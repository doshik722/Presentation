# Здесь размещаем общую основу PPO и GRPO: конфиг, очередь заданий, сбор эпизодов, награду, log p, журнал и манифест.
"""Общая основа RL (``docs/grpo.md``; PPO будет использовать её же).

Части без torch (конфиг, сиды, очередь заданий, журнал, награда и аналитика, сбор эпизодов) проверяются тестами
без модели; части с torch (log p кусками, KL, шаг оптимизатора, слияние SFT) — тестами на крошечной модели
(``tests/test_rl_torch.py``, запускаются на сервере).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..environment import Episode, build_prompt, run_episode
from ..tasks import Task

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NAME_GRAMMEMES = frozenset({"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"})
HASHED_SOURCES = ("environment.py", "dsl.py", "rewards.py", "prompts.py", "tasks.py", "rendering/builder.py",
                  "tools/wiki_tools.py", "tools/cache.py", "training/common.py", "training/grpo.py",
                  "training/vllm_client.py")


# --------------------------------------------------------------------------- конфиг

@dataclass
class RLConfig:
    """Параметры запуска (``configs/algorithms/grpo.yaml``); обоснования — ``docs/grpo.md``."""

    algorithm: str = "grpo"
    runs_dir: str = "runs/grpo"
    policy_model: str = "t-tech/T-lite-it-2.1"  # стартовая и опорная модель — исходная T-lite (решение 13.11.2026)
    policy_revision: str = "main"  # хеш коммита из кеша HF пишется в манифест (``model_commit``)
    vllm_url: str = "http://127.0.0.1:8000"
    vllm_base_name: str = "base"  # --served-model-name исходной модели на сервере vLLM
    wiki_snapshot: str = "data/sources/wiki_cache"  # снимок кеша: копируется в папку запуска на старте
    # шаг
    tasks_per_mode: int = 4
    group_size: int = 8
    steps: int = 150
    # эпизод
    max_new_tokens: int = 12000  # решение 13.11.2026 (было 8000)
    max_context: int = 25000
    max_calls: int = 10
    temperature: float = 0.9
    workers: int = 64  # одновременных эпизодов (запросы к vLLM он сам собирает в пакеты)
    # модель и оптимизатор
    lora_rank: int = 32
    lora_alpha: int = 32
    learning_rate: float = 1e-5
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    logits_chunk: int = 1024
    # награда
    reward_floor: float = -1.0  # нижний порог награды эпизода (решение 12.11.2026)
    # loss
    kl_beta: float = 0.02
    clip_eps: float = 0.2
    std_floor: float = 0.02
    # validation и сохранение
    eval_every: int = 10
    val_topics: int = 8
    val_samples: int = 2
    keep_states: int = 2
    # контроль
    kl_old_new_episodes: int = 4  # на скольких эпизодах шага считать KL(old‖new) после обновления
    max_logprob_mismatch: dict = field(default_factory=lambda: {"mean_abs": None, "p99": None})  # по пилоту
    max_skipped_updates: float = 0.05
    embedder_device: str = "cuda:1"
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "RLConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known - {"implemented"}
        if unknown:
            raise ValueError(f"неизвестные параметры конфига: {sorted(unknown)}")
        values = {k: v for k, v in data.items() if k in known} | overrides
        if "betas" in values:
            values["betas"] = tuple(values["betas"])
        return cls(**values)

    def to_json(self) -> dict:
        data = dataclasses.asdict(self)
        data["betas"] = list(self.betas)
        return data


# --------------------------------------------------------------------------- сиды и очередь заданий

def stable_seed(*parts) -> int:
    """Сид из идентификаторов (запуск / шаг / задание / номер ответа / ход): не зависит от порядка обработки
    и от ``hash()`` Python (тот меняется от запуска к запуску)."""
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


class TaskQueue:
    """Две очереди заданий — с поиском и без, по ``per_mode`` из каждой на шаг. Темы каждой очереди перемешиваются
    заново в каждой эпохе (сид — от эпохи и режима); в конце эпохи шаг неполный (остаток, решение 08.11.2026)."""

    MODES = ("search", "plain")

    def __init__(self, ids: list[str], per_mode: int, seed: int):
        self.ids = sorted(ids)
        self.per_mode = per_mode
        self.seed = seed
        self.epoch = 0
        self.position = 0

    def _order(self, mode: str) -> list[str]:
        ids = list(self.ids)
        random.Random(stable_seed(self.seed, "order", mode, self.epoch)).shuffle(ids)
        return ids

    def next(self) -> list[tuple[str, str]]:
        """Задания следующего шага: [(режим, id темы)] — сначала с поиском, потом без."""
        if self.position >= len(self.ids):
            self.epoch += 1
            self.position = 0
        end = min(self.position + self.per_mode, len(self.ids))
        batch = [(mode, task_id) for mode in self.MODES for task_id in self._order(mode)[self.position:end]]
        self.position = end
        return batch

    def state(self) -> dict:
        return {"epoch": self.epoch, "position": self.position}

    def load_state(self, state: dict) -> None:
        self.epoch, self.position = state["epoch"], state["position"]


def validation_jobs(topics: list[str], samples: int) -> list[tuple[str, str, int]]:
    """Задания validation: (режим, тема, номер ответа); сиды — без номера шага (одинаковые у всех чекпоинтов)."""
    return [(mode, topic, s) for mode in TaskQueue.MODES for topic in topics for s in range(samples)]


def fixed_validation_topics(all_ids: list[str], n: int) -> list[str]:
    """Малый validation: ``n`` тем, одинаковых для всех запусков (сид не зависит от запуска)."""
    return sorted(random.Random(stable_seed("validation-topics")).sample(sorted(all_ids), n))


# --------------------------------------------------------------------------- эпизоды

@dataclass
class Rollout:
    """Эпизод с тем, что нужно обучению и журналу."""

    task_id: str
    mode: str
    sample: int
    ids: list[int]
    mask: list[int]
    gen_logprobs: list[float]  # log p генератора для токенов модели (в порядке маски), nan — неизвестно
    text: str  # эпизод после промпта, как его видела модель
    answer: str
    finish: str
    calls: list[dict]
    truncated_outputs: int
    seconds: float
    reward: float = math.nan
    components: dict = field(default_factory=dict)
    analytics: dict = field(default_factory=dict)

    @property
    def model_tokens(self) -> int:
        return sum(self.mask)

    @classmethod
    def from_episode(cls, task_id: str, mode: str, sample: int, episode: Episode, tokenizer,
                     seconds: float) -> "Rollout":
        logprobs = []
        for segment in episode.segments:
            if segment.trainable:
                values = segment.logprobs if segment.logprobs is not None else [math.nan] * len(segment.ids)
                logprobs += list(values)
        text = tokenizer.decode(episode.ids[len(episode.prompt_ids):], skip_special_tokens=False)
        return cls(task_id, mode, sample, episode.ids, episode.loss_mask, logprobs, text, episode.answer,
                   episode.finish, episode.calls, episode.truncated_outputs, seconds)

    def to_json(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if k not in ("ids", "mask", "gen_logprobs")}


@dataclass
class Job:
    task: Task  # задание с уже выставленным режимом (``request.search``)
    mode: str
    sample: int
    seed_parts: tuple  # части сида; к ним добавляется номер реплики


def mode_task(task: Task, mode: str) -> Task:
    return dataclasses.replace(task, request=dataclasses.replace(task.request, search=(mode == "search")))


class LockedTokenizer:
    """Токенизатор для потоков: кодирование и декодирование — по одному за раз (быстрый токенизатор HF при
    одновременных вызовах из разных потоков может упасть с «Already borrowed»)."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    def encode(self, *args, **kwargs):
        with self._lock:
            return self._tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        with self._lock:
            return self._tokenizer.decode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def collect(jobs: list[Job], tokenizer, prompt: Callable, make_generate: Callable[[Job], Callable],
            make_tools: Callable[[], object], config: RLConfig) -> list[Rollout]:
    """Эпизоды заданий параллельно (потоки; генерацию собирает в пакеты сервер vLLM). Ошибка эпизода останавливает
    шаг — эпизоды не выбрасываются молча (это сместило бы данные)."""
    prompts = [prompt(job.task.request) for job in jobs]  # шаблон чата — в основном потоке
    safe = LockedTokenizer(tokenizer)

    def one(item) -> Rollout:
        job, prompt_ids = item
        start = time.monotonic()
        tools = make_tools() if job.mode == "search" else None
        episode = run_episode(prompt_ids, safe, make_generate(job), tools,
                              max_new_tokens=config.max_new_tokens, max_context=config.max_context)
        return Rollout.from_episode(job.task.id, job.mode, job.sample, episode, safe, time.monotonic() - start)

    with ThreadPoolExecutor(max_workers=max(1, min(config.workers, len(jobs)))) as pool:
        return list(pool.map(one, zip(jobs, prompts)))


def episode_prompt(tokenizer) -> Callable:
    from ..prompts import system_prompt

    def prompt(request):
        return build_prompt(tokenizer, system_prompt(request), request)

    return prompt


def stop_ids(tokenizer) -> list[int]:
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return [end_id] + ([tokenizer.eos_token_id] if tokenizer.eos_token_id not in (None, end_id) else [])


def vllm_generator(client, model_name: str, temperature: float, stops: list[int], seed_parts: tuple) -> Callable:
    """``generate`` для эпизода: реплики по очереди, сид каждой — от идентификаторов эпизода и номера реплики."""
    turn = [0]

    def generate(ids, budget):
        seed = stable_seed(*seed_parts, turn[0])
        turn[0] += 1
        return client.generate(ids, budget, model_name, temperature, seed, stops)

    return generate


# --------------------------------------------------------------------------- награда и аналитика

class RewardModel:
    """Принятая награда (``rewards.score``, решение 07.11.2026) по готовому эпизоду и аналитика опоры на источники.
    Контекст эпизода (вызовы, показанная модели выдача, допустимые ``#SOURCE``) берётся из текста эпизода — то есть
    только то, что модель действительно видела (после обрезаний). Награда эпизода обрезается снизу на ``floor``
    (решение 12.11.2026); необрезанная остаётся в компонентах (``total``)."""

    def __init__(self, root: str | Path, device: str | None = None, embedder=None, lemmatizer=None, weights=None,
                 floor: float = -1.0):
        from ..rewards import Lemmatizer, RewardWeights, SentenceEmbedder, load_key_points

        self.points = load_key_points(Path(root) / "data" / "tasks" / "key_points" / "key_points.json")
        self.embedder = embedder or SentenceEmbedder("BAAI/bge-m3", device=device)
        self.lemmatizer = lemmatizer or Lemmatizer()
        self.weights = weights or RewardWeights()
        self.floor = floor
        self._lock = threading.Lock()  # эмбеддер и лемматизатор — по одному вызову за раз

    def __call__(self, task: Task, rollout: Rollout) -> None:
        from ..dsl import parse
        from ..perturb import task_context
        from ..rendering import build_presentation, pick_theme
        from ..rewards import VISUAL_KINDS, score

        search = rollout.mode == "search"
        context = task_context(task, rollout.text, search)
        deck = parse(rollout.answer)
        report = build_presentation(deck, None, pick_theme(task.request.theme, task.request.text),
                                    sources=(context["sources"] or {}).get, show_sources=task.request.sources)
        with self._lock:
            result = score(deck, self.points[task.id], conclusion=context["conclusion"],
                           slide_range=tuple(context["slide_range"]) if context["slide_range"] else None,
                           embedder=self.embedder, lemmatizer=self.lemmatizer, report=report,
                           images_allowed=context["images_allowed"], tool_text=context["tool_text"],
                           calls=context["calls"], search=search, weights=self.weights)
            analytics = grounding(deck, report, context["tool_text"], self.lemmatizer) if search else {}
        if search:
            analytics |= search_stats(rollout.calls, rollout.text)
        rollout.reward = max(float(result.total), self.floor)
        rollout.components = {k: float(v) for k, v in result.components().items()}
        rollout.analytics = analytics | {"slides": len(deck.slides), "parse_rate": deck.parse_rate,
                                         "format_errors": deck.error_counts(),
                                         **episode_stats(deck, result, context["slide_range"] is not None, VISUAL_KINDS),
                                         "conclusion_expected": bool(context["conclusion"])}


def slide_facts(text: str, lemmatizer) -> set[str]:
    """Числа и имена текста слайда — то, что можно сверить с источником буквально (решение 08.11.2026).
    Имена — слова с заглавной буквы, которые словарь считает именем, фамилией, отчеством, географическим названием
    или организацией, и слова латиницей с заглавной буквы; числа — как в награде («6 400» → «6400»)."""
    import re

    facts = {lemma for lemma, pos in lemmatizer.tokens(text) if pos == "NUMB"}
    for word in re.findall(r"[A-ZА-ЯЁ][\w-]+", text):
        if re.fullmatch(r"[A-Za-z][\w-]*", word):
            facts.add(word.lower())
            continue
        parsed = lemmatizer.analyzer.parse(word)
        if parsed and NAME_GRAMMEMES & set(parsed[0].tag.grammemes):
            facts.add(parsed[0].normal_form.replace("ё", "е"))
    return facts


def episode_stats(deck, result, explicit: bool, visual_kinds) -> dict:
    """Аналитика ответа (только журнал, не награда): есть ли план (``#PLAN``) и заключение (последний слайд
    из трёх и больше с заголовком «Вывод(ы)», «Заключение», «Итог(и)» — как в ``evaluation``); ожидаемое число слайдов (из запроса — ``explicit``, иначе
    по тезисам темы); визуалы, которые модель написала (все и картинки); проверенные на связь со слайдом и ниже
    порога — картинки и остальные виды отдельно (``rewards.VISUAL_TAU``)."""
    kinds = [e.kind for s in deck.slides for e in s.elements if e.kind in visual_kinds]
    checked, unrelated = result.visual_checked, result.visual_unrelated
    from ..evaluation import CONCLUSION

    low, high = result.expected_slides
    n = len(deck.slides)
    return {"has_plan": bool(deck.plan),
            "has_conclusion": bool(n >= 3 and CONCLUSION.match(deck.slides[-1].title or "")),
            "slides_low": low, "slides_high": high, "slides_explicit": explicit,
            "visuals": len(kinds), "images": kinds.count("IMAGE"),
            "images_checked": checked.get("IMAGE", 0), "images_unrelated": unrelated.get("IMAGE", 0),
            "other_checked": sum(v for k, v in checked.items() if k != "IMAGE"),
            "other_unrelated": sum(v for k, v in unrelated.items() if k != "IMAGE")}


def search_stats(calls: list[dict], text: str) -> dict:
    """Аналитика поиска в эпизоде (только журнал, не награда): вызовы и ошибочные (неразобранные) вызовы; поиски,
    поиски без результатов, запросы без кириллицы; поиски с результатами и из них «повторные» — больше половины
    найденных статей уже находили прежние поиски эпизода (запрос с одним добавленным словом, который ничего
    нового не даёт). Выдача — из текста эпизода по порядку вызовов (у вызова, после которого эпизод кончился,
    её нет)."""
    import re

    responses = re.findall(r"<tool_response>\n(.*?)\n</tool_response>", text, re.DOTALL)
    out = {"tool_calls": len(calls), "tool_errors": sum(bool(c.get("error")) for c in calls), "searches": 0,
           "searches_nothing": 0, "searches_en": 0, "searches_with_results": 0, "searches_repeated": 0}
    seen: set[str] = set()
    for call, response in zip(calls, responses + [None] * len(calls)):
        if call.get("name") != "search" or call.get("error") or response is None:
            continue
        out["searches"] += 1
        out["searches_en"] += not re.search("[а-яё]", call.get("argument") or "", re.IGNORECASE)
        if "ничего не найдено" in response:
            out["searches_nothing"] += 1
            continue
        found = set(re.findall(r"^(d\d+) \| (?:ru|en) \| ", response, re.MULTILINE))
        if found:
            out["searches_with_results"] += 1
            out["searches_repeated"] += len(found & seen) > len(found) / 2
            seen |= found
    return out


def grounding(deck, report, tool_text: str | None, lemmatizer) -> dict:
    """Опора на прочитанное (только аналитика): сколько чисел и имён со слайдов встречается в выдаче, показанной
    модели, и сколько слайдов ссылаются на прочитанное (``#SOURCE`` из выдачи)."""
    from ..rewards import slide_texts

    seen = lemmatizer.tokens(tool_text or "")
    seen_words = {lemma for lemma, _ in seen} | {w.lower() for w in (tool_text or "").split()}
    facts = set().union(*(slide_facts(t.full, lemmatizer) for t in slide_texts(deck, report))) if deck.slides else set()
    sourced = {item.slide for item in report.items if item.kind == "SOURCE" and item.status == "ok"}
    return {"facts": len(facts), "facts_found": sum(1 for f in facts if f in seen_words),
            "slides_with_source": len(sourced)}


def summarize(rollouts: list[Rollout]) -> dict:
    """Сводка шага по эпизодам (журнал): награда и компоненты по режимам, завершения, вызовы, аналитика."""
    import statistics

    def mean(values):
        values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return statistics.mean(values) if values else None

    out: dict = {"episodes": len(rollouts)}
    for mode in TaskQueue.MODES:
        part = [r for r in rollouts if r.mode == mode]
        if not part:
            continue
        prefix = f"{mode}_"
        out[prefix + "reward"] = mean([r.reward for r in part])
        out[prefix + "floored"] = mean([float(r.components.get("total", r.reward) < r.reward) for r in part])
        for key in sorted({k for r in part for k in r.components}):
            out[prefix + key] = mean([r.components.get(key) for r in part])
        out[prefix + "calls"] = mean([len(r.calls) for r in part])
        out[prefix + "model_tokens"] = mean([r.model_tokens for r in part])
        out[prefix + "tokens"] = mean([len(r.ids) for r in part])
        # доли оборванных эпизодов: контекст исчерпан (выдача и реплики) / предел токенов модели; с обрезанной выдачей
        out[prefix + "finish_context"] = mean([float(r.finish == "context") for r in part])
        out[prefix + "finish_length"] = mean([float(r.finish == "length") for r in part])
        out[prefix + "truncated"] = mean([float(r.truncated_outputs > 0) for r in part])
        # формат DSL: доля корректных строк, доля ответов без ошибок разбора, ошибки по кодам (сумма за шаг)
        parsed = [r.analytics for r in part if "parse_rate" in r.analytics]
        out[prefix + "parse_rate"] = mean([a["parse_rate"] for a in parsed])
        out[prefix + "format_clean"] = mean([float(not a.get("format_errors")) for a in parsed if "format_errors" in a])
        errors: dict[str, int] = {}
        for a in parsed:
            for code, count in (a.get("format_errors") or {}).items():
                errors[code] = errors.get(code, 0) + count
        out[prefix + "format_errors"] = dict(sorted(errors.items(), key=lambda kv: -kv[1]))
        # слайды: попадание в ожидаемый диапазон — для запросов с числом слайдов и без него отдельно
        ranged = [r.analytics for r in part if "slides_high" in r.analytics]
        out[prefix + "slides"] = mean([a["slides"] for a in ranged])
        for name, group in (("explicit", [a for a in ranged if a["slides_explicit"]]),
                            ("auto", [a for a in ranged if not a["slides_explicit"]])):
            out[f"{prefix}in_range_{name}"] = mean([float(a["slides_low"] <= a["slides"] <= a["slides_high"])
                                                    for a in group])
            out[f"{prefix}slides_off_{name}"] = mean([max(a["slides_low"] - a["slides"], a["slides"] - a["slides_high"], 0)
                                                     for a in group])
        # план и заключение: доли ответов; заключение — ещё и среди тем, где оно ожидается (тезис-заключение)
        out[prefix + "with_plan"] = mean([float(a["has_plan"]) for a in ranged if "has_plan" in a])
        out[prefix + "with_conclusion"] = mean([float(a["has_conclusion"]) for a in ranged if "has_conclusion" in a])
        out[prefix + "with_conclusion_expected"] = mean([float(a["has_conclusion"]) for a in ranged
                                                         if a.get("conclusion_expected") and "has_conclusion" in a])
        # визуалы: сколько на эпизод и доля не связанных со слайдом среди проверенных
        out[prefix + "visuals"] = mean([a["visuals"] for a in ranged])
        out[prefix + "images"] = mean([a["images"] for a in ranged])
        for name in ("images", "other"):
            checked = sum(a[f"{name}_checked"] for a in ranged)
            out[f"{prefix}{name}_unrelated_share"] = sum(a[f"{name}_unrelated"] for a in ranged) / checked \
                if checked else None
        if mode == "search":
            out["search_share"] = mean([float(any(not c.get("error") for c in r.calls)) for r in part])
            with_facts = [r for r in part if r.analytics.get("facts")]
            out["grounding"] = mean([r.analytics["facts_found"] / r.analytics["facts"] for r in with_facts])
            out["grounding_no_facts"] = len(part) - len(with_facts)
            out["slides_with_source"] = mean([r.analytics.get("slides_with_source", 0) / r.analytics["slides"]
                                              for r in part if r.analytics.get("slides")])
            out["tool_calls"] = {name: sum(1 for r in part for c in r.calls if c.get("name") == name)
                                 for name in ("search", "open", "read")}
            total = {k: sum(r.analytics.get(k, 0) for r in part) for k in
                     ("tool_calls", "tool_errors", "searches", "searches_nothing", "searches_en",
                      "searches_with_results", "searches_repeated")}
            share = lambda a, b: total[a] / total[b] if total[b] else None  # noqa: E731
            out["tool_error_share"] = share("tool_errors", "tool_calls")
            out["search_nothing_share"] = share("searches_nothing", "searches")
            out["search_en_share"] = share("searches_en", "searches")
            out["search_repeated_share"] = share("searches_repeated", "searches_with_results")
    out["finish"] = {k: sum(1 for r in rollouts if r.finish == k) for k in sorted({r.finish for r in rollouts})}
    out["truncated_outputs"] = sum(r.truncated_outputs for r in rollouts)
    out["parse_rate"] = mean([r.analytics.get("parse_rate") for r in rollouts])
    return out


# --------------------------------------------------------------------------- журнал

class Journal:
    """Журнал JSONL: записи только дописываются (обучение и судья пишут каждый в свой файл). При продолжении
    дописывается отметка ``resume``: записи шагов после восстановленной точки, сделанные до неё, при чтении
    отбрасываются (``read_journal``)."""

    def __init__(self, path: str | Path, session: int):
        self.path = Path(path)
        self.session = session
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": self.session, **record},
                          ensure_ascii=False, default=_json_default)
        with self._lock, self.path.open("a", encoding="utf-8") as out:
            out.write(line + "\n")

    def resume(self, from_step: int) -> None:
        self.write({"type": "resume", "from_step": from_step})


def _json_default(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def read_journal(path: str | Path) -> list[dict]:
    """Записи журнала без отменённых: шаги после точки продолжения, записанные до отметки ``resume``."""
    path = Path(path)
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept: list[dict] = []
    for record in records:
        if record.get("type") == "resume":
            kept = [r for r in kept if r.get("step") is None or r["step"] <= record["from_step"]]
            continue
        kept.append(record)
    return kept


# --------------------------------------------------------------------------- папка запуска и манифест

def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: str | Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def versions() -> dict:
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "sentence-transformers", "pymorphy3"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def code_hashes(package_root: str | Path) -> dict:
    root = Path(package_root)
    return {name: file_hash(root / name) for name in HASHED_SOURCES if (root / name).exists()}


def snapshot_info(folder: str | Path) -> dict:
    folder = Path(folder)
    return {p.name: {"lines": sum(1 for _ in p.open(encoding="utf-8")), "sha256": file_hash(p)}
            for p in sorted(folder.glob("*.jsonl"))}


def new_run_dir(config: RLConfig, root: str | Path) -> Path:
    """Папка нового запуска: имя — время первого старта и сид (идентификатор запуска, при продолжении не меняется)."""
    name = time.strftime("%Y%m%d_%H%M") + f"_seed{config.seed}"
    path = Path(root) / config.runs_dir / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def start_manifest(run_dir: Path, config: RLConfig, root: str | Path, extra: dict) -> dict:
    root = Path(root)
    manifest = {"run": run_dir.name, "config": config.to_json(), "versions": versions(), "git": git_state(root),
                "code": code_hashes(root / "src" / "presentation_rl"), "sessions": [], **extra}
    manifest["policy_model"] = {"id": config.policy_model, "revision": config.policy_revision,
                                "commit": model_commit(config.policy_model, config.policy_revision)}
    return manifest


def model_commit(model_id: str, revision: str) -> str | None:
    """Хеш коммита модели: ревизия, если это уже хеш (закреплена в конфиге), иначе — куда указывает ссылка ревизии
    в кеше HF (``refs/<ревизия>``; тот же снимок читают ``transformers`` и vLLM); None — ссылки нет."""
    import re

    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    try:
        from huggingface_hub import constants

        ref = Path(constants.HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "refs" / revision
        return ref.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def add_session(run_dir: Path, manifest: dict, from_step: int, root: str | Path, extra: dict | None = None) -> int:
    session = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "from_step": from_step, "git": git_state(root),
               "versions": versions(), **(extra or {})}
    manifest["sessions"].append(session)
    write_json(run_dir / "manifest.json", manifest)
    return len(manifest["sessions"])


def check_resume(manifest: dict, config: RLConfig, root: str | Path) -> None:
    """Продолжение — только тот же эксперимент: те же настройки и тот же код (иначе это новый запуск)."""
    if manifest["config"] != config.to_json():
        changed = sorted(k for k in config.to_json() if manifest["config"].get(k) != config.to_json()[k])
        raise ValueError(f"настройки изменились ({changed}) — это новый запуск, а не продолжение")
    code = code_hashes(Path(root) / "src" / "presentation_rl")
    if manifest["code"] != code:
        changed = sorted(k for k in code if manifest["code"].get(k) != code[k])
        raise ValueError(f"код изменился ({changed}) — это новый запуск, а не продолжение")


def write_json(path: Path, data) -> None:
    """Атомарная запись: временный файл и переименование (прерванная запись не портит прежний файл)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def copy_snapshot(snapshot: str | Path, run_dir: Path) -> Path:
    """Каждый запуск начинается с одного и того же снимка кеша Википедии; новые ответы пишутся в копию запуска."""
    target = run_dir / "wiki_cache"
    if not target.exists():
        shutil.copytree(snapshot, target)
    return target


def save_episodes(folder: Path, rollouts: list[Rollout]) -> None:
    """Эпизоды validation: текст эпизода, ответ и метаданные (их читает судья)."""
    folder.mkdir(parents=True, exist_ok=True)
    for r in rollouts:
        stem = f"{r.mode}_{r.task_id}_{r.sample}"
        (folder / f"{stem}.txt").write_text(r.text, encoding="utf-8")
        (folder / f"{stem}_answer.txt").write_text(r.answer, encoding="utf-8")
        write_json(folder / f"{stem}.json", r.to_json())


# --------------------------------------------------------------------------- torch: log p, KL, шаг

def backbone_and_head(model):
    """Тело трансформера (до lm_head, с финальной нормировкой) и lm_head — у обычной модели и у PeftModel."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def target_positions(mask: list[int]) -> list[int]:
    """Позиции токенов модели (маска 1); токен в позиции p предсказывается выходом позиции p − 1."""
    return [p for p in range(1, len(mask)) if mask[p]]


def _chunk_logprobs(hidden, targets, weight, temperature: float):
    import torch

    logits = torch.nn.functional.linear(hidden, weight).float() / temperature
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(1, targets[:, None])[:, 0]
    entropy = -(logp_all.exp() * logp_all).sum(-1)
    return logp, entropy.detach()


def sequence_logprobs(model, ids: list[int], mask: list[int], temperature: float, chunk: int = 1024):
    """log p токенов модели при температуре T (как при генерации) и полная энтропия распределения в этих позициях.

    Логиты по словарю (~150 тыс.) считаются только в нужных позициях и кусками по ``chunk``; каждый кусок — под
    перевычислением (checkpoint): в памяти остаются выходы тела, а не логиты всего эпизода."""
    import torch
    from torch.utils.checkpoint import checkpoint

    device = next(model.parameters()).device
    if not target_positions(mask):
        empty = torch.zeros(0, device=device)
        return empty, empty
    body, head = backbone_and_head(model)
    input_ids = torch.tensor([ids], device=device)
    positions = torch.tensor([p - 1 for p in target_positions(mask)], device=device)
    hidden = body(input_ids=input_ids, use_cache=False).last_hidden_state[0]
    hidden = hidden.index_select(0, positions)
    targets = input_ids[0].index_select(0, positions + 1)
    logps, entropies = [], []
    for start in range(0, len(positions), chunk):
        part = slice(start, start + chunk)
        if torch.is_grad_enabled():
            logp, entropy = checkpoint(_chunk_logprobs, hidden[part], targets[part], head.weight, temperature,
                                       use_reentrant=False)
        else:
            logp, entropy = _chunk_logprobs(hidden[part], targets[part], head.weight, temperature)
        logps.append(logp)
        entropies.append(entropy)
    return torch.cat(logps), torch.cat(entropies)


def kl_terms(logp, ref_logp):
    """k2 = ½(log π_θ − log π_ref)² — в loss (её градиент при выборке своей политикой — градиент KL(π_θ‖π_ref));
    k3 = e^d − d − 1, d = log π_ref − log π_θ — несмещённая оценка значения KL, только в журнал."""
    import torch

    k2 = 0.5 * (logp - ref_logp) ** 2
    d = (ref_logp - logp).detach()
    k3 = torch.exp(d) - d - 1
    return k2, k3


def policy_terms(logp, advantage: float, old_logp=None, clip_eps: float = 0.2):
    """Слагаемое политики PPO/GRPO на токенах: −min(ρA, clip(ρ)A). ``old_logp=None`` — одно обновление на пакет:
    ρ = exp(log p − sg(log p)) = 1, градиент −A∇log p (константа 1 дала бы нулевой градиент)."""
    import torch

    old = logp.detach() if old_logp is None else old_logp
    ratio = torch.exp(logp - old)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    terms = -torch.minimum(ratio * advantage, clipped * advantage)
    clip_fraction = ((ratio - 1).abs() > clip_eps).float().mean()
    return terms, clip_fraction.detach()


def mismatch_stats(train_logp: list[float], gen_logp: list[float]) -> dict:
    """Расхождение log p генератора и обучающего прохода на одних токенах: среднее со знаком, среднее |Δ|, p99 |Δ|."""
    diffs = [t - g for t, g in zip(train_logp, gen_logp) if not math.isnan(g)]
    if not diffs:
        return {"tokens": 0}
    absolute = sorted(abs(d) for d in diffs)
    return {"tokens": len(diffs), "mean": sum(diffs) / len(diffs), "mean_abs": sum(absolute) / len(absolute),
            "p99": absolute[min(len(absolute) - 1, int(0.99 * len(absolute)))]}


def mismatch_exceeded(stats: dict, limits: dict) -> str | None:
    for key in ("mean_abs", "p99"):
        if limits.get(key) is not None and stats.get(key) is not None and stats[key] > limits[key]:
            return f"расхождение log p vLLM и обучения: {key} = {stats[key]:.4f} > {limits[key]}"
    return None


# --------------------------------------------------------------------------- модель, адаптер, слияние SFT

def setup_torch() -> None:
    """Обучение: наивное ядро внимания выключено — при неподходящих входах будет ошибка, а не тихий переход
    на память O(L²) (решение 08.11.2026)."""
    import torch

    torch.backends.cuda.enable_math_sdp(False)


def load_policy(config: RLConfig, root: str | Path, device: str = "cuda:0", adapter: str | Path | None = None):
    """Исходная модель во float16 + RL-LoRA во float32 (новый — с сидом запуска; ``adapter`` — продолжение)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.policy_model, revision=config.policy_revision)
    model = AutoModelForCausalLM.from_pretrained(config.policy_model, revision=config.policy_revision,
                                                 torch_dtype=torch.float16, device_map={"": device},
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        torch.manual_seed(stable_seed(config.seed, "lora"))
        model = get_peft_model(model, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.0,
                                                 bias="none", target_modules=list(TARGET_MODULES),
                                                 task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.train()
    return model, tokenizer


def export_adapter(model, folder: Path) -> Path:
    """Атомарное сохранение адаптера: временная папка → полная запись → переименование."""
    tmp = folder.with_name(folder.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return folder


def merge_sft(base_id: str, revision: str, adapter: str | Path, out_dir: str | Path, device: str = "cuda:0") -> dict:
    """Слияние SFT-адаптера с базой: W + ΔW считается во float32, результат сохраняется во float16 (решение
    08.11.2026). Рядом — ``merged_hash.json`` (хеши весов, база, адаптер) для манифестов запусков."""
    import torch
    from peft import PeftModel
    from peft.tuners.lora import Linear as LoraLinear
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(base_id, revision=revision, torch_dtype=torch.float16,
                                                 device_map={"": device})
    model = PeftModel.from_pretrained(model, adapter)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLinear):
                delta = module.get_delta_weight("default").float()
                weight = module.get_base_layer().weight
                weight.data = (weight.data.float() + delta).to(torch.float16)
    model = model.unload()  # адаптер убирается без повторного вливания
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    info = {"path": str(out_dir), "base": base_id, "revision": revision, "adapter": str(adapter),
            "adapter_hash": {p.name: file_hash(p) for p in sorted(Path(adapter).glob("*.safetensors"))},
            "weights": {p.name: file_hash(p) for p in sorted(out_dir.glob("*.safetensors"))}}
    write_json(out_dir / "merged_hash.json", info)
    return info
