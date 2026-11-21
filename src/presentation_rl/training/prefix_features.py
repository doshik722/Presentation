# Здесь считаю признаки оценки уже закрытых слайдов для критика PPO.
from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from ..dsl import ParseLimits, parse
from ..environment import END, TOOL_RESULT, Tokenizer, ToolCall, _ARGUMENTS, _CALL_RE, parse_tool_calls
from ..key_points import expected_theses
from ..perturb import _RESPONSE_RE, episode_context
from ..rendering import build_presentation, pick_theme
from ..rewards import score
from ..tasks import Task
from .common import RewardModel, mode_task, stable_seed, target_positions

if TYPE_CHECKING:
    from .ppo import PPOConfig

_SLIDE_LINE = re.compile(r"^[^\S\n]*#SLIDE(?![A-Z_])[^\n]*\n", re.MULTILINE | re.IGNORECASE | re.ASCII)
_MAX_LINE = ParseLimits().max_line_length
_COMPONENTS = (
    "total", "coverage", "keys", "length", "visual", "forbidden_images", "format", "text_overflow",
    "bad_sources", "tools", "contradictions", "sources", "slide_overflow", "render_errors", "fake_quotes",
    "repeats", "formula_over", "unrelated_visuals",
)
FEATURE_NAMES = _COMPONENTS + (
    "floored_total", "uncovered_fraction", "closed_slides", "slides_over_low", "slides_over_high",
    "slides_above_high", "expected_low", "expected_high", "theses", "conclusion_expected",
    "search", "images_allowed", "answer_started", "observed_calls", "model_budget_used", "context_left",
    "actions_since_score",
)


@dataclass(frozen=True)
class PrefixEvent:
    """Состояние перед действием в position, то есть по ids[:position]."""

    position: int
    answer: str
    tool_history: str
    answer_started: bool


@dataclass(frozen=True)
class PrefixFeatures:
    """Строки признаков в порядке target_positions(mask) и число вычислений score."""

    values: np.ndarray
    score_calls: int


def _closed_answer(text: str) -> tuple[str, bool]:
    """Оставляю текст до последней завершённой строки с #SLIDE."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = None
    for match in _SLIDE_LINE.finditer(text):
        if len(match[0].strip()) <= _MAX_LINE:
            start = match.start()
    return (text[:start].rstrip(), True) if start is not None else ("", False)


def _observed_exchange(calls: list[ToolCall], block: str) -> str:
    """Сопоставляю вызовы одной реплики с выдачей из следующего блока среды."""
    parts = []
    for call, response in zip(calls, _RESPONSE_RE.findall(block)):
        if call.error:
            body = "{}"  # неразобранный вызов занимает своё место в последовательности
        else:
            body = json.dumps({"name": call.name, "arguments": {_ARGUMENTS[call.name]: call.argument}},
                              ensure_ascii=False).replace("<", r"\u003c")
        parts.extend((f"<tool_call>{body}</tool_call>", TOOL_RESULT.format(response)))
    return "".join(parts)


def prefix_events(ids: list[int], mask: list[int], tokenizer: Tokenizer,
                  newline_tokens: dict[int, bool] | None = None) -> Iterator[PrefixEvent]:
    """Обновляю закрытую часть при видимом #SLIDE и в начале реплики после выдачи.

    В байтовом токенизаторе T-lite перевод строки виден уже при декодировании отдельного токена.
    Само событие и его текст проверяю декодированием целого префикса текущей реплики.
    """
    if len(ids) != len(mask) or (mask and mask[0]):
        raise ValueError("нужна маска той же длины, что ids, с промптом перед первым действием")
    newline_tokens = {} if newline_tokens is None else newline_tokens
    parts: list[str] = []
    calls: list[ToolCall] = []
    tool_history = ""
    start = 0

    for is_action, span in groupby(mask):
        end = start + sum(1 for _ in span)
        if is_action:
            previous = "".join(part + "\n" for part in parts)
            closed = _closed_answer(previous)
            yield PrefixEvent(start, closed[0], tool_history, closed[1])
            for p in range(start + 1, end):
                token = ids[p - 1]
                if token not in newline_tokens:
                    piece = tokenizer.decode([token], skip_special_tokens=False)
                    newline_tokens[token] = "\n" in piece or "\r" in piece
                if not newline_tokens[token]:
                    continue
                text = tokenizer.decode(ids[start:p], skip_special_tokens=False)
                # Сохраняю переводы строк вокруг вызовов. Незакрытый вызов пока не часть ответа.
                clean = _CALL_RE.sub("", text).partition("<tool_call>")[0]
                current = _closed_answer(previous + clean)
                if current != closed:
                    closed = current
                    yield PrefixEvent(p, closed[0], tool_history, closed[1])
            turn = tokenizer.decode(ids[start:end], skip_special_tokens=False).removesuffix(END)
            answer, calls = parse_tool_calls(turn)
            if answer:
                parts.append(answer)
        elif start:
            tool_history += _observed_exchange(calls, tokenizer.decode(ids[start:end], skip_special_tokens=False))
        start = end


class PrefixFeatureExtractor:
    """Считаю score по закрытым слайдам с отчётом их сборки и видимой на тот момент выдачей."""

    def __init__(self, tokenizer: Tokenizer, reward: RewardModel, *, max_new_tokens: int, max_context: int):
        if min(max_new_tokens, max_context) <= 0:
            raise ValueError("лимиты токенов должны быть положительными")
        self.tokenizer, self.reward = tokenizer, reward
        self.max_new_tokens, self.max_context = max_new_tokens, max_context
        self._newline_tokens: dict[int, bool] = {}

    def __call__(self, task: Task, ids: list[int], mask: list[int]) -> PrefixFeatures:
        actions = np.asarray(target_positions(mask), dtype=np.int64)
        values = np.empty((len(actions), len(FEATURE_NAMES)), dtype=np.float32)
        events = list(prefix_events(ids, mask, self.tokenizer, self._newline_tokens))
        if not events:
            return PrefixFeatures(values, 0)

        request = task.request
        search = request.search
        points = self.reward.points[task.id]
        conclusion = expected_theses(task)[2]
        slide_range = request.slide_range()
        theme = pick_theme(request.theme, request.text)
        starts = np.searchsorted(actions, [event.position for event in events]).tolist() + [len(actions)]
        previous_history = None

        for i, event in enumerate(events):
            if event.tool_history != previous_history:
                context = episode_context(event.tool_history, search)
                previous_history = event.tool_history
            deck = parse(event.answer)
            report = build_presentation(deck, None, theme, sources=context["sources"].get,
                                        show_sources=request.sources)
            result = score(deck, points, conclusion=conclusion, slide_range=slide_range, report=report,
                           embedder=self.reward.embedder, lemmatizer=self.reward.lemmatizer,
                           images_allowed=request.images, tool_text=context["tool_text"], calls=context["calls"],
                           search=search, weights=self.reward.weights)
            components = result.components()
            low, high = result.expected_slides
            n = len(deck.slides)
            block = values[starts[i]:starts[i + 1]]
            block[:, :-3] = (
                *(components[name] for name in _COMPONENTS),
                max(result.total, self.reward.floor), result.uncovered / max(1, len(points)), n,
                n / max(1, low), n / max(1, high), max(0, n - high), low, high, len(points), conclusion,
                search, request.images, event.answer_started, len(context["calls"]),
            )
            block[:, -1] = np.arange(len(block))

        values[:, -3] = np.arange(len(actions)) / self.max_new_tokens
        values[:, -2] = np.maximum(0, self.max_context - actions) / self.max_context
        return PrefixFeatures(values, len(events))


def prepare_prefix_warmup(episodes: Sequence[dict], tasks: Mapping[str, Task],
                          extract: PrefixFeatureExtractor, config: PPOConfig) -> dict:
    """Готовлю буфер PPO на всех сохранённых эпизодах и записываю время каждого пакета."""
    import torch

    from .ppo import sample_positions

    per_mode = config.tasks_per_mode * config.group_size
    episodes = sorted(episodes, key=lambda e: e["step"])  # порядок внутри шага сохраняется
    ids = {(e["step"], e["mode"], e["task"], e["sample"]) for e in episodes}
    batches = {step: list(group) for step, group in groupby(episodes, key=lambda e: e["step"])}
    if len(ids) != len(episodes) or set(batches) != set(range(1, config.warmup_steps + 1)):
        raise ValueError("в разогреве пропущены шаги или продублированы эпизоды")
    if any(len(batch) != 2 * per_mode or sum(e["mode"] == "plain" for e in batch) != per_mode
           or sum(e["mode"] == "search" for e in batch) != per_mode for batch in batches.values()):
        raise ValueError(f"нужен полный разогрев: по {per_mode} эпизодов каждого режима на шаг")

    counts = [sum(e["mask"][1:]) for e in episodes]
    keep = sample_positions(counts, config.warmup_max_positions, stable_seed(config.seed, "warmup-positions"))
    data = {"features": [], "positions": [], "counts": [], "rewards": [], "tasks": [], "modes": [],
            "timing": [], "feature_names": list(FEATURE_NAMES)}
    for step, rows in groupby(zip(episodes, keep, counts), key=lambda row: row[0]["step"]):
        began = perf_counter()
        score_calls = 0
        for episode, positions, count in rows:
            if not positions:
                continue
            task = mode_task(tasks[episode["task"]], episode["mode"])
            result = extract(task, episode["ids"], episode["mask"])
            data["features"].append(torch.from_numpy(result.values[positions]))
            data["positions"].append(positions)
            data["counts"].append(count)
            data["rewards"].append(episode["reward"])
            data["tasks"].append(episode["task"])
            data["modes"].append(episode["mode"])
            score_calls += result.score_calls
        seconds = perf_counter() - began
        data["timing"].append({"step": step, "seconds": seconds, "score_calls": score_calls})
        print(f"признаки: шаг {step}/{config.warmup_steps} · {seconds:.1f} с · "
              f"{score_calls} оценок", flush=True)
    return data


def fit_prefix_head(data: dict, config: PPOConfig, holdout: set[str], lr: float = 1e-3, *,
                    kind: str = "linear", reference: Sequence | None = None, use_prefix: bool = True) -> dict:
    """Проверяю голову на Z, на T-lite или на обоих наборах признаков."""
    import torch

    from .ppo import fit_warmup_head, head_values, paired_improvement

    train = [i for i, topic in enumerate(data["tasks"]) if topic not in holdout]
    if not train or len(train) == len(data["features"]):
        raise ValueError("нужны непустые обучающая и отложенная части")
    if not use_prefix and reference is None:
        raise ValueError("для головы на T-lite нужны признаки опорной модели")
    if reference is not None and len(reference) != len(data["features"]):
        raise ValueError("буферы признаков содержат разное число эпизодов")

    scaler = None
    if use_prefix:
        scale, mean = torch.std_mean(torch.cat([data["features"][i] for i in train]), dim=0, unbiased=False)
        scale = torch.where(scale > 1e-8, scale, 1.0)
        features = [(x - mean) / scale for x in data["features"]]
        scaler = {"mean": mean.tolist(), "scale": scale.tolist(), "names": data["feature_names"]}
        if reference is not None:
            features = [torch.cat((h, z), dim=1) for h, z in zip(reference, features)]
    else:
        features = list(reference)
    data = data | {"features": features}
    fit = fit_warmup_head(data, config, kind, lr, holdout)
    fit["feature_scaler"] = scaler
    fit["input_dim"] = features[0].shape[1]

    errors_by_mode: dict[str, list[tuple[int, float]]] = {}
    paired_rows = []
    for x, r, topic, mode in zip(data["features"], data["rewards"], data["tasks"], data["modes"]):
        if topic not in holdout:
            continue
        baseline_error = (r - fit["mode_means"][mode]) ** 2
        vs = head_values(fit["head"], x, config.head_device)
        errors_by_mode.setdefault(mode, []).append((len(vs), baseline_error))
        paired_rows.append({"task": topic, "mode": mode,
                            "d": baseline_error - statistics.fmean((r - v) ** 2 for v in vs)})
    metrics = fit["holdout"]
    metrics["paired_mode"] = paired_improvement(paired_rows)
    for mode, rows in errors_by_mode.items():
        metrics[f"{mode}_all_mse_mode"] = sum(n * error for n, error in rows) / sum(n for n, _ in rows)
    all_rows = [row for rows in errors_by_mode.values() for row in rows]
    metrics["all_mse_mode"] = sum(n * error for n, error in all_rows) / sum(n for n, _ in all_rows)
    return fit


def aligned_reference_features(data: dict, reference: dict, episodes: Sequence[dict], revision: str) -> list:
    """Проверяю, что признаки T-lite и Z относятся к одним эпизодам и действиям."""
    ids_hash = [stable_seed(*e["ids"]) for e in sorted(episodes, key=lambda e: e["step"]) if any(e["mask"][1:])]
    for name in ("positions", "counts", "rewards", "tasks", "modes"):
        if reference.get(name) != data[name]:
            raise ValueError(f"буферы T-lite и Z расходятся по {name}")
    if reference.get("ids_hash") != ids_hash or reference.get("revision") != revision:
        raise ValueError("буфер T-lite рассчитан по другим токенам или ревизии модели")
    if reference.get("layer") != "last_hidden_state":
        raise ValueError("в буфере T-lite нужен последний скрытый слой")
    features = reference.get("features", [])
    if len(features) != len(data["positions"]) or any(
        h.ndim != 2 or len(h) != len(pos) for h, pos in zip(features, data["positions"])
    ):
        raise ValueError("не совпадают размеры буферов T-lite и Z")
    return features


def prefix_warmup_code_hash() -> str:
    """Отпечаток извлечения Z и выбора позиций. Изменения в обучении голов на него не влияют."""
    from hashlib import sha256
    from pathlib import Path

    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return sha256(source.split(b"\ndef fit_prefix_head(", 1)[0]).hexdigest()


def prefix_cache_matches(saved: dict, expected: dict) -> bool:
    """Принимаю старый буфер Z из 5780238 только при том же извлечении и тех же входах."""
    if saved == expected:
        return True
    name = "training/prefix_features.py"
    saved_hash = saved.get("code", {}).get(name)
    if expected["code"][name] != "26bbaf61c5d797548d7cc0ba0023488eecba83446cfb352e5add7eb9a74658c9":
        return False
    if saved_hash not in {
        "78a89104d2b58f4f44df1bd790d61f787a906f7fcd82d4da16f027fb3d939cde",
        "cf37cf026ec8b0eb814d16c931f4d75e6aad0ac93d7d3a48e0208cdead8bff71",
    }:
        return False
    return saved == expected | {"code": expected["code"] | {name: saved_hash}}
