"""Оценка награды на испорченных презентациях (decisions.md, раздел 7, решение 01.11.2026).

Варианты строит ``perturb.py`` (``data/perturbation/variants/<id>.json``) — один раз и без награды. Здесь:
сборка всех вариантов (отчёт сборщика нужен награде и для предусловий), кеш эмбеддингов на диске, награда
по сетке конфигураций, естественный разброс награды по 8 ответам T-lite (σ, пороги m = 0,25·σ и ε = 0,1·σ),
вердикты по парам, сводка по семействам и режимам, вложенная перекрёстная проверка по темам и бутстреп по темам.

Знак компонентов — «как входит в награду»: у штрафов обратный, Δ < 0 всегда значит «хуже».
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .dsl import parse
from .key_points import KeyPoint
from .perturb import task_context
from .rendering.builder import build_presentation
from .rewards import RewardWeights, score

PENALTIES = frozenset({"length", "forbidden_images", "slide_overflow", "render_errors", "fake_quotes", "bad_sources",
                       "format", "text_overflow", "tools", "repeats"})
M_SHARE, EPS_SHARE, SIGMA_FLOOR = 0.25, 0.1, 0.005


def signed(components: dict[str, float]) -> dict[str, float]:
    return {k: (-v if k in PENALTIES else v) for k, v in components.items()}


# --------------------------------------------------------------------------- кеш эмбеддингов

class CachedEmbedder:
    """Эмбеддер sentence-transformers с кешем на диске. Ключ кеша — модель, её ревизия, префиксы, усечение
    (max_seq_length) и нормализация: при смене любой из них векторы считаются заново (другой файл)."""

    def __init__(self, name: str, cache_dir: str | Path, device: str | None = None, batch_size: int = 32,
                 revision: str | None = None):
        from sentence_transformers import SentenceTransformer  # группа зависимостей reward

        self.model = SentenceTransformer(name, device=device, revision=revision)
        self.prefixes = "e5" in name.lower()
        self.batch_size = batch_size
        self.signature = {"model": name, "revision": revision or _snapshot(name),
                          "prefix": "query: / passage: " if self.prefixes else "",
                          "max_seq_length": int(self.model.max_seq_length), "normalize": True}
        digest = hashlib.sha1(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()[:12]
        self.path = Path(cache_dir) / f"{name.replace('/', '__')}_{digest}.npz"
        self.vectors: dict[str, np.ndarray] = {}
        self.added = 0
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self.vectors = dict(zip(data["keys"].tolist(), data["vectors"]))

    def encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        keys = [f"{kind}\t{t}" for t in texts]
        missing = [t for t in dict.fromkeys(texts) if f"{kind}\t{t}" not in self.vectors]
        if missing:
            inputs = [f"{kind}: {t}" if self.prefixes else t for t in missing]
            vectors = self.model.encode(inputs, batch_size=self.batch_size, normalize_embeddings=True)
            for text, vector in zip(missing, vectors):
                self.vectors[f"{kind}\t{text}"] = np.asarray(vector, dtype=np.float32)
            self.added += len(missing)
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self.vectors[k] for k in keys])

    def save(self) -> None:
        if not self.added:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.vectors)
        np.savez(self.path, keys=np.array(keys), vectors=np.stack([self.vectors[k] for k in keys]),
                 signature=json.dumps(self.signature))
        self.added = 0


def _snapshot(name: str) -> str:
    """Коммит локальной копии модели с Hugging Face; «unknown», если узнать нельзя."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(name, local_files_only=True)).name
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- данные

@dataclass
class Item:
    """Презентация для оценки: текст, контекст награды и результат сборки."""

    key: tuple  # (task, id варианта) или ("natural", run, файл) или ("conflict", id, "a"/"b")
    task: str
    text: str
    context: dict
    deck: object = None
    report: object = None


def load_variants(folder: str | Path) -> tuple[dict[tuple, Item], list[dict]]:
    """Варианты и пары из ``data/perturbation/variants``; у пары — режим поиска исходника."""
    items, pairs = {}, []
    for path in sorted(Path(folder).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        task, base = record["task"], record["context"]
        for vid, variant in record["variants"].items():
            items[(task, vid)] = Item((task, vid), task, variant["text"], {**base, **variant["context"]})
        for pair in record["pairs"]:
            pairs.append({**pair, "task": task, "mode": "search" if base["search"] else "plain"})
    return items, pairs


def load_natural(run_dirs: Iterable[str | Path], tasks: dict, with_points: set[str]) -> dict[tuple, Item]:
    """Ответы T-lite из прогонов (``<mode>_<id>_<s>_answer.txt`` и полный эпизод) — для естественного разброса."""
    items = {}
    for run in run_dirs:
        run = Path(run)
        for answer in sorted(run.glob("*_answer.txt")):
            mode, task, sample = answer.name[:-len("_answer.txt")].split("_")
            if task not in with_points or task not in tasks:
                continue
            episode = answer.with_name(f"{mode}_{task}_{sample}.txt")
            if not episode.exists():
                continue
            context = task_context(tasks[task], episode.read_text(encoding="utf-8"), mode == "search")
            key = ("natural", run.name, mode, task, sample)
            items[key] = Item(key, task, answer.read_text(encoding="utf-8"), context)
    return items


def _side(better: str, first: str) -> str | None:
    """Ответ судьи («1», «2», «равны») → презентация пары: «a», «b» или «equal»; None — ответа нет."""
    better = str(better).strip().lower()
    second = "b" if first == "a" else "a"
    if better == "1":
        return first
    if better == "2":
        return second
    if better in ("равны", "=", "0", "equal"):
        return "equal"
    return None


def load_conflicts(folder: str | Path, contexts: dict[str, dict]) -> tuple[dict[tuple, Item], list[dict]]:
    """Пары с конфликтом достоинств и решения судей.

    Ответы — ``answers_<судья>.json`` (номер пары, «1»/«2»/«равны») в порядке ``key.json``; ответы того же судьи
    при обратном порядке презентаций — ``answers_<судья>_reversed.json``. Решение судьи по паре — «a», «b»,
    «equal» или None: судья не определился (в двух порядках выбрал разное) или не ответил."""
    folder = Path(folder)
    conflicts = json.loads((folder / "pairs.json").read_text(encoding="utf-8"))
    items = {}
    for pair in conflicts:
        for side in ("a", "b"):
            key = ("conflict", pair["id"], side)
            items[key] = Item(key, pair["task"], pair[side], dict(contexts[pair["task"]]))
    key_path = folder / "key.json"
    judged: dict[str, dict] = {}
    if key_path.exists():
        order = {k["pair"]: k for k in json.loads(key_path.read_text(encoding="utf-8"))}
        for path in sorted(folder.glob("answers_*.json")):
            name = path.stem[len("answers_"):]
            if name.endswith("_reversed"):
                continue
            answers = {str(a["pair"]).zfill(2): a for a in json.loads(path.read_text(encoding="utf-8"))}
            reversed_path = folder / f"answers_{name}_reversed.json"
            back = {str(a["pair"]).zfill(2): a for a in json.loads(reversed_path.read_text(encoding="utf-8"))} \
                if reversed_path.exists() else None
            for number, answer in answers.items():
                entry = order[number]
                choice = _side(answer["better"], entry["first"])
                if back is not None:
                    flipped = "b" if entry["first"] == "a" else "a"
                    other = _side(back[number]["better"], flipped) if number in back else None
                    choice = choice if choice == other else None
                judged.setdefault(entry["id"], {})[name] = choice
    for pair in conflicts:
        pair["judges"] = judged.get(pair["id"], {})
    return items, conflicts


def combine_judges(conflicts: list[dict], primary: str, secondary: str, secondary_kinds: Iterable[str],
                   name: str = "judge") -> None:
    """Сводный судья ``name``: решение ``primary`` там, где он отвечал (его «равны» — тоже его ответ), иначе —
    ``secondary``, но только на парах видов ``secondary_kinds`` (решение 01.11.2026: автор и Codex на повторах)."""
    kinds = set(secondary_kinds)
    for pair in conflicts:
        judges = pair["judges"]
        if primary in judges:
            judges[name] = judges[primary]
        elif pair["kind"] in kinds and secondary in judges:
            judges[name] = judges[secondary]


def author_answers(path: str | Path) -> list[dict]:
    """Ответы автора из текстового файла: строки «02: 1 — причина» или «02: 2. причина» (пустой ответ
    пропускается)."""
    import re

    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        head, sep, rest = line.partition(":")
        if not sep or not head.strip().isdigit():
            continue
        match = re.match(r"\s*(1|2|равны)\b[\s.,;:—–-]*(.*)$", rest, re.IGNORECASE)
        if match:
            result.append({"pair": head.strip().zfill(2), "better": match.group(1).lower(),
                           "reason": match.group(2).strip()})
    return result


def build_all(items: dict[tuple, Item], log: Callable[[str], None] = print) -> None:
    """Собирает каждую презентацию один раз (pptx во временную папку): отчёт сборщика нужен награде."""
    folder = Path(tempfile.mkdtemp())
    for k, item in enumerate(items.values()):
        if item.deck is not None:
            continue
        item.deck = parse(item.text)
        sources = item.context.get("sources") or {}
        item.report = build_presentation(item.deck, folder / "deck.pptx", None, sources.get, True)
        if (k + 1) % 200 == 0:
            log(f"собрано {k + 1} из {len(items)}")


def check_pairs(pairs: list[dict], items: dict[tuple, Item]) -> tuple[list[dict], list[str]]:
    """Предусловия по отчёту сборщика: пара, где задуманный дефект не получился, отбрасывается до оценки."""
    kept, dropped = [], []
    for pair in pairs:
        item = items[(pair["task"], pair["variant"])]
        checks = pair.get("checks") or {}
        problem = ""
        if checks.get("parse_errors") and not item.deck.errors:
            problem = "нет ошибок разбора"
        if "status" in checks and not any(i.status == checks["status"] for i in item.report.items):
            problem = f"нет статуса {checks['status']}"
        if problem:
            dropped.append(f"{pair['id']}: {problem}")
        else:
            kept.append(pair)
    return kept, dropped


# --------------------------------------------------------------------------- награда по сетке

@dataclass(frozen=True)
class Config:
    embedder: str
    alpha: float
    length: float
    keys: float  # вес ключей при весе покрытия 1

    def weights(self, base: RewardWeights | None = None) -> RewardWeights:
        return replace(base or RewardWeights(), alpha=self.alpha, length=self.length, keys=self.keys)

    @property
    def name(self) -> str:
        return f"{self.embedder.split('/')[-1]}|a={self.alpha}|len={self.length}|keys={self.keys}"


def grid(embedders: Sequence[str], alphas: Sequence[float], lengths: Sequence[float],
         keys: Sequence[float]) -> list[Config]:
    return [Config(e, a, l, k) for e, a, l, k in itertools.product(embedders, alphas, lengths, keys)]


def score_items(items: dict[tuple, Item], points: dict[str, list[KeyPoint]], config: Config, embedder,
                lemmatizer) -> dict[tuple, dict[str, float]]:
    """Компоненты награды (со знаком «как в награде») для каждой презентации при конфигурации ``config``."""
    weights = config.weights()
    result = {}
    for key, item in items.items():
        c = item.context
        slide_range = tuple(c["slide_range"]) if c.get("slide_range") else None
        r = score(item.deck, points[item.task], conclusion=c["conclusion"], slide_range=slide_range,
                  embedder=embedder, lemmatizer=lemmatizer, report=item.report, images_allowed=c["images_allowed"],
                  tool_text=c.get("tool_text"), calls=c.get("calls"), search=c["search"], weights=weights)
        result[key] = signed(r.components())
    return result


def sigma(scores: dict[tuple, dict[str, float]]) -> tuple[float, list[dict]]:
    """σ — медиана стандартного отклонения награды внутри группы (прогон, режим, тема) по ответам T-lite."""
    groups: dict[tuple, list[float]] = {}
    for key, comp in scores.items():
        if key[0] == "natural":
            groups.setdefault(key[1:4], []).append(comp["total"])
    rows = [{"run": g[0], "mode": g[1], "task": g[2], "n": len(v), "std": statistics.pstdev(v),
             "mean": statistics.mean(v)} for g, v in groups.items() if len(v) >= 2]
    value = statistics.median(r["std"] for r in rows) if rows else 0.0
    return max(SIGMA_FLOOR, value), rows


# --------------------------------------------------------------------------- вердикты

def verdict(delta: float, expect: str | None, m: float, eps: float) -> str | None:
    """ok — ожидание выполнено; weak — направление верное, но сдвиг меньше m; tie — |Δ| ≤ ε при ожидании
    падения; violation — нарушено. None — ожидания нет."""
    if expect is None:
        return None
    if expect == "down":
        return "ok" if delta <= -m else "tie" if abs(delta) <= eps else "weak" if delta < 0 else "violation"
    if expect == "up":
        return "ok" if delta >= m else "tie" if abs(delta) <= eps else "weak" if delta > 0 else "violation"
    if expect == "not_up":
        return "ok" if delta <= eps else "violation"
    if expect == "same":
        return "ok" if abs(delta) <= eps else "violation"
    raise ValueError(f"неизвестное ожидание {expect}")


def evaluate_pairs(pairs: list[dict], scores: dict[tuple, dict[str, float]], m: float, eps: float) -> list[dict]:
    rows = []
    for pair in pairs:
        base, variant = scores[(pair["task"], pair["base"])], scores[(pair["task"], pair["variant"])]
        d_total = variant["total"] - base["total"]
        d_target = variant[pair["target"]] - base[pair["target"]] if pair["target"] else None
        row = {k: pair[k] for k in ("id", "task", "mode", "family", "kind", "strength", "target", "diagnostic")}
        row.update(d_total=d_total, d_target=d_target,
                   total=verdict(d_total, pair["expect_total"], m, eps),
                   component=verdict(d_target, pair["expect_target"], m, eps) if d_target is not None else None)
        # компонент упал (как задумано), а награда выросла
        row["component_down_total_up"] = bool(d_target is not None and d_target <= -m and d_total > eps)
        rows.append(row)
    return rows


def outcome(delta: float, eps: float) -> str:
    """Выбор награды в паре с конфликтом: «a», «b» или «equal» (|Δ| ≤ ε)."""
    return "a" if delta > eps else "b" if delta < -eps else "equal"


def conflict_rows(conflicts: list[dict], scores: dict[tuple, dict[str, float]], judge: str, eps: float) -> list[dict]:
    """Совпадает ли выбор награды (a / b / равны) с выбором судьи. Пропускаются пары, где судья не определился
    или ответил «равны» (решение 01.11.2026: «равны» значит «всё равно», и ничьей от награды не требуем)."""
    rows = []
    for pair in conflicts:
        prefer = pair["judges"].get(judge)
        if prefer in (None, "equal"):
            continue
        d = scores[("conflict", pair["id"], "a")]["total"] - scores[("conflict", pair["id"], "b")]["total"]
        reward = outcome(d, eps)
        rows.append({"id": pair["id"], "task": pair["task"], "kind": pair["kind"], "decides": pair["decides"],
                     "d_a_minus_b": d, "judge": prefer, "reward": reward, "agree": reward == prefer,
                     "agree_soft": 1.0 if reward == prefer else 0.5 if "equal" in (reward, prefer) else 0.0})
    return rows


def judges_agreement(conflicts: list[dict], first: str, second: str) -> tuple[int, int]:
    """Сколько пар оба судьи решили одинаково и сколько решили оба (без «не определился» и «равны»)."""
    both = [c for c in conflicts if c["judges"].get(first) not in (None, "equal")
            and c["judges"].get(second) not in (None, "equal")]
    return sum(c["judges"][first] == c["judges"][second] for c in both), len(both)


def pair_success(row: dict) -> float | None:
    """Выполнение ожиданий пары: среднее по компоненту и всей награде; выполнено — 1, «слабо» — 0,5, иначе 0.
    None — ожиданий нет или пара диагностическая."""
    if row["diagnostic"]:
        return None
    marks = [1.0 if v == "ok" else 0.5 if v == "weak" else 0.0 for v in (row["component"], row["total"])
             if v is not None]
    return sum(marks) / len(marks) if marks else None


def summarize(rows: list[dict], by: Sequence[str] = ("family", "kind", "mode")) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in by), []).append(row)
    out = []
    for key, part in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        totals = [r["d_total"] for r in part]
        targets = [r["d_target"] for r in part if r["d_target"] is not None]
        entry = dict(zip(by, key))
        entry.update(n=len(part), d_total_median=statistics.median(totals), d_total_mean=statistics.mean(totals),
                     d_target_median=statistics.median(targets) if targets else None,
                     d_target_mean=statistics.mean(targets) if targets else None,
                     component_down_total_up=sum(r["component_down_total_up"] for r in part))
        for name in ("total", "component"):
            marks = [r[name] for r in part if r[name] is not None]
            for state in ("ok", "weak", "tie", "violation"):
                entry[f"{name}_{state}"] = sum(v == state for v in marks) / len(marks) if marks else None
        out.append(entry)
    return out


# --------------------------------------------------------------------------- выбор конфигурации

CORE = frozenset({"coverage", "keys", "length", "invariance"})  # главное: порог провала строже
REPORT_ONLY = frozenset({"chain"})  # цепочки повторяют формат, достоверность и инструменты — только в отчёт
CURRENT_WEIGHTS = (1.0, 0.5)  # текущие λ и вес ключей


@dataclass(frozen=True)
class Rules:
    """Правила выбора, зафиксированные до запуска (decisions.md, раздел 7, 01.11.2026)."""

    fail_core: float = 0.7  # семейство из CORE провалено, если ожидания выполнены меньше чем на 0,7
    fail_other: float = 0.5
    close_scale: float = 1.0  # «близкие» — в пределах close_scale · половины ширины бутстреп-интервала лучшей
    boot: int = 300
    seed: int = 0


def tally(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Тема → семейство → [сумма выполнения, число пар] (без диагностики и семейств только для отчёта)."""
    out: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = pair_success(row)
        if value is None or row["family"] in REPORT_ONLY:
            continue
        cell = out.setdefault(row["task"], {}).setdefault(row["family"], [0.0, 0])
        cell[0] += value
        cell[1] += 1
    return out


def conflict_tally(rows: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in rows:
        cell = out.setdefault(row["task"], [0.0, 0])
        cell[0] += row["agree"]
        cell[1] += 1
    return out


def family_shares(table: dict, weights: dict[str, int]) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for task, count in weights.items():
        for family, (s, n) in table.get(task, {}).items():
            cell = sums.setdefault(family, [0.0, 0])
            cell[0] += count * s
            cell[1] += count * n
    return {f: s / n for f, (s, n) in sums.items() if n}


def agreement(table: dict, weights: dict[str, int]) -> float:
    s = sum(c * table[t][0] for t, c in weights.items() if t in table)
    n = sum(c * table[t][1] for t, c in weights.items() if t in table)
    return s / n if n else float("nan")


def failed_families(families: dict[str, float], rules: Rules) -> set[str]:
    return {f for f, v in families.items() if v < (rules.fail_core if f in CORE else rules.fail_other)}


def criterion(families: dict[str, float], exempt: set[str], rules: Rules = Rules()) -> float:
    """−inf — провалено семейство (кроме проваленных всеми конфигурациями); иначе средняя доля по семействам."""
    if failed_families(families, rules) - exempt:
        return -math.inf
    return statistics.mean(families.values()) if families else -math.inf


def _neighbors(config: Config, configs: Sequence[Config]) -> list[Config]:
    axes = {f: sorted({getattr(c, f) for c in configs}) for f in ("embedder", "alpha", "length", "keys")}
    out = []
    for c in configs:
        diff = [f for f in axes if getattr(c, f) != getattr(config, f)]
        if len(diff) == 1 and diff[0] != "embedder":
            values = axes[diff[0]]
            if abs(values.index(getattr(c, diff[0])) - values.index(getattr(config, diff[0]))) == 1:
                out.append(c)
    return out


def _blocks_score(fn: Callable[[dict[str, int]], float], blocks: Sequence[Sequence[str]]) -> float:
    values = [fn({t: 1 for t in block}) for block in blocks]
    values = [v for v in values if not math.isnan(v)]
    return statistics.mean(values) if values else float("nan")


def _resample(tasks: Sequence[str], rng: random.Random) -> dict[str, int]:
    drawn = [rng.choice(list(tasks)) for _ in tasks]
    return {t: drawn.count(t) for t in set(drawn)}


def choose(tallies: dict[Config, dict], conflicts: dict[Config, dict] | None, blocks: Sequence[Sequence[str]],
           rules: Rules = Rules()) -> tuple[Config, dict]:
    """Процедура выбора на темах ``blocks`` (оценка конфигурации — среднее по блокам).

    1) Допустимые — без проваленных семейств (семейства, проваленные всеми, — ограничение награды, не отсев).
    2) Среди допустимых — по совпадению с судьёй (``conflicts``); без судьи — по критерию искажений, λ и вес
       ключей текущие. 3) Близкие к лучшей — в пределах close_scale · половины ширины бутстреп-интервала; из них —
       с лучшим средним по соседям в сетке, при равенстве — по критерию искажений. 4) С судьёй: новые λ и вес
       ключей — только если бутстреп-интервал разницы совпадения с текущими (те же эмбеддер и α) выше нуля."""
    configs = list(tallies)
    tasks = sorted({t for block in blocks for t in block})
    weights = {t: 1 for t in tasks}
    shares = {c: family_shares(tallies[c], weights) for c in configs}
    names = set().union(*(s.keys() for s in shares.values()))
    exempt = {f for f in names if all(f in failed_families(shares[c], rules) for c in configs)}
    crit = {c: criterion(shares[c], exempt, rules) for c in configs}  # допустимость — по всем темам вместе
    feasible = [c for c in configs if crit[c] > -math.inf]
    info = {"exempt": sorted(exempt), "feasible": len(feasible), "judge": conflicts is not None}
    if not feasible:
        feasible = configs
        info["note"] = "допустимых нет: выбор среди всех"
    if conflicts is not None:
        def value(c: Config, w: dict[str, int]) -> float:
            return agreement(conflicts[c], w)
    else:
        current = [c for c in feasible if (c.length, c.keys) == CURRENT_WEIGHTS]
        feasible = current or feasible

        def value(c: Config, w: dict[str, int]) -> float:
            shares_w = family_shares(tallies[c], w)
            return statistics.mean(shares_w.values()) if shares_w else float("nan")
    score = {c: _blocks_score(lambda w, c=c: value(c, w), blocks) for c in feasible}
    best = max(feasible, key=lambda c: (score[c], crit[c]))
    rng = random.Random(rules.seed)
    samples = sorted(value(best, _resample(tasks, rng)) for _ in range(rules.boot))
    samples = [s for s in samples if s > -math.inf and not math.isnan(s)] or [score[best]]
    half = (samples[int(0.975 * (len(samples) - 1))] - samples[int(0.025 * (len(samples) - 1))]) / 2
    close = [c for c in feasible if score[c] >= score[best] - rules.close_scale * half]

    def stability(c: Config) -> float:
        around = [score[n] for n in _neighbors(c, feasible) if n in score]
        return statistics.mean(around + [score[c]])

    chosen = max(close, key=lambda c: (stability(c), crit[c]))
    info.update(best=best.name, close=[c.name for c in close], halfwidth=half)
    if conflicts is not None and (chosen.length, chosen.keys) != CURRENT_WEIGHTS:
        base = Config(chosen.embedder, chosen.alpha, *CURRENT_WEIGHTS)
        if base in score:
            rng = random.Random(rules.seed + 1)
            diffs = []
            for _ in range(rules.boot):
                w = _resample(tasks, rng)
                diffs.append(value(chosen, w) - value(base, w))
            diffs.sort()
            low = diffs[int(0.025 * (len(diffs) - 1))]
            info["vs_current_low"] = low
            if not low > 0:
                info["kept_current"] = chosen.name
                chosen = base
    info["chosen"] = chosen.name
    return chosen, info


def stratified_folds(modes: dict[str, str], k: int, seed: int) -> list[list[str]]:
    """Разбиение тем на k блоков с поровну тем каждого режима поиска."""
    rng = random.Random(seed)
    result: list[list[str]] = [[] for _ in range(k)]
    position = 0
    for mode in sorted(set(modes.values())):
        group = sorted(t for t, m in modes.items() if m == mode)
        rng.shuffle(group)
        for task in group:
            result[position % k].append(task)
            position += 1
    return result


def nested_cv(tallies: dict[Config, dict], conflicts: dict[Config, dict] | None, modes: dict[str, str],
              outer: int = 4, inner: int = 3, rules: Rules = Rules()) -> dict:
    """Вложенная проверка по темам: вся процедура выбора (``choose``) — на внутренних блоках обучающих тем,
    оценка выбранной конфигурации — на отложенных темах. Итог — ``choose`` на всех темах (блоки — внешние)."""
    blocks = stratified_folds(modes, outer, rules.seed)
    report: dict = {"outer": []}
    for k, test in enumerate(blocks):
        train = {t: m for t, m in modes.items() if t not in test}
        chosen, info = choose(tallies, conflicts, stratified_folds(train, inner, rules.seed + k + 1), rules)
        weights = {t: 1 for t in test}
        exempt = set(info["exempt"])
        row = {"test": test, "chosen": chosen.name,
               "test_criterion": criterion(family_shares(tallies[chosen], weights), exempt, rules),
               "train_criterion": criterion(family_shares(tallies[chosen], {t: 1 for t in train}), exempt, rules)}
        if conflicts is not None:
            row["test_agreement"] = agreement(conflicts[chosen], weights)
            row["train_agreement"] = agreement(conflicts[chosen], {t: 1 for t in train})
        report["outer"].append(row)
    final, info = choose(tallies, conflicts, blocks, rules)
    report["final"], report["final_info"] = final.name, info
    names = [o["chosen"] for o in report["outer"]]
    configs = {c.name: c for c in tallies}
    near = [n for n in names if n == final.name or configs[n] in _neighbors(final, list(tallies))]
    gaps = [o["train_criterion"] - o["test_criterion"] for o in report["outer"]
            if o["train_criterion"] > -math.inf and o["test_criterion"] > -math.inf]
    report["stable"] = len(near) >= outer - 1  # та же конфигурация или соседняя хотя бы в 3 из 4
    report["optimism"] = statistics.mean(gaps) if gaps else None
    report["reliable"] = bool(report["stable"] and report["optimism"] is not None and report["optimism"] <= 0.05)
    return report


def bootstrap(table: dict, tasks: Sequence[str], n: int = 1000, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Бутстреп по темам: 2,5%, медиана и 97,5% доли выполненных ожиданий по семействам."""
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n):
        for family, value in family_shares(table, _resample(tasks, rng)).items():
            samples.setdefault(family, []).append(value)
    out = {}
    for family, values in samples.items():
        values.sort()
        out[family] = (values[int(0.025 * (len(values) - 1))], statistics.median(values),
                       values[int(0.975 * (len(values) - 1))])
    return out


def mode_failures(rows: list[dict], rules: Rules = Rules()) -> list[str]:
    """Предупреждения: семейство провалено в одном режиме поиска (в отсев не идёт)."""
    out = []
    for mode in sorted({r["mode"] for r in rows}):
        table = tally([r for r in rows if r["mode"] == mode])
        shares = family_shares(table, {t: 1 for t in table})
        out += [f"{f} ({mode}): {shares[f]:.2f}" for f in sorted(failed_families(shares, rules))]
    return out


def identifiable(conflict_by: dict[Config, list[dict]]) -> list[dict]:
    """Различают ли пары с конфликтом λ и вес ключей: число разных наборов выборов награды при тех же эмбеддере
    и α (1 — параметры по этим парам не выбрать)."""
    groups: dict[tuple, set] = {}
    for config, rows in conflict_by.items():
        groups.setdefault((config.embedder, config.alpha), set()).add(tuple(r["reward"] for r in rows))
    return [{"embedder": e, "alpha": a, "distinct_rankings": len(v)} for (e, a), v in sorted(groups.items())]


EXTENSION = {"alpha": {0.4: 0.6}, "length": {0.5: 0.25, 2.0: 4.0}, "keys": {0.5: 0.25, 1.0: 1.5}}


def edge(config: Config, configs: Sequence[Config], axes: Sequence[str] = ("alpha", "length", "keys")) -> dict[str, float]:
    """Оси, по которым выбранная конфигурация на краю сетки, и значение для расширения (одно расширение).
    ``axes`` — только оси, которые действительно выбирались (без судьи λ и вес ключей не выбираются)."""
    out = {}
    for axis, extra in EXTENSION.items():
        if axis not in axes:
            continue
        values = sorted({getattr(c, axis) for c in configs})
        value = getattr(config, axis)
        if len(values) > 1 and value in (values[0], values[-1]) and value in extra and extra[value] not in values:
            out[axis] = extra[value]
    return out
