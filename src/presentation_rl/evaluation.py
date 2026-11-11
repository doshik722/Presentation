# Здесь применяем общие метрики к сохранённым ответам без повторной генерации.
"""Оценка сохранённых эпизодов: формат, слайды, поиск, картинки, награда, признаки вырождения; парное сравнение
моделей по темам с интервалом бутстрепа. Эпизод на диске — ``<режим>_<id>_<номер>``: ``.json`` (конец, вызовы,
ошибка, время), ``.txt`` (эпизод после промпта), ``_answer.txt`` (ответ). Так пишут ``notebooks/sft_eval.ipynb``
и ``notebooks/sft_select.ipynb``."""
from __future__ import annotations

import collections
import csv
import json
import random
import re
import statistics
from pathlib import Path

CONCLUSION = re.compile(r"^\s*(вывод|выводы|заключение|итог|итоги)", re.IGNORECASE)
RESPONSE = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.DOTALL)
CYRILLIC = re.compile("[а-яё]", re.IGNORECASE)
LATIN = re.compile("[a-z]", re.IGNORECASE)
CJK = re.compile("[぀-ヿ一-鿿]")
WORD = re.compile(r"[^\W\d_]+")


def search_metrics(calls: list[dict], responses: list[str]) -> dict:
    """Запросы к Википедии: сколько вызовов каждого инструмента, доля запросов на английском и с найденным."""
    searches = [(c, r) for c, r in zip(calls, responses + [""] * len(calls)) if c["name"] == "search" and not c["error"]]
    return {"calls_search": len(searches), "calls_open": sum(c["name"] == "open" for c in calls),
            "calls_read": sum(c["name"] == "read" for c in calls),
            "queries_en": statistics.mean(not CYRILLIC.search(c["argument"] or "") for c, _ in searches) if searches else None,
            "queries_found": statistics.mean("ничего не найдено" not in r for _, r in searches) if searches else None}


def image_metrics(deck, embedder) -> dict:
    """Картинки: сколько, длина описания и доля связанных со слайдом (порог награды для картинок)."""
    from .rewards import VISUAL_TAU, pairwise_cosine, slide_body

    images = [(e.data.get("text", ""), slide_body(s, set())) for s in deck.slides for e in s.elements if e.kind == "IMAGE"]
    if not images:
        return {"images": 0, "image_words": None, "images_related": None}
    sims = pairwise_cosine(embedder, [t for t, _ in images], [b for _, b in images])
    return {"images": len(images), "image_words": statistics.mean(len(t.split()) for t, _ in images),
            "images_related": statistics.mean(float(v) >= VISUAL_TAU["IMAGE"] for v in sims)}


def degeneration(answer: str, finish: str) -> dict:
    """Признаки вырождения ответа (EXP-017): испорченные слова (смесь кириллицы и латиницы в одном слове, иероглифы),
    самый частый повтор 8 слов подряд, доля повторённых строк длиннее 20 знаков, ответ не закончен. Вырожденный —
    не закончен, или повтор 8 слов не меньше 4 раз, или испорченных слов не меньше 5."""
    words = WORD.findall(answer)
    broken = sum((bool(CYRILLIC.search(w)) and bool(LATIN.search(w))) or bool(CJK.search(w)) for w in words)
    tokens = re.findall(r"\w+", answer.lower())
    grams = collections.Counter(tuple(tokens[i:i + 8]) for i in range(len(tokens) - 7))
    loop = max(grams.values(), default=0)
    lines = [line.strip() for line in answer.splitlines() if len(line.strip()) > 20]
    repeated = (len(lines) - len(set(lines))) / len(lines) if lines else 0.0
    unfinished = finish != "stop"
    return {"broken_words": broken, "glitch": broken > 0, "loop_8gram": loop, "repeated_lines": repeated,
            "unfinished": unfinished, "degenerate": unfinished or loop >= 4 or broken >= 5}


def evaluate_folder(folder: str | Path, model: str, tasks: dict, points: dict, embedder, lemmatizer,
                    build_pptx: bool = True) -> list[dict]:
    """Строка на каждый сохранённый эпизод папки: метрики формата, поиска, картинок, вырождения и компоненты награды
    (для тем с тезисами). Ошибочные эпизоды остаются строкой с ``error``."""
    from .dsl import parse
    from .perturb import task_context
    from .rendering import build_presentation, pick_theme
    from .rewards import score

    folder = Path(folder)
    rows = []
    for meta in sorted(folder.glob("*_*_*.json")):
        row = {"model": model, **json.loads(meta.read_text(encoding="utf-8"))}
        rows.append(row)
        if row.get("error"):
            continue
        task, stem = tasks[row["id"]], meta.stem
        text = (folder / f"{stem}.txt").read_text(encoding="utf-8")
        answer = (folder / f"{stem}_answer.txt").read_text(encoding="utf-8")
        context = task_context(task, text, row["mode"] == "search")
        deck = parse(answer)
        report = build_presentation(deck, folder / f"{stem}.pptx" if build_pptx else None,
                                    pick_theme(task.request.theme, task.request.text),
                                    sources=(context["sources"] or {}).get, show_sources=task.request.sources)
        n = len(deck.slides)
        statuses = [i.status for i in report.items if i.kind == "SOURCE"]
        row.update(parse_rate=deck.parse_rate, slides=n, stop=row["finish"] == "stop",
                   conclusion=bool(n >= 3 and CONCLUSION.match(deck.slides[-1].title or "")),
                   searched=row["mode"] == "search" and row["calls"] > 0,
                   calls_in_plain=row["mode"] == "plain" and row["calls"] > 0,
                   sources_ok=statuses.count("ok"), sources_bad=statuses.count("bad_source"),
                   slides_with_source=len({i.slide for i in report.items if i.kind == "SOURCE" and i.status == "ok"}) / n
                   if n else 0.0, **image_metrics(deck, embedder), **degeneration(answer, row["finish"]))
        if row["mode"] == "search":
            row.update(search_metrics(context["calls"], RESPONSE.findall(text)))
        if row["id"] in points:
            slide_range = tuple(context["slide_range"]) if context["slide_range"] else None
            r = score(deck, points[row["id"]], conclusion=context["conclusion"], slide_range=slide_range,
                      embedder=embedder, lemmatizer=lemmatizer, report=report, images_allowed=context["images_allowed"],
                      tool_text=context["tool_text"], calls=context["calls"], search=context["search"])
            low, high = r.expected_slides
            row.update(in_range=low <= n <= high, **{f"reward_{k}": v for k, v in r.components().items()})
    return rows


def topic_means(rows: list[dict], model: str, metric: str, mode: str) -> dict[str, float]:
    """Среднее метрики по ответам каждой темы (все номера ответов) для модели и режима."""
    values: dict[str, list[float]] = {}
    for r in rows:
        if r["model"] == model and r["mode"] == mode and not r.get("error") and r.get(metric) is not None:
            values.setdefault(r["id"], []).append(float(r[metric]))
    return {k: statistics.mean(v) for k, v in values.items()}


def paired(rows: list[dict], metric: str, mode: str, base: str, other: str, n_boot: int = 2000,
           seed: int = 0) -> dict | None:
    """Разница ``other − base`` по темам (среднее ответов темы у каждой модели) с 95% интервалом бутстрепа по темам."""
    a, b = topic_means(rows, base, metric, mode), topic_means(rows, other, metric, mode)
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [b[k] - a[k] for k in keys]
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(n_boot))
    return {"mode": mode, "metric": metric, "n": len(keys), "base": statistics.mean(a[k] for k in keys),
            "other": statistics.mean(b[k] for k in keys), "diff": statistics.mean(diffs),
            "low": boot[int(0.025 * n_boot)], "high": boot[int(0.975 * n_boot)]}


def reward_spread(rows: list[dict], model: str, mode: str = "plain") -> float | None:
    """Разброс награды внутри темы: стандартное отклонение по ответам темы, среднее по темам (на нём учится GRPO)."""
    values: dict[str, list[float]] = {}
    for r in rows:
        if r["model"] == model and r["mode"] == mode and not r.get("error") and r.get("reward_total") is not None:
            values.setdefault(r["id"], []).append(float(r["reward_total"]))
    spreads = [statistics.pstdev(v) for v in values.values() if len(v) >= 2]
    return statistics.mean(spreads) if spreads else None


def write_csv(path: str | Path, items: list[dict]) -> None:
    items = list(items)
    if not items:
        return
    fields = list(dict.fromkeys(k for r in items for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig — для Excel
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(items)
