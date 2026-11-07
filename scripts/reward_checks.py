# Здесь проверяем на сервере суженное числовое противоречие и то, находит ли награда слайд, где раскрыт тезис.
"""Две проверки награды (решение 07.11.2026, после второго прогона EXP-016).

1. Числовое противоречие: все срабатывания прежнего правила и суженного (другое число той же единицы, которое
   само есть в ключах темы, противоречием не считается) на ответах T-lite и исходниках — с текстом слайдов тезиса,
   для ручной разметки «настоящая ошибка / другая величина»; и сколько искажений «неверное число» ловит каждое.
2. Слайд тезиса: на размеченных исходниках (16 + 4 новых) — совпадает ли лучший слайд награды с размеченным
   и попадает ли размеченный слайд в слайды, где ищутся ключи.

Запуск из корня репозитория: ``python scripts/reward_checks.py``. Результат — ``runs/reward_checks/<дата>/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from presentation_rl import perturb_eval as pe  # noqa: E402
from presentation_rl.rewards import (Lemmatizer, RewardWeights, contradicts, load_key_points, number_units,  # noqa: E402
                                     score, slide_texts)
from presentation_rl.tasks import load_tasks  # noqa: E402

RUNS = ["runs/variance/20261020_0331", "runs/reward_sources/20261027_2150"]
SETS = ["data/perturbation", "data/perturbation/holdout"]
CONTRADICTION_CONFIG = ("BAAI/bge-m3", 0.2)
SLIDE_CONFIGS = [("BAAI/bge-m3", 0.2), ("BAAI/bge-m3", 0.0), ("intfloat/multilingual-e5-small", 0.0),
                 ("intfloat/multilingual-e5-small", 0.2)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def run_score(item, points, embedder, lemmatizer, alpha):
    c = item.context
    slide_range = tuple(c["slide_range"]) if c.get("slide_range") else None
    return score(item.deck, points[item.task], conclusion=c["conclusion"], slide_range=slide_range, embedder=embedder,
                 lemmatizer=lemmatizer, report=item.report, images_allowed=c["images_allowed"],
                 tool_text=c.get("tool_text"), calls=c.get("calls"), search=c["search"],
                 weights=RewardWeights(alpha=alpha))


def contradiction_flags(item, points, result, lemmatizer) -> list[dict]:
    """Срабатывания прежнего и суженного правила по ключам с числом и единицей."""
    texts = slide_texts(item.deck, item.report)
    pairs = [number_units(lemmatizer.tokens(t.full)) for t in texts]
    allowed = set().union(*(number_units(lemmatizer.tokens(k)) for p in points for k in p.key))
    rows = []
    for point, thesis in zip(points, result.theses):
        here = set().union(*(pairs[s - 1] for s in thesis.key_slides)) if thesis.key_slides else set()
        for key in dict.fromkeys(point.key):
            key_pairs = number_units(lemmatizer.tokens(key))
            if not key_pairs:
                continue
            old, new = contradicts(key_pairs, here), contradicts(key_pairs, here, allowed)
            if old or new:
                units = {u for _, u in key_pairs}
                rows.append({"thesis": point.title, "key": key, "old": old, "new": new,
                             "same_unit": "; ".join(sorted(f"{n} {u}" + (" (ключ темы)" if (n, u) in allowed else "")
                                                           for n, u in here if u in units and (n, u) not in key_pairs)),
                             "slides": " | ".join(f"[{s}] " + texts[s - 1].full.replace("\n", " ")[:500]
                                                  for s in thesis.key_slides)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=None)
    parser.add_argument("--cache", default="runs/perturbation/embeddings")
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / "runs" / "reward_checks" / time.strftime("%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    points = load_key_points(ROOT / "data" / "tasks" / "key_points" / "key_points.json")
    tasks = {t.id: t for t in load_tasks(ROOT / "data" / "tasks" / "tasks.json")}

    items, pairs, annotations = {}, [], {}
    for folder in SETS:
        part, part_pairs = pe.load_variants(ROOT / folder / "variants")
        items.update(part)
        pairs += [{**p, "set": folder} for p in part_pairs]
        annotations.update({t: {**a, "set": folder}
                            for t, a in json.loads((ROOT / folder / "annotations.json").read_text(encoding="utf-8")).items()})
    wrong = [p for p in pairs if p["kind"] == "wrong_number"]
    natural = pe.load_natural([ROOT / r for r in RUNS], tasks, set(annotations))
    needed = {k for k in items if k[1] == "original"} | {(p["task"], p["variant"]) for p in wrong}
    work = {**{k: items[k] for k in needed}, **natural}
    print(f"презентаций: исходников и неверных чисел {len(needed)}, ответов T-lite {len(natural)}", flush=True)
    pe.build_all(work)
    lemmatizer = Lemmatizer(units=True)

    # 1. числовое противоречие
    name, alpha = CONTRADICTION_CONFIG
    embedder = pe.CachedEmbedder(name, ROOT / args.cache)
    flags, counts = [], {}
    for key, item in work.items():
        result = run_score(item, points, embedder, lemmatizer, alpha)
        rows = contradiction_flags(item, points[item.task], result, lemmatizer)
        counts[key] = (sum(r["old"] for r in rows), sum(r["new"] for r in rows))
        kind = "natural" if key[0] == "natural" else "original" if key[1] == "original" else "wrong_number"
        flags += [{"kind": kind, "item": "/".join(map(str, key)), "task": item.task, **r} for r in rows]
    embedder.save()
    write_csv(out / "contradiction_flags.csv", flags)
    caught = [{"pair": p["id"], "set": p["set"],
               "old": counts[(p["task"], p["variant"])][0] > counts[(p["task"], "original")][0],
               "new": counts[(p["task"], p["variant"])][1] > counts[(p["task"], "original")][1]} for p in wrong]
    write_csv(out / "contradiction_caught.csv", caught)
    summary = {"contradiction": {
        "natural_items": sum(k[0] == "natural" for k in work),
        "natural_flagged_old": sum(k[0] == "natural" and c[0] > 0 for k, c in counts.items()),
        "natural_flagged_new": sum(k[0] == "natural" and c[1] > 0 for k, c in counts.items()),
        "originals_flagged_new": sum(k[1] == "original" and c[1] > 0 for k, c in counts.items() if k[0] != "natural"),
        "wrong_number_pairs": len(caught),
        "caught_old": sum(c["old"] for c in caught), "caught_new": sum(c["new"] for c in caught)}}

    # 2. слайд тезиса
    accuracy = []
    for name, alpha in SLIDE_CONFIGS:
        embedder = pe.CachedEmbedder(name, ROOT / args.cache)
        for task, ann in sorted(annotations.items()):
            item = items[(task, "original")]
            result = run_score(item, points, embedder, lemmatizer, alpha)
            last = len(result.theses) - 1 if item.context["conclusion"] and len(result.theses) > 1 else None
            for i, (thesis, marked) in enumerate(zip(result.theses, ann["theses"])):
                accuracy.append({"embedder": name.split("/")[-1], "alpha": alpha, "set": ann["set"], "task": task,
                                 "thesis": i, "title": points[task][i].title, "conclusion": i == last,
                                 "marked": " ".join(map(str, marked)), "best": thesis.best_slide,
                                 "key_slides": " ".join(map(str, thesis.key_slides)),
                                 "best_ok": thesis.best_slide in marked,
                                 "key_ok": any(s in marked for s in thesis.key_slides)})
        embedder.save()
    write_csv(out / "slide_accuracy.csv", accuracy)
    summary["slides"] = []
    for name, alpha in SLIDE_CONFIGS:
        for folder in SETS + [None]:
            part = [r for r in accuracy if r["embedder"] == name.split("/")[-1] and r["alpha"] == alpha
                    and not r["conclusion"] and (folder is None or r["set"] == folder)]
            if part:
                summary["slides"].append({"embedder": name.split("/")[-1], "alpha": alpha, "set": folder or "все",
                                          "theses": len(part),
                                          "best_ok": round(statistics.mean(r["best_ok"] for r in part), 3),
                                          "key_ok": round(statistics.mean(r["key_ok"] for r in part), 3)})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
