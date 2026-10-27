# Здесь собираем презентации для калибровки оценки text_overflow: их потом меряет PowerPoint (measure_pptx.ps1).
"""Собирает PPTX из готовых ответов DSL и сохраняет отчёт сборщика о переполнении текста.

Источники: примеры SFT (``data/tasks/sft/sft_dataset.jsonl``) и, если указаны, папки с ответами модели
(``*_answer.txt`` из прогонов вроде EXP-010). Результат — ``<out>/decks/*.pptx`` и ``<out>/builder.csv``:
по каждому слайду оценка сборщика (заполненность колонки, переполнение и его размер).

    PYTHONPATH=src python scripts/text_fit/build_decks.py runs/text_fit/<дата> [папка с *_answer.txt ...]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from presentation_rl.rendering import build_from_text

ROOT = Path(__file__).resolve().parents[2]
NEEDED_RE = re.compile(r"нужно около ([\d.]+) дюйма, доступно ([\d.]+)")


def sources(extra: list[Path]):
    for line in (ROOT / "data/tasks/sft/sft_dataset.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        yield f"sft_{row['id']}_{row['model']}", row["answer"]
    for folder in extra:
        for path in sorted(folder.rglob("*_answer.txt")):
            yield f"{folder.name}_{path.stem.removesuffix('_answer')}", path.read_text(encoding="utf-8")


def main() -> None:
    out = Path(sys.argv[1])
    decks = out / "decks"
    decks.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, text in sources([Path(p) for p in sys.argv[2:]]):
        _, report = build_from_text(text, decks / f"{name}.pptx")
        overflow = {}
        for item in report.items:
            if item.status == "text_overflow" and item.kind == "TEXT":
                match = NEEDED_RE.search(item.message)
                overflow[item.slide] = match.groups() if match else ("", "")
        for stats in report.slides:
            needed, available = overflow.get(stats.slide, ("", ""))
            rows.append({"file": f"{name}.pptx", "slide": stats.slide, "text_fill": stats.text_fill,
                         "visuals": stats.visuals, "text_overflow": int(stats.slide in overflow),
                         "needed_in": needed, "available_in": available})
    with open(out / "builder.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(r["text_overflow"] for r in rows)
    print(f"{len({r['file'] for r in rows})} презентаций, {len(rows)} слайдов, text_overflow — {flagged}")


if __name__ == "__main__":
    main()
