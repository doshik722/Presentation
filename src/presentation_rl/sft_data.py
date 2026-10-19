# Здесь собираем примеры для SFT из сохранённых ответов составителей и проверяем, что ничего не потеряно и не перепутано.
"""Сборка примеров для SFT.

Файлы в ``data/tasks/sft``:

- ``assignment.json`` — какие задания в каком пакете у какой модели;
- ``offline.json`` — задания без картинок;
- ``<модель>/answer<k>.txt`` — ответ модели на пакет k, как есть;
- ``<модель>/answer<k>_fix.txt`` — исправления по отдельной просьбе: заменяют презентации с теми же id;
- остальные файлы (например, ``answer3_v4.txt``) — история, сборка их не читает.

В ответе каждая презентация начинается строкой с id (``### t001`` или просто ``t001``).
Если презентация в блоке кода, берётся только блок; текст вне блока (например, с какими
источниками составитель сверял факты) сохраняется отдельно как ``comment`` и в обучение не идёт. Пример — запрос в том виде, в каком его увидит модель
(«Поиск: нет», картинки — по ``offline.json``), и ответ в DSL.

Ошибки (пример в набор не идёт или набор неполон): нет задания пакета, лишний или чужой id,
ошибки разбора, формула не рисуется, ``#IMAGE`` при «Картинки: нет». Замечания (пример идёт): нет ``#PLAN``,
число слайдов не совпадает с требованием запроса.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .dsl import parse
from .rendering.visuals import RenderError, render_formula
from .tasks import Task, load_tasks

MODELS = ("claude", "codex", "gemini")
_ID_RE = re.compile(r"^\s*(?:#{1,3}\s*)?(t\d{3})\s*$")
_FENCE_RE = re.compile(r"^\s*```")


@dataclass
class Example:
    id: str
    model: str
    batch: str
    images: bool
    request: str
    answer: str
    fixed: bool = False
    comment: str = ""  # текст составителя вне блока кода: для сверки фактов, не для обучения


@dataclass
class Report:
    examples: list[Example] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    waiting: list[str] = field(default_factory=list)  # пакеты, ответа на которые ещё нет

    def text(self) -> str:
        lines = [f"Примеров: {len(self.examples)}; ошибок: {len(self.errors)}; замечаний: {len(self.notes)}"]
        if self.waiting:
            lines.append("Ещё нет ответа: " + ", ".join(self.waiting))
        lines += ["Ошибка: " + e for e in self.errors]
        lines += ["Замечание: " + n for n in self.notes]
        return "\n".join(lines)


def split_answer(text: str, with_comments: bool = False):
    """Делит ответ составителя на презентации по строкам с id.

    Если у презентации есть блок кода, берётся только его содержимое, остальное — комментарий.
    Возвращает {id: dsl} или, при ``with_comments``, {id: (dsl, комментарий)}.
    """
    raw: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if match := _ID_RE.match(line):
            current = match.group(1)
            raw[current] = []
        elif current is not None:
            raw[current].append(line)
    result = {}
    for key, lines in raw.items():
        inside, outside, in_block, had_block = [], [], False, False
        for line in lines:
            if _FENCE_RE.match(line):
                in_block, had_block = not in_block, True
            elif in_block:
                inside.append(line)
            else:
                outside.append(line)
        dsl = inside if had_block else outside
        comment = outside if had_block else []
        pair = ("\n".join(dsl).strip(), "\n".join(comment).strip())
        result[key] = pair if with_comments else pair[0]
    return result


def _request(task: Task, images: bool) -> str:
    return replace(task.request, search=False, images=images).render()


def _broken_formulas(deck) -> list[int]:
    broken = []
    with tempfile.TemporaryDirectory() as tmp:
        for slide in deck.slides:
            for element in slide.elements_of("FORMULA"):
                try:
                    render_formula(element.data["latex"], Path(tmp) / "f.png")
                except RenderError:
                    broken.append(element.line)
    return broken


def collect(sft_dir: str | Path, tasks_path: str | Path) -> Report:
    """Собирает все примеры, на которые уже есть ответы, и проверяет их."""
    sft_dir = Path(sft_dir)
    tasks = {t.id: t for t in load_tasks(tasks_path, split="sft")}
    plan = json.loads((sft_dir / "assignment.json").read_text(encoding="utf-8"))
    no_images = set(json.loads((sft_dir / "offline.json").read_text(encoding="utf-8")))
    report = Report()
    seen: dict[str, str] = {}

    for model in MODELS:
        for batch, ids in plan[model].items():
            if not ids:
                continue
            k = batch.removeprefix("batch")
            main = sft_dir / model / f"answer{k}.txt"
            if not main.exists() or main.stat().st_size == 0:
                report.waiting.append(f"{model}/{batch}")
                continue
            parts = split_answer(main.read_text(encoding="utf-8"), with_comments=True)
            fix = sft_dir / model / f"answer{k}_fix.txt"
            fixed = split_answer(fix.read_text(encoding="utf-8"), with_comments=True) if fix.exists() else {}
            for extra in sorted(set(fixed) - set(parts)):
                report.errors.append(f"{model}/{batch}: исправление {extra} — такого задания в ответе нет")
            parts.update(fixed)

            for extra in sorted(set(parts) - set(ids)):
                report.errors.append(f"{model}/{batch}: лишний id {extra}")
            for task_id in ids:
                where = f"{model}/{batch}/{task_id}"
                if task_id not in parts:
                    report.errors.append(f"{where}: нет презентации")
                    continue
                if task_id in seen:
                    report.errors.append(f"{where}: задание уже взято из {seen[task_id]}")
                    continue
                seen[task_id] = where
                answer, comment = parts[task_id]
                deck = parse(answer)
                images = task_id not in no_images
                bad = [f"{e.code} (строка {e.line})" for e in deck.errors]
                if not images and any(e.kind == "IMAGE" for s in deck.slides for e in s.elements):
                    bad.append("#IMAGE при «Картинки: нет»")
                bad += [f"формула не рисуется (строка {line})" for line in _broken_formulas(deck)]
                if bad:
                    report.errors.append(f"{where}: " + "; ".join(bad))
                    continue
                if not deck.plan:
                    report.notes.append(f"{where}: нет #PLAN")
                need = tasks[task_id].request.slide_range()
                if need and not need[0] <= len(deck.slides) <= need[1]:
                    report.notes.append(f"{where}: слайдов {len(deck.slides)}, в запросе {need[0]}–{need[1]}")
                report.examples.append(
                    Example(
                        task_id, model, batch, images, _request(tasks[task_id], images), answer,
                        task_id in fixed, comment,
                    )
                )
    report.examples.sort(key=lambda e: e.id)
    return report


def main(argv: list[str] | None = None) -> int:
    """``python -m presentation_rl.sft_data [папка проекта]``: пишет sft_dataset.jsonl и отчёт."""
    root = Path((argv or sys.argv[1:] or ["."])[0])
    sft_dir = root / "data" / "tasks" / "sft"
    report = collect(sft_dir, root / "data" / "tasks" / "tasks.json")
    with open(sft_dir / "sft_dataset.jsonl", "w", encoding="utf-8") as out:
        for example in report.examples:
            out.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    (sft_dir / "sft_report.txt").write_text(report.text() + "\n", encoding="utf-8")
    print(report.text())
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
