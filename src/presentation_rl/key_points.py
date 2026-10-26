# Здесь собираем тезисы (key_points) из ответов составителей и проверяем их формат до ручной проверки.
"""Сборка тезисов от составителей.

Файлы в ``data/tasks/key_points``:

- ``assignment.json`` — какие темы в каком пакете у какой модели (пакеты ``train01``, ``validation01``, ``test01``…);
- ``<модель>/<пакет>.txt`` — пакет, отправленный модели;
- ``<модель>/<пакет>_answer.json`` — ответ модели на пакет, как есть;
- ``<модель>/<пакет>_fix.json`` — исправления по отдельной просьбе: заменяют темы с теми же id.

Ответ — JSON ``{"topics": [{"id", "key_points": [{"title", "statement", "key"}], "sources"}]}``; текст вокруг JSON
и обёртка в блок кода допускаются, о них пишется замечание.

Ошибки (тема не принимается): JSON не читается, нет темы пакета, лишний или повторный id, у тезиса нет title,
statement или key, больше 13 тезисов без заключения, у validation и test нет источников.
Замечания (тема принимается, но стоит посмотреть): число тезисов не по числу слайдов, последний тезис не похож
на заключение, title длиннее 8 слов, statement длиннее 25 слов, в key не 5–7 строк, строка key длиннее 4 слов
или похожа на формулу, короткое число без единиц в key, повторы key внутри темы, убраны ссылки.
Ссылки (``[название](https://…)`` и голые URL) из title, statement и key вырезаются: в train источников нет,
и составитель иногда вписывает их прямо в тезис.

Это проверка формата черновиков; содержание тезисов проверяется вручную (``docs/decisions.md``, раздел 6).
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .tasks import Task, load_tasks

MODELS = ("claude", "codex", "gemini")
MAX_THESES = 13  # без заключения
_CONCLUSION_RE = re.compile(r"^\s*(вывод|выводы|заключение|итог|итоги|conclusion|висновок|висновки|підсумок|підсумки)",
                            re.IGNORECASE)
_FORMULA_RE = re.compile(r"[\\^_={}]")
_IDENTIFIER_RE = re.compile(r"[a-z]{2,}(_[a-z0-9]{2,})+", re.IGNORECASE)  # dag_id, start_date — не формулы; x_1 — формула
_LINK_RE = re.compile(r"\s*\[[^\]]*\]\(https?://(?:[^()\s]|\([^()\s]*\))*\)|\s*https?://\S+")  # в URL бывают скобки: …_(b)


@dataclass
class KeyPoint:
    title: str
    statement: str
    key: list[str]


@dataclass
class Topic:
    id: str
    model: str
    batch: str
    key_points: list[KeyPoint]
    sources: list[dict] = field(default_factory=list)
    fixed: bool = False


@dataclass
class Report:
    topics: list[Topic] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    waiting: list[str] = field(default_factory=list)  # пакеты, ответа на которые ещё нет

    def text(self) -> str:
        lines = [f"Тем: {len(self.topics)}; ошибок: {len(self.errors)}; замечаний: {len(self.notes)}"]
        if self.waiting:
            lines.append(f"Нет ответа ({len(self.waiting)}): " + ", ".join(self.waiting))
        lines += [f"Ошибка: {e}" for e in self.errors]
        lines += [f"Замечание: {n}" for n in self.notes]
        return "\n".join(lines)


def read_answer(text: str) -> tuple[dict | None, str]:
    """JSON ответа и замечание (текст вокруг JSON); None — JSON не читается."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None, ""
    outside = (text[:start] + text[end + 1:]).replace("```json", "").replace("```", "").strip()
    try:
        # strict=False: при копировании из окна чата длинные строки иногда приходят с переносами внутри значений
        data = json.loads(text[start:end + 1], strict=False)
    except json.JSONDecodeError:
        return None, ""
    return _squeeze(data) if isinstance(data, dict) else None, ("текст вне JSON" if outside else "")


def _squeeze(value):
    """Переносы и лишние пробелы внутри строк — в один пробел («углеродный\nслед» → «углеродный след»)."""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return [_squeeze(v) for v in value]
    if isinstance(value, dict):
        return {_squeeze(k): _squeeze(v) for k, v in value.items()}  # и ключи: «"\n  statement "» → «statement»
    return value


def _unlink(value):
    """Строка без ссылок; не строки — как есть (их отбракует проверка формата)."""
    return _LINK_RE.sub("", value).strip() if isinstance(value, str) else value


def expected_theses(task: Task) -> tuple[int, int, bool]:
    """Сколько тезисов без заключения ожидается и нужно ли заключение.

    Один тезис — на 1–2 содержательных слайда; титульный слайд не считается; при числе слайдов больше трёх
    последний слайд — заключение. Не задано число слайдов — 1–13 тезисов и заключение.
    """
    slides = task.request.slide_range()
    if slides is None:
        return 1, MAX_THESES, True
    low, high = slides
    content_low = max(1, low - 1 - (low > 3))
    content_high = max(1, high - 1 - (high > 3))
    # при больших презентациях граница мягкая (+1): промпт даёт ориентир («10–12 слайдов — 6–10»), а не точное правило; это замечание, не ошибка
    return min(MAX_THESES, max(1, math.ceil(content_low / 2))), min(MAX_THESES, content_high + (high > 3)), high > 3


def check_topic(raw: dict, task: Task, where: str, report: Report) -> list[KeyPoint] | None:
    points = raw.get("key_points")
    if not isinstance(points, list) or not points:
        report.errors.append(f"{where}: нет key_points")
        return None
    result: list[KeyPoint] = []
    linked: list[int] = []
    for n, point in enumerate(points, 1):
        if not isinstance(point, dict):
            report.errors.append(f"{where}: тезис {n} — не объект")
            return None
        raw = point.get("title"), point.get("statement"), point.get("key")
        title, statement = _unlink(raw[0]), _unlink(raw[1])
        key = [_unlink(k) for k in raw[2]] if isinstance(raw[2], list) else raw[2]
        if (title, statement, key) != raw:
            linked.append(n)
        if not (isinstance(title, str) and title.strip() and isinstance(statement, str) and statement.strip()
                and isinstance(key, list) and key and all(isinstance(k, str) and k.strip() for k in key)):
            report.errors.append(f"{where}: тезис {n} — нет title, statement или key")
            return None
        result.append(KeyPoint(title.strip(), statement.strip(), [k.strip() for k in key]))
        if len(title.split()) > 8:
            report.notes.append(f"{where}: тезис {n} — title длиннее 8 слов")
        if len(statement.split()) > 25:
            report.notes.append(f"{where}: тезис {n} — statement длиннее 25 слов ({len(statement.split())})")
        if not 5 <= len(key) <= 7:
            report.notes.append(f"{where}: тезис {n} — в key {len(key)} строк, а не 5–7")
        long = [k for k in key if len(k.split()) > 4]
        if long:
            report.notes.append(f"{where}: тезис {n} — key длиннее 4 слов: {long}")
        formulas = [k for k in key if _FORMULA_RE.search(k) and not _IDENTIFIER_RE.fullmatch(k)]
        if formulas:
            report.notes.append(f"{where}: тезис {n} — key похож на формулу: {formulas}")
        bare = [k for k in key if k.isdigit() and len(k) < 3]  # «6», «10»: без единиц найдётся где угодно
        if bare:
            report.notes.append(f"{where}: тезис {n} — key — короткое число без единиц: {bare}")

    if linked:
        report.notes.append(f"{where}: убраны ссылки из тезисов {', '.join(map(str, linked))}")
    low, high, conclusion = expected_theses(task)
    has_conclusion = bool(_CONCLUSION_RE.match(result[-1].title))
    content = len(result) - (1 if conclusion else 0)
    if content > MAX_THESES:
        report.errors.append(f"{where}: {content} тезисов без заключения, больше {MAX_THESES}")
        return None
    if not low <= content <= high:
        report.notes.append(f"{where}: {content} тезисов без заключения, ожидается {low}–{high}")
    if conclusion and not has_conclusion:
        report.notes.append(f"{where}: последний тезис не похож на заключение: «{result[-1].title}»")
    keys = [k.lower() for p in result for k in p.key]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        report.notes.append(f"{where}: повторы key: {repeated}")
    return result


def collect(kp_dir: str | Path, tasks: dict[str, Task]) -> Report:
    kp_dir = Path(kp_dir)
    plan = json.loads((kp_dir / "assignment.json").read_text(encoding="utf-8"))
    report = Report()
    seen: dict[str, str] = {}
    for model in MODELS:
        for batch, ids in plan.get(model, {}).items():
            answer = kp_dir / model / f"{batch}_answer.json"
            if not answer.exists():
                report.waiting.append(f"{model}/{batch}")
                continue
            raw_topics: dict[str, dict] = {}
            fixed: set[str] = set()
            for path, is_fix in ((answer, False), (kp_dir / model / f"{batch}_fix.json", True)):
                if not path.exists():
                    continue
                data, note = read_answer(path.read_text(encoding="utf-8"))
                if data is None or not isinstance(data.get("topics"), list):
                    report.errors.append(f"{model}/{path.name}: JSON не читается или нет topics")
                    continue
                if note:
                    report.notes.append(f"{model}/{path.name}: {note}")
                for item in data["topics"]:
                    tid = str(item.get("id", "")) if isinstance(item, dict) else ""
                    if tid not in ids:
                        report.errors.append(f"{model}/{path.name}: лишний id {tid or '(пусто)'}")
                        continue
                    if tid in raw_topics and not is_fix:
                        report.errors.append(f"{model}/{path.name}: повторный id {tid}")
                        continue
                    raw_topics[tid] = item
                    if is_fix:
                        fixed.add(tid)
            for tid in ids:
                where = f"{model}/{batch}/{tid}"
                if tid not in raw_topics:
                    report.errors.append(f"{where}: темы нет в ответе")
                    continue
                if tid in seen:
                    report.errors.append(f"{where}: тема уже есть в {seen[tid]}")
                    continue
                task = tasks[tid]
                points = check_topic(raw_topics[tid], task, where, report)
                if points is None:
                    continue
                sources = raw_topics[tid].get("sources") or []
                if task.split in ("validation", "test") and not sources:
                    report.errors.append(f"{where}: нет источников (нужны для validation и test)")
                    continue
                seen[tid] = where
                report.topics.append(Topic(tid, model, batch, points, sources if isinstance(sources, list) else [],
                                           tid in fixed))
    return report


def main(argv: list[str] | None = None) -> int:
    """``python -m presentation_rl.key_points [папка проекта]``: пишет key_points.json и отчёт."""
    root = Path((argv or sys.argv[1:] or ["."])[0])
    kp_dir = root / "data" / "tasks" / "key_points"
    tasks = {t.id: t for t in load_tasks(root / "data" / "tasks" / "tasks.json")}
    report = collect(kp_dir, tasks)
    (kp_dir / "key_points.json").write_text(
        json.dumps([asdict(t) for t in report.topics], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (kp_dir / "key_points_report.txt").write_text(report.text() + "\n", encoding="utf-8")
    print(report.text())
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
