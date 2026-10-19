# Здесь описываем запрос пользователя на презентацию (PresentationRequest), задание набора (Task) и загрузку data/tasks/tasks.json.
"""Запрос на презентацию и задания набора.

``PresentationRequest`` — то, что заполняет пользователь демо и что получает
модель: текст запроса и необязательные поля (аудитория, число слайдов, тип
темы, разрешён ли поиск по Википедии, нужны ли картинки). Поиск и картинки
независимы: поиск — инструменты модели во время генерации, картинки по
описаниям из ``#IMAGE`` после генерации ищет сборщик в Wikimedia Commons. Палитра оформления (``theme``) модели не передаётся:
её применяет сборщик, ``pick_theme(request.theme, request.text)``.
Задание набора (``Task``) — это такой запрос плюс сведения для деления на выборки
и для отчёта.

Число слайдов: если оно задано полем или упомянуто в тексте («на 7 слайдов»,
«5–7 слайдов»), требование берётся оттуда; поле важнее текста. Больше 30 слайдов
не бывает: так ограничен парсер DSL, и длиннее ответ не помещается в генерацию.
Если число не задано нигде, требования нет: модель решает сама, сколько слайдов
нужно для раскрытия темы.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from .rendering.themes import THEME_NAMES

Slides = Union[int, tuple[int, int]]

MAX_SLIDES = 30  # совпадает с ParseLimits.max_slides
DOMAINS = ("technical", "humanities", "auto")
SPLITS = ("sft", "train", "validation", "test")
SOURCES = ("syllabus", "wildchat", "composed", "author")

_DOMAIN_NAMES = {"technical": "техническая", "humanities": "гуманитарная"}

# Слайды в запросах иногда называют страницами: «15 страниц», «7-страничная».
_SLIDE = r"(?:слайд|страни[цч])"
# «5-7 слайдов», «5–7 слайдов», «от 5 до 7 слайдов»
_RANGE_RE = re.compile(
    r"(?:от\s*)?(\d{1,3})\s*(?:-|–|—|до)\s*(\d{1,3})\s*(?:-?(?:ти|и|х)\s*)?" + _SLIDE, re.IGNORECASE
)
# «10 слайдов», «на 5-ти слайдах», «из 12 слайдов», «не менее 8 слайдов», «7-страничная»
_NUMBER_RE = re.compile(r"(\d{1,3})\s*(?:-?(?:ти|и|х|ми|ть)\s*)?-?" + _SLIDE, re.IGNORECASE)
# «количество слайдов: 8», «число слайдов = 8», «всего слайдов будет 16»
_COUNT_AFTER_RE = re.compile(
    r"(?:количеств\w*|числ\w*|всего)\s+слайд\w*\s*(?:будет\s*)?[:=—–-]?\s*(\d{1,3})", re.IGNORECASE
)
_WORD_NUMBERS = {
    "один": 1, "одного": 1, "одном": 1, "два": 2, "двух": 2, "три": 3, "трёх": 3, "трех": 3,
    "четыре": 4, "четырёх": 4, "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "одиннадцать": 11, "одиннадцати": 11, "двенадцать": 12, "двенадцати": 12,
    "пятнадцать": 15, "пятнадцати": 15, "двадцать": 20, "двадцати": 20,
}
# «пять слайдов», «к девяти слайдам»; не «Один слайд — один гриб» и не «Два слайда из них»
_WORD_RE = re.compile(
    r"\b(" + "|".join(_WORD_NUMBERS) + r")\s+слайд\w*(?!\s*(?:[—–-]\s*од|из\s+них))", re.IGNORECASE
)
# «текст для слайда», «речь к слайду» — один слайд; «для слайда 2» — ссылка на номер, не число
_SINGLE_RE = re.compile(r"\b(?:для|к)\s+слайд[ау]\b(?!\s*№?\s*\d)", re.IGNORECASE)


def parse_slides(text: str) -> Slides | None:
    """Число слайдов, упомянутое в тексте запроса, или None.

    Номера слайдов вроде «Слайд 4: …» не считаются: число должно стоять перед
    словом «слайд» или после «количество слайдов». Просьба больше 30 слайдов
    урезается до 30.
    """
    found: Slides | None = None
    if match := _RANGE_RE.search(text):
        low, high = int(match.group(1)), int(match.group(2))
        found = (low, high) if low < high else low
    elif match := _NUMBER_RE.search(text):
        found = int(match.group(1))
    elif match := _COUNT_AFTER_RE.search(text):
        found = int(match.group(1))
    elif match := _WORD_RE.search(text):
        found = _WORD_NUMBERS[match.group(1).lower()]
    elif _SINGLE_RE.search(text):
        found = 1
    if found is None:
        return None
    if isinstance(found, tuple):
        low, high = min(found[0], MAX_SLIDES), min(found[1], MAX_SLIDES)
        found = (low, high) if low < high else low
    else:
        found = min(found, MAX_SLIDES)
    return found if _valid_slides(found) else None


def _valid_slides(value: Slides) -> bool:
    if isinstance(value, int):
        return 1 <= value <= MAX_SLIDES
    low, high = value
    return 1 <= low < high <= MAX_SLIDES


@dataclass(frozen=True)
class PresentationRequest:
    """Запрос пользователя: текст и необязательные поля формы."""

    text: str
    audience: str | None = None
    slides: Slides | None = None
    domain: str = "auto"
    search: bool = True  # False — у модели нет инструментов поиска
    images: bool = True  # False — #IMAGE не используется, нужен другой визуал
    sources: bool = True  # False — не показывать источники на слайдах; модели не передаётся, их убирает сборщик
    theme: str = "auto"  # палитра оформления; auto — сборщик выбирает по тексту запроса

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("пустой текст запроса")
        if self.domain not in DOMAINS:
            raise ValueError(f"неизвестный тип темы: {self.domain}")
        if self.theme != "auto" and self.theme not in THEME_NAMES:
            raise ValueError(f"неизвестная палитра: {self.theme}")
        if self.slides is not None and not _valid_slides(self.slides):
            raise ValueError(f"недопустимое число слайдов: {self.slides}")

    def slide_range(self) -> tuple[int, int] | None:
        """Требование к числу слайдов: поле, иначе число из текста; None — не задано."""
        value = self.slides if self.slides is not None else parse_slides(self.text)
        if value is None:
            return None
        return (value, value) if isinstance(value, int) else value

    def render(self) -> str:
        """Текст запроса для промпта модели: сам запрос и заполненные поля."""
        lines = [self.text.strip(), ""]
        if self.audience:
            lines.append(f"Аудитория: {self.audience}")
        if slides := self.slide_range():
            low, high = slides
            lines.append(f"Слайдов: {low}" if low == high else f"Слайдов: {low}–{high}")
        if self.domain != "auto":
            lines.append(f"Тип темы: {_DOMAIN_NAMES[self.domain]}")
        lines.append("Поиск: есть" if self.search else "Поиск: нет")
        lines.append("Картинки: есть" if self.images else "Картинки: нет")
        return "\n".join(lines)


@dataclass(frozen=True)
class Task:
    """Задание набора: запрос и сведения для деления на выборки и отчёта."""

    id: str
    split: str
    source: str
    request: PresentationRequest
    topic: str | None
    domain: str
    scenario: str
    family: str
    discipline: str | None = None
    has_course: bool = False
    has_material: bool = False
    wildchat_id: str | None = None
    extra: dict = field(default_factory=dict, compare=False, repr=False)


def task_from_record(record: dict, search: bool = True, images: bool = True, sources: bool = True) -> Task:
    """Задание из записи tasks.json.

    Темы из рабочих программ становятся запросом «Сделай презентацию на тему…»
    с аудиторией и типом темы; остальные запросы подаются текстом пользователя,
    тип темы модель определяет сама.
    """
    slides = record.get("slides")
    if isinstance(slides, list):
        slides = tuple(slides)
    if record["source"] == "syllabus":
        text = f"Сделай презентацию на тему «{record['topic']}»."
        domain = record["domain"]
    else:
        text = record["request"]
        domain = "auto"
    request = PresentationRequest(
        text=text,
        audience=record.get("audience"),
        slides=slides,
        domain=domain,
        search=search,
        images=images,
        sources=sources,
    )
    known = {
        "id", "split", "source", "topic", "domain", "scenario", "family", "discipline",
        "has_course", "has_material", "wildchat_id", "request", "audience", "slides",
    }
    return Task(
        id=record["id"],
        split=record["split"],
        source=record["source"],
        request=request,
        topic=record.get("topic"),
        domain=record["domain"],
        scenario=record["scenario"],
        family=record["family"],
        discipline=record.get("discipline"),
        has_course=bool(record.get("has_course")),
        has_material=bool(record.get("has_material")),
        wildchat_id=record.get("wildchat_id"),
        extra={k: v for k, v in record.items() if k not in known},
    )


def check_records(records: list[dict]) -> list[str]:
    """Проблемы в записях tasks.json; пустой список — всё в порядке."""
    problems: list[str] = []
    seen: set[str] = set()
    family_split: dict[str, str] = {}
    for i, record in enumerate(records):
        where = record.get("id", f"запись {i}")
        for key in ("id", "split", "source", "domain", "scenario", "family"):
            if not record.get(key):
                problems.append(f"{where}: нет поля {key}")
        if record.get("id") in seen:
            problems.append(f"{where}: повторяется id")
        seen.add(record.get("id"))
        if record.get("split") not in SPLITS:
            problems.append(f"{where}: неизвестная выборка {record.get('split')}")
        if record.get("source") not in SOURCES:
            problems.append(f"{where}: неизвестный источник {record.get('source')}")
        if record.get("domain") not in ("technical", "humanities"):
            problems.append(f"{where}: неизвестный тип темы {record.get('domain')}")
        if record.get("source") == "syllabus" and not record.get("topic"):
            problems.append(f"{where}: у темы из программы нет topic")
        if record.get("source") != "syllabus" and not (record.get("request") or "").strip():
            problems.append(f"{where}: у запроса нет текста")
        family = record.get("family")
        if family in family_split and family_split[family] != record.get("split"):
            problems.append(f"{where}: семейство {family} попало в разные выборки")
        family_split.setdefault(family, record.get("split"))
        try:
            task_from_record(record)
        except (ValueError, KeyError) as exc:
            problems.append(f"{where}: {exc}")
    return problems


def load_tasks(
    path: str | Path, split: str | None = None, search: bool = True, images: bool = True, sources: bool = True,
) -> list[Task]:
    """Читает tasks.json, проверяет записи и возвращает задания (все или одной выборки)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data["tasks"] if isinstance(data, dict) else data
    problems = check_records(records)
    if problems:
        raise ValueError("ошибки в наборе заданий:\n" + "\n".join(problems[:20]))
    tasks = [task_from_record(r, search=search, images=images, sources=sources) for r in records]
    return [t for t in tasks if split is None or t.split == split]
