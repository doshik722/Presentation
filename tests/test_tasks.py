# Здесь проверяем запрос на презентацию: число слайдов из поля и текста, сборку текста для модели и проверку tasks.json.
import json
from pathlib import Path

import pytest

from presentation_rl.tasks import (
    PresentationRequest,
    check_records,
    load_tasks,
    parse_slides,
    task_from_record,
)

TASKS = Path(__file__).resolve().parents[1] / "data" / "tasks" / "tasks.json"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Сделай презентацию на 10 слайдов", 10),
        ("презентация на 5-ти слайдах про кошек", 5),
        ("план на 14 слайдов", 14),
        ("нужно 5-7 слайдов", (5, 7)),
        ("от 8 до 10 слайдов", (8, 10)),
        ("Количество слайдов: 6", 6),
        ("текст для одного слайда", 1),
        ("пять слайдов, не больше", 5),
        ("Количество слайдов = 15. Например, для слайда 2 распиши…", 15),
        ("Два слайда из них — приветствие. Всего слайдов будет 16", 16),
        ("Напиши содержимое одиннадцати слайдов", 11),
        ("Создать 7-страничную презентацию", 7),
        ("презентация должна составлять 17 страниц", 17),
        ("Составь текст для слайда презентации", 1),
        ("Один слайд — один гриб", None),
        ("например, для слайда 2 распиши", None),
        ("Сделай презентацию про инфляцию", None),
        ("Слайд 4: Преимущества решения", None),  # номер слайда, а не их число
        ("Сделай презентацию на 40 слайдов", 30),  # больше 30 урезаем
        ("от 25 до 40 слайдов", (25, 30)),
        ("на 0 слайдов", None),
    ],
)
def test_parse_slides(text, expected):
    assert parse_slides(text) == expected


def test_slide_range_field_beats_text():
    req = PresentationRequest("Сделай презентацию на 25 слайдов", slides=8)
    assert req.slide_range() == (8, 8)


def test_slide_range_from_text_and_default():
    assert PresentationRequest("Сделай презентацию на 7 слайдов").slide_range() == (7, 7)
    assert PresentationRequest("Сделай 5–7 слайдов").slide_range() == (5, 7)
    assert PresentationRequest("Рацион на день").slide_range() is None


def test_render_contains_only_filled_fields():
    text = PresentationRequest("Рацион на день").render()
    assert "Аудитория" not in text and "Тип темы" not in text and "Слайдов" not in text
    assert "Поиск: есть" in text and "Картинки: есть" in text

    text = PresentationRequest(
        "Сделай презентацию на тему «Инфляция».", audience="финансовый отдел",
        slides=8, domain="technical", search=False, images=False,
    ).render()
    assert "Аудитория: финансовый отдел" in text
    assert "Слайдов: 8" in text
    assert "Тип темы: техническая" in text
    assert "Поиск: нет" in text and "Картинки: нет" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(text="  "), dict(text="x", domain="physics"), dict(text="x", slides=0),
        dict(text="x", slides=(7, 5)), dict(text="x", theme="rainbow"),
    ],
)
def test_request_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        PresentationRequest(**kwargs)


def _record(**overrides):
    record = {
        "id": "t001", "split": "train", "source": "wildchat", "domain": "technical",
        "scenario": "учёба", "family": "f1", "topic": "Инфляция",
        "request": "Сделай презентацию про инфляцию на 6 слайдов", "audience": None, "slides": None,
    }
    record.update(overrides)
    return record


def test_task_from_record_syllabus_and_wildchat():
    syllabus = task_from_record(_record(source="syllabus", request=None, audience="студенты 1–2 курса"))
    assert syllabus.request.text == "Сделай презентацию на тему «Инфляция»."
    assert syllabus.request.domain == "technical"
    wildchat = task_from_record(_record())
    assert wildchat.request.domain == "auto"
    assert wildchat.request.slide_range() == (6, 6)


def test_check_records_finds_problems():
    records = [
        _record(),
        _record(id="t001"),  # повтор id
        _record(id="t003", split="test"),  # семейство f1 уже в train
        _record(id="t004", family="f4", request=" "),  # пустой запрос
        _record(id="t005", family="f5", domain="physics"),
    ]
    problems = "\n".join(check_records(records))
    assert "повторяется id" in problems
    assert "разные выборки" in problems
    assert "нет текста" in problems
    assert "неизвестный тип темы" in problems


@pytest.mark.skipif(not TASKS.exists(), reason="нет data/tasks/tasks.json")
def test_real_task_file_loads():
    tasks = load_tasks(TASKS)
    assert len({t.id for t in tasks}) == len(tasks)
    sizes = {s: len(load_tasks(TASKS, split=s)) for s in ("sft", "train", "validation", "test")}
    assert sum(sizes.values()) == len(tasks)
    for task in tasks:
        if slides := task.request.slide_range():
            low, high = slides
            assert 1 <= low <= high <= 30
        assert task.request.render()


def test_theme_is_not_shown_to_model():
    text = PresentationRequest("Рацион на день", theme="forest").render()
    assert "forest" not in text and "палитр" not in text.lower()


def test_search_and_images_are_independent():
    text = PresentationRequest("Фотоэффект", search=False, images=True).render()
    assert "Поиск: нет" in text and "Картинки: есть" in text



def test_sources_preference_is_not_shown_to_model():
    assert "Источник" not in PresentationRequest("Фотоэффект", sources=False).render()
    assert not PresentationRequest("Фотоэффект", sources=False).sources
