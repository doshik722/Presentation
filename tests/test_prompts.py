# Здесь проверяем системный промпт: режимы с поиском и без, строки формата и примеры разбираются парсером.
import re

from presentation_rl.dsl import parse
from presentation_rl.prompts import system_prompt
from presentation_rl.tasks import PresentationRequest

WITH_SEARCH = system_prompt(PresentationRequest("Фотоэффект"))
NO_SEARCH = system_prompt(PresentationRequest("Фотоэффект", search=False))


def test_search_only_in_search_mode():
    assert "## Поиск по Википедии" in WITH_SEARCH and "<tool_call>" in WITH_SEARCH
    assert "Сначала поиск" in WITH_SEARCH
    # без поиска о поиске и инструментах ни слова
    for word in ("Википеди", "поиск", "tool_call", "#SOURCE"):
        assert word not in NO_SEARCH


def test_format_examples_parse_without_errors():
    for prompt in (WITH_SEARCH, NO_SEARCH):
        examples = re.findall(r"`(#(?:TEXT|BULLET|FORMULA|CHART|PLOT|DIAGRAM|TIMELINE|QUOTE|IMAGE|NOTES|TITLE): [^`]+)`",
                              prompt)
        assert len(examples) >= 12
        deck = parse("#SLIDE\n" + "\n#SLIDE\n".join(e.replace("…", "далее") for e in examples))
        assert deck.errors == []


def test_example_presentations_parse_without_errors():
    for prompt, slides in ((WITH_SEARCH, 4), (NO_SEARCH, 2)):
        example = prompt[prompt.index("## Пример"):]
        lines = [line for line in example.splitlines() if line.startswith("#") and not line.startswith("##")]
        deck = parse("\n".join(lines))
        assert deck.errors == [] and deck.warnings == [] and len(deck.slides) == slides
