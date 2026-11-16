# Здесь проверяем системный промпт: режимы с поиском и без, строки формата и примеры разбираются парсером.
import re

from presentation_rl.dsl import parse
from presentation_rl.prompts import SPECIFIC_RULE, system_prompt
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


def test_no_images_mode_does_not_mention_images():
    for search in (True, False):
        with_images = system_prompt(PresentationRequest("Фотоэффект", search=search))
        without = system_prompt(PresentationRequest("Фотоэффект", search=search, images=False))
        assert "#IMAGE" in with_images and "Картинки —" in with_images
        assert "#IMAGE" not in without and "картин" not in without.lower()
        example = without[without.index("## Пример"):]
        lines = [line for line in example.splitlines() if line.startswith("#") and not line.startswith("##")]
        assert parse("\n".join(lines)).errors == []


def test_specific_rule_is_optional_and_follows_text_volume_rule():
    for search in (True, False):
        for images in (True, False):
            request = PresentationRequest("Фотоэффект", search=search, images=images)
            plain, specific = system_prompt(request), system_prompt(request, specific=True)
            assert SPECIFIC_RULE not in plain and specific.count(SPECIFIC_RULE) == 1
            assert specific.replace(SPECIFIC_RULE, "") == plain  # больше ничего не меняется
            assert specific.index("однородные пункты") < specific.index(SPECIFIC_RULE) < specific.index("визуальный элемент")
