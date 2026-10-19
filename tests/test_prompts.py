# Здесь проверяем системный промпт: раздел о поиске только с поиском, примеры строк формата разбираются парсером.
import re

from presentation_rl.dsl import parse
from presentation_rl.prompts import system_prompt
from presentation_rl.tasks import PresentationRequest


def test_search_section_only_with_search():
    assert "## Поиск по Википедии" in system_prompt(PresentationRequest("Фотоэффект"))
    assert "Википедии" not in system_prompt(PresentationRequest("Фотоэффект", search=False))


def test_format_examples_parse_without_errors():
    examples = re.findall(r"`(#(?:TEXT|BULLET|FORMULA|CHART|PLOT|DIAGRAM|TIMELINE|QUOTE|IMAGE|NOTES|TITLE): [^`]+)`",
                          system_prompt(PresentationRequest("Фотоэффект")))
    assert len(examples) >= 12
    deck = parse("#SLIDE\n" + "\n#SLIDE\n".join(e.replace("…", "далее") for e in examples))
    assert deck.errors == []
