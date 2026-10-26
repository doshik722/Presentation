# Здесь проверяем разбор DSL на корректных, испорченных и оборванных примерах из data/examples.
import ast
from pathlib import Path

import pytest

from presentation_rl.dsl import DSLValueError, ParseLimits, compile_expression, parse

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"


def read_example(name: str) -> str:
    return (EXAMPLES / name).read_text(encoding="utf-8")


# ------------------------------------------------------------- ручные примеры


@pytest.mark.parametrize("name", ["gradient_descent.txt", "serf_reform.txt"])
def test_good_examples_parse_without_errors(name):
    deck = parse(read_example(name))
    assert deck.errors == []
    assert deck.warnings == []
    assert len(deck.slides) == 12
    assert deck.parse_rate == 1.0
    assert all(slide.title for slide in deck.slides)
    assert deck.plan  # #PLAN сохранён отдельно


def test_broken_example_matches_readme_table():
    deck = parse(read_example("broken.txt"))
    errors = {(issue.line, issue.code) for issue in deck.errors}
    assert errors == {
        (2, "untagged"),
        (3, "outside_slide"),
        (6, "duplicate_title"),
        (8, "unknown_tag"),
        (9, "missing_colon"),
        (10, "empty_value"),
        (11, "bad_value"),
        (16, "bad_value"),
        (19, "bad_value"),
    }
    warnings = {(issue.line, issue.code) for issue in deck.warnings}
    assert warnings == {(12, "empty_slide"), (13, "missing_title")}
    assert deck.fence_lines == 2
    assert deck.content_lines == 18
    assert deck.valid_lines == 9
    assert deck.parse_rate == pytest.approx(0.5)
    assert deck.slides_before_cleanup == 3
    assert len(deck.slides) == 2
    assert [e.kind for e in deck.slides[1].elements] == ["BULLET", "CHART", "DIAGRAM", "IMAGE"]


# ------------------------------------------------------------- общие правила


def test_empty_text_gives_zero_rate():
    deck = parse("")
    assert deck.slides == [] and deck.content_lines == 0 and deck.parse_rate == 0.0


def test_tags_are_case_insensitive_and_space_before_colon_is_allowed():
    deck = parse("#slide\n#Title : Заголовок\n#bullet: пункт")
    assert deck.errors == []
    assert deck.slides[0].title == "Заголовок"
    assert deck.slides[0].elements[0].data["text"] == "пункт"


def test_slide_number_is_ignored_silently():
    deck = parse("#SLIDE 3\n#TITLE: Т")
    assert deck.errors == [] and deck.warnings == []


def test_markdown_heading_is_untagged():
    deck = parse("#SLIDE\n# Заголовок\n#TITLE: Т")
    assert [(i.line, i.code) for i in deck.errors] == [(2, "untagged")]


def test_plan_is_not_counted_as_content():
    deck = parse("#PLAN: план\n#SLIDE\n#TITLE: Т")
    assert deck.content_lines == 2 and deck.plan == ["план"]


def test_title_only_slide_is_kept():
    deck = parse("#SLIDE\n#TITLE: Раздел")
    assert len(deck.slides) == 1 and deck.warnings == []


def test_text_and_bullets_are_different_elements():
    deck = parse("#SLIDE\n#TITLE: Т\n#TEXT: Связное рассуждение.\n#BULLET: пункт")
    assert [(e.kind, e.data["text"]) for e in deck.slides[0].elements] == [
        ("TEXT", "Связное рассуждение."),
        ("BULLET", "пункт"),
    ]


def test_notes_and_sources_are_collected():
    deck = parse("#SLIDE\n#TITLE: Т\n#NOTES: раз\n#NOTES: два\n#SOURCE: https://example.org")
    slide = deck.slides[0]
    assert slide.notes == ["раз", "два"] and slide.sources == ["https://example.org"]


def test_limits():
    limits = ParseLimits(max_slides=1, max_elements_per_slide=1, max_line_length=30)
    deck = parse(
        "#SLIDE\n#TITLE: Т\n#BULLET: а\n#BULLET: б\n#SLIDE\n#BULLET: " + "x" * 40,
        limits,
    )
    codes = [(i.line, i.code) for i in deck.errors]
    assert (4, "limit_exceeded") in codes
    assert (5, "limit_exceeded") in codes
    assert (6, "too_long") in codes
    assert len(deck.slides) == 1


# ------------------------------------------------------------- значения тегов


def first_element(text: str):
    deck = parse("#SLIDE\n#TITLE: Т\n" + text)
    return deck, (deck.slides[0].elements[0] if deck.slides and deck.slides[0].elements else None)


def test_quote_with_and_without_attribution():
    _, quote = first_element("#QUOTE: «Слова — с тире внутри» — Автор, источник")
    assert quote.data == {"text": "Слова — с тире внутри", "attribution": "Автор, источник"}
    deck, quote = first_element("#QUOTE: Просто слова")
    assert quote.data["attribution"] is None
    assert deck.warnings[0].code == "quote_without_attribution"


@pytest.mark.parametrize("raw", [r"$a^2$", r"$$a^2$$", r"\(a^2\)", r"a^2"])
def test_formula_delimiters_are_stripped(raw):
    _, formula = first_element("#FORMULA: " + raw)
    assert formula.data["latex"] == "a^2"


def test_chart_is_parsed():
    _, chart = first_element("#CHART: type=bar; title=Точность; x=train,test; y=0.98,0.8; unit=доля")
    assert chart.data == {
        "type": "bar",
        "categories": ["train", "test"],
        "values": [0.98, 0.8],
        "title": "Точность",
        "unit": "доля",
    }


@pytest.mark.parametrize(
    "value",
    [
        "type=bar; x=a,b; y=1",  # разная длина
        "type=bar; x=a,b; y=1,5,2",  # «1,5» с запятой
        "type=area; x=a,b; y=1,2",  # неизвестный тип
        "type=pie; x=a,b; y=-1,2",  # отрицательное значение круговой
        "type=bar; x=a; y=1",  # одна точка
        "type=bar; x=a,b; y=1,nan",  # не конечное число
        "type=bar; x=a,b",  # нет y
        "type=bar; x=a,b; y=1,2; title",  # пара без =
    ],
)
def test_bad_charts(value):
    deck, element = first_element("#CHART: " + value)
    assert element is None and deck.errors[0].code == "bad_value"


def test_unknown_chart_key_is_warning():
    deck, chart = first_element("#CHART: type=bar; x=a,b; y=1,2; color=red")
    assert chart is not None and deck.warnings[0].code == "unknown_key"


def test_plot_is_parsed():
    _, plot = first_element("#PLOT: y = x^4 - 3*x^2 + sin(x); x=-2..2; title=График")
    assert plot.data == {"expression": "x^4 - 3*x^2 + sin(x)", "x_min": -2.0, "x_max": 2.0, "title": "График"}


@pytest.mark.parametrize(
    "value",
    [
        "y = x^; x=0..1",  # незаконченное выражение
        "y = __import__('os'); x=0..1",  # чужой код
        "y = x.real; x=0..1",  # атрибуты запрещены
        "y = 5; x=0..1",  # не зависит от x
        "y = sin; x=0..1",  # функция без аргумента
        "y = x^2",  # нет диапазона
        "y = x^2; x=1..0",  # a >= b
        "f = x^2; x=0..1",  # не y
    ],
)
def test_bad_plots(value):
    deck, element = first_element("#PLOT: " + value)
    assert element is None and deck.errors[0].code == "bad_value"


@pytest.mark.parametrize(
    "short, full",
    [
        ("1.2 - 0.15x", "1.2 - 0.15*x"),
        ("1 - exp(-0.4x)", "1 - exp(-0.4*x)"),
        ("3sin(x) + 2 x", "3*sin(x) + 2*x"),
        ("2(x+1)(x-1)", "2*(x+1)*(x-1)"),
        ("x(1-x)", "x*(1-x)"),
        ("2pi x", "2*pi*x"),
        ("2e^x", "2*e^x"),
    ],
)
def test_plot_implicit_multiplication(short, full):
    assert ast.dump(compile_expression(short)) == ast.dump(compile_expression(full))


def test_plot_number_with_exponent_is_kept():
    tree = compile_expression("2e-3*x")
    assert ast.dump(tree) == ast.dump(ast.parse("2e-3*x", mode="eval"))


def test_source_with_several_ids():
    deck = parse("#SLIDE\n#TITLE: Слайд\n#SOURCE: d1, d2#3; d4\n#SOURCE: d5")
    assert deck.errors == [] and deck.slides[0].sources == ["d1", "d2#3", "d4", "d5"]


def test_flow_diagram_accepts_unicode_arrow():
    _, diagram = first_element("#DIAGRAM: type=flow; Древняя Русь →|раздробленность| Московское царство → Империя")
    assert diagram.data["nodes"] == ["Древняя Русь", "Московское царство", "Империя"]
    assert diagram.data["edges"][0]["label"] == "раздробленность"


def test_compile_expression_rejects_code():
    with pytest.raises(DSLValueError):
        compile_expression("().__class__")


def test_flow_diagram_with_labels_and_branches():
    _, diagram = first_element(
        "#DIAGRAM: type=flow; Причины -->|вызвали| Экономика; Причины -> Армия; Экономика -> Реформа; Армия -> Реформа"
    )
    assert diagram.data["nodes"] == ["Причины", "Экономика", "Армия", "Реформа"]
    assert diagram.data["edges"][0] == {"source": "Причины", "target": "Экономика", "label": "вызвали"}
    assert len(diagram.data["edges"]) == 4


def test_tree_diagram():
    _, diagram = first_element("#DIAGRAM: type=tree; Корень > А, Б; А > В")
    assert diagram.data["nodes"] == ["Корень", "А", "Б", "В"]
    assert {(e["source"], e["target"]) for e in diagram.data["edges"]} == {
        ("Корень", "А"),
        ("Корень", "Б"),
        ("А", "В"),
    }


@pytest.mark.parametrize(
    "value",
    [
        "type=flow; A",  # нет стрелки
        "type=graph; A -> B",  # неизвестный тип
        "A -> B",  # нет type
        "type=tree; A > B; C > B",  # два родителя
        "type=tree; A > B; C > D",  # два корня
        "type=flow; " + " -> ".join(f"У{i}" for i in range(9)),  # больше 8 узлов
    ],
)
def test_bad_diagrams(value):
    deck, element = first_element("#DIAGRAM: " + value)
    assert element is None and deck.errors[0].code == "bad_value"


def test_timeline():
    _, timeline = first_element("#TIMELINE: 1853–1856 — Крымская война; 1861 — Манифест")
    assert timeline.data["items"] == [
        {"date": "1853–1856", "event": "Крымская война"},
        {"date": "1861", "event": "Манифест"},
    ]


@pytest.mark.parametrize("value", ["1861 — одно событие", "1861 Манифест; 1864 — реформа"])
def test_bad_timelines(value):
    deck, element = first_element("#TIMELINE: " + value)
    assert element is None and deck.errors[0].code == "bad_value"
