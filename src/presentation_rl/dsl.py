# Здесь описываем структуру презентации (Deck, Slide, Element) и разбираем текст DSL по спецификации docs/dsl_spec.md.
"""Парсер DSL презентаций.

Каждая строка разбирается независимо: ошибочная строка теряется сама,
остальные попадают в презентацию. Парсер ничего не исправляет и не
переносит между слайдами; всё, что он отбросил, записано в ``Deck.errors``.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any

# Теги со значением. #SLIDE значения не имеет и обрабатывается отдельно.
VALUE_TAGS = frozenset(
    {
        "TITLE",
        "TEXT",
        "BULLET",
        "NOTES",
        "PLAN",
        "SOURCE",
        "QUOTE",
        "FORMULA",
        "CHART",
        "PLOT",
        "DIAGRAM",
        "TIMELINE",
        "IMAGE",
    }
)
SLIDE_TAG = "SLIDE"

# Элементы, которые рисуются на слайде в порядке появления.
ELEMENT_TAGS = frozenset({"TEXT", "BULLET", "QUOTE", "FORMULA", "CHART", "PLOT", "DIAGRAM", "TIMELINE", "IMAGE"})
# Крупные визуальные элементы: сборщик рисует не больше двух на слайд.
LARGE_VISUALS = frozenset({"CHART", "PLOT", "DIAGRAM", "TIMELINE", "IMAGE"})

CHART_TYPES = ("bar", "line", "pie")
DIAGRAM_TYPES = ("flow", "tree")
PLOT_FUNCTIONS = ("sin", "cos", "tan", "exp", "log", "sqrt", "abs")
PLOT_CONSTANTS = ("pi", "e")
# Неявное умножение в #PLOT: «0.15x», «3sin(x)», «2(x+1)», «(x+1)(x-1)», «x(1-x)» читаются со знаком *.
# Запись числа вроде «2e-3» не трогаем: это число, а не 2·e − 3.
_IMPLICIT_MUL_RE = re.compile(
    r"(?<=[\d)])\s*(?=[a-zA-Z(])(?!e[+-]?\d)"  # после числа или «)»: 0.15x, 3sin(x), 2(x+1), (x+1)(x-1)
    r"|(?<=(?<![a-zA-Z])x)\s*(?=[a-zA-Z(])"  # после x: x(1-x), x sin(x)
    r"|(?<=(?<![a-zA-Z])pi)\s*(?=[a-zA-Z(])"  # после pi: 2pi x
)

_TAG_RE = re.compile(r"^#([A-Za-z_]+)(.*)$")
_FENCE_RE = re.compile(r"^```")
# Стрелка потока: -> или -->, с необязательной подписью в стиле Mermaid: -->|текст|
_ARROW_RE = re.compile(r"\s*(?:-{1,2}>|→)\s*(?:\|([^|]*)\|)?\s*")  # стрелка: ->, --> или →
_QUOTE_SPLIT_RE = re.compile(r"\s+(?:—|--|–)\s+")
_TIMELINE_SPLIT_RE = re.compile(r"\s+(?:—|–|-)\s+")
_NODE_FORBIDDEN = ("->", ">", "|", ";", ",")


class DSLValueError(ValueError):
    """Значение тега не соответствует синтаксису из спецификации."""


@dataclass(frozen=True)
class ParseLimits:
    """Лимиты разбора; в экспериментах задаются конфигурацией."""

    max_slides: int = 30
    max_elements_per_slide: int = 20
    max_line_length: int = 1000
    max_chart_points: int = 12
    min_chart_points: int = 2
    max_diagram_nodes: int = 8
    min_timeline_items: int = 2
    max_timeline_items: int = 8


@dataclass
class Issue:
    """Ошибка разбора или предупреждение о дефекте презентации."""

    code: str
    message: str
    line: int | None = None
    text: str = ""


@dataclass
class Element:
    """Элемент слайда. ``data`` зависит от ``kind`` (см. функции ``_parse_*``)."""

    kind: str
    line: int
    raw: str
    data: dict[str, Any]


@dataclass
class Slide:
    line: int
    title: str | None = None
    title_line: int | None = None
    elements: list[Element] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.title is None and not self.elements and not self.notes and not self.sources

    def elements_of(self, *kinds: str) -> list[Element]:
        return [e for e in self.elements if e.kind in kinds]


@dataclass
class Deck:
    """Разобранная презентация и диагностика разбора."""

    slides: list[Slide] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    content_lines: int = 0
    valid_lines: int = 0
    fence_lines: int = 0
    slides_before_cleanup: int = 0

    @property
    def parse_rate(self) -> float:
        """Доля корректных содержательных строк; 0, если содержательных строк нет."""
        return self.valid_lines / self.content_lines if self.content_lines else 0.0

    def error_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.errors:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return counts

    def warning_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.warnings:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return counts


# --------------------------------------------------------------------------- разбор


def parse(text: str, limits: ParseLimits | None = None) -> Deck:
    """Разбирает текст DSL в презентацию. Никогда не бросает исключений из-за содержимого."""
    limits = limits or ParseLimits()
    deck = Deck()
    current: Slide | None = None
    all_slides: list[Slide] = []

    for number, raw_line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if _FENCE_RE.match(line):
            deck.fence_lines += 1
            continue

        match = _TAG_RE.match(line)
        tag = match.group(1).upper() if match else None

        # #PLAN не входит в содержательные строки ни при каком раскладе.
        if tag == "PLAN":
            value = _split_value(match.group(2)) if match else None
            if value:
                deck.plan.append(value)
            continue

        deck.content_lines += 1

        def error(code: str, message: str) -> None:
            deck.errors.append(Issue(code=code, message=message, line=number, text=line))

        if len(line) > limits.max_line_length:
            error("too_long", f"строка длиннее {limits.max_line_length} символов")
            continue
        if match is None:
            error("untagged", "строка не начинается с тега вида #ТЕГ:")
            continue

        if tag == SLIDE_TAG:
            if len(all_slides) >= limits.max_slides:
                error("limit_exceeded", f"больше {limits.max_slides} слайдов")
                current = None
                continue
            current = Slide(line=number)
            all_slides.append(current)
            deck.valid_lines += 1
            continue

        if tag not in VALUE_TAGS:
            error("unknown_tag", f"неизвестный тег #{tag}")
            continue

        value = _split_value(match.group(2))
        if value is None:
            error("missing_colon", f"после #{tag} нет двоеточия")
            continue
        if value == "":
            error("empty_value", f"у #{tag} пустое значение")
            continue
        if current is None:
            code = "limit_exceeded" if len(all_slides) >= limits.max_slides else "outside_slide"
            error(code, "элемент стоит до первого #SLIDE" if code == "outside_slide" else "слайд сверх лимита")
            continue

        if tag == "TITLE":
            if current.title is not None:
                error("duplicate_title", f"у слайда уже есть заголовок (строка {current.title_line})")
                continue
            current.title = value
            current.title_line = number
        elif tag == "NOTES":
            current.notes.append(value)
        elif tag == "SOURCE":  # несколько id через запятую или «;» — несколько источников
            current.sources.extend(ref for ref in (part.strip() for part in re.split(r"[,;]", value)) if ref)
        else:
            if len(current.elements) >= limits.max_elements_per_slide:
                error("limit_exceeded", f"больше {limits.max_elements_per_slide} элементов на слайде")
                continue
            try:
                data, element_warnings = _parse_value(tag, value, limits)
            except DSLValueError as exc:
                error("bad_value", str(exc))
                continue
            for code, message in element_warnings:
                deck.warnings.append(Issue(code=code, message=message, line=number, text=line))
            current.elements.append(Element(kind=tag, line=number, raw=line, data=data))
        deck.valid_lines += 1

    deck.slides_before_cleanup = len(all_slides)
    for slide in all_slides:
        if slide.is_empty():
            deck.warnings.append(Issue("empty_slide", "пустой слайд удалён", line=slide.line))
            continue
        if slide.title is None:
            deck.warnings.append(Issue("missing_title", "у слайда нет заголовка", line=slide.line))
        deck.slides.append(slide)
    return deck


def _split_value(rest: str) -> str | None:
    """Возвращает значение после двоеточия; None, если двоеточия нет."""
    rest = rest.lstrip()
    if not rest.startswith(":"):
        return None
    return rest[1:].strip()


def _parse_value(tag: str, value: str, limits: ParseLimits) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    if tag in ("TEXT", "BULLET", "IMAGE"):
        return {"text": value}, []
    if tag == "QUOTE":
        return _parse_quote(value)
    if tag == "FORMULA":
        return {"latex": _strip_math_delimiters(value)}, []
    if tag == "CHART":
        return _parse_chart(value, limits)
    if tag == "PLOT":
        return _parse_plot(value)
    if tag == "DIAGRAM":
        return _parse_diagram(value, limits)
    if tag == "TIMELINE":
        return _parse_timeline(value, limits), []
    raise AssertionError(f"нет разбора для #{tag}")


# ------------------------------------------------------------------- значения тегов


def _parse_quote(value: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    parts = _QUOTE_SPLIT_RE.split(value)
    if len(parts) >= 2 and parts[-1].strip():
        quote = " — ".join(parts[:-1]).strip()
        attribution = parts[-1].strip()
        if quote:
            return {"text": _strip_quotes(quote), "attribution": attribution}, []
    return {"text": _strip_quotes(value), "attribution": None}, [
        ("quote_without_attribution", "у цитаты нет автора и источника")
    ]


def _strip_quotes(text: str) -> str:
    text = text.strip()
    for left, right in (("«", "»"), ('"', '"'), ("“", "”")):
        if len(text) > 1 and text.startswith(left) and text.endswith(right):
            return text[1:-1].strip()
    return text


def _strip_math_delimiters(latex: str) -> str:
    latex = latex.strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if len(latex) > len(left) + len(right) and latex.startswith(left) and latex.endswith(right):
            return latex[len(left) : -len(right)].strip()
    return latex


def _parse_pairs(segments: list[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for segment in segments:
        if "=" not in segment:
            raise DSLValueError(f"ожидалась пара ключ=значение, получено «{segment}»")
        key, val = segment.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise DSLValueError(f"пустой ключ в «{segment}»")
        if key in pairs:
            raise DSLValueError(f"ключ {key} повторяется")
        pairs[key] = val.strip()
    return pairs


def _segments(value: str) -> list[str]:
    return [s.strip() for s in value.split(";") if s.strip()]


def _parse_number(text: str) -> float:
    try:
        number = float(text.strip())
    except ValueError as exc:
        raise DSLValueError(f"«{text.strip()}» не число (дробная часть пишется через точку)") from exc
    if not math.isfinite(number):
        raise DSLValueError(f"«{text.strip()}» не конечное число")
    return number


def _parse_chart(value: str, limits: ParseLimits) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    pairs = _parse_pairs(_segments(value))
    for key in ("type", "x", "y"):
        if not pairs.get(key):
            raise DSLValueError(f"у графика нет обязательного ключа {key}")
    chart_type = pairs["type"].lower()
    if chart_type not in CHART_TYPES:
        raise DSLValueError(f"тип графика {chart_type} не из {', '.join(CHART_TYPES)}")
    categories = [c.strip() for c in pairs["x"].split(",")]
    if any(not c for c in categories):
        raise DSLValueError("пустая подпись в x")
    values = [_parse_number(v) for v in pairs["y"].split(",")]
    if len(categories) != len(values):
        raise DSLValueError(f"в x {len(categories)} подписей, а в y {len(values)} чисел")
    if not limits.min_chart_points <= len(values) <= limits.max_chart_points:
        raise DSLValueError(f"точек должно быть от {limits.min_chart_points} до {limits.max_chart_points}")
    if chart_type == "pie" and (any(v < 0 for v in values) or sum(values) <= 0):
        raise DSLValueError("для круговой диаграммы значения не могут быть отрицательными, а сумма — нулевой")
    warnings = [
        ("unknown_key", f"неизвестный ключ графика {key} пропущен")
        for key in pairs
        if key not in ("type", "x", "y", "title", "unit")
    ]
    data = {
        "type": chart_type,
        "categories": categories,
        "values": values,
        "title": pairs.get("title") or None,
        "unit": pairs.get("unit") or None,
    }
    return data, warnings


def _parse_plot(value: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    segments = _segments(value)
    if not segments:
        raise DSLValueError("пустой график функции")
    head = segments[0]
    lhs, sep, expression = head.partition("=")
    if not sep or lhs.strip().lower() != "y" or not expression.strip():
        raise DSLValueError("график функции должен начинаться с y = выражение")
    expression = expression.strip()
    compile_expression(expression)
    pairs = _parse_pairs(segments[1:])
    if "x" not in pairs:
        raise DSLValueError("у графика функции нет диапазона x=a..b")
    low_text, dots, high_text = pairs["x"].partition("..")
    if not dots:
        raise DSLValueError("диапазон x записывается как a..b")
    low, high = _parse_number(low_text), _parse_number(high_text)
    if low >= high:
        raise DSLValueError("в диапазоне x=a..b должно быть a < b")
    warnings = [
        ("unknown_key", f"неизвестный ключ графика функции {key} пропущен")
        for key in pairs
        if key not in ("x", "title")
    ]
    return {"expression": expression, "x_min": low, "x_max": high, "title": pairs.get("title") or None}, warnings


def compile_expression(expression: str) -> ast.Expression:
    """Проверяет выражение #PLOT и возвращает его дерево; ничего не выполняет.

    Разрешены числа, x, pi, e, операции + - * / ^, скобки и функции из PLOT_FUNCTIONS; знак умножения
    можно опускать (``0.15x``, ``2(x+1)``).
    """
    source = _IMPLICIT_MUL_RE.sub("*", expression).replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise DSLValueError(f"выражение «{expression}» записано с ошибкой") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Name) and (node.id == "x" or node.id in PLOT_CONSTANTS or node.id in PLOT_FUNCTIONS):
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in PLOT_FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            continue
        raise DSLValueError(f"в выражении «{expression}» недопустимая конструкция")
    # Имя функции без вызова (например, «y = sin») тоже ошибка.
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in PLOT_FUNCTIONS and id(node) not in called:
            raise DSLValueError(f"функция {node.id} используется без аргумента")
    if not any(isinstance(n, ast.Name) and n.id == "x" for n in ast.walk(tree)):
        raise DSLValueError("выражение не зависит от x")
    return tree


def _check_label(label: str, what: str) -> str:
    label = label.strip()
    if not label:
        raise DSLValueError(f"пустая подпись {what}")
    for forbidden in _NODE_FORBIDDEN:
        if forbidden in label:
            raise DSLValueError(f"в подписи {what} «{label}» недопустим символ {forbidden}")
    return label


def _parse_diagram(value: str, limits: ParseLimits) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    segments = _segments(value)
    if not segments or not segments[0].lower().replace(" ", "").startswith("type="):
        raise DSLValueError("схема должна начинаться с type=flow или type=tree")
    diagram_type = segments[0].split("=", 1)[1].strip().lower()
    if diagram_type not in DIAGRAM_TYPES:
        raise DSLValueError(f"тип схемы {diagram_type} не из {', '.join(DIAGRAM_TYPES)}")
    body = segments[1:]
    if not body:
        raise DSLValueError("в схеме нет ни одной связи")

    nodes: list[str] = []
    edges: list[dict[str, Any]] = []

    def node(label: str) -> str:
        label = _check_label(label, "узла")
        if label not in nodes:
            nodes.append(label)
        return label

    if diagram_type == "flow":
        for chain in body:
            parts = _ARROW_RE.split(chain)
            # re.split с группой: [узел, подпись|None, узел, подпись|None, узел, ...]
            if len(parts) < 3:
                raise DSLValueError(f"в цепочке «{chain}» нет стрелки ->")
            chain_nodes = [node(p) for p in parts[0::2]]
            labels = parts[1::2]
            for i, label in enumerate(labels):
                edge_label = _check_label(label, "стрелки") if label and label.strip() else None
                edges.append({"source": chain_nodes[i], "target": chain_nodes[i + 1], "label": edge_label})
    else:
        parents: dict[str, str] = {}
        for segment in body:
            parent_text, sep, children_text = segment.partition(">")
            if not sep:
                raise DSLValueError(f"в «{segment}» нет знака >")
            parent = node(parent_text)
            children = [c for c in children_text.split(",")]
            if not any(c.strip() for c in children):
                raise DSLValueError(f"у узла «{parent}» нет потомков")
            for child_text in children:
                child = node(child_text)
                if child == parent:
                    raise DSLValueError(f"узел «{child}» не может быть потомком самого себя")
                if child in parents:
                    raise DSLValueError(f"у узла «{child}» два родителя")
                parents[child] = parent
                edges.append({"source": parent, "target": child, "label": None})
        roots = [n for n in nodes if n not in parents]
        if len(roots) != 1:
            raise DSLValueError("у дерева должен быть ровно один корень")
        # Проверка циклов: от каждого узла поднимаемся к корню.
        for start in nodes:
            seen = {start}
            current = start
            while current in parents:
                current = parents[current]
                if current in seen:
                    raise DSLValueError("в дереве есть цикл")
                seen.add(current)

    if len(nodes) > limits.max_diagram_nodes:
        raise DSLValueError(f"в схеме {len(nodes)} узлов, допустимо не больше {limits.max_diagram_nodes}")
    if len(nodes) < 2:
        raise DSLValueError("в схеме меньше двух узлов")
    return {"type": diagram_type, "nodes": nodes, "edges": edges}, []


def _parse_timeline(value: str, limits: ParseLimits) -> dict[str, Any]:
    items = []
    for segment in _segments(value):
        parts = _TIMELINE_SPLIT_RE.split(segment, maxsplit=1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise DSLValueError(f"пункт «{segment}» должен иметь вид «дата — событие»")
        items.append({"date": parts[0].strip(), "event": parts[1].strip()})
    if not limits.min_timeline_items <= len(items) <= limits.max_timeline_items:
        raise DSLValueError(
            f"в ленте времени должно быть от {limits.min_timeline_items} до {limits.max_timeline_items} пунктов"
        )
    return {"items": items}
