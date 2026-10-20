# Здесь достаём из HTML статьи Википедии таблицы и карточку статьи: в простом тексте статьи их нет.
"""Таблицы и карточка статьи Википедии простым текстом.

Текст статьи берём из простого текста API (``extracts``), а он выбрасывает таблицы целиком: от раздела-списка
остаётся одна подпись (EXP-009: «Лауреаты — граждане СССР» — одна строка про серый цвет). Числа и списки чаще всего
именно в таблицах, поэтому таблицы и карточку (infobox) берём из HTML статьи (``action=parse``) и дописываем
в конец своего раздела:

- таблица (``wikitable``, в русской Википедии и старый ``standard``) — строка «Таблица: подпись», затем строки ячеек через « | »; объединённые ячейки
  (rowspan, colspan) повторяются в каждой строке и столбце, которые они покрывают;
- карточка (``infobox``) — «Карточка статьи:», затем строки «поле: значение»; дописывается к введению.

Сноски, скрытый текст и ключи сортировки выбрасываются, формулы — LaTeX в ``$…$``. Раздел таблицы — по последнему
заголовку перед ней; заголовки HTML сопоставляются с заголовками простого текста по порядку и названию.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_HEADINGS = {"h2": 1, "h3": 2, "h4": 3, "h5": 4, "h6": 5}
# Что выбрасываем вместе с содержимым: сноски, ссылки «править», скрытое, ключи сортировки, служебные блоки.
_DROP_CLASSES = {"reference", "mw-editsection", "sortkey", "noprint", "mw-cite-backlink", "navbox", "metadata",
                 "mw-empty-elt", "toc"}
_DROP_TAGS = {"style", "script", "img"}
_SPACE = re.compile(r"\s+")


@dataclass
class _Node:
    tag: str
    attrs: dict
    children: list = field(default_factory=list)  # _Node или str

    @property
    def classes(self) -> set[str]:
        return set((self.attrs.get("class") or "").split())


class _TreeBuilder(HTMLParser):
    """Минимальное дерево HTML: Википедия отдаёт корректную разметку, так что лечить вложенность не нужно."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(_Node(tag, dict(attrs)))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):  # закрываем до ближайшего такого же тега
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _hidden(node: _Node) -> bool:
    if node.tag in _DROP_TAGS or node.attrs.get("id") == "toc":
        return True
    style = (node.attrs.get("style") or "").replace(" ", "").lower()
    return bool(node.classes & _DROP_CLASSES) or "display:none" in style


def _text(node: _Node) -> str:
    """Видимый текст узла: формулы — LaTeX в $…$, <br> — «; »."""
    if "mwe-math-element" in node.classes:
        tex = _find(node, lambda n: n.tag == "annotation" and n.attrs.get("encoding") == "application/x-tex")
        if tex is not None:
            latex = _raw(tex).strip()
            if latex.startswith("{\\displaystyle") and latex.endswith("}"):
                latex = latex[len("{\\displaystyle"):-1]
            return f" ${latex.strip()}$ "
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        elif child.tag == "br":
            parts.append("; ")
        elif not _hidden(child):
            text = _text(child)
            parts.append(f" {text} " if child.tag in ("p", "div", "li", "table", "tr") else text)
    return "".join(parts)


def _raw(node: _Node) -> str:
    return "".join(c if isinstance(c, str) else _raw(c) for c in node.children)


def _clean(text: str) -> str:
    text = _SPACE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"(;\s*)+", "; ", text).strip("; ").strip()
    return text


def _find(node: _Node, test) -> _Node | None:
    for child in node.children:
        if isinstance(child, _Node):
            if test(child):
                return child
            found = _find(child, test)
            if found is not None:
                return found
    return None


def _rows(table: _Node) -> list[list[_Node]]:
    """Строки таблицы без строк вложенных таблиц."""
    rows: list[list[_Node]] = []

    def walk(node: _Node) -> None:
        for child in node.children:
            if not isinstance(child, _Node) or child.tag == "table" or _hidden(child):
                continue
            if child.tag == "tr":
                rows.append([c for c in child.children if isinstance(c, _Node) and c.tag in ("th", "td")])
            else:
                walk(child)

    walk(table)
    return rows


def _span(cell: _Node, name: str) -> int:
    try:
        return max(1, min(int(cell.attrs.get(name, "1")), 50))
    except ValueError:
        return 1


def _grid(table: _Node) -> list[list[tuple[str, bool]]]:
    """Таблица как сетка (текст, заголовок ли ячейка); объединённые ячейки повторены."""
    grid: list[list[tuple[str, bool]]] = []
    pending: dict[int, tuple[int, tuple[str, bool]]] = {}  # столбец → (сколько ещё строк, ячейка) для rowspan
    for cells in _rows(table):
        row: list[tuple[str, bool]] = []
        column = 0
        queue = list(cells)
        while queue or any(col >= column for col in pending):
            if column in pending:
                left, value = pending[column]
                row.append(value)
                pending[column] = (left - 1, value)
                if left - 1 == 0:
                    del pending[column]
                column += 1
                continue
            if not queue:
                column += 1
                continue
            cell = queue.pop(0)
            value = (_clean(_text(cell)), cell.tag == "th")
            for _ in range(_span(cell, "colspan")):
                row.append(value)
                if _span(cell, "rowspan") > 1:
                    pending[column] = (_span(cell, "rowspan") - 1, value)
                column += 1
        if any(text for text, _ in row):
            grid.append(row)
    return grid


def _dedupe(row: list[str]) -> list[str]:
    """Ячейка с colspan повторена в соседних столбцах — в строке текста оставляем её один раз."""
    out: list[str] = []
    for text in row:
        if not out or text != out[-1]:
            out.append(text)
    return out


def table_text(table: _Node) -> str:
    """Таблица ``wikitable`` простым текстом."""
    caption = _find(table, lambda n: n.tag == "caption")
    lines = ["Таблица: " + _clean(_text(caption)) if caption is not None else "Таблица:"]
    for row in _grid(table):
        lines.append(" | ".join(_dedupe([text for text, _ in row])))
    return "\n".join(lines)


def infobox_text(table: _Node) -> str:
    """Карточка статьи простым текстом: «поле: значение», заголовки групп — отдельной строкой."""
    lines = ["Карточка статьи:"]
    for row in _grid(table):
        texts = _dedupe([text for text, _ in row])
        if not any(texts):
            continue
        heads = [text for text, is_head in row if is_head and text]
        if len(texts) >= 2 and heads and row[0][1] and not row[-1][1]:
            lines.append(f"{texts[0]}: {' '.join(texts[1:])}")
        else:
            lines.append(" ".join(t for t in texts if t))
    return "\n".join(lines) if len(lines) > 1 else ""


@dataclass
class HtmlTables:
    headings: list[str]  # заголовки разделов в порядке HTML
    tables: list[tuple[int, str]]  # (сколько заголовков было до таблицы: 0 — введение, текст таблицы)


def html_tables(html: str) -> HtmlTables:
    builder = _TreeBuilder()
    builder.feed(html)
    result = HtmlTables([], [])

    def walk(node: _Node) -> None:
        for child in node.children:
            if not isinstance(child, _Node) or _hidden(child):
                continue
            if child.tag in _HEADINGS:
                result.headings.append(_clean(_text(child)))
            elif child.tag == "table" and "infobox" in child.classes:
                text = infobox_text(child)
                if text:
                    result.tables.append((0, text))  # карточка — всегда к введению
            elif child.tag == "table" and child.classes & {"wikitable", "standard"}:  # standard — старый стиль ruwiki
                result.tables.append((len(result.headings), table_text(child)))
            elif child.tag != "table":
                walk(child)

    walk(builder.root)
    return result


def _norm(heading: str) -> str:
    return _SPACE.sub(" ", heading).strip().lower()


def align_headings(html_headings: list[str], text_headings: list[str]) -> list[int]:
    """Номер заголовка простого текста (с 1) для каждого заголовка HTML; 0 — не нашёлся (таблица уйдёт в раздел
    предыдущего найденного заголовка). Идём по порядку: одинаковые заголовки в разных частях статьи не путаются."""
    result: list[int] = []
    position = 0
    for heading in html_headings:
        target = _norm(heading)
        found = next((i for i in range(position, len(text_headings)) if _norm(text_headings[i]) == target), None)
        if found is None:
            result.append(result[-1] if result else 0)
        else:
            result.append(found + 1)
            position = found + 1
    return result
