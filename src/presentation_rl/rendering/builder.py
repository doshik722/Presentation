# Здесь собираем PPTX из разобранной презентации: макеты, текст, нативные графики, схемы и ленту времени; отчёт о неотрисованных элементах.
"""Сборщик презентаций.

Один сборщик для всех моделей и агентов. Он не исправляет содержание,
не переставляет слайды и не добавляет своего текста: всё, что не удалось
нарисовать, записывается в ``RenderReport`` с причиной.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.chart import XL_CHART_TYPE, XL_DATA_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Inches, Pt

from ..dsl import LARGE_VISUALS, Deck, Element, ParseLimits, Slide, parse
from .themes import Theme
from .visuals import DPI, RenderError, render_formula, render_plot

SLIDE_W = 13.333
SLIDE_H = 7.5
MARGIN = 0.6
MAX_LARGE_VISUALS = 2
MAX_FORMULAS = 3

# Размеры шрифтов фиксированы: сборщик не ужимает текст, а сообщает о переполнении.
TITLE_SIZE = 30
TITLE_SLIDE_SIZE = 40
SECTION_SIZE = 36
BODY_SIZE = 20
ATTRIBUTION_SIZE = 16
CAPTION_SIZE = 12
DATE_SIZE = 14
SMALL_SIZE = 10
NODE_SIZES = (14, 13, 12, 11, 10)  # единственное исключение: подпись внутри узла схемы

PROSE = ("BULLET", "TEXT")  # элементы текстового потока


# ------------------------------------------------------------------ оформление


def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


# ---------------------------------------------------------------------- отчёт


@dataclass
class ElementReport:
    slide: int  # номер слайда в презентации, с 1
    kind: str
    status: str  # ok | render_error | overflow | placeholder | text_overflow | bad_source
    line: int | None = None
    message: str = ""


@dataclass
class SlideStats:
    """Заполненность слайда: основа для метрик «пустой слайд» и «слайд без визуала»."""

    slide: int
    text_fill: float | None  # доля высоты текстовой колонки, занятая текстом; None — текста нет
    visuals: int  # нарисованные крупные визуальные элементы
    formulas: int
    sparse: bool  # нет визуала, и текст занимает меньше половины места


@dataclass
class RenderReport:
    path: Path | None
    pptx_created: bool = False
    slide_count: int = 0
    items: list[ElementReport] = field(default_factory=list)
    slides: list[SlideStats] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # источники по порядку номеров
    sources_slide: bool = False  # добавлен ли автоматический последний слайд «Источники» (в slide_count он входит)

    def count(self, status: str) -> int:
        return sum(1 for item in self.items if item.status == status)

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts


@dataclass
class _Source:
    number: int
    title: str
    section: str
    url: str
    lang: str

    def label(self) -> str:
        place = f", раздел «{self.section}»" if self.section else ""
        wiki = "Википедия" if self.lang == "ru" else "англ. Википедия"
        return f"{self.title}{place} ({wiki})"


# По id из #SOURCE возвращает {"title", "section", "url", "lang"} или None, если id не из этого эпизода
# (например, WikiTools.source). Без него любой #SOURCE — ошибка: поиска не было.
SourceResolver = Callable[[str], "dict | None"]

@dataclass
class _Box:
    x: float
    y: float
    w: float
    h: float


# ------------------------------------------------------------------ публичный API


def build_from_text(
    text: str,
    path: str | Path,
    limits: ParseLimits | None = None,
    theme: Theme | None = None,
    sources: SourceResolver | None = None,
    show_sources: bool = True,
) -> tuple[Deck, RenderReport]:
    """Разбирает текст DSL и собирает PPTX."""
    deck = parse(text, limits)
    return deck, build_presentation(deck, path, theme, sources, show_sources)


def build_presentation(
    deck: Deck, path: str | Path, theme: Theme | None = None, sources: SourceResolver | None = None,
    show_sources: bool = True,
) -> RenderReport:
    """Собирает PPTX из разобранной презентации и сохраняет его в ``path``.

    Источники (``#SOURCE``) по id получает ``sources``; на слайде — строка с названиями и ссылками, а если источников
    больше двух или строка не помещается — номера, и тогда в конце добавляется слайд «Источники» со всем списком.
    ``show_sources=False`` (пользователь не хочет источников на слайдах): id проверяются и попадают в отчёт,
    но на слайды не выводятся.
    """
    theme = theme or Theme()
    path = Path(path)
    report = RenderReport(path=path)
    if not deck.slides:
        return report
    slide_sources = _resolve_sources(deck, sources, report)
    if not show_sources:
        slide_sources = [[] for _ in slide_sources]
    numbered = any(_source_numbers(items) for items in slide_sources)

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]

    with tempfile.TemporaryDirectory() as tmp:
        assets = Path(tmp)
        for index, slide_data in enumerate(deck.slides, start=1):
            slide = prs.slides.add_slide(blank)
            _SlideBuilder(slide, slide_data, index, len(deck.slides), theme, assets, report,
                          slide_sources[index - 1]).build()
        if numbered:
            index = len(deck.slides) + 1
            sources_slide = Slide(line=0, title="Источники")
            _SlideBuilder(prs.slides.add_slide(blank), sources_slide, index, index, theme, assets, report).sources_list()
            report.sources_slide = True
        path.parent.mkdir(parents=True, exist_ok=True)
        prs.save(path)

    try:
        report.slide_count = len(Presentation(path).slides)
        report.pptx_created = True
    except Exception:  # файл не открывается — это тоже результат, а не падение
        report.pptx_created = False
    return report


def _resolve_sources(deck: Deck, resolve: SourceResolver | None, report: RenderReport) -> list[list[_Source]]:
    """Источники каждого слайда; номера сквозные, по первому появлению; одна ссылка — один номер."""
    by_url: dict[str, _Source] = {}
    result: list[list[_Source]] = []
    for index, slide in enumerate(deck.slides, start=1):
        items: list[_Source] = []
        for ref in slide.sources:
            info = resolve(ref) if resolve else None
            if info is None:
                report.items.append(ElementReport(index, "SOURCE", "bad_source", slide.line,
                                                  f"«{ref}» не принят: id не из результатов инструментов этого "
                                                  "эпизода или источники не запрошены"))
                continue
            if info["url"] not in by_url:
                by_url[info["url"]] = _Source(len(by_url) + 1, info["title"], info.get("section", ""), info["url"],
                                              info.get("lang", "ru"))
                report.sources.append({"number": len(by_url), "ref": ref, **info})
            source = by_url[info["url"]]
            if source not in items:
                items.append(source)
            report.items.append(ElementReport(index, "SOURCE", "ok", slide.line))
        result.append(items)
    return result


def _source_numbers(items: list[_Source]) -> bool:
    """Нужны ли на слайде номера вместо названий: источников больше двух или строка не помещается."""
    if not items:
        return False
    if len(items) > 2:
        return True
    text = ("Источник: " if len(items) == 1 else "Источники: ") + "; ".join(s.label() for s in items)
    return not _fits(text, SLIDE_W - 2 * MARGIN - 1.0, 0.4, SMALL_SIZE)


# ------------------------------------------------------------ сборка одного слайда


class _SlideBuilder:
    def __init__(self, slide, data: Slide, index: int, total: int, theme: Theme, assets: Path, report: RenderReport,
                 sources: list[_Source] | None = None):
        self.slide = slide
        self.sources = sources or []
        self.data = data
        self.index = index
        self.total = total
        self.t = theme
        self.assets = assets
        self.report = report
        self.is_title_slide = index == 1
        self.text_fill: float | None = None

    # --- общая раскладка

    def build(self) -> None:
        text_blocks, visuals = self._split_elements()
        dark = self.is_title_slide
        if dark:
            fill = self.slide.background.fill
            fill.solid()
            fill.fore_color.rgb = _rgb(self.t.title_bg)

        body_bottom = SLIDE_H - (0.85 if self.sources else 0.7)
        if self.is_title_slide:
            body = self._title_slide_header(text_blocks, visuals)
        elif not text_blocks and not visuals:
            self._section_title()
            body = None
        else:
            title_h = self._title(_Box(MARGIN, 0.35, SLIDE_W - 2 * MARGIN, 1.0))
            body = _Box(MARGIN, 0.35 + title_h + 0.25, SLIDE_W - 2 * MARGIN, 0)
            body.h = body_bottom - body.y

        if body is not None:
            self._layout(body, text_blocks, visuals, dark)
        drawn = sum(
            1 for item in self.report.items
            if item.slide == self.index and item.kind in LARGE_VISUALS and item.status in ("ok", "placeholder")
        )
        formulas = sum(1 for item in self.report.items if item.slide == self.index and item.kind == "FORMULA" and item.status == "ok")
        self.report.slides.append(
            SlideStats(
                slide=self.index,
                text_fill=None if self.text_fill is None else round(self.text_fill, 2),
                visuals=drawn,
                formulas=formulas,
                sparse=drawn == 0 and (self.text_fill is None or self.text_fill < 0.5),
            )
        )

        if self.sources:
            self._sources(dark)
        if not self.is_title_slide:
            self._slide_number()
        if self.data.notes:
            self.slide.notes_slide.notes_text_frame.text = "\n".join(self.data.notes)

    def _split_elements(self) -> tuple[list[list[Element]], list[Element]]:
        """Делит элементы на блоки текстовой колонки и крупные визуальные элементы."""
        text_blocks: list[list[Element]] = []
        visuals: list[Element] = []
        formulas = 0
        for element in self.data.elements:
            if element.kind in LARGE_VISUALS:
                if len(visuals) >= MAX_LARGE_VISUALS:
                    self._record(element, "overflow", f"на слайде уже {MAX_LARGE_VISUALS} крупных визуальных элемента")
                    continue
                visuals.append(element)
                continue
            if element.kind == "FORMULA":
                if formulas >= MAX_FORMULAS:
                    self._record(element, "overflow", f"на слайде уже {MAX_FORMULAS} формулы")
                    continue
                formulas += 1
            # Подряд идущие пункты и абзацы — один текстовый поток без зазоров между ними.
            if element.kind in PROSE and text_blocks and text_blocks[-1][0].kind in PROSE:
                text_blocks[-1].append(element)
            else:
                text_blocks.append([element])
        return text_blocks, visuals

    def _layout(self, body: _Box, text_blocks, visuals, dark: bool) -> None:
        gap = 0.4
        if not visuals:
            self._text_column(body, text_blocks, dark)
            return
        if not text_blocks:
            if len(visuals) == 1:
                self._visual(visuals[0], body)
            else:
                half = (body.h - gap) / 2
                self._visual(visuals[0], _Box(body.x, body.y, body.w, half))
                self._visual(visuals[1], _Box(body.x, body.y + half + gap, body.w, half))
            return
        if len(visuals) == 1 and self._is_wide(visuals[0]) and not self.is_title_slide:
            needed = self._text_height(text_blocks, body.w) - 0.2
            allowed = body.h - 2.2  # визуалу оставляем не меньше 2.2 дюйма
            text_h = min(needed, allowed)
            self._text_column(_Box(body.x, body.y, body.w, text_h), text_blocks, dark)
            self.text_fill = min(1.0, needed / allowed)
            top = body.y + text_h + 0.35
            self._visual(visuals[0], _Box(body.x, top, body.w, body.y + body.h - top))
            return
        share = 0.52 if len(visuals) == 1 else 0.46
        text_w = body.w * share
        self._text_column(_Box(body.x, body.y, text_w, body.h), text_blocks, dark)
        vis = _Box(body.x + text_w + gap, body.y, body.w - text_w - gap, body.h)
        if len(visuals) == 1:
            self._visual(visuals[0], vis)
        else:
            half = (vis.h - 0.3) / 2
            self._visual(visuals[0], _Box(vis.x, vis.y, vis.w, half))
            self._visual(visuals[1], _Box(vis.x, vis.y + half + 0.3, vis.w, half))

    @staticmethod
    def _is_wide(element: Element) -> bool:
        """Элементы, которым нужна вся ширина слайда: лента времени, поток, широкое дерево."""
        if element.kind == "TIMELINE":
            return True
        if element.kind != "DIAGRAM":
            return False
        if element.data["type"] == "flow":
            return True
        parents = {e["source"] for e in element.data["edges"]}
        leaves = [n for n in element.data["nodes"] if n not in parents]
        return len(leaves) >= 4

    # --- заголовки

    def _title(self, box: _Box) -> float:
        title = self.data.title
        if not title:
            return 0.4
        size = TITLE_SIZE
        lines = _estimate_lines(title, box.w - 0.1, size, bold=True, heading=True)
        height = max(0.6, lines * size * 1.2 / 72 + 0.1)
        tf = self._textbox(_Box(box.x, box.y, box.w, height), anchor=MSO_ANCHOR.TOP)
        self._add_run(tf.paragraphs[0], title, size, self.t.ink, font=self.t.heading_font, bold=True)
        return height

    def _section_title(self) -> None:
        title = self.data.title or ""
        tf = self._textbox(_Box(MARGIN, 2.6, SLIDE_W - 2 * MARGIN, 1.6), anchor=MSO_ANCHOR.MIDDLE)
        self._add_run(tf.paragraphs[0], title, SECTION_SIZE, self.t.ink, font=self.t.heading_font, bold=True)

    def _title_slide_header(self, text_blocks, visuals) -> _Box:
        title = self.data.title or ""
        width = SLIDE_W - 2 * MARGIN if not visuals else 7.0
        size = TITLE_SLIDE_SIZE
        lines = _estimate_lines(title, width - 0.1, size, bold=True, heading=True)
        height = lines * size * 1.2 / 72 + 0.1
        top = 1.2 if text_blocks else (SLIDE_H - height) / 2
        # Заголовок прижат к нижнему краю своей области: если он всё же займёт больше строк, чем по оценке,
        # лишняя строка уйдёт вверх, в пустое поле, а не на подзаголовок.
        tf = self._textbox(_Box(MARGIN, top, width, height), anchor=MSO_ANCHOR.BOTTOM)
        self._add_run(tf.paragraphs[0], title, size, self.t.title_text, font=self.t.heading_font, bold=True)
        body_top = top + height + 0.4
        if visuals:
            # Визуальный элемент титульного слайда — справа на всю высоту.
            self._visual(visuals[0], _Box(MARGIN + width + 0.4, 1.0, SLIDE_W - MARGIN - (MARGIN + width + 0.4), 5.4))
            for extra in visuals[1:]:
                self._record(extra, "overflow", "на титульном слайде рисуется один визуальный элемент")
            visuals.clear()
            return _Box(MARGIN, body_top, width, SLIDE_H - 0.8 - body_top)
        return _Box(MARGIN, body_top, width, SLIDE_H - 0.8 - body_top)

    # --- текстовая колонка

    def _text_height(self, blocks, width: float, size: int = BODY_SIZE) -> float:
        total = 0.0
        for block in blocks:
            total += self._block_height(block, width, size) + 0.2
        return total

    def _block_height(self, block: list[Element], width: float, size: int) -> float:
        kind = block[0].kind
        if kind in PROSE:
            return sum(
                _estimate_lines(e.data["text"], width - 0.1 - (0.32 if e.kind == "BULLET" else 0.0), size) * size * 1.2 / 72
                + size * 0.5 / 72
                for e in block
            ) + 0.1
        if kind == "QUOTE":
            e = block[0]
            inner = width - 0.6
            # Курсив Cambria шире Calibri: считаем как жирный.
            height = _estimate_lines("«" + e.data["text"] + "»", inner, size, bold=True) * size * 1.25 / 72
            if e.data.get("attribution"):
                height += _estimate_lines(e.data["attribution"], inner, ATTRIBUTION_SIZE) * ATTRIBUTION_SIZE * 1.25 / 72 + 0.12
            return height + 0.45
        if kind == "FORMULA":
            return 0.75
        return 0.0

    def _text_column(self, box: _Box, blocks, dark: bool) -> None:
        size = BODY_SIZE
        needed = self._text_height(blocks, box.w, size) - 0.2
        self.text_fill = min(1.0, needed / box.h) if box.h > 0 else 1.0
        if needed > box.h + 0.05:
            # Шрифт не уменьшаем: переполнение видно на слайде и попадает в отчёт.
            self.report.items.append(
                ElementReport(
                    self.index,
                    "TEXT",
                    "text_overflow",
                    blocks[0][0].line,
                    f"текст не помещается: нужно около {needed:.1f} дюйма, доступно {box.h:.1f}",
                )
            )

        y = box.y
        for block in blocks:
            kind = block[0].kind
            if kind in PROSE:
                h = self._block_height(block, box.w, size)
                self._paragraphs(_Box(box.x, y, box.w, h), block, size, dark)
            elif kind == "QUOTE":
                h = self._block_height(block, box.w, size)
                self._quote(_Box(box.x, y, box.w, h), block[0], size, dark)
            else:
                h = self._formula(_Box(box.x, y, box.w, 0.75), block[0], dark)
            y += h + 0.2

    def _paragraphs(self, box: _Box, block: list[Element], size: int, dark: bool) -> None:
        """Пункты (#BULLET) и абзацы (#TEXT) одного текстового потока."""
        tf = self._textbox(box, anchor=MSO_ANCHOR.TOP)
        color = self.t.title_body if dark else self.t.body
        for i, element in enumerate(block):
            paragraph = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            self._add_run(paragraph, element.data["text"], size, color)
            paragraph.space_after = Pt(size * 0.5)
            if element.kind == "BULLET":
                _set_bullet(paragraph, self.t.accent, indent_in=0.32)
            self._record(element, "ok")

    def _quote(self, box: _Box, element: Element, size: int, dark: bool) -> None:
        shape = self.slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(box.x), Inches(box.y), Inches(box.w), Inches(box.h)
        )
        shape.adjustments[0] = 0.08
        shape.fill.solid()
        shape.fill.fore_color.rgb = _rgb(self.t.accent_soft)
        shape.line.fill.background()
        shape.shadow.inherit = False
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for side in ("margin_left", "margin_right"):
            setattr(tf, side, Inches(0.25))
        tf.margin_top = tf.margin_bottom = Inches(0.12)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        self._add_run(p, "«" + element.data["text"] + "»", size, self.t.ink, font=self.t.heading_font, italic=True)
        if element.data.get("attribution"):
            a = tf.add_paragraph()
            a.alignment = PP_ALIGN.RIGHT
            a.space_before = Pt(6)
            self._add_run(a, "— " + element.data["attribution"], ATTRIBUTION_SIZE, self.t.muted)
        self._record(element, "ok")

    def _formula(self, box: _Box, element: Element, dark: bool) -> float:
        png = self.assets / f"formula_{self.index}_{element.line}.png"
        try:
            width_px, height_px = render_formula(
                element.data["latex"], png, color="#" + (self.t.title_text if dark else self.t.ink)
            )
        except RenderError as exc:
            self._record(element, "render_error", str(exc))
            return -0.2  # место не занимаем
        width_in, height_in = width_px / DPI, height_px / DPI
        scale = min(1.0, (box.w - 0.3) / width_in, 0.9 / height_in)
        width_in, height_in = width_in * scale, height_in * scale
        self.slide.shapes.add_picture(str(png), Inches(box.x + 0.3), Inches(box.y), Inches(width_in), Inches(height_in))
        self._record(element, "ok")
        return height_in

    # --- визуальные элементы

    def _visual(self, element: Element, box: _Box) -> None:
        handlers = {
            "CHART": self._chart,
            "PLOT": self._plot,
            "TIMELINE": self._timeline,
            "DIAGRAM": self._diagram,
            "IMAGE": self._image_placeholder,
        }
        handlers[element.kind](element, box)

    def _chart(self, element: Element, box: _Box) -> None:
        d = element.data
        chart_data = CategoryChartData()
        chart_data.categories = d["categories"]
        chart_data.add_series(d["title"] or "Значения", d["values"])
        kind = {
            "bar": XL_CHART_TYPE.COLUMN_CLUSTERED,
            "line": XL_CHART_TYPE.LINE_MARKERS,
            "pie": XL_CHART_TYPE.PIE,
        }[d["type"]]
        frame = self.slide.shapes.add_chart(kind, Inches(box.x), Inches(box.y), Inches(box.w), Inches(box.h), chart_data)
        chart = frame.chart
        chart.font.name = self.t.body_font
        chart.font.size = Pt(11)
        chart.font.color.rgb = _rgb(self.t.body)
        if d["title"]:
            chart.has_title = True
            chart.chart_title.text_frame.text = d["title"]
            title_font = chart.chart_title.text_frame.paragraphs[0].runs[0].font
            title_font.size = Pt(13)
            title_font.bold = True
            title_font.color.rgb = _rgb(self.t.ink)
        else:
            chart.has_title = False
        plot = chart.plots[0]
        series = plot.series[0]

        if d["type"] == "pie":
            for i, point in enumerate(series.points):
                point.format.fill.solid()
                point.format.fill.fore_color.rgb = _rgb(self.t.palette[i % len(self.t.palette)])
            chart.has_legend = True
            chart.legend.position = XL_LEGEND_POSITION.BOTTOM
            chart.legend.include_in_layout = False
            chart.legend.font.size = Pt(11)
            chart.legend.font.color.rgb = _rgb(self.t.body)
            plot.has_data_labels = True
            labels = plot.data_labels
            labels.show_category_name = False
            labels.show_percentage = True
            labels.show_value = False
            labels.number_format = "0%"
            labels.number_format_is_linked = False
            labels.position = XL_DATA_LABEL_POSITION.INSIDE_END
            labels.font.size = Pt(12)
            labels.font.bold = True
            labels.font.color.rgb = _rgb("FFFFFF")
        else:
            chart.has_legend = False
            if d["type"] == "bar":
                plot.gap_width = 70
                series.format.fill.solid()
                series.format.fill.fore_color.rgb = _rgb(self.t.accent)
            else:
                series.smooth = False
                series.format.line.color.rgb = _rgb(self.t.accent)
                series.format.line.width = Pt(2.5)
                series.marker.style = XL_MARKER_STYLE.CIRCLE
                series.marker.size = 7
                series.marker.format.fill.solid()
                series.marker.format.fill.fore_color.rgb = _rgb(self.t.accent)
                series.marker.format.line.color.rgb = _rgb(self.t.accent)
            plot.has_data_labels = True
            labels = plot.data_labels
            labels.show_value = True
            labels.font.size = Pt(10)
            labels.font.color.rgb = _rgb(self.t.body)
            labels.position = XL_DATA_LABEL_POSITION.OUTSIDE_END if d["type"] == "bar" else XL_DATA_LABEL_POSITION.ABOVE
            value_axis = chart.value_axis
            value_axis.has_major_gridlines = True
            value_axis.major_gridlines.format.line.color.rgb = _rgb(self.t.grid)
            value_axis.format.line.fill.background()
            value_axis.tick_labels.font.size = Pt(10)
            value_axis.tick_labels.font.color.rgb = _rgb(self.t.muted)
            if d["unit"]:
                value_axis.has_title = True
                value_axis.axis_title.text_frame.text = d["unit"]
                axis_font = value_axis.axis_title.text_frame.paragraphs[0].runs[0].font
                axis_font.size = Pt(10)
                axis_font.bold = False
                axis_font.color.rgb = _rgb(self.t.muted)
            category_axis = chart.category_axis
            category_axis.tick_labels.font.size = Pt(10)
            category_axis.format.line.color.rgb = _rgb(self.t.grid)
        self._record(element, "ok")

    def _plot(self, element: Element, box: _Box) -> None:
        d = element.data
        png = self.assets / f"plot_{self.index}_{element.line}.png"
        try:
            render_plot(
                d["expression"], d["x_min"], d["x_max"], png, d["title"], size_in=(box.w, box.h),
                line_color="#" + self.t.accent, text_color="#" + self.t.ink,
            )
        except RenderError as exc:
            self._record(element, "render_error", str(exc))
            return
        self.slide.shapes.add_picture(str(png), Inches(box.x), Inches(box.y), Inches(box.w), Inches(box.h))
        self._record(element, "ok")

    def _image_placeholder(self, element: Element, box: _Box) -> None:
        """Поиск в Wikimedia Commons ещё не подключён: на месте картинки рисуется заглушка с описанием."""
        shape = self.slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(box.x), Inches(box.y), Inches(box.w), Inches(box.h)
        )
        shape.adjustments[0] = 0.04
        shape.fill.solid()
        shape.fill.fore_color.rgb = _rgb(self.t.placeholder_fill)
        shape.line.color.rgb = _rgb(self.t.placeholder_line)
        shape.line.width = Pt(1.25)
        shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        shape.shadow.inherit = False
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        self._add_run(p, "Изображение", CAPTION_SIZE, self.t.muted, bold=True)
        q = tf.add_paragraph()
        q.alignment = PP_ALIGN.CENTER
        self._add_run(q, element.data["text"], CAPTION_SIZE, self.t.muted, italic=True)
        self._record(element, "placeholder", "поиск картинок ещё не подключён")

    def _timeline(self, element: Element, box: _Box) -> None:
        items = element.data["items"]
        n = len(items)
        cell = box.w / n
        date_size = DATE_SIZE
        event_size = CAPTION_SIZE
        axis_y = box.y + min(0.75, box.h * 0.3)
        text_top = axis_y + 0.22
        text_h = box.y + box.h - text_top
        if not all(_fits(item["event"], cell - 0.2, text_h, event_size) for item in items):
            self.report.items.append(
                ElementReport(self.index, "TIMELINE", "text_overflow", element.line, "подписи событий не помещаются")
            )

        first_x = box.x + cell / 2
        last_x = box.x + box.w - cell / 2
        line = self.slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(first_x - 0.2), Inches(axis_y), Inches(last_x + 0.2), Inches(axis_y)
        )
        line.line.color.rgb = _rgb(self.t.placeholder_line)
        line.line.width = Pt(2)

        dot = 0.2
        for i, item in enumerate(items):
            cx = box.x + cell * (i + 0.5)
            marker = self.slide.shapes.add_shape(
                MSO_SHAPE.OVAL, Inches(cx - dot / 2), Inches(axis_y - dot / 2), Inches(dot), Inches(dot)
            )
            marker.fill.solid()
            marker.fill.fore_color.rgb = _rgb(self.t.accent)
            marker.line.color.rgb = _rgb("FFFFFF")
            marker.line.width = Pt(1.5)
            marker.shadow.inherit = False

            date_box = self._textbox(
                _Box(cx - cell / 2 + 0.05, axis_y - 0.62, cell - 0.1, 0.5), anchor=MSO_ANCHOR.BOTTOM
            )
            date_box.paragraphs[0].alignment = PP_ALIGN.CENTER
            self._add_run(date_box.paragraphs[0], item["date"], date_size, self.t.accent, bold=True)

            event_box = self._textbox(_Box(cx - cell / 2 + 0.07, text_top, cell - 0.14, text_h), anchor=MSO_ANCHOR.TOP)
            event_box.paragraphs[0].alignment = PP_ALIGN.CENTER
            self._add_run(event_box.paragraphs[0], item["event"], event_size, self.t.body)
        self._record(element, "ok")

    # --- схемы

    def _diagram(self, element: Element, box: _Box) -> None:
        if element.data["type"] == "tree":
            _TreeDrawer(self, element.data, box).draw()
        else:
            _FlowDrawer(self, element.data, box).draw()
        self._record(element, "ok")

    # --- подвал

    def _sources(self, dark: bool) -> None:
        """Строка источников внизу слайда: названия со ссылками или номера из списка на последнем слайде."""
        color = self.t.title_body if dark else self.t.muted
        tf = self._textbox(_Box(MARGIN, SLIDE_H - 0.72, SLIDE_W - 2 * MARGIN - 1.0, 0.4), anchor=MSO_ANCHOR.BOTTOM)
        paragraph = tf.paragraphs[0]
        numbers = _source_numbers(self.sources)
        self._add_run(paragraph, "Источник: " if len(self.sources) == 1 else "Источники: ", SMALL_SIZE, color)
        for i, source in enumerate(self.sources):
            if i:
                self._add_run(paragraph, ", " if numbers else "; ", SMALL_SIZE, color)
            run = self._add_run(paragraph, str(source.number) if numbers else source.label(), SMALL_SIZE, color)
            run.hyperlink.address = source.url

    def sources_list(self) -> None:
        """Автоматический последний слайд «Источники»: все источники по номерам, со ссылками."""
        title_h = self._title(_Box(MARGIN, 0.35, SLIDE_W - 2 * MARGIN, 1.0))
        box = _Box(MARGIN, 0.35 + title_h + 0.25, SLIDE_W - 2 * MARGIN, SLIDE_H - 0.7 - (0.6 + title_h))
        sources = self.report.sources
        labels = [_Source(s["number"], s["title"], s.get("section", ""), s["url"], s.get("lang", "ru")).label()
                  for s in sources]
        lines = [f"{n}. {label}" for n, label in enumerate(labels, 1)]
        size = next((size for size in (16, 14, 12, 11, 10) if sum(
            _estimate_lines(line, box.w - 0.1, size) * size * 1.2 / 72 + size * 0.4 / 72 for line in lines) <= box.h), 10)
        tf = self._textbox(box)
        for i, (source, label) in enumerate(zip(sources, labels)):
            paragraph = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            paragraph.space_after = Pt(size * 0.4)
            self._add_run(paragraph, f"{source['number']}. ", size, self.t.muted)
            run = self._add_run(paragraph, label, size, self.t.ink)
            run.hyperlink.address = source["url"]
        self._slide_number()

    def _slide_number(self) -> None:
        tf = self._textbox(_Box(SLIDE_W - MARGIN - 0.8, SLIDE_H - 0.62, 0.8, 0.3), anchor=MSO_ANCHOR.BOTTOM)
        tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
        self._add_run(tf.paragraphs[0], str(self.index), SMALL_SIZE, self.t.muted)

    # --- примитивы

    def _textbox(self, box: _Box, anchor=MSO_ANCHOR.TOP):
        shape = self.slide.shapes.add_textbox(Inches(box.x), Inches(box.y), Inches(box.w), Inches(max(box.h, 0.2)))
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.05)
        tf.margin_top = tf.margin_bottom = Inches(0.03)
        return tf

    def _add_run(self, paragraph, text: str, size: float, color: str, font: str | None = None, bold=False, italic=False):
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.name = font or self.t.body_font
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = _rgb(color)
        return run

    def _record(self, element: Element, status: str, message: str = "") -> None:
        self.report.items.append(ElementReport(self.index, element.kind, status, element.line, message))


# ----------------------------------------------------------------------- схемы


class _DiagramDrawer:
    """Общие примитивы для схем: узлы и соединители из фигур PowerPoint."""

    def __init__(self, builder: _SlideBuilder, data: dict, box: _Box):
        self.b = builder
        self.t = builder.t
        self.shapes = builder.slide.shapes
        self.data = data
        self.box = box

    def node(self, label: str, x: float, y: float, w: float, h: float, emphasis: bool = False):
        shape = self.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.adjustments[0] = 0.18
        shape.fill.solid()
        shape.fill.fore_color.rgb = _rgb(self.t.accent if emphasis else self.t.accent_soft)
        shape.line.color.rgb = _rgb(self.t.accent)
        shape.line.width = Pt(1.25)
        shape.shadow.inherit = False
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.06)
        tf.margin_top = tf.margin_bottom = Inches(0.03)
        size = _fit_size(label, w - 0.14, h - 0.08, NODE_SIZES, bold=emphasis)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        self.b._add_run(p, label, size, "FFFFFF" if emphasis else self.t.ink, bold=emphasis)
        return shape

    def style_line(self, line_format, arrow: bool = True) -> None:
        line_format.color.rgb = _rgb(self.t.connector)
        line_format.width = Pt(1.5)
        if arrow:
            ln = line_format._get_or_add_ln()
            tail = ln.find(qn("a:tailEnd"))
            if tail is None:
                tail = OxmlElement("a:tailEnd")
                ln.append(tail)
            tail.set("type", "triangle")
            tail.set("w", "med")
            tail.set("len", "med")

    def polyline(self, points: list[tuple[float, float]], arrow: bool = True):
        builder = self.shapes.build_freeform(Inches(points[0][0]), Inches(points[0][1]), scale=1.0)
        builder.add_line_segments([(Inches(x), Inches(y)) for x, y in points[1:]], close=False)
        shape = builder.convert_to_shape()
        shape.fill.background()
        shape.shadow.inherit = False
        self.style_line(shape.line, arrow)
        return shape

    def label(self, text: str, cx: float, cy: float, width: float) -> None:
        size = CAPTION_SIZE
        lines = _estimate_lines(text, width - 0.1, size)
        height = lines * size * 1.2 / 72 + 0.08
        tf = self.b._textbox(_Box(cx - width / 2, cy - height, width, height), anchor=MSO_ANCHOR.BOTTOM)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        self.b._add_run(p, text, size, self.t.muted, italic=True)


class _FlowDrawer(_DiagramDrawer):
    """Поток: послойная раскладка слева направо; стрелки назад — дугой снизу."""

    def draw(self) -> None:
        nodes: list[str] = self.data["nodes"]
        edges: list[dict] = self.data["edges"]
        back = self._back_edges(nodes, edges)
        forward = [e for i, e in enumerate(edges) if i not in back]
        layer = {n: 0 for n in nodes}
        for _ in range(len(nodes)):
            for e in forward:
                layer[e["target"]] = max(layer[e["target"]], layer[e["source"]] + 1)
        layers: list[list[str]] = [[] for _ in range(max(layer.values()) + 1)]
        for n in nodes:
            layers[layer[n]].append(n)
        count = len(layers)
        rows = max(len(col) for col in layers)

        box = self.box
        skip = [e for e in forward if layer[e["target"]] - layer[e["source"]] > 1]
        labelled_back = any(edges[i]["label"] for i in back)
        top_reserve = 0.55 if skip else 0.0
        bottom_reserve = (0.35 + 0.18 * (len(back) - 1) + (0.3 if labelled_back else 0.0)) if back else 0.0
        adjacent_labels = [
            e["label"] for e in forward if e["label"] and layer[e["target"]] - layer[e["source"]] == 1
        ]
        # Промежуток между слоями вмещает самое длинное слово подписи (в две строки максимум).
        gap_x = 0.55
        if adjacent_labels:
            longest = max(_longest_word(label) for label in adjacent_labels)
            gap_x = max(1.0, longest * CAPTION_SIZE * 0.62 / 72 + 0.2)
        node_w = min(2.4, (box.w - (count - 1) * gap_x) / count)
        if node_w < 1.0:  # слишком узко: жертвуем промежутками, но не ниже разумного
            gap_x = max(0.4, (box.w - count * 1.0) / max(count - 1, 1))
            node_w = min(2.4, (box.w - (count - 1) * gap_x) / count)
        label_space = 0.5 if adjacent_labels and rows == 1 else 0.0
        gap_y = 0.45 if adjacent_labels else 0.3
        node_h = 0.85 if rows == 1 else 0.75
        drawing_h = top_reserve + label_space + rows * node_h + (rows - 1) * gap_y + bottom_reserve
        if drawing_h > box.h:  # не помещаемся по высоте — сжимаем блоки
            node_h = max(0.45, (box.h - top_reserve - label_space - bottom_reserve - (rows - 1) * gap_y) / rows)
            drawing_h = top_reserve + label_space + rows * node_h + (rows - 1) * gap_y + bottom_reserve
        offset = min(max(box.h - drawing_h, 0) / 2, 0.35)
        area_top = box.y + offset + top_reserve + label_space
        area_h = rows * node_h + (rows - 1) * gap_y
        total_w = count * node_w + (count - 1) * gap_x
        x0 = box.x + (box.w - total_w) / 2

        pos: dict[str, tuple[float, float]] = {}
        shapes: dict[str, object] = {}
        for li, column in enumerate(layers):
            column_h = len(column) * node_h + (len(column) - 1) * gap_y
            y0 = area_top + (area_h - column_h) / 2
            for ri, n in enumerate(column):
                x = x0 + li * (node_w + gap_x)
                y = y0 + ri * (node_h + gap_y)
                pos[n] = (x, y)
                shapes[n] = self.node(n, x, y, node_w, node_h)

        def center(n):
            x, y = pos[n]
            return x + node_w / 2, y + node_h / 2

        for e in forward:
            s, t = e["source"], e["target"]
            (sx, sy), (tx, ty) = pos[s], pos[t]
            if layer[t] - layer[s] == 1:
                connector = self.shapes.add_connector(
                    MSO_CONNECTOR.STRAIGHT,
                    Inches(sx + node_w), Inches(sy + node_h / 2), Inches(tx), Inches(ty + node_h / 2),
                )
                connector.begin_connect(shapes[s], 3)
                connector.end_connect(shapes[t], 1)
                self.style_line(connector.line)
                if e["label"]:
                    mx = (sx + node_w + tx) / 2
                    my = min(sy, ty) + node_h / 2 - 0.05
                    self.label(e["label"], mx, my, gap_x - 0.06)
            else:
                route_y = area_top - label_space - top_reserve + 0.2
                scx, tcx = center(s)[0], center(t)[0]
                self.polyline([(scx, sy), (scx, route_y), (tcx, route_y), (tcx, ty)])
                if e["label"]:
                    self.label(e["label"], (scx + tcx) / 2, route_y - 0.02, abs(tcx - scx) * 0.6)

        for k, i in enumerate(back):
            e = edges[i]
            s, t = e["source"], e["target"]
            (sx, sy), (tx, ty) = pos[s], pos[t]
            route_y = area_top + area_h + 0.25 + 0.18 * k
            scx, tcx = sx + node_w / 2, tx + node_w / 2
            if abs(scx - tcx) < 0.01:  # ребро в соседний узел того же слоя
                scx += node_w / 4
                tcx -= node_w / 4
            self.polyline([(scx, sy + node_h), (scx, route_y), (tcx, route_y), (tcx, ty + node_h)])
            if e["label"]:
                self.label(e["label"], (scx + tcx) / 2, route_y + 0.3, abs(scx - tcx) * 0.6)

    @staticmethod
    def _back_edges(nodes: list[str], edges: list[dict]) -> set[int]:
        """Рёбра, замыкающие цикл при обходе в глубину в порядке появления узлов."""
        out: dict[str, list[int]] = {n: [] for n in nodes}
        for i, e in enumerate(edges):
            out[e["source"]].append(i)
        state = {n: 0 for n in nodes}  # 0 — не посещён, 1 — в стеке, 2 — готов
        back: set[int] = set()

        def visit(n: str) -> None:
            state[n] = 1
            for i in out[n]:
                t = edges[i]["target"]
                if state[t] == 1:
                    back.add(i)
                elif state[t] == 0:
                    visit(t)
            state[n] = 2

        for n in nodes:
            if state[n] == 0:
                visit(n)
        return back


class _TreeDrawer(_DiagramDrawer):
    """Дерево: уровни сверху вниз, родитель над центром своих потомков."""

    def draw(self) -> None:
        nodes: list[str] = self.data["nodes"]
        children: dict[str, list[str]] = {n: [] for n in nodes}
        has_parent: set[str] = set()
        for e in self.data["edges"]:
            children[e["source"]].append(e["target"])
            has_parent.add(e["target"])
        root = next(n for n in nodes if n not in has_parent)

        depth: dict[str, int] = {}
        order: list[str] = []

        def walk(n: str, d: int) -> None:
            depth[n] = d
            order.append(n)
            for c in children[n]:
                walk(c, d + 1)

        walk(root, 0)
        leaves = [n for n in order if not children[n]]
        levels = max(depth.values()) + 1

        box = self.box
        gap_x, gap_y = 0.25, 0.5
        node_w = min(2.6, (box.w - (len(leaves) - 1) * gap_x) / len(leaves))
        node_h = max(0.45, min(0.9, (box.h - (levels - 1) * gap_y) / levels))
        total_w = len(leaves) * node_w + (len(leaves) - 1) * gap_x
        total_h = levels * node_h + (levels - 1) * gap_y
        x0 = box.x + (box.w - total_w) / 2
        y0 = box.y + min(max(box.h - total_h, 0) / 2, 0.35)

        cx: dict[str, float] = {}
        for i, leaf in enumerate(leaves):
            cx[leaf] = x0 + i * (node_w + gap_x) + node_w / 2
        for n in reversed(order):
            if children[n]:
                cx[n] = sum(cx[c] for c in children[n]) / len(children[n])

        shapes = {}
        for n in order:
            y = y0 + depth[n] * (node_h + gap_y)
            shapes[n] = self.node(n, cx[n] - node_w / 2, y, node_w, node_h, emphasis=(n == root))
        for n in order:
            for c in children[n]:
                py = y0 + depth[n] * (node_h + gap_y) + node_h
                cy = y0 + depth[c] * (node_h + gap_y)
                connector = self.shapes.add_connector(
                    MSO_CONNECTOR.ELBOW, Inches(cx[n]), Inches(py), Inches(cx[c]), Inches(cy)
                )
                connector.begin_connect(shapes[n], 2)
                connector.end_connect(shapes[c], 0)
                self.style_line(connector.line, arrow=False)


# ------------------------------------------------------------- оценка текста


def _estimate_lines(text: str, width_in: float, size_pt: float, bold: bool = False, heading: bool = False) -> int:
    """Грубая оценка числа строк при переносе по словам (средняя ширина символа Calibri/Cambria).

    Для шрифта заголовков (жирный Cambria) берём запас: на кириллице средний символ около 0,57–0,59 em,
    а LibreOffice переносит ещё раньше, чем дают метрики шрифта.
    """
    factor = 0.68 if heading else 0.6 if bold else 0.56
    per_line = max(1, int(width_in / (size_pt * factor / 72)))
    lines, current = 1, 0
    # После дефиса строка тоже может переноситься («Нормативно-» / «правовое»): считаем части отдельными словами.
    for word in text.replace("-", "- ").split():
        length = len(word)
        if current == 0:
            current = length
        elif current + 1 + length <= per_line:
            current += 1 + length
        else:
            lines += 1
            current = length
        while current > per_line:
            lines += 1
            current -= per_line
    return lines


def _longest_word(text: str) -> int:
    return max((len(w) for w in text.split()), default=1)


def _fits(text: str, width_in: float, height_in: float, size: float, bold: bool = False) -> bool:
    """Помещается ли текст в прямоугольник без разрыва слов."""
    if _longest_word(text) * size * (0.66 if bold else 0.62) / 72 > width_in:
        return False
    return _estimate_lines(text, width_in, size, bold) * size * 1.2 / 72 <= height_in


def _fit_size(text: str, width_in: float, height_in: float, sizes: tuple[int, ...], bold: bool = False) -> int:
    for size in sizes:
        if _fits(text, width_in, height_in, size, bold):
            return size
    return sizes[-1]


def _set_bullet(paragraph, color_hex: str, indent_in: float = 0.3) -> None:
    """Маркер списка через свойства абзаца (а не символ в тексте)."""
    p_pr = paragraph._p.get_or_add_pPr()
    p_pr.set("marL", str(Emu(Inches(indent_in))))
    p_pr.set("indent", str(-Emu(Inches(indent_in))))
    for tag in ("a:buClr", "a:buSzPct", "a:buFont", "a:buChar", "a:buNone", "a:buAutoNum"):
        for existing in p_pr.findall(qn(tag)):
            p_pr.remove(existing)
    bu_clr = OxmlElement("a:buClr")
    srgb = OxmlElement("a:srgbClr")
    srgb.set("val", color_hex)
    bu_clr.append(srgb)
    bu_font = OxmlElement("a:buFont")
    bu_font.set("typeface", "Arial")
    bu_char = OxmlElement("a:buChar")
    bu_char.set("char", "•")
    for element in (bu_clr, bu_font, bu_char):
        p_pr.insert_element_before(element, "a:tabLst", "a:defRPr", "a:extLst")

