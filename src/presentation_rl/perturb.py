"""Испорченные презентации для проверки награды (decisions.md, раздел 7, решения 29.10 и 01.11.2026).

Исходники — исправленные вручную ответы T-lite (``data/perturbation/originals/<id>.txt``), разметка —
``data/perturbation/annotations.json`` (какие слайды раскрывают какой тезис, независимые слайды, числа ключей,
словоформы, факты для диагностики, перефразировки). Варианты строятся правкой текста DSL по слайдам; парсер
не нужен, награда не используется — пары не зависят от проверяемой конфигурации. Всё сохраняется один раз
(``data/perturbation/variants/<id>.json``): полный текст каждого варианта, контекст эпизода (вызовы, выдача
инструментов, допустимые источники) и манифест пар с ожиданиями.

Ожидания — для компонента «как он входит в награду»: у штрафов знак обратный, «down» всегда значит «хуже».
``down`` — Δ ≤ −m; ``not_up`` — Δ ≤ ε; ``same`` — |Δ| ≤ ε; m и ε считает оценка. ``checks`` — предусловия,
которые проверяет оценка по отчёту сборщика до подсчёта награды (дефект действительно получился).

Запуск: ``python -m presentation_rl.perturb`` из корня репозитория.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dsl import LARGE_VISUALS
from .environment import NO_SEARCH_MESSAGE, parse_tool_calls
from .key_points import _CONCLUSION_RE, KeyPoint, expected_theses
from .rendering.builder import MAX_FORMULAS, MAX_LARGE_VISUALS
from .rewards import FORMULA_TOTAL_PER_SLIDE, _normalize_quote, load_key_points, quote_found, type_limit
from .sft_search import _Ledger
from .tasks import load_tasks

DOWN, NOT_UP, SAME = "down", "not_up", "same"
TEXTUAL = ("TEXT", "BULLET")
BODY = ("TEXT", "BULLET", "QUOTE", "FORMULA", "CHART", "PLOT", "DIAGRAM", "TIMELINE", "IMAGE")
VISUAL = ("CHART", "PLOT", "DIAGRAM", "TIMELINE", "IMAGE", "QUOTE", "FORMULA")
THESES_PER_KIND = 3  # тезисных искажений каждого вида на исходник: первый, средний и последний подходящий

IMAGE_LINE = "#IMAGE: абстрактный декоративный узор"
CHART_LINE = "#CHART: type=bar; x=А, Б, В; y=3, 5, 2; title=Показатели"
FORMULA_LINE = "#FORMULA: a + b = c"
FAKE_QUOTES = (
    "Знание — фундамент любого успешного дела — Неизвестный автор",
    "Тот, кто понимает прошлое, управляет будущим — Неизвестный автор",
    "Каждое открытие начинается с простого вопроса — Неизвестный автор",
)
UNTAGGED = ("**Ключевая мысль**", "Это важно запомнить.", "- дополнительный пункт", "```", "Итак, перейдём дальше.",
            "## Подробнее")

_TAG_RE = re.compile(r"^#([A-Za-z_]+)\s*:\s?(.*)$")
_RESPONSE_RE = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.DOTALL)


def line_tag(line: str) -> tuple[str | None, str]:
    match = _TAG_RE.match(line.strip())
    return (match.group(1).upper(), match.group(2)) if match else (None, line)


# --------------------------------------------------------------------------- текст по слайдам

@dataclass
class Doc:
    """Текст DSL: строки до первого ``#SLIDE`` и слайды (первая строка слайда — ``#SLIDE``); пустые строки
    отбрасываются, при сборке слайды разделяются пустой строкой — для парсера это ничего не меняет."""

    head: list[str]
    slides: list[list[str]]

    @classmethod
    def parse(cls, text: str) -> "Doc":
        head: list[str] = []
        slides: list[list[str]] = []
        for line in text.replace("\r\n", "\n").split("\n"):
            if not line.strip():
                continue
            if line.strip().upper() == "#SLIDE":
                slides.append([line.strip()])
            elif slides:
                slides[-1].append(line)
            else:
                head.append(line)
        return cls(head, slides)

    def text(self) -> str:
        parts = ["\n".join(self.head)] if self.head else []
        parts += ["\n".join(s) for s in self.slides]
        return "\n\n".join(parts) + "\n"

    def copy(self) -> "Doc":
        return Doc(list(self.head), [list(s) for s in self.slides])

    def __len__(self) -> int:
        return len(self.slides)

    def slide(self, number: int) -> list[str]:
        return self.slides[number - 1]

    def title(self, number: int) -> str:
        return next((v for t, v in map(line_tag, self.slide(number)) if t == "TITLE"), "")

    def values(self, number: int, *tags: str) -> list[str]:
        return [v for t, v in map(line_tag, self.slide(number)) if t in tags]

    def count(self, number: int, *tags: str) -> int:
        return len(self.values(number, *tags))

    def remove(self, number: int, *tags: str) -> None:
        self.slides[number - 1] = [l for l in self.slide(number) if line_tag(l)[0] not in tags]

    def add(self, number: int, *lines: str) -> None:
        self.slides[number - 1].extend(lines)

    def replace_textual(self, number: int, new_lines: list[str]) -> None:
        """Строки ``#TEXT``/``#BULLET`` слайда заменяются на ``new_lines`` (на месте первой из них)."""
        slide, out, done = self.slide(number), [], False
        for line in slide:
            if line_tag(line)[0] in TEXTUAL:
                if not done:
                    out.extend(new_lines)
                    done = True
                continue
            out.append(line)
        if not done:
            out.extend(new_lines)
        self.slides[number - 1] = out


def paraphrase(doc: Doc, number: int, texts: list[str]) -> Doc:
    """Строки ``#TEXT``/``#BULLET`` слайда по порядку заменяются текстами перефразировки (тег сохраняется)."""
    result = doc.copy()
    slide, k = result.slide(number), 0
    for i, line in enumerate(slide):
        tag, _ = line_tag(line)
        if tag in TEXTUAL:
            if k >= len(texts):
                raise ValueError(f"на слайде {number} строк текста больше, чем в перефразировке")
            slide[i] = f"#{tag}: {texts[k]}"
            k += 1
    if k != len(texts):
        raise ValueError(f"на слайде {number} строк текста {k}, в перефразировке {len(texts)}")
    return result


# --------------------------------------------------------------------------- контекст эпизода

def episode_context(text: str, search: bool) -> dict:
    """Вызовы (как ``Episode.calls``), выдача инструментов и допустимые ``#SOURCE`` из полного текста эпизода
    (``runs/.../<mode>_<id>_<s>.txt``). Источники — как у ``WikiTools.valid_source``: статья из выдачи поиска,
    раздел открытой статьи или прочитанный раздел."""
    responses = _RESPONSE_RE.findall(text)
    _, calls = parse_tool_calls(_RESPONSE_RE.sub("", text))
    ledger = _Ledger()
    records = []
    for k, call in enumerate(calls):
        output = responses[k] if k < len(responses) else ""
        record = {"name": call.name, "argument": call.argument, "error": call.error, "chars": len(output)}
        if not search:
            record["disabled"] = True
        elif not call.error and output:
            ledger.check(call, output, f"вызов {k + 1}", [], [])
        records.append(record)
    sources: dict[str, dict] = {}
    for doc in ledger.docs:
        url = f"https://{doc.lang}.wikipedia.org/wiki/{doc.title.replace(' ', '_')}"
        for index in [None, *sorted(set(doc.toc) | doc.read)]:
            ref = f"d{doc.number}" if index is None else f"d{doc.number}#{index}"
            if ledger.source_text(ref) is None:
                continue
            section = "" if index is None else doc.toc.get(index, "")
            sources[ref] = {"title": doc.title, "lang": doc.lang, "section": section,
                            "url": url + (f"#{section.replace(' ', '_')}" if section else "")}
    return {"calls": records, "tool_text": "\n".join(responses) if search else None, "sources": sources}


# --------------------------------------------------------------------------- пары

@dataclass
class Pair:
    id: str
    family: str
    kind: str
    base: str  # id варианта-основы («original» — исходник)
    variant: str
    target: str | None  # компонент RewardResult.components()
    expect_target: str | None
    expect_total: str | None
    strength: int = 1
    slide: int | None = None
    thesis: int | None = None
    diagnostic: bool = False  # фиксирует ограничение награды; в выбор конфигурации не входит
    checks: dict = field(default_factory=dict)  # предусловия по отчёту сборщика
    changed: dict = field(default_factory=dict)  # строки, убранные и добавленные относительно основы


@dataclass
class Generated:
    variants: dict[str, dict] = field(default_factory=dict)  # id → {"text", "context"} (context — поправки)
    pairs: list[Pair] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # искажения, у которых не выполнены предусловия


def _changed(base: str, variant: str) -> dict:
    removed, added = [], []
    for line in difflib.ndiff(base.splitlines(), variant.splitlines()):
        if line.startswith("- ") and line[2:].strip():
            removed.append(line[2:][:200])
        elif line.startswith("+ ") and line[2:].strip():
            added.append(line[2:][:200])
    return {"removed": removed[:20], "added": added[:20]}


class _Builder:
    def __init__(self, task_id: str, original: Doc, annotation: dict, points: list[KeyPoint], context: dict,
                 donor: Doc):
        self.task, self.doc, self.ann, self.points, self.ctx, self.donor = task_id, original, annotation, points, \
            context, donor
        self.out = Generated()
        self.out.variants["original"] = {"text": original.text(), "context": {}}
        self.n = len(original)
        self.conclusion_point = context["conclusion"] and len(points) > 1
        self.has_conclusion_slide = self.n > 2 and bool(_CONCLUSION_RE.match(original.title(self.n)))
        last_content = self.n - 1 if self.has_conclusion_slide else self.n
        self.content = list(range(2, last_content + 1))
        # N в награде — по числу слайдов и требованию заключения (rewards.score)
        self.N = max(1, self.n - 1 - (1 if self.conclusion_point and self.n > 2 else 0)) if self.n > 1 else 1

    # --- основа

    def variant(self, name: str, doc: Doc | None = None, context: dict | None = None) -> str:
        key = name
        text = doc.text() if doc is not None else self.out.variants["original"]["text"]
        self.out.variants[key] = {"text": text, "context": context or {}}
        return key

    def pair(self, family: str, kind: str, variant: str, target: str | None, expect_target: str | None,
             expect_total: str | None, base: str = "original", suffix: str = "", **extra) -> None:
        pair_id = f"{self.task}/{kind}" + (f"/{suffix}" if suffix else "")
        changed = _changed(self.out.variants[base]["text"], self.out.variants[variant]["text"])
        self.out.pairs.append(Pair(pair_id, family, kind, base, variant, target, expect_target, expect_total,
                                   changed=changed, **extra))

    def skip(self, kind: str, reason: str) -> None:
        self.out.skipped.append(f"{self.task}/{kind}: {reason}")

    # --- тезисы

    def thesis_slides(self, i: int) -> list[int]:
        return self.ann["theses"][i]

    def exclusive_theses(self) -> list[int]:
        """Тезисы, раскрытые ровно на одном содержательном слайде, которого нет у других тезисов."""
        result = []
        last = len(self.points) - 1 if self.conclusion_point else None
        for i, slides in enumerate(self.ann["theses"]):
            if i == last or len(slides) != 1:
                continue
            s = slides[0]
            if s not in self.content or any(s in other for j, other in enumerate(self.ann["theses"]) if j != i):
                continue
            result.append(i)
        return result

    def spread(self, items: list[int]) -> list[int]:
        if len(items) <= THESES_PER_KIND:
            return items
        return sorted({items[0], items[len(items) // 2], items[-1]})

    def donor_textual(self) -> tuple[str, list[str]]:
        """Заголовок и текст содержательного слайда чужой темы (средний)."""
        n = len(self.donor)
        number = max(2, n // 2)
        lines = [l for l in self.donor.slide(number) if line_tag(l)[0] in TEXTUAL]
        return self.donor.title(number), lines

    def free_large(self, doc: Doc, number: int) -> int:
        return MAX_LARGE_VISUALS - doc.count(number, *LARGE_VISUALS)

    def place(self, doc: Doc, line: str, count: int, kinds=LARGE_VISUALS, per_slide=MAX_LARGE_VISUALS,
              slides: list[int] | None = None) -> bool:
        """Добавляет ``count`` строк ``line`` на содержательные слайды со свободным местом; False — места нет."""
        for number in slides or self.content:
            while count and doc.count(number, *kinds) < per_slide:
                doc.add(number, line)
                count -= 1
        return count == 0

    # --- семейства

    def coverage(self) -> None:
        theses = self.spread(self.exclusive_theses())
        if not theses:
            self.skip("coverage", "нет тезиса, раскрытого на одном отдельном слайде")
        donor_title, donor_lines = self.donor_textual()
        for i in theses:
            s = self.thesis_slides(i)[0]
            point = self.points[i]
            extra = {"slide": s, "thesis": i}
            doc = self.doc.copy()
            del doc.slides[s - 1]
            self.pair("coverage", "drop_explanation", self.variant(f"drop_explanation/T{i}", doc), "coverage", DOWN,
                      DOWN, suffix=f"T{i}", **extra)
            doc = self.doc.copy()
            doc.remove(s, *BODY)
            self.pair("coverage", "empty_body", self.variant(f"empty_body/T{i}", doc), "coverage", DOWN, DOWN,
                      suffix=f"T{i}", **extra)
            doc = self.doc.copy()
            doc.remove(s, *BODY, "TITLE")
            doc.slides[s - 1].insert(1, f"#TITLE: {point.title}")
            self.pair("coverage", "title_only_thesis", self.variant(f"title_only_thesis/T{i}", doc), "coverage",
                      DOWN, DOWN, suffix=f"T{i}", **extra)
            doc = self.doc.copy()
            doc.remove(s, *BODY)
            doc.add(s, *donor_lines)
            self.pair("coverage", "foreign_body", self.variant(f"foreign_body/T{i}", doc), "coverage", DOWN, DOWN,
                      suffix=f"T{i}", **extra)
            doc = self.doc.copy()
            doc.replace_textual(s, ["#TEXT: " + ", ".join(dict.fromkeys(point.key))])
            self.pair("coverage", "keys_only", self.variant(f"keys_only/T{i}", doc), "coverage", DOWN, DOWN,
                      suffix=f"T{i}", **extra)
            texts = self.doc.values(s, *TEXTUAL)
            doc = self.doc.copy()
            doc.replace_textual(s, [f"#NOTES: {t}" for t in texts])
            self.pair("coverage", "hide_in_notes", self.variant(f"hide_in_notes/T{i}", doc), "coverage", DOWN,
                      DOWN, suffix=f"T{i}", **extra)
            doc = self.doc.copy()
            doc.replace_textual(s, [])
            doc.head.extend(f"#PLAN: {t}" for t in texts)
            self.pair("coverage", "hide_in_plan", self.variant(f"hide_in_plan/T{i}", doc), "coverage", DOWN, DOWN,
                      suffix=f"T{i}", **extra)
            if self.ctx["images_allowed"] and self.free_large(self.doc, s) > 0:
                doc = self.doc.copy()
                doc.replace_textual(s, ["#IMAGE: " + " ".join(texts)[:990]])
                self.pair("coverage", "hide_in_image", self.variant(f"hide_in_image/T{i}", doc), "coverage", DOWN,
                          DOWN, suffix=f"T{i}", **extra)
            else:
                self.skip(f"hide_in_image/T{i}", "картинки запрещены или на слайде нет места для крупного элемента")
            if len(texts) >= 2:  # объяснение на двух слайдах: ключи ищутся на двух лучших слайдах
                doc = self.doc.copy()
                half = (len(texts) + 1) // 2
                textual = [l for l in doc.slide(s) if line_tag(l)[0] in TEXTUAL]
                doc.slides[s - 1] = [l for l in doc.slide(s) if l not in textual[half:]]
                doc.slides.insert(s, ["#SLIDE", f"#TITLE: {self.doc.title(s)} (продолжение)", *textual[half:]])
                self.pair("coverage", "split_explanation", self.variant(f"split_explanation/T{i}", doc), "keys",
                          SAME, None, suffix=f"T{i}", **extra)
        if self.has_conclusion_slide and self.conclusion_point:
            last = len(self.points) - 1
            doc = self.doc.copy()
            del doc.slides[-1]
            self.pair("coverage", "no_conclusion", self.variant("no_conclusion", doc), "coverage", DOWN, DOWN,
                      slide=self.n, thesis=last)
            doc = self.doc.copy()
            doc.remove(self.n, *BODY)
            doc.add(self.n, "#TEXT: Спасибо за внимание!")
            self.pair("coverage", "conclusion_empty", self.variant("conclusion_empty", doc), "coverage", DOWN, DOWN,
                      slide=self.n, thesis=last)
            doc = self.doc.copy()
            doc.slides[-1] = list(doc.slides[-2])
            self.pair("coverage", "conclusion_replaced", self.variant("conclusion_replaced", doc), "coverage", DOWN,
                      DOWN, slide=self.n, thesis=last)
        else:
            self.skip("conclusion", "в исходнике нет слайда-заключения или тема его не требует")
        if len(self.content) >= 3:
            doc = Doc(list(self.doc.head), [list(self.doc.slide(1))])
            merged = ["#SLIDE", "#TITLE: Основные положения"]
            for s in self.content:  # по строке на слайд: иначе сверх лимита элементов на слайде
                merged.append("#TEXT: " + " ".join(self.doc.values(s, *TEXTUAL))[:990])
            doc.slides.append(merged)
            if self.has_conclusion_slide:
                doc.slides.append(list(self.doc.slide(self.n)))
            self.pair("coverage", "overloaded_slide", self.variant("overloaded_slide", doc), "coverage", DOWN, DOWN)

    def keys(self) -> None:
        for k, edit in enumerate(self.ann["numbers"]):
            doc = self._edit(edit)
            self.pair("keys", "wrong_number", self.variant(f"wrong_number/{k}", doc), "keys", DOWN, DOWN,
                      suffix=str(k), slide=edit["slide"])
        for k, edit in enumerate(self.ann["forms"]):
            doc = self._edit(edit)
            self.pair("keys", "key_wordform", self.variant(f"key_wordform/{k}", doc), "keys", SAME, SAME,
                      suffix=str(k), slide=edit["slide"])
        if not self.ann["numbers"]:
            self.skip("wrong_number", "в ключах нет чисел")

    def _edit(self, edit: dict) -> Doc:
        doc = self.doc.copy()
        slide = doc.slide(edit["slide"])
        hits = [i for i, l in enumerate(slide) if edit["old"] in l]
        if len(hits) != 1 or slide[hits[0]].count(edit["old"]) != 1:
            raise ValueError(f"{self.task}: «{edit['old']}» на слайде {edit['slide']} встречается не один раз")
        slide[hits[0]] = slide[hits[0]].replace(edit["old"], edit["new"])
        return doc

    def length(self) -> None:
        middle = self.content[len(self.content) // 2]
        plans = {1: [middle], 2: self.content[::2], 3: list(self.content)}
        for strength, slides in plans.items():
            doc = self.doc.copy()
            for s in sorted(slides, reverse=True):
                doc.slides.insert(s, list(doc.slide(s)))
            # дубли всех слайдов — за верхней границей диапазона; один-два дубля могут остаться внутри него
            expect = DOWN if strength == 3 or self.ctx["slide_range"] is not None else None
            self.pair("length", "duplicate_slides", self.variant(f"duplicate_slides/{strength}", doc), "length", expect,
                      DOWN if strength == 3 else NOT_UP, suffix=str(strength), strength=strength)
        donor_slides = [list(self.donor.slide(s)) for s in (2, 3) if s <= len(self.donor)]
        for strength in (1, 2):
            doc = self.doc.copy()
            at = self.content[-1]
            for slide in donor_slides[:strength]:
                doc.slides.insert(at, [slide[0]] + [l for l in slide[1:] if line_tag(l)[0] in ("TITLE", *TEXTUAL)])
                at += 1
            self.pair("length", "foreign_slides", self.variant(f"foreign_slides/{strength}", doc), "length", None,
                      NOT_UP, suffix=str(strength), strength=strength)

    def visual(self) -> None:
        doc_visuals = sum(self.doc.count(s, *VISUAL) for s in range(1, self.n + 1))
        if doc_visuals:
            doc = self.doc.copy()
            for s in range(1, self.n + 1):
                doc.remove(s, *VISUAL)
            self.pair("visual", "no_visuals", self.variant("no_visuals", doc), "visual", DOWN, DOWN)
        else:
            self.skip("no_visuals", "в исходнике нет визуала")
        kinds = [("image", IMAGE_LINE, "IMAGE")] if self.ctx["images_allowed"] else []
        kinds.append(("chart", CHART_LINE, "CHART"))
        for name, line, kind in kinds:
            have = sum(self.doc.count(s, kind) for s in range(1, self.n + 1))
            limit = type_limit(kind, self.N)
            self._limit_chain(name, line, max(0, limit - have), LARGE_VISUALS, MAX_LARGE_VISUALS, (1, 3), DOWN)
        have = sum(self.doc.count(s, "FORMULA") for s in range(1, self.n + 1))
        self._limit_chain("formula", FORMULA_LINE, max(0, FORMULA_TOTAL_PER_SLIDE * self.N - have), ("FORMULA",),
                          MAX_FORMULAS, (3,), DOWN)
        # сверх ограничения слайда: основа — слайд, заполненный до предела, вариант — на один элемент больше
        for name, line, kinds_, limit in (("large", IMAGE_LINE if self.ctx["images_allowed"] else CHART_LINE,
                                           LARGE_VISUALS, MAX_LARGE_VISUALS),
                                          ("formula", FORMULA_LINE, ("FORMULA",), MAX_FORMULAS)):
            s = self.content[0]
            base = self.doc.copy()
            self.place(base, line, limit - base.count(s, *kinds_), kinds_, limit, [s])
            base_id = self.variant(f"slide_full/{name}", base)
            doc = base.copy()
            doc.add(s, line)
            self.pair("visual", "slide_overflow", self.variant(f"slide_overflow/{name}", doc), "slide_overflow", DOWN,
                      DOWN, base=base_id, suffix=name, slide=s, checks={"status": "overflow"})
        charts = [(s, i) for s in range(1, self.n + 1) for i, l in enumerate(self.doc.slide(s))
                  if line_tag(l)[0] == "CHART"]
        if charts:
            s, i = charts[0]
            doc = self.doc.copy()
            doc.slide(s)[i] = re.sub(r"(y\s*=[^;]*)", r"\1, 7", doc.slide(s)[i], count=1)  # чисел больше, чем подписей
            self.pair("visual", "broken_chart", self.variant("broken_chart", doc), "format", DOWN, DOWN, slide=s,
                      checks={"parse_errors": True})
        else:
            self.skip("broken_chart", "в исходнике нет #CHART")
        for kind, line, target, expect, status in (
                ("undefined_plot", "#PLOT: y = log(x); x=-10..1", "render_errors", DOWN, "undefined_plot"),
                ("flat_plot", "#PLOT: y = 0*x + 3; x=0..10", "visual", NOT_UP, "flat_plot")):
            doc = self.doc.copy()
            if not self.place(doc, line, 1):
                self.skip(kind, "нет места для крупного элемента")
                continue
            self.pair("visual", kind, self.variant(kind, doc), target, expect, expect, checks={"status": status})

    def _limit_chain(self, name: str, line: str, to_limit: int, kinds, per_slide: int, extras, expect_big) -> None:
        base = self.doc.copy()
        if not self.place(base, line, to_limit, kinds, per_slide):
            self.skip(f"{name}_excess", "не хватает места на слайдах, чтобы дойти до предела")
            return
        base_id = "original"
        if to_limit:
            base_id = self.variant(f"{name}_at_limit", base)
            self.pair("visual", "irrelevant_visual", base_id, "visual", None, NOT_UP, suffix=name, strength=to_limit)
        for extra in extras:
            doc = base.copy()
            if not self.place(doc, line, extra, kinds, per_slide):
                self.skip(f"{name}_excess/+{extra}", "не хватает места на слайдах")
                continue
            expect = NOT_UP if extra == 1 else expect_big
            self.pair("visual", f"{name}_excess", self.variant(f"{name}_excess/{extra}", doc), "visual", expect,
                      expect, base=base_id, suffix=f"+{extra}", strength=extra)

    def grounding(self) -> None:
        """Цитаты, картинки при запрете, источники."""
        if self.ctx["search"] and self.ctx["tool_text"] is not None:
            source = _normalize_quote(self.ctx["tool_text"])
            quotes = [q for q in FAKE_QUOTES if not quote_found(q.rsplit(" — ", 1)[0], source)]
            ids = {}
            for strength in (1, 3):
                doc = self.doc.copy()
                for s, quote in zip(self.content, quotes[:strength]):
                    doc.add(s, f"#QUOTE: {quote}")
                ids[strength] = self.variant(f"fake_quote/{strength}", doc)
                self.pair("grounding", "fake_quote", ids[strength], "fake_quotes", DOWN, DOWN, suffix=str(strength),
                          strength=strength)
            self.pair("chain", "fake_quote_chain", ids[3], "fake_quotes", DOWN, DOWN, base=ids[1], strength=3)
        else:
            self.skip("fake_quote", "режим без поиска: цитаты ни плюс, ни минус")
        forbid = {"images_allowed": False}
        base = self.doc.copy()
        for s in range(1, self.n + 1):
            base.remove(s, "IMAGE")
        doc = self.doc.copy()
        if not any(self.doc.count(s, "IMAGE") for s in range(1, self.n + 1)):
            self.place(doc, IMAGE_LINE, 1)
        base_id = self.variant("no_images", base, forbid)
        self.pair("grounding", "forbidden_image", self.variant("forbidden_image", doc, forbid), "forbidden_images",
                  DOWN, DOWN, base=base_id)
        ids = {}
        for strength in (1, 3):
            doc = self.doc.copy()
            for s in self.content[:strength]:
                doc.add(s, "#SOURCE: d99")
            ids[strength] = self.variant(f"bad_source/{strength}", doc)
            self.pair("grounding", "bad_source", ids[strength], "bad_sources", DOWN, DOWN, suffix=str(strength),
                      strength=strength)
        self.pair("chain", "bad_source_chain", ids[3], "bad_sources", DOWN, DOWN, base=ids[1], strength=3)

    def format(self) -> None:
        ids = {}
        for strength in (1, 3, 6):
            doc = self.doc.copy()
            for k in range(strength):
                doc.add(self.content[k % len(self.content)], UNTAGGED[k])
            ids[strength] = self.variant(f"untagged/{strength}", doc)
            self.pair("format", "untagged_lines", ids[strength], "format", DOWN, DOWN, suffix=str(strength),
                      strength=strength, checks={"parse_errors": True})
        self.pair("chain", "untagged_chain", ids[3], "format", DOWN, DOWN, base=ids[1], suffix="1-3", strength=3)
        self.pair("chain", "untagged_chain", ids[6], "format", DOWN, DOWN, base=ids[3], suffix="3-6", strength=6)
        # разбавление: к битым строкам добавлены пустые корректные — награда не должна вырасти
        self.pair("format", "dilution", self.variant("dilution", self._dilute(ids[3])), "format", NOT_UP, NOT_UP,
                  base=ids[3])
        s = self.content[0]
        doc = self.doc.copy()
        slide = doc.slide(s)
        k = next((i for i, l in enumerate(slide) if line_tag(l)[0] == "TEXT"), None)
        if k is not None:
            slide[k] = slide[k].replace("#TEXT:", "#TXET:", 1)
            self.pair("format", "tag_typo", self.variant("tag_typo", doc), "format", DOWN, DOWN, slide=s,
                      checks={"parse_errors": True})
        doc = self.doc.copy()
        doc.add(s, "#PLOT: x^2 + 1")
        self.pair("format", "bad_syntax", self.variant("bad_syntax", doc), "format", DOWN, DOWN, slide=s,
                  checks={"parse_errors": True})

    def _dilute(self, base_id: str) -> Doc:
        doc = Doc.parse(self.out.variants[base_id]["text"])
        for k in range(10):
            doc.add(self.content[k % len(self.content)], "#NOTES: —")
        return doc

    def text_overflow(self) -> None:
        theses = self.exclusive_theses() or [None]
        i = theses[0]
        s = self.thesis_slides(i)[0] if i is not None else self.content[0]
        text = " ".join(self.doc.values(s, *TEXTUAL)) or self.doc.title(s)
        long = (" ".join([text] * 6))[:990]
        doc = self.doc.copy()
        doc.add(s, f"#TEXT: {long}")
        self.pair("text", "text_overflow", self.variant("text_overflow", doc), "text_overflow", DOWN, DOWN, slide=s,
                  thesis=i, checks={"status": "text_overflow"})

    def tools(self) -> None:
        calls = self.ctx["calls"]
        if self.ctx["search"]:
            done = [c for c in calls if not c["error"]]
            if done:
                repeat = calls + [dict(done[0])]
                self.pair("tools", "repeat_call", self.variant("repeat_call", None, {"calls": repeat}), "tools", DOWN,
                          DOWN)
            else:
                self.skip("repeat_call", "в эпизоде нет вызовов")
            ids = {}
            for strength in (1, 3):
                extra = [{"name": "search", "argument": f"дополнительный запрос {w}", "error": "", "chars": 600}
                         for w in ("первый", "второй", "третий")[:strength]]
                ids[strength] = self.variant(f"extra_calls/{strength}", None, {"calls": calls + extra})
                self.pair("tools", "extra_calls", ids[strength], "tools", DOWN, DOWN, suffix=str(strength),
                          strength=strength)
            self.pair("chain", "extra_calls_chain", ids[3], "tools", DOWN, DOWN, base=ids[1], strength=3)
        else:
            extra = {"name": "search", "argument": self.points[0].title, "error": "", "chars": len(NO_SEARCH_MESSAGE),
                     "disabled": True}
            self.pair("tools", "call_in_plain", self.variant("call_in_plain", None, {"calls": calls + [extra]}),
                      "tools", DOWN, DOWN)

    def invariance(self) -> None:
        both = self.doc
        for item in self.ann["paraphrases"]:
            doc = paraphrase(self.doc, item["slide"], item["texts"])
            both = paraphrase(both, item["slide"], item["texts"])
            self.pair("invariance", "paraphrase", self.variant(f"paraphrase/{item['slide']}", doc), "coverage", SAME,
                      SAME, suffix=str(item["slide"]), slide=item["slide"])
        if len(self.ann["paraphrases"]) > 1:
            self.pair("invariance", "paraphrase", self.variant("paraphrase/all", both), "coverage", SAME, SAME,
                      suffix="all", strength=2)
        for a, b in self.ann["independent"]:
            if a not in self.content or b not in self.content:
                self.skip(f"reorder/{a}-{b}", "переставлять можно только содержательные слайды")
                continue
            doc = self.doc.copy()
            doc.slides[a - 1], doc.slides[b - 1] = doc.slides[b - 1], doc.slides[a - 1]
            self.pair("invariance", "reorder", self.variant(f"reorder/{a}-{b}", doc), "coverage", SAME, SAME,
                      suffix=f"{a}-{b}")
        if not self.ann["independent"]:
            self.skip("reorder", "нет независимых слайдов")

    def diagnostics(self) -> None:
        for k, edit in enumerate(self.ann["facts"]):
            doc = self._edit(edit)
            self.pair("diagnostic", edit["kind"], self.variant(f"fact/{k}", doc), "coverage", DOWN, DOWN,
                      suffix=str(k), slide=edit["slide"], diagnostic=True)
        theses = self.exclusive_theses()
        if theses:  # объяснение после 512 токенов эмбеддера: впереди длинный чужой текст
            i = theses[0]
            s = self.thesis_slides(i)[0]
            _, donor_lines = self.donor_textual()
            filler = " ".join(line_tag(l)[1] for l in donor_lines)
            filler = (" ".join([filler] * 8))[:990]
            doc = self.doc.copy()
            slide = doc.slide(s)
            k = next(j for j, l in enumerate(slide) if line_tag(l)[0] in TEXTUAL)
            slide[k:k] = [f"#TEXT: {filler}", f"#TEXT: {filler}"]
            self.pair("diagnostic", "after_512_tokens", self.variant("after_512_tokens", doc), "coverage", None, None,
                      slide=s, thesis=i, diagnostic=True)

    def run(self) -> Generated:
        for step in (self.coverage, self.keys, self.length, self.visual, self.grounding, self.format,
                     self.text_overflow, self.tools, self.invariance, self.diagnostics):
            step()
        if self.ctx["slide_range"] is not None:  # число слайдов задано: больше заданного — штраф длины
            doc = self.doc.copy()
            for s in self.content[:2]:
                doc.slides.insert(s, list(doc.slide(s)))
            self.pair("length", "over_count", self.variant("over_count", doc), "length", DOWN, DOWN)
        return self.out


def generate(task_id: str, original: str, annotation: dict, points: list[KeyPoint], context: dict,
             donor: str) -> Generated:
    """Все варианты и пары одного исходника; ``context`` — контекст эпизода и задания (см. ``task_context``),
    ``donor`` — исходник другой темы (чужое содержание)."""
    return _Builder(task_id, Doc.parse(original), annotation, points, context, Doc.parse(donor)).run()


def task_context(task, episode_text: str, search: bool) -> dict:
    """Контекст оценки: задание (число слайдов, заключение, картинки, режим) и эпизод (вызовы, выдача, источники)."""
    context = {"search": search, "images_allowed": task.request.images,
               "slide_range": list(task.request.slide_range()) if task.request.slide_range() else None,
               "conclusion": expected_theses(task)[2]}
    context.update(episode_context(episode_text, search))
    return context


# --------------------------------------------------------------------------- пары с конфликтом достоинств

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[«\"(A-ZА-ЯЁ0-9$])")


def _sentences(line: str) -> int:
    return len(_SENTENCE_RE.split(line_tag(line)[1]))


def _halves(lines: list[str]) -> tuple[list[str], list[str]] | None:
    """Строки текста слайда пополам: по строкам, а одну строку — по предложениям (тег сохраняется);
    None — делить нечего. Ровность разреза проверяет ``_Conflicts.split``."""
    if len(lines) >= 2:
        half = (len(lines) + 1) // 2
        return lines[:half], lines[half:]
    if not lines:
        return None
    tag, value = line_tag(lines[0])
    sentences = _SENTENCE_RE.split(value)
    if len(sentences) < 2:
        return None
    half = (len(sentences) + 1) // 2
    return [f"#{tag}: " + " ".join(sentences[:half])], [f"#{tag}: " + " ".join(sentences[half:])]


def _enough(text_lines: list[str], visuals: list[str]) -> bool:
    """Половина слайда не выглядит обрубком: два предложения или две строки — или предложение и визуальный элемент."""
    return len(text_lines) >= 2 or sum(map(_sentences, text_lines)) >= 2 or bool(text_lines and visuals)


class _Conflicts(_Builder):
    """Пары, где у каждой презентации своё достоинство; какая лучше — решает судья вслепую (не награда).
    По ним выбираются λ (длина) и соотношение покрытия и ключей; изолированные искажения задают только знак."""

    even = False  # разрез, по которому размечены пары (по строкам и предложениям, визуал остаётся на первом слайде);
    # True — ровный разрез из проверки судьи (data/perturbation/conflicts/rounds/recut_check, решение 01.11.2026)

    def split(self, share: float) -> Doc:
        """Объяснения на доле ``share`` содержательных слайдов (через один, если половина) разбиты на два слайда;
        при ровном разрезе (``even``) слайд, который ровно не делится, остаётся целым."""
        slides = self.content if share >= 1 else self.content[::2]
        doc, inserted = self.doc.copy(), 0
        for s in self.content:
            number = s + inserted
            if s not in slides:
                continue
            textual = [l for l in doc.slide(number) if line_tag(l)[0] in TEXTUAL]
            halves = _halves(textual)
            if halves is None:
                continue
            first, second = halves
            visuals = [l for l in doc.slide(number) if line_tag(l)[0] in BODY and l not in textual]
            moved = visuals[len(visuals) // 2:] if self.even else []  # ровный разрез: визуал поровну
            if self.even and not (_enough(first, visuals[:len(visuals) // 2]) and _enough(second, moved)):
                continue  # ровный разрез: половина без двух предложений и без визуала не делится
            kept = [l for l in doc.slide(number) if l not in textual and l not in moved]
            doc.slides[number - 1] = kept[:2] + first + kept[2:]
            doc.slides.insert(number, ["#SLIDE", f"#TITLE: {self.doc.title(s)} (продолжение)", *second, *moved])
            inserted += 1
        return doc

    def drop(self, count: int, avoid: set[int] = frozenset()) -> tuple[Doc, list[int]]:
        """Без ``count`` слайдов-тезисов (тезисы на отдельном слайде, из середины; иначе — содержательные слайды)."""
        theses = [self.thesis_slides(i)[0] for i in self.exclusive_theses()
                  if self.thesis_slides(i)[0] not in avoid]
        pool = theses if len(theses) >= count else [x for x in self.content if x not in avoid]
        middle = len(pool) // 2
        chosen = sorted(pool[max(0, middle - count // 2):][:count])
        doc = self.doc.copy()
        for s in sorted(chosen, reverse=True):
            del doc.slides[s - 1]
        return doc, chosen

    def keys_of_slide(self, s: int) -> list[str]:
        return list(dict.fromkeys(k for i, slides in enumerate(self.ann["theses"]) if s in slides
                                  for k in self.points[i].key))

    def build(self) -> list[dict]:
        """Пять пар на тему: растяжение (половина или все объяснения на двух слайдах) против потери одного или
        двух тезисов и повторы против отсутствия заключения (λ); объяснение с ошибкой в числе против списка
        ключей (вес ключей)."""
        result = []

        def add(kind: str, decides: str, a: Doc, b: Doc, note: str) -> None:
            result.append({"id": f"{self.task}/{kind}", "task": self.task, "kind": kind, "decides": decides,
                           "a": a.text(), "b": b.text(), "note": note})

        one, lost_one = self.drop(1)
        two, lost_two = self.drop(2)
        add("stretch_half_vs_missing1", "length", self.split(0.5), one,
            f"a — половина объяснений на двух слайдах; b — без слайда {lost_one}")
        add("stretch_all_vs_missing1", "length", self.split(1), one,
            f"a — все объяснения на двух слайдах; b — без слайда {lost_one}")
        add("stretch_all_vs_missing2", "length", self.split(1), two,
            f"a — все объяснения на двух слайдах; b — без слайдов {lost_two}")
        longer = self.doc.copy()
        for x in sorted({self.content[0], self.content[len(self.content) // 2]}, reverse=True):
            longer.slides.insert(x, list(longer.slide(x)))
        if self.has_conclusion_slide:
            shorter = self.doc.copy()
            del shorter.slides[-1]
            note = "a — два слайда повторены; b — без заключения"
        else:
            shorter, other = self.drop(1, set(lost_one))
            note = f"a — два слайда повторены; b — без слайда {other}"
        add("repeat_vs_missing", "length", longer, shorter, note)
        if self.ann["numbers"]:
            edit = self.ann["numbers"][0]
            s, a = edit["slide"], self._edit(edit)
            note = f"a — объяснение с неверным числом ({edit['old']} → {edit['new']}); b — голый список ключей"
        else:
            theses = self.exclusive_theses()
            s = self.thesis_slides(theses[len(theses) // 2])[0] if theses else self.content[len(self.content) // 2]
            a = self.doc.copy()
            first = _SENTENCE_RE.split(" ".join(self.doc.values(s, *TEXTUAL)))[0]
            a.replace_textual(s, [f"#TEXT: {first}"])
            note = "a — от объяснения осталось первое предложение; b — голый список ключей"
        b = self.doc.copy()
        b.replace_textual(s, ["#TEXT: " + ", ".join(self.keys_of_slide(s))])
        add("text_vs_keys", "keys", a, b, note + f" (слайд {s})")
        return result


def conflict_pairs(task_id: str, original: str, annotation: dict, points: list[KeyPoint], context: dict,
                   even: bool = False) -> list[dict]:
    builder = _Conflicts(task_id, Doc.parse(original), annotation, points, context, Doc.parse(original))
    builder.even = even
    return builder.build()


_VIEW = {"TEXT": "{}", "BULLET": "• {}", "QUOTE": "Цитата: {}", "FORMULA": "Формула: {}", "CHART": "Диаграмма: {}",
         "PLOT": "График функции: {}", "DIAGRAM": "Схема: {}", "TIMELINE": "Хронология: {}", "IMAGE": "[Картинка: {}]",
         "NOTES": "Заметки докладчика: {}"}


def readable(text: str, sources: dict) -> str:
    """Презентация для судьи: слайды, видимые элементы и заметки; источники — названиями статей."""
    out = []
    for k, slide in enumerate(Doc.parse(text).slides, 1):
        lines = slide[1:]
        title = next((v for t, v in map(line_tag, lines) if t == "TITLE"), "(без заголовка)")
        out.append(f"### Слайд {k}. {title}")
        for tag, value in map(line_tag, lines):
            if tag in _VIEW:
                out.append(_VIEW[tag].format(value))
            elif tag == "SOURCE":
                names = [sources.get(r.strip(), {}).get("title", f"{r.strip()} (нет в выдаче)")
                         for r in re.split(r"[,;]", value) if r.strip()]
                out.append("Источник: " + "; ".join(f"Википедия — {n}" for n in names))
        out.append("")
    return "\n".join(out)


RUBRIC = """# Какая презентация лучше

В каждом файле — запрос пользователя и две презентации на него (текстом: слайды, видимые элементы, заметки
докладчика). Если есть раздел «Справка», в нём проверенные числа по теме: они верны. Другие факты проверять
не нужно и искать в интернете не нужно — судите по тому, что видно.

Нужно решить, какую из двух презентаций вы бы отдали пользователю. Смотрите на то, что важно ему:
- насколько полно и верно раскрыта тема запроса (главные положения; числа сверяйте со справкой);
- объяснение, а не перечень слов;
- соответствие запросу: число слайдов, если оно задано; уместная длина, если не задано; есть ли заключение;
- повторы и лишнее;
- уместность визуальных элементов.

Длиннее — не значит лучше. Если презентации действительно равноценны, можно ответить «равны», но старайтесь
выбрать.

Ответ — в файл `answers.json`: список объектов
`{"pair": "07", "better": "1" | "2" | "равны", "confidence": 1 | 2 | 3, "reason": "одно предложение"}`,
по одному на каждый файл, в порядке номеров. Другие файлы и папки не открывайте.
"""


def export_blind(conflicts: list[dict], requests: dict[str, str], sources: dict[str, dict], out_dir: Path,
                 facts: dict[str, list[str]] | None = None, seed: int = 0) -> list[dict]:
    """Файлы для слепого судьи: случайный порядок пар и презентаций внутри пары; рядом — папка ``<out>_reversed``
    с теми же номерами и презентациями в обратном порядке (проверка склонности судьи к позиции).
    Возвращает ключ основного порядка."""
    import random

    rng = random.Random(seed)
    order = list(conflicts)
    rng.shuffle(order)
    key = [{"pair": f"{n:02d}", "id": pair["id"], "first": "a" if rng.random() < 0.5 else "b"}
           for n, pair in enumerate(order, 1)]
    for folder, flip in ((out_dir, False), (out_dir.with_name(out_dir.name + "_reversed"), True)):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "README.md").write_text(RUBRIC, encoding="utf-8")
        for pair, entry in zip(order, key):
            first = entry["first"] if not flip else ("b" if entry["first"] == "a" else "a")
            second = "b" if first == "a" else "a"
            task = pair["task"]
            body = [f"# Пара {entry['pair']}", "", "## Запрос пользователя", "", requests[task], ""]
            if facts and facts.get(task):
                body += ["## Справка: проверенные числа по теме", ""] + [f"- {f}" for f in facts[task]] + [""]
            body += ["## Презентация 1", "", readable(pair[first], sources[task]),
                     "## Презентация 2", "", readable(pair[second], sources[task])]
            (folder / f"{entry['pair']}.md").write_text("\n".join(body), encoding="utf-8")
    return key


# --------------------------------------------------------------------------- запуск

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".", help="корень репозитория")
    parser.add_argument("--out", default="data/perturbation/variants")
    parser.add_argument("--blind", default=None,
                        help="папка для слепого судьи (вне репозитория): пары с конфликтом достоинств")
    args = parser.parse_args(argv)
    root = Path(args.root)
    base = root / "data" / "perturbation"
    annotations = json.loads((base / "annotations.json").read_text(encoding="utf-8"))
    points = load_key_points(root / "data" / "tasks" / "key_points" / "key_points.json")
    tasks = {t.id: t for t in load_tasks(root / "data" / "tasks" / "tasks.json")}
    ids = sorted(annotations)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    total, skipped, conflicts, requests, sources = 0, [], [], {}, {}
    for k, task_id in enumerate(ids):
        ann = annotations[task_id]
        original = (base / "originals" / f"{task_id}.txt").read_text(encoding="utf-8")
        donor = (base / "originals" / f"{ids[(k + 1) % len(ids)]}.txt").read_text(encoding="utf-8")
        episode = (root / f"{ann['source']}.txt").read_text(encoding="utf-8")
        context = task_context(tasks[task_id], episode, ann["search"])
        result = generate(task_id, original, ann, points[task_id], context, donor)
        record = {"task": task_id, "source": ann["source"], "context": context, "variants": result.variants,
                  "pairs": [asdict(p) for p in result.pairs], "skipped": result.skipped}
        (out / f"{task_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        conflicts += conflict_pairs(task_id, original, ann, points[task_id], context)
        requests[task_id] = tasks[task_id].request.render()
        sources[task_id] = context["sources"]
        total += len(result.pairs)
        skipped += result.skipped
        print(f"{task_id}: вариантов {len(result.variants)}, пар {len(result.pairs)}, пропущено {len(result.skipped)}")
    print(f"всего пар {total}; пропущено искажений {len(skipped)}")
    for line in skipped:
        print("  -", line)
    folder = base / "conflicts"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pairs.json").write_text(json.dumps(conflicts, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"пар с конфликтом достоинств {len(conflicts)}")
    if args.blind:
        facts_path = folder / "facts.json"
        facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else None
        key = export_blind(conflicts, requests, sources, Path(args.blind), facts)
        (folder / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"файлы для судьи — {args.blind}, ключ — {folder / 'key.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
