# Здесь считаем содержательную часть награды: покрытие тезисов, ключи, длину презентации и визуал.
"""Награда за презентацию — содержательная часть (``docs/decisions.md``, раздел 7).

Компоненты:

- покрытие тезисов: ``sim(i, s) = w_t · cos(title_i, T_s) + (1 − w_t) · [(1 − α) · cos(statement_i, C_s)
  + α · лекс(statement_i, C_s)]``; для тезиса берётся лучший слайд, итог — среднее по тезисам. Тезис-заключение
  сравнивается только с последним слайдом; последний слайд в выбор лучшего для остальных тезисов не входит
  (иначе слайд-итог, перечисляющий всё, «покрывал» бы все тезисы сразу);
- ключи: доля ``key`` тезиса, найденных на двух его лучших слайдах (заключение — на последнем), с точностью
  до окончаний; каждый ключ считается один раз; среднее по тезисам;
- длина: штраф за число слайдов вне ожидаемого диапазона (заданного в запросе или выведенного из числа тезисов);
- визуал: каждый тип до ⌈0,5·N⌉ элементов, всего до 2·N; в пределах — +1, сверх — −1, сумма делится на 2·N
  (N — содержательные слайды); засчитывается только нарисованное без ошибок; элемент, не нарисованный из-за
  ограничения слайда (статус ``overflow``: 2 крупных, 3 формулы), — тоже −1; цитата — только найденная в ответах
  инструментов эпизода; ``#IMAGE`` — только при «Картинки: есть», иначе это отдельный штраф;
- ошибки: элемент, который сборщик не смог отрисовать (``render_error``), и график, у которого функция не определена
  на большей части отрезка (``undefined_plot``), — −0,5 в визуальной сумме; неудачный график (``flat_plot``: выглядит
  прямой, хотя функция не линейная, или сжат выбросом) не засчитывается и не штрафуется; ``#SOURCE``
  с id не из этого эпизода (``bad_source``) — штраф за каждый;
- формат: штраф ``1 − parse_rate`` (доля битых строк) и штраф за каждый слайд, где текст не поместился
  в колонку (``text_overflow`` у текста; подписи ленты времени — дефект визуала, не считаются);
- инструменты: при поиске — стоимость каждого вызова и штраф за каждый повтор (тот же инструмент и тот же аргумент;
  у ``search`` — тот же набор знаменательных слов в начальной форме, у ``open``/``read`` — тот же id); без поиска —
  штраф за каждый вызов.

Текст слайда для сравнения — только видимое: заголовок; ``#TEXT``, ``#BULLET``, ``#QUOTE``, подписи ``#DIAGRAM``,
``#TIMELINE`` и ``#CHART``. Не идут ``#NOTES``, ``#IMAGE``, ``#SOURCE``, ``#PLAN``, ``#FORMULA`` и элементы, которые
сборщик не нарисовал (``overflow``, ``render_error``). Текст с ``text_overflow`` нарисован и считается видимым.

Эмбеддер и лемматизатор подставляются: в обучении — ``SentenceEmbedder`` и ``Lemmatizer`` (pymorphy3), в тестах —
простые заменители. Веса — временные, подбираются на испорченных презентациях.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .dsl import Deck, Element
from .key_points import KeyPoint

VISUAL_KINDS = ("CHART", "PLOT", "DIAGRAM", "TIMELINE", "IMAGE", "FORMULA", "QUOTE")
HIDDEN_STATUSES = frozenset({"overflow", "render_error"})  # сборщик элемент не нарисовал
RENDER_ERROR_VALUE = 0.5  # во сколько раз штраф за неотрисованный элемент меньше штрафа за излишек
COUNTED_STATUSES = frozenset({"ok", "placeholder"})
ERROR_STATUSES = frozenset({"render_error", "undefined_plot"})  # −0,5; flat_plot — ни плюса, ни минуса  # placeholder — картинка, которую при обучении не скачивают
LENGTH_FACTOR = 1.3  # содержательных слайдов на тезис, если число слайдов не задано
VISUAL_TYPE_SHARE = 0.5  # предел на тип: ⌈0,5·N⌉
VISUAL_TOTAL_PER_SLIDE = 2  # общий предел: 2·N


# --------------------------------------------------------------------------- текст слайдов


@dataclass
class SlideText:
    title: str
    content: str

    @property
    def full(self) -> str:
        return f"{self.title}\n{self.content}".strip()


def _hidden(report, slide_number: int) -> set[int]:
    """Строки элементов слайда, которые сборщик не нарисовал (номер слайда — с 1)."""
    if report is None:
        return set()
    return {item.line for item in report.items
            if item.slide == slide_number and item.status in HIDDEN_STATUSES and item.line is not None}


def element_text(element: Element) -> str:
    """Видимый текст элемента для сравнения; пусто — элемент в сравнение не идёт."""
    d = element.data
    if element.kind in ("TEXT", "BULLET"):
        return d.get("text", "")
    if element.kind == "QUOTE":
        return " ".join(filter(None, [d.get("text"), d.get("attribution")]))
    if element.kind == "DIAGRAM":
        labels = list(d.get("nodes", [])) + [e["label"] for e in d.get("edges", []) if e.get("label")]
        return ". ".join(labels)
    if element.kind == "TIMELINE":
        return ". ".join(f"{item['date']} — {item['event']}" for item in d.get("items", []))
    if element.kind == "CHART":
        return ". ".join(filter(None, [d.get("title"), ", ".join(d.get("categories", [])), d.get("unit")]))
    return ""  # IMAGE, FORMULA, PLOT


def slide_texts(deck: Deck, report=None) -> list[SlideText]:
    result = []
    for number, slide in enumerate(deck.slides, 1):
        hidden = _hidden(report, number)
        parts = [element_text(e) for e in slide.elements if e.line not in hidden]
        result.append(SlideText(slide.title or "", "\n".join(p for p in parts if p)))
    return result


# --------------------------------------------------------------------------- лемматизация

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[a-zа-яё]+(?:-[a-zа-яё]+)*", re.IGNORECASE)
_YEAR_WORDS = frozenset({"г", "гг", "год"})
_FUNCTION_POS = frozenset({"PREP", "CONJ", "PRCL", "INTJ", "NPRO"})


class Lemmatizer:
    """Слова в начальной форме; числа — как есть («6 400» → «6400», «3,4» → «3.4»), «г.» и «год» после числа
    убираются, ё → е. ``analyzer`` — объект с ``parse(word) → [разбор с normal_form и tag.POS]`` (pymorphy3)."""

    def __init__(self, analyzer=None):
        if analyzer is None:
            import pymorphy3  # группа зависимостей reward

            analyzer = pymorphy3.MorphAnalyzer()
        self.analyzer = analyzer
        self._cache: dict[str, tuple[str, str | None]] = {}

    def _word(self, word: str) -> tuple[str, str | None]:
        if word not in self._cache:
            parsed = self.analyzer.parse(word)
            if parsed:
                self._cache[word] = (parsed[0].normal_form.replace("ё", "е"), parsed[0].tag.POS)
            else:
                self._cache[word] = (word, None)
        return self._cache[word]

    def tokens(self, text: str) -> list[tuple[str, str | None]]:
        text = text.lower().replace("ё", "е")
        text = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", text)  # 6 400 → 6400
        result: list[tuple[str, str | None]] = []
        for raw in _TOKEN_RE.findall(text):
            if raw[0].isdigit():
                result.append((raw.replace(",", "."), "NUMB"))
                continue
            lemma, pos = self._word(raw)
            if lemma in _YEAR_WORDS and result and result[-1][1] == "NUMB":
                continue
            result.append((lemma, pos))
        return result

    def lemmas(self, text: str) -> list[str]:
        return [lemma for lemma, _ in self.tokens(text)]

    def content_lemmas(self, text: str) -> set[str]:
        """Знаменательные слова и числа (без предлогов, союзов, частиц, местоимений)."""
        return {lemma for lemma, pos in self.tokens(text) if pos not in _FUNCTION_POS}


def contains(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Есть ли ``needle`` подряд в ``haystack``."""
    n = len(needle)
    if not n:
        return False
    return any(list(haystack[i:i + n]) == list(needle) for i in range(len(haystack) - n + 1))


def lexical_share(statement: str, slide_content: str, lemmatizer: Lemmatizer) -> float:
    """Доля знаменательных слов утверждения, найденных на слайде; длина слайда в формулу не входит."""
    words = lemmatizer.content_lemmas(statement)
    if not words:
        return 0.0
    return len(words & set(lemmatizer.lemmas(slide_content))) / len(words)


# --------------------------------------------------------------------------- эмбеддер


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        """Строки — нормированные векторы; ``kind`` — ``query`` (тезисы) или ``passage`` (слайды)."""


class SentenceEmbedder:
    """Эмбеддер на sentence-transformers. Для e5 добавляет префиксы ``query:``/``passage:``."""

    def __init__(self, name: str = "intfloat/multilingual-e5-small", device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # группа зависимостей reward

        self.model = SentenceTransformer(name, device=device)
        self.prefixes = name.lower().find("e5") >= 0
        self.batch_size = batch_size
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if (kind, t) not in self._cache]
        if missing:
            inputs = [f"{kind}: {t}" if self.prefixes else t for t in missing]
            vectors = self.model.encode(inputs, batch_size=self.batch_size, normalize_embeddings=True)
            for text, vector in zip(missing, vectors):
                self._cache[(kind, text)] = np.asarray(vector, dtype=np.float32)
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._cache[(kind, t)] for t in texts])


def _cosine(embedder: Embedder, queries: list[str], passages: list[str]) -> np.ndarray:
    """Матрица косинусов; пустой текст даёт 0."""
    result = np.zeros((len(queries), len(passages)), dtype=np.float32)
    q_idx = [i for i, t in enumerate(queries) if t.strip()]
    p_idx = [j for j, t in enumerate(passages) if t.strip()]
    if q_idx and p_idx:
        q = embedder.encode([queries[i] for i in q_idx], "query")
        p = embedder.encode([passages[j] for j in p_idx], "passage")
        result[np.ix_(q_idx, p_idx)] = q @ p.T
    return result


# --------------------------------------------------------------------------- длина и визуал


def expected_slides(slide_range: tuple[int, int] | None, content_theses: int, conclusion: bool) -> tuple[int, int, bool]:
    """Ожидаемый диапазон числа слайдов и штрафуется ли недостаток.

    Число задано в запросе — оно, штраф в обе стороны. Не задано — титульный + от C до ⌈1,3·C⌉ содержательных +
    заключение; недостаток не штрафуется (его наказывает покрытие).
    """
    if slide_range is not None:
        return slide_range[0], slide_range[1], True
    extra = 1 + (1 if conclusion else 0)
    return content_theses + extra, math.ceil(LENGTH_FACTOR * content_theses) + extra, False


def length_penalty(n_slides: int, low: int, high: int, both_sides: bool) -> float:
    if n_slides > high:
        return (n_slides - high) / high
    if both_sides and n_slides < low:
        return (low - n_slides) / high
    return 0.0


def _normalize_quote(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return " ".join(re.findall(r"[0-9a-zа-я]+", text))


def visual_score(deck: Deck, report, content_slides: int, images_allowed: bool,
                 tool_text: str | None) -> tuple[float, dict[str, int], int, int, int]:
    """Балл за визуал в [−∞, 1], засчитанные элементы по типам, число запрещённых ``#IMAGE``, число элементов,
    не нарисованных из-за ограничения слайда, и число элементов с ошибкой отрисовки."""
    statuses: dict[tuple[int, int], str] = {}
    if report is not None:
        for item in report.items:
            if item.line is not None:
                statuses.setdefault((item.slide, item.line), item.status)
    quotes_source = _normalize_quote(tool_text) if tool_text else ""
    counted: dict[str, int] = {}
    forbidden_images = 0
    slide_overflow = render_errors = 0
    for number, slide in enumerate(deck.slides, 1):
        for element in slide.elements:
            if element.kind not in VISUAL_KINDS:
                continue
            status = statuses.get((number, element.line), "ok" if report is None else None)
            if status == "overflow":  # сверх ограничения слайда: не нарисован, −1
                slide_overflow += 1
                continue
            if status in ERROR_STATUSES:  # не отрисовался или почти пуст: −0,5
                render_errors += 1
                continue
            if element.kind == "IMAGE" and not images_allowed:
                if status not in HIDDEN_STATUSES:
                    forbidden_images += 1
                continue
            if element.kind == "QUOTE":
                quote = _normalize_quote(element.data.get("text", ""))
                if status in HIDDEN_STATUSES or not quote or quote not in quotes_source:
                    continue
            elif status not in COUNTED_STATUSES:
                continue
            counted[element.kind] = counted.get(element.kind, 0) + 1
    n = max(1, content_slides)
    per_type = math.ceil(VISUAL_TYPE_SHARE * n)
    total_limit = VISUAL_TOTAL_PER_SLIDE * n
    within = sum(min(c, per_type) for c in counted.values())
    over = sum(max(0, c - per_type) for c in counted.values())
    over += max(0, within - total_limit)
    within = min(within, total_limit)
    value = (within - over - slide_overflow - RENDER_ERROR_VALUE * render_errors) / total_limit
    return value, counted, forbidden_images, slide_overflow, render_errors


def _call_key(call, lemmatizer: Lemmatizer) -> tuple:
    name = call.get("name", "")
    argument = (call.get("argument") or "").strip()
    if name == "search":
        words = lemmatizer.content_lemmas(argument)
        return name, frozenset(words) if words else argument.lower()
    return name, argument


def tool_penalty(calls: Sequence[dict] | None, lemmatizer: Lemmatizer, call_cost: float,
                 repeat_cost: float, search: bool = True, disabled_cost: float = 0.1) -> tuple[float, int]:
    """Штраф за вызовы инструментов эпизода (``Episode.calls``) и число повторов.

    Неразобранный вызов стоит как обычный, но в повторы не идёт: его аргумент не известен.
    Без поиска (``search=False``) каждый вызов — ``disabled_cost``, повторы не считаются.
    """
    if not calls:
        return 0.0, 0
    if not search:
        return disabled_cost * len(calls), 0
    seen: set[tuple] = set()
    repeats = 0
    for call in calls:
        if call.get("error"):
            continue
        key = _call_key(call, lemmatizer)
        if key in seen:
            repeats += 1
        seen.add(key)
    return call_cost * len(calls) + repeat_cost * repeats, repeats


# --------------------------------------------------------------------------- награда


@dataclass
class RewardWeights:
    """Временные веса; подбираются на испорченных презентациях (train/validation)."""

    title: float = 0.15  # доля заголовка в сходстве тезиса со слайдом
    alpha: float = 0.0  # доля лексической части в сходстве утверждения (кандидат)
    coverage: float = 1.0
    keys: float = 0.5
    length: float = 1.0
    visual: float = 0.2
    forbidden_image: float = 0.2  # за каждый #IMAGE при «Картинки: нет»
    tool_call: float = 0.02  # стоимость каждого вызова инструмента (временная)
    tool_repeat: float = 0.1  # штраф за каждый повтор вызова (решение 27.10.2026)
    tool_disabled: float = 0.1  # штраф за каждый вызов при «Поиск: нет» (решение 27.10.2026)
    bad_source: float = 0.1  # за каждый #SOURCE с id не из этого эпизода (решение 27.10.2026)
    format: float = 0.1  # вес штрафа 1 − parse_rate (временный, малый)
    text_overflow: float = 0.1  # за каждый слайд с не поместившимся текстом (решение 27.10.2026)


@dataclass
class ThesisResult:
    best_slide: int | None  # номер слайда с 1; None — слайдов нет
    similarity: float
    key_slides: list[int]
    found_keys: list[str]
    missing_keys: list[str]


@dataclass
class RewardResult:
    total: float
    coverage: float
    keys: float
    length: float
    visual: float
    forbidden_images: int
    theses: list[ThesisResult] = field(default_factory=list)
    visuals: dict[str, int] = field(default_factory=dict)
    expected_slides: tuple[int, int] = (0, 0)
    slide_overflow: int = 0  # визуальные элементы сверх ограничения слайда
    render_errors: int = 0  # визуальные элементы, которые сборщик не смог отрисовать
    bad_sources: int = 0  # #SOURCE с id не из этого эпизода
    format: float = 0.0  # 1 − parse_rate
    text_overflow: int = 0  # слайды с не поместившимся текстом
    tools: float = 0.0  # штраф за вызовы инструментов
    repeats: int = 0  # повторы вызовов

    def components(self) -> dict[str, float]:
        return {"total": self.total, "coverage": self.coverage, "keys": self.keys, "length": self.length,
                "visual": self.visual, "forbidden_images": self.forbidden_images,
                "slide_overflow": self.slide_overflow, "render_errors": self.render_errors,
                "bad_sources": self.bad_sources, "format": self.format,
                "text_overflow": self.text_overflow, "tools": self.tools, "repeats": self.repeats}


def similarity_matrix(points: Sequence[KeyPoint], texts: Sequence[SlideText], embedder: Embedder,
                      lemmatizer: Lemmatizer | None, weights: RewardWeights) -> np.ndarray:
    titles = _cosine(embedder, [p.title for p in points], [t.title for t in texts])
    statements = _cosine(embedder, [p.statement for p in points], [t.content for t in texts])
    if weights.alpha and lemmatizer is not None:
        lexical = np.array([[lexical_share(p.statement, t.content, lemmatizer) for t in texts] for p in points],
                           dtype=np.float32).reshape(len(points), len(texts))
        statements = (1 - weights.alpha) * statements + weights.alpha * lexical
    return weights.title * titles + (1 - weights.title) * statements


def score(deck: Deck, points: Sequence[KeyPoint], *, conclusion: bool, slide_range: tuple[int, int] | None,
          embedder: Embedder, lemmatizer: Lemmatizer, report=None, images_allowed: bool = True,
          tool_text: str | None = None, calls: Sequence[dict] | None = None, search: bool = True,
          weights: RewardWeights | None = None) -> RewardResult:
    """Награда за разобранную презентацию по тезисам темы.

    ``conclusion`` — последний тезис является заключением (``key_points.expected_theses(task)[2]``);
    ``slide_range`` — число слайдов из запроса (``task.request.slide_range()``); ``report`` — отчёт сборщика
    (без него все элементы считаются нарисованными); ``tool_text`` — ответы инструментов эпизода (для цитат);
    ``calls`` — вызовы инструментов эпизода (``Episode.calls``); ``search`` — режим «Поиск: есть».
    """
    weights = weights or RewardWeights()
    texts = slide_texts(deck, report)
    n = len(texts)
    has_conclusion = conclusion and len(points) > 1
    content_points = list(points[:-1]) if has_conclusion else list(points)
    low, high, both = expected_slides(slide_range, len(content_points), has_conclusion)
    length = length_penalty(n, low, high, both)
    content_slides = n - 1 - (1 if has_conclusion and n > 2 else 0) if n > 1 else n
    visual, visuals, forbidden, slide_overflow, render_errors = visual_score(deck, report, content_slides,
                                                                             images_allowed, tool_text)
    tools, repeats = tool_penalty(calls, lemmatizer, weights.tool_call, weights.tool_repeat, search,
                                  weights.tool_disabled)
    bad_sources = sum(1 for item in report.items if item.status == "bad_source") if report is not None else 0
    format_penalty = 1.0 - deck.parse_rate
    overflow_slides = len({item.slide for item in report.items if item.status == "text_overflow" and item.kind == "TEXT"}) \
        if report is not None else 0

    theses: list[ThesisResult] = []
    if n and points:
        sim = similarity_matrix(points, texts, embedder, lemmatizer, weights)
        slide_lemmas = [lemmatizer.lemmas(t.full) for t in texts]
        # последний слайд — заключение: в выбор лучшего для содержательных тезисов не входит
        pool = list(range(n - 1)) if has_conclusion and n > 1 else list(range(n))
        for i, point in enumerate(points):
            if has_conclusion and i == len(points) - 1:
                candidates = [n - 1]
            else:
                candidates = pool
            ranked = sorted(candidates, key=lambda j: -float(sim[i, j]))
            best = ranked[0]
            key_slides = ranked[:1] if has_conclusion and i == len(points) - 1 else ranked[:2]
            found, missing = [], []
            for key in dict.fromkeys(point.key):  # каждый ключ — один раз
                needle = lemmatizer.lemmas(key)
                (found if any(contains(slide_lemmas[j], needle) for j in key_slides) else missing).append(key)
            theses.append(ThesisResult(best + 1, float(sim[i, best]), [j + 1 for j in key_slides], found, missing))
    else:
        theses = [ThesisResult(None, 0.0, [], [], list(dict.fromkeys(p.key))) for p in points]

    coverage = float(np.mean([t.similarity for t in theses])) if theses else 0.0
    keys = float(np.mean([len(t.found_keys) / max(1, len(t.found_keys) + len(t.missing_keys)) for t in theses])) \
        if theses else 0.0
    total = (weights.coverage * coverage + weights.keys * keys - weights.length * length
             + weights.visual * visual - weights.forbidden_image * forbidden - tools
             - weights.bad_source * bad_sources - weights.format * format_penalty
             - weights.text_overflow * overflow_slides)
    return RewardResult(total, coverage, keys, length, visual, forbidden, theses, visuals, (low, high),
                        slide_overflow, render_errors, bad_sources, format_penalty, overflow_slides, tools, repeats)


def load_key_points(path) -> dict[str, list[KeyPoint]]:
    """Тезисы из ``data/tasks/key_points/key_points.json``: id темы → тезисы."""
    import json
    from pathlib import Path

    topics = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t["id"]: [KeyPoint(p["title"], p["statement"], list(p["key"])) for p in t["key_points"]] for t in topics}

