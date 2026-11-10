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
- визуал: каждый тип до ⌈0,5·N⌉ элементов (картинки — до N), всего до 2·N; в пределах — +1, сверх — −1, сумма делится на 2·N
  (N — содержательные слайды); формулы считаются отдельно: +0,7 за каждую до 2·N, сверх — −0,2 (в той же сумме); засчитывается только нарисованное без ошибок; элемент, не нарисованный из-за
  ограничения слайда (статус ``overflow``: 2 крупных, 3 формулы), — тоже −1; цитата — только найденная в ответах
  инструментов эпизода; ``#IMAGE`` — только при «Картинки: есть», иначе это отдельный штраф;
- цитата, которой нет в выдаче поиска эпизода, — −0,25 в визуальной сумме (только при поиске; многоточие в цитате —
  пропуск, части ищутся по порядку);
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

Изменения по итогам проверки на искажениях (решения 07.11.2026, EXP-014 и EXP-016). Каждое включается отдельно;
по умолчанию включены принятые — прямые штрафы, связь визуала со слайдом (пороги ``VISUAL_TAU``), числа и единицы
в лемматизаторе; остальные выключены (не подтвердились, оставлены для воспроизведения абляций):

- ``RewardWeights.relative`` — покрытие тезиса относительно фона: ``(sim − b)/(1 − b)`` без отсечения нулём; фон b —
  ``thesis_backgrounds`` по чужим презентациям, передаётся в ``score(backgrounds=...)``;
- ``RewardWeights.keys_gate`` — ключ засчитывается только на слайде, где сходство с тезисом выше фона;
- ``Lemmatizer(units=True)`` — «N тысяч» → число, «миллион» = «млн», «миллиард» = «млрд», «₽», «руб.» → «рубль»,
  «%», «процентный» → «процент»;
- ``RewardWeights.contradiction`` — штраф за ключ, у которого связки «число + единица» на его слайдах нет, а та же
  единица стоит с другим числом, которого нет в ключах темы;
- ``RewardWeights.direct_penalties`` — переполнение слайда, ошибка отрисовки, формула сверх предела и выдуманная цитата —
  прямые штрафы, а не доли визуальной суммы;
- ``RewardWeights.visual_tau`` — визуал засчитывается, только если его текст близок к тексту слайда (порог τ по виду
  визуала; вид без порога — за ``visual_unverified`` бонуса); одинаковый элемент (вид и текст) засчитывается один раз,
  повтор — как элемент сверх предела (−1 в сумме; формула — как формула сверх предела).

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
FAKE_QUOTE_PENALTY = 0.25  # цитата, которой нет в выдаче поиска эпизода (только при поиске; решение 29.10.2026)
RENDER_ERROR_VALUE = 0.5  # во сколько раз штраф за неотрисованный элемент меньше штрафа за излишек
COUNTED_STATUSES = frozenset({"ok", "placeholder"})
CHECKED_VISUALS = frozenset({"CHART", "PLOT", "DIAGRAM", "TIMELINE", "IMAGE"})  # связь со слайдом проверяется
# пороги связи визуала со слайдом: bge-m3, 16 исходников (EXP-016, perturb_eval.calibrate_tau); у графиков, хронологий
# и графиков функций мало примеров — без порога, за половину бонуса. Для другого эмбеддера — откалибровать заново.
VISUAL_TAU = {"IMAGE": 0.411, "DIAGRAM": 0.522}
ERROR_STATUSES = frozenset({"render_error", "undefined_plot"})  # −0,5; flat_plot — ни плюса, ни минуса  # placeholder — картинка, которую при обучении не скачивают
LENGTH_FACTOR = 1.3  # содержательных слайдов на тезис, если число слайдов не задано
VISUAL_TYPE_SHARE = 0.5  # предел на тип: ⌈0,5·N⌉
IMAGE_TYPE_SHARE = 1  # предел картинок: N — бывают запросы «по фото на слайд» (решение 29.10.2026)


def type_limit(kind: str, content_slides: int) -> int:
    """Предел засчитываемых элементов одного визуального типа: ⌈0,5·N⌉, у картинок — N."""
    return math.ceil((IMAGE_TYPE_SHARE if kind == "IMAGE" else VISUAL_TYPE_SHARE) * max(1, content_slides))
VISUAL_TOTAL_PER_SLIDE = 2  # общий предел: 2·N (без формул)
FORMULA_VALUE = 0.7  # формула — не визуальный элемент: свой бонус и свой предел (решение 28.10.2026)
FORMULA_TOTAL_PER_SLIDE = 2  # предел формул: 2·N
FORMULA_OVER_PENALTY = 0.2  # за каждую формулу сверх 2·N (бонуса за неё нет)


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


def visual_text(element: Element) -> str:
    """Текст визуала для связи со слайдом и поиска одинаковых элементов."""
    d = element.data
    if element.kind == "IMAGE":
        return d.get("text", "")
    if element.kind == "PLOT":
        return ". ".join(filter(None, [d.get("title"), d.get("expression")]))
    if element.kind == "FORMULA":
        return d.get("latex", "")
    if element.kind == "QUOTE":
        return d.get("text", "")
    return element_text(element)


def slide_body(slide, hidden: set[int]) -> str:
    """Текст слайда без заголовка (``#TEXT``, ``#BULLET``, ``#QUOTE``); если его нет — заголовок."""
    parts = [element_text(e) for e in slide.elements if e.kind in ("TEXT", "BULLET", "QUOTE") and e.line not in hidden]
    body = "\n".join(p for p in parts if p)
    return body or (slide.title or "")


def slide_texts(deck: Deck, report=None) -> list[SlideText]:
    result = []
    for number, slide in enumerate(deck.slides, 1):
        hidden = _hidden(report, number)
        parts = [element_text(e) for e in slide.elements if e.line not in hidden]
        result.append(SlideText(slide.title or "", "\n".join(p for p in parts if p)))
    return result


# --------------------------------------------------------------------------- лемматизация

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[a-zа-яё]+(?:-[a-zа-яё]+)*|[%₽$€]", re.IGNORECASE)
_UNIT_SYMBOLS = {"%": "процент", "₽": "рубль", "$": "доллар", "€": "евро"}
_UNIT_LEMMAS = {"процентный": "процент", "руб": "рубль", "миллион": "млн", "миллиард": "млрд"}
_UNIT_NOUNS = frozenset({"процент", "рубль", "доллар", "евро", "млн", "млрд"})
_THOUSAND = frozenset({"тысяча", "тыс"})
_YEAR_WORDS = frozenset({"г", "гг", "год"})
_FUNCTION_POS = frozenset({"PREP", "CONJ", "PRCL", "INTJ", "NPRO"})


class Lemmatizer:
    """Слова в начальной форме; числа — как есть («6 400» → «6400», «3,4» → «3.4»), «г.» и «год» после числа
    убираются, ё → е. ``analyzer`` — объект с ``parse(word) → [разбор с normal_form и tag.POS]`` (pymorphy3).

    ``units`` — числа и единицы (решение 07.11.2026; False — прежнее поведение): «N тысяч» → N·1000, «миллион» → «млн», «миллиард» → «млрд»,
    «₽», «руб.» → «рубль», «%», «процентный» → «процент» («4-процентным» → «4 процент»)."""

    def __init__(self, analyzer=None, units: bool = True):
        if analyzer is None:
            import pymorphy3  # группа зависимостей reward

            analyzer = pymorphy3.MorphAnalyzer()
        self.analyzer = analyzer
        self.units = units
        self._cache: dict[str, tuple[str, str | None]] = {}

    def _word(self, word: str) -> tuple[str, str | None]:
        if word not in self._cache:
            parsed = self.analyzer.parse(word)
            if parsed:
                pos = parsed[0].tag.POS  # у pymorphy3 — не строка: сравнение с «NUMB» бросает ValueError
                self._cache[word] = (parsed[0].normal_form.replace("ё", "е"), str(pos) if pos is not None else None)
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
            if raw in _UNIT_SYMBOLS:
                if self.units:
                    result.append((_UNIT_SYMBOLS[raw], "NOUN"))
                continue
            lemma, pos = self._word(raw)
            after_number = bool(result) and result[-1][1] == "NUMB"
            if lemma in _YEAR_WORDS and after_number:
                continue
            if self.units:
                if lemma in _THOUSAND and after_number:
                    result[-1] = (_times_thousand(result[-1][0]), "NUMB")
                    continue
                lemma = _UNIT_LEMMAS.get(lemma, lemma)
                if lemma in _UNIT_NOUNS:
                    pos = "NOUN"
            result.append((lemma, pos))
        return result

    def lemmas(self, text: str) -> list[str]:
        return [lemma for lemma, _ in self.tokens(text)]

    def content_lemmas(self, text: str) -> set[str]:
        """Знаменательные слова и числа (без предлогов, союзов, частиц, местоимений)."""
        return {lemma for lemma, pos in self.tokens(text) if pos not in _FUNCTION_POS}


def _times_thousand(number: str) -> str:
    value = float(number) * 1000
    return str(int(value)) if value.is_integer() else str(value)


def number_units(tokens: Sequence[tuple[str, str | None]]) -> set[tuple[str, str]]:
    """Связки «число + единица»: единица — существительное или слово без части речи (латиница, сокращение
    вне словаря: «нм», «TPS») сразу после числа («8.5 процент», «6 октябрь», «300 dpi»)."""
    return {(tokens[i][0], tokens[i + 1][0]) for i in range(len(tokens) - 1)
            if tokens[i][1] == "NUMB" and tokens[i + 1][1] in ("NOUN", None)}


def contradicts(key_pairs: set[tuple[str, str]], slide_pairs: set[tuple[str, str]],
                allowed: set[tuple[str, str]] = frozenset()) -> bool:
    """Верной связки нет, а та же единица стоит с другим числом. ``allowed`` — связки из ключей всей темы:
    такое число — другая величина той же темы («план 12 млн» рядом с ключом «1,2 млн»), не противоречие."""
    return any((number, unit) not in slide_pairs
               and any(u == unit and n != number and (n, u) not in allowed for n, u in slide_pairs)
               for number, unit in key_pairs)


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

    def __init__(self, name: str = "BAAI/bge-m3", device: str | None = None, batch_size: int = 32):
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


def pairwise_cosine(embedder: Embedder, left: list[str], right: list[str]) -> np.ndarray:
    """Косинус каждой пары (оба текста — как ``passage``); пустой текст даёт 0."""
    result = np.zeros(len(left), dtype=np.float32)
    idx = [i for i in range(len(left)) if left[i].strip() and right[i].strip()]
    if idx:
        a = embedder.encode([left[i] for i in idx], "passage")
        b = embedder.encode([right[i] for i in idx], "passage")
        result[idx] = np.sum(a * b, axis=1)
    return result


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
    text = text.replace("\u0301", "").lower().replace("ё", "е")  # знак ударения Википедии («по́ля») — не граница слова
    return " ".join(re.findall(r"[0-9a-zа-я]+", text))


def quote_found(quote: str, source: str) -> bool:
    """Есть ли цитата в выдаче (``source`` — уже ``_normalize_quote``). Многоточие — пропуск: части цитаты
    ищутся по порядку («Без выражения… даже величайшая любовь может умереть»)."""
    parts = [p for p in (_normalize_quote(x) for x in re.split(r"…|\.\.\.", quote)) if p]
    position = 0
    for part in parts:
        found = source.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return bool(parts)


@dataclass
class VisualResult:
    value: float  # балл за визуал в [−∞, 1]
    counted: dict[str, int]  # засчитанные элементы по типам (до пределов)
    forbidden: int = 0  # #IMAGE при «Картинки: нет»
    slide_overflow: int = 0  # не нарисованы из-за ограничения слайда
    render_errors: int = 0  # ошибка отрисовки или неопределённый график
    fake_quotes: int = 0  # цитаты, которых нет в выдаче поиска
    formula_over: int = 0  # формулы сверх 2·N
    unrelated: int = 0  # визуал, не связанный со слайдом (ниже τ)
    duplicates: int = 0  # повтор уже засчитанного элемента


def visual_details(deck: Deck, report, content_slides: int, images_allowed: bool, tool_text: str | None, *,
                   direct: bool = False, embedder: Embedder | None = None, tau: dict[str, float] | None = None,
                   unverified: float = 0.5) -> VisualResult:
    """Визуал презентации (``tool_text`` — выдача эпизода; None — режим без поиска: цитаты ни плюс, ни минус).

    ``direct`` — переполнение, ошибки отрисовки, формулы сверх предела и выдуманные цитаты не входят в сумму (их
    штрафует ``score`` напрямую). ``tau`` — пороги связи визуала со слайдом по видам (None — без проверки): ниже
    порога — ни бонуса, ни штрафа; вид из ``CHECKED_VISUALS`` без порога — за ``unverified`` бонуса; одинаковые
    элементы (вид и текст) засчитываются один раз, повтор — как элемент сверх предела. Нужен ``embedder``.
    """
    statuses: dict[tuple[int, int], str] = {}
    if report is not None:
        for item in report.items:
            if item.line is not None:
                statuses.setdefault((item.slide, item.line), item.status)
    quotes_source = _normalize_quote(tool_text) if tool_text else ""
    result = VisualResult(0.0, {})
    entries: list[tuple[str, float]] = []  # вид, вес бонуса
    checks: list[tuple[int, str, str]] = []  # номер в entries, текст визуала, текст слайда
    seen: set[tuple[str, str]] = set()
    duplicate_formulas = 0
    for number, slide in enumerate(deck.slides, 1):
        body = None
        for element in slide.elements:
            if element.kind not in VISUAL_KINDS:
                continue
            status = statuses.get((number, element.line), "ok" if report is None else None)
            if status == "overflow":  # сверх ограничения слайда: не нарисован
                result.slide_overflow += 1
                continue
            if status in ERROR_STATUSES:  # не отрисовался или почти пуст
                result.render_errors += 1
                continue
            if element.kind == "IMAGE" and not images_allowed:
                if status not in HIDDEN_STATUSES:
                    result.forbidden += 1
                continue
            if element.kind == "QUOTE":
                if status in HIDDEN_STATUSES:
                    continue
                if not quote_found(element.data.get("text", ""), quotes_source):
                    result.fake_quotes += tool_text is not None
                    continue
            elif status not in COUNTED_STATUSES:
                continue
            weight = 1.0
            if tau is not None:
                text = visual_text(element)
                key = (element.kind, _normalize_quote(text))
                if key in seen:  # повтор — как элемент сверх предела
                    result.duplicates += 1
                    duplicate_formulas += element.kind == "FORMULA"
                    continue
                seen.add(key)
                if element.kind in CHECKED_VISUALS:
                    if element.kind in tau:
                        if body is None:
                            body = slide_body(slide, _hidden(report, number))
                        checks.append((len(entries), text, body))
                    else:
                        weight = unverified
            entries.append((element.kind, weight))
    if checks:
        if embedder is None:
            raise ValueError("для проверки связи визуала со слайдом нужен эмбеддер")
        unrelated = set()
        sims = pairwise_cosine(embedder, [c[1] for c in checks], [c[2] for c in checks])
        for (index, _, _), sim in zip(checks, sims):
            if sim < tau[entries[index][0]]:
                unrelated.add(index)
        result.unrelated = len(unrelated)
        entries = [e for i, e in enumerate(entries) if i not in unrelated]

    by_kind: dict[str, list[float]] = {}
    for kind, weight in entries:
        by_kind.setdefault(kind, []).append(weight)
    result.counted = {kind: len(ws) for kind, ws in by_kind.items()}
    n = max(1, content_slides)
    total_limit = VISUAL_TOTAL_PER_SLIDE * n
    within: list[float] = []
    over = 0
    for kind, ws in by_kind.items():
        if kind == "FORMULA":
            continue
        ws = sorted(ws, reverse=True)
        limit = type_limit(kind, n)
        within += ws[:limit]
        over += max(0, len(ws) - limit)
    within.sort(reverse=True)
    over += max(0, len(within) - total_limit) + result.duplicates - duplicate_formulas
    formulas = len(by_kind.get("FORMULA", []))
    formula_within = min(formulas, FORMULA_TOTAL_PER_SLIDE * n)
    result.formula_over = formulas - formula_within + duplicate_formulas
    value = sum(within[:total_limit]) + FORMULA_VALUE * formula_within - over
    if not direct:
        value -= (FORMULA_OVER_PENALTY * result.formula_over + result.slide_overflow
                  + RENDER_ERROR_VALUE * result.render_errors + FAKE_QUOTE_PENALTY * result.fake_quotes)
    result.value = value / total_limit
    return result


def visual_score(deck: Deck, report, content_slides: int, images_allowed: bool,
                 tool_text: str | None) -> tuple[float, dict[str, int], int, int, int, int]:
    """Прежний вид ``visual_details``: балл, засчитанные по типам, запрещённые ``#IMAGE``, переполнение,
    ошибки отрисовки, выдуманные цитаты."""
    r = visual_details(deck, report, content_slides, images_allowed, tool_text)
    return r.value, r.counted, r.forbidden, r.slide_overflow, r.render_errors, r.fake_quotes


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
    alpha: float = 0.2  # доля лексической части в сходстве утверждения (решение 07.11.2026, EXP-014)
    coverage: float = 1.0
    keys: float = 0.5
    length: float = 1.0
    visual: float = 0.2
    forbidden_image: float = 0.2  # за каждый #IMAGE при «Картинки: нет»
    tool_call: float = 0.003  # стоимость каждого вызова при поиске (решение 08.11.2026; было 0,02 — ~1,4σ за 5 вызовов)
    tool_repeat: float = 0.1  # штраф за каждый повтор вызова (решение 27.10.2026)
    tool_disabled: float = 0.1  # штраф за каждый вызов при «Поиск: нет» (решение 27.10.2026)
    bad_source: float = 0.1  # за каждый #SOURCE с id не из этого эпизода (решение 27.10.2026)
    format: float = 0.1  # вес штрафа 1 − parse_rate (временный, малый)
    text_overflow: float = 0.1  # за каждый слайд с не поместившимся текстом (решение 27.10.2026)
    # изменения 07.11.2026 (EXP-014, EXP-016): не подтвердились — выключены, оставлены для воспроизведения абляций
    relative: bool = False  # покрытие относительно фона (нужен score(backgrounds=...))
    keys_gate: bool = False  # ключ — только на слайде со сходством выше фона (нужен фон)
    contradiction: float = 0.0  # за ключ с числовым противоречием (в абляциях — 0,05)
    # приняты
    direct_penalties: bool = True  # прямые штрафы вместо долей визуальной суммы
    overflow_penalty: float = 0.05  # элемент сверх ограничения слайда
    render_penalty: float = 0.05  # ошибка отрисовки, неопределённый график
    formula_over_penalty: float = 0.05  # формула сверх 2·N
    fake_quote_penalty: float = 0.1  # цитата, которой нет в выдаче
    visual_tau: dict[str, float] | None = field(default_factory=lambda: dict(VISUAL_TAU))  # None — без проверки
    visual_unverified: float = 0.5  # доля бонуса за вид визуала без порога


@dataclass
class ThesisResult:
    best_slide: int | None  # номер слайда с 1; None — слайдов нет
    similarity: float
    key_slides: list[int]
    found_keys: list[str]
    missing_keys: list[str]
    background: float | None = None  # фон тезиса (если задан)
    contradictions: int = 0  # ключи с числовым противоречием


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
    fake_quotes: int = 0  # цитаты, которых нет в выдаче поиска
    bad_sources: int = 0  # #SOURCE с id не из этого эпизода
    format: float = 0.0  # 1 − parse_rate
    text_overflow: int = 0  # слайды с не поместившимся текстом
    tools: float = 0.0  # штраф за вызовы инструментов
    repeats: int = 0  # повторы вызовов
    contradictions: int = 0  # ключи с числовым противоречием
    formula_over: int = 0  # формулы сверх 2·N
    unrelated_visuals: int = 0  # визуал ниже порога связи со слайдом

    def components(self) -> dict[str, float]:
        return {"total": self.total, "coverage": self.coverage, "keys": self.keys, "length": self.length,
                "visual": self.visual, "forbidden_images": self.forbidden_images,
                "slide_overflow": self.slide_overflow, "render_errors": self.render_errors, "fake_quotes": self.fake_quotes,
                "bad_sources": self.bad_sources, "format": self.format,
                "text_overflow": self.text_overflow, "tools": self.tools, "repeats": self.repeats,
                "contradictions": self.contradictions, "formula_over": self.formula_over,
                "unrelated_visuals": self.unrelated_visuals}


def similarity_matrix(points: Sequence[KeyPoint], texts: Sequence[SlideText], embedder: Embedder,
                      lemmatizer: Lemmatizer | None, weights: RewardWeights) -> np.ndarray:
    titles = _cosine(embedder, [p.title for p in points], [t.title for t in texts])
    statements = _cosine(embedder, [p.statement for p in points], [t.content for t in texts])
    if weights.alpha and lemmatizer is not None:
        lexical = np.array([[lexical_share(p.statement, t.content, lemmatizer) for t in texts] for p in points],
                           dtype=np.float32).reshape(len(points), len(texts))
        statements = (1 - weights.alpha) * statements + weights.alpha * lexical
    return weights.title * titles + (1 - weights.title) * statements


def thesis_backgrounds(points: Sequence[KeyPoint], foreign: Sequence[Sequence[SlideText]], embedder: Embedder,
                       lemmatizer: Lemmatizer | None, weights: RewardWeights, *, conclusion: bool) -> list[float]:
    """Фон тезисов: среднее по чужим презентациям (``slide_texts`` каждой) лучшего сходства тезиса с их слайдами
    по той же формуле, что в ``score``; тезис-заключение — сходство с последним слайдом."""
    has_conclusion = conclusion and len(points) > 1
    rows = []
    for texts in foreign:
        n = len(texts)
        if not n:
            continue
        sim = similarity_matrix(points, texts, embedder, lemmatizer, weights)
        pool = list(range(n - 1)) if has_conclusion and n > 1 else list(range(n))
        rows.append([float(sim[i, n - 1]) if has_conclusion and i == len(points) - 1 else float(sim[i, pool].max())
                     for i in range(len(points))])
    if not rows:
        raise ValueError("нет чужих презентаций для фона")
    return [float(x) for x in np.mean(rows, axis=0)]


def score(deck: Deck, points: Sequence[KeyPoint], *, conclusion: bool, slide_range: tuple[int, int] | None,
          embedder: Embedder, lemmatizer: Lemmatizer, report=None, images_allowed: bool = True,
          tool_text: str | None = None, calls: Sequence[dict] | None = None, search: bool = True,
          weights: RewardWeights | None = None, backgrounds: Sequence[float] | None = None) -> RewardResult:
    """Награда за разобранную презентацию по тезисам темы.

    ``conclusion`` — последний тезис является заключением (``key_points.expected_theses(task)[2]``);
    ``slide_range`` — число слайдов из запроса (``task.request.slide_range()``); ``report`` — отчёт сборщика
    (без него все элементы считаются нарисованными); ``tool_text`` — ответы инструментов эпизода (для цитат);
    ``calls`` — вызовы инструментов эпизода (``Episode.calls``); ``search`` — режим «Поиск: есть»;
    ``backgrounds`` — фон каждого тезиса (``thesis_backgrounds``), нужен при ``relative`` и ``keys_gate``.
    """
    weights = weights or RewardWeights()
    if (weights.relative or weights.keys_gate) and (backgrounds is None or len(backgrounds) != len(points)):
        raise ValueError("для покрытия относительно фона и ключей по фону нужен фон каждого тезиса")
    texts = slide_texts(deck, report)
    n = len(texts)
    has_conclusion = conclusion and len(points) > 1
    content_points = list(points[:-1]) if has_conclusion else list(points)
    low, high, both = expected_slides(slide_range, len(content_points), has_conclusion)
    length = length_penalty(n, low, high, both)
    content_slides = n - 1 - (1 if has_conclusion and n > 2 else 0) if n > 1 else n
    v = visual_details(deck, report, content_slides, images_allowed, tool_text, direct=weights.direct_penalties,
                       embedder=embedder, tau=weights.visual_tau, unverified=weights.visual_unverified)
    tools, repeats = tool_penalty(calls, lemmatizer, weights.tool_call, weights.tool_repeat, search,
                                  weights.tool_disabled)
    bad_sources = sum(1 for item in report.items if item.status == "bad_source") if report is not None else 0
    format_penalty = 1.0 - deck.parse_rate
    overflow_slides = len({item.slide for item in report.items if item.status == "text_overflow" and item.kind == "TEXT"}) \
        if report is not None else 0

    theses: list[ThesisResult] = []
    if n and points:
        sim = similarity_matrix(points, texts, embedder, lemmatizer, weights)
        slide_tokens = [lemmatizer.tokens(t.full) for t in texts]
        slide_lemmas = [[lemma for lemma, _ in tokens] for tokens in slide_tokens]
        slide_pairs = [number_units(tokens) for tokens in slide_tokens]
        topic_pairs = set().union(*(number_units(lemmatizer.tokens(k)) for p in points for k in p.key))
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
            b = float(backgrounds[i]) if backgrounds is not None else None
            # ключи — на слайдах, где тезис раскрыт (выше фона), если включено
            counted_slides = [j for j in key_slides if sim[i, j] > b] if weights.keys_gate else key_slides
            pairs_here = set().union(*(slide_pairs[j] for j in key_slides))
            found, missing = [], []
            contradictions = 0
            for key in dict.fromkeys(point.key):  # каждый ключ — один раз
                key_tokens = lemmatizer.tokens(key)
                needle = [lemma for lemma, _ in key_tokens]
                (found if any(contains(slide_lemmas[j], needle) for j in counted_slides) else missing).append(key)
                contradictions += contradicts(number_units(key_tokens), pairs_here, topic_pairs)
            theses.append(ThesisResult(best + 1, float(sim[i, best]), [j + 1 for j in key_slides], found, missing,
                                       b, contradictions))
    else:
        theses = [ThesisResult(None, 0.0, [], [], list(dict.fromkeys(p.key)),
                               float(backgrounds[i]) if backgrounds is not None else None)
                  for i, p in enumerate(points)]

    def thesis_coverage(t: ThesisResult) -> float:
        if not weights.relative:
            return t.similarity
        return (t.similarity - t.background) / max(1e-6, 1 - t.background)  # без отсечения нулём

    coverage = float(np.mean([thesis_coverage(t) for t in theses])) if theses else 0.0
    contradictions = sum(t.contradictions for t in theses)
    keys = float(np.mean([len(t.found_keys) / max(1, len(t.found_keys) + len(t.missing_keys)) for t in theses])) \
        if theses else 0.0
    total = (weights.coverage * coverage + weights.keys * keys - weights.length * length
             + weights.visual * v.value - weights.forbidden_image * v.forbidden - tools
             - weights.bad_source * bad_sources - weights.format * format_penalty
             - weights.text_overflow * overflow_slides - weights.contradiction * contradictions)
    if weights.direct_penalties:  # один элемент — один штраф; бонуса за ненарисованное нет
        total -= (weights.overflow_penalty * v.slide_overflow + weights.render_penalty * v.render_errors
                  + weights.formula_over_penalty * v.formula_over + weights.fake_quote_penalty * v.fake_quotes)
    return RewardResult(total, coverage, keys, length, v.value, v.forbidden, theses, v.counted, (low, high),
                        v.slide_overflow, v.render_errors, v.fake_quotes, bad_sources, format_penalty, overflow_slides,
                        tools, repeats, contradictions, v.formula_over, v.unrelated)


def load_key_points(path) -> dict[str, list[KeyPoint]]:
    """Тезисы из ``data/tasks/key_points/key_points.json``: id темы → тезисы."""
    import json
    from pathlib import Path

    topics = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t["id"]: [KeyPoint(p["title"], p["statement"], list(p["key"])) for p in t["key_points"]] for t in topics}

