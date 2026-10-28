# Здесь проверяем логику содержательной награды на подставных эмбеддере и лемматизаторе, без загрузки моделей.
import math
from types import SimpleNamespace

import numpy as np
import pytest

from presentation_rl.dsl import parse
from presentation_rl.key_points import KeyPoint
from presentation_rl.rendering.builder import ElementReport, RenderReport
from presentation_rl.rewards import (Lemmatizer, RewardWeights, expected_slides, length_penalty, lexical_share, score,
                                     slide_texts, tool_penalty, visual_score)

FORMS = {"льда": "лед", "лед": "лед", "фототока": "фототок", "фототок": "фототок", "года": "год", "г": "г",
         "матрицы": "матрица", "матрица": "матрица", "градиентного": "градиентный", "градиентный": "градиентный",
         "спуска": "спуск", "спуск": "спуск"}
FUNCTION = {"в": "PREP", "и": "CONJ", "не": "PRCL", "до": "PREP"}


class FakeAnalyzer:
    """Начальная форма — из словаря FORMS, иначе само слово; часть речи — из FUNCTION, иначе NOUN."""

    def parse(self, word):
        return [SimpleNamespace(normal_form=FORMS.get(word, word), tag=SimpleNamespace(POS=FUNCTION.get(word, "NOUN")))]


class FakeEmbedder:
    """Сходство по общим начальным формам слов: мешок слов с хешированием."""

    def __init__(self):
        self.lemmatizer = Lemmatizer(FakeAnalyzer())

    def encode(self, texts, kind):
        rows = []
        for text in texts:
            v = np.zeros(256, dtype=np.float32)
            for lemma in self.lemmatizer.content_lemmas(text):
                v[hash(lemma) % 256] += 1
            norm = np.linalg.norm(v)
            rows.append(v / norm if norm else v)
        return np.stack(rows)


LEM = Lemmatizer(FakeAnalyzer())
EMB = FakeEmbedder()


def deck_of(*slides):
    lines = []
    for slide in slides:
        lines.append("#SLIDE")
        lines += slide
    return parse("\n".join(lines))


def test_only_visible_text_goes_to_comparison():
    deck = deck_of(["#TITLE: Заголовок", "#TEXT: видимый текст", "#NOTES: скрытые заметки",
                    "#IMAGE: картинка кота", "#FORMULA: E = mc^2", "#DIAGRAM: type=flow; узел альфа -> узел бета",
                    "#TIMELINE: 1905 — первое событие; 1915 — второе событие",
                    "#CHART: type=bar; title=Доли; x=один,два; y=1,2"])
    text = slide_texts(deck)[0]
    assert text.title == "Заголовок"
    for visible in ("видимый текст", "узел альфа", "1905 — первое событие", "Доли", "один, два"):
        assert visible in text.content
    for hidden in ("скрытые", "кота", "mc"):
        assert hidden not in text.content
    diagram_line = next(e.line for e in deck.slides[0].elements if e.kind == "DIAGRAM")
    report = RenderReport(path=None, items=[ElementReport(1, "DIAGRAM", "overflow", diagram_line)])
    assert "узел альфа" not in slide_texts(deck, report)[0].content


def test_keys_match_word_forms_numbers_and_year_words():
    assert LEM.lemmas("В 1888 г. изучали фототока") == ["в", "1888", "изучали", "фототок"]
    assert LEM.lemmas("6 400 вольт и 3,4 нм") == ["6400", "вольт", "и", "3.4", "нм"]
    deck = deck_of(["#TITLE: Опыты", "#TEXT: В 1888 г. изучали фототока на слое льда при 6 400 вольт"])
    point = KeyPoint("Опыты", "Опыты с фототоком", ["фототок", "1888", "лед", "6400", "фототок"])
    result = score(deck, [point], conclusion=False, slide_range=None, embedder=EMB, lemmatizer=LEM)
    assert result.theses[0].found_keys == ["фототок", "1888", "лед", "6400"]  # повтор ключа считается один раз
    assert result.keys == 1.0


def test_conclusion_is_compared_only_with_last_slide_and_last_slide_is_not_a_best_slide():
    deck = deck_of(["#TITLE: Тема"], ["#TITLE: Матрица", "#TEXT: матрица и определитель"],
                   ["#TITLE: Ранг", "#TEXT: ранг системы"], ["#TITLE: Итог", "#TEXT: матрица определитель ранг итог"])
    points = [KeyPoint("Матрица", "матрица и определитель", ["матрица"]),
              KeyPoint("Ранг", "ранг системы", ["ранг"]),
              KeyPoint("Вывод: итог", "итог матрица определитель ранг", ["итог"])]
    result = score(deck, points, conclusion=True, slide_range=None, embedder=EMB, lemmatizer=LEM)
    assert [t.best_slide for t in result.theses] == [2, 3, 4]
    assert result.theses[2].key_slides == [4]
    assert 4 not in result.theses[0].key_slides + result.theses[1].key_slides
    plain = score(deck, points[:2], conclusion=False, slide_range=None, embedder=EMB, lemmatizer=LEM)
    assert any(4 in t.key_slides for t in plain.theses)  # без заключения последний слайд — обычный


def test_keys_are_searched_only_on_two_best_slides():
    deck = deck_of(["#TITLE: Альфа", "#TEXT: альфа бета"], ["#TITLE: Альфа", "#TEXT: альфа гамма"],
                   ["#TITLE: Прочее", "#TEXT: дельта ключик"])
    point = KeyPoint("Альфа", "альфа бета гамма", ["ключик", "бета"])
    result = score(deck, [point], conclusion=False, slide_range=None, embedder=EMB, lemmatizer=LEM)
    assert result.theses[0].found_keys == ["бета"]
    assert result.theses[0].missing_keys == ["ключик"]


def test_expected_slides_and_length_penalty():
    assert expected_slides(None, 8, True) == (10, math.ceil(1.3 * 8) + 2, False)
    low, high, both = expected_slides(None, 8, True)
    assert length_penalty(high, low, high, both) == 0
    assert length_penalty(5, low, high, both) == 0  # короткую наказывает покрытие
    assert length_penalty(high + 2, low, high, both) == pytest.approx(2 / high)
    assert expected_slides((10, 10), 8, True) == (10, 10, True)
    assert length_penalty(12, 10, 10, True) == pytest.approx(0.2)
    assert length_penalty(8, 10, 10, True) == pytest.approx(0.2)


def test_image_limit_is_one_per_content_slide():
    slides = [["#TITLE: Тема"]] + [["#TITLE: Гриб", "#IMAGE: гриб"] for _ in range(5)] + [["#TITLE: Итог"]]
    deck = deck_of(*slides)  # N = 4: картинок до N = 4 (не ⌈0,5·N⌉ = 2), всего до 8
    value, counted, _, _, _, _ = visual_score(deck, None, 4, images_allowed=True, tool_text=None)
    assert counted == {"IMAGE": 5}
    assert value == pytest.approx((4 - 1) / 8)  # пятая сверх предела — −1


def test_visual_limits_quotes_and_images():
    slides = [["#TITLE: Тема"]]
    slides += [["#TITLE: Слайд", "#TIMELINE: 1905 — а; 1906 — б"] for _ in range(3)]
    slides += [["#TITLE: Цитаты", "#QUOTE: «Знание — сила» — Бэкон", "#QUOTE: «Выдумано мной» — Никто",
                "#IMAGE: кот"]]
    slides += [["#TITLE: Итог"]]
    deck = deck_of(*slides)  # 6 слайдов, содержательных N = 4: на тип ⌈0,5·4⌉ = 2, всего 2·4 = 8
    value, counted, forbidden, _, _, _ = visual_score(deck, None, 4, images_allowed=False, tool_text="… знание — сила …")
    assert counted == {"TIMELINE": 3, "QUOTE": 1}  # выдуманная цитата не засчитана, картинка запрещена
    assert forbidden == 1
    assert value == pytest.approx((2 + 1 - 1 - 0.25) / 8)  # третья лента сверх предела — −1, выдуманная цитата — −0,25
    value, counted, forbidden, _, _, _ = visual_score(deck, None, 4, images_allowed=True, tool_text=None)
    assert counted == {"TIMELINE": 3, "IMAGE": 1} and forbidden == 0  # без поиска цитаты не засчитываются
    line = deck.slides[1].elements[0].line
    report = RenderReport(path=None, items=[ElementReport(2, "TIMELINE", "render_error", line)]
                          + [ElementReport(s, "TIMELINE", "ok", deck.slides[s - 1].elements[0].line) for s in (3, 4)])
    value, counted, _, _, errors, _ = visual_score(deck, report, 4, images_allowed=False, tool_text=None)
    assert counted == {"TIMELINE": 2} and errors == 1  # с ошибкой отрисовки не засчитывается
    assert value == pytest.approx((2 - 0.5) / 8)  # и даёт −0,5


def test_lexical_share_does_not_punish_long_slides():
    short = lexical_share("матрица и определитель", "матрица", LEM)
    long = lexical_share("матрица и определитель", "матрица " + "слово " * 50 + "определитель", LEM)
    assert short == pytest.approx(0.5) and long == pytest.approx(1.0)  # «и» не считается


def test_score_combines_components_and_handles_empty_deck():
    deck = deck_of(["#TITLE: Матрица", "#TEXT: матрица и определитель"])
    points = [KeyPoint("Матрица", "матрица и определитель", ["матрица"])]
    weights = RewardWeights(alpha=0.3)
    result = score(deck, points, conclusion=False, slide_range=(1, 1), embedder=EMB, lemmatizer=LEM, weights=weights)
    assert result.length == 0 and result.keys == 1.0 and result.coverage > 0.5
    assert result.total == pytest.approx(weights.coverage * result.coverage + weights.keys * result.keys
                                         + weights.visual * result.visual)
    empty = score(parse(""), points, conclusion=False, slide_range=(1, 1), embedder=EMB, lemmatizer=LEM)
    assert empty.coverage == 0 and empty.keys == 0 and empty.length == pytest.approx(1.0)


def test_visual_over_slide_limit_is_minus_one():
    deck = deck_of(["#TITLE: Тема"], ["#TITLE: Слайд", "#TIMELINE: 1905 — а; 1906 — б", "#TIMELINE: 1907 — в; 1908 — г",
                                      "#TIMELINE: 1909 — д; 1910 — е"], ["#TITLE: Итог"])
    lines = [e.line for e in deck.slides[1].elements if e.kind == "TIMELINE"]
    report = RenderReport(path=None, items=[ElementReport(2, "TIMELINE", "ok", lines[0]),
                                            ElementReport(2, "TIMELINE", "ok", lines[1]),
                                            ElementReport(2, "TIMELINE", "overflow", lines[2])])
    value, counted, _, overflow, _, _ = visual_score(deck, report, 1, images_allowed=True, tool_text=None)
    # N = 1: на тип ⌈0,5⌉ = 1, всего 2; вторая лента — сверх предела типа (−1), третья не нарисована (−1)
    assert counted == {"TIMELINE": 2} and overflow == 1
    assert value == pytest.approx((1 - 1 - 1) / 2)


def test_tool_repeats_by_word_forms_and_ids():
    calls = [{"name": "search", "argument": "Градиентный спуск"},
             {"name": "search", "argument": "спуск градиентного"},  # те же слова в другой форме и порядке — повтор
             {"name": "search", "argument": "градиентный спуск в матрица"},  # новое слово — не повтор
             {"name": "open", "argument": "d1"}, {"name": "read", "argument": "d1#2"},
             {"name": "read", "argument": " d1#2 "},  # тот же раздел — повтор
             {"name": "read", "argument": "d1#3"},
             {"name": "", "argument": "", "error": "не JSON"}, {"name": "", "argument": "", "error": "не JSON"}]
    penalty, repeats = tool_penalty(calls, LEM, call_cost=0.02, repeat_cost=0.1)
    assert repeats == 2  # неразобранные вызовы в повторы не идут
    assert penalty == pytest.approx(0.02 * len(calls) + 0.1 * 2)
    assert tool_penalty(None, LEM, 0.02, 0.1) == (0.0, 0)
    deck = deck_of(["#TITLE: Матрица", "#TEXT: матрица и определитель"])
    points = [KeyPoint("Матрица", "матрица и определитель", ["матрица"])]
    without = score(deck, points, conclusion=False, slide_range=(1, 1), embedder=EMB, lemmatizer=LEM)
    with_calls = score(deck, points, conclusion=False, slide_range=(1, 1), embedder=EMB, lemmatizer=LEM, calls=calls)
    assert with_calls.repeats == 2 and without.total - with_calls.total == pytest.approx(penalty)


def test_no_search_calls_bad_sources_and_format():
    calls = [{"name": "search", "argument": "спуск", "disabled": True}] * 2
    assert tool_penalty(calls, LEM, 0.02, 0.1, search=False, disabled_cost=0.1) == (pytest.approx(0.2), 0)
    deck = parse("#SLIDE\n#TITLE: Матрица\n#TEXT: матрица и определитель\n#SOURCE: d9\nПрезентация завершена")
    points = [KeyPoint("Матрица", "матрица и определитель", ["матрица"])]
    report = RenderReport(path=None, items=[ElementReport(1, "SOURCE", "bad_source", 4)])
    weights = RewardWeights()
    result = score(deck, points, conclusion=False, slide_range=(1, 1), embedder=EMB, lemmatizer=LEM, report=report,
                   calls=calls, search=False)
    assert result.bad_sources == 1 and result.format == pytest.approx(1 - deck.parse_rate) and result.format > 0
    assert result.tools == pytest.approx(0.2)
    assert result.total == pytest.approx(weights.coverage * result.coverage + weights.keys * result.keys
                                         + weights.visual * result.visual - 0.2 - weights.bad_source
                                         - weights.format * result.format)


def test_bad_plots_flat_is_zero_undefined_is_error():
    deck = deck_of(["#TITLE: Тема"], ["#TITLE: А", "#PLOT: y = sin(x); x=0..0.1"], ["#TITLE: Б", "#PLOT: y = log(x); x=-10..1"])
    lines = [deck.slides[i].elements[0].line for i in (1, 2)]
    report = RenderReport(path=None, items=[ElementReport(2, "PLOT", "flat_plot", lines[0]),
                                            ElementReport(3, "PLOT", "undefined_plot", lines[1])])
    value, counted, _, _, errors, _ = visual_score(deck, report, 2, images_allowed=True, tool_text=None)
    assert counted == {} and errors == 1 and value == pytest.approx(-0.5 / 4)


def test_text_overflow_is_penalized_once_per_slide():
    deck = deck_of(["#TITLE: Матрица", "#TEXT: матрица и определитель", "#TIMELINE: 1905 — а; 1906 — б"])
    points = [KeyPoint("Матрица", "матрица и определитель", ["матрица"])]
    line = deck.slides[0].elements[0].line
    report = RenderReport(path=None, items=[ElementReport(1, "TEXT", "text_overflow", line),
                                            ElementReport(1, "TEXT", "text_overflow", line),
                                            ElementReport(1, "TIMELINE", "text_overflow", line + 1)])
    base = score(deck, points, conclusion=False, slide_range=(1, 1), embedder=EMB, lemmatizer=LEM)
    over = score(deck, points, conclusion=False, slide_range=(1, 1), embedder=EMB, lemmatizer=LEM, report=report)
    assert over.text_overflow == 1  # подписи ленты не считаются, слайд — один раз
    assert "матрица и определитель" in slide_texts(deck, report)[0].content  # текст с переполнением виден


def test_formulas_are_counted_separately_from_visuals():
    # N = 1: визуал — до ⌈0,5⌉ = 1 на тип и 2 всего; формулы — до 2 и по 0,7, в общий предел не входят
    deck = deck_of(["#TITLE: Тема"], ["#TITLE: Слайд", "#TIMELINE: 1905 — а; 1906 — б", "#FORMULA: E = mc^2",
                                      "#FORMULA: p = mv", "#FORMULA: F = ma"], ["#TITLE: Итог"])
    value, counted, _, _, _, _ = visual_score(deck, None, 1, images_allowed=True, tool_text=None)
    assert counted == {"TIMELINE": 1, "FORMULA": 3}
    assert value == pytest.approx((1 + 0.7 * 2 - 0.2) / 2)  # третья формула сверх 2·N — −0,2 и без бонуса


def test_quote_with_ellipsis_and_fake_quote_penalty():
    from presentation_rl.rewards import quote_found, _normalize_quote
    source = _normalize_quote("«Без выражения», — предупреждает он, — «даже величайшая любовь может умереть»")
    assert quote_found("Без выражения… даже величайшая любовь может умереть", source)
    assert not quote_found("даже величайшая любовь… Без выражения", source)  # порядок частей важен
    deck = deck_of(["#TITLE: Тема"], ["#TITLE: А", "#QUOTE: Без выражения… даже величайшая любовь может умереть — Р. Стернберг",
                                      "#QUOTE: Выдуманные слова — Никто"], ["#TITLE: Итог"])
    value, counted, _, _, _, fake = visual_score(deck, None, 1, images_allowed=True, tool_text="«Без выражения», — предупреждает он, — «даже величайшая любовь может умереть»")
    assert counted == {"QUOTE": 1} and fake == 1 and value == pytest.approx((1 - 0.25) / 2)
    _, _, _, _, _, fake = visual_score(deck, None, 1, images_allowed=True, tool_text=None)
    assert fake == 0  # без поиска цитаты ни плюс, ни минус
