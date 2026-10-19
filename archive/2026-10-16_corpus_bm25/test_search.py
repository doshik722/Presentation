from presentation_rl.tools.search import Analyzer, SectionIndex, Weights
from presentation_rl.tools.wiki import Article, Section


def make(title, sections, pageid):
    return Article("ru", pageid, title, 1, "", [Section(i, 1, h, t) for i, (h, t) in enumerate(sections)])


ARTICLES = [
    make("Градиентный спуск", [("", "Градиентный спуск — метод нахождения минимума функции."),
                               ("Выбор шага", "Шаг обучения определяет скорость сходимости метода.")], 1),
    make("Метод Ньютона", [("", "Итерационный метод нахождения корня функции."),
                           ("Начальное приближение", "Метод сходится, если начальное приближение близко к корню.")], 2),
    make("Строевой шаг", [("", "Строевой шаг — элемент строевой подготовки, шаг с вытянутой ногой.")], 3),
]


def test_analyzer_stems_and_drops_stop_words():
    assert Analyzer("ru")("Градиентного спуска и ёлки") == ["градиентн", "спуск", "елк"]


def test_title_beats_text_and_one_hit_per_article():
    index = SectionIndex(ARTICLES, "ru")
    hits = index.search("градиентный спуск", k=3)
    assert hits[0].article.title == "Градиентный спуск"
    assert len({h.article.key for h in hits}) == len(hits)


def test_heading_points_to_section():
    index = SectionIndex(ARTICLES, "ru")
    hit = index.search("начальное приближение", k=1)[0]
    assert (hit.article.title, hit.section) == ("Метод Ньютона", 1)


def test_article_field_lifts_related_sections():
    index = SectionIndex(ARTICLES, "ru")
    # «шаг обучения»: без поля «вся статья» строевой шаг конкурирует только по слову «шаг»
    hits = index.search("шаг обучения метод", k=2, weights=Weights(article=1.0))
    assert hits[0].article.title == "Градиентный спуск"


def test_other_language_and_duplicates_skipped():
    en = Article("en", 9, "Gradient descent", 1, "", [Section(0, 1, "", "Gradient descent is a method.")])
    index = SectionIndex(ARTICLES + [ARTICLES[0], en], "ru")
    assert len(index.articles) == 3
