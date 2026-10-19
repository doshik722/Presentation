# Архив: тесты клиента Википедии и сборщика корпуса (tools/corpus.py) на 16.10.2026; в проекте не запускаются.
# Здесь проверяем клиент Википедии и сборку корпуса без сети: ответы API подставляются заранее.
import json
import urllib.parse

import pytest

from presentation_rl.tools.corpus import CorpusStore, TopicPlan, collect_topic
from presentation_rl.tools.wiki import WikiClient, split_sections

ARTICLE_TEXT = """Градиентный спуск — метод нахождения минимума.

== Описание ==
Шаг против градиента.

=== Выбор шага ===
Шаг подбирают.

== Пустой раздел ==

== Примечания ==
1. Сноска.

=== Подраздел примечаний ===
Тоже служебное.

== История ==
Коши, 1847.
"""


def test_split_sections_drops_service_and_empty_sections():
    sections = split_sections(ARTICLE_TEXT)
    assert [(s.index, s.level, s.heading) for s in sections] == [
        (0, 1, ""), (1, 1, "Описание"), (2, 2, "Выбор шага"), (3, 1, "История"),
    ]
    assert sections[0].text == "Градиентный спуск — метод нахождения минимума."
    assert sections[3].text == "Коши, 1847."


class FakeApi:
    """Отвечает по (язык, action/list/prop, главный параметр); записывает запросы."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, url, headers):
        lang = url.split("//")[1].split(".")[0]
        params = dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))
        self.calls.append((lang, params))
        if params.get("list") == "search":
            key = ("search", lang, params["srsearch"])
        elif params.get("action") == "parse":
            key = ("parse", lang, params["page"])
        elif params.get("prop") == "langlinks":
            key = ("langlinks", lang, params["titles"])
        else:
            key = ("article", lang, params["titles"])
        return json.loads(json.dumps(self.responses.get(key, {"query": {}})))


def _search(*titles):
    return {"query": {"search": [{"title": t, "pageid": i + 1, "sectiontitle": "", "snippet": "<b>x</b>"} for i, t in enumerate(titles)]}}


def _article(pageid, title, text="Текст.", disambiguation=False):
    page = {"pageid": pageid, "title": title, "fullurl": f"https://w/{title}", "revisions": [{"revid": 7}], "extract": text}
    if disambiguation:
        page["pageprops"] = {"disambiguation": ""}
    return {"query": {"pages": [page]}}


def test_client_parses_search_langlinks_and_skips_disambiguation():
    api = FakeApi({
        ("search", "ru", "спуск"): _search("Градиентный спуск"),
        ("langlinks", "ru", "Градиентный спуск|ГС"): {"query": {
            "redirects": [{"from": "ГС", "to": "Градиентный спуск"}],
            "pages": [{"title": "Градиентный спуск", "langlinks": [{"lang": "en", "title": "Gradient descent"}]}]}},
        ("article", "ru", "Спуск"): _article(5, "Спуск", disambiguation=True),
    })
    client = WikiClient(fetch=api, min_interval=0)
    hits = client.search("ru", "спуск")
    assert hits[0].title == "Градиентный спуск" and hits[0].snippet == "x"
    assert client.langlinks("ru", ["Градиентный спуск", "ГС"], "en") == {
        "Градиентный спуск": "Gradient descent", "ГС": "Gradient descent"}
    assert client.article("ru", "Спуск") is None
    assert all(p["maxlag"] == "5" and p["format"] == "json" for _, p in api.calls)


def test_client_needs_contact_for_real_requests(monkeypatch):
    monkeypatch.delenv("WIKI_CONTACT", raising=False)
    with pytest.raises(ValueError):
        WikiClient()


def test_collect_topic_and_resume(tmp_path):
    api = FakeApi({
        ("search", "ru", "Градиентный спуск"): _search("Градиентный спуск", "Метод Ньютона"),
        ("search", "ru", "шаг обучения"): _search("Скорость обучения"),
        ("parse", "ru", "Градиентный спуск"): {"parse": {"links": [{"ns": 0, "title": "Градиент", "exists": True}, {"ns": 14, "title": "Категория:X"}]}},
        ("parse", "ru", "Метод Ньютона"): {"parse": {"links": [{"ns": 0, "title": "Градиент", "exists": True}]}},
        ("langlinks", "ru", "Градиентный спуск|Метод Ньютона|Скорость обучения|Градиент"): {"query": {"pages": [
            {"title": "Градиентный спуск", "langlinks": [{"title": "Gradient descent"}]}]}},
        ("search", "en", "Gradient descent"): _search("Gradient descent", "Stochastic gradient descent"),
        ("article", "ru", "Градиентный спуск"): _article(1, "Градиентный спуск", ARTICLE_TEXT),
        ("article", "ru", "Метод Ньютона"): _article(2, "Метод Ньютона"),
        ("article", "ru", "Скорость обучения"): _article(3, "Скорость обучения"),
        ("article", "ru", "Градиент"): _article(4, "Градиент"),
        ("article", "en", "Gradient descent"): _article(10, "Gradient descent"),
        ("article", "en", "Stochastic gradient descent"): _article(11, "Stochastic gradient descent"),
    })
    client = WikiClient(fetch=api, min_interval=0)
    store = CorpusStore(tmp_path)
    record = collect_topic(client, store, TopicPlan("gd", "Градиентный спуск", ["шаг обучения"]), top_n=2)
    assert record["top_ru"] == ["Градиентный спуск", "Метод Ньютона"]
    assert record["articles_ru"] == ["ru:1", "ru:2", "ru:3", "ru:4"]  # «Градиент» из двух введений — один раз
    assert record["articles_en"] == ["en:10", "en:11"]
    assert record["live_ru"]["шаг обучения"] == [{"title": "Скорость обучения", "section": ""}]
    assert list(record["live_en"]) == ["Gradient descent"]

    # Повторная загрузка: статьи берутся из файла, а не из API.
    reopened = CorpusStore(tmp_path)
    calls_before = len(api.calls)
    assert reopened.get_article(client, "ru", "Метод Ньютона").pageid == 2
    assert len(api.calls) == calls_before
    assert len(reopened.unique_articles()) == 6


def test_clean_math_keeps_latex_inline():
    from presentation_rl.tools.wiki import clean_math

    text = ("Пусть функция имеет вид:\n\n  \n    \n      F\n      (\n      x\n      )\n    \n"
            "    {\\displaystyle F(x)\\to \\min }\n  \n.\nВместо \n  \n    F\n    {\\displaystyle F}\n  \n берут \n"
            "  \n    {\\displaystyle -F}\n  \n\nДалее.")
    assert clean_math(text) == "Пусть функция имеет вид: $F(x)\\to \\min$.\nВместо $F$ берут $-F$\nДалее."
    assert clean_math("Без формул.\n  отступ") == "Без формул.\n  отступ"
