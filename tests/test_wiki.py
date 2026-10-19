# Здесь проверяем клиент Википедии без сети: ответы API подставляются заранее.
import json
import urllib.parse

import pytest

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


def test_clean_math_keeps_latex_inline():
    from presentation_rl.tools.wiki import clean_math

    text = ("Пусть функция имеет вид:\n\n  \n    \n      F\n      (\n      x\n      )\n    \n"
            "    {\\displaystyle F(x)\\to \\min }\n  \n.\nВместо \n  \n    F\n    {\\displaystyle F}\n  \n берут \n"
            "  \n    {\\displaystyle -F}\n  \n\nДалее.")
    assert clean_math(text) == "Пусть функция имеет вид: $F(x)\\to \\min$.\nВместо $F$ берут $-F$\nДалее."
    assert clean_math("Без формул.\n  отступ") == "Без формул.\n  отступ"


def test_client_retries_and_records_failures():
    import urllib.error

    answers = [urllib.error.URLError("timed out"), {"query": {"search": []}}]

    def fetch(url, headers):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    client = WikiClient(fetch=fetch, min_interval=0)
    import presentation_rl.tools.wiki as wiki
    sleep, wiki.time.sleep = wiki.time.sleep, lambda s: None
    try:
        assert client.search("ru", "спуск") == []
    finally:
        wiki.time.sleep = sleep
    assert client.requests == 2 and client.failures == ["сеть: timed out"]
