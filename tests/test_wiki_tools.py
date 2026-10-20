# Здесь проверяем инструменты search, open, read и сохранение ответов без сети: ответы API подставляются заранее.
import json
import urllib.parse

import pytest

from presentation_rl.tools.cache import CacheMiss, WikiCache
from presentation_rl.tools.wiki import WikiClient
from presentation_rl.tools.wiki_tools import WikiTools, _fit_cap

LEAD = "Общая идея метода: двигаться против градиента. " * 10  # подводка раздела с подразделами, ~480 символов
ARTICLE = f"""Градиентный спуск — метод нахождения локального минимума функции.

== Описание ==
{LEAD}

=== Выбор шага ===
Шаг подбирают так, чтобы значение функции убывало.

== Пустой раздел ==

=== Только подраздел ===
Текст подраздела.

== Длинный раздел ==
{"Первое предложение длинного абзаца. " * 40}
{"Второй абзац. " * 40}

== Литература ==
Книги.
"""


def page(index, title, pageid, extract, link="", disambiguation=False, lang="ru"):
    data = {"index": index, "title": title, "pageid": pageid, "lastrevid": pageid * 10,
            "fullurl": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}", "extract": extract}
    if link:
        data["langlinks"] = [{"lang": "en", "title": link}]
    if disambiguation:
        data["pageprops"] = {"disambiguation": ""}
    return data


RESPONSES = {
    ("search", "ru", "градиентный спуск"): {"query": {"pages": [
        page(2, "Метод Ньютона", 2, "Метод Ньютона — итерационный метод.", "Newton's method"),
        page(3, "Спуск", 3, "Спуск может означать:", disambiguation=True),
        page(1, "Градиентный спуск", 1, "Градиентный спуск — метод нахождения минимума. " * 10, "Gradient descent"),
    ]}},
    ("search", "en", "gradient descent"): {"query": {"pages": [
        page(1, "Gradient descent", 11, "Gradient descent is a method.", lang="en"),
    ]}},
    ("titles", "en", "Gradient descent|Newton's method"): {"query": {"pages": [
        page(0, "Gradient descent", 11, "Gradient descent is a method.", lang="en"),
        page(0, "Newton's method", 12, "Newton's method is an algorithm.", lang="en"),
    ]}},
    ("titles", "ru", ""): {"query": {}},
    ("article", "ru", "Градиентный спуск"): {"query": {"pages": [{
        "pageid": 1, "title": "Градиентный спуск", "revisions": [{"revid": 15}],
        "fullurl": "https://ru.wikipedia.org/wiki/Градиентный_спуск", "extract": ARTICLE,
    }]}},
}


class FakeApi:
    def __init__(self):
        self.calls = 0

    def __call__(self, url, headers):
        self.calls += 1
        lang = url.split("//")[1].split(".")[0]
        params = dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))
        if params.get("action") == "parse":  # HTML статьи для таблиц: в этих тестах таблиц нет
            return {"parse": {"text": ""}}
        if params.get("generator") == "search":
            key = ("search", lang, params["gsrsearch"])
        elif "exsectionformat" in params:
            key = ("article", lang, params["titles"])
        else:
            key = ("titles", lang, params["titles"])
        return json.loads(json.dumps(RESPONSES.get(key, {"query": {}})))


@pytest.fixture
def setup(tmp_path):
    api = FakeApi()
    cache = WikiCache(tmp_path, WikiClient(fetch=api, min_interval=0))
    return WikiTools(cache), api, tmp_path


def test_search_orders_skips_disambiguation_and_adds_other_language(setup):
    tools, _, _ = setup
    lines = tools.search("градиентный спуск").splitlines()
    assert lines[1].startswith("d1 | ru | Градиентный спуск | Градиентный спуск — метод")
    assert lines[2].startswith("d2 | ru | Метод Ньютона")
    assert lines[3].startswith("d3 | en | Gradient descent")
    assert lines[4].startswith("d4 | en | Newton's method")
    assert len(lines) == 5  # страница значений «Спуск» пропущена
    assert lines[1].endswith("…") and len(lines[1].split(" | ")[3]) <= 201


def test_latin_query_goes_to_english_and_ids_are_reused(setup):
    tools, _, _ = setup
    tools.search("градиентный спуск")
    out = tools.search("gradient descent")
    assert out.splitlines()[1].startswith("d3 | en | Gradient descent")  # та же статья — тот же id


def test_saved_answers_are_reused_without_network(setup):
    tools, api, folder = setup
    first = tools.search("градиентный спуск") + tools.open("d1")
    calls = api.calls
    tools.reset()
    assert tools.search("градиентный спуск") + tools.open("d1") == first
    assert api.calls == calls
    offline = WikiTools(WikiCache(folder))  # без клиента: только сохранённое
    assert offline.search("градиентный спуск") + offline.open("d1") == first
    with pytest.raises(CacheMiss):
        offline.search("что-то новое")


def test_open_shows_intro_toc_and_leads(setup):
    tools, _, _ = setup
    tools.search("градиентный спуск")
    out = tools.open("d1")
    assert out.splitlines()[0] == "d1 | ru | Градиентный спуск"
    assert "d1#0 Введение\nГрадиентный спуск — метод нахождения локального минимума функции." in out
    assert "d1#1 Описание\n" + LEAD.strip() in out
    assert "d1#2   Выбор шага\nШаг" not in out  # текст конечного раздела — только через read
    assert "d1#3 Пустой раздел\nd1#4   Только подраздел" in out
    assert "Литература" not in out


def test_open_cuts_only_longest_texts_to_fit_limit(setup):
    tools, _, _ = setup
    tools.limit = 400
    tools.search("градиентный спуск")
    out = tools.open("d1")
    assert len(out) <= 400
    assert "Градиентный спуск — метод нахождения локального минимума функции." in out  # короткое введение целиком
    assert "[дальше: read d1#1]" in out


def test_read_section_truncation_and_empty_parent(setup):
    tools, _, _ = setup
    tools.limit = 1000
    tools.search("градиентный спуск")
    assert tools.read("d1#2").endswith("Шаг подбирают так, чтобы значение функции убывало.")
    long = tools.read("d1#5")
    assert "[обрезано: ещё" in long and len(long) < 1150
    assert "Подразделы: d1#4 Только подраздел." in tools.read("d1#3")
    assert "нет раздела #9" in tools.read("d1#9")


def test_sources_need_search_or_open(setup):
    tools, _, _ = setup
    tools.search("градиентный спуск")
    assert tools.valid_source("d1") and not tools.valid_source("d1#2") and not tools.valid_source("d7")
    tools.open("d1")
    assert tools.valid_source("d1#2")
    source = tools.source("d1#2")
    assert source["section"] == "Выбор шага" and source["revid"] == 15
    assert source["url"].endswith("#" + urllib.parse.quote("Выбор_шага"))
    assert tools.source("d2")["title"] == "Метод Ньютона"


def test_call_limit_and_errors(setup):
    tools, _, _ = setup
    tools.max_calls = 2
    assert "Статьи d1 в этом поиске не было" in tools.call("open", "d1")
    assert "Неверный id" in tools.call("read", "раздел 3")
    assert "Лимит вызовов исчерпан" in tools.call("search", "градиентный спуск")
    assert len(tools.calls) == 2


def test_fit_cap():
    assert _fit_cap([100, 200], 400) is None
    assert _fit_cap([100, 1000, 1000], 1100) == 500
