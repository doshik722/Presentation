# Здесь проверяем таблицы и карточку статьи из HTML Википедии и их место в разделах статьи — без сети.
import json
import urllib.parse

from presentation_rl.tools.wiki import WikiClient, split_sections
from presentation_rl.tools.wiki_tables import align_headings, html_tables

# Разметка как у action=parse: заголовки в новом (mw-heading) и старом (mw-headline) виде, сноски, ключи сортировки,
# объединённые ячейки, формула, карточка, навигационная таблица внизу.
HTML = """<div class="mw-content-ltr mw-parser-output" lang="ru">
<table class="infobox infobox-0ab1" data-name="Карточка ГЭС">
<tbody><tr><th colspan="2" class="infobox-above">Саяно-Шушенская ГЭС</th></tr>
<tr><td colspan="2"><span typeof="mw:File"><img src="x.jpg" alt=""></span><br>Вид на плотину</td></tr>
<tr><th>Страна</th><td><span class="flagicon"><img src="f.png"></span> Россия</td></tr>
<tr><th>Река</th><td><a href="/wiki/Енисей">Енисей</a></td></tr>
<tr><th>Мощность</th><td>6400 МВт<sup class="reference"><a href="#cite_note-1">[1]</a></sup></td></tr>
</tbody></table>
<p>Саяно-Шушенская ГЭС — крупнейшая электростанция России.</p>
<div class="mw-heading mw-heading2"><h2 id="История">История</h2><span class="mw-editsection">[<a>править</a>]</span></div>
<p>Строительство началось в 1963 году.</p>
<div class="mw-heading mw-heading3"><h3 id="Ввод">Ввод в эксплуатацию</h3></div>
<table class="wikitable sortable">
<caption>Гидроагрегаты</caption>
<tbody><tr><th>Агрегат</th><th>Год пуска</th><th>Мощность, МВт</th></tr>
<tr><td>1</td><td rowspan="2">1978</td><td><span class="sortkey" style="display:none">0640</span>640</td></tr>
<tr><td>2</td><td>640</td></tr>
<tr><td colspan="2">Итого</td><td>1280</td></tr>
</tbody></table>
<h2><span class="mw-headline" id="Формула">Расчёт</span><span class="mw-editsection">[править]</span></h2>
<table class="wikitable"><tr><th>Величина</th><th>Формула</th></tr>
<tr><td>Мощность</td><td><span class="mwe-math-element"><math><semantics><mrow><mi>P</mi></mrow>
<annotation encoding="application/x-tex">{\\displaystyle P=\\rho gQH}</annotation></semantics></math>
<img class="mwe-math-fallback-image-inline" alt="P=\\rho gQH"></span></td></tr></table>
<div class="mw-heading mw-heading2"><h2 id="Примечания">Примечания</h2></div>
<table class="wikitable"><tr><td>служебная таблица</td><td>x</td></tr></table>
<div class="navbox"><table class="nowraplinks"><tr><td>ГЭС России</td></tr></table></div>
</div>"""

EXTRACT = """Саяно-Шушенская ГЭС — крупнейшая электростанция России.

== История ==
Строительство началось в 1963 году.

=== Ввод в эксплуатацию ===

== Расчёт ==

== Примечания ==
1. Сноска.
"""


def test_html_tables_text():
    found = html_tables(HTML)
    assert found.headings == ["История", "Ввод в эксплуатацию", "Расчёт", "Примечания"]
    (n0, card), (n1, units), (n2, formula), (n3, _) = found.tables
    assert (n0, n1, n2, n3) == (0, 2, 3, 4)
    assert card == ("Карточка статьи:\nСаяно-Шушенская ГЭС\nВид на плотину\nСтрана: Россия\nРека: Енисей\n"
                    "Мощность: 6400 МВт")
    assert units == ("Таблица: Гидроагрегаты\nАгрегат | Год пуска | Мощность, МВт\n1 | 1978 | 640\n2 | 1978 | 640\n"
                     "Итого | 1280")
    assert formula.splitlines()[-1] == "Мощность | $P=\\rho gQH$"


def test_align_headings_by_order_and_name():
    assert align_headings(["Содержание", "История", "Итоги", "История"], ["История", "Итоги", "Ссылки", "История"]) \
        == [0, 1, 2, 4]


def test_tables_go_to_their_sections_and_service_sections_are_dropped():
    found = html_tables(HTML)
    tables = {}
    numbers = align_headings(found.headings, ["История", "Ввод в эксплуатацию", "Расчёт", "Примечания"])
    for count, table in found.tables:
        tables.setdefault(numbers[count - 1] if count else 0, []).append(table)
    sections = split_sections(EXTRACT, tables)
    assert [s.heading for s in sections] == ["", "История", "Ввод в эксплуатацию", "Расчёт"]
    assert sections[0].text.startswith("Саяно-Шушенская ГЭС — крупнейшая") and "Мощность: 6400 МВт" in sections[0].text
    assert sections[2].text.startswith("Таблица: Гидроагрегаты")  # раздел из одной таблицы теперь не пустой
    assert "служебная таблица" not in json.dumps([s.text for s in sections], ensure_ascii=False)


def test_last_section_before_several_service_sections_is_kept():
    text = "Введение.\n\n== История ==\nВажный текст.\n\n== Примечания ==\n1.\n\n== Ссылки ==\nсайт\n"
    assert [(s.heading, s.text) for s in split_sections(text)] == [("", "Введение."), ("История", "Важный текст.")]


def test_client_article_adds_tables_from_same_revision():
    calls = []

    def fetch(url, headers):
        params = dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))
        calls.append(params)
        if params.get("action") == "parse":
            return {"parse": {"text": HTML}}
        return {"query": {"pages": [{"pageid": 5, "title": "Саяно-Шушенская ГЭС", "fullurl": "https://w/x",
                                     "revisions": [{"revid": 77}], "extract": EXTRACT}]}}

    article = WikiClient(fetch=fetch, min_interval=0).article("ru", "Саяно-Шушенская ГЭС")
    assert calls[1]["action"] == "parse" and calls[1]["oldid"] == "77"
    assert "1 | 1978 | 640" in article.sections[2].text
    assert "служебная таблица" not in json.dumps([s.text for s in article.sections], ensure_ascii=False)


def test_service_section_tables_dropped_even_if_headings_do_not_match():
    def fetch(url, headers):
        if "action=parse" in url:
            return {"parse": {"text": HTML}}
        return {"query": {"pages": [{"pageid": 5, "title": "X", "revisions": [{"revid": 1}], "extract": "Текст."}]}}

    article = WikiClient(fetch=fetch, min_interval=0).article("ru", "X")
    assert "служебная таблица" not in article.sections[0].text and "Гидроагрегаты" in article.sections[0].text


def test_several_values_in_one_cell_are_separated():
    html = """<table class="wikitable"><tr><th>Год</th><th>Лауреаты</th></tr>
<tr><td>1958</td><td><div class="plainlist"><ul><li><a>Павел Черенков</a></li><li><a>Игорь Тамм</a></li></ul></div></td></tr>
<tr><td>1964</td><td><p>Николай Басов</p><p>Александр Прохоров</p></td></tr>
<tr><td>1975</td><td><a>Андрей Сахаров</a><br><small>премия мира</small></td></tr></table>"""
    (_, text), = html_tables(html).tables
    assert text.splitlines()[1:] == ["Год | Лауреаты", "1958 | Павел Черенков; Игорь Тамм",
                                     "1964 | Николай Басов; Александр Прохоров", "1975 | Андрей Сахаров; премия мира"]
