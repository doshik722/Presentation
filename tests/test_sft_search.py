# Здесь проверяем сборку примеров SFT с поиском: строение эпизода, id выдачи, источники, цитаты и сборку эпизода средой.
import json
from pathlib import Path

from presentation_rl.environment import NEXT_TURN
from presentation_rl.sft_search import collect, replay

from test_environment import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data" / "tasks" / "tasks.json"

GOOD = """[модель]
#PLAN: история → свойства → выводы
#PLAN: ищу определение и историю
<tool_call>
{"name": "search", "arguments": {"query": "определитель"}}
</tool_call>
[программа]
Результаты поиска «определитель»:
d1 | ru | Определитель | Определитель — скалярная величина, которая характеризует квадратную матрицу…
d2 | en | Determinant | In mathematics, the determinant is a scalar-valued function…
[модель]
<tool_call>
{"name": "open", "arguments": {"id": "d1"}}
</tool_call>
[программа]
d1 | ru | Определитель
d1#0 Введение
Определитель — скалярная величина, которая характеризует квадратную матрицу.
d1#1 История
d1#2 Свойства
[модель]
<tool_call>
{"name": "read", "arguments": {"id": "d1#1"}}
</tool_call>
[программа]
d1#1 | ru | Определитель — История
Впервые определители встречаются у Лейбница в 1693 году. Габриэль Крамер в 1750 году дал правило решения систем: «Определитель есть сумма произведений».
[модель]
#SLIDE
#TITLE: Определители
#TEXT: О том, откуда взялись определители и зачем они нужны.
#SLIDE
#TITLE: История
#TIMELINE: 1693 — Лейбниц; 1750 — правило Крамера
#QUOTE: Определитель есть сумма произведений — Крамер, «Введение в анализ»
#SOURCE: d1#1
#SLIDE
#TITLE: Свойства
#FORMULA: \\det(AB) = \\det A \\cdot \\det B
#SLIDE
#TITLE: Выводы
#TEXT: Определитель отвечает на вопрос, есть ли у системы единственное решение.
"""


def make_dir(tmp_path, episode, fix=None):
    folder = tmp_path / "sft_search"
    (folder / "claude").mkdir(parents=True)
    (folder / "assignment.json").write_text(json.dumps({"claude": {"batch1": ["t007"]}}), encoding="utf-8")
    (folder / "claude" / "answer1.txt").write_text("### t007\n```\n" + episode + "```\n", encoding="utf-8")
    offline = tmp_path / "offline.json"
    offline.write_text(json.dumps(["t007"]), encoding="utf-8")
    return collect(folder, TASKS, offline)


def test_good_episode_passes_and_replays_through_environment(tmp_path):
    report = make_dir(tmp_path, GOOD)
    assert report.errors == [] and len(report.examples) == 1
    assert not [n for n in report.notes if "чисел нет" in n]
    example = report.examples[0]
    assert example.answer.startswith("#PLAN: история") and "Поиск: есть" in example.request
    tok = CharTokenizer()
    episode = replay(example, tok, tok.encode("запрос" + NEXT_TURN))
    assert [c["name"] for c in episode.calls] == ["search", "open", "read"]
    assert [s.trainable for s in episode.segments] == [True, False, True, False, True, False, True]
    assert "Лейбница в 1693 году" in episode.segments[5].text and episode.answer == example.answer


def test_wrong_ids_sources_quotes_and_counts_are_errors(tmp_path):
    bad = (GOOD.replace("d2 | en | Determinant", "d3 | en | Determinant")  # id не по порядку
               .replace("#SOURCE: d1#1", "#SOURCE: d1#1, d5")  # такого id не было
               .replace("Определитель есть сумма произведений — Крамер", "Определитель придумал я — Крамер")
               .replace('{"name": "read", "arguments": {"id": "d1#1"}}', '{"name": "read", "arguments": {"id": "d1#7"}}'))
    report = make_dir(tmp_path, bad)
    text = "\n".join(report.errors)
    assert report.examples == []
    assert "программа дала бы d2" in text
    assert "d5: такого id в выдаче не было" in text
    assert "цитаты нет в выдаче" in text
    assert "read d1#7 — такого раздела в оглавлении нет" in text


def test_missing_program_answer_and_numbers_not_in_source(tmp_path):
    report = make_dir(tmp_path, GOOD.replace("1750 — правило Крамера", "1754 — правило Крамера"))
    assert any("чисел нет в источнике: 1754" in n for n in report.notes) and report.errors == []
    broken = GOOD.replace("[программа]\nd1#1 | ru", "d1#1 | ru", 1)  # ответ программы слился с репликой
    report = make_dir(tmp_path / "b", broken)
    assert any("вызовов 1, ответов программы 0" in e for e in report.errors)


def test_markdown_links_are_removed_with_a_note(tmp_path):
    linked = (GOOD.replace("#PLAN: ищу определение и историю", "#PLAN: ищу [определитель](https://ru.wikipedia.org/wiki/Определитель_(математика)) и историю")
                  .replace("Результаты поиска «определитель»:", "Результаты поиска «определитель» ([Википедия](https://ru.wikipedia.org/wiki/Определитель)):"))
    report = make_dir(tmp_path, linked)
    assert report.errors == [] and any("убраны ссылки Markdown — 2" in n for n in report.notes)
    assert "#PLAN: ищу определитель и историю" in report.examples[0].answer
    assert "английская статья открыта или прочитана в 0 из 1" in report.text()


def test_glued_program_answers_are_split_by_their_first_lines(tmp_path):
    two_calls = GOOD.replace(
        '{"name": "read", "arguments": {"id": "d1#1"}}\n</tool_call>\n',
        '{"name": "read", "arguments": {"id": "d1#1"}}\n</tool_call>\n<tool_call>\n'
        '{"name": "search", "arguments": {"query": "правило Крамера"}}\n</tool_call>\n')
    two_calls = two_calls.replace(
        "«Определитель есть сумма произведений».\n",
        "«Определитель есть сумма произведений».\n\nРезультаты поиска «правило Крамера»:\n"
        "d3 | ru | Метод Крамера | Метод Крамера — способ решения квадратных систем линейных уравнений…\n")
    report = make_dir(tmp_path, two_calls)
    assert report.errors == [] and any("ответы программы разделены (1 блока → 2)" in n for n in report.notes)
    example = report.examples[0]
    assert [t.role for t in example.turns].count("program") == 4
    tok = CharTokenizer()
    episode = replay(example, tok, tok.encode("запрос" + NEXT_TURN))
    assert [c["name"] for c in episode.calls] == ["search", "open", "read", "search"]


def test_subsection_indent_before_id_is_moved_after_it(tmp_path):
    indented = GOOD.replace("d1#2 Свойства", "d1#2 Свойства\n  d1#3 Определитель произведения")
    report = make_dir(tmp_path, indented)
    assert report.errors == [] and any("оглавление приведено к виду инструмента (отступы, «Введение») — 1" in n for n in report.notes)
    program = [t.text for t in report.examples[0].turns if t.role == "program"][1]
    assert "\nd1#3   Определитель произведения" in program


TEACHER = """[модель]
#PLAN: идея метода → выбор шага → выводы
#PLAN: ищу градиентный спуск; английская статья d2 — для сравнения
<tool_call>
{"name": "search", "arguments": {"query": "градиентный спуск"}}
</tool_call>
[программа]
Результаты поиска «градиентный спуск»:
d1 | ru | Градиентный спуск | Градиентный спуск — метод нахождения минимума…
d2 | en | Gradient descent | Gradient descent is a method.
[модель]
<tool_call>
{"name": "open", "arguments": {"id": "d1"}}
</tool_call>
[программа]
d1 | ru | Градиентный спуск
d1#0 Введение
Градиентный спуск — метод нахождения… [обрезано: ещё 30 символов]
d1#1 Описание
  d1#2 Выбор шага
[модель]
<tool_call>
{"name": "read", "arguments": {"id": "d1#2"}}
</tool_call>
[программа]
d1#2 | ru | Градиентный спуск — Выбор шага
Шаг подбирают так… [обрезано: ещё 30 символов]
[модель]
#SLIDE
#TITLE: Градиентный спуск
#TEXT: Как найти минимум функции, двигаясь против градиента.
#SLIDE
#TITLE: Выбор шага
#TEXT: Шаг подбирают так, чтобы значение функции убывало.
#FORMULA: x_{k+1} = x_k - \\lambda \\nabla f(x_k)
#SOURCE: d1#2, d2
#SLIDE
#TITLE: Идея метода
#TEXT: Двигаться против градиента.
#FORMULA: \\nabla f(x)
#SLIDE
#TITLE: Выводы
#TEXT: Удачный шаг — половина успеха метода.
"""


def real_tools(tmp_path):
    from presentation_rl.tools.cache import WikiCache
    from presentation_rl.tools.wiki import WikiClient
    from presentation_rl.tools.wiki_tools import WikiTools
    from test_wiki_tools import FakeApi
    return WikiTools(WikiCache(tmp_path / "cache", WikiClient(fetch=FakeApi(), min_interval=0)))


def test_real_output_is_substituted_and_ids_are_renumbered(tmp_path):
    from presentation_rl.sft_search import substitute
    from presentation_rl.tasks import load_tasks
    report = make_dir(tmp_path, TEACHER)
    assert report.errors == []
    task = {t.id: t for t in load_tasks(TASKS, split="sft")}["t007"]
    example, problems, notes = substitute(report.examples[0], real_tools(tmp_path), task)
    assert problems == [] and any("d2→d3" in n for n in notes)
    programs = [t.text for t in example.turns if t.role == "program"]
    assert "d2 | ru | Метод Ньютона" in programs[0]  # настоящая выдача целиком
    assert programs[2].endswith("Шаг подбирают так, чтобы значение функции убывало.")
    assert "#SOURCE: d1#2, d3" in example.answer and "английская статья d3" in example.answer


def test_teacher_text_not_in_real_section_is_rejected(tmp_path):
    from presentation_rl.sft_search import substitute
    from presentation_rl.tasks import load_tasks
    report = make_dir(tmp_path, TEACHER.replace("Шаг подбирают так… [обрезано", "Шаг выбирают наугад… [обрезано"))
    task = {t.id: t for t in load_tasks(TASKS, split="sft")}["t007"]
    example, problems, _ = substitute(report.examples[0], real_tools(tmp_path), task)
    assert example is None and any("текста составителя нет в настоящем разделе" in p for p in problems)


def test_glued_fragment_from_elsewhere_is_rejected(tmp_path):
    from presentation_rl.sft_search import substitute
    from presentation_rl.tasks import load_tasks
    glued = TEACHER.replace("Шаг подбирают так… [обрезано", "Шаг подбирают так… Общая идея метода: двигаться против градиента. [обрезано")
    report = make_dir(tmp_path, glued)
    task = {t.id: t for t in load_tasks(TASKS, split="sft")}["t007"]
    example, problems, _ = substitute(report.examples[0], real_tools(tmp_path), task)
    assert example is None and any("Общая идея метода" in p for p in problems)
