# Здесь проверяем сборку тезисов от составителей: ошибки формата, замечания, исправления, источники для test.
import json

from presentation_rl.key_points import collect, expected_theses, read_answer
from presentation_rl.tasks import PresentationRequest, Task


def task(tid, split="train", slides=None):
    return Task(tid, split, "syllabus", PresentationRequest(f"Тема {tid}", slides=slides), f"Тема {tid}",
                "technical", "study", f"f{tid}")


def point(title, words=6, keys=6):
    return {"title": title, "statement": " ".join(["слово"] * words), "key": [f"ключ {i}" for i in range(keys)]}


def write(folder, plan, answers):
    (folder / "assignment.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    for name, text in answers.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_read_answer_accepts_code_fence_and_notes_text():
    data, note = read_answer('Вот ответ:\n```json\n{"topics": []}\n```')
    assert data == {"topics": []} and note
    assert read_answer("не json") == (None, "")
    wrapped, _ = read_answer('{"topics": [{"key": ["углеродный\nслед"]}]}')  # перенос строки внутри значения
    assert wrapped == {"topics": [{"key": ["углеродный след"]}]}


def test_expected_theses():
    assert expected_theses(task("t001", slides=1)) == (1, 1, False)
    assert expected_theses(task("t001", slides=10)) == (4, 9, True)
    assert expected_theses(task("t001")) == (1, 13, True)


def test_collect_errors_notes_fix_and_sources(tmp_path):
    tasks = {"t001": task("t001", slides=5), "t002": task("t002", split="test"), "t003": task("t003"),
             "t004": task("t004")}
    good = [point("Часть 1"), point("Часть 2"), point("Часть 3", words=30), point("Вывод: главное")]
    plan = {"claude": {"train01": ["t001", "t003"], "test01": ["t002"]}, "codex": {"train01": ["t004"]}}
    answers = {
        "claude/train01_answer.json": json.dumps({"topics": [
            {"id": "t001", "key_points": good},
            {"id": "t003", "key_points": [{"title": "Без ключей", "statement": "Есть"}]},
            {"id": "t999", "key_points": good},
        ]}, ensure_ascii=False),
        "claude/train01_fix.json": json.dumps({"topics": [{"id": "t003", "key_points": good}]}, ensure_ascii=False),
        "claude/test01_answer.json": json.dumps({"topics": [{"id": "t002", "key_points": good}]}, ensure_ascii=False),
    }
    write(tmp_path, plan, answers)
    report = collect(tmp_path, tasks)
    assert [t.id for t in report.topics] == ["t001", "t003"]
    assert next(t for t in report.topics if t.id == "t003").fixed
    assert any("лишний id t999" in e for e in report.errors)
    assert any("t002: нет источников" in e for e in report.errors)
    assert report.waiting == ["codex/train01"]
    assert any("statement длиннее 25 слов" in n for n in report.notes)


def test_links_are_removed_with_note():
    from presentation_rl.key_points import Report, check_topic
    linked = point("Часть 1")
    linked["statement"] += " [Источник](https://example.org/a_(b)) https://example.org/c"
    linked["key"][0] += " https://example.org"
    report = Report()
    points = check_topic({"key_points": [linked, point("Часть 2")]}, task("t001", slides=3), "w", report)
    assert points[0].statement == " ".join(["слово"] * 6) and points[0].key[0] == "ключ 0"
    assert any("убраны ссылки из тезисов 1" in n for n in report.notes)


def test_identifiers_and_ukrainian_conclusion():
    from presentation_rl.key_points import Report, check_topic
    code = point("Параметры DAG")
    code["key"] = ["dag_id", "start_date", "x_1", "R^n", "оператор"]
    report = Report()
    check_topic({"key_points": [code, point("Частина 2"), point("Частина 3"), point("Висновок: головне")]},
                task("t001", slides=5), "w", report)
    assert [n for n in report.notes if "формулу" in n] == ["w: тезис 1 — key похож на формулу: ['x_1', 'R^n']"]
    assert not any("заключение" in n for n in report.notes)
