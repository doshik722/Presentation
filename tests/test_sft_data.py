# Здесь проверяем сборку примеров для SFT: деление ответа по id, исправления, проверки и ожидание пакетов.
import json

from presentation_rl.sft_data import collect, split_answer

GOOD = "#PLAN: а → б\n#SLIDE\n#TITLE: Тема\n#TEXT: Текст\n#DIAGRAM: type=flow; А -> Б"


def test_split_answer_handles_both_id_styles_and_fences():
    parts = split_answer("### t001\n```\n#SLIDE\n#TITLE: A\n```\n\nt002\n#SLIDE\n#TITLE: B")
    assert parts == {"t001": "#SLIDE\n#TITLE: A", "t002": "#SLIDE\n#TITLE: B"}


def test_text_outside_code_block_is_a_comment():
    parts = split_answer("### t001\n```\n#SLIDE\n#TITLE: A\n```\nСверено с [MIT](https://x)\n", with_comments=True)
    assert parts == {"t001": ("#SLIDE\n#TITLE: A", "Сверено с [MIT](https://x)")}


def _tasks(tmp_path):
    records = [
        {"id": i, "split": "sft", "source": "syllabus", "domain": "technical", "scenario": "учёба",
         "family": f"f{i}", "topic": f"Тема {i}", "request": None, "audience": None, "slides": None}
        for i in ("t001", "t002", "t003")
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": records}, ensure_ascii=False), encoding="utf-8")
    return path


def test_collect_applies_fixes_and_reports_problems(tmp_path):
    sft = tmp_path / "sft"
    for m in ("claude", "codex", "gemini"):
        (sft / m).mkdir(parents=True)
    plan = {"claude": {"batch1": ["t001", "t002"]}, "codex": {"batch1": ["t003"]}, "gemini": {"batch1": []}}
    (sft / "assignment.json").write_text(json.dumps(plan), encoding="utf-8")
    (sft / "offline.json").write_text(json.dumps(["t002"]), encoding="utf-8")
    with_image = GOOD + "\n#IMAGE: фото"
    (sft / "claude" / "answer1.txt").write_text(f"### t001\n{GOOD}\n### t002\n{with_image}", encoding="utf-8")
    (sft / "claude" / "answer1_fix.txt").write_text(f"### t002\n{GOOD}", encoding="utf-8")
    report = collect(sft, _tasks(tmp_path))
    assert [e.id for e in report.examples] == ["t001", "t002"]
    t002 = report.examples[1]
    assert t002.fixed and "#IMAGE" not in t002.answer
    assert "Картинки: нет" in t002.request and "Поиск: нет" in t002.request
    assert report.waiting == ["codex/batch1"] and not report.errors


def test_collect_rejects_forbidden_image_and_missing_task(tmp_path):
    sft = tmp_path / "sft"
    for m in ("claude", "codex", "gemini"):
        (sft / m).mkdir(parents=True)
    plan = {"claude": {"batch1": ["t001", "t002"]}, "codex": {"batch1": []}, "gemini": {"batch1": []}}
    (sft / "assignment.json").write_text(json.dumps(plan), encoding="utf-8")
    (sft / "offline.json").write_text(json.dumps(["t001"]), encoding="utf-8")
    (sft / "claude" / "answer1.txt").write_text(f"### t001\n{GOOD}\n#IMAGE: фото", encoding="utf-8")
    report = collect(sft, _tasks(tmp_path))
    assert not report.examples
    assert any("#IMAGE при «Картинки: нет»" in e for e in report.errors)
    assert any("t002: нет презентации" in e for e in report.errors)


def test_collect_rejects_formula_that_does_not_render(tmp_path):
    sft = tmp_path / "sft"
    for m in ("claude", "codex", "gemini"):
        (sft / m).mkdir(parents=True)
    plan = {"claude": {"batch1": ["t001"]}, "codex": {"batch1": []}, "gemini": {"batch1": []}}
    (sft / "assignment.json").write_text(json.dumps(plan), encoding="utf-8")
    (sft / "offline.json").write_text("[]", encoding="utf-8")
    (sft / "claude" / "answer1.txt").write_text(f"### t001\n{GOOD}\n#FORMULA: \\begin{{pmatrix}} 1 \\end{{pmatrix}}", encoding="utf-8")
    report = collect(sft, _tasks(tmp_path))
    assert not report.examples and any("формула не рисуется" in e for e in report.errors)
