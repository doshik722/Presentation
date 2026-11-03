# Здесь проверяем построение испорченных презентаций: правку текста по слайдам, предусловия, контекст эпизода и пары.
from presentation_rl.dsl import parse
from presentation_rl.key_points import KeyPoint
from presentation_rl.perturb import DOWN, NOT_UP, SAME, Doc, episode_context, generate, paraphrase

ORIGINAL = """#PLAN: тема → первое → второе → третье → итоги
#SLIDE
#TITLE: Тема
#TEXT: О чём пойдёт речь
#SLIDE
#TITLE: Первое
#TEXT: Объяснение первого: событие случилось в 1957 году.
#BULLET: пункт первого
#SOURCE: d1
#SLIDE
#TITLE: Второе
#TEXT: Объяснение второго.
#IMAGE: схема второго
#SLIDE
#TITLE: Третье
#TEXT: Объяснение третьего.
#SLIDE
#TITLE: Заключение
#TEXT: Итоги всей темы.
"""
DONOR = """#SLIDE
#TITLE: Чужая тема
#TEXT: Чужое вступление
#SLIDE
#TITLE: Чужой слайд
#TEXT: Совсем другое содержание.
#SLIDE
#TITLE: Ещё чужой
#TEXT: Другое содержание ещё.
"""
POINTS = [KeyPoint("Первое", "Первое случилось в 1957 году", ["1957"]),
          KeyPoint("Второе", "Второе устроено так", ["второе"]),
          KeyPoint("Третье", "Третье устроено иначе", ["третье"]),
          KeyPoint("Итоги", "Итоги темы", ["итог"])]
ANNOTATION = {"source": "runs/x/search_t1_0", "search": True, "theses": [[2], [3], [4], [5]], "independent": [[3, 4]],
              "numbers": [{"slide": 2, "old": "1957", "new": "1967"}],
              "forms": [{"slide": 2, "old": "в 1957 году", "new": "в 1957 г."}],
              "facts": [{"slide": 3, "kind": "negation", "old": "Объяснение второго.", "new": "Второе не объяснено."}],
              "paraphrases": [{"slide": 4, "texts": ["Третье объяснено иначе."]}]}
CONTEXT = {"search": True, "images_allowed": True, "slide_range": None, "conclusion": True,
           "calls": [{"name": "search", "argument": "тема", "error": "", "chars": 120}],
           "tool_text": "Результаты поиска «тема»:\nd1 | ru | Тема | Начало статьи", "sources": {"d1": {"title": "Тема", "lang": "ru", "section": "", "url": "https://ru.wikipedia.org/wiki/Тема"}}}


def _generated(annotation=ANNOTATION, context=CONTEXT):
    return generate("t1", ORIGINAL, annotation, POINTS, context, DONOR)


def _by_kind(result, kind):
    return [p for p in result.pairs if p.kind == kind]


def test_doc_round_trip_keeps_the_deck():
    doc = Doc.parse(ORIGINAL)
    assert len(doc) == 5 and doc.title(3) == "Второе" and doc.head == ["#PLAN: тема → первое → второе → третье → итоги"]
    before, after = parse(ORIGINAL), parse(doc.text())
    assert [[e.raw for e in s.elements] for s in before.slides] == [[e.raw for e in s.elements] for s in after.slides]
    assert before.plan == after.plan and not after.errors


def test_paraphrase_replaces_text_lines_in_order():
    doc = paraphrase(Doc.parse(ORIGINAL), 2, ["Новый текст.", "Новый пункт"])
    assert doc.slide(2)[2:4] == ["#TEXT: Новый текст.", "#BULLET: Новый пункт"]
    try:
        paraphrase(Doc.parse(ORIGINAL), 2, ["Только одна строка"])
    except ValueError:
        pass
    else:
        raise AssertionError("число строк перефразировки не проверено")


def test_every_pair_refers_to_saved_variants_and_only_format_defects_break_parsing():
    result = _generated()
    broken = {"untagged_lines", "untagged_chain", "dilution", "tag_typo", "bad_syntax", "broken_chart"}
    ids = [p.id for p in result.pairs]
    assert len(ids) == len(set(ids))
    for pair in result.pairs:
        assert pair.base in result.variants and pair.variant in result.variants
        deck = parse(result.variants[pair.variant]["text"])
        assert bool(deck.errors) == (pair.kind in broken or pair.variant.startswith("untagged")), pair.id


def test_coverage_defects_touch_only_their_slide():
    result = _generated()
    drop = next(p for p in _by_kind(result, "drop_explanation") if p.thesis == 1)
    assert [s.title for s in parse(result.variants[drop.variant]["text"]).slides] == ["Тема", "Первое", "Третье",
                                                                                       "Заключение"]
    hidden = next(p for p in _by_kind(result, "hide_in_notes") if p.thesis == 1)
    slide = parse(result.variants[hidden.variant]["text"]).slides[2]
    assert [e.kind for e in slide.elements] == ["IMAGE"] and slide.notes == ["Объяснение второго."]
    keys = next(p for p in _by_kind(result, "keys_only") if p.thesis == 0)
    assert "#TEXT: 1957" in result.variants[keys.variant]["text"]
    assert (drop.target, drop.expect_target, drop.expect_total) == ("coverage", DOWN, DOWN)


def test_thesis_shared_with_another_slide_is_not_perturbed():
    annotation = dict(ANNOTATION, theses=[[2, 3], [3], [4], [5]])
    result = _generated(annotation)
    assert {p.thesis for p in _by_kind(result, "drop_explanation")} == {2}


def test_keys_and_invariance_pairs_follow_annotation():
    result = _generated()
    wrong = _by_kind(result, "wrong_number")[0]
    text = result.variants[wrong.variant]["text"]
    assert "в 1967 году" in text and "1957" not in text and wrong.expect_target == DOWN
    form = _by_kind(result, "key_wordform")[0]
    assert (form.expect_target, form.expect_total) == (SAME, SAME)
    reorder = _by_kind(result, "reorder")[0]
    assert [s.title for s in parse(result.variants[reorder.variant]["text"]).slides][2:4] == ["Третье", "Второе"]
    assert _by_kind(result, "negation")[0].diagnostic


def test_limits_chain_starts_at_the_limit():
    result = _generated()
    at_limit = parse(result.variants["image_at_limit"]["text"])
    assert sum(e.kind == "IMAGE" for s in at_limit.slides for e in s.elements) == 3  # N = 3 содержательных слайда
    excess = {p.strength: p for p in _by_kind(result, "image_excess")}
    assert excess[1].base == "image_at_limit" and excess[1].expect_target == NOT_UP and excess[3].expect_target == DOWN


def test_context_variants_change_only_the_context():
    result = _generated()
    forbidden = _by_kind(result, "forbidden_image")[0]
    assert result.variants[forbidden.variant]["context"] == {"images_allowed": False}
    assert "#IMAGE" not in result.variants[forbidden.base]["text"]
    repeat = _by_kind(result, "repeat_call")[0]
    calls = result.variants[repeat.variant]["context"]["calls"]
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result.variants[repeat.variant]["text"] == result.variants["original"]["text"]
    plain = _generated(context=dict(CONTEXT, search=False, tool_text=None, calls=[]))
    assert not _by_kind(plain, "fake_quote") and _by_kind(plain, "call_in_plain")


def test_episode_context_reads_calls_outputs_and_sources():
    text = ('#PLAN: план\n<tool_call>\n{"name": "search", "arguments": {"query": "тема"}}\n</tool_call><|im_end|>\n'
            "<|im_start|>user\n<tool_response>\nРезультаты поиска «тема»:\nd1 | ru | Тема | Начало статьи\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n#SLIDE\n#TITLE: Тема\n")
    context = episode_context(text, search=True)
    output = "Результаты поиска «тема»:\nd1 | ru | Тема | Начало статьи"
    assert context["calls"] == [{"name": "search", "argument": "тема", "error": "", "chars": len(output)}]
    assert list(context["sources"]) == ["d1"] and "Начало статьи" in context["tool_text"]
    plain = episode_context(text, search=False)
    assert plain["calls"][0]["disabled"] and plain["tool_text"] is None and plain["sources"] == {}


def test_conflict_pairs_and_blind_view(tmp_path):
    from presentation_rl.perturb import conflict_pairs, export_blind, readable

    original = ORIGINAL.replace("#TEXT: Объяснение третьего.",
                                "#TEXT: Третье устроено так. Это видно по двум признакам. Первый связан с формой. "
                                "Второй — с назначением.")
    loose = {p["kind"]: p for p in conflict_pairs("t1", original, ANNOTATION, POINTS, CONTEXT)}
    assert len(parse(loose["stretch_all_vs_missing1"]["a"]).slides) == len(parse(original).slides) + 2
    pairs = {p["kind"]: p for p in conflict_pairs("t1", original, ANNOTATION, POINTS, CONTEXT, even=True)}
    assert set(pairs) == {"stretch_half_vs_missing1", "stretch_all_vs_missing1", "stretch_all_vs_missing2",
                          "repeat_vs_missing", "text_vs_keys"}
    n = len(parse(original).slides)
    half, full = parse(pairs["stretch_half_vs_missing1"]["a"]), parse(pairs["stretch_all_vs_missing1"]["a"])
    assert n < len(half.slides) <= len(full.slides) == n + 1 and not full.errors
    # разрезан только слайд, который делится ровно: по два предложения на каждой половине
    assert "#TEXT: Третье устроено так. Это видно по двум признакам." in pairs["stretch_all_vs_missing1"]["a"]
    assert "Объяснение второго.\n#IMAGE" in pairs["stretch_all_vs_missing1"]["a"]  # одно предложение — не делится
    assert len(parse(pairs["stretch_all_vs_missing1"]["b"]).slides) == n - 1
    assert len(parse(pairs["stretch_all_vs_missing2"]["b"]).slides) == n - 2
    assert "1967" in pairs["text_vs_keys"]["a"] and "#TEXT: 1957" in pairs["text_vs_keys"]["b"]
    view = readable(ORIGINAL, CONTEXT["sources"])
    assert "### Слайд 2. Первое" in view and "Источник: Википедия — Тема" in view and "#" not in view.replace("###", "")
    folder = tmp_path / "judge"
    key = export_blind(list(pairs.values()), {"t1": "Сделай презентацию"}, {"t1": CONTEXT["sources"]}, folder,
                       {"t1": ["Событие — 1957 год."]})
    assert sorted(k["id"] for k in key) == sorted(p["id"] for p in pairs.values())
    main, back = (folder / "01.md").read_text(encoding="utf-8"), (tmp_path / "judge_reversed" / "01.md").read_text(
        encoding="utf-8")
    assert "Событие — 1957 год." in main and main.count("## Презентация") == 2
    first, second = main.split("## Презентация 2")
    assert second.strip() == back.split("## Презентация 2")[0].split("## Презентация 1")[1].strip()
