# Здесь проверяем оценку награды на испорченных презентациях: вердикты, сборку и награду пар, выбор конфигурации.
import json

from presentation_rl.perturb import generate
from presentation_rl.dsl import parse
from presentation_rl.perturb_eval import (VARIANTS, Config, Item, Rules, ablation_configs, ablation_rows, author_answers,
                                          build_all, calibrate_tau, check_pairs, choose, conflict_rows, edge,
                                          evaluate_pairs, family_shares, load_conflicts, load_foreign, nested_cv,
                                          score_items, signed, stratified_folds, tally, verdict)
from presentation_rl.rewards import Lemmatizer

from test_perturb import ANNOTATION, CONTEXT, DONOR, ORIGINAL, POINTS
from test_rewards import FakeAnalyzer, FakeEmbedder


def test_verdicts_use_margins():
    assert verdict(-0.3, "down", 0.25, 0.1) == "ok"
    assert verdict(-0.2, "down", 0.25, 0.1) == "weak"
    assert verdict(0.05, "down", 0.25, 0.1) == "tie"
    assert verdict(0.2, "down", 0.25, 0.1) == "violation"
    assert verdict(0.08, "not_up", 0.25, 0.1) == "ok" and verdict(0.2, "not_up", 0.25, 0.1) == "violation"
    assert verdict(-0.15, "same", 0.25, 0.1) == "violation" and verdict(None or 0.0, None, 0.25, 0.1) is None


def test_penalties_change_sign():
    assert signed({"total": 1.0, "coverage": 0.5, "length": 0.2, "tools": 0.1}) == \
        {"total": 1.0, "coverage": 0.5, "length": -0.2, "tools": -0.1}


def test_pairs_are_built_scored_and_judged_end_to_end():
    result = generate("t1", ORIGINAL, ANNOTATION, POINTS, CONTEXT, DONOR)
    items = {("t1", vid): Item(("t1", vid), "t1", v["text"], {**CONTEXT, **v["context"]})
             for vid, v in result.variants.items()}
    pairs = [{**p.__dict__, "task": "t1", "mode": "search"} for p in result.pairs]
    build_all(items, log=lambda _: None)
    pairs, dropped = check_pairs(pairs, items)
    assert all("text_overflow" in d for d in dropped), dropped  # короткий текст может и поместиться
    lemmatizer = Lemmatizer(FakeAnalyzer())
    scores = score_items(items, {"t1": POINTS}, Config("fake", 0.0, 1.0, 0.5), FakeEmbedder(), lemmatizer)
    rows = evaluate_pairs(pairs, scores, m=0.01, eps=0.004)
    by_id = {r["id"]: r for r in rows}
    assert by_id["t1/drop_explanation/T1"]["total"] == "ok"
    assert by_id["t1/bad_source/1"]["component"] == "ok" and by_id["t1/repeat_call"]["total"] == "ok"
    assert by_id["t1/paraphrase/4"]["d_total"] is not None
    shares = family_shares(tally(rows), {"t1": 1})
    assert set(shares) >= {"coverage", "format", "tools", "invariance"} and "chain" not in shares


def test_conflict_answers_are_mapped_through_the_blind_key_and_both_orders(tmp_path):
    pairs = [{"id": f"t1/x{i}", "task": "t1", "kind": "x", "decides": "length", "a": ORIGINAL, "b": ORIGINAL}
             for i in range(3)]
    (tmp_path / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    (tmp_path / "key.json").write_text(json.dumps([{"pair": f"0{i + 1}", "id": f"t1/x{i}", "first": "b"}
                                                   for i in range(3)]), encoding="utf-8")
    (tmp_path / "answers_codex.json").write_text(json.dumps(
        [{"pair": "01", "better": "1"}, {"pair": "02", "better": "1"}, {"pair": "03", "better": "равны"}]), encoding="utf-8")
    # в обратном порядке: 01 — та же презентация (b), 02 — другая, 03 — снова «равны»
    (tmp_path / "answers_codex_reversed.json").write_text(json.dumps(
        [{"pair": "01", "better": "2"}, {"pair": "02", "better": "1"}, {"pair": "03", "better": "равны"}]),
        encoding="utf-8")
    items, conflicts = load_conflicts(tmp_path, {"t1": CONTEXT})
    assert [c["judges"]["codex"] for c in conflicts] == ["b", None, "equal"] and len(items) == 6
    scores = {("conflict", f"t1/x{i}", s): {"total": v} for i, (a, b) in enumerate([(0.5, 0.7), (0.5, 0.5), (0.6, 0.6)])
              for s, v in (("a", a), ("b", b))}
    rows = conflict_rows(conflicts, scores, "codex", 0.01)
    assert [(r["reward"], r["agree"]) for r in rows] == [("b", True)]  # «равны» судьи не учитывается
    (tmp_path / "author.txt").write_text("Пары: 01, 02\n01: 2 — короче\n02:\n03: 1. обе плохие\n04:равны\n",
                                         encoding="utf-8")
    assert author_answers(tmp_path / "author.txt") == [{"pair": "01", "better": "2", "reason": "короче"},
                                                       {"pair": "03", "better": "1", "reason": "обе плохие"},
                                                       {"pair": "04", "better": "равны", "reason": ""}]


def _table(scores):
    """Тема → семейство «f» с долей выполнения ``scores[тема]``."""
    return {t: {"f": [v, 1]} for t, v in scores.items()}


def test_choose_drops_failed_configs_prefers_stable_region_and_keeps_current_weights_without_evidence():
    tasks = [f"t{i}" for i in range(8)]
    modes = {t: ("search" if i % 2 else "plain") for i, t in enumerate(tasks)}
    configs = [Config("e", a, 1.0, 0.5) for a in (0.0, 0.2, 0.4, 0.6)]
    good = {t: 0.9 for t in tasks}
    tallies = {configs[0]: _table({t: 0.3 for t in tasks}),  # провал семейства — отсев
               configs[1]: _table(good), configs[2]: _table({t: 0.88 for t in tasks}),
               configs[3]: _table({t: 0.86 for t in tasks})}
    chosen, info = choose(tallies, None, stratified_folds(modes, 4, 0), Rules(fail_other=0.5))
    assert chosen in configs[1:3] and info["feasible"] == 3
    # с судьёй: λ = 2 совпадает чуть лучше, но без устойчивого преимущества — остаются текущие веса
    pair = [Config("e", 0.0, 1.0, 0.5), Config("e", 0.0, 2.0, 0.5)]
    tallies = {c: _table(good) for c in pair}
    conflicts = {pair[0]: {t: [3, 5] for t in tasks}, pair[1]: {t: [3 + (i == 0), 5] for i, t in enumerate(tasks)}}
    chosen, info = choose(tallies, conflicts, stratified_folds(modes, 4, 0))
    assert chosen == pair[0]
    conflicts[pair[1]] = {t: [5, 5] for t in tasks}  # явное преимущество во всех темах
    assert choose(tallies, conflicts, stratified_folds(modes, 4, 0))[0] == pair[1]


def test_folds_are_balanced_by_mode_and_cv_reports_stability():
    modes = {f"t{i}": ("search" if i < 8 else "plain") for i in range(16)}
    blocks = stratified_folds(modes, 4, seed=0)
    assert sorted(sum(blocks, [])) == sorted(modes)
    assert all(sum(modes[t] == "search" for t in b) == 2 and len(b) == 4 for b in blocks)
    configs = [Config("e", a, 1.0, 0.5) for a in (0.0, 0.2, 0.4)]
    tallies = {c: _table({t: 0.9 if c is configs[1] else 0.6 for t in modes}) for c in configs}
    report = nested_cv(tallies, None, modes)
    assert report["final"] == configs[1].name and report["stable"] and report["reliable"]
    assert edge(configs[2], configs) == {"alpha": 0.6} and edge(configs[1], configs) == {}


def test_ablation_variants_score_end_to_end(tmp_path):
    configs = ablation_configs("fake", 0.2, [0.5, 1.0], [0.5, 1.0])
    assert [c.variant for c in configs[:len(VARIANTS)]] == list(VARIANTS)
    assert len(configs) == len(VARIANTS) + 3 and Config("fake", 0.2, 1.0, 0.5).name == "fake|a=0.2|len=1.0|keys=0.5"
    assert configs[-1].name.endswith("|all") and configs[0].weights() == Config("fake", 0.2, 1.0, 0.5).weights()
    (tmp_path / "sft.jsonl").write_text("\n".join(json.dumps({"answer": ORIGINAL}) for _ in range(2)), encoding="utf-8")
    foreign = load_foreign(tmp_path / "sft.jsonl")
    assert len(foreign) == 2 and foreign[0]
    result = generate("t1", ORIGINAL, ANNOTATION, POINTS, CONTEXT, DONOR)
    items = {("t1", vid): Item(("t1", vid), "t1", v["text"], {**CONTEXT, **v["context"]})
             for vid, v in result.variants.items()}
    pairs = [{**p.__dict__, "task": "t1", "mode": "search"} for p in result.pairs]
    build_all(items, log=lambda _: None)
    pairs, _ = check_pairs(pairs, items)
    lemmatizer, embedder = Lemmatizer(FakeAnalyzer()), FakeEmbedder()
    tau = {"IMAGE": 0.0}
    scores = {c: score_items(items, {"t1": POINTS}, c, embedder, lemmatizer, foreign, tau) for c in configs}
    base = configs[0]
    assert scores[configs[1]][("t1", "original")]["coverage"] < scores[base][("t1", "original")]["coverage"]  # фон — сам исходник
    assert "contradictions" in scores[base][("t1", "original")]
    rows = ablation_rows(pairs, scores, {c: 0.01 for c in configs}, base)
    assert {r["variant"] for r in rows} == set(VARIANTS)
    try:
        score_items(items, {"t1": POINTS}, configs[1], embedder, lemmatizer)
    except ValueError:
        pass
    else:
        raise AssertionError("без фона вариант relative должен падать")


def test_tau_is_calibrated_on_own_slides_against_foreign_visuals():
    def deck(word, n):
        return parse("\n".join(f"#SLIDE\n#TITLE: Слайд\n#TEXT: {word}{i} текст\n#IMAGE: {word}{i} фото" for i in range(n)))

    decks = {"a": deck("альфа", 10), "b": deck("бета", 10), "c": deck("гамма", 10)}
    domains = {"a": "x", "b": "x", "c": "y"}
    tau, report = calibrate_tau(decks, domains, FakeEmbedder(), minimum=15)
    row = next(r for r in report if r["kind"] == "IMAGE")
    assert row["positives"] == 30 and row["negatives"] == 2 * 10 * 10  # a↔b, у c своей темы того же типа нет
    assert row["positives_pass"] >= 0.95 and row["negatives_filtered"] == 1.0 and tau == {"IMAGE": row["tau"]}
    assert not next(r for r in report if r["kind"] == "DIAGRAM")["verified"]
