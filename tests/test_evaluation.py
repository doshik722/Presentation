# Здесь проверяем признаки вырождения и парное сравнение моделей по темам.
from presentation_rl.evaluation import degeneration, paired, reward_spread, topic_means


def test_degeneration_loop_and_unfinished():
    normal = "#SLIDE\n#TITLE: Введение\n#TEXT: Солнечная энергия дешевеет каждый год.\n"
    assert not degeneration(normal, "stop")["degenerate"]
    loop = "#SLIDE\n#TITLE: x\n" + "услуга на " * 40
    result = degeneration(loop, "stop")
    assert result["loop_8gram"] >= 4 and result["degenerate"]
    assert degeneration(normal, "length")["degenerate"]


def test_degeneration_broken_words():
    text = "слoво " * 5  # латинская «o» внутри русского слова
    result = degeneration(text, "stop")
    assert result["broken_words"] == 5 and result["glitch"] and result["degenerate"]


def rows_for(model, rewards):
    return [{"model": model, "mode": "plain", "id": t, "sample": s, "reward_total": v}
            for t, values in rewards.items() for s, v in enumerate(values)]


def test_paired_uses_topic_means():
    rows = rows_for("base", {"t1": [0.0, 1.0], "t2": [1.0, 1.0]}) + rows_for("sft", {"t1": [1.0, 1.0], "t2": [1.0, 1.0]})
    assert topic_means(rows, "base", "reward_total", "plain") == {"t1": 0.5, "t2": 1.0}
    result = paired(rows, "reward_total", "plain", "base", "sft")
    assert result["n"] == 2 and abs(result["diff"] - 0.25) < 1e-9
    assert result["low"] <= result["diff"] <= result["high"]


def test_reward_spread_and_errors_skipped():
    rows = rows_for("base", {"t1": [0.0, 2.0], "t2": [1.0, 1.0]})
    rows.append({"model": "base", "mode": "plain", "id": "t1", "sample": 9, "reward_total": 100.0, "error": "boom"})
    assert reward_spread(rows, "base") == 0.5
