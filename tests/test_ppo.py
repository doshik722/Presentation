# Здесь проверяем части PPO без torch: GAE, нормировку, меры критика, выборку позиций разогрева, деление и конфиг.
import math
from pathlib import Path

import pytest

from presentation_rl.training.common import RLConfig
from presentation_rl.training.grpo import RunStopped
from presentation_rl.training.ppo import (POSITIONS, PPOConfig, critic_metrics, exp028_verdict, gae,
                                          leave_one_out_means, normalize,
                                          paired_improvement, sample_positions, slice_index, split_topics)

ROOT = Path(__file__).resolve().parents[1]


def explicit(values, reward, lam):
    """A_t = λⁿ·R + (1 − λ)·Σ_{j=1}^{n} λ^{j−1}·V_{t+j} − V_t, n = T − t (docs/ppo.md, раздел 6)."""
    T = len(values) - 1
    out = []
    for t in range(len(values)):
        n = T - t
        target = lam ** n * reward + (1 - lam) * sum(lam ** (j - 1) * values[t + j] for j in range(1, n + 1))
        out.append(target - values[t])
    return out


def test_gae_lambda_one_is_reward_minus_value_and_targets_are_reward():
    values = [0.5, 0.6, 0.4, 0.4]
    a, g = gae(values, 0.3, 1.0)
    assert a == pytest.approx([-0.2, -0.3, -0.1, -0.1]) and g == pytest.approx([0.3] * 4)


def test_gae_recursion_matches_explicit_formula_for_any_lambda():
    values = [0.5, 0.6, 0.4, 0.4]
    for lam in (0.0, 0.5, 0.95, 0.999, 1.0):
        a, g = gae(values, 0.3, lam)
        assert a == pytest.approx(explicit(values, 0.3, lam))
        assert g == pytest.approx([x + v for x, v in zip(a, values)])
    assert gae(values, 0.3, 0.0)[0] == pytest.approx([0.1, -0.2, 0.0, -0.1])  # λ = 0: сдвиги ожидания
    assert gae(values, 0.3, 0.5)[0] == pytest.approx([-0.0125, -0.225, -0.05, -0.1])  # пример subtleties.md
    assert gae([], 0.3, 1.0) == ([], [])


def test_normalize_over_all_tokens_with_floor_and_stops_on_nonfinite():
    out, stats = normalize([[1.0, 3.0], [5.0]], 0.02)
    flat = [x for e in out for x in e]
    assert sum(flat) == pytest.approx(0.0) and stats["raw_mean"] == pytest.approx(3.0)
    assert math.sqrt(sum(x * x for x in flat) / 3) == pytest.approx(1.0)
    same, stats = normalize([[0.4, 0.4]], 0.02)
    assert same == [[0.0, 0.0]] and stats["floor_used"]
    with pytest.raises(RunStopped):
        normalize([[float("nan")]], 0.02)


def test_slice_index_and_critic_metrics():
    assert [slice_index(5, n) for n in POSITIONS] == [0, 1, 2, 3, 4]
    rewards, modes = [0.2, 0.6, 0.4, 0.8], ["search", "plain", "search", "plain"]
    perfect = [[r] * 3 for r in rewards]
    m = critic_metrics(perfect, rewards, modes, 0.5, {"search": 0.3, "plain": 0.7})
    assert m["all_mse"] == 0 and m["all_bias"] == 0 and m["all_ev"] == pytest.approx(1.0)
    assert m["all_mse_const"] == pytest.approx(sum((r - 0.5) ** 2 for r in rewards) / 4)
    assert m["first_mse_mode"] == pytest.approx(0.01)
    shifted = critic_metrics([[r + 0.1] * 3 for r in rewards], rewards, modes, 0.5)
    assert shifted["all_bias"] == pytest.approx(0.1) and shifted["all_ev"] == pytest.approx(1.0)  # EV смещения не видит
    constant = critic_metrics([[0.5] * 3 for _ in rewards], rewards, modes, 0.5)
    assert constant["all_mse"] == pytest.approx(constant["all_mse_const"])


def test_paired_improvement_by_topics():
    rows = [{"task": f"t{i}", "mode": m, "d": 0.01 + 0.001 * i} for i in range(10) for m in ("search", "plain")]
    better = paired_improvement(rows)
    assert better["estimate"] > 0 and better["low"] > 0 and better["topics"] == 10
    worse = paired_improvement([{**r, "d": -r["d"]} for r in rows])
    assert worse["high"] < 0
    mixed = paired_improvement([{**r, "d": r["d"] * (1 if int(r["task"][1:]) % 2 else -1)} for r in rows])
    assert mixed["low"] < mixed["estimate"] < mixed["high"]


def test_leave_one_out_means():
    assert leave_one_out_means([1.0, 2.0, 3.0, 5.0], ["a", "a", "a", "b"]) == [2.5, 2.0, 1.5, None]


def test_sample_positions_keeps_ends_respects_cap_and_is_deterministic():
    counts = [1000, 5, 0, 2000]
    keep = sample_positions(counts, 300, seed=7)
    assert keep == sample_positions(counts, 300, seed=7)
    assert keep[2] == [] and keep[1][0] == 0 and keep[1][-1] == 4
    assert all(k[0] == 0 and k[-1] == n - 1 for k, n in zip(keep, counts) if n)
    assert 200 < sum(len(k) for k in keep) < 420
    assert sample_positions([3, 4], 100, seed=0) == [[0, 1, 2], [0, 1, 2, 3]]


def test_split_topics_by_topic_deterministic():
    ids = [f"t{i}" for i in range(40)] * 2
    held = split_topics(ids, 0.2, seed=3)
    assert held == split_topics(ids, 0.2, seed=3) and len(held) == 8


def test_ppo_config_loads_and_shares_common_settings_with_grpo():
    ppo = PPOConfig.load(ROOT / "configs" / "algorithms" / "ppo.yaml")
    grpo = RLConfig.load(ROOT / "configs" / "algorithms" / "grpo.yaml")
    assert ppo.algorithm == "ppo" and ppo.gae_lambda == 1.0 and ppo.ppo_epochs == 2 and ppo.head_betas == (0.9, 0.999)
    common = {k: v for k, v in grpo.to_json().items() if k not in ("algorithm", "runs_dir")}
    assert {k: ppo.to_json()[k] for k in common} == common  # условия сравнения одинаковые
    assert ppo.to_json()["head_betas"] == [0.9, 0.999]
    with pytest.raises(TypeError):
        PPOConfig(nope=1)


def test_exp028_verdict():
    base = {"all_mse": 0.01, "all_mse_const": 0.02, "first_mse": 0.019, "first_mse_const": 0.02,
            "search_all_mse": 0.012, "search_all_mse_const": 0.02}
    assert exp028_verdict({**base, "paired": {"low": 0.001, "high": 0.01}}) == "пригоден"
    assert exp028_verdict({**base, "paired": {"low": -0.001, "high": 0.01}}) == "данных недостаточно"
    assert exp028_verdict({**base, "paired": {"low": -0.01, "high": -0.001}}) == "не пригоден"
    worse = {**base, "first_mse": 0.03, "paired": {"low": 0.001, "high": 0.01}}
    assert exp028_verdict(worse).startswith("данных недостаточно (хуже константы: first")
