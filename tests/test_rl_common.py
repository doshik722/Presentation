# Здесь проверяем части RL без torch: сиды, очередь заданий, преимущество, журнал, сбор эпизодов, клиент vLLM.
import math
from types import SimpleNamespace

import pytest

from presentation_rl.environment import END, NEXT_TURN
from presentation_rl.training import vllm_client
from presentation_rl.training.common import (Job, Journal, RLConfig, Rollout, TaskQueue, check_resume, collect,
                                             fixed_validation_topics, grounding, mismatch_exceeded, mismatch_stats,
                                             read_journal, slide_facts, stable_seed, summarize, target_positions,
                                             validation_jobs, vllm_generator)
from presentation_rl.training.grpo import RunStopped, advantages

from test_environment import CharTokenizer


def test_stable_seed_is_deterministic_and_distinct():
    assert stable_seed(0, "train", 5, "t1", "search", 3) == stable_seed(0, "train", 5, "t1", "search", 3)
    assert stable_seed(0, "train", 5, "t1", "search", 3) != stable_seed(0, "train", 5, "t1", "search", 4)
    assert 0 <= stable_seed("x") < 2**31


def test_task_queue_epochs_partial_step_and_budget():
    ids = [f"t{i:03d}" for i in range(274)]
    queue = TaskQueue(ids, per_mode=4, seed=0)
    steps = [queue.next() for _ in range(150)]
    assert all(len(s) == 8 for s in steps[:68]) and len(steps[68]) == 4  # шаг 69 — неполный: 2 + 2
    assert len(steps[137]) == 4 and len(steps[69]) == 8  # шаг 138 — второй неполный
    first_epoch = [t for s in steps[:69] for t in s]
    for mode in ("search", "plain"):
        assert sorted(t for m, t in first_epoch if m == mode) == ids  # каждая тема — ровно раз за эпоху
    assert sum(len(s) for s in steps) * 8 == 9536
    assert [m for m, _ in steps[0]] == ["search"] * 4 + ["plain"] * 4
    # порядок по режимам разный, по эпохам — разный
    assert [t for m, t in steps[0] if m == "search"] != [t for m, t in steps[0] if m == "plain"]
    assert [t for m, t in steps[69] if m == "search"] != [t for m, t in steps[0] if m == "search"]
    # продолжение с сохранённого состояния даёт ту же последовательность
    other = TaskQueue(ids, 4, 0)
    for _ in range(10):
        other.next()
    resumed = TaskQueue(ids, 4, 0)
    resumed.load_state(other.state())
    assert [resumed.next() for _ in range(70)] == steps[10:80]


def test_advantages_group_mean_batch_std_and_floor():
    rewards = [0.70, 0.75, 0.80, 0.85, 0.90, 0.5, 0.5, 0.5]
    groups = [0, 0, 0, 0, 0, 1, 1, 1]
    values, stats = advantages(rewards, groups, floor=0.02)
    std = (sum((r - sum(rewards) / 8) ** 2 for r in rewards) / 8) ** 0.5
    assert stats["std_batch"] == pytest.approx(std) and not stats["floor_used"]
    assert values[0] == pytest.approx(-0.10 / std) and values[4] == pytest.approx(0.10 / std)
    assert values[5:] == [0.0, 0.0, 0.0] and stats["tied_groups"] == 1
    assert stats["low_spread_groups"] == pytest.approx(0.5)  # группа 1 — без разброса
    small, stats = advantages([0.500, 0.501, 0.502, 0.503], [0, 0, 0, 0], floor=0.02)
    assert stats["floor_used"] and small[0] == pytest.approx(-0.0015 / 0.02)
    with pytest.raises(RunStopped):
        advantages([0.1, math.nan], [0, 0])


def test_journal_resume_drops_cancelled_steps(tmp_path):
    path = tmp_path / "steps.jsonl"
    journal = Journal(path, session=1)
    for step in range(1, 13):
        journal.write({"step": step, "value": step})
    second = Journal(path, session=2)
    second.resume(10)
    for step in range(11, 14):
        second.write({"step": step, "value": 100 + step})
    records = read_journal(path)
    assert [r["step"] for r in records] == list(range(1, 14))
    assert [r["value"] for r in records if r["step"] > 10] == [111, 112, 113]
    assert all(r["session"] == 2 for r in records if r["step"] > 10)


def test_validation_topics_and_jobs():
    ids = [f"v{i:02d}" for i in range(25)]
    topics = fixed_validation_topics(ids, 8)
    assert topics == fixed_validation_topics(list(reversed(ids)), 8) and len(set(topics)) == 8
    jobs = validation_jobs(topics, 2)
    assert len(jobs) == 32 and {m for m, _, _ in jobs} == {"search", "plain"}


def test_mismatch_stats_and_limits():
    stats = mismatch_stats([-0.1, -0.2, -0.3, -0.4], [-0.1, -0.25, math.nan, -0.5])
    assert stats["tokens"] == 3 and stats["mean"] == pytest.approx((0 + 0.05 + 0.1) / 3)
    assert stats["mean_abs"] == pytest.approx(0.05) and stats["p99"] == pytest.approx(0.1)
    assert mismatch_exceeded(stats, {"mean_abs": 0.01, "p99": None})
    assert mismatch_exceeded(stats, {"mean_abs": None, "p99": None}) is None


def test_config_rejects_unknown_and_resume_needs_same_config(tmp_path):
    (tmp_path / "c.yaml").write_text("steps: 10\nbetas: [0.9, 0.99]\nlr: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="неизвестные"):
        RLConfig.load(tmp_path / "c.yaml")
    config = RLConfig(steps=10)
    manifest = {"config": RLConfig(steps=20).to_json(), "code": {}}
    with pytest.raises(ValueError, match="новый запуск"):
        check_resume(manifest, config, tmp_path)


def test_target_positions_follow_mask():
    assert target_positions([0, 0, 1, 1, 0, 1]) == [2, 3, 5]


class _Task(SimpleNamespace):
    pass


def test_collect_runs_episodes_in_parallel_and_keeps_logprobs():
    tok = CharTokenizer()
    request = SimpleNamespace(search=False)
    jobs = [Job(_Task(id=f"t{i}", request=request), "plain", s, ("x", i, s)) for i in range(3) for s in range(2)]
    replies = {}

    def make_generate(job):
        def generate(ids, budget):
            new = tok.encode(f"#SLIDE\n#TITLE: {job.task.id}/{job.sample}" + END)
            replies[(job.task.id, job.sample)] = new
            return new, True, [-0.25] * len(new)
        return generate

    rollouts = collect(jobs, tok, lambda req: tok.encode("п" + NEXT_TURN), make_generate, lambda: None,
                       RLConfig(workers=4))
    assert [(r.task_id, r.sample) for r in rollouts] == [(j.task.id, j.sample) for j in jobs]
    for r in rollouts:
        assert r.answer == f"#SLIDE\n#TITLE: {r.task_id}/{r.sample}" and r.finish == "stop"
        assert len(r.gen_logprobs) == r.model_tokens == len(target_positions(r.mask))
        assert r.text.endswith(END)

    def broken(job):
        def generate(ids, budget):
            raise RuntimeError("сервер упал")
        return generate

    with pytest.raises(RuntimeError, match="сервер упал"):
        collect(jobs, tok, lambda req: tok.encode("п"), broken, lambda: None, RLConfig(workers=2))


def test_vllm_generator_seeds_per_turn():
    calls = []

    class Client:
        def generate(self, ids, budget, model, temperature, seed, stops):
            calls.append((model, temperature, seed, tuple(stops)))
            return [1], True, [-0.1]

    generate = vllm_generator(Client(), "run_step3", 0.9, [7], ("seed", "train", 3, "t1", "plain", 0))
    generate([0], 10)
    generate([0, 1], 10)
    assert calls[0][2] == stable_seed("seed", "train", 3, "t1", "plain", 0, 0)
    assert calls[1][2] == stable_seed("seed", "train", 3, "t1", "plain", 0, 1)
    assert calls[0][:2] == ("run_step3", 0.9) and calls[0][3] == (7,)


def test_vllm_client_payload_and_parsing():
    client = vllm_client.VLLMClient("http://127.0.0.1:1")
    sent = {}

    def fake(path, payload=None, method=None):
        sent.update(payload)
        return {"choices": [{"finish_reason": "stop", "logprobs": {
            "tokens": ["token_id:5", "token_id:6"], "token_logprobs": [-0.5, None]}}]}

    client._request = fake
    new, finished, logprobs = client.generate([1, 2], 50, "m", 0.9, 11, [99])
    assert new == [5, 6, 99] and finished and logprobs[0] == -0.5 and math.isnan(logprobs[1])
    assert math.isnan(logprobs[2])  # стоп-токен не вернулся — дописан с неизвестным log p
    assert sent["top_k"] == -1 and sent["top_p"] == 1.0 and sent["min_p"] == 0.0 and sent["repetition_penalty"] == 1.0
    assert sent["prompt"] == [1, 2] and sent["seed"] == 11 and sent["return_tokens_as_token_ids"]


class _Grammemes(SimpleNamespace):
    pass


class NameAnalyzer:
    NAMES = {"Кальвин": ("кальвин", {"Surn"}), "Франклин": ("франклин", {"Surn"}), "Растение": ("растение", set())}

    def parse(self, word):
        form, grammemes = self.NAMES.get(word, (word.lower(), set()))
        return [SimpleNamespace(normal_form=form, tag=SimpleNamespace(POS="NOUN", grammemes=frozenset(grammemes)))]


def test_slide_facts_and_grounding():
    from presentation_rl.dsl import parse
    from presentation_rl.rewards import Lemmatizer

    lem = Lemmatizer(NameAnalyzer())
    assert slide_facts("Растение и Кальвин в 1953 году, КПД 3,5 и Python", lem) == {"кальвин", "1953", "3.5", "python"}
    deck = parse("#SLIDE\n#TITLE: Фотосинтез\n#TEXT: Кальвин описал цикл в 1950 году, Франклин — 1953.")
    report = SimpleNamespace(items=[SimpleNamespace(kind="SOURCE", status="ok", slide=1)])
    result = grounding(deck, report, "Мельвин Кальвин открыл цикл в 1950 году.", lem)
    assert result == {"facts": 4, "facts_found": 2, "slides_with_source": 1}


def test_summarize_by_mode():
    def rollout(mode, reward, calls, analytics):
        return Rollout("t", mode, 0, [0, 1, 2], [0, 1, 1], [math.nan, math.nan], "", "", "stop", calls, 0, 1.0,
                       reward, {"coverage": reward}, analytics)

    rollouts = [rollout("search", 1.0, [{"name": "search"}], {"facts": 4, "facts_found": 2, "slides": 2,
                                                              "slides_with_source": 1, "parse_rate": 1.0}),
                rollout("search", 0.5, [], {"facts": 0, "facts_found": 0, "slides": 2, "slides_with_source": 0,
                                            "parse_rate": 1.0}),
                rollout("plain", 0.2, [], {"slides": 2, "parse_rate": 0.5})]
    out = summarize(rollouts)
    assert out["search_reward"] == pytest.approx(0.75) and out["plain_reward"] == pytest.approx(0.2)
    assert out["search_share"] == pytest.approx(0.5) and out["grounding"] == pytest.approx(0.5)
    assert out["grounding_no_facts"] == 1 and out["slides_with_source"] == pytest.approx(0.25)
    assert out["finish"] == {"stop": 3} and out["parse_rate"] == pytest.approx(5 / 6)


def test_target_positions_select_exactly_the_model_tokens(tmp_path):
    """Сдвиг на позицию и маска на эпизоде с двумя репликами и выдачей: токены в позициях обучения — ровно
    то, что выбрала модель (включая <|im_end|>), без вставленного средой."""
    from presentation_rl.environment import run_episode
    from presentation_rl.tools.cache import WikiCache
    from presentation_rl.tools.wiki import WikiClient
    from presentation_rl.tools.wiki_tools import WikiTools

    from test_environment import CALL, Script
    from test_wiki_tools import FakeApi

    tok = CharTokenizer()
    tools = WikiTools(WikiCache(tmp_path, WikiClient(fetch=FakeApi(), min_interval=0)))
    episode = run_episode(tok.encode("п" + NEXT_TURN), tok, Script(tok, [CALL + END, "#SLIDE" + END]), tools)
    model_ids = [i for s in episode.segments if s.trainable for i in s.ids]
    assert [episode.ids[p] for p in target_positions(episode.loss_mask)] == model_ids
    assert model_ids[-1] == tok.encode(END)[0] and len(model_ids) == episode.generated_tokens
