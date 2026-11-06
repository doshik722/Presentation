# Здесь проверяем сборку примеров SFT в эпизоды: токены как у среды, маска только на репликах модели, отложенные темы.
import json
from pathlib import Path

import pytest

from presentation_rl.environment import END, NEXT_TURN, TOOL_RESULT
from presentation_rl.tasks import load_tasks
from presentation_rl.training.sft import (Sample, build_samples, drop_long, plain_episode, request_for,
                                          search_episode, split)

from test_environment import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOK = CharTokenizer()


def _prompt(request):
    return TOK.encode("запрос" + NEXT_TURN)


def _first(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8").splitlines()[0])


def test_plain_example_is_one_model_reply_after_the_prompt():
    record = _first("data/tasks/sft/sft_dataset.jsonl")
    prompt = _prompt(None)
    episode = plain_episode(record, prompt, TOK)
    assert episode.ids == prompt + TOK.encode(record["answer"] + END)
    assert episode.loss_mask == [0] * len(prompt) + [1] * (len(episode.ids) - len(prompt))


def test_search_example_keeps_tool_outputs_out_of_the_loss():
    record = _first("data/tasks/sft_search/sft_search_dataset.jsonl")
    episode = search_episode(record, _prompt(None), TOK)
    text = TOK.decode(episode.ids)
    trained = TOK.decode([t for t, m in zip(episode.ids, episode.loss_mask) if m])
    output = next(t["text"] for t in record["turns"] if t["role"] == "program")
    assert TOOL_RESULT.format(output) in text and output not in trained  # выдача — в эпизоде, но не в loss
    assert "<tool_call>" in trained and "#SLIDE" in trained


def test_request_must_match_the_task():
    tasks = {t.id: t for t in load_tasks(ROOT / "data" / "tasks" / "tasks.json", split="sft")}
    record = _first("data/tasks/sft/sft_dataset.jsonl")
    assert request_for(record, tasks).render() == record["request"]
    with pytest.raises(ValueError):
        request_for({**record, "request": record["request"] + "\nАудитория: другая"}, tasks)


def test_all_examples_build_and_held_out_topics_do_not_leak():
    samples = build_samples(ROOT, TOK, _prompt)
    assert len(samples) == 137 and {s.kind for s in samples} == {"plain", "search"}
    assert all(len(s.ids) == len(s.mask) and 0 < s.target_tokens < len(s.ids) for s in samples)
    train, held = split(samples, 9, seed=0)
    assert len({s.id for s in held}) == 9 and not {s.id for s in train} & {s.id for s in held}
    assert split(samples, 9, seed=0)[1] == held  # разбиение воспроизводимо


def test_long_episodes_are_dropped_not_cut():
    samples = [Sample("t1", "plain", "x", [1] * 5, [0, 0, 1, 1, 1]), Sample("t2", "plain", "x", [1] * 50, [1] * 50)]
    kept, dropped = drop_long(samples, 10)
    assert [s.id for s in kept] == ["t1"] and [s.id for s in dropped] == ["t2"] and len(kept[0].ids) == 5


try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="нет torch")
def test_collate_masks_labels():
    from presentation_rl.training.sft import collate

    batch = collate([Sample("t1", "plain", "x", [5, 6, 7, 8], [0, 0, 1, 1])])
    assert batch["labels"].tolist() == [[-100, -100, 7, 8]] and batch["input_ids"].shape == (1, 4)
