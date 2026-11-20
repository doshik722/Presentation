# Здесь проверяю, что критик видит только закрытые слайды и уже полученную выдачу.
import numpy as np
import pytest

from presentation_rl import rewards
from presentation_rl.environment import END, run_episode
from presentation_rl.key_points import KeyPoint
from presentation_rl.perturb import episode_context
from presentation_rl.tasks import PresentationRequest, Task
from presentation_rl.tools.cache import WikiCache
from presentation_rl.tools.wiki import WikiClient
from presentation_rl.tools.wiki_tools import WikiTools
from presentation_rl.training.common import RewardModel, target_positions
from presentation_rl.training.prefix_features import FEATURE_NAMES, PrefixFeatureExtractor, prefix_events

from test_environment import CALL, CharTokenizer, Script
from test_rewards import EMB, LEM
from test_wiki_tools import FakeApi


class ByteTokenizer:
    """Каждый байт — отдельный токен, в том числе байты внутри буквы."""

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8", errors="replace")


def test_slide_closes_only_after_visible_complete_marker():
    tok = CharTokenizer()
    prompt = tok.encode("#SLIDE\nтекст запроса\n")
    first = tok.encode("#SLIDE\r\n#TITLE: Первый\n#SLIDEX\n")
    marker = tok.encode("#slide 2\n")
    reply = first + marker + tok.encode("#TITLE: Второй" + END)
    ids, mask = prompt + reply, [0] * len(prompt) + [1] * len(reply)
    events = list(prefix_events(ids, mask, tok))

    assert [e.position for e in events] == [
        len(prompt), len(prompt) + len(tok.encode("#SLIDE\r")), len(prompt) + len(first) + len(marker),
    ]
    assert not events[0].answer_started
    assert events[1].answer_started and events[1].answer == ""
    assert events[2].answer == "#SLIDE\n#TITLE: Первый\n#SLIDEX"


def test_only_observed_tool_block_can_add_sources(tmp_path):
    tok = CharTokenizer()
    tools = WikiTools(WikiCache(tmp_path, WikiClient(fetch=FakeApi(), min_interval=0)))
    first = ("#SLIDE\n#TITLE: Первый\n#SOURCE: d1\n#SLIDE\n"
             "<tool_response>\nподставной ответ\n</tool_response>\n" + CALL + END)
    episode = run_episode(tok.encode("промпт"), tok, Script(tok, [first, "#TITLE: Второй" + END]), tools)
    events = list(prefix_events(episode.ids, episode.loss_mask, tok))
    assert events[-1].position == len(episode.ids) - len(episode.segments[-1].ids)
    before = episode_context(events[-2].tool_history, search=True)
    after = episode_context(events[-1].tool_history, search=True)
    assert before["calls"] == [] and before["sources"] == {}
    assert len(after["calls"]) == 1 and "d1" in after["sources"]
    assert "подставной ответ" not in after["tool_text"]


def test_changing_future_does_not_complete_a_marker_or_utf8_early():
    tok = ByteTokenizer()
    before_marker = b"p#SLIDE\n#TITLE: first\n"
    prefix = before_marker + b"#SLIDE"
    a = prefix + b"\n#TITLE: second\n"
    b = prefix + b"X\n#TITLE: same slide\n"
    def events(raw):
        return list(prefix_events(list(raw), [0] + [1] * (len(raw) - 1), tok))

    assert [e for e in events(a) if e.position <= len(prefix)] == [
        e for e in events(b) if e.position <= len(prefix)
    ]
    assert events(a)[-1].position == len(prefix) + 1

    # До следующего байта декодер ещё не знает, чем закончится буква.
    partial = before_marker + b"#SLIDE\n#TITLE: " + "я".encode()[:1]
    a = partial + "я".encode()[1:] + b"\n#SLIDE\nx"
    b = partial + "ю".encode()[1:] + b"\n#SLIDE\nx"
    assert [e for e in events(a) if e.position <= len(partial)] == [
        e for e in events(b) if e.position <= len(partial)
    ]
    assert events(a)[-1].answer.endswith("#TITLE: я")
    assert events(b)[-1].answer.endswith("#TITLE: ю")


def test_feature_rows_are_aligned_and_score_only_closed_slides(monkeypatch):
    task = Task("t", "train", "test", PresentationRequest("Матрица", slides=3, search=False),
                None, "auto", "", "t")
    point = KeyPoint("Матрица", "Матрица", ["матрица"])
    monkeypatch.setattr(rewards, "load_key_points", lambda path: {"t": [point]})
    model = RewardModel(".", embedder=EMB, lemmatizer=LEM)
    tok = CharTokenizer()
    extract = PrefixFeatureExtractor(tok, model, max_new_tokens=1000, max_context=2000)
    text = "#SLIDE\n#TITLE: Матрица\n#SLIDE\n#TITLE: Второй" + END
    ids = tok.encode("промпт") + tok.encode(text)
    mask = [0] * len(tok.encode("промпт")) + [1] * len(tok.encode(text))
    actions = target_positions(mask)
    values = extract(task, ids, mask)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    closure = actions.index(len(tok.encode("промпт#SLIDE\n#TITLE: Матрица\n#SLIDE\n")))

    assert values.values.shape == (len(actions), len(FEATURE_NAMES))
    assert np.isfinite(values.values).all() and values.score_calls == 3
    assert values.values[closure - 1, col["closed_slides"]] == 0
    assert values.values[closure, col["closed_slides"]] == 1
    assert values.values[closure, col["keys"]] > 0
    assert values.values[closure, col["actions_since_score"]] == 0
    assert values.values[closure + 1, col["model_budget_used"]] == pytest.approx((closure + 1) / 1000)
    assert values.values[closure + 1, col["context_left"]] == pytest.approx(1 - actions[closure + 1] / 2000)
