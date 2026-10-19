# Здесь проверяем эпизод с вызовами инструментов без модели: токенизатор посимвольный, ответы модели заданы заранее.
import pytest

from presentation_rl.environment import END, NEXT_TURN, parse_tool_calls, run_episode
from presentation_rl.tools.cache import WikiCache
from presentation_rl.tools.wiki import WikiClient
from presentation_rl.tools.wiki_tools import WikiTools

from test_wiki_tools import FakeApi

SPECIAL = ["<|im_start|>", "<|im_end|>"]


class CharTokenizer:
    """Посимвольный токенизатор; специальные токены — один id, как у настоящего."""

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for n, token in enumerate(SPECIAL):
                if text.startswith(token, i):
                    ids.append(n)
                    i += len(token)
                    break
            else:
                ids.append(ord(text[i]) + 10)
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(SPECIAL[i] if i < 10 else chr(i - 10) for i in ids)


class Script:
    """Модель-заглушка: по очереди отдаёт заранее заданные реплики (с <|im_end|>, если реплика закончена)."""

    def __init__(self, tokenizer, replies):
        self.tokenizer = tokenizer
        self.replies = list(replies)
        self.inputs = []

    def __call__(self, ids, max_new_tokens):
        self.inputs.append(list(ids))
        reply = self.replies.pop(0)
        new = self.tokenizer.encode(reply)[:max_new_tokens]
        return new, reply.endswith(END) and len(new) == len(self.tokenizer.encode(reply))


@pytest.fixture
def tools(tmp_path):
    return WikiTools(WikiCache(tmp_path, WikiClient(fetch=FakeApi(), min_interval=0)))


CALL = '<tool_call>\n{"name": "search", "arguments": {"query": "градиентный спуск"}}\n</tool_call>'


def test_parse_tool_calls():
    clean, calls = parse_tool_calls("#PLAN: ищу\n" + CALL)
    assert clean == "#PLAN: ищу" and (calls[0].name, calls[0].argument, calls[0].error) == ("search", "градиентный спуск", "")
    _, calls = parse_tool_calls('<tool_call>{"name": "read", "arguments": "{\\"id\\": \\"d1#2\\"}"}</tool_call>')
    assert (calls[0].name, calls[0].argument) == ("read", "d1#2")
    assert parse_tool_calls("<tool_call>не json</tool_call>")[1][0].error
    assert "нет инструмента" in parse_tool_calls('<tool_call>{"name": "web", "arguments": {}}</tool_call>')[1][0].error
    clean, calls = parse_tool_calls("#SLIDE\n<tool_call>{\"name\": \"search\"")
    assert clean == "#SLIDE" and "не закрыт" in calls[0].error


def test_episode_with_search_open_and_answer(tools):
    tok = CharTokenizer()
    prompt = tok.encode("<|im_start|>user\nзапрос" + NEXT_TURN)
    script = Script(tok, [
        "#PLAN: ищу\n" + CALL + END,
        '<tool_call>\n{"name": "open", "arguments": {"id": "d1"}}\n</tool_call>' + END,
        "#SLIDE\n#TITLE: Градиентный спуск\n#SOURCE: d1#0" + END,
    ])
    episode = run_episode(prompt, tok, script, tools)
    assert episode.finish == "stop"
    assert episode.answer == "#PLAN: ищу\n#SLIDE\n#TITLE: Градиентный спуск\n#SOURCE: d1#0"
    assert [c["name"] for c in episode.calls] == ["search", "open"]
    assert [s.trainable for s in episode.segments] == [True, False, True, False, True]
    # Модель видит ответ инструмента в формате шаблона T-lite и продолжает новой репликой.
    second_input = tok.decode(script.inputs[1])
    assert second_input.endswith("</tool_call>" + END + "\n<|im_start|>user\n<tool_response>\nРезультаты поиска «градиентный спуск»:"
                                 + second_input.split("Результаты поиска «градиентный спуск»:")[1])
    assert "</tool_response>" + NEXT_TURN in second_input
    # Маска: токены модели — 1, промпт и ответы инструментов — 0; последовательность — ровно то, что видела модель.
    assert len(episode.ids) == len(episode.loss_mask)
    assert sum(episode.loss_mask) == episode.generated_tokens
    assert episode.ids[:len(script.inputs[2])] == script.inputs[2]
    assert tools.valid_source("d1#0") and episode.articles["d1"]["opened"]


def test_bad_call_costs_a_call_and_returns_error(tools):
    tok = CharTokenizer()
    tools.max_calls = 1
    script = Script(tok, ["<tool_call>не json</tool_call>" + END, CALL + END, "#SLIDE" + END])
    episode = run_episode(tok.encode("п" + NEXT_TURN), tok, script, tools)
    assert "Ошибка вызова" in episode.segments[1].text
    assert "Лимит вызовов исчерпан" in episode.segments[3].text
    assert episode.answer == "#SLIDE" and episode.finish == "stop"


def test_length_limit_and_no_tools():
    tok = CharTokenizer()
    episode = run_episode(tok.encode("п"), tok, Script(tok, ["#SLIDE\n#TITLE: очень длинный" + END]), None, max_new_tokens=8)
    assert episode.finish == "length" and episode.answer == "#SLIDE\n#"
    episode = run_episode(tok.encode("п"), tok, Script(tok, [CALL + END]), None)
    assert episode.finish == "stop" and episode.answer.startswith("<tool_call>")  # без поиска вызов — просто текст
