# Здесь реализуем эпизод: генерация с вызовами инструментов, ответы инструментов в контексте, итоговый текст для парсера.
"""Эпизод — один прогон модели на одном задании.

Модель пишет ответ; если в ходе ответа она вызывает инструмент (``<tool_call>{"name": …, "arguments": …}</tool_call>``
и конец реплики), среда выполняет вызов, дописывает результат (``<tool_response>`` в реплике user — формат шаблона
чата T-lite / Qwen3, см. ``configs/models/tlite_21_chat_template.txt``) и модель продолжает новой репликой.

Эпизод хранится токенами, а не текстом: для RL логарифмы вероятностей считаются по тем же токенам, которые модель
сгенерировала (повторная токенизация текста может дать другие токены). Каждый кусок помечен: написала его модель
(входит в loss) или это ответ инструмента и разметка шаблона (не входит).

Генерация передаётся функцией ``generate(ids, max_new_tokens) -> (новые id, закончила ли модель реплику)``, поэтому
среда не зависит от того, чем генерировать (transformers, vLLM, заглушка в тестах).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .tasks import PresentationRequest
from .tools.wiki_tools import TOOL_SPECS, WikiTools

# Разметка шаблона чата T-lite-it-2.1 (как у Qwen3).
END = "<|im_end|>"
TOOL_RESULTS_START = "\n<|im_start|>user"
TOOL_RESULT = "\n<tool_response>\n{}\n</tool_response>"
NEXT_TURN = END + "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"  # как add_generation_prompt шаблона

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ARGUMENTS = {"search": "query", "open": "id", "read": "id"}

Generate = Callable[[list[int], int], tuple[list[int], bool]]


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


@dataclass
class Segment:
    ids: list[int]
    text: str
    trainable: bool  # True — сгенерировала модель (входит в loss); False — ответы инструментов и разметка


@dataclass
class ToolCall:
    name: str
    argument: str
    error: str = ""  # непусто — вызов не разобран, инструмент не вызывался


@dataclass
class Episode:
    prompt_ids: list[int]
    segments: list[Segment] = field(default_factory=list)
    answer: str = ""  # всё, что написала модель, без вызовов инструментов — это получает парсер DSL
    calls: list[dict] = field(default_factory=list)
    finish: str = ""  # stop — модель закончила; length — кончились токены; max_turns — слишком много реплик
    articles: dict = field(default_factory=dict)  # таблица id статей эпизода (для лога и проверки источников)

    @property
    def ids(self) -> list[int]:
        return self.prompt_ids + [i for s in self.segments for i in s.ids]

    @property
    def loss_mask(self) -> list[int]:
        """1 — токен модели (в loss), 0 — промпт, ответы инструментов и разметка."""
        return [0] * len(self.prompt_ids) + [int(s.trainable) for s in self.segments for _ in s.ids]

    @property
    def generated_tokens(self) -> int:
        return sum(len(s.ids) for s in self.segments if s.trainable)


def build_prompt(tokenizer, system_prompt: str, request: PresentationRequest) -> list[int]:
    """Начало эпизода токенами: системный промпт, описания инструментов (если есть поиск) и запрос."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": request.render()}]
    text = tokenizer.apply_chat_template(
        messages, tools=TOOL_SPECS if request.search else None, add_generation_prompt=True, tokenize=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Вызовы инструментов из реплики модели и текст реплики без них."""
    calls: list[ToolCall] = []
    for raw in _CALL_RE.findall(text):
        calls.append(_parse_call(raw))
    clean = _CALL_RE.sub("", text)
    if "<tool_call>" in clean:  # открытый, но не закрытый вызов
        calls.append(ToolCall("", "", "вызов не закрыт: нет </tool_call>"))
        clean = clean[:clean.index("<tool_call>")]
    return clean.strip(), calls


def _parse_call(raw: str) -> ToolCall:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ToolCall("", "", 'не JSON; нужно {"name": …, "arguments": {…}}')
    if not isinstance(data, dict):
        return ToolCall("", "", 'нужен объект {"name": …, "arguments": {…}}')
    name = data.get("name")
    arguments = data.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {_ARGUMENTS.get(name, "value"): arguments}
    if name not in _ARGUMENTS:
        return ToolCall(str(name or ""), "", f"нет инструмента «{name}»; есть search, open, read")
    key = _ARGUMENTS[name]
    if not isinstance(arguments, dict) or not isinstance(arguments.get(key), str):
        return ToolCall(name, "", f"у {name} один аргумент — {key} (строка)")
    return ToolCall(name, arguments[key])


def run_episode(
    prompt_ids: list[int],
    tokenizer: Tokenizer,
    generate: Generate,
    tools: WikiTools | None = None,
    max_new_tokens: int = 8000,
) -> Episode:
    """Генерация с вызовами инструментов до конца ответа, лимита токенов или лимита реплик.

    ``tools=None`` — поиска нет: вызовы не выполняются, текст реплики целиком идёт парсеру (там он станет ошибкой).
    """
    episode = Episode(list(prompt_ids))
    if tools is not None:
        tools.reset()
    ids = list(prompt_ids)
    budget = max_new_tokens
    parts: list[str] = []
    turns = tools.max_calls + 2 if tools is not None else 1
    episode.finish = "max_turns"
    for _ in range(turns):
        new, finished = generate(ids, budget)
        budget -= len(new)
        text = tokenizer.decode(new, skip_special_tokens=False)
        episode.segments.append(Segment(list(new), text, True))
        ids += new
        content = text[:-len(END)] if text.endswith(END) else text
        if tools is None:
            parts.append(content.strip())
            episode.finish = "stop" if finished else "length"
            break
        clean, calls = parse_tool_calls(content)
        parts.append(clean)
        if not finished:
            episode.finish = "length"
            break
        if not calls:
            episode.finish = "stop"
            break
        results = [_execute(tools, call, episode) for call in calls]
        block = TOOL_RESULTS_START + "".join(TOOL_RESULT.format(r) for r in results) + NEXT_TURN
        if not text.endswith(END):  # модель закончила реплику без <|im_end|> (например, по eos) — дописываем
            block = END + "\n" + block.lstrip("\n")
        block_ids = tokenizer.encode(block, add_special_tokens=False)
        episode.segments.append(Segment(block_ids, block, False))
        ids += block_ids
        if budget <= 0:
            episode.finish = "length"
            break
    episode.answer = "\n".join(p for p in parts if p)
    if tools is not None:
        episode.articles = tools.table()
    return episode


def _execute(tools: WikiTools, call: ToolCall, episode: Episode) -> str:
    """Выполняет вызов; неразобранный вызов тоже тратит лимит — модель учится на своих ошибках."""
    if call.error:
        if len(tools.calls) >= tools.max_calls:
            result = f"Лимит вызовов исчерпан ({tools.max_calls}). Пишите презентацию по уже найденному."
        else:
            result = f"Ошибка вызова: {call.error}."
            tools.calls.append({"name": call.name, "argument": call.argument, "chars": len(result), "error": True})
    else:
        result = tools.call(call.name, call.argument)
    episode.calls.append({"name": call.name, "argument": call.argument, "error": call.error, "chars": len(result)})
    return result
