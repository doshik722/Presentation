# Здесь считаю признаки оценки уже закрытых слайдов для критика PPO.
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby

import numpy as np

from ..dsl import ParseLimits, parse
from ..environment import END, TOOL_RESULT, Tokenizer, ToolCall, _ARGUMENTS, _CALL_RE, parse_tool_calls
from ..key_points import expected_theses
from ..perturb import _RESPONSE_RE, episode_context
from ..rendering import build_presentation, pick_theme
from ..rewards import score
from ..tasks import Task
from .common import RewardModel, target_positions

_SLIDE_LINE = re.compile(r"^[^\S\n]*#SLIDE(?![A-Z_])[^\n]*\n", re.MULTILINE | re.IGNORECASE | re.ASCII)
_MAX_LINE = ParseLimits().max_line_length
_COMPONENTS = (
    "total", "coverage", "keys", "length", "visual", "forbidden_images", "format", "text_overflow",
    "bad_sources", "tools", "contradictions", "sources", "slide_overflow", "render_errors", "fake_quotes",
    "repeats", "formula_over", "unrelated_visuals",
)
FEATURE_NAMES = _COMPONENTS + (
    "floored_total", "uncovered_fraction", "closed_slides", "slides_over_low", "slides_over_high",
    "slides_above_high", "expected_low", "expected_high", "theses", "conclusion_expected",
    "search", "images_allowed", "answer_started", "observed_calls", "model_budget_used", "context_left",
    "actions_since_score",
)


@dataclass(frozen=True)
class PrefixEvent:
    """Состояние перед действием в position, то есть по ids[:position]."""

    position: int
    answer: str
    tool_history: str
    answer_started: bool


@dataclass(frozen=True)
class PrefixFeatures:
    """Строки признаков в порядке target_positions(mask) и число вычислений score."""

    values: np.ndarray
    score_calls: int


def _closed_answer(text: str) -> tuple[str, bool]:
    """Оставляю текст до последней завершённой строки с #SLIDE."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = None
    for match in _SLIDE_LINE.finditer(text):
        if len(match[0].strip()) <= _MAX_LINE:
            start = match.start()
    return (text[:start].rstrip(), True) if start is not None else ("", False)


def _observed_exchange(calls: list[ToolCall], block: str) -> str:
    """Сопоставляю вызовы одной реплики с выдачей из следующего блока среды."""
    parts = []
    for call, response in zip(calls, _RESPONSE_RE.findall(block)):
        if call.error:
            body = "{}"  # неразобранный вызов занимает своё место в последовательности
        else:
            body = json.dumps({"name": call.name, "arguments": {_ARGUMENTS[call.name]: call.argument}},
                              ensure_ascii=False).replace("<", r"\u003c")
        parts.extend((f"<tool_call>{body}</tool_call>", TOOL_RESULT.format(response)))
    return "".join(parts)


def prefix_events(ids: list[int], mask: list[int], tokenizer: Tokenizer,
                  newline_tokens: dict[int, bool] | None = None) -> Iterator[PrefixEvent]:
    """Обновляю закрытую часть при видимом #SLIDE и в начале реплики после выдачи.

    В байтовом токенизаторе T-lite перевод строки виден уже при декодировании отдельного токена.
    Само событие и его текст проверяю декодированием целого префикса текущей реплики.
    """
    if len(ids) != len(mask) or (mask and mask[0]):
        raise ValueError("нужна маска той же длины, что ids, с промптом перед первым действием")
    newline_tokens = {} if newline_tokens is None else newline_tokens
    parts: list[str] = []
    calls: list[ToolCall] = []
    tool_history = ""
    start = 0

    for is_action, span in groupby(mask):
        end = start + sum(1 for _ in span)
        if is_action:
            previous = "".join(part + "\n" for part in parts)
            closed = _closed_answer(previous)
            yield PrefixEvent(start, closed[0], tool_history, closed[1])
            for p in range(start + 1, end):
                token = ids[p - 1]
                if token not in newline_tokens:
                    piece = tokenizer.decode([token], skip_special_tokens=False)
                    newline_tokens[token] = "\n" in piece or "\r" in piece
                if not newline_tokens[token]:
                    continue
                text = tokenizer.decode(ids[start:p], skip_special_tokens=False)
                # Сохраняю переводы строк вокруг вызовов. Незакрытый вызов пока не часть ответа.
                clean = _CALL_RE.sub("", text).partition("<tool_call>")[0]
                current = _closed_answer(previous + clean)
                if current != closed:
                    closed = current
                    yield PrefixEvent(p, closed[0], tool_history, closed[1])
            turn = tokenizer.decode(ids[start:end], skip_special_tokens=False).removesuffix(END)
            answer, calls = parse_tool_calls(turn)
            if answer:
                parts.append(answer)
        elif start:
            tool_history += _observed_exchange(calls, tokenizer.decode(ids[start:end], skip_special_tokens=False))
        start = end


class PrefixFeatureExtractor:
    """Считаю score по закрытым слайдам с отчётом их сборки и видимой на тот момент выдачей."""

    def __init__(self, tokenizer: Tokenizer, reward: RewardModel, *, max_new_tokens: int, max_context: int):
        if min(max_new_tokens, max_context) <= 0:
            raise ValueError("лимиты токенов должны быть положительными")
        self.tokenizer, self.reward = tokenizer, reward
        self.max_new_tokens, self.max_context = max_new_tokens, max_context
        self._newline_tokens: dict[int, bool] = {}

    def __call__(self, task: Task, ids: list[int], mask: list[int]) -> PrefixFeatures:
        actions = np.asarray(target_positions(mask), dtype=np.int64)
        values = np.empty((len(actions), len(FEATURE_NAMES)), dtype=np.float32)
        events = list(prefix_events(ids, mask, self.tokenizer, self._newline_tokens))
        if not events:
            return PrefixFeatures(values, 0)

        request = task.request
        search = request.search
        points = self.reward.points[task.id]
        conclusion = expected_theses(task)[2]
        slide_range = request.slide_range()
        theme = pick_theme(request.theme, request.text)
        starts = np.searchsorted(actions, [event.position for event in events]).tolist() + [len(actions)]
        previous_history = None

        for i, event in enumerate(events):
            if event.tool_history != previous_history:
                context = episode_context(event.tool_history, search)
                previous_history = event.tool_history
            deck = parse(event.answer)
            report = build_presentation(deck, None, theme, sources=context["sources"].get,
                                        show_sources=request.sources)
            result = score(deck, points, conclusion=conclusion, slide_range=slide_range, report=report,
                           embedder=self.reward.embedder, lemmatizer=self.reward.lemmatizer,
                           images_allowed=request.images, tool_text=context["tool_text"], calls=context["calls"],
                           search=search, weights=self.reward.weights)
            components = result.components()
            low, high = result.expected_slides
            n = len(deck.slides)
            block = values[starts[i]:starts[i + 1]]
            block[:, :-3] = (
                *(components[name] for name in _COMPONENTS),
                max(result.total, self.reward.floor), result.uncovered / max(1, len(points)), n,
                n / max(1, low), n / max(1, high), max(0, n - high), low, high, len(points), conclusion,
                search, request.images, event.answer_started, len(context["calls"]),
            )
            block[:, -1] = np.arange(len(block))

        values[:, -3] = np.arange(len(actions)) / self.max_new_tokens
        values[:, -2] = np.maximum(0, self.max_context - actions) / self.max_context
        return PrefixFeatures(values, len(events))
