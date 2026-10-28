# Здесь проверяем и собираем примеры SFT с поиском: эпизоды составителей — реплики модели с вызовами и выдача программы.
"""Сборка примеров SFT с поиском (``data/tasks/sft_search``).

Составитель пишет эпизод целиком (промпт — ``sft_search_prompt.txt``): реплики модели с ``#PLAN`` и вызовами
инструментов, ответы программы и последнюю реплику — презентацию. Файлы — как у примеров без поиска
(``sft_data.py``): ``assignment.json``, ``<модель>/answer<k>.txt`` на пакет k, исправления — ``answer<k>_fix.txt``;
задания без картинок — ``data/tasks/sft/offline.json``. Эпизод — в блоке кода под строкой ``### id``; реплика модели
начинается строкой ``[модель]``, ответ программы на один вызов — строкой ``[программа]``.

Ошибки (пример в набор не идёт):

- строение эпизода: число ответов программы не равно числу вызовов реплики, вызов не разобран, реплика без вызовов
  не последняя, в последней реплике есть вызов, больше ``MAX_CALLS`` вызовов, повтор вызова слово в слово;
- выдача не в формате наших инструментов: заголовок, id статей не по порядку первого появления, ``open``/``read``
  статьи, которой не было в выдаче, раздел, номера которого модель не видела, заголовок раздела не как в оглавлении;
- презентация: ошибки разбора, формула не рисуется, ``#IMAGE`` при «Картинки: нет», ``#SOURCE`` с id, которого
  не было в выдаче (как ``WikiTools.valid_source``), цитата, которой нет в выдаче.

Если ответов программы меньше, чем вызовов (составитель поставил одну метку ``[программа]`` на несколько ответов),
блок режется по первым строкам ответов наших инструментов — «Результаты поиска «…»:», «d3 | ru | Название»,
«d1#5 | ru | Статья — Раздел»; части должны совпасть с вызовами по числу и порядку. Текст выдачи не меняется,
пишется замечание.

Ссылки Markdown (``[текст](https://…)``) программа убирает сама — остаётся текст, у заголовка выдачи поиска
пропадает приписка со ссылкой — и пишет замечание: модель не видит адресов и не должна учиться их писать.
Оставшийся адрес в реплике модели — ошибка.

Замечания (пример идёт, но стоит посмотреть): числа со слайда с ``#SOURCE`` (кроме ``#FORMULA`` и ``#PLOT``)
не нашлись в тексте, на который он ссылается; пределы визуала из награды превышены; нет ``#PLAN``; нет заключения; число слайдов не как в запросе;
ответ программы длиннее ``OUTPUT_LIMIT``; нет ни одного вызова.

Настоящая выдача (решение 29.10.2026). Составитель пишет вызовы и презентацию, а его выдача — набросок: Codex
приводит настоящий текст, но обрезает разделы до одной-двух фраз. ``substitute`` повторяет вызовы эпизода через
настоящие инструменты (``WikiTools``) и подставляет их ответы — ровно то, что модель увидит при RL. Заодно сверяет:
статьи выдачи поиска составителя должны найтись в настоящей (id составителя переводятся в настоящие во всех
репликах — в вызовах, ``#PLAN`` и ``#SOURCE``), заголовки оглавления и разделов — совпасть, каждый кусок текста
составителя (между многоточиями и пометками обрезки) — найтись в настоящем ответе. Эпизод с расхождением в набор не идёт. После подстановки эпизод проверяется
заново — ``#SOURCE``, числа и цитаты уже по настоящей выдаче.

``replay`` прогоняет проверенный эпизод через ``environment.run_episode`` — той же функцией, что и роллауты RL:
реплики модели подаются как сгенерированные, выдача — как ответы инструментов. Так пример SFT получает ровно
ту разметку чата и ту маску loss, что и эпизоды при обучении с подкреплением.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .dsl import LARGE_VISUALS, parse
from .environment import END, Episode, ToolCall, parse_tool_calls, run_episode
from .rendering.builder import MAX_FORMULAS, MAX_LARGE_VISUALS
from .rewards import FORMULA_TOTAL_PER_SLIDE, VISUAL_TOTAL_PER_SLIDE, _normalize_quote, quote_found, type_limit
from .sft_data import MODELS, _broken_formulas, split_answer
from .tasks import Task, load_tasks

MAX_CALLS = 10  # как WikiTools.max_calls
OUTPUT_LIMIT = 1700  # 1500 символов текста по промпту составителей и строка заголовка
OTHER_VISUALS = ("CHART", "PLOT", "DIAGRAM", "TIMELINE", "IMAGE", "QUOTE")  # визуал награды без формул
CONCLUSION_RE = re.compile(r"заключ|вывод|итог|резюме|подвед", re.IGNORECASE)

_MARK_RE = re.compile(r"^\[(модель|программа)\]$")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://(?:[^()\s]|\([^()\s]*\))*\)")
_SEARCH_HEAD_RE = re.compile(r"^(Результаты поиска «.*?»)\s*\([^()]*\):\s*$", re.MULTILINE)
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_TOC_INDENT_RE = re.compile(r"^([ \t]+)(d\d+#\d+) ", re.MULTILINE)  # «  d4#3 Раздел» → «d4#3   Раздел», как у open
# Первая строка ответа каждого инструмента — по ним режется склеенный блок [программа].
_OUTPUT_HEADS = (
    ("search", re.compile(r"^(?:Результаты поиска «.*»:|По запросу «.*» ничего не найдено.*)$")),
    ("open", re.compile(r"^d\d+ \| (?:ru|en) \| [^|]+$")),  # оглавление: три поля (у строки выдачи поиска — четыре)
    ("read", re.compile(r"^d\d+#\d+ \| (?:ru|en) \| .+$")),
)
_RESULT_RE = re.compile(r"^d(\d+) \| (ru|en) \| (.+?) \| (.*)$")
_ARTICLE_RE = re.compile(r"^d(\d+)$")
_SECTION_RE = re.compile(r"^d(\d+)(?:#(\d+))?$")


@dataclass
class Turn:
    role: str  # model | program
    text: str


@dataclass
class Example:
    id: str
    model: str
    batch: str
    images: bool
    request: str
    turns: list[Turn]
    answer: str  # то, что получит парсер DSL: все реплики модели без вызовов
    fixed: bool = False
    comment: str = ""
    english: bool = False  # открыта или прочитана статья на английском


@dataclass
class Report:
    examples: list[Example] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    waiting: list[str] = field(default_factory=list)
    searches: dict[str, list[int]] = field(default_factory=dict)  # составитель → [выдач поиска, из них без другого языка]

    def text(self) -> str:
        lines = [f"Примеров: {len(self.examples)}; ошибок: {len(self.errors)}; замечаний: {len(self.notes)}"]
        for model in sorted({e.model for e in self.examples} | set(self.searches)):
            mine = [e for e in self.examples if e.model == model]
            total, one_lang = self.searches.get(model, [0, 0])
            lines.append(f"{model}: английская статья открыта или прочитана в {sum(e.english for e in mine)} "
                         f"из {len(mine)} примеров; выдач поиска без статей на другом языке — {one_lang} из {total}")
        if self.waiting:
            lines.append("Ещё нет ответа: " + ", ".join(self.waiting))
        lines += ["Ошибка: " + e for e in self.errors]
        lines += ["Замечание: " + n for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------- строение эпизода


def split_turns(text: str) -> tuple[list[Turn], bool]:
    """Реплики эпизода по строкам ``[модель]`` и ``[программа]``; второе — был ли текст до первой метки."""
    turns: list[Turn] = []
    role, lines, stray = None, [], False
    for line in text.splitlines():
        if match := _MARK_RE.match(line.strip()):
            if role:
                turns.append(Turn(role, "\n".join(lines).strip()))
            role, lines = ("model" if match.group(1) == "модель" else "program"), []
        elif role is None:
            stray = stray or bool(line.strip())
        else:
            lines.append(line)
    if role:
        turns.append(Turn(role, "\n".join(lines).strip()))
    return turns, stray


@dataclass
class _Step:
    raw: str  # реплика модели как есть
    clean: str  # реплика без вызовов
    calls: list[ToolCall]
    outputs: list[str]


def _output_kind(line: str) -> str | None:
    return next((name for name, pattern in _OUTPUT_HEADS if pattern.match(line.strip())), None)


def _split_outputs(outputs: list[str], calls: list[ToolCall]) -> list[str] | None:
    """Режет склеенные ответы программы по их первым строкам; None — части не совпали с вызовами."""
    pieces: list[list[str]] = []
    for block in outputs:
        for k, line in enumerate(block.splitlines()):
            if k == 0 or _output_kind(line):
                pieces.append([])
            pieces[-1].append(line)
    texts = ["\n".join(p).strip() for p in pieces]
    if len(texts) != len(calls):
        return None
    if [_output_kind(t.splitlines()[0]) for t in texts] != [c.name for c in calls]:
        return None
    return texts


def _steps(turns: list[Turn], errors: list[str], notes: list[str] | None = None) -> list[_Step]:
    steps: list[_Step] = []
    i = 0
    if not turns or turns[0].role != "model":
        errors.append("эпизод начинается не с реплики модели")
        return steps
    while i < len(turns):
        clean, calls = parse_tool_calls(turns[i].text)
        for call in calls:
            if call.error:
                errors.append(f"реплика {len(steps) + 1}: вызов не разобран — {call.error}")
        j = i + 1
        outputs = []
        while j < len(turns) and turns[j].role == "program":
            outputs.append(turns[j].text)
            j += 1
        if len(outputs) < len(calls) and outputs and (split := _split_outputs(outputs, calls)) is not None:
            if notes is not None:
                notes.append(f"реплика {len(steps) + 1}: ответы программы разделены ({len(outputs)} блока → {len(split)})")
            outputs = split
        if len(outputs) != len(calls):
            errors.append(f"реплика {len(steps) + 1}: вызовов {len(calls)}, ответов программы {len(outputs)}")
        steps.append(_Step(turns[i].text, clean, calls, outputs))
        i = j
    for k, step in enumerate(steps[:-1], 1):
        if not step.calls:
            errors.append(f"реплика {k} без вызовов, а эпизод на ней не кончается")
    if steps and steps[-1].calls:
        errors.append("в последней реплике есть вызов: презентации после него нет")
    return steps


# --------------------------------------------------------------------------- выдача программы


@dataclass
class _Doc:
    number: int
    lang: str
    title: str
    snippet: str
    toc: dict[int, str] = field(default_factory=dict)  # раздел → заголовок (после open)
    texts: dict[int, str] = field(default_factory=dict)  # раздел → текст, который модель видела
    read: set[int] = field(default_factory=set)


class _Ledger:
    """Id эпизода, как их выдаёт WikiTools: статьи по порядку первого появления, разделы — по оглавлению."""

    def __init__(self):
        self.docs: list[_Doc] = []
        self.searches = 0
        self.one_language = 0  # выдачи поиска без версий статей на другом языке
        self.english = False

    def by_ref(self, number: int) -> _Doc | None:
        return self.docs[number - 1] if 1 <= number <= len(self.docs) else None

    def check(self, call: ToolCall, output: str, where: str, errors: list[str], notes: list[str]) -> None:
        lines = output.splitlines()
        if not lines:
            errors.append(f"{where}: пустой ответ программы")
            return
        if len(output) > OUTPUT_LIMIT:
            notes.append(f"{where}: ответ программы {len(output)} символов (по промпту — до 1500)")
        argument = call.argument.strip()
        handler = {"search": self._search, "open": self._open, "read": self._read}[call.name]
        handler(argument, lines, where, errors, notes)

    def _search(self, query: str, lines: list[str], where: str, errors: list[str], notes: list[str]) -> None:
        if lines[0].startswith("По запросу"):
            notes.append(f"{where}: поиск ничего не нашёл — в образцах запросы должны быть удачными")
            return
        if lines[1:] and all(not line.strip() or "не найден" in line for line in lines[1:]):
            notes.append(f"{where}: поиск ничего не нашёл — в образцах запросы должны быть удачными")
            return
        if lines[0] != f"Результаты поиска «{query}»:":
            errors.append(f"{where}: заголовок выдачи поиска «{lines[0][:60]}»")
        query_lang = "ru" if _CYRILLIC_RE.search(query) else "en"
        self.searches += 1
        if not any(f" | {'en' if query_lang == 'ru' else 'ru'} | " in line for line in lines[1:]):
            self.one_language += 1
        for line in lines[1:]:
            match = _RESULT_RE.match(line)
            if not match:
                errors.append(f"{where}: строка выдачи не в формате «id | язык | название | начало»: «{line[:60]}»")
                continue
            number, lang, title, snippet = int(match.group(1)), match.group(2), match.group(3), match.group(4)
            known = next((d for d in self.docs if (d.lang, d.title) == (lang, title)), None)
            expected = known.number if known else len(self.docs) + 1
            if number != expected:
                errors.append(f"{where}: у статьи «{title}» id d{number}, программа дала бы d{expected}")
                continue
            if known is None:
                self.docs.append(_Doc(number, lang, title, snippet))
            if len(snippet) > 210:
                notes.append(f"{where}: начало введения у d{number} длиннее 200 символов")

    def _open(self, ref: str, lines: list[str], where: str, errors: list[str], notes: list[str]) -> None:
        match = _ARTICLE_RE.match(ref)
        doc = self.by_ref(int(match.group(1))) if match else None
        if doc is None:
            errors.append(f"{where}: open {ref} — такой статьи в выдаче не было")
            return
        if lines[0] != f"d{doc.number} | {doc.lang} | {doc.title}":
            errors.append(f"{where}: заголовок оглавления «{lines[0][:60]}» вместо «d{doc.number} | {doc.lang} | {doc.title}»")
        head_re = re.compile(rf"^d{doc.number}#(\d+) (\s*)(.+)$")
        current = None
        toc: dict[int, str] = {}
        for line in lines[1:]:
            if match := head_re.match(line):
                index = int(match.group(1))
                if index != len(toc):
                    errors.append(f"{where}: раздел d{doc.number}#{index} не по порядку оглавления")
                toc[index] = match.group(3).strip()
                current = index
            elif current is not None:
                doc.texts[current] = (doc.texts.get(current, "") + "\n" + line).strip()
            else:
                errors.append(f"{where}: строка до первого раздела оглавления: «{line[:60]}»")
        if not toc:
            errors.append(f"{where}: в оглавлении нет разделов")
        elif toc.get(0) != "Введение":
            errors.append(f"{where}: раздел #0 должен быть «Введение»")
        doc.toc = toc
        self.english = self.english or doc.lang == "en"

    def _read(self, ref: str, lines: list[str], where: str, errors: list[str], notes: list[str]) -> None:
        match = _SECTION_RE.match(ref)
        doc = self.by_ref(int(match.group(1))) if match else None
        if doc is None:
            errors.append(f"{where}: read {ref} — такой статьи в выдаче не было")
            return
        index = int(match.group(2) or 0)
        if index and not doc.toc:
            errors.append(f"{where}: read {ref} — оглавление статьи не открыто, номер раздела модели неизвестен")
            return
        if doc.toc and index not in doc.toc:
            errors.append(f"{where}: read {ref} — такого раздела в оглавлении нет")
            return
        heading = doc.toc.get(index, "Введение")
        expected = f"d{doc.number}#{index} | {doc.lang} | {doc.title} — {heading}"
        if lines[0] != expected:
            errors.append(f"{where}: заголовок раздела «{lines[0][:70]}» вместо «{expected}»")
        doc.texts[index] = "\n".join(lines[1:])
        doc.read.add(index)
        self.english = self.english or doc.lang == "en"

    # --- источники

    def source_text(self, ref: str) -> str | None:
        """Текст, на который может ссылаться ``#SOURCE``; None — id недопустим (как ``WikiTools.valid_source``)."""
        match = _SECTION_RE.match(ref.strip())
        doc = self.by_ref(int(match.group(1))) if match else None
        if doc is None:
            return None
        if match.group(2) is None:  # статья: начало введения из выдачи поиска и введение, если открыта
            return doc.snippet + "\n" + doc.texts.get(0, "")
        index = int(match.group(2))
        if index in doc.read or index in doc.toc:  # заголовок раздела модель тоже видела («Праведный халифат (632—661)»)
            return doc.toc.get(index, "") + "\n" + doc.texts.get(index, "")
        return None


# --------------------------------------------------------------------------- проверка презентации


def _numbers(text: str) -> set[str]:
    """Числа от двух цифр («6 400» → 6400, «1,5» → 1.5)."""
    text = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", text)
    found = set()
    for raw in re.findall(r"\d+(?:[.,]\d+)?", text):
        value = raw.replace(",", ".")
        if len(value.replace(".", "")) >= 2:
            found.add(value)
    return found


def _check_deck(deck, ledger: _Ledger, outputs: str, images: bool, where: str,
                errors: list[str], notes: list[str]) -> None:
    if not images and any(e.kind == "IMAGE" for s in deck.slides for e in s.elements):
        errors.append(f"{where}: #IMAGE при «Картинки: нет»")
    quotes = _normalize_quote(outputs)
    for number, slide in enumerate(deck.slides, 1):
        for element in slide.elements_of("QUOTE"):
            if not quote_found(element.data.get("text", ""), quotes):
                errors.append(f"{where}: слайд {number} — цитаты нет в выдаче (награда её не засчитает)")
        cited = []
        for ref in slide.sources:
            text = ledger.source_text(ref)
            if text is None:
                errors.append(f"{where}: слайд {number} — #SOURCE {ref}: такого id в выдаче не было")
            else:
                cited.append(text)
        if cited:
            # в формулах и выражениях графиков — константы и параметры, а не факты источника
            # у #CHART списки через запятую, дробная часть — через точку: запятая там не десятичная
            shown = " ".join([slide.title or ""] + [e.raw.replace(",", " ") if e.kind == "CHART" else e.raw
                                                    for e in slide.elements if e.kind not in ("FORMULA", "PLOT")])
            missing = sorted(_numbers(shown) - _numbers("\n".join(cited)))
            if missing:
                notes.append(f"{where}: слайд {number} — чисел нет в источнике: {', '.join(missing)}")
        large = sum(e.kind in LARGE_VISUALS for e in slide.elements)
        formulas = len(slide.elements_of("FORMULA"))
        if large > MAX_LARGE_VISUALS or formulas > MAX_FORMULAS:
            notes.append(f"{where}: слайд {number} — крупных элементов {large}, формул {formulas} (сборщик нарисует 2 и 3)")
    content = max(1, len(deck.slides) - 2) if len(deck.slides) > 3 else max(1, len(deck.slides) - 1)
    counts = {kind: sum(len(s.elements_of(kind)) for s in deck.slides) for kind in OTHER_VISUALS + ("FORMULA",)}
    over = {kind: c for kind, c in counts.items() if kind != "FORMULA" and c > type_limit(kind, content)}
    if over:
        notes.append(f"{where}: элементов одного типа больше предела (⌈0,5·N⌉, картинок — N; N = {content}): "
                     + ", ".join(f"{k} {c}" for k, c in over.items()))
    total = sum(c for kind, c in counts.items() if kind != "FORMULA")
    if total > VISUAL_TOTAL_PER_SLIDE * content:
        notes.append(f"{where}: визуальных элементов {total}, предел 2·N = {VISUAL_TOTAL_PER_SLIDE * content}")
    if counts["FORMULA"] > FORMULA_TOTAL_PER_SLIDE * content:
        notes.append(f"{where}: формул {counts['FORMULA']}, предел 2·N = {FORMULA_TOTAL_PER_SLIDE * content}")
    if len(deck.slides) > 3 and not CONCLUSION_RE.search(deck.slides[-1].title or ""):
        notes.append(f"{where}: последний слайд «{deck.slides[-1].title}» — заключение ли это?")


def unlink(text: str) -> tuple[str, int]:
    """Убирает ссылки Markdown: ``[текст](адрес)`` → ``текст``; у заголовка выдачи поиска — приписку со ссылкой."""
    text, links = _MD_LINK_RE.subn(r"\1", text)
    text, heads = _SEARCH_HEAD_RE.subn(r"\1:", text)
    return text, links


def fix_toc_indent(text: str) -> tuple[str, int]:
    """Отступ подраздела в оглавлении — после id, как у ``WikiTools.open``, а не перед ним; введение и у английских
    статей называется «Введение» (так его называет инструмент), а не «Introduction»."""
    text, moved = _TOC_INDENT_RE.subn(lambda m: m.group(2) + " " + m.group(1).replace("\t", "  "), text)
    text, intro = re.subn(r"^(d\d+#0 |d\d+#0 \| en \| .+ — )Introduction$", r"\1Введение", text, flags=re.MULTILINE)
    return text, moved + intro


def check_episode(text: str, task: Task, images: bool, where: str, ledger: "_Ledger | None" = None,
                  real: bool = False) -> tuple[list[Turn], str, list[str], list[str]]:
    """Проверяет эпизод составителя: реплики, ответ для парсера, ошибки, замечания."""
    errors: list[str] = []
    notes: list[str] = []
    text, links = unlink(text)
    if links:
        notes.append(f"{where}: убраны ссылки Markdown — {links}")
    text, moved = fix_toc_indent(text)
    if moved:
        notes.append(f"{where}: оглавление приведено к виду инструмента (отступы, «Введение») — {moved}")
    turns, stray = split_turns(text)
    for k, turn in enumerate(t for t in turns if t.role == "model"):
        if re.search(r"https?://", turn.text):
            errors.append(f"{where}: в реплике модели {k + 1} адрес ссылки")
    if stray:
        errors.append(f"{where}: текст до первой метки [модель]")
    steps = _steps(turns, errors, notes)
    # реплики после разрезания склеенных ответов: каждому вызову — свой ответ программы
    turns = [t for s in steps for t in [Turn("model", s.raw)] + [Turn("program", o) for o in s.outputs]]
    errors[:] = [e if e.startswith(where) else f"{where}: {e}" for e in errors]
    notes[:] = [n if n.startswith(where) else f"{where}: {n}" for n in notes]
    ledger = ledger if ledger is not None else _Ledger()
    seen_calls = set()
    calls = 0
    for k, step in enumerate(steps, 1):
        for call, output in zip(step.calls, step.outputs):
            calls += 1
            key = (call.name, call.argument.strip())
            if key in seen_calls:
                errors.append(f"{where}: повтор вызова {call.name}({call.argument})")
            seen_calls.add(key)
            if not call.error:
                found: list[str] = []
                ledger.check(call, output, f"{where}: реплика {k}, {call.name}({call.argument})", errors, found)
                notes += [n for n in found if not (real and "символов (по промпту" in n)]
    if calls > MAX_CALLS:
        errors.append(f"{where}: вызовов {calls}, предел {MAX_CALLS}")
    if calls == 0:
        notes.append(f"{where}: в эпизоде нет ни одного вызова")
    answer = "\n".join(s.clean for s in steps if s.clean)
    deck = parse(answer)
    errors += [f"{where}: {e.code} (строка {e.line})" for e in deck.errors]
    errors += [f"{where}: формула не рисуется (строка {line})" for line in _broken_formulas(deck)]
    outputs = "\n".join(o for s in steps for o in s.outputs)
    _check_deck(deck, ledger, outputs, images, where, errors, notes)
    if not deck.plan:
        notes.append(f"{where}: нет #PLAN")
    need = task.request.slide_range()
    if need and not need[0] <= len(deck.slides) <= need[1]:
        notes.append(f"{where}: слайдов {len(deck.slides)}, в запросе {need[0]}–{need[1]}")
    return turns, answer, errors, notes


def collect(sft_dir: str | Path, tasks_path: str | Path, offline_path: str | Path) -> Report:
    """Собирает все эпизоды, на которые уже есть ответы, и проверяет их."""
    sft_dir = Path(sft_dir)
    tasks = {t.id: t for t in load_tasks(tasks_path, split="sft")}
    plan = json.loads((sft_dir / "assignment.json").read_text(encoding="utf-8"))
    no_images = set(json.loads(Path(offline_path).read_text(encoding="utf-8")))
    report = Report()
    for model in MODELS:
        for batch, ids in plan.get(model, {}).items():
            k = batch.removeprefix("batch")
            main = sft_dir / model / f"answer{k}.txt"
            if not main.exists() or main.stat().st_size == 0:
                report.waiting.append(f"{model}/{batch}")
                continue
            parts = split_answer(main.read_text(encoding="utf-8"), with_comments=True)
            fix = sft_dir / model / f"answer{k}_fix.txt"
            fixed = split_answer(fix.read_text(encoding="utf-8"), with_comments=True) if fix.exists() else {}
            for extra in sorted(set(fixed) - set(parts)):
                report.errors.append(f"{model}/{batch}: исправление {extra} — такого задания в ответе нет")
            parts.update(fixed)
            for extra in sorted(set(parts) - set(ids)):
                report.errors.append(f"{model}/{batch}: лишний id {extra}")
            for task_id in ids:
                where = f"{model}/{batch}/{task_id}"
                if task_id not in parts:
                    report.errors.append(f"{where}: нет эпизода")
                    continue
                text, comment = parts[task_id]
                images = task_id not in no_images
                ledger = _Ledger()
                turns, answer, errors, notes = check_episode(text, tasks[task_id], images, where, ledger)
                report.notes += notes
                counts = report.searches.setdefault(model, [0, 0])
                counts[0] += ledger.searches
                counts[1] += ledger.one_language
                if errors:
                    report.errors += errors
                    continue
                request = replace(tasks[task_id].request, search=True, images=images).render()
                report.examples.append(Example(task_id, model, batch, images, request, turns, answer,
                                               task_id in fixed, comment, ledger.english))
    report.examples.sort(key=lambda e: e.id)
    return report


# --------------------------------------------------------------------------- настоящая выдача

_ID_RE = re.compile(r"\bd(\d+)(?=#\d+|\b)")


def _remap(text: str, mapping: dict[int, int]) -> str:
    return _ID_RE.sub(lambda m: f"d{mapping.get(int(m.group(1)), int(m.group(1)))}", text)


def _plain(text: str) -> str:
    """Для сравнения текстов: без ударений, пометок обрезки, многоточий и лишних пробелов."""
    text = text.replace("\u0301", "")
    text = re.sub(r"\[(обрезано|дальше)[^\]]*\]", " ", text)
    return " ".join(text.replace("…", " ").split())


def _missing_fragments(teacher_body: str, real: str) -> list[str]:
    """Куски текста составителя (между многоточиями и пометками обрезки), которых нет в настоящей выдаче."""
    real = _plain(real)
    pieces = re.split(r"…|\[(?:обрезано|дальше)[^\]]*\]|\n", teacher_body.replace("\u0301", ""))
    return [p for p in (_plain(piece) for piece in pieces) if len(p) >= 15 and p not in real]


def _results(text: str) -> list[tuple[int, str, str]]:
    return [(int(m.group(1)), m.group(2), m.group(3)) for line in text.splitlines()[1:]
            if (m := _RESULT_RE.match(line))]


def _toc(text: str) -> dict[int, str]:
    return {int(m.group(1)): m.group(2).strip() for line in text.splitlines()[1:]
            if (m := re.match(r"^d\d+#(\d+) (.+)$", line))}


def _compare(call: ToolCall, teacher: str, real: str, mapping: dict[int, int], where: str) -> list[str]:
    """Расхождения выдачи составителя с настоящей; для поиска пополняет перевод id составителя в настоящие."""
    problems = []
    if call.name == "search":
        found = {(lang, title): number for number, lang, title in _results(real)}
        for number, lang, title in _results(teacher):
            if (lang, title) in found:
                mapping[number] = found[(lang, title)]
            else:
                problems.append(f"{where}: в настоящей выдаче нет статьи «{title}» ({lang})")
    elif call.name == "open":
        if teacher.splitlines()[0] != real.splitlines()[0]:
            problems.append(f"{where}: оглавление другой статьи: «{real.splitlines()[0]}»")
        real_toc = _toc(real)
        for index, heading in _toc(teacher).items():
            if real_toc.get(index) != heading:
                problems.append(f"{where}: раздел #{index} «{heading}», в настоящем оглавлении — «{real_toc.get(index)}»")
        texts = "\n".join(line for line in teacher.splitlines()[1:] if not re.match(r"^d\d+#\d+ ", line))
        for piece in _missing_fragments(texts, real):
            problems.append(f"{where}: текста составителя нет в настоящем оглавлении: «{piece[:60]}»")
    else:  # read
        if teacher.splitlines()[0] != real.splitlines()[0]:
            problems.append(f"{where}: заголовок «{teacher.splitlines()[0][:70]}», настоящий — «{real.splitlines()[0][:70]}»")
        for piece in _missing_fragments("\n".join(teacher.splitlines()[1:]), real):
            problems.append(f"{where}: текста составителя нет в настоящем разделе: «{piece[:60]}»")
    return problems


def episode_text(turns: list[Turn]) -> str:
    return "\n".join(("[модель]" if t.role == "model" else "[программа]") + "\n" + t.text for t in turns)


def substitute(example: Example, tools, task: Task) -> tuple[Example | None, list[str], list[str]]:
    """Подставляет настоящую выдачу вместо выдачи составителя; возвращает пример (None — расхождения),
    расхождения и ошибки, замечания проверки по настоящей выдаче."""
    where = f"{example.model}/{example.batch}/{example.id}"
    tools.reset()
    mapping: dict[int, int] = {}
    turns: list[Turn] = []
    pending: list[ToolCall] = []
    problems: list[str] = []
    for turn in example.turns:
        if turn.role == "model":
            text = _remap(turn.text, mapping)
            turns.append(Turn("model", text))
            pending = parse_tool_calls(text)[1]
            continue
        call = pending.pop(0)
        real = tools.call(call.name, call.argument)
        teacher = _remap(turn.text, mapping) if call.name != "search" else turn.text
        problems += _compare(call, teacher, real, mapping, f"{where}: {call.name}({call.argument})")
        turns.append(Turn("program", real))
    if problems:
        return None, problems, []
    # реплики модели — с окончательным переводом id (#PLAN мог упомянуть статью раньше, чем её нашёл поиск)
    originals = iter(t.text for t in example.turns if t.role == "model")
    turns = [Turn("model", _remap(next(originals), mapping)) if t.role == "model" else t for t in turns]
    turns, answer, errors, notes = check_episode(episode_text(turns), task, example.images, where, real=True)
    if errors:
        return None, errors, notes
    renumbered = [f"{where}: id составителя переведены в настоящие: "
                  + ", ".join(f"d{a}→d{b}" for a, b in sorted(mapping.items()) if a != b)] if any(
        a != b for a, b in mapping.items()) else []
    return replace(example, turns=turns, answer=answer), [], renumbered + notes


# --------------------------------------------------------------------------- сборка эпизода


class ReplayTools:
    """Инструменты для ``run_episode``, которые возвращают выдачу составителя по порядку вызовов."""

    def __init__(self, outputs: list[tuple[str, str, str]], max_calls: int = MAX_CALLS):
        self.outputs = outputs  # (инструмент, аргумент, ответ программы)
        self.max_calls = max_calls
        self.reset()

    def reset(self) -> None:
        self.calls: list[dict] = []
        self._next = 0

    def call(self, name: str, argument: str) -> str:
        expected_name, expected_argument, text = self.outputs[self._next]
        if (name, argument) != (expected_name, expected_argument):
            raise ValueError(f"вызов {name}({argument}) вместо {expected_name}({expected_argument})")
        self._next += 1
        self.calls.append({"name": name, "argument": argument, "chars": len(text)})
        return text

    def table(self) -> dict:
        return {}


def replay(example: Example, tokenizer, prompt_ids: list[int]) -> Episode:
    """Эпизод SFT той же функцией, что и роллауты RL: реплики модели — как сгенерированные, выдача — как ответы
    инструментов. ``prompt_ids`` — ``environment.build_prompt`` с ``prompts.system_prompt`` для запроса примера."""
    replies, outputs = [], []
    pending: list[ToolCall] = []
    for turn in example.turns:
        if turn.role == "model":
            replies.append(turn.text + END)
            pending = parse_tool_calls(turn.text)[1]
        else:
            call = pending.pop(0)
            outputs.append((call.name, call.argument, turn.text))

    def generate(ids, budget):
        new = tokenizer.encode(replies.pop(0), add_special_tokens=False)
        return new, True

    episode = run_episode(prompt_ids, tokenizer, generate, ReplayTools(outputs), max_new_tokens=10**6)
    if episode.finish != "stop" or episode.answer != example.answer or replies:
        raise ValueError(f"{example.id}: эпизод собрался не так, как написан ({episode.finish})")
    return episode


def main(argv: list[str] | None = None) -> int:
    """``python -m presentation_rl.sft_search [папка проекта] [--real]``.

    Без ``--real`` — проверка ответов составителей: ``sft_search_report.txt``. С ``--real`` (нужен доступ к Википедии
    и ``WIKI_CONTACT``, ответы сохраняются в ``data/sources/wiki_cache``) — ещё и подстановка настоящей выдачи:
    набор для SFT ``sft_search_dataset.jsonl`` и ``sft_search_real_report.txt``.
    """
    args = argv if argv is not None else sys.argv[1:]
    real = "--real" in args
    root = Path(next((a for a in args if not a.startswith("--")), "."))
    sft_dir = root / "data" / "tasks" / "sft_search"
    tasks_path = root / "data" / "tasks" / "tasks.json"
    report = collect(sft_dir, tasks_path, root / "data" / "tasks" / "sft" / "offline.json")
    (sft_dir / "sft_search_report.txt").write_text(report.text() + "\n", encoding="utf-8")
    print(report.text())
    if not real:
        return 1 if report.errors else 0

    from .tools.cache import WikiCache
    from .tools.wiki import WikiClient
    from .tools.wiki_tools import WikiTools

    tasks = {t.id: t for t in load_tasks(tasks_path, split="sft")}
    tools = WikiTools(WikiCache(root / "data" / "sources" / "wiki_cache", WikiClient()))
    final = Report()
    for example in report.examples:
        result, problems, notes = substitute(example, tools, tasks[example.id])
        final.errors += problems
        final.notes += notes
        if result is not None:
            final.examples.append(result)
    with open(sft_dir / "sft_search_dataset.jsonl", "w", encoding="utf-8") as out:
        for example in final.examples:
            out.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    (sft_dir / "sft_search_real_report.txt").write_text(final.text() + "\n", encoding="utf-8")
    print("\nС настоящей выдачей:\n" + final.text())
    return 1 if final.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
