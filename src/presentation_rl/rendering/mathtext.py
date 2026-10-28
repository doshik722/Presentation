# Здесь переводим формулы LaTeX в подмножество, которое рисует matplotlib mathtext.
"""Замены команд LaTeX, которых нет в mathtext, на равнозначные по виду.

Модели пишут обычный LaTeX с amsmath, а рисуем мы через mathtext (ставится
вместе с matplotlib, внешний LaTeX не нужен). Правильную формулу нельзя
штрафовать только за то, что mathtext не знает команду, поэтому перед
отрисовкой переписываем то, что умеем переписать без потери смысла:

- синонимы: ``\\le`` → ``\\leq``, ``\\implies`` → ``\\Rightarrow``, ``\\tfrac`` → ``\\frac`` …;
- команды оформления без смысла для слайда: ``\\displaystyle``, ``\\big``, ``\\tag{1}`` …;
- ``\\xrightarrow{a}`` → ``\\overset{a}{\\longrightarrow}``;
- остаток от деления: ``\\pmod{N}`` → ``\\;(\\mathrm{mod}\\;N)``, ``\\bmod`` и ``\\mod`` → ``\\mathrm{mod}``;
- окружения ``cases`` и ``aligned``: строки ставим друг под другом через ``\\genfrac``;
- сокращённая дробь без скобок: ``\\frac12`` → ``\\frac{1}{2}``, ``\\frac1a`` → ``\\frac{1}{a}`` (в LaTeX так можно,
  mathtext требует скобки); так же шрифт и акцент над одной буквой: ``\\mathbf w`` → ``\\mathbf{w}``, ``\\hat x`` → ``\\hat{x}``.

Что не умеем переписать (матрицы, ``\\ce``, ``\\underbrace`` …), оставляем как есть:
сборщик отметит такую формулу как ``render_error``.
"""
from __future__ import annotations

import re

# Синонимы: команда → команда mathtext с тем же видом.
_RENAME = {
    "le": "leq",
    "ge": "geq",
    "iff": "Leftrightarrow",
    "implies": "Rightarrow",
    "impliedby": "Leftarrow",
    "gets": "leftarrow",
    "land": "wedge",
    "lor": "vee",
    "lnot": "neg",
    "tfrac": "frac",
    "cfrac": "frac",
    "stackrel": "overset",
    "textbf": "text",
    "textit": "text",
    "textrm": "text",
    "emph": "text",
    "mbox": "text",
    "operatorname": "operatorname",  # \operatorname* → \operatorname: звёздочку снимаем
}

# Символы, которые пишем напрямую.
_SYMBOL = {"lvert": "|", "rvert": "|", "lVert": r"\|", "rVert": r"\|"}

# Оформление без смысла для слайда: удаляем команду.
_DROP = {
    "displaystyle", "textstyle", "limits", "nolimits", "nonumber", "notag",
    "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr",
    "biggl", "biggr", "Biggl", "Biggr", "bigm", "Bigm",
}

# Удаляем команду вместе с аргументом в фигурных скобках.
_DROP_WITH_ARG = {"tag", "label", "color"}

# Окружения: строки ставим друг под другом; у cases — с фигурной скобкой слева.
_BRACE_ENVS = ("cases", "dcases")
_STACK_ENVS = ("aligned", "align", "gathered", "gather", "split")

_COMMAND = re.compile(r"\\\\|\\([A-Za-z]+)(\*?)")
_ENV = re.compile(r"\\begin\{(\w+)\*?\}(.*?)\\end\{\1\*?\}", re.DOTALL)
_NESTED_TEXT = re.compile(r"\\text\{\s*\\text\{([^{}]*)\}\s*\}")
_SHORT_ARG = re.compile(r"\\(mathbf|mathrm|mathit|mathcal|mathbb|mathsf|mathtt|boldsymbol|vec|hat|bar|tilde|dot|ddot)"
                        r"(?:\s+|(?=[0-9]))([0-9A-Za-z])")
_SHORT_FRAC = re.compile(r"\\([dt]?frac|binom)\s*([0-9A-Za-z]|\{[^{}]*\})\s*([0-9A-Za-z]|\{[^{}]*\})")


def _braced(token: str) -> str:
    return token if token.startswith("{") else "{" + token + "}"


def to_mathtext(latex: str) -> str:
    """Переписывает формулу LaTeX в вид, который понимает mathtext."""
    text = _ENV.sub(_convert_env, latex)
    text = _convert_commands(text)
    text = _SHORT_ARG.sub(lambda m: "\\" + m.group(1) + "{" + m.group(2) + "}", text)
    text = _SHORT_FRAC.sub(lambda m: "\\" + m.group(1) + _braced(m.group(2)) + _braced(m.group(3)), text)
    while True:  # \textbf внутри \text превратился в \text{\text{…}}
        collapsed = _NESTED_TEXT.sub(r"\\text{\1}", text)
        if collapsed == text:
            return text
        text = collapsed


def _convert_env(match: re.Match) -> str:
    name, body = match.group(1), match.group(2)
    if name not in _BRACE_ENVS + _STACK_ENVS:
        return match.group(0)  # матрицы и прочее: не переписываем
    rows = [row.strip() for row in re.split(r"\\\\", body) if row.strip()]
    if not rows:
        return ""
    joiner = r" \quad " if name in _BRACE_ENVS else " "  # в cases & разделяет значение и условие
    rows = [joiner.join(cell.strip() for cell in row.split("&") if cell.strip()) for row in rows]
    stacked = _stack(rows)
    return rf"\left\{{ {stacked} \right." if name in _BRACE_ENVS else stacked


def _stack(rows: list[str]) -> str:
    # \genfrac без черты и в размере основного текста — строки одна под другой.
    if len(rows) == 1:
        return rows[0]
    return rf"\genfrac{{}}{{}}{{0}}{{0}}{{{rows[0]}}}{{{_stack(rows[1:])}}}"


def _convert_commands(text: str) -> str:
    out: list[str] = []
    i = 0
    while True:
        match = _COMMAND.search(text, i)
        if match is None:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i : match.start()])
        i = match.end()
        name = match.group(1)
        if name is None:  # \\ — перенос строки, не команда
            out.append(match.group(0))
        elif name in _DROP:
            pass
        elif name in _DROP_WITH_ARG:
            _, i = _argument(text, i)
        elif name == "textcolor":
            _, i = _argument(text, i)
            body, i = _argument(text, i)
            out.append("{" + _convert_commands(body) + "}")
        elif name == "pmod":
            body, i = _argument(text, i)
            out.append(r"\;(\mathrm{mod}\;" + _convert_commands(body) + ")")
        elif name in ("bmod", "mod"):
            out.append(r"\;\mathrm{mod}\;")
        elif name == "boxed":
            body, i = _argument(text, i)
            out.append("{" + _convert_commands(body) + "}")
        elif name in ("xrightarrow", "xleftarrow"):
            below, i = _optional(text, i)
            above, i = _argument(text, i)
            arrow = r"\longrightarrow" if name == "xrightarrow" else r"\longleftarrow"
            result = rf"\overset{{{_convert_commands(above)}}}{{{arrow}}}"
            if below:
                result = rf"\underset{{{_convert_commands(below)}}}{{{result}}}"
            out.append(result)
        elif name in _SYMBOL:
            out.append(_SYMBOL[name])
        elif name in _RENAME:
            out.append("\\" + _RENAME[name])
        else:
            out.append(match.group(0))


def _argument(text: str, i: int) -> tuple[str, int]:
    """Аргумент команды: содержимое {…} или один символ. Возвращает (аргумент, позиция после)."""
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return "", i
    if text[i] != "{":
        return text[i], i + 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{" and (j == 0 or text[j - 1] != "\\"):
            depth += 1
        elif text[j] == "}" and text[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
    return text[i + 1 :], len(text)  # скобка не закрыта: mathtext сообщит об ошибке сам


def _optional(text: str, i: int) -> tuple[str, int]:
    """Необязательный аргумент [...], если он есть."""
    j = i
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "[":
        end = text.find("]", j)
        if end != -1:
            return text[j + 1 : end], end + 1
    return "", i
