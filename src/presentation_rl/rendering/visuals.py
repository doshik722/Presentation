# Здесь превращаем формулы (#FORMULA) и графики функций (#PLOT) в PNG для вставки на слайд.
"""Растровые визуальные элементы: формулы и графики функций.

Используется matplotlib без графического окна. Ошибки отрисовки превращаются
в ``RenderError``: сборщик записывает их в отчёт и не вставляет элемент.
"""
from __future__ import annotations

import ast
import logging
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..dsl import compile_expression  # noqa: E402
from .mathtext import to_mathtext  # noqa: E402

DPI = 200


class RenderError(RuntimeError):
    """Элемент не удалось отрисовать."""


class _GlyphWatcher(logging.Handler):
    """Слушает журнал mathtext и собирает сообщения о символах, которых нет в шрифте.

    Такие символы mathtext не считает ошибкой: рисует вместо них заглушку и пишет
    предупреждение в журнал. Пример — ``x^\\text{\\textasciicircum}2``: внутри ``\\text{}``
    ``\\t`` читается как табуляция.
    """

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.missing: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "does not have a glyph" in message:
            self.missing.append(message)


def render_formula(latex: str, path: Path, fontsize: int = 26, color: str = "#1F2A44") -> tuple[int, int]:
    """Рисует формулу через matplotlib mathtext. Возвращает размер PNG в пикселях.

    Сначала переписываем команды, которых нет в mathtext (``to_mathtext``).
    Сломанной считаем формулу, на которой mathtext падает, и формулу, в которой
    хотя бы один символ нарисован заглушкой.
    """
    fig = plt.figure(figsize=(0.01, 0.01))
    watcher = _GlyphWatcher()
    log = logging.getLogger("matplotlib.mathtext")
    log.addHandler(watcher)
    try:
        fig.text(0, 0, f"${to_mathtext(latex)}$", fontsize=fontsize, color=color)
        fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.04)
    except Exception as exc:  # mathtext сообщает об ошибках разными типами исключений
        raise RenderError(f"формулу не удалось отрисовать: {_first_line(exc)}") from exc
    finally:
        log.removeHandler(watcher)
        plt.close(fig)
    if watcher.missing:
        Path(path).unlink(missing_ok=True)
        raise RenderError(f"в формуле символ, которого нет в шрифте: {watcher.missing[0]}")
    return _png_size(path)


def evaluate_expression(expression: str, x: np.ndarray) -> np.ndarray:
    """Вычисляет проверенное выражение #PLOT на массиве x без eval."""
    tree = compile_expression(expression)
    functions = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
    }
    constants = {"pi": math.pi, "e": math.e}

    def walk(node: ast.AST) -> np.ndarray | float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return x if node.id == "x" else constants[node.id]
        if isinstance(node, ast.UnaryOp):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return np.divide(left, right)
            return np.power(left, right)
        if isinstance(node, ast.Call):
            return functions[node.func.id](walk(node.args[0]))
        raise RenderError("недопустимая конструкция в выражении")

    with np.errstate(all="ignore"):
        result = walk(tree)
    return np.broadcast_to(np.asarray(result, dtype=float), x.shape)


def render_plot(
    expression: str,
    x_min: float,
    x_max: float,
    path: Path,
    title: str | None = None,
    size_in: tuple[float, float] = (6.0, 3.6),
    line_color: str = "#2A9D8F",
    text_color: str = "#1F2A44",
) -> tuple[int, int]:
    """Рисует график функции y(x). Возвращает размер PNG в пикселях."""
    x = np.linspace(x_min, x_max, 600)
    y = evaluate_expression(expression, x).astype(float)
    finite = np.isfinite(y)
    if finite.mean() < 0.5:
        raise RenderError("на большей части диапазона функция не определена")
    y = np.where(finite, y, np.nan)

    fig, ax = plt.subplots(figsize=size_in)
    try:
        ax.plot(x, y, color=line_color, linewidth=2.4)
        ax.axhline(0, color="#B8BFC9", linewidth=0.8, zorder=0)
        if x_min < 0 < x_max:
            ax.axvline(0, color="#B8BFC9", linewidth=0.8, zorder=0)
        ax.grid(True, color="#E6E9EE", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8A94A6")
        ax.tick_params(colors=text_color, labelsize=10)
        ax.set_xlabel("x", color=text_color, fontsize=11)
        ax.set_ylabel("y", color=text_color, fontsize=11, rotation=0, labelpad=12)
        if title:
            # Длинный заголовок переносим, чтобы он не обрезался краем рисунка.
            chars_per_line = max(20, int(size_in[0] * 0.8 * 72 / (12 * 0.62)))
            ax.set_title(textwrap.fill(title, chars_per_line), color=text_color, fontsize=12, loc="left", pad=10)
        fig.tight_layout()
        fig.savefig(path, dpi=DPI, transparent=False, facecolor="white")
    except Exception as exc:
        raise RenderError(f"график не удалось отрисовать: {_first_line(exc)}") from exc
    finally:
        plt.close(fig)
    return _png_size(path)


def _png_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else exc.__class__.__name__
