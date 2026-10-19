# Здесь проверяем замены команд LaTeX на команды mathtext: что переписываем и что оставляем как есть.
import pytest

from presentation_rl.rendering.mathtext import to_mathtext


@pytest.mark.parametrize(
    "latex, expected",
    [
        # синонимы
        (r"a \le b \ge c", r"a \leq b \geq c"),
        (r"p \implies q \iff r", r"p \Rightarrow q \Leftrightarrow r"),
        (r"\tfrac{1}{2} + \cfrac{1}{3}", r"\frac{1}{2} + \frac{1}{3}"),
        (r"\lvert x \rvert + \lVert v \rVert", r"| x | + \| v \|"),
        (r"\operatorname*{argmax}_x f(x)", r"\operatorname{argmax}_x f(x)"),
        (r"\stackrel{def}{=}", r"\overset{def}{=}"),
        # текстовые стили — в \text, вложенный \text схлопывается
        (r"\textbf{скорость} = \emph{v}", r"\text{скорость} = \text{v}"),
        (r"\text{\textbf{x}}", r"\text{x}"),
        # оформление удаляем
        (r"\displaystyle \sum\limits_{i=1}^n i", r" \sum_{i=1}^n i"),
        (r"\big( x \big)", r"( x )"),
        (r"x = 1 \tag{1}", r"x = 1 "),
        (r"\boxed{x = 1}", r"{x = 1}"),
        (r"\textcolor{red}{x} + \color{blue} y", r"{x} +  y"),
        # стрелки с подписью
        (r"a \xrightarrow{\text{свет}} b", r"a \overset{\text{свет}}{\longrightarrow} b"),
        (r"a \xleftarrow[t]{k} b", r"a \underset{t}{\overset{k}{\longleftarrow}} b"),
        # похожие команды не задеваем: \left не \le, \geqslant не \ge
        (r"\left( x \right) \leq \geqslant", r"\left( x \right) \leq \geqslant"),
        # остаток от деления
        (r"a \equiv b \pmod{N}", r"a \equiv b \;(\mathrm{mod}\;N)"),
        (r"a \bmod n", r"a \;\mathrm{mod}\; n"),
        # перенос строки \\ не команда
        (r"a \\ b", r"a \\ b"),
    ],
)
def test_commands_are_rewritten(latex, expected):
    assert to_mathtext(latex) == expected


def test_cases_become_brace_with_stacked_rows():
    latex = r"|x| = \begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}"
    expected = r"|x| = \left\{ \genfrac{}{}{0}{0}{x \quad x \geq 0}{-x \quad x < 0} \right."
    assert to_mathtext(latex) == expected


def test_three_rows_are_nested():
    latex = r"\begin{cases} 1 & x>0 \\ 0 & x=0 \\ -1 & x<0 \end{cases}"
    expected = (
        r"\left\{ \genfrac{}{}{0}{0}{1 \quad x>0}"
        r"{\genfrac{}{}{0}{0}{0 \quad x=0}{-1 \quad x<0}} \right."
    )
    assert to_mathtext(latex) == expected


def test_aligned_rows_are_stacked_without_ampersands():
    latex = r"\begin{aligned} y &= (x+1)^2 \\ &= x^2 + 2x + 1 \end{aligned}"
    assert to_mathtext(latex) == r"\genfrac{}{}{0}{0}{y = (x+1)^2}{= x^2 + 2x + 1}"


@pytest.mark.parametrize(
    "latex",
    [
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",  # матрицы не переписываем
        r"\ce{6CO2 + 6H2O -> C6H12O6 + 6O2}",  # химия mhchem
        r"\underbrace{a + b}_{n}",
        r"\theta_{t+1} = \theta_t - \eta \nabla J",  # то, что mathtext и так знает
    ],
)
def test_unknown_or_supported_commands_are_left_as_is(latex):
    assert to_mathtext(latex) == latex
