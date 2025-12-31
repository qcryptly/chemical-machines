"""
Chemical Machines Symbols Module

A module for rendering LaTeX math expressions with support for different notation styles.

Notation Styles:
    - standard: Default LaTeX math notation
    - physicist: Physics notation (hbar, vectors with arrows, etc.)
    - chemist: Chemistry notation (reaction arrows, chemical formulas)
    - braket: Dirac bra-ket notation for quantum mechanics
    - engineering: Engineering notation (j for imaginary, etc.)

Usage:
    from cm.symbols import latex, Math, set_notation

    # Simple LaTeX rendering
    latex(r"E = mc^2")

    # Using the Math builder
    m = Math()
    m.frac("a", "b").plus().sqrt("c")
    m.render()

    # Change notation style
    set_notation("physicist")
    latex(r"\\hbar \\omega")

    # Bra-ket notation
    set_notation("braket")
    m = Math()
    m.bra("psi").ket("phi")
    m.render()
"""

from typing import Optional, List, Union
from . import views

# Current notation style
_notation_style = "standard"

# Current line height (default: normal)
_line_height = "normal"


def set_line_height(height: str):
    """
    Set the line height for math rendering.

    Args:
        height: CSS line-height value (e.g., "1", "1.5", "2", "normal", "1.2em")

    Example:
        set_line_height("1.5")
        set_line_height("2")
        set_line_height("normal")
    """
    global _line_height
    _line_height = height


def get_line_height() -> str:
    """Get the current line height setting."""
    return _line_height

# Notation-specific LaTeX preambles/macros
_NOTATION_MACROS = {
    "standard": "",
    "physicist": r"""
        \newcommand{\vect}[1]{\vec{#1}}
        \newcommand{\grad}{\nabla}
        \newcommand{\curl}{\nabla \times}
        \newcommand{\divg}{\nabla \cdot}
        \newcommand{\lapl}{\nabla^2}
        \newcommand{\ddt}{\frac{d}{dt}}
        \newcommand{\ddx}{\frac{d}{dx}}
        \newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
    """,
    "chemist": r"""
        \newcommand{\ce}[1]{\mathrm{#1}}
        \newcommand{\rightmark}{\rightarrow}
        \newcommand{\leftmark}{\leftarrow}
        \newcommand{\eqmark}{\rightleftharpoons}
        \newcommand{\yields}{\rightarrow}
        \newcommand{\equilibrium}{\rightleftharpoons}
        \newcommand{\gas}{\uparrow}
        \newcommand{\precipitate}{\downarrow}
        \newcommand{\aq}{_{(aq)}}
        \newcommand{\solid}{_{(s)}}
        \newcommand{\liquid}{_{(l)}}
        \newcommand{\gasphase}{_{(g)}}
    """,
    "braket": r"""
        \newcommand{\bra}[1]{\langle #1 |}
        \newcommand{\ket}[1]{| #1 \rangle}
        \newcommand{\braket}[2]{\langle #1 | #2 \rangle}
        \newcommand{\expval}[1]{\langle #1 \rangle}
        \newcommand{\matelem}[3]{\langle #1 | #2 | #3 \rangle}
        \newcommand{\op}[1]{\hat{#1}}
        \newcommand{\comm}[2]{[#1, #2]}
        \newcommand{\anticomm}[2]{\{#1, #2\}}
        \newcommand{\dagger}{\dagger}
    """,
    "engineering": r"""
        \newcommand{\j}{\mathrm{j}}
        \newcommand{\ohm}{\Omega}
        \newcommand{\simark}[1]{\,\mathrm{#1}}
        \newcommand{\phasor}[1]{\tilde{#1}}
        \newcommand{\magnitude}[1]{|#1|}
        \newcommand{\phase}[1]{\angle #1}
        \newcommand{\conj}[1]{#1^*}
        \newcommand{\re}{\mathrm{Re}}
        \newcommand{\im}{\mathrm{Im}}
    """
}


def set_notation(style: str):
    """
    Set the notation style for math rendering.

    Args:
        style: One of 'standard', 'physicist', 'chemist', 'braket', 'engineering'

    Example:
        set_notation("physicist")
        set_notation("braket")
    """
    global _notation_style
    if style not in _NOTATION_MACROS:
        raise ValueError(f"Unknown notation style: {style}. "
                        f"Available: {list(_NOTATION_MACROS.keys())}")
    _notation_style = style


def get_notation() -> str:
    """Get the current notation style."""
    return _notation_style


def _get_macros() -> str:
    """Get the LaTeX macros for the current notation style."""
    return _NOTATION_MACROS.get(_notation_style, "")


def _get_line_height_style() -> str:
    """Get inline style for line height if not default."""
    if _line_height != "normal":
        return f' style="line-height: {_line_height};"'
    return ""


def latex(expression: str, display: bool = True, label: Optional[str] = None,
          justify: str = "center"):
    """
    Render a LaTeX math expression.

    Args:
        expression: LaTeX math expression (without delimiters)
        display: If True, render as display math (centered, block). If False, inline.
        label: Optional label/caption for the expression
        justify: Alignment - 'left', 'center', or 'right'

    Example:
        latex(r"E = mc^2")
        latex(r"\\int_0^\\infty e^{-x} dx = 1", display=True)
        latex(r"x^2", display=False)  # Inline math
        latex(r"F = ma", justify="left")
    """
    if display:
        delim_start, delim_end = r"\[", r"\]"
    else:
        delim_start, delim_end = r"\(", r"\)"

    justify_class = f"cm-math-{justify}" if justify in ("left", "center", "right") else "cm-math-center"
    line_height_style = _get_line_height_style()
    html_content = f'<div class="cm-math {justify_class}"{line_height_style}>{delim_start}{expression}{delim_end}</div>'

    if label:
        html_content = f'<div class="cm-math-labeled"{line_height_style}><span class="cm-math-label">{label}</span>{html_content}</div>'

    views.html(html_content)


def equation(expression: str, number: Optional[Union[int, str]] = None):
    """
    Render a numbered equation.

    Args:
        expression: LaTeX math expression
        number: Optional equation number or label

    Example:
        equation(r"F = ma", number=1)
        equation(r"E = mc^2", number="2.1")
    """
    line_height_style = _get_line_height_style()
    if number is not None:
        html_content = f'''
        <div class="cm-equation"{line_height_style}>
            <span class="cm-equation-content">\\[{expression}\\]</span>
            <span class="cm-equation-number">({number})</span>
        </div>
        '''
    else:
        html_content = f'<div class="cm-equation"{line_height_style}>\\[{expression}\\]</div>'

    views.html(html_content)


def align(*equations: str):
    """
    Render aligned equations (useful for multi-step derivations).

    Args:
        *equations: LaTeX expressions with & for alignment points

    Example:
        align(
            r"x &= a + b",
            r"&= c + d",
            r"&= e"
        )
    """
    aligned = r" \\ ".join(equations)
    line_height_style = _get_line_height_style()
    html_content = f'<div class="cm-math"{line_height_style}>\\[\\begin{{aligned}}{aligned}\\end{{aligned}}\\]</div>'
    views.html(html_content)


def matrix(data: List[List], style: str = "pmatrix"):
    """
    Render a matrix.

    Args:
        data: 2D list of matrix elements
        style: Matrix style - 'pmatrix' (parentheses), 'bmatrix' (brackets),
               'vmatrix' (vertical bars), 'Vmatrix' (double bars), 'matrix' (none)

    Example:
        matrix([[1, 2], [3, 4]])
        matrix([[1, 0], [0, 1]], style="bmatrix")
    """
    rows = []
    for row in data:
        row_str = " & ".join(str(x) for x in row)
        rows.append(row_str)

    matrix_content = r" \\ ".join(rows)
    line_height_style = _get_line_height_style()
    html_content = f'<div class="cm-math"{line_height_style}>\\[\\begin{{{style}}}{matrix_content}\\end{{{style}}}\\]</div>'
    views.html(html_content)


def bullets(*expressions: str, display: bool = True):
    """
    Render a bulleted list of LaTeX expressions.

    Args:
        *expressions: LaTeX expressions for each bullet point
        display: If True, use display math. If False, inline math.

    Example:
        bullets(
            r"x^2 + y^2 = r^2",
            r"e^{i\\pi} + 1 = 0",
            r"\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}"
        )
    """
    delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
    list_items = "\n".join(f'<li>{delim_start}{expr}{delim_end}</li>' for expr in expressions)
    line_height_style = _get_line_height_style()
    html_content = f'<ul class="cm-math-list bulleted"{line_height_style}>{list_items}</ul>'
    views.html(html_content)


def numbered(*expressions: str, start: int = 1, display: bool = True):
    """
    Render a numbered list of LaTeX expressions.

    Args:
        *expressions: LaTeX expressions for each numbered item
        start: Starting number (default: 1)
        display: If True, use display math. If False, inline math.

    Example:
        numbered(
            r"F = ma",
            r"E = mc^2",
            r"p = mv"
        )
    """
    delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
    list_items = "\n".join(f'<li>{delim_start}{expr}{delim_end}</li>' for expr in expressions)
    line_height_style = _get_line_height_style()
    html_content = f'<ol class="cm-math-list numbered" start="{start}"{line_height_style}>{list_items}</ol>'
    views.html(html_content)


def items(*expressions: str, display: bool = True):
    """
    Render a plain list of LaTeX expressions (no bullets or numbers).

    Args:
        *expressions: LaTeX expressions for each item
        display: If True, use display math. If False, inline math.

    Example:
        items(
            r"\\text{First equation: } x = 1",
            r"\\text{Second equation: } y = 2"
        )
    """
    delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
    item_list = "\n".join(f'<li>{delim_start}{expr}{delim_end}</li>' for expr in expressions)
    line_height_style = _get_line_height_style()
    html_content = f'<ul class="cm-math-list none"{line_height_style}>{item_list}</ul>'
    views.html(html_content)


class Math:
    """
    A builder class for constructing LaTeX expressions programmatically.

    Example:
        m = Math()
        m.frac("a", "b").plus().sqrt("c").equals().text("result")
        m.render()

        # Bra-ket notation (with braket style)
        m = Math()
        m.bra("psi").ket("phi")
        m.render()
    """

    def __init__(self):
        self._parts: List[str] = []

    def _append(self, content: str) -> "Math":
        self._parts.append(content)
        return self

    def raw(self, latex: str) -> "Math":
        """Add raw LaTeX content."""
        return self._append(latex)

    def text(self, content: str) -> "Math":
        """Add text (non-italic) content."""
        return self._append(f"\\text{{{content}}}")

    def var(self, name: str) -> "Math":
        """Add a variable."""
        return self._append(name)

    # Basic operations
    def plus(self) -> "Math":
        return self._append(" + ")

    def minus(self) -> "Math":
        return self._append(" - ")

    def times(self) -> "Math":
        return self._append(" \\times ")

    def cdot(self) -> "Math":
        return self._append(" \\cdot ")

    def div(self) -> "Math":
        return self._append(" \\div ")

    def equals(self) -> "Math":
        return self._append(" = ")

    def approx(self) -> "Math":
        return self._append(" \\approx ")

    def neq(self) -> "Math":
        return self._append(" \\neq ")

    def lt(self) -> "Math":
        return self._append(" < ")

    def gt(self) -> "Math":
        return self._append(" > ")

    def leq(self) -> "Math":
        return self._append(" \\leq ")

    def geq(self) -> "Math":
        return self._append(" \\geq ")

    # Fractions and roots
    def frac(self, num: str, denom: str) -> "Math":
        """Add a fraction."""
        return self._append(f"\\frac{{{num}}}{{{denom}}}")

    def sqrt(self, content: str, n: Optional[str] = None) -> "Math":
        """Add a square root or nth root."""
        if n:
            return self._append(f"\\sqrt[{n}]{{{content}}}")
        return self._append(f"\\sqrt{{{content}}}")

    # Subscripts and superscripts
    def sub(self, content: str) -> "Math":
        """Add a subscript."""
        return self._append(f"_{{{content}}}")

    def sup(self, content: str) -> "Math":
        """Add a superscript."""
        return self._append(f"^{{{content}}}")

    def subsup(self, sub: str, sup: str) -> "Math":
        """Add both subscript and superscript."""
        return self._append(f"_{{{sub}}}^{{{sup}}}")

    # Greek letters
    def alpha(self) -> "Math": return self._append("\\alpha")
    def beta(self) -> "Math": return self._append("\\beta")
    def gamma(self) -> "Math": return self._append("\\gamma")
    def delta(self) -> "Math": return self._append("\\delta")
    def epsilon(self) -> "Math": return self._append("\\epsilon")
    def zeta(self) -> "Math": return self._append("\\zeta")
    def eta(self) -> "Math": return self._append("\\eta")
    def theta(self) -> "Math": return self._append("\\theta")
    def iota(self) -> "Math": return self._append("\\iota")
    def kappa(self) -> "Math": return self._append("\\kappa")
    def lambda_(self) -> "Math": return self._append("\\lambda")
    def mu(self) -> "Math": return self._append("\\mu")
    def nu(self) -> "Math": return self._append("\\nu")
    def xi(self) -> "Math": return self._append("\\xi")
    def pi(self) -> "Math": return self._append("\\pi")
    def rho(self) -> "Math": return self._append("\\rho")
    def sigma(self) -> "Math": return self._append("\\sigma")
    def tau(self) -> "Math": return self._append("\\tau")
    def upsilon(self) -> "Math": return self._append("\\upsilon")
    def phi(self) -> "Math": return self._append("\\phi")
    def chi(self) -> "Math": return self._append("\\chi")
    def psi(self) -> "Math": return self._append("\\psi")
    def omega(self) -> "Math": return self._append("\\omega")

    # Capital Greek
    def Gamma(self) -> "Math": return self._append("\\Gamma")
    def Delta(self) -> "Math": return self._append("\\Delta")
    def Theta(self) -> "Math": return self._append("\\Theta")
    def Lambda(self) -> "Math": return self._append("\\Lambda")
    def Xi(self) -> "Math": return self._append("\\Xi")
    def Pi(self) -> "Math": return self._append("\\Pi")
    def Sigma(self) -> "Math": return self._append("\\Sigma")
    def Phi(self) -> "Math": return self._append("\\Phi")
    def Psi(self) -> "Math": return self._append("\\Psi")
    def Omega(self) -> "Math": return self._append("\\Omega")

    # Calculus
    def integral(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "Math":
        """Add an integral sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\int_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\int_{{{lower}}}")
        return self._append("\\int")

    def sum(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "Math":
        """Add a summation sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\sum_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\sum_{{{lower}}}")
        return self._append("\\sum")

    def prod(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "Math":
        """Add a product sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\prod_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\prod_{{{lower}}}")
        return self._append("\\prod")

    def lim(self, var: str, to: str) -> "Math":
        """Add a limit."""
        return self._append(f"\\lim_{{{var} \\to {to}}}")

    def deriv(self, func: str = "", var: str = "x") -> "Math":
        """Add a derivative."""
        if func:
            return self._append(f"\\frac{{d{func}}}{{d{var}}}")
        return self._append(f"\\frac{{d}}{{d{var}}}")

    def partial(self, func: str = "", var: str = "x") -> "Math":
        """Add a partial derivative."""
        if func:
            return self._append(f"\\frac{{\\partial {func}}}{{\\partial {var}}}")
        return self._append(f"\\frac{{\\partial}}{{\\partial {var}}}")

    def nabla(self) -> "Math":
        return self._append("\\nabla")

    # Brackets and grouping
    def paren(self, content: str) -> "Math":
        """Add parentheses."""
        return self._append(f"\\left({content}\\right)")

    def bracket(self, content: str) -> "Math":
        """Add square brackets."""
        return self._append(f"\\left[{content}\\right]")

    def brace(self, content: str) -> "Math":
        """Add curly braces."""
        return self._append(f"\\left\\{{{content}\\right\\}}")

    def abs(self, content: str) -> "Math":
        """Add absolute value bars."""
        return self._append(f"\\left|{content}\\right|")

    def norm(self, content: str) -> "Math":
        """Add norm double bars."""
        return self._append(f"\\left\\|{content}\\right\\|")

    # Quantum mechanics / Bra-ket
    def bra(self, content: str) -> "Math":
        """Add a bra <content|."""
        return self._append(f"\\langle {content} |")

    def ket(self, content: str) -> "Math":
        """Add a ket |content>."""
        return self._append(f"| {content} \\rangle")

    def braket(self, bra: str, ket: str) -> "Math":
        """Add a braket <bra|ket>."""
        return self._append(f"\\langle {bra} | {ket} \\rangle")

    def expval(self, operator: str) -> "Math":
        """Add an expectation value <operator>."""
        return self._append(f"\\langle {operator} \\rangle")

    def matelem(self, bra: str, op: str, ket: str) -> "Math":
        """Add a matrix element <bra|op|ket>."""
        return self._append(f"\\langle {bra} | {op} | {ket} \\rangle")

    def op(self, name: str) -> "Math":
        """Add an operator with hat."""
        return self._append(f"\\hat{{{name}}}")

    def dagger(self) -> "Math":
        """Add a dagger superscript."""
        return self._append("^\\dagger")

    def comm(self, a: str, b: str) -> "Math":
        """Add a commutator [a, b]."""
        return self._append(f"[{a}, {b}]")

    # Physics
    def vec(self, content: str) -> "Math":
        """Add a vector with arrow."""
        return self._append(f"\\vec{{{content}}}")

    def hbar(self) -> "Math":
        return self._append("\\hbar")

    def infty(self) -> "Math":
        return self._append("\\infty")

    # Chemistry
    def ce(self, formula: str) -> "Math":
        """Add a chemical formula (upright text)."""
        return self._append(f"\\mathrm{{{formula}}}")

    def yields(self) -> "Math":
        """Add a reaction arrow."""
        return self._append(" \\rightarrow ")

    def equilibrium(self) -> "Math":
        """Add an equilibrium arrow."""
        return self._append(" \\rightleftharpoons ")

    # Special functions
    def sin(self, arg: str = "") -> "Math":
        return self._append(f"\\sin{{{arg}}}" if arg else "\\sin")

    def cos(self, arg: str = "") -> "Math":
        return self._append(f"\\cos{{{arg}}}" if arg else "\\cos")

    def tan(self, arg: str = "") -> "Math":
        return self._append(f"\\tan{{{arg}}}" if arg else "\\tan")

    def ln(self, arg: str = "") -> "Math":
        return self._append(f"\\ln{{{arg}}}" if arg else "\\ln")

    def log(self, arg: str = "", base: Optional[str] = None) -> "Math":
        if base:
            return self._append(f"\\log_{{{base}}}{{{arg}}}" if arg else f"\\log_{{{base}}}")
        return self._append(f"\\log{{{arg}}}" if arg else "\\log")

    def exp(self, arg: str = "") -> "Math":
        return self._append(f"\\exp{{{arg}}}" if arg else "\\exp")

    # Spacing
    def space(self) -> "Math":
        return self._append("\\ ")

    def quad(self) -> "Math":
        return self._append("\\quad")

    def qquad(self) -> "Math":
        return self._append("\\qquad")

    # Build and render
    def build(self) -> str:
        """Build and return the LaTeX string."""
        return "".join(self._parts)

    def render(self, display: bool = True, label: Optional[str] = None,
               justify: str = "center"):
        """Render the built expression to HTML output."""
        latex(self.build(), display=display, label=label, justify=justify)

    def clear(self) -> "Math":
        """Clear the builder."""
        self._parts = []
        return self

    def __str__(self) -> str:
        return self.build()


# Convenience functions for common expressions
def fraction(num: str, denom: str):
    """Render a simple fraction."""
    latex(f"\\frac{{{num}}}{{{denom}}}")


def sqrt(content: str, n: Optional[str] = None):
    """Render a square root or nth root."""
    if n:
        latex(f"\\sqrt[{n}]{{{content}}}")
    else:
        latex(f"\\sqrt{{{content}}}")


# Chemical notation helpers
def chemical(formula: str):
    """
    Render a chemical formula.

    Example:
        chemical("H2O")
        chemical("2H2 + O2 -> 2H2O")
    """
    # Simple conversion: numbers after letters become subscripts
    import re
    # Convert trailing numbers to subscripts
    result = re.sub(r'([A-Za-z])(\d+)', r'\1_{\2}', formula)
    # Convert arrows
    result = result.replace('->', r'\rightarrow')
    result = result.replace('<->', r'\rightleftharpoons')
    result = result.replace('<=>', r'\rightleftharpoons')

    latex(f"\\mathrm{{{result}}}")


def reaction(reactants: str, products: str, reversible: bool = False):
    """
    Render a chemical reaction.

    Example:
        reaction("2H2 + O2", "2H2O")
        reaction("N2 + 3H2", "2NH3", reversible=True)
    """
    arrow = r"\rightleftharpoons" if reversible else r"\rightarrow"
    chemical(f"{reactants} {arrow} {products}")
