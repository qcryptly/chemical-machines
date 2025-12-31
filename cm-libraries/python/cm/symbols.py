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
    def bra(self, content) -> "Math":
        """Add a bra <content|. Content can be a string or list of quantum numbers."""
        if isinstance(content, (list, tuple)):
            content = ", ".join(str(c) for c in content)
        return self._append(f"\\langle {content} |")

    def ket(self, content) -> "Math":
        """Add a ket |content>. Content can be a string or list of quantum numbers."""
        if isinstance(content, (list, tuple)):
            content = ", ".join(str(c) for c in content)
        return self._append(f"| {content} \\rangle")

    def braket(self, bra, ket) -> "Math":
        """Add a braket <bra|ket>. Arguments can be strings or lists of quantum numbers."""
        if isinstance(bra, (list, tuple)):
            bra = ", ".join(str(c) for c in bra)
        if isinstance(ket, (list, tuple)):
            ket = ", ".join(str(c) for c in ket)
        return self._append(f"\\langle {bra} | {ket} \\rangle")

    def expval(self, operator) -> "Math":
        """Add an expectation value <operator>."""
        if isinstance(operator, (list, tuple)):
            operator = ", ".join(str(c) for c in operator)
        return self._append(f"\\langle {operator} \\rangle")

    def matelem(self, bra, op, ket) -> "Math":
        """Add a matrix element <bra|op|ket>."""
        if isinstance(bra, (list, tuple)):
            bra = ", ".join(str(c) for c in bra)
        if isinstance(ket, (list, tuple)):
            ket = ", ".join(str(c) for c in ket)
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

    # Symbolic Determinant Expansion
    @staticmethod
    def _symbolic_determinant(m) -> List[tuple]:
        """
        Compute symbolic determinant expansion as list of (sign, [elements]) tuples.
        Uses cofactor expansion along the first row.
        """
        import numpy as np
        m = np.asarray(m)

        if m.shape[0] == 1:
            return [(1, [m[0, 0]])]

        if m.shape[0] == 2:
            return [(1, [m[0, 0], m[1, 1]]), (-1, [m[0, 1], m[1, 0]])]

        def distribute(sign, element, terms):
            """Distribute an element and sign across symbolic terms."""
            return [(sign * s, [element] + elems) for s, elems in terms]

        def interleave(lists):
            """Interleave terms from multiple cofactor expansions."""
            result = []
            max_len = max(len(lst) for lst in lists)
            for i in range(max_len):
                for lst in lists:
                    if i < len(lst):
                        result.append(lst[i])
            return result

        cofactors = []
        for col in range(m.shape[1]):
            sign = (-1) ** col
            minor = np.delete(np.delete(m, 0, axis=0), col, axis=1)
            sub_terms = Math._symbolic_determinant(minor)
            cofactors.append(distribute(sign, m[0, col], sub_terms))

        return interleave(cofactors)

    def determinant_bra(self, matrix) -> "Math":
        """
        Render symbolic determinant expansion using bra notation ⟨a,b,c|.

        Each term in the determinant expansion is rendered as a bra with
        the product of elements shown as comma-separated values.

        Args:
            matrix: 2D array-like (numpy array or nested list)

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_bra(y)
            m.render()
        """
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'bra')

    def determinant_ket(self, matrix) -> "Math":
        """
        Render symbolic determinant expansion using ket notation |a,b,c⟩.

        Each term in the determinant expansion is rendered as a ket with
        the product of elements shown as comma-separated values.

        Args:
            matrix: 2D array-like (numpy array or nested list)

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_ket(y)
            m.render()
        """
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'ket')

    def determinant_braket(self, matrix, bra_label: str = "\\psi") -> "Math":
        """
        Render symbolic determinant expansion using braket notation ⟨ψ|a,b,c⟩.

        Each term in the determinant expansion is rendered as a braket with
        a fixed bra label and the product elements as the ket.

        Args:
            matrix: 2D array-like (numpy array or nested list)
            bra_label: Label for the bra side (default: ψ)

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_braket(y, bra_label="\\phi")
            m.render()
        """
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'braket', bra_label=bra_label)

    def determinant_product(self, matrix) -> "Math":
        """
        Render symbolic determinant expansion as product notation (a·b·c).

        Each term in the determinant expansion is rendered as a product
        of elements using cdot.

        Args:
            matrix: 2D array-like (numpy array or nested list)

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_product(y)
            m.render()
        """
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'product')

    def determinant_subscript(self, matrix, var: str = "a") -> "Math":
        """
        Render symbolic determinant expansion using subscript notation (a_{ij}).

        Each element is rendered with row/column subscripts.

        Args:
            matrix: 2D array-like (numpy array or nested list)
            var: Variable name for elements (default: 'a')

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_subscript(y, var="a")
            m.render()
        """
        import numpy as np
        matrix = np.asarray(matrix)
        n = matrix.shape[0]

        # Create symbolic matrix with subscripts
        sym_matrix = [[f"{var}_{{{i+1}{j+1}}}" for j in range(n)] for i in range(n)]
        terms = self._symbolic_determinant(sym_matrix)
        return self._render_symbolic_terms(terms, 'product')

    def _render_symbolic_terms(self, terms: List[tuple], notation: str,
                                bra_label: str = None) -> "Math":
        """
        Internal method to render symbolic determinant terms in various notations.

        Args:
            terms: List of (sign, [elements]) tuples
            notation: One of 'bra', 'ket', 'braket', 'product'
            bra_label: Label for bra side (only used with 'braket' notation)
        """
        first = True

        for sign, elements in terms:
            # Handle sign
            if sign > 0:
                if not first:
                    self.plus()
            else:
                self.minus()
            first = False

            # Render elements in chosen notation
            if notation == 'bra':
                self.bra(elements)
            elif notation == 'ket':
                self.ket(elements)
            elif notation == 'braket':
                self.braket(bra_label, elements)
            elif notation == 'product':
                # Render as product with cdots
                elem_strs = [str(e) for e in elements]
                product_str = " \\cdot ".join(elem_strs)
                self._append(f"({product_str})")

        return self

    def slater_determinant(self, orbitals: List[str], normalize: bool = True) -> "Math":
        """
        Render a Slater determinant in standard physics notation.

        A Slater determinant represents an antisymmetrized product of
        single-particle wavefunctions (orbitals).

        Args:
            orbitals: List of orbital labels (e.g., ['1s', '2s', '2p'])
            normalize: Include normalization factor 1/√n!

        Example:
            m = Math()
            m.slater_determinant(['\\phi_1', '\\phi_2', '\\phi_3'])
            m.render()
        """
        n = len(orbitals)

        if normalize:
            self._append(f"\\frac{{1}}{{\\sqrt{{{n}!}}}}")

        # Build determinant matrix representation
        self._append("\\begin{vmatrix}")

        for i in range(n):
            row_parts = []
            for orb in orbitals:
                row_parts.append(f"{orb}(\\mathbf{{r}}_{{{i+1}}})")
            self._append(" & ".join(row_parts))
            if i < n - 1:
                self._append(" \\\\ ")

        self._append("\\end{vmatrix}")

        return self

    def slater_ket(self, orbitals: List[str], normalize: bool = True) -> "Math":
        """
        Render a Slater determinant in occupation number (ket) notation.

        Args:
            orbitals: List of orbital labels
            normalize: Include normalization factor

        Example:
            m = Math()
            m.slater_ket(['1s↑', '1s↓', '2s↑'])
            m.render()
            # Renders: |1s↑, 1s↓, 2s↑⟩
        """
        n = len(orbitals)

        if normalize:
            self._append(f"\\frac{{1}}{{\\sqrt{{{n}!}}}}")

        self.ket(orbitals)
        return self

    # Inner Products of Determinants
    @staticmethod
    def _compute_inner_product_terms(bra_terms: List[tuple], ket_terms: List[tuple],
                                      orthogonal: bool = False,
                                      orthogonal_states: Optional[List[str]] = None) -> List[tuple]:
        """
        Compute inner product terms between two symbolic determinant expansions.

        When orthogonality is specified, applies Kronecker delta: ⟨φᵢ|φⱼ⟩ = δᵢⱼ
        Non-matching overlaps evaluate to zero.

        Args:
            bra_terms: Symbolic determinant expansion for bra (from _symbolic_determinant)
            ket_terms: Symbolic determinant expansion for ket (from _symbolic_determinant)
            orthogonal: If True, all states are orthonormal (⟨i|j⟩ = δᵢⱼ)
            orthogonal_states: Optional list of states that are orthogonal to each other.
                              If provided, only these states follow orthogonality rules.

        Returns:
            List of (sign, bra_elements, ket_elements) tuples representing surviving terms
        """
        result = []

        for bra_sign, bra_elems in bra_terms:
            for ket_sign, ket_elems in ket_terms:
                combined_sign = bra_sign * ket_sign

                if orthogonal:
                    # Check if all elements match (Kronecker delta condition)
                    # For orthonormal states: ⟨a,b,c|d,e,f⟩ = δ_ad·δ_be·δ_cf
                    if len(bra_elems) != len(ket_elems):
                        continue

                    # Check pairwise orthogonality
                    all_match = True
                    for b, k in zip(bra_elems, ket_elems):
                        b_str, k_str = str(b), str(k)
                        if orthogonal_states is not None:
                            # Only apply orthogonality to specified states
                            if b_str in orthogonal_states and k_str in orthogonal_states:
                                if b_str != k_str:
                                    all_match = False
                                    break
                        else:
                            # All states are orthogonal
                            if b_str != k_str:
                                all_match = False
                                break

                    if all_match:
                        result.append((combined_sign, bra_elems, ket_elems))
                else:
                    # No orthogonality - keep all terms
                    result.append((combined_sign, bra_elems, ket_elems))

        return result

    def determinant_inner_product(self, bra_matrix, ket_matrix,
                                   orthogonal: bool = False,
                                   orthogonal_states: Optional[List[str]] = None,
                                   show_zeros: bool = False) -> "Math":
        """
        Render the inner product of two symbolic determinant expansions.

        When orthogonality is enabled, applies the Kronecker delta condition:
        ⟨φᵢ|φⱼ⟩ = δᵢⱼ (non-matching overlaps evaluate to zero).

        Args:
            bra_matrix: 2D array-like for the bra determinant
            ket_matrix: 2D array-like for the ket determinant
            orthogonal: If True, all states are orthonormal
            orthogonal_states: Optional list of states that are mutually orthogonal
            show_zeros: If True, show all terms with zeros; if False, only show surviving terms

        Example:
            m = Math()
            bra = np.array([['a', 'b'], ['c', 'd']])
            ket = np.array([['a', 'b'], ['c', 'd']])
            m.determinant_inner_product(bra, ket, orthogonal=True)
            m.render()
        """
        bra_terms = self._symbolic_determinant(bra_matrix)
        ket_terms = self._symbolic_determinant(ket_matrix)

        inner_terms = self._compute_inner_product_terms(
            bra_terms, ket_terms, orthogonal, orthogonal_states
        )

        if not inner_terms:
            self._append("0")
            return self

        first = True
        for sign, bra_elems, ket_elems in inner_terms:
            # Handle sign
            if sign > 0:
                if not first:
                    self.plus()
            else:
                self.minus()
            first = False

            # Render as ⟨bra|ket⟩ for each pair
            if orthogonal:
                # When orthogonal and matching, render as 1 (or simplified form)
                # Show the overlaps that survived
                overlaps = []
                for b, k in zip(bra_elems, ket_elems):
                    overlaps.append(f"\\langle {b} | {k} \\rangle")
                self._append("(" + " ".join(overlaps) + ")")
            else:
                # Show full inner product notation
                overlaps = []
                for b, k in zip(bra_elems, ket_elems):
                    overlaps.append(f"\\langle {b} | {k} \\rangle")
                self._append("(" + " ".join(overlaps) + ")")

        return self

    def determinant_inner_product_simplified(self, bra_matrix, ket_matrix,
                                              orthogonal: bool = True) -> "Math":
        """
        Render the simplified inner product assuming orthonormal states.

        When states are orthonormal, ⟨φᵢ|φⱼ⟩ = δᵢⱼ, so matching terms become 1.
        This method shows the final simplified result.

        Args:
            bra_matrix: 2D array-like for the bra determinant
            ket_matrix: 2D array-like for the ket determinant
            orthogonal: If True (default), simplify matching terms to 1

        Example:
            m = Math()
            bra = np.array([['a', 'b'], ['c', 'd']])
            ket = np.array([['a', 'b'], ['c', 'd']])
            m.determinant_inner_product_simplified(bra, ket)
            m.render()
            # With orthogonality: terms where all bra elements match ket elements survive
        """
        bra_terms = self._symbolic_determinant(bra_matrix)
        ket_terms = self._symbolic_determinant(ket_matrix)

        inner_terms = self._compute_inner_product_terms(
            bra_terms, ket_terms, orthogonal=orthogonal
        )

        if not inner_terms:
            self._append("0")
            return self

        # Count surviving terms
        positive_count = sum(1 for sign, _, _ in inner_terms if sign > 0)
        negative_count = sum(1 for sign, _, _ in inner_terms if sign < 0)

        net_result = positive_count - negative_count

        if net_result == 0:
            self._append("0")
        elif net_result > 0:
            self._append(str(net_result))
        else:
            self._append(str(net_result))

        return self

    def slater_inner_product(self, bra_orbitals: List[str], ket_orbitals: List[str],
                              orthogonal: bool = True,
                              normalize: bool = True) -> "Math":
        """
        Render the inner product of two Slater determinants.

        For orthonormal orbitals, ⟨Ψ|Φ⟩ = δ_{Ψ,Φ} (determinants are orthogonal
        unless they have the same set of occupied orbitals).

        Args:
            bra_orbitals: List of orbital labels for bra Slater determinant
            ket_orbitals: List of orbital labels for ket Slater determinant
            orthogonal: If True, orbitals are orthonormal
            normalize: Include normalization factors

        Example:
            m = Math()
            m.slater_inner_product(['a', 'b', 'c'], ['a', 'b', 'c'], orthogonal=True)
            m.render()
        """
        n_bra = len(bra_orbitals)
        n_ket = len(ket_orbitals)

        if normalize:
            # Normalization: 1/sqrt(n!) for each determinant
            self._append(f"\\frac{{1}}{{{n_bra}!}}")

        # Create symbolic matrices for the Slater determinants
        # Each row i represents electron i, each column j represents orbital j
        import numpy as np
        bra_matrix = np.array([[orb for orb in bra_orbitals] for _ in range(n_bra)])
        ket_matrix = np.array([[orb for orb in ket_orbitals] for _ in range(n_ket)])

        # For proper Slater determinant inner product, we need permutation matching
        # The inner product expands to n! terms, with orthogonality reducing this

        if orthogonal:
            # For orthonormal orbitals: ⟨Ψ|Φ⟩ = 1 if same orbitals, 0 otherwise
            # (accounting for antisymmetry and normalization)
            bra_set = set(str(o) for o in bra_orbitals)
            ket_set = set(str(o) for o in ket_orbitals)

            if bra_set == ket_set:
                # Same occupied orbitals - result is 1 (after normalization)
                if not normalize:
                    import math
                    self._append(f"{math.factorial(n_bra)}")
                else:
                    self._append("1")
            else:
                self._append("0")
        else:
            # Show the full expansion without orthogonality simplification
            self._append("\\sum_{P} (-1)^P ")
            overlaps = []
            for i, (b, k) in enumerate(zip(bra_orbitals, ket_orbitals)):
                overlaps.append(f"\\langle {b} | {k} \\rangle")
            self._append(" ".join(overlaps))

        return self

    def determinant_overlap_expansion(self, bra_matrix, ket_matrix,
                                       notation: str = 'braket') -> "Math":
        """
        Render the full overlap expansion of two determinants without simplification.

        Shows all pairwise inner products between bra and ket determinant terms.

        Args:
            bra_matrix: 2D array-like for the bra determinant
            ket_matrix: 2D array-like for the ket determinant
            notation: 'braket' for ⟨a|b⟩ notation, 'product' for (a*·b) notation

        Example:
            m = Math()
            bra = np.array([['a', 'b'], ['c', 'd']])
            ket = np.array([['e', 'f'], ['g', 'h']])
            m.determinant_overlap_expansion(bra, ket)
            m.render()
        """
        bra_terms = self._symbolic_determinant(bra_matrix)
        ket_terms = self._symbolic_determinant(ket_matrix)

        # Show bra expansion
        self._append("\\Bigl(")
        first = True
        for sign, elements in bra_terms:
            if sign > 0:
                if not first:
                    self._append(" + ")
            else:
                self._append(" - ")
            first = False
            self.bra(elements)
        self._append("\\Bigr)")

        self._append("\\Bigl(")
        first = True
        for sign, elements in ket_terms:
            if sign > 0:
                if not first:
                    self._append(" + ")
            else:
                self._append(" - ")
            first = False
            self.ket(elements)
        self._append("\\Bigr)")

        return self

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
