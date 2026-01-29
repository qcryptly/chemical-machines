"""
Chemical Machines Symbols Display Module

LaTeX rendering, MathBuilder, and legacy display classes.
"""

from typing import Optional, List, Union, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC

from .. import views
from .core import Expr, Var, Const, _get_sympy

__all__ = [
    # Legacy classes
    'Symmetry',
    'SpinOrbital',
    'SlaterState',
    'OverlapResult',
    # Rendering
    'MathBuilder',
    'latex',
    'equation',
    'align',
    'matrix',
    'bullets',
    'numbered',
    'items',
    'get_notation',
    'set_notation',
    'set_line_height',
    # Chemistry helpers
    'chemical',
    'reaction',
    'fraction',
    'sqrt',
]


# =============================================================================
# LEGACY SUPPORT - Original Classes Below
# =============================================================================

class Symmetry(Enum):
    """Symmetry types for spin-orbital basis functions."""
    NONE = "none"           # No symmetry (keep all terms)
    SPIN = "spin"           # Spin symmetry (α/β orthogonal)
    SPATIAL = "spatial"     # Spatial symmetry (different spatial orbitals orthogonal)
    ORTHONORMAL = "orthonormal"  # Full orthonormality (⟨φᵢ|φⱼ⟩ = δᵢⱼ)


@dataclass
class SpinOrbital:
    """
    Represents a spin-orbital: a spatial orbital combined with spin.

    Attributes:
        label: Orbital label (e.g., "1s", "2p", "φ₁")
        spin: Spin state ("α", "β", "↑", "↓", or None for spinless)
        spatial_quantum_numbers: Optional dict of quantum numbers (n, l, m, etc.)
        symmetry_group: Optional symmetry group label for point group symmetry
    """
    label: str
    spin: Optional[str] = None
    spatial_quantum_numbers: Optional[Dict[str, int]] = None
    symmetry_group: Optional[str] = None

    def __post_init__(self):
        # Normalize spin notation
        if self.spin in ("↑", "up", "alpha"):
            self.spin = "α"
        elif self.spin in ("↓", "down", "beta"):
            self.spin = "β"

    @property
    def full_label(self) -> str:
        """Return full label including spin if present."""
        if self.spin:
            spin_symbol = "↑" if self.spin == "α" else "↓"
            return f"{self.label}{spin_symbol}"
        return self.label

    @property
    def latex_label(self) -> str:
        """Return LaTeX-formatted label."""
        if self.spin:
            spin_symbol = r"\uparrow" if self.spin == "α" else r"\downarrow"
            return f"{self.label}_{{{spin_symbol}}}"
        return self.label

    def is_orthogonal_to(self, other: "SpinOrbital", symmetry: Symmetry) -> bool:
        """
        Check if this orbital is orthogonal to another under given symmetry.

        Args:
            other: Another SpinOrbital
            symmetry: Symmetry type to apply

        Returns:
            True if orbitals are orthogonal, False otherwise
        """
        if symmetry == Symmetry.NONE:
            return False

        if symmetry == Symmetry.SPIN:
            # Different spins are orthogonal
            if self.spin and other.spin and self.spin != other.spin:
                return True
            return False

        if symmetry == Symmetry.SPATIAL:
            # Different spatial orbitals are orthogonal
            if self.label != other.label:
                return True
            return False

        if symmetry == Symmetry.ORTHONORMAL:
            # Full orthonormality: both spin and spatial must match
            return self.label != other.label or self.spin != other.spin

        return False

    def __eq__(self, other):
        if isinstance(other, SpinOrbital):
            return self.label == other.label and self.spin == other.spin
        return False

    def __hash__(self):
        return hash((self.label, self.spin))

    def __str__(self):
        return self.full_label


@dataclass
class SlaterState:
    """
    Represents a Slater determinant state as a list of occupied spin-orbitals.

    This is the primary interface for working with Slater determinants.
    Instead of specifying a full matrix, you provide a 1D list of occupied
    orbitals, and the class handles the determinant structure automatically.

    The electron coordinates are implicitly coupled with the spin-orbitals:
    electron 1 is associated with the first orbital, electron 2 with the second, etc.

    Attributes:
        orbitals: List of occupied spin-orbitals
        symmetry: Symmetry type for orthogonality rules
        basis_name: Optional name for the basis set

    Example:
        # Create a 3-electron state
        state = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])

        # Or with explicit SpinOrbital objects
        orbs = [SpinOrbital("1s", "α"), SpinOrbital("1s", "β"), SpinOrbital("2s", "α")]
        state = SlaterState(orbs, symmetry=Symmetry.ORTHONORMAL)

        # Use with Math class
        m = Math()
        m.slater_bra(state)  # ⟨1s↑, 1s↓, 2s↑|
    """
    orbitals: List[SpinOrbital]
    symmetry: Symmetry = Symmetry.ORTHONORMAL
    basis_name: Optional[str] = None

    @classmethod
    def from_labels(cls, labels: List[str], symmetry: Symmetry = Symmetry.ORTHONORMAL,
                    basis_name: Optional[str] = None) -> "SlaterState":
        """
        Create a SlaterState from string labels.

        Parses spin notation from labels:
        - "1s↑" or "1sα" or "1s_up" -> SpinOrbital("1s", "α")
        - "2p↓" or "2pβ" or "2p_down" -> SpinOrbital("2p", "β")
        - "φ1" -> SpinOrbital("φ1", None) (no spin)

        Args:
            labels: List of orbital labels with optional spin notation
            symmetry: Symmetry type for orthogonality rules
            basis_name: Optional name for the basis set
        """
        orbitals = []
        for label in labels:
            # Parse spin from label
            spin = None
            base_label = label

            for spin_marker, spin_val in [
                ("↑", "α"), ("↓", "β"),
                ("α", "α"), ("β", "β"),
                ("_up", "α"), ("_down", "β"),
                ("_alpha", "α"), ("_beta", "β"),
            ]:
                if spin_marker in label:
                    spin = spin_val
                    base_label = label.replace(spin_marker, "")
                    break

            orbitals.append(SpinOrbital(base_label, spin))

        return cls(orbitals, symmetry, basis_name)

    @classmethod
    def from_occupation(cls, occupation: Dict[str, int],
                        symmetry: Symmetry = Symmetry.ORTHONORMAL) -> "SlaterState":
        """
        Create a SlaterState from occupation numbers.

        Args:
            occupation: Dict mapping orbital labels to occupation (0, 1, or 2)
            symmetry: Symmetry type

        Example:
            state = SlaterState.from_occupation({"1s": 2, "2s": 1})
            # Creates: |1s↑, 1s↓, 2s↑⟩
        """
        orbitals = []
        for label, n in occupation.items():
            if n >= 1:
                orbitals.append(SpinOrbital(label, "α"))
            if n >= 2:
                orbitals.append(SpinOrbital(label, "β"))
        return cls(orbitals, symmetry)

    @property
    def n_electrons(self) -> int:
        """Number of electrons in this state."""
        return len(self.orbitals)

    @property
    def labels(self) -> List[str]:
        """List of full orbital labels."""
        return [orb.full_label for orb in self.orbitals]

    @property
    def latex_labels(self) -> List[str]:
        """List of LaTeX-formatted orbital labels."""
        return [orb.latex_label for orb in self.orbitals]

    def overlap_with(self, other: "SlaterState") -> Tuple[bool, Optional[int]]:
        """
        Compute overlap with another Slater state under current symmetry.

        For orthonormal orbitals:
        - ⟨Ψ|Φ⟩ = 0 if different occupied orbitals
        - ⟨Ψ|Φ⟩ = (-1)^P if same orbitals (P = permutation parity)

        Returns:
            (is_nonzero, sign) where sign is +1, -1, or None if zero
        """
        if self.n_electrons != other.n_electrons:
            return (False, None)

        if self.symmetry == Symmetry.NONE:
            return (True, None)  # Can't determine without explicit computation

        # Check if same set of orbitals (for orthonormal case)
        if self.symmetry == Symmetry.ORTHONORMAL:
            self_set = set(self.orbitals)
            other_set = set(other.orbitals)

            if self_set != other_set:
                return (False, None)

            # Compute permutation parity
            # Find the permutation that maps other's order to self's order
            parity = self._compute_permutation_parity(other)
            return (True, parity)

        return (True, None)

    def _compute_permutation_parity(self, other: "SlaterState") -> int:
        """Compute the parity of permutation between two states with same orbitals."""
        # Build index map
        other_indices = {orb: i for i, orb in enumerate(other.orbitals)}
        perm = [other_indices[orb] for orb in self.orbitals]

        # Count inversions
        inversions = 0
        n = len(perm)
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inversions += 1

        return 1 if inversions % 2 == 0 else -1

    def to_matrix(self) -> List[List[str]]:
        """
        Convert to symbolic matrix representation for determinant expansion.

        Returns n×n matrix where element [i][j] represents orbital j evaluated
        at electron i's coordinates: φⱼ(rᵢ)
        """
        n = self.n_electrons
        matrix = []
        for i in range(n):
            row = []
            for j, orb in enumerate(self.orbitals):
                # Element represents φⱼ(rᵢ) - orbital j at electron i's position
                row.append(orb.latex_label)
            matrix.append(row)
        return matrix

    def __str__(self):
        return f"|{', '.join(self.labels)}⟩"

    def __repr__(self):
        return f"SlaterState({self.labels}, symmetry={self.symmetry.value})"

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


def _render_scrollable(expression: str, display: bool = True, label: Optional[str] = None,
                       justify: str = "center"):
    """
    Render a LaTeX expression in a scrollable container for long equations.

    This wraps the math output in a div with horizontal scrolling enabled,
    useful for very long equations that would otherwise overflow.
    """
    if display:
        delim_start, delim_end = r"\[", r"\]"
    else:
        delim_start, delim_end = r"\(", r"\)"

    justify_class = f"cm-math-{justify}" if justify in ("left", "center", "right") else "cm-math-center"
    line_height_style = _get_line_height_style()

    # Wrap in scrollable container
    html_content = f'<div class="cm-math-scroll"><div class="cm-math {justify_class}"{line_height_style}>{delim_start}{expression}{delim_end}</div></div>'

    if label:
        html_content = f'<div class="cm-math-labeled"{line_height_style}><span class="cm-math-label">{label}</span>{html_content}</div>'

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


class MathBuilder:
    """
    A builder class for constructing LaTeX expressions programmatically.

    NOTE: This is the legacy API. For new code, use the Math factory:
        from cm.symbols import Math
        x = Math.var("x")
        expr = x**2 + 1
        expr.render()

    Legacy Example:
        m = MathBuilder()
        m.frac("a", "b").plus().sqrt("c").equals().text("result")
        m.render()

        # Bra-ket notation (with braket style)
        m = MathBuilder()
        m.bra("psi").ket("phi")
        m.render()

        # Operator overloads for inner products
        m_g = MathBuilder()
        m_g.determinant_bra(sm_g).equals().bra('\\phi_1')

        m_1 = MathBuilder()
        m_1.determinant_ket(sm_1).equals().ket('\\phi_2')

        m_s = m_g @ m_1  # Inner product of both sides
        m_s.render()
    """

    def __init__(self):
        self._parts: List[str] = []
        # Track equation structure for operator overloads
        self._lhs_parts: List[str] = []  # Left-hand side of equation
        self._rhs_parts: List[str] = []  # Right-hand side of equation
        self._has_equals: bool = False
        # Track determinant matrices for inner product computation
        self._lhs_det_matrix = None  # Determinant matrix on LHS
        self._rhs_det_matrix = None  # Determinant matrix on RHS
        self._lhs_det_type: str = None  # 'bra' or 'ket'
        self._rhs_det_type: str = None  # 'bra' or 'ket'
        # Track simple bra/ket labels
        self._lhs_bra_label = None
        self._rhs_bra_label = None
        self._lhs_ket_label = None
        self._rhs_ket_label = None
        # Track if this is an operator (for matrix element notation)
        self._is_operator: bool = False
        self._lhs_operator_str: str = None  # Operator expression for LHS
        self._rhs_operator_str: str = None  # Operator expression for RHS
        # Track pending operator for chained @ operations (bra @ op @ ket)
        self._pending_operator: "Math" = None
        # Track SlaterState objects for inner products
        self._lhs_slater_state: "SlaterState" = None
        self._rhs_slater_state: "SlaterState" = None
        self._lhs_slater_type: str = None  # 'bra' or 'ket'
        self._rhs_slater_type: str = None  # 'bra' or 'ket'

    def _append(self, content: str) -> "MathBuilder":
        self._parts.append(content)
        # Track LHS vs RHS for operator overloads
        if self._has_equals:
            self._rhs_parts.append(content)
        else:
            self._lhs_parts.append(content)
        return self

    def raw(self, latex: str) -> "MathBuilder":
        """Add raw LaTeX content."""
        return self._append(latex)

    def text(self, content: str) -> "MathBuilder":
        """Add text (non-italic) content."""
        return self._append(f"\\text{{{content}}}")

    def var(self, name: str) -> "MathBuilder":
        """Add a variable."""
        return self._append(name)

    # Basic operations
    def plus(self) -> "MathBuilder":
        return self._append(" + ")

    def minus(self) -> "MathBuilder":
        return self._append(" - ")

    def times(self) -> "MathBuilder":
        return self._append(" \\times ")

    def cdot(self) -> "MathBuilder":
        return self._append(" \\cdot ")

    def div(self) -> "MathBuilder":
        return self._append(" \\div ")

    def equals(self) -> "MathBuilder":
        self._has_equals = True
        return self._append(" = ")

    def approx(self) -> "MathBuilder":
        return self._append(" \\approx ")

    def neq(self) -> "MathBuilder":
        return self._append(" \\neq ")

    def lt(self) -> "MathBuilder":
        return self._append(" < ")

    def gt(self) -> "MathBuilder":
        return self._append(" > ")

    def leq(self) -> "MathBuilder":
        return self._append(" \\leq ")

    def geq(self) -> "MathBuilder":
        return self._append(" \\geq ")

    # Fractions and roots
    def frac(self, num: str, denom: str) -> "MathBuilder":
        """Add a fraction."""
        return self._append(f"\\frac{{{num}}}{{{denom}}}")

    def sqrt(self, content: str, n: Optional[str] = None) -> "MathBuilder":
        """Add a square root or nth root."""
        if n:
            return self._append(f"\\sqrt[{n}]{{{content}}}")
        return self._append(f"\\sqrt{{{content}}}")

    # Subscripts and superscripts
    def sub(self, content: str) -> "MathBuilder":
        """Add a subscript."""
        return self._append(f"_{{{content}}}")

    def sup(self, content: str) -> "MathBuilder":
        """Add a superscript."""
        return self._append(f"^{{{content}}}")

    def subsup(self, sub: str, sup: str) -> "MathBuilder":
        """Add both subscript and superscript."""
        return self._append(f"_{{{sub}}}^{{{sup}}}")

    # Greek letters
    def alpha(self) -> "MathBuilder": return self._append("\\alpha")
    def beta(self) -> "MathBuilder": return self._append("\\beta")
    def gamma(self) -> "MathBuilder": return self._append("\\gamma")
    def delta(self) -> "MathBuilder": return self._append("\\delta")
    def epsilon(self) -> "MathBuilder": return self._append("\\epsilon")
    def zeta(self) -> "MathBuilder": return self._append("\\zeta")
    def eta(self) -> "MathBuilder": return self._append("\\eta")
    def theta(self) -> "MathBuilder": return self._append("\\theta")
    def iota(self) -> "MathBuilder": return self._append("\\iota")
    def kappa(self) -> "MathBuilder": return self._append("\\kappa")
    def lambda_(self) -> "MathBuilder": return self._append("\\lambda")
    def mu(self) -> "MathBuilder": return self._append("\\mu")
    def nu(self) -> "MathBuilder": return self._append("\\nu")
    def xi(self) -> "MathBuilder": return self._append("\\xi")
    def pi(self) -> "MathBuilder": return self._append("\\pi")
    def rho(self) -> "MathBuilder": return self._append("\\rho")
    def sigma(self) -> "MathBuilder": return self._append("\\sigma")
    def tau(self) -> "MathBuilder": return self._append("\\tau")
    def upsilon(self) -> "MathBuilder": return self._append("\\upsilon")
    def phi(self) -> "MathBuilder": return self._append("\\phi")
    def chi(self) -> "MathBuilder": return self._append("\\chi")
    def psi(self) -> "MathBuilder": return self._append("\\psi")
    def omega(self) -> "MathBuilder": return self._append("\\omega")

    # Capital Greek
    def Gamma(self) -> "MathBuilder": return self._append("\\Gamma")
    def Delta(self) -> "MathBuilder": return self._append("\\Delta")
    def Theta(self) -> "MathBuilder": return self._append("\\Theta")
    def Lambda(self) -> "MathBuilder": return self._append("\\Lambda")
    def Xi(self) -> "MathBuilder": return self._append("\\Xi")
    def Pi(self) -> "MathBuilder": return self._append("\\Pi")
    def Sigma(self) -> "MathBuilder": return self._append("\\Sigma")
    def Phi(self) -> "MathBuilder": return self._append("\\Phi")
    def Psi(self) -> "MathBuilder": return self._append("\\Psi")
    def Omega(self) -> "MathBuilder": return self._append("\\Omega")

    # Calculus
    def integral(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "MathBuilder":
        """Add an integral sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\int_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\int_{{{lower}}}")
        return self._append("\\int")

    def sum(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "MathBuilder":
        """Add a summation sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\sum_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\sum_{{{lower}}}")
        return self._append("\\sum")

    def prod(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "MathBuilder":
        """Add a product sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\prod_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\prod_{{{lower}}}")
        return self._append("\\prod")

    def lim(self, var: str, to: str) -> "MathBuilder":
        """Add a limit."""
        return self._append(f"\\lim_{{{var} \\to {to}}}")

    def deriv(self, func: str = "", var: str = "x") -> "MathBuilder":
        """Add a derivative."""
        if func:
            return self._append(f"\\frac{{d{func}}}{{d{var}}}")
        return self._append(f"\\frac{{d}}{{d{var}}}")

    def partial(self, func: str = "", var: str = "x") -> "MathBuilder":
        """Add a partial derivative."""
        if func:
            return self._append(f"\\frac{{\\partial {func}}}{{\\partial {var}}}")
        return self._append(f"\\frac{{\\partial}}{{\\partial {var}}}")

    def nabla(self) -> "MathBuilder":
        return self._append("\\nabla")

    # Brackets and grouping
    def paren(self, content: str) -> "MathBuilder":
        """Add parentheses."""
        return self._append(f"\\left({content}\\right)")

    def bracket(self, content: str) -> "MathBuilder":
        """Add square brackets."""
        return self._append(f"\\left[{content}\\right]")

    def brace(self, content: str) -> "MathBuilder":
        """Add curly braces."""
        return self._append(f"\\left\\{{{content}\\right\\}}")

    def abs(self, content: str) -> "MathBuilder":
        """Add absolute value bars."""
        return self._append(f"\\left|{content}\\right|")

    def norm(self, content: str) -> "MathBuilder":
        """Add norm double bars."""
        return self._append(f"\\left\\|{content}\\right\\|")

    # Quantum mechanics / Bra-ket
    def bra(self, content) -> "MathBuilder":
        """Add a bra <content|. Content can be a string or list of quantum numbers."""
        original_content = content
        if isinstance(content, (list, tuple)):
            content = ", ".join(str(c) for c in content)
        # Track for operator overloads
        if self._has_equals:
            self._rhs_bra_label = original_content
        else:
            self._lhs_bra_label = original_content
        return self._append(f"\\langle {content} |")

    def ket(self, content) -> "MathBuilder":
        """Add a ket |content>. Content can be a string or list of quantum numbers."""
        original_content = content
        if isinstance(content, (list, tuple)):
            content = ", ".join(str(c) for c in content)
        # Track for operator overloads
        if self._has_equals:
            self._rhs_ket_label = original_content
        else:
            self._lhs_ket_label = original_content
        return self._append(f"| {content} \\rangle")

    def braket(self, bra, ket) -> "MathBuilder":
        """Add a braket <bra|ket>. Arguments can be strings or lists of quantum numbers."""
        if isinstance(bra, (list, tuple)):
            bra = ", ".join(str(c) for c in bra)
        if isinstance(ket, (list, tuple)):
            ket = ", ".join(str(c) for c in ket)
        return self._append(f"\\langle {bra} | {ket} \\rangle")

    def expval(self, operator) -> "MathBuilder":
        """Add an expectation value <operator>."""
        if isinstance(operator, (list, tuple)):
            operator = ", ".join(str(c) for c in operator)
        return self._append(f"\\langle {operator} \\rangle")

    def matelem(self, bra, op, ket) -> "MathBuilder":
        """Add a matrix element <bra|op|ket>."""
        if isinstance(bra, (list, tuple)):
            bra = ", ".join(str(c) for c in bra)
        if isinstance(ket, (list, tuple)):
            ket = ", ".join(str(c) for c in ket)
        return self._append(f"\\langle {bra} | {op} | {ket} \\rangle")

    def op(self, name: str) -> "MathBuilder":
        """Add an operator with hat."""
        return self._append(f"\\hat{{{name}}}")

    def dagger(self) -> "MathBuilder":
        """Add a dagger superscript."""
        return self._append("^\\dagger")

    def comm(self, a: str, b: str) -> "MathBuilder":
        """Add a commutator [a, b]."""
        return self._append(f"[{a}, {b}]")

    # Physics
    def vec(self, content: str) -> "MathBuilder":
        """Add a vector with arrow."""
        return self._append(f"\\vec{{{content}}}")

    def hbar(self) -> "MathBuilder":
        return self._append("\\hbar")

    def infty(self) -> "MathBuilder":
        return self._append("\\infty")

    # Chemistry
    def ce(self, formula: str) -> "MathBuilder":
        """Add a chemical formula (upright text)."""
        return self._append(f"\\mathrm{{{formula}}}")

    def yields(self) -> "MathBuilder":
        """Add a reaction arrow."""
        return self._append(" \\rightarrow ")

    def equilibrium(self) -> "MathBuilder":
        """Add an equilibrium arrow."""
        return self._append(" \\rightleftharpoons ")

    # Special functions
    def sin(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\sin{{{arg}}}" if arg else "\\sin")

    def cos(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\cos{{{arg}}}" if arg else "\\cos")

    def tan(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\tan{{{arg}}}" if arg else "\\tan")

    def ln(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\ln{{{arg}}}" if arg else "\\ln")

    def log(self, arg: str = "", base: Optional[str] = None) -> "MathBuilder":
        if base:
            return self._append(f"\\log_{{{base}}}{{{arg}}}" if arg else f"\\log_{{{base}}}")
        return self._append(f"\\log{{{arg}}}" if arg else "\\log")

    def exp(self, arg: str = "") -> "MathBuilder":
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

    def determinant_bra(self, matrix) -> "MathBuilder":
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
        import numpy as np
        matrix = np.asarray(matrix)
        # Track for operator overloads
        if self._has_equals:
            self._rhs_det_matrix = matrix
            self._rhs_det_type = 'bra'
        else:
            self._lhs_det_matrix = matrix
            self._lhs_det_type = 'bra'
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'bra')

    def determinant_ket(self, matrix) -> "MathBuilder":
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
        import numpy as np
        matrix = np.asarray(matrix)
        # Track for operator overloads
        if self._has_equals:
            self._rhs_det_matrix = matrix
            self._rhs_det_type = 'ket'
        else:
            self._lhs_det_matrix = matrix
            self._lhs_det_type = 'ket'
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'ket')

    def determinant_braket(self, matrix, bra_label: str = "\\psi") -> "MathBuilder":
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

    def determinant_product(self, matrix) -> "MathBuilder":
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

    def determinant_subscript(self, matrix, var: str = "a") -> "MathBuilder":
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
                                bra_label: str = None) -> "MathBuilder":
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

    def slater_determinant(self, orbitals: List[str], normalize: bool = True) -> "MathBuilder":
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

    def slater_ket(self, orbitals: List[str], normalize: bool = True) -> "MathBuilder":
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

    def slater_bra_state(self, state: "SlaterState", normalize: bool = False) -> "MathBuilder":
        """
        Render a SlaterState as a bra ⟨orbitals|.

        This method works with the new SlaterState class that represents
        a Slater determinant as a 1D list of occupied spin-orbitals.

        Args:
            state: SlaterState object
            normalize: Include normalization factor 1/√n!

        Example:
            state = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
            m = Math()
            m.slater_bra_state(state)
            m.render()
            # Renders: ⟨1s↑, 1s↓, 2s↑|
        """
        n = state.n_electrons

        if normalize:
            self._append(f"\\frac{{1}}{{\\sqrt{{{n}!}}}}")

        # Track the SlaterState for operator overloads
        if self._has_equals:
            self._rhs_slater_state = state
            self._rhs_slater_type = 'bra'
        else:
            self._lhs_slater_state = state
            self._lhs_slater_type = 'bra'

        self.bra(state.latex_labels)
        return self

    def slater_ket_state(self, state: "SlaterState", normalize: bool = False) -> "MathBuilder":
        """
        Render a SlaterState as a ket |orbitals⟩.

        This method works with the new SlaterState class that represents
        a Slater determinant as a 1D list of occupied spin-orbitals.

        Args:
            state: SlaterState object
            normalize: Include normalization factor 1/√n!

        Example:
            state = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
            m = Math()
            m.slater_ket_state(state)
            m.render()
            # Renders: |1s↑, 1s↓, 2s↑⟩
        """
        n = state.n_electrons

        if normalize:
            self._append(f"\\frac{{1}}{{\\sqrt{{{n}!}}}}")

        # Track the SlaterState for operator overloads
        if self._has_equals:
            self._rhs_slater_state = state
            self._rhs_slater_type = 'ket'
        else:
            self._lhs_slater_state = state
            self._lhs_slater_type = 'ket'

        self.ket(state.latex_labels)
        return self

    def slater_matrix_element(self, bra_state: "SlaterState", operator: str,
                               ket_state: "SlaterState",
                               apply_symmetry: bool = True) -> "MathBuilder":
        """
        Render a matrix element ⟨bra|op|ket⟩ between two SlaterStates.

        When apply_symmetry is True and states have orthonormal symmetry,
        uses Slater-Condon rules to simplify:
        - If states differ by more than 2 orbitals: result is 0
        - If states are identical: sum of one-electron terms + two-electron terms
        - If states differ by 1 orbital: one-electron + two-electron terms
        - If states differ by 2 orbitals: only two-electron terms

        Args:
            bra_state: SlaterState for bra
            operator: LaTeX string for operator (e.g., "\\hat{H}")
            ket_state: SlaterState for ket
            apply_symmetry: Use symmetry to simplify (default True)

        Example:
            psi = SlaterState.from_labels(["1s↑", "1s↓"])
            phi = SlaterState.from_labels(["1s↑", "2s↑"])
            m = Math()
            m.slater_matrix_element(psi, "\\hat{H}", phi)
            m.render()
        """
        # Render the full matrix element notation
        bra_str = ", ".join(bra_state.latex_labels)
        ket_str = ", ".join(ket_state.latex_labels)
        self._append(f"\\langle {bra_str} | {operator} | {ket_str} \\rangle")

        return self

    def slater_overlap(self, bra_state: "SlaterState", ket_state: "SlaterState",
                        simplify: bool = True) -> "MathBuilder":
        """
        Render and optionally simplify the overlap ⟨bra|ket⟩ between two SlaterStates.

        When simplify is True and states have orthonormal symmetry:
        - Returns 0 if different sets of orbitals
        - Returns ±1 based on permutation parity if same orbitals

        Args:
            bra_state: SlaterState for bra
            ket_state: SlaterState for ket
            simplify: Apply symmetry simplifications (default True)

        Example:
            psi = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
            phi = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
            m = Math()
            m.slater_overlap(psi, phi)
            m.render()
            # Renders: 1 (same orbitals)
        """
        if simplify and bra_state.symmetry == Symmetry.ORTHONORMAL:
            is_nonzero, sign = bra_state.overlap_with(ket_state)

            if not is_nonzero:
                self._append("0")
            elif sign is not None:
                self._append(str(sign))
            else:
                # Render full overlap
                bra_str = ", ".join(bra_state.latex_labels)
                ket_str = ", ".join(ket_state.latex_labels)
                self._append(f"\\langle {bra_str} | {ket_str} \\rangle")
        else:
            # Render without simplification
            bra_str = ", ".join(bra_state.latex_labels)
            ket_str = ", ".join(ket_state.latex_labels)
            self._append(f"\\langle {bra_str} | {ket_str} \\rangle")

        return self

    def slater_condon_rule(self, bra_state: "SlaterState", ket_state: "SlaterState",
                           operator_type: str = "one_electron") -> "MathBuilder":
        """
        Apply Slater-Condon rules to determine which terms survive.

        This analyzes the difference between two Slater states and renders
        the appropriate Slater-Condon expression.

        Args:
            bra_state: SlaterState for bra
            ket_state: SlaterState for ket
            operator_type: "one_electron" for ĥ(i), "two_electron" for ĝ(i,j)

        Returns:
            Math object with the Slater-Condon expression
        """
        bra_set = set(bra_state.orbitals)
        ket_set = set(ket_state.orbitals)

        # Find differing orbitals
        only_in_bra = bra_set - ket_set
        only_in_ket = ket_set - bra_set
        n_diff = len(only_in_bra)

        if n_diff > 2:
            # More than 2 orbitals differ: matrix element is zero
            self._append("0")
            return self

        if n_diff == 0:
            # Same orbitals (diagonal case)
            if operator_type == "one_electron":
                # Sum over all occupied orbitals
                self._append("\\sum_i ")
                self._append("\\langle i | \\hat{h} | i \\rangle")
            else:
                # Two-electron: sum over pairs
                self._append("\\frac{1}{2} \\sum_{i \\neq j} ")
                self._append("\\left[ \\langle ij | \\hat{g} | ij \\rangle - \\langle ij | \\hat{g} | ji \\rangle \\right]")

        elif n_diff == 1:
            # Single excitation
            orb_bra = list(only_in_bra)[0]
            orb_ket = list(only_in_ket)[0]

            if operator_type == "one_electron":
                self._append(f"\\langle {orb_bra.latex_label} | \\hat{{h}} | {orb_ket.latex_label} \\rangle")
            else:
                # Two-electron with common orbitals
                common = bra_set & ket_set
                self._append("\\sum_j ")
                self._append(f"\\left[ \\langle {orb_bra.latex_label} j | \\hat{{g}} | {orb_ket.latex_label} j \\rangle ")
                self._append(f"- \\langle {orb_bra.latex_label} j | \\hat{{g}} | j {orb_ket.latex_label} \\rangle \\right]")

        else:  # n_diff == 2
            # Double excitation - only two-electron operator contributes
            if operator_type == "one_electron":
                self._append("0")
            else:
                orbs_bra = list(only_in_bra)
                orbs_ket = list(only_in_ket)
                b1, b2 = orbs_bra[0].latex_label, orbs_bra[1].latex_label
                k1, k2 = orbs_ket[0].latex_label, orbs_ket[1].latex_label

                self._append(f"\\langle {b1} {b2} | \\hat{{g}} | {k1} {k2} \\rangle ")
                self._append(f"- \\langle {b1} {b2} | \\hat{{g}} | {k2} {k1} \\rangle")

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
                                   show_zeros: bool = False) -> "MathBuilder":
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
                                              orthogonal: bool = True) -> "MathBuilder":
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
                              normalize: bool = True) -> "MathBuilder":
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
                                       notation: str = 'braket') -> "MathBuilder":
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
    def space(self) -> "MathBuilder":
        return self._append("\\ ")

    def quad(self) -> "MathBuilder":
        return self._append("\\quad")

    def qquad(self) -> "MathBuilder":
        return self._append("\\qquad")

    # Line breaks for multi-line equations
    def newline(self) -> "MathBuilder":
        """Add a line break (\\\\) for use in aligned environments."""
        return self._append(" \\\\ ")

    def br(self) -> "MathBuilder":
        """Alias for newline() - add a line break."""
        return self.newline()

    def align_eq(self) -> "MathBuilder":
        """Add alignment marker (&) followed by equals sign for aligned environments."""
        return self._append(" &= ")

    def align_mark(self) -> "MathBuilder":
        """Add alignment marker (&) for aligned environments."""
        return self._append(" & ")

    # Build and render
    def build(self) -> str:
        """Build and return the LaTeX string."""
        return "".join(self._parts)

    def render(self, display: bool = True, label: Optional[str] = None,
               justify: str = "center", multiline: bool = False, scrollable: bool = False):
        """
        Render the built expression to HTML output.

        Args:
            display: If True, render as display math (block). If False, inline.
            label: Optional label/caption for the expression.
            justify: Alignment - 'left', 'center', or 'right'.
            multiline: If True, wrap in aligned environment for line breaks.
            scrollable: If True, wrap in scrollable container for long equations.
        """
        expr = self.build()

        # Wrap in aligned environment for multi-line equations
        if multiline or " \\\\ " in expr:
            expr = f"\\begin{{aligned}} {expr} \\end{{aligned}}"

        # Use scrollable container if requested
        if scrollable:
            _render_scrollable(expr, display=display, label=label, justify=justify)
        else:
            latex(expr, display=display, label=label, justify=justify)

    def clear(self) -> "MathBuilder":
        """Clear the builder."""
        self._parts = []
        return self

    def __str__(self) -> str:
        return self.build()

    # Operator overloads for inner products
    def __matmul__(self, other: "Math") -> "MathBuilder":
        """
        Matrix multiplication operator (@) for computing inner products.

        When two Math objects are combined with @, it computes the inner product
        of both the LHS and RHS of their equations separately.

        For bra @ ket operations:
        - LHS: ⟨bra_lhs| @ |ket_lhs⟩ = ⟨bra_lhs|ket_lhs⟩
        - RHS: ⟨bra_rhs| @ |ket_rhs⟩ = ⟨bra_rhs|ket_rhs⟩

        For bra @ op @ ket operations (matrix elements):
        - LHS: ⟨bra_lhs|op_lhs|ket_lhs⟩
        - RHS: ⟨bra_rhs|op_rhs|ket_rhs⟩

        Example:
            m_g = Math()
            m_g.determinant_bra(sm_g).equals().bra('\\phi_1')

            m_1 = Math()
            m_1.determinant_ket(sm_1).equals().ket('\\phi_2')

            m_s = m_g @ m_1  # Inner product of both sides
            m_s.render()

            # With operator:
            my_op = Math().var("h")
            m_o = m_g @ my_op @ m_1  # Matrix element ⟨...|h|...⟩
            m_o.render()
        """
        if not isinstance(other, Math):
            raise TypeError(f"unsupported operand type(s) for @: 'Math' and '{type(other).__name__}'")

        # Check what kind of objects we're dealing with
        self_is_bra = (self._lhs_det_type == 'bra' or self._lhs_bra_label is not None or
                       self._lhs_slater_type == 'bra')
        self_has_pending_op = self._pending_operator is not None
        other_is_ket = (other._lhs_det_type == 'ket' or other._lhs_ket_label is not None or
                        other._lhs_slater_type == 'ket')
        other_is_bra = (other._lhs_det_type == 'bra' or other._lhs_bra_label is not None or
                        other._lhs_slater_type == 'bra')

        # Case 1: bra @ op (store operator for later, return intermediate result)
        # The "other" is an operator (not a bra or ket)
        if self_is_bra and not other_is_ket and not other_is_bra:
            result = Math()
            # Copy bra information to result
            result._lhs_det_matrix = self._lhs_det_matrix
            result._lhs_det_type = self._lhs_det_type
            result._lhs_bra_label = self._lhs_bra_label
            result._rhs_det_matrix = self._rhs_det_matrix
            result._rhs_det_type = self._rhs_det_type
            result._rhs_bra_label = self._rhs_bra_label
            result._has_equals = self._has_equals
            # Copy SlaterState info
            result._lhs_slater_state = self._lhs_slater_state
            result._lhs_slater_type = self._lhs_slater_type
            result._rhs_slater_state = self._rhs_slater_state
            result._rhs_slater_type = self._rhs_slater_type
            # Store operator for when we encounter the ket
            result._pending_operator = other
            return result

        # Case 2: (bra @ op) @ ket - we have a pending operator
        if self_has_pending_op and other_is_ket:
            result = Math()
            op = self._pending_operator

            # Get operator strings for LHS and RHS
            # If operator doesn't have equals, use same expression for both sides
            lhs_op_str = "".join(op._lhs_parts) if op._lhs_parts else op.build()
            if op._has_equals:
                rhs_op_str = "".join(op._rhs_parts) if op._rhs_parts else op.build()
            else:
                rhs_op_str = lhs_op_str

            # Compute LHS matrix element
            lhs_computed = self._compute_side_matrix_element(
                self._lhs_det_matrix, self._lhs_det_type, self._lhs_bra_label,
                lhs_op_str,
                other._lhs_det_matrix, other._lhs_det_type, other._lhs_ket_label
            )
            result._append(lhs_computed)

            # If both have equations, compute RHS matrix element
            if self._has_equals and other._has_equals:
                result._has_equals = True
                result._append(" = ")

                rhs_computed = self._compute_side_matrix_element(
                    self._rhs_det_matrix, self._rhs_det_type, self._rhs_bra_label,
                    rhs_op_str,
                    other._rhs_det_matrix, other._rhs_det_type, other._rhs_ket_label
                )
                result._append(rhs_computed)
            elif self._has_equals or other._has_equals:
                # One has equals - use operator's expression for RHS
                result._has_equals = True
                result._append(" = ")

                # Use RHS if available, otherwise LHS
                bra_matrix = self._rhs_det_matrix if self._has_equals else self._lhs_det_matrix
                bra_type = self._rhs_det_type if self._has_equals else self._lhs_det_type
                bra_label = self._rhs_bra_label if self._has_equals else self._lhs_bra_label
                ket_matrix = other._rhs_det_matrix if other._has_equals else other._lhs_det_matrix
                ket_type = other._rhs_det_type if other._has_equals else other._lhs_det_type
                ket_label = other._rhs_ket_label if other._has_equals else other._lhs_ket_label

                rhs_computed = self._compute_side_matrix_element(
                    bra_matrix, bra_type, bra_label,
                    rhs_op_str,
                    ket_matrix, ket_type, ket_label
                )
                result._append(rhs_computed)

            return result

        # Case 3: Simple bra @ ket (no operator)
        if self_is_bra and other_is_ket:
            result = Math()

            # Check if we're using SlaterState objects
            if self._lhs_slater_state is not None and other._lhs_slater_state is not None:
                # Use SlaterState-aware inner product
                lhs_computed = self._compute_slater_inner_product(
                    self._lhs_slater_state, other._lhs_slater_state
                )
                result._append(lhs_computed)

                # If both have equations, compute RHS inner product
                if self._has_equals and other._has_equals:
                    result._has_equals = True
                    result._append(" = ")

                    rhs_computed = self._compute_slater_inner_product(
                        self._rhs_slater_state, other._rhs_slater_state
                    )
                    result._append(rhs_computed)
            else:
                # Fall back to original matrix-based inner product
                lhs_computed = self._compute_side_inner_product(
                    self._lhs_det_matrix, self._lhs_det_type, self._lhs_bra_label, self._lhs_ket_label,
                    other._lhs_det_matrix, other._lhs_det_type, other._lhs_bra_label, other._lhs_ket_label
                )
                result._append(lhs_computed)

                # If both have equations, compute RHS inner product
                if self._has_equals and other._has_equals:
                    result._has_equals = True
                    result._append(" = ")

                    rhs_computed = self._compute_side_inner_product(
                        self._rhs_det_matrix, self._rhs_det_type, self._rhs_bra_label, self._rhs_ket_label,
                        other._rhs_det_matrix, other._rhs_det_type, other._rhs_bra_label, other._rhs_ket_label
                    )
                    result._append(rhs_computed)

            return result

        # Invalid combination
        if not self_is_bra and not self_has_pending_op:
            raise ValueError("Left operand of @ must contain bra notation (use determinant_bra or bra)")
        raise ValueError("Right operand of @ must contain ket notation or be an operator expression")

    def _compute_slater_inner_product(self, bra_state: "SlaterState",
                                       ket_state: "SlaterState") -> str:
        """
        Compute the inner product ⟨bra|ket⟩ between two SlaterState objects.

        Uses the symmetry settings on the bra_state to determine simplification.
        For orthonormal states, applies Kronecker delta rules.
        """
        if bra_state.symmetry == Symmetry.ORTHONORMAL:
            is_nonzero, sign = bra_state.overlap_with(ket_state)

            if not is_nonzero:
                return "0"
            elif sign is not None:
                return str(sign)

        # Render full overlap notation
        bra_str = ", ".join(bra_state.latex_labels)
        ket_str = ", ".join(ket_state.latex_labels)
        return f"\\langle {bra_str} | {ket_str} \\rangle"

    def _compute_side_inner_product(self, bra_matrix, bra_det_type, bra_label, bra_ket_label,
                                     ket_matrix, ket_det_type, ket_bra_label, ket_label) -> str:
        """
        Compute the inner product for one side (LHS or RHS) of the equation.

        Handles combinations of:
        - determinant_bra @ determinant_ket -> combined braket notation ⟨a,b,c|d,e,f⟩
        - bra @ ket -> simple braket
        - determinant_bra @ ket -> simplified notation
        - bra @ determinant_ket -> simplified notation
        """
        import numpy as np

        # Case 1: Both are determinants
        if bra_matrix is not None and ket_matrix is not None:
            if bra_det_type != 'bra':
                raise ValueError("Left side must be bra notation for inner product")
            if ket_det_type != 'ket':
                raise ValueError("Right side must be ket notation for inner product")

            # Check shape compatibility
            bra_shape = np.asarray(bra_matrix).shape
            ket_shape = np.asarray(ket_matrix).shape
            if bra_shape != ket_shape:
                raise ValueError(f"Shape mismatch: bra has shape {bra_shape}, ket has shape {ket_shape}")

            # Compute inner product terms
            bra_terms = self._symbolic_determinant(bra_matrix)
            ket_terms = self._symbolic_determinant(ket_matrix)
            inner_terms = self._compute_inner_product_terms(bra_terms, ket_terms, orthogonal=False)

            # Build result string using combined braket notation ⟨a,b,c|d,e,f⟩
            parts = []
            first = True
            for sign, bra_elems, ket_elems in inner_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False

                # Combined braket: ⟨bra_elements|ket_elements⟩
                bra_str = ", ".join(str(b) for b in bra_elems)
                ket_str = ", ".join(str(k) for k in ket_elems)
                parts.append(f"\\langle {bra_str} | {ket_str} \\rangle")

            return "".join(parts) if parts else "0"

        # Case 2: Simple bra @ simple ket
        elif bra_label is not None and ket_label is not None:
            bra_str = bra_label
            ket_str = ket_label
            if isinstance(bra_str, (list, tuple)):
                bra_str = ", ".join(str(c) for c in bra_str)
            if isinstance(ket_str, (list, tuple)):
                ket_str = ", ".join(str(c) for c in ket_str)
            return f"\\langle {bra_str} | {ket_str} \\rangle"

        # Case 3: determinant_bra @ simple ket
        elif bra_matrix is not None and ket_label is not None:
            ket_str = ket_label
            if isinstance(ket_str, (list, tuple)):
                ket_str = ", ".join(str(c) for c in ket_str)

            # Render determinant bra terms contracted with ket
            bra_terms = self._symbolic_determinant(bra_matrix)
            parts = []
            first = True
            for sign, elements in bra_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False
                elem_str = ", ".join(str(e) for e in elements)
                parts.append(f"\\langle {elem_str} | {ket_str} \\rangle")

            return "".join(parts)

        # Case 4: simple bra @ determinant_ket
        elif bra_label is not None and ket_matrix is not None:
            bra_str = bra_label
            if isinstance(bra_str, (list, tuple)):
                bra_str = ", ".join(str(c) for c in bra_str)

            # Render bra contracted with determinant ket terms
            ket_terms = self._symbolic_determinant(ket_matrix)
            parts = []
            first = True
            for sign, elements in ket_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False
                elem_str = ", ".join(str(e) for e in elements)
                parts.append(f"\\langle {bra_str} | {elem_str} \\rangle")

            return "".join(parts)

        else:
            raise ValueError("Cannot compute inner product: incompatible operands")

    def _compute_side_matrix_element(self, bra_matrix, bra_det_type, bra_label,
                                      operator_str: str,
                                      ket_matrix, ket_det_type, ket_label) -> str:
        """
        Compute the matrix element ⟨bra|op|ket⟩ for one side of the equation.

        Handles combinations of:
        - determinant_bra @ op @ determinant_ket -> ⟨a,b,c|op|d,e,f⟩ terms
        - bra @ op @ ket -> simple ⟨bra|op|ket⟩
        - determinant_bra @ op @ ket -> determinant terms with operator
        - bra @ op @ determinant_ket -> bra with determinant ket terms
        """
        import numpy as np

        # Case 1: Both are determinants
        if bra_matrix is not None and ket_matrix is not None:
            if bra_det_type != 'bra':
                raise ValueError("Left side must be bra notation for matrix element")
            if ket_det_type != 'ket':
                raise ValueError("Right side must be ket notation for matrix element")

            # Check shape compatibility
            bra_shape = np.asarray(bra_matrix).shape
            ket_shape = np.asarray(ket_matrix).shape
            if bra_shape != ket_shape:
                raise ValueError(f"Shape mismatch: bra has shape {bra_shape}, ket has shape {ket_shape}")

            # Compute matrix element terms
            bra_terms = self._symbolic_determinant(bra_matrix)
            ket_terms = self._symbolic_determinant(ket_matrix)
            inner_terms = self._compute_inner_product_terms(bra_terms, ket_terms, orthogonal=False)

            # Build result string using matrix element notation ⟨a,b,c|op|d,e,f⟩
            parts = []
            first = True
            for sign, bra_elems, ket_elems in inner_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False

                # Matrix element: ⟨bra_elements|operator|ket_elements⟩
                bra_str = ", ".join(str(b) for b in bra_elems)
                ket_str = ", ".join(str(k) for k in ket_elems)
                parts.append(f"\\langle {bra_str} | {operator_str} | {ket_str} \\rangle")

            return "".join(parts) if parts else "0"

        # Case 2: Simple bra @ op @ simple ket
        elif bra_label is not None and ket_label is not None:
            bra_str = bra_label
            ket_str = ket_label
            if isinstance(bra_str, (list, tuple)):
                bra_str = ", ".join(str(c) for c in bra_str)
            if isinstance(ket_str, (list, tuple)):
                ket_str = ", ".join(str(c) for c in ket_str)
            return f"\\langle {bra_str} | {operator_str} | {ket_str} \\rangle"

        # Case 3: determinant_bra @ op @ simple ket
        elif bra_matrix is not None and ket_label is not None:
            ket_str = ket_label
            if isinstance(ket_str, (list, tuple)):
                ket_str = ", ".join(str(c) for c in ket_str)

            bra_terms = self._symbolic_determinant(bra_matrix)
            parts = []
            first = True
            for sign, elements in bra_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False
                elem_str = ", ".join(str(e) for e in elements)
                parts.append(f"\\langle {elem_str} | {operator_str} | {ket_str} \\rangle")

            return "".join(parts)

        # Case 4: simple bra @ op @ determinant_ket
        elif bra_label is not None and ket_matrix is not None:
            bra_str = bra_label
            if isinstance(bra_str, (list, tuple)):
                bra_str = ", ".join(str(c) for c in bra_str)

            ket_terms = self._symbolic_determinant(ket_matrix)
            parts = []
            first = True
            for sign, elements in ket_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False
                elem_str = ", ".join(str(e) for e in elements)
                parts.append(f"\\langle {bra_str} | {operator_str} | {elem_str} \\rangle")

            return "".join(parts)

        else:
            raise ValueError("Cannot compute matrix element: incompatible operands")

    def __rmatmul__(self, other):
        """Right matrix multiplication - not typically used but included for completeness."""
        raise TypeError("Right @ operation not supported. Use bra @ ket order.")

    def __add__(self, other: "Math") -> "MathBuilder":
        """
        Addition operator (+) for combining Math expressions.

        Concatenates two Math expressions with a plus sign between them.

        Example:
            m1 = Math().var('a')
            m2 = Math().var('b')
            m3 = m1 + m2  # Results in "a + b"
        """
        if not isinstance(other, Math):
            raise TypeError(f"unsupported operand type(s) for +: 'Math' and '{type(other).__name__}'")

        result = Math()
        result._parts = self._parts.copy()
        result._parts.append(" + ")
        result._parts.extend(other._parts)
        return result

    def __sub__(self, other: "Math") -> "MathBuilder":
        """
        Subtraction operator (-) for combining Math expressions.

        Concatenates two Math expressions with a minus sign between them.

        Example:
            m1 = Math().var('a')
            m2 = Math().var('b')
            m3 = m1 - m2  # Results in "a - b"
        """
        if not isinstance(other, Math):
            raise TypeError(f"unsupported operand type(s) for -: 'Math' and '{type(other).__name__}'")

        result = Math()
        result._parts = self._parts.copy()
        result._parts.append(" - ")
        result._parts.extend(other._parts)
        return result

    def __mul__(self, other: "Math") -> "MathBuilder":
        """
        Multiplication operator (*) for combining Math expressions.

        Concatenates two Math expressions with a cdot between them.

        Example:
            m1 = Math().var('a')
            m2 = Math().var('b')
            m3 = m1 * m2  # Results in "a \\cdot b"
        """
        if not isinstance(other, Math):
            raise TypeError(f"unsupported operand type(s) for *: 'Math' and '{type(other).__name__}'")

        result = Math()
        result._parts = self._parts.copy()
        result._parts.append(" \\cdot ")
        result._parts.extend(other._parts)
        return result


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
