"""
Chemical Machines Quantum Mechanics Module

Provides tools for working with Slater determinants, spin-orbitals,
and quantum mechanical matrix elements using spherical harmonic basis functions.

Example:
    from cm import qm

    # Create spin-orbitals with quantum numbers (spin, L, m)
    orbitals = qm.basis_sh([(1, 0, 0), (-1, 0, 0), (1, 1, 0)])

    # Create Slater determinant
    psi = qm.slater(orbitals)
    psi.render()

    # Inner products with automatic orthogonality
    phi = qm.slater(qm.basis_sh([(1, 0, 0), (-1, 0, 0), (1, 1, 1)]))
    overlap = psi @ phi
    overlap.render()  # Renders 0 (orthogonal states)

    # Matrix elements with Hamiltonian
    H = qm.hamiltonian()
    matrix_elem = psi @ H @ phi
    matrix_elem.render()
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from . import views


@dataclass
class SpinOrbital:
    """
    A spin-orbital defined by quantum numbers.

    Attributes:
        spin: +1 for spin-up (α), -1 for spin-down (β)
        L: Angular momentum quantum number (0=s, 1=p, 2=d, 3=f, ...)
        m: Magnetic quantum number (-L <= m <= L)
        n: Optional principal quantum number
    """
    spin: int  # +1 or -1
    L: int     # Angular momentum
    m: int     # Magnetic quantum number
    n: Optional[int] = None  # Principal quantum number (optional)

    def __post_init__(self):
        if self.spin not in (1, -1):
            raise ValueError(f"spin must be +1 or -1, got {self.spin}")
        if self.L < 0:
            raise ValueError(f"L must be non-negative, got {self.L}")
        if abs(self.m) > self.L:
            raise ValueError(f"|m| must be <= L, got m={self.m}, L={self.L}")

    @property
    def spin_label(self) -> str:
        """Return spin as α or β."""
        return "α" if self.spin == 1 else "β"

    @property
    def spin_arrow(self) -> str:
        """Return spin as ↑ or ↓."""
        return "↑" if self.spin == 1 else "↓"

    @property
    def L_label(self) -> str:
        """Return L as spectroscopic notation (s, p, d, f, ...)."""
        labels = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k']
        if self.L < len(labels):
            return labels[self.L]
        return f"L{self.L}"

    @property
    def label(self) -> str:
        """Human-readable label like '2p↑(m=1)'."""
        n_str = str(self.n) if self.n is not None else ""
        return f"{n_str}{self.L_label}{self.spin_arrow}(m={self.m})"

    @property
    def short_label(self) -> str:
        """Short label like '2p↑' (without m)."""
        n_str = str(self.n) if self.n is not None else ""
        return f"{n_str}{self.L_label}{self.spin_arrow}"

    @property
    def latex_label(self) -> str:
        """LaTeX formatted label."""
        spin_tex = r"\uparrow" if self.spin == 1 else r"\downarrow"
        n_str = str(self.n) if self.n is not None else ""
        return f"{n_str}{self.L_label}_{{{spin_tex}, m={self.m}}}"

    @property
    def ket_label(self) -> str:
        """Label for use in ket notation: |L, m, σ⟩."""
        spin_tex = r"\uparrow" if self.spin == 1 else r"\downarrow"
        if self.n is not None:
            return f"{self.n}, {self.L}, {self.m}, {spin_tex}"
        return f"{self.L}, {self.m}, {spin_tex}"

    def is_orthogonal_to(self, other: "SpinOrbital") -> bool:
        """
        Check if this orbital is orthogonal to another.

        Orbitals are orthogonal if any quantum number differs:
        - Different spin
        - Different L
        - Different m
        - Different n (if both specified)
        """
        if self.spin != other.spin:
            return True
        if self.L != other.L:
            return True
        if self.m != other.m:
            return True
        if self.n is not None and other.n is not None and self.n != other.n:
            return True
        return False

    def __eq__(self, other):
        if not isinstance(other, SpinOrbital):
            return False
        return (self.spin == other.spin and
                self.L == other.L and
                self.m == other.m and
                self.n == other.n)

    def __hash__(self):
        return hash((self.spin, self.L, self.m, self.n))

    def __repr__(self):
        return f"SpinOrbital(spin={self.spin}, L={self.L}, m={self.m}, n={self.n})"


def basis_sh_element(spin: int, L: int, m: int, n: Optional[int] = None) -> SpinOrbital:
    """
    Create a single spin-orbital basis element with spherical harmonic quantum numbers.

    Args:
        spin: +1 for spin-up (α), -1 for spin-down (β)
        L: Angular momentum quantum number (0=s, 1=p, 2=d, 3=f, ...)
        m: Magnetic quantum number (-L <= m <= L)
        n: Optional principal quantum number

    Returns:
        SpinOrbital object

    Example:
        # 1s spin-up orbital
        orbital = basis_sh_element(1, L=0, m=0, n=1)

        # 2p spin-down with m=-1
        orbital = basis_sh_element(-1, L=1, m=-1, n=2)
    """
    return SpinOrbital(spin=spin, L=L, m=m, n=n)


def basis_sh(quantum_numbers: List[Tuple]) -> List[SpinOrbital]:
    """
    Create a list of spin-orbital basis elements from quantum number tuples.

    Args:
        quantum_numbers: List of tuples (spin, L, m) or (spin, L, m, n)
            - spin: +1 for spin-up, -1 for spin-down
            - L: Angular momentum quantum number
            - m: Magnetic quantum number
            - n: Optional principal quantum number

    Returns:
        List of SpinOrbital objects

    Example:
        # Helium ground state: 1s↑ 1s↓
        orbitals = basis_sh([(1, 0, 0), (-1, 0, 0)])

        # With principal quantum numbers
        orbitals = basis_sh([(1, 0, 0, 1), (-1, 0, 0, 1), (1, 1, 0, 2)])
    """
    result = []
    for qn in quantum_numbers:
        if len(qn) == 3:
            spin, L, m = qn
            result.append(SpinOrbital(spin=spin, L=L, m=m))
        elif len(qn) == 4:
            spin, L, m, n = qn
            result.append(SpinOrbital(spin=spin, L=L, m=m, n=n))
        else:
            raise ValueError(f"Expected tuple of 3 or 4 elements (spin, L, m[, n]), got {len(qn)}")
    return result


class SlaterDeterminant:
    """
    Represents a Slater determinant as an antisymmetrized product of spin-orbitals.

    The Slater determinant automatically handles:
    - Orthogonality between spin-orbitals
    - Antisymmetry under particle exchange
    - Inner products using Slater-Condon rules

    Example:
        orbitals = basis_sh([(1, 0, 0), (-1, 0, 0)])
        psi = SlaterDeterminant(orbitals)
        psi.render()
    """

    def __init__(self, orbitals: List[SpinOrbital]):
        """
        Create a Slater determinant from a list of spin-orbitals.

        Args:
            orbitals: List of SpinOrbital objects (one per electron)
        """
        self.orbitals = list(orbitals)
        self.n_electrons = len(orbitals)

        # Check for duplicate orbitals (Pauli exclusion)
        orbital_set = set(orbitals)
        if len(orbital_set) != len(orbitals):
            raise ValueError("Duplicate orbitals detected - violates Pauli exclusion principle")

    @property
    def orbital_set(self) -> set:
        """Set of orbitals (for comparison)."""
        return set(self.orbitals)

    def overlap(self, other: "SlaterDeterminant") -> Tuple[bool, Optional[int]]:
        """
        Compute overlap ⟨self|other⟩ using orthonormality.

        Returns:
            (is_nonzero, sign) where:
            - is_nonzero: True if overlap is non-zero
            - sign: +1 or -1 based on permutation parity (None if zero)
        """
        if self.n_electrons != other.n_electrons:
            return (False, None)

        if self.orbital_set != other.orbital_set:
            return (False, None)

        # Same orbitals - compute permutation parity
        parity = self._permutation_parity(other)
        return (True, parity)

    def _permutation_parity(self, other: "SlaterDeterminant") -> int:
        """Compute parity of permutation between self and other."""
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

    def n_excitations(self, other: "SlaterDeterminant") -> int:
        """Count number of orbital differences between two determinants."""
        return len(self.orbital_set - other.orbital_set)

    def excitation_orbitals(self, other: "SlaterDeterminant") -> Tuple[List[SpinOrbital], List[SpinOrbital]]:
        """
        Get orbitals that differ between two determinants.

        Returns:
            (in_self, in_other): Orbitals only in self, orbitals only in other
        """
        only_self = list(self.orbital_set - other.orbital_set)
        only_other = list(other.orbital_set - self.orbital_set)
        return (only_self, only_other)

    def ket_labels(self) -> List[str]:
        """Get LaTeX labels for ket notation."""
        return [orb.ket_label for orb in self.orbitals]

    def render(self, normalize: bool = False):
        """
        Render the Slater determinant as LaTeX.

        Args:
            normalize: Include 1/√n! normalization factor
        """
        labels = ", ".join(self.ket_labels())

        if normalize:
            n = self.n_electrons
            latex = f"\\frac{{1}}{{\\sqrt{{{n}!}}}} | {labels} \\rangle"
        else:
            latex = f"| {labels} \\rangle"

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def __matmul__(self, other: Union["SlaterDeterminant", "Operator"]) -> "MatrixElement":
        """
        Compute inner product or begin matrix element.

        psi @ phi -> ⟨psi|phi⟩ (overlap)
        psi @ H -> partial matrix element (needs another @)
        """
        if isinstance(other, SlaterDeterminant):
            # Direct inner product
            return Overlap(self, other)
        elif isinstance(other, Operator):
            # Partial matrix element - store bra and operator
            return PartialMatrixElement(self, other)
        else:
            raise TypeError(f"Cannot use @ with SlaterDeterminant and {type(other)}")

    def __repr__(self):
        labels = [orb.short_label for orb in self.orbitals]
        return f"SlaterDeterminant([{', '.join(labels)}])"


class Operator:
    """
    Represents a quantum mechanical operator.

    Supports one-electron and two-electron operators for
    computing matrix elements with Slater determinants.
    """

    def __init__(self, symbol: str = "H", operator_type: str = "hamiltonian"):
        """
        Create an operator.

        Args:
            symbol: LaTeX symbol for the operator (default "H")
            operator_type: "hamiltonian", "one_electron", or "two_electron"
        """
        self.symbol = symbol
        self.operator_type = operator_type

    @property
    def latex(self) -> str:
        """LaTeX representation with hat."""
        return f"\\hat{{{self.symbol}}}"

    def __repr__(self):
        return f"Operator({self.symbol})"


class PartialMatrixElement:
    """
    Intermediate result of bra @ operator, waiting for ket.
    """

    def __init__(self, bra: SlaterDeterminant, operator: Operator):
        self.bra = bra
        self.operator = operator

    def __matmul__(self, ket: SlaterDeterminant) -> "MatrixElement":
        """Complete the matrix element with a ket."""
        if not isinstance(ket, SlaterDeterminant):
            raise TypeError(f"Expected SlaterDeterminant, got {type(ket)}")
        return MatrixElement(self.bra, self.operator, ket)


class Overlap:
    """
    Represents an overlap integral ⟨bra|ket⟩.
    """

    def __init__(self, bra: SlaterDeterminant, ket: SlaterDeterminant):
        self.bra = bra
        self.ket = ket

        # Compute overlap
        self.is_nonzero, self.sign = bra.overlap(ket)

    @property
    def value(self) -> Union[int, str]:
        """Numeric value if computable, else symbolic."""
        if not self.is_nonzero:
            return 0
        return self.sign

    def render(self, simplify: bool = True):
        """
        Render the overlap.

        Args:
            simplify: If True, show computed value. If False, show full notation.
        """
        if simplify:
            html = f'<div class="cm-math cm-math-center">\\[ {self.value} \\]</div>'
        else:
            bra_labels = ", ".join(self.bra.ket_labels())
            ket_labels = ", ".join(self.ket.ket_labels())
            latex = f"\\langle {bra_labels} | {ket_labels} \\rangle"
            html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'

        views.html(html)

    def __repr__(self):
        return f"Overlap(value={self.value})"


class MatrixElement:
    """
    Represents a matrix element ⟨bra|operator|ket⟩.

    Uses Slater-Condon rules to determine which terms survive.
    """

    def __init__(self, bra: SlaterDeterminant, operator: Operator, ket: SlaterDeterminant):
        self.bra = bra
        self.operator = operator
        self.ket = ket

        # Analyze excitation level
        self.n_excitations = bra.n_excitations(ket)
        self.excitations = bra.excitation_orbitals(ket)

    @property
    def is_zero(self) -> bool:
        """Matrix element is zero if more than 2 excitations."""
        return self.n_excitations > 2

    def render(self, apply_slater_condon: bool = True):
        """
        Render the matrix element.

        Args:
            apply_slater_condon: If True, apply Slater-Condon rules to simplify.
        """
        if apply_slater_condon and self.is_zero:
            html = '<div class="cm-math cm-math-center">\\[ 0 \\]</div>'
            views.html(html)
            return

        if apply_slater_condon and self.n_excitations == 0:
            # Diagonal element - sum over occupied orbitals
            latex = self._render_diagonal()
        elif apply_slater_condon and self.n_excitations == 1:
            # Single excitation
            latex = self._render_single_excitation()
        elif apply_slater_condon and self.n_excitations == 2:
            # Double excitation
            latex = self._render_double_excitation()
        else:
            # Full notation
            bra_labels = ", ".join(self.bra.ket_labels())
            ket_labels = ", ".join(self.ket.ket_labels())
            latex = f"\\langle {bra_labels} | {self.operator.latex} | {ket_labels} \\rangle"

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def _render_diagonal(self) -> str:
        """Render diagonal matrix element (same determinant)."""
        if self.operator.operator_type == "one_electron":
            return "\\sum_i \\langle i | \\hat{h} | i \\rangle"
        else:
            # Full Hamiltonian
            return ("\\sum_i \\langle i | \\hat{h} | i \\rangle + "
                    "\\frac{1}{2} \\sum_{i \\neq j} \\left[ "
                    "\\langle ij | \\hat{g} | ij \\rangle - "
                    "\\langle ij | \\hat{g} | ji \\rangle \\right]")

    def _render_single_excitation(self) -> str:
        """Render single excitation matrix element."""
        only_bra, only_ket = self.excitations
        p = only_bra[0].ket_label
        q = only_ket[0].ket_label

        if self.operator.operator_type == "one_electron":
            return f"\\langle {p} | \\hat{{h}} | {q} \\rangle"
        else:
            return (f"\\langle {p} | \\hat{{h}} | {q} \\rangle + "
                    f"\\sum_j \\left[ \\langle {p} j | \\hat{{g}} | {q} j \\rangle - "
                    f"\\langle {p} j | \\hat{{g}} | j {q} \\rangle \\right]")

    def _render_double_excitation(self) -> str:
        """Render double excitation matrix element."""
        only_bra, only_ket = self.excitations
        p, q = only_bra[0].ket_label, only_bra[1].ket_label
        r, s = only_ket[0].ket_label, only_ket[1].ket_label

        if self.operator.operator_type == "one_electron":
            return "0"
        else:
            return (f"\\langle {p} {q} | \\hat{{g}} | {r} {s} \\rangle - "
                    f"\\langle {p} {q} | \\hat{{g}} | {s} {r} \\rangle")

    def __repr__(self):
        return f"MatrixElement(n_exc={self.n_excitations}, zero={self.is_zero})"


# Convenience functions

def slater(orbitals: List[SpinOrbital]) -> SlaterDeterminant:
    """
    Create a Slater determinant from a list of spin-orbitals.

    Args:
        orbitals: List of SpinOrbital objects

    Returns:
        SlaterDeterminant object

    Example:
        orbitals = basis_sh([(1, 0, 0), (-1, 0, 0)])
        psi = slater(orbitals)
        psi.render()
    """
    return SlaterDeterminant(orbitals)


def hamiltonian(symbol: str = "H") -> Operator:
    """
    Create a Hamiltonian operator.

    Args:
        symbol: LaTeX symbol (default "H")

    Returns:
        Operator object for use in matrix elements

    Example:
        H = hamiltonian()
        matrix_elem = psi @ H @ phi
    """
    return Operator(symbol=symbol, operator_type="hamiltonian")


def one_electron_operator(symbol: str = "h") -> Operator:
    """
    Create a one-electron operator.

    Args:
        symbol: LaTeX symbol (default "h")

    Returns:
        Operator object
    """
    return Operator(symbol=symbol, operator_type="one_electron")


def two_electron_operator(symbol: str = "g") -> Operator:
    """
    Create a two-electron operator.

    Args:
        symbol: LaTeX symbol (default "g")

    Returns:
        Operator object
    """
    return Operator(symbol=symbol, operator_type="two_electron")


# Common Hamiltonian alias
H = hamiltonian()


# =============================================================================
# RELATIVISTIC QUANTUM MECHANICS
# =============================================================================

@dataclass
class DiracSpinor:
    """
    A four-component Dirac spinor defined by relativistic quantum numbers.

    In relativistic quantum mechanics, spin-orbit coupling is inherent, so we use
    total angular momentum j and its projection mⱼ instead of separate L and spin.

    The κ quantum number encodes both orbital (l) and total (j) angular momentum:
        κ = -(j + 1/2) for j = l + 1/2  (spin-orbit aligned)
        κ = +(j + 1/2) for j = l - 1/2  (spin-orbit anti-aligned)

    Attributes:
        n: Principal quantum number (1, 2, 3, ...)
        kappa: Relativistic angular momentum quantum number
        mj: Projection of total angular momentum (-j ≤ mⱼ ≤ j, half-integer)

    The relationship between κ and l, j:
        l = |κ| - 1 if κ > 0, else |κ|
        j = |κ| - 1/2

    Examples:
        1s₁/₂: n=1, κ=-1, mⱼ=±1/2
        2s₁/₂: n=2, κ=-1, mⱼ=±1/2
        2p₁/₂: n=2, κ=+1, mⱼ=±1/2
        2p₃/₂: n=2, κ=-2, mⱼ=±1/2, ±3/2
    """
    n: int           # Principal quantum number
    kappa: int       # Relativistic κ quantum number (non-zero integer)
    mj: float        # Projection of j (half-integer: ±1/2, ±3/2, ...)

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.kappa == 0:
            raise ValueError("κ cannot be zero")
        j = self.j
        if abs(self.mj) > j:
            raise ValueError(f"|mⱼ| must be <= j={j}, got mⱼ={self.mj}")
        # mj must be half-integer
        if (2 * self.mj) != int(2 * self.mj):
            raise ValueError(f"mⱼ must be half-integer, got {self.mj}")

    @property
    def j(self) -> float:
        """Total angular momentum quantum number (half-integer)."""
        return abs(self.kappa) - 0.5

    @property
    def l(self) -> int:
        """Orbital angular momentum quantum number."""
        if self.kappa > 0:
            return self.kappa
        else:
            return -self.kappa - 1

    @property
    def l_small(self) -> int:
        """Orbital angular momentum for small component (l̃ = 2j - l)."""
        return int(2 * self.j - self.l)

    @property
    def is_spin_orbit_aligned(self) -> bool:
        """True if j = l + 1/2 (κ < 0)."""
        return self.kappa < 0

    @property
    def L_label(self) -> str:
        """Return l as spectroscopic notation (s, p, d, f, ...)."""
        labels = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k']
        if self.l < len(labels):
            return labels[self.l]
        return f"l{self.l}"

    @property
    def j_label(self) -> str:
        """Return j as fraction string (1/2, 3/2, 5/2, ...)."""
        j2 = int(2 * self.j)
        return f"{j2}/2"

    @property
    def mj_label(self) -> str:
        """Return mⱼ as fraction string."""
        mj2 = int(2 * self.mj)
        sign = "+" if mj2 >= 0 else ""
        return f"{sign}{mj2}/2"

    @property
    def label(self) -> str:
        """Human-readable label like '2p₃/₂(mⱼ=+1/2)'."""
        return f"{self.n}{self.L_label}_{self.j_label}(mⱼ={self.mj_label})"

    @property
    def short_label(self) -> str:
        """Short label like '2p₃/₂'."""
        return f"{self.n}{self.L_label}_{self.j_label}"

    @property
    def latex_label(self) -> str:
        """LaTeX formatted label."""
        j_frac = f"\\frac{{{int(2*self.j)}}}{{2}}"
        return f"{self.n}{self.L_label}_{{{j_frac}}}"

    @property
    def ket_label(self) -> str:
        """Label for use in ket notation: |n, κ, mⱼ⟩."""
        mj2 = int(2 * self.mj)
        sign = "+" if mj2 >= 0 else ""
        return f"{self.n}, {self.kappa}, {sign}{mj2}/2"

    @property
    def spectroscopic_label(self) -> str:
        """Standard spectroscopic notation like '2p₃/₂'."""
        j2 = int(2 * self.j)
        return f"{self.n}{self.L_label}_{{{j2}/2}}"

    def is_orthogonal_to(self, other: "DiracSpinor") -> bool:
        """
        Check if this spinor is orthogonal to another.

        Spinors are orthogonal if any quantum number differs.
        """
        if self.n != other.n:
            return True
        if self.kappa != other.kappa:
            return True
        if self.mj != other.mj:
            return True
        return False

    def __eq__(self, other):
        if not isinstance(other, DiracSpinor):
            return False
        return (self.n == other.n and
                self.kappa == other.kappa and
                self.mj == other.mj)

    def __hash__(self):
        return hash((self.n, self.kappa, self.mj))

    def __repr__(self):
        return f"DiracSpinor(n={self.n}, κ={self.kappa}, mⱼ={self.mj})"


def kappa_from_lj(l: int, j: float) -> int:
    """
    Compute κ from orbital angular momentum l and total angular momentum j.

    Args:
        l: Orbital angular momentum (0, 1, 2, ...)
        j: Total angular momentum (l ± 1/2)

    Returns:
        κ quantum number

    Example:
        kappa_from_lj(0, 0.5)  # 1s₁/₂ -> κ = -1
        kappa_from_lj(1, 0.5)  # 2p₁/₂ -> κ = +1
        kappa_from_lj(1, 1.5)  # 2p₃/₂ -> κ = -2
    """
    if abs(j - l - 0.5) < 0.01:  # j = l + 1/2
        return -(l + 1)
    elif abs(j - l + 0.5) < 0.01:  # j = l - 1/2
        return l
    else:
        raise ValueError(f"j must be l ± 1/2, got l={l}, j={j}")


def dirac_spinor(n: int, kappa: int, mj: float) -> DiracSpinor:
    """
    Create a Dirac spinor with relativistic quantum numbers.

    Args:
        n: Principal quantum number (1, 2, 3, ...)
        kappa: Relativistic κ quantum number
            κ = -(l+1) for j = l + 1/2
            κ = +l     for j = l - 1/2
        mj: Projection of total angular momentum (half-integer)

    Returns:
        DiracSpinor object

    Example:
        # 1s₁/₂ with mⱼ = +1/2
        psi = dirac_spinor(n=1, kappa=-1, mj=0.5)

        # 2p₃/₂ with mⱼ = -3/2
        psi = dirac_spinor(n=2, kappa=-2, mj=-1.5)
    """
    return DiracSpinor(n=n, kappa=kappa, mj=mj)


def dirac_spinor_lj(n: int, l: int, j: float, mj: float) -> DiracSpinor:
    """
    Create a Dirac spinor using (n, l, j, mⱼ) notation.

    This is often more intuitive than using κ directly.

    Args:
        n: Principal quantum number
        l: Orbital angular momentum (0=s, 1=p, 2=d, ...)
        j: Total angular momentum (l ± 1/2)
        mj: Projection of j (half-integer)

    Returns:
        DiracSpinor object

    Example:
        # 2p₃/₂ with mⱼ = +1/2
        psi = dirac_spinor_lj(n=2, l=1, j=1.5, mj=0.5)
    """
    kappa = kappa_from_lj(l, j)
    return DiracSpinor(n=n, kappa=kappa, mj=mj)


def basis_dirac(quantum_numbers: List[Tuple]) -> List[DiracSpinor]:
    """
    Create a list of Dirac spinor basis elements.

    Args:
        quantum_numbers: List of tuples in one of two formats:
            (n, κ, mⱼ) - using κ directly
            (n, l, j, mⱼ) - using l and j

    Returns:
        List of DiracSpinor objects

    Example:
        # Helium ground state (relativistic)
        spinors = basis_dirac([
            (1, -1, 0.5),   # 1s₁/₂ mⱼ=+1/2
            (1, -1, -0.5),  # 1s₁/₂ mⱼ=-1/2
        ])

        # Using (n, l, j, mⱼ) format
        spinors = basis_dirac([
            (1, 0, 0.5, 0.5),   # 1s₁/₂ mⱼ=+1/2
            (1, 0, 0.5, -0.5),  # 1s₁/₂ mⱼ=-1/2
        ])
    """
    result = []
    for qn in quantum_numbers:
        if len(qn) == 3:
            n, kappa, mj = qn
            result.append(DiracSpinor(n=n, kappa=kappa, mj=mj))
        elif len(qn) == 4:
            n, l, j, mj = qn
            kappa = kappa_from_lj(l, j)
            result.append(DiracSpinor(n=n, kappa=kappa, mj=mj))
        else:
            raise ValueError(f"Expected tuple of 3 (n, κ, mⱼ) or 4 (n, l, j, mⱼ) elements, got {len(qn)}")
    return result


class DiracDeterminant:
    """
    Represents a Slater determinant of four-component Dirac spinors.

    Used for relativistic many-electron calculations where spin-orbit
    coupling is treated from the start.

    Example:
        spinors = basis_dirac([(1, -1, 0.5), (1, -1, -0.5)])
        psi = DiracDeterminant(spinors)
        psi.render()
    """

    def __init__(self, spinors: List[DiracSpinor]):
        """
        Create a Dirac determinant from a list of four-component spinors.

        Args:
            spinors: List of DiracSpinor objects (one per electron)
        """
        self.spinors = list(spinors)
        self.n_electrons = len(spinors)

        # Check for duplicate spinors (Pauli exclusion)
        spinor_set = set((s.n, s.kappa, s.mj) for s in spinors)
        if len(spinor_set) != len(spinors):
            raise ValueError("Duplicate spinors detected - violates Pauli exclusion principle")

    @property
    def spinor_set(self) -> set:
        """Set of spinors (for comparison)."""
        return set(self.spinors)

    @property
    def total_mj(self) -> float:
        """Total Mⱼ projection (sum of individual mⱼ values)."""
        return sum(s.mj for s in self.spinors)

    def overlap(self, other: "DiracDeterminant") -> Tuple[bool, Optional[int]]:
        """
        Compute overlap ⟨self|other⟩ using orthonormality.

        Returns:
            (is_nonzero, sign) where:
            - is_nonzero: True if overlap is non-zero
            - sign: +1 or -1 based on permutation parity (None if zero)
        """
        if self.n_electrons != other.n_electrons:
            return (False, None)

        if self.spinor_set != other.spinor_set:
            return (False, None)

        # Same spinors - compute permutation parity
        parity = self._permutation_parity(other)
        return (True, parity)

    def _permutation_parity(self, other: "DiracDeterminant") -> int:
        """Compute parity of permutation between self and other."""
        other_indices = {s: i for i, s in enumerate(other.spinors)}
        perm = [other_indices[s] for s in self.spinors]

        # Count inversions
        inversions = 0
        n = len(perm)
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inversions += 1

        return 1 if inversions % 2 == 0 else -1

    def n_excitations(self, other: "DiracDeterminant") -> int:
        """Count number of spinor differences between two determinants."""
        return len(self.spinor_set - other.spinor_set)

    def excitation_spinors(self, other: "DiracDeterminant") -> Tuple[List[DiracSpinor], List[DiracSpinor]]:
        """
        Get spinors that differ between two determinants.

        Returns:
            (in_self, in_other): Spinors only in self, spinors only in other
        """
        only_self = list(self.spinor_set - other.spinor_set)
        only_other = list(other.spinor_set - self.spinor_set)
        return (only_self, only_other)

    def ket_labels(self) -> List[str]:
        """Get LaTeX labels for ket notation."""
        return [s.ket_label for s in self.spinors]

    def render(self, normalize: bool = False, notation: str = "kappa"):
        """
        Render the Dirac determinant as LaTeX.

        Args:
            normalize: Include 1/√n! normalization factor
            notation: "kappa" for |n,κ,mⱼ⟩ or "spectroscopic" for |2p₃/₂, mⱼ⟩
        """
        if notation == "spectroscopic":
            labels = ", ".join(f"{s.spectroscopic_label}({s.mj_label})" for s in self.spinors)
        else:
            labels = ", ".join(self.ket_labels())

        if normalize:
            n = self.n_electrons
            latex = f"\\frac{{1}}{{\\sqrt{{{n}!}}}} | {labels} \\rangle"
        else:
            latex = f"| {labels} \\rangle"

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def __matmul__(self, other: Union["DiracDeterminant", "RelativisticOperator"]) -> "MatrixElement":
        """
        Compute inner product or begin matrix element.

        psi @ phi -> ⟨psi|phi⟩ (overlap)
        psi @ H_D -> partial matrix element (needs another @)
        """
        if isinstance(other, DiracDeterminant):
            return DiracOverlap(self, other)
        elif isinstance(other, RelativisticOperator):
            return PartialDiracMatrixElement(self, other)
        else:
            raise TypeError(f"Cannot use @ with DiracDeterminant and {type(other)}")

    def __repr__(self):
        labels = [s.short_label for s in self.spinors]
        return f"DiracDeterminant([{', '.join(labels)}])"


class RelativisticOperator:
    """
    Represents a relativistic quantum mechanical operator.

    Supports the Dirac Hamiltonian and various two-electron interactions:
    - Coulomb: Standard 1/r₁₂ interaction
    - Gaunt: Magnetic interaction (α₁·α₂)/r₁₂
    - Breit: Full retardation correction
    - Coulomb-Breit: Coulomb + Breit (most accurate)
    """

    def __init__(self, symbol: str = "H_D", operator_type: str = "dirac_coulomb"):
        """
        Create a relativistic operator.

        Args:
            symbol: LaTeX symbol for the operator
            operator_type: One of:
                - "dirac": One-electron Dirac operator (cα·p + βmc² + V)
                - "dirac_coulomb": Dirac + Coulomb two-electron
                - "dirac_coulomb_gaunt": Dirac + Coulomb + Gaunt
                - "dirac_coulomb_breit": Dirac + Coulomb + Breit (full)
        """
        self.symbol = symbol
        self.operator_type = operator_type

    @property
    def latex(self) -> str:
        """LaTeX representation."""
        return f"\\hat{{{self.symbol}}}"

    @property
    def includes_two_electron(self) -> bool:
        """Whether this operator includes two-electron terms."""
        return self.operator_type != "dirac"

    @property
    def two_electron_type(self) -> Optional[str]:
        """Type of two-electron interaction."""
        if "breit" in self.operator_type:
            return "coulomb_breit"
        elif "gaunt" in self.operator_type:
            return "coulomb_gaunt"
        elif "coulomb" in self.operator_type:
            return "coulomb"
        return None

    def __repr__(self):
        return f"RelativisticOperator({self.symbol}, {self.operator_type})"


class PartialDiracMatrixElement:
    """
    Intermediate result of bra @ relativistic_operator, waiting for ket.
    """

    def __init__(self, bra: DiracDeterminant, operator: RelativisticOperator):
        self.bra = bra
        self.operator = operator

    def __matmul__(self, ket: DiracDeterminant) -> "DiracMatrixElement":
        """Complete the matrix element with a ket."""
        if not isinstance(ket, DiracDeterminant):
            raise TypeError(f"Expected DiracDeterminant, got {type(ket)}")
        return DiracMatrixElement(self.bra, self.operator, ket)


class DiracOverlap:
    """
    Represents an overlap integral ⟨bra|ket⟩ for Dirac determinants.
    """

    def __init__(self, bra: DiracDeterminant, ket: DiracDeterminant):
        self.bra = bra
        self.ket = ket
        self.is_nonzero, self.sign = bra.overlap(ket)

    @property
    def value(self) -> Union[int, str]:
        """Numeric value if computable, else symbolic."""
        if not self.is_nonzero:
            return 0
        return self.sign

    def render(self, simplify: bool = True):
        """Render the overlap."""
        if simplify:
            html = f'<div class="cm-math cm-math-center">\\[ {self.value} \\]</div>'
        else:
            bra_labels = ", ".join(self.bra.ket_labels())
            ket_labels = ", ".join(self.ket.ket_labels())
            latex = f"\\langle {bra_labels} | {ket_labels} \\rangle"
            html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def __repr__(self):
        return f"DiracOverlap(value={self.value})"


class DiracMatrixElement:
    """
    Represents a matrix element ⟨bra|operator|ket⟩ for Dirac determinants.

    Uses relativistic Slater-Condon rules. The key difference from
    non-relativistic case is the form of the two-electron integrals.
    """

    def __init__(self, bra: DiracDeterminant, operator: RelativisticOperator,
                 ket: DiracDeterminant):
        self.bra = bra
        self.operator = operator
        self.ket = ket

        self.n_excitations = bra.n_excitations(ket)
        self.excitations = bra.excitation_spinors(ket)

    @property
    def is_zero(self) -> bool:
        """Matrix element is zero if more than 2 excitations."""
        return self.n_excitations > 2

    def render(self, apply_slater_condon: bool = True, show_components: bool = False):
        """
        Render the matrix element.

        Args:
            apply_slater_condon: If True, apply relativistic Slater-Condon rules
            show_components: If True, show large/small component structure
        """
        if apply_slater_condon and self.is_zero:
            html = '<div class="cm-math cm-math-center">\\[ 0 \\]</div>'
            views.html(html)
            return

        if apply_slater_condon and self.n_excitations == 0:
            latex = self._render_diagonal(show_components)
        elif apply_slater_condon and self.n_excitations == 1:
            latex = self._render_single_excitation(show_components)
        elif apply_slater_condon and self.n_excitations == 2:
            latex = self._render_double_excitation(show_components)
        else:
            bra_labels = ", ".join(self.bra.ket_labels())
            ket_labels = ", ".join(self.ket.ket_labels())
            latex = f"\\langle {bra_labels} | {self.operator.latex} | {ket_labels} \\rangle"

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def _render_diagonal(self, show_components: bool) -> str:
        """Render diagonal matrix element."""
        if not self.operator.includes_two_electron:
            # One-electron Dirac only
            return "\\sum_i \\langle i | \\hat{h}_D | i \\rangle"

        two_e = self._two_electron_symbol()
        if show_components:
            # Show large-large and small-small contributions
            return (
                "\\sum_i \\langle i | \\hat{h}_D | i \\rangle + "
                f"\\frac{{1}}{{2}} \\sum_{{i \\neq j}} \\left[ "
                f"\\langle ij | {two_e} | ij \\rangle - "
                f"\\langle ij | {two_e} | ji \\rangle \\right]"
            )
        else:
            return (
                "\\sum_i \\langle i | \\hat{h}_D | i \\rangle + "
                f"\\frac{{1}}{{2}} \\sum_{{i \\neq j}} \\left[ "
                f"\\langle ij | {two_e} | ij \\rangle - "
                f"\\langle ij | {two_e} | ji \\rangle \\right]"
            )

    def _render_single_excitation(self, show_components: bool) -> str:
        """Render single excitation matrix element."""
        only_bra, only_ket = self.excitations
        p = only_bra[0].ket_label
        q = only_ket[0].ket_label

        if not self.operator.includes_two_electron:
            return f"\\langle {p} | \\hat{{h}}_D | {q} \\rangle"

        two_e = self._two_electron_symbol()
        return (
            f"\\langle {p} | \\hat{{h}}_D | {q} \\rangle + "
            f"\\sum_j \\left[ \\langle {p} j | {two_e} | {q} j \\rangle - "
            f"\\langle {p} j | {two_e} | j {q} \\rangle \\right]"
        )

    def _render_double_excitation(self, show_components: bool) -> str:
        """Render double excitation matrix element."""
        only_bra, only_ket = self.excitations
        p, q = only_bra[0].ket_label, only_bra[1].ket_label
        r, s = only_ket[0].ket_label, only_ket[1].ket_label

        if not self.operator.includes_two_electron:
            return "0"

        two_e = self._two_electron_symbol()
        return (
            f"\\langle {p} {q} | {two_e} | {r} {s} \\rangle - "
            f"\\langle {p} {q} | {two_e} | {s} {r} \\rangle"
        )

    def _two_electron_symbol(self) -> str:
        """Get LaTeX symbol for two-electron operator."""
        two_e_type = self.operator.two_electron_type
        if two_e_type == "coulomb_breit":
            return "\\hat{g}_{CB}"
        elif two_e_type == "coulomb_gaunt":
            return "\\hat{g}_{CG}"
        else:
            return "\\hat{g}_C"

    def __repr__(self):
        return f"DiracMatrixElement(n_exc={self.n_excitations}, zero={self.is_zero})"


# Relativistic convenience functions

def dirac_slater(spinors: List[DiracSpinor]) -> DiracDeterminant:
    """
    Create a Dirac determinant from a list of four-component spinors.

    Args:
        spinors: List of DiracSpinor objects

    Returns:
        DiracDeterminant object

    Example:
        spinors = basis_dirac([(1, -1, 0.5), (1, -1, -0.5)])
        psi = dirac_slater(spinors)
        psi.render()
    """
    return DiracDeterminant(spinors)


def dirac_hamiltonian(two_electron: str = "coulomb") -> RelativisticOperator:
    """
    Create a Dirac-Coulomb or Dirac-Coulomb-Breit Hamiltonian.

    Args:
        two_electron: Type of two-electron interaction:
            - "none": One-electron Dirac only
            - "coulomb": Coulomb interaction (default)
            - "coulomb_gaunt": Coulomb + Gaunt (magnetic)
            - "coulomb_breit": Coulomb + full Breit (most accurate)

    Returns:
        RelativisticOperator object

    Example:
        H_DC = dirac_hamiltonian("coulomb")
        H_DCB = dirac_hamiltonian("coulomb_breit")
    """
    if two_electron == "none":
        return RelativisticOperator(symbol="H_D", operator_type="dirac")
    elif two_electron == "coulomb":
        return RelativisticOperator(symbol="H_{DC}", operator_type="dirac_coulomb")
    elif two_electron == "coulomb_gaunt":
        return RelativisticOperator(symbol="H_{DCG}", operator_type="dirac_coulomb_gaunt")
    elif two_electron == "coulomb_breit":
        return RelativisticOperator(symbol="H_{DCB}", operator_type="dirac_coulomb_breit")
    else:
        raise ValueError(f"Unknown two_electron type: {two_electron}")


# Pre-defined relativistic operators
H_DC = dirac_hamiltonian("coulomb")         # Dirac-Coulomb
H_DCB = dirac_hamiltonian("coulomb_breit")  # Dirac-Coulomb-Breit
