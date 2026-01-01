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
