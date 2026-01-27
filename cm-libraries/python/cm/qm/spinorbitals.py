"""
Chemical Machines QM Spin-Orbitals Module

Spin-orbital classes, Slater determinants, and matrix elements.
"""

from typing import Optional, List, Union, Dict, Set, Tuple, Any
from dataclasses import dataclass, field

from ..symbols import Expr, Var, Const, _ensure_expr, _get_sympy
from .. import views
from .coordinates import Coordinate3D, CoordinateType, spherical_coord

__all__ = [
    'SpinOrbital',
    'SlaterDeterminant',
    'Operator',
    'Overlap',
    'MatrixElement',
    'spin_orbital',
    'basis_orbital',
    'basis_orbitals',
    'slater',
    'hamiltonian',
    'one_electron_operator',
    'two_electron_operator',
]


# =============================================================================
# SPIN-ORBITAL (NEW TUPLE-BASED FORMAT)
# =============================================================================


class SpinOrbital(Expr):
    """
    A spin-orbital defined as (vec3, n, l, m, spin, t=None).

    This class represents a single-particle quantum state with:
    - Spatial dependence via Coordinate3D (vec3)
    - Principal quantum number n
    - Angular momentum quantum number l
    - Magnetic quantum number m
    - Spin: +1 (alpha/up) or -1 (beta/down)
    - Optional time dependence t

    The orbital can be written symbolically as:
        phi_{n,l,m,sigma}(r, t) = R_nl(r) Y_l^m(theta, phi) chi_sigma [* e^(-iEt/hbar)]

    Attributes:
        vec3: Coordinate3D - spatial coordinate (symbolic)
        n: int - principal quantum number (can be None for basis-only specification)
        l: int - angular momentum quantum number (0=s, 1=p, 2=d, ...)
        m: int - magnetic quantum number (-l <= m <= l)
        spin: int - +1 for spin-up (alpha), -1 for spin-down (beta)
        t: Optional[Expr] - time parameter for time-dependent orbitals

    Example:
        from cm.symbols import Math
        from cm import qm

        # Create coordinate
        coord = qm.spherical_coord()

        # Create 1s spin-up orbital
        orbital = qm.SpinOrbital(coord, n=1, l=0, m=0, spin=1)

        # Time-dependent orbital
        t = Math.var("t")
        orbital_td = qm.SpinOrbital(coord, n=1, l=0, m=0, spin=1, t=t)
    """

    def __init__(self, vec3: Coordinate3D, n: Optional[int], l: int, m: int, spin: int,
                 t: Optional[Expr] = None, center: Optional[Union[int, str]] = None):
        super().__init__()

        # Validation
        if not isinstance(vec3, Coordinate3D):
            raise TypeError(f"vec3 must be Coordinate3D, got {type(vec3)}")
        if spin not in (1, -1):
            raise ValueError(f"spin must be +1 or -1, got {spin}")
        if l < 0:
            raise ValueError(f"l must be non-negative, got {l}")
        if abs(m) > l:
            raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
        if n is not None and n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        self._vec3 = vec3
        self._n = n
        self._l = l
        self._m = m
        self._spin = spin
        self._t = _ensure_expr(t) if t is not None else None
        self._center = center  # Atomic center label for molecular orbitals

    # Property accessors
    @property
    def vec3(self) -> Coordinate3D:
        """Spatial coordinate."""
        return self._vec3

    @property
    def n(self) -> Optional[int]:
        """Principal quantum number."""
        return self._n

    @property
    def l(self) -> int:
        """Angular momentum quantum number."""
        return self._l

    # Alias for backward compatibility
    @property
    def L(self) -> int:
        """Angular momentum quantum number (alias for l)."""
        return self._l

    @property
    def m(self) -> int:
        """Magnetic quantum number."""
        return self._m

    @property
    def spin(self) -> int:
        """Spin: +1 (up/alpha) or -1 (down/beta)."""
        return self._spin

    @property
    def t(self) -> Optional[Expr]:
        """Time parameter (None if time-independent)."""
        return self._t

    @property
    def is_time_dependent(self) -> bool:
        """True if orbital has explicit time dependence."""
        return self._t is not None

    @property
    def center(self) -> Optional[Union[int, str]]:
        """Atomic center label (for molecular orbitals)."""
        return self._center

    # Tuple representation (for compatibility and hashing)
    def as_tuple(self) -> tuple:
        """Return tuple representation (vec3, n, l, m, spin, t, center)."""
        return (self._vec3, self._n, self._l, self._m, self._spin, self._t, self._center)

    # Quantum number tuple (for orthogonality checks)
    @property
    def quantum_numbers(self) -> tuple:
        """Return (center, n, l, m, spin) tuple for orthogonality.

        Includes center label so orbitals on different atoms are distinguishable.
        """
        return (self._center, self._n, self._l, self._m, self._spin)

    # Labels (preserved from old API with updates)
    @property
    def spin_label(self) -> str:
        """Return spin as alpha/beta."""
        return "alpha" if self._spin == 1 else "beta"

    @property
    def spin_symbol(self) -> str:
        """Return spin as Greek letter (for LaTeX)."""
        return r"\alpha" if self._spin == 1 else r"\beta"

    @property
    def spin_arrow(self) -> str:
        """Return spin as up/down arrow."""
        return "↑" if self._spin == 1 else "↓"

    @property
    def spin_arrow_latex(self) -> str:
        """Return spin as LaTeX up/down arrow."""
        return r"\uparrow" if self._spin == 1 else r"\downarrow"

    @property
    def l_label(self) -> str:
        """Return l as spectroscopic notation (s, p, d, f, ...)."""
        labels = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k']
        if self._l < len(labels):
            return labels[self._l]
        return f"l{self._l}"

    # Alias for backward compatibility
    @property
    def L_label(self) -> str:
        """Return l as spectroscopic notation (alias for l_label)."""
        return self.l_label

    @property
    def center_label(self) -> str:
        """Subscript label for atomic center (e.g., '_A', '_0')."""
        if self._center is None:
            return ""
        return f"_{self._center}"

    @property
    def label(self) -> str:
        """Human-readable label like '2p↑(m=1)' or '2p↑_A(m=1)' with center."""
        n_str = str(self._n) if self._n is not None else ""
        return f"{n_str}{self.l_label}{self.center_label}{self.spin_arrow}(m={self._m})"

    @property
    def short_label(self) -> str:
        """Short label like '2p↑' or '2p↑_0' with center (without m)."""
        n_str = str(self._n) if self._n is not None else ""
        return f"{n_str}{self.l_label}{self.center_label}{self.spin_arrow}"

    @property
    def latex_label(self) -> str:
        """LaTeX formatted label."""
        n_str = str(self._n) if self._n is not None else ""
        center = f"_{{{self._center}}}" if self._center is not None else ""
        return f"{n_str}{self.l_label}{center}_{{{self.spin_arrow_latex}, m={self._m}}}"

    @property
    def ket_label(self) -> str:
        """Label for use in ket notation: |n, l, m, σ⟩ or |center, n, l, m, σ⟩."""
        if self._center is not None:
            if self._n is not None:
                return f"{self._center}, {self._n}, {self._l}, {self._m}, {self.spin_arrow_latex}"
            return f"{self._center}, {self._l}, {self._m}, {self.spin_arrow_latex}"
        if self._n is not None:
            return f"{self._n}, {self._l}, {self._m}, {self.spin_arrow_latex}"
        return f"{self._l}, {self._m}, {self.spin_arrow_latex}"

    # Orthogonality
    def is_orthogonal_to(self, other: "SpinOrbital") -> bool:
        """
        Check orthogonality based on quantum numbers.

        Orbitals are orthogonal if any quantum number differs,
        including the atomic center for molecular orbitals.
        """
        if not isinstance(other, SpinOrbital):
            return False
        # Different centers are orthogonal (atomic orbital approximation)
        if self._center is not None and other._center is not None:
            if self._center != other._center:
                return True
        if self._spin != other._spin:
            return True
        if self._l != other._l:
            return True
        if self._m != other._m:
            return True
        if self._n is not None and other._n is not None and self._n != other._n:
            return True
        return False

    # Expr interface
    def to_sympy(self):
        """
        Convert to SymPy representation.

        Returns a symbolic function phi_{n,l,m,spin}(r, theta, phi, [t]).
        """
        sp = _get_sympy()

        # Create a symbolic function representing the orbital
        name = f"phi_{self._n or ''}{self.l_label}_{self._m}_{self.spin_label}"

        if self._t is not None:
            return sp.Function(name)(
                self._vec3.c1.to_sympy(),
                self._vec3.c2.to_sympy(),
                self._vec3.c3.to_sympy(),
                self._t.to_sympy()
            )
        else:
            return sp.Function(name)(
                self._vec3.c1.to_sympy(),
                self._vec3.c2.to_sympy(),
                self._vec3.c3.to_sympy()
            )

    def to_latex(self) -> str:
        """Generate LaTeX representation."""
        # Base: phi_{n l m}^{sigma}
        n_str = str(self._n) if self._n is not None else ""
        subscript = f"{n_str}{self.l_label},{self._m}"
        superscript = self.spin_symbol

        # Coordinate part
        coord_latex = self._vec3.to_latex()

        if self._t is not None:
            t_latex = self._t.to_latex()
            return f"\\phi_{{{subscript}}}^{{{superscript}}}\\left({coord_latex}, {t_latex}\\right)"
        else:
            return f"\\phi_{{{subscript}}}^{{{superscript}}}\\left({coord_latex}\\right)"

    def to_latex_time_evolution(self, energy: Optional[Expr] = None) -> str:
        """
        Render with explicit time evolution factor.

        If energy is provided, shows e^{-iEt/hbar} phase factor.
        """
        base = self.to_latex()

        if self._t is None:
            return base

        if energy is not None:
            E_latex = energy.to_latex()
            t_latex = self._t.to_latex()
            phase = f"e^{{-i{E_latex}{t_latex}/\\hbar}}"
            return f"{base} \\cdot {phase}"
        else:
            return base

    def render_wavefunction(self, style: str = "compact"):
        """
        Render as explicit wavefunction.

        Args:
            style: "compact" for phi notation, "explicit" for R(r)Y(theta,phi)chi
        """
        if style == "explicit" and self._vec3.coord_type == CoordinateType.SPHERICAL:
            r_latex = self._vec3.c1.to_latex()
            theta_latex = self._vec3.c2.to_latex()
            phi_latex = self._vec3.c3.to_latex()
            n = self._n or ""
            l = self._l
            m = self._m

            latex = f"R_{{{n},{l}}}({r_latex}) Y_{{{l}}}^{{{m}}}({theta_latex}, {phi_latex}) \\chi_{{{self.spin_symbol}}}"

            if self._t is not None:
                latex += f" \\cdot e^{{-iE_{{{n},{l}}} {self._t.to_latex()}/\\hbar}}"
        else:
            latex = self.to_latex()

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def _get_free_variables(self) -> Set['Var']:
        result = self._vec3._get_free_variables()
        if self._t is not None:
            result |= self._t._get_free_variables()
        return result

    def __eq__(self, other):
        if not isinstance(other, SpinOrbital):
            return False
        return (self._vec3 == other._vec3 and
                self._n == other._n and
                self._l == other._l and
                self._m == other._m and
                self._spin == other._spin and
                self._center == other._center and
                ((self._t is None and other._t is None) or
                 (self._t is not None and other._t is not None and
                  self._t.to_latex() == other._t.to_latex())))

    def __hash__(self):
        t_hash = self._t.to_latex() if self._t else None
        return hash(('SpinOrbital', hash(self._vec3), self._n, self._l,
                     self._m, self._spin, self._center, t_hash))

    def __repr__(self):
        t_str = f", t={self._t}" if self._t else ""
        c_str = f", center={self._center}" if self._center is not None else ""
        return f"SpinOrbital(vec3={self._vec3}, n={self._n}, l={self._l}, m={self._m}, spin={self._spin}{c_str}{t_str})"


# =============================================================================
# SPIN-ORBITAL FACTORY FUNCTIONS (NEW API)
# =============================================================================


def spin_orbital(vec3: Coordinate3D, n: Optional[int], l: int, m: int, spin: int,
                 t: Optional[Expr] = None) -> SpinOrbital:
    """
    Create a spin-orbital with explicit parameters.

    Args:
        vec3: Coordinate3D spatial coordinate
        n: Principal quantum number (can be None)
        l: Angular momentum quantum number
        m: Magnetic quantum number
        spin: +1 for up, -1 for down
        t: Optional time parameter

    Returns:
        SpinOrbital object

    Example:
        coord = qm.spherical_coord()
        orbital = qm.spin_orbital(coord, n=1, l=0, m=0, spin=1)
    """
    return SpinOrbital(vec3, n, l, m, spin, t)


def basis_orbital(spec: tuple, vec3: Optional[Coordinate3D] = None,
                  t: Optional[Expr] = None) -> SpinOrbital:
    """
    Create a spin-orbital from a tuple specification.

    Args:
        spec: Tuple of (n, l, m, spin) or (l, m, spin)
        vec3: Optional coordinate (defaults to spherical_coord())
        t: Optional time parameter

    Returns:
        SpinOrbital object

    Example:
        orbital = qm.basis_orbital((1, 0, 0, 1))  # 1s up: (n, l, m, spin)
        orbital = qm.basis_orbital((0, 0, 1))     # s orbital, spin up (no n)
    """
    if vec3 is None:
        vec3 = spherical_coord()

    if len(spec) == 4:
        n, l, m, spin = spec
    elif len(spec) == 3:
        l, m, spin = spec
        n = None
    else:
        raise ValueError(f"Expected tuple of 3 (l, m, spin) or 4 (n, l, m, spin), got {len(spec)}")

    return SpinOrbital(vec3, n, l, m, spin, t)


def basis_orbitals(specs: List[tuple], vec3: Optional[Coordinate3D] = None,
                   t: Optional[Expr] = None) -> List[SpinOrbital]:
    """
    Create multiple spin-orbitals from tuple specifications.

    Args:
        specs: List of tuples (n, l, m, spin) or (l, m, spin)
        vec3: Shared coordinate (all orbitals use same spatial variables)
        t: Optional shared time parameter

    Returns:
        List of SpinOrbital objects

    Example:
        # Helium ground state
        orbitals = qm.basis_orbitals([
            (1, 0, 0, 1),   # 1s up: (n, l, m, spin)
            (1, 0, 0, -1),  # 1s down
        ])
    """
    if vec3 is None:
        vec3 = spherical_coord()
    return [basis_orbital(spec, vec3, t) for spec in specs]


# =============================================================================
# DEPRECATED FACTORY FUNCTIONS (OLD API)
# =============================================================================


def basis_sh_element(spin: int, L: int, m: int, n: Optional[int] = None) -> SpinOrbital:
    """
    DEPRECATED: Use spin_orbital() or basis_orbital() instead.

    Create a single spin-orbital basis element with spherical harmonic quantum numbers.

    Args:
        spin: +1 for spin-up (α), -1 for spin-down (β)
        L: Angular momentum quantum number (0=s, 1=p, 2=d, 3=f, ...)
        m: Magnetic quantum number (-L <= m <= L)
        n: Optional principal quantum number

    Returns:
        SpinOrbital object
    """
    warnings.warn(
        "basis_sh_element is deprecated. Use spin_orbital() or basis_orbital() instead.\n"
        "New format: SpinOrbital(coord, n, l, m, spin) or basis_orbital((n, l, m, spin))",
        DeprecationWarning,
        stacklevel=2
    )
    coord = spherical_coord()
    return SpinOrbital(coord, n, L, m, spin)


def basis_sh(quantum_numbers: List[Tuple]) -> List[SpinOrbital]:
    """
    DEPRECATED: Use basis_orbitals() instead.

    Create a list of spin-orbital basis elements from quantum number tuples.

    Note: Old format was (spin, L, m, [n]), new format is (n, l, m, spin).

    Args:
        quantum_numbers: List of tuples (spin, L, m) or (spin, L, m, n)
            - spin: +1 for spin-up, -1 for spin-down
            - L: Angular momentum quantum number
            - m: Magnetic quantum number
            - n: Optional principal quantum number

    Returns:
        List of SpinOrbital objects
    """
    warnings.warn(
        "basis_sh is deprecated. Use basis_orbitals() instead.\n"
        "Old format: (spin, L, m, [n]) -> New format: (n, l, m, spin)",
        DeprecationWarning,
        stacklevel=2
    )
    coord = spherical_coord()
    result = []
    for qn in quantum_numbers:
        if len(qn) == 3:
            spin, L, m = qn
            result.append(SpinOrbital(coord, None, L, m, spin))
        elif len(qn) == 4:
            spin, L, m, n = qn
            result.append(SpinOrbital(coord, n, L, m, spin))
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
        coord = qm.spherical_coord()
        orbitals = qm.basis_orbitals([
            (1, 0, 0, 1),   # 1s up
            (1, 0, 0, -1),  # 1s down
        ])
        psi = qm.SlaterDeterminant(orbitals)
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

        # Check for duplicate orbitals (Pauli exclusion) using quantum numbers
        qn_set = set(orb.quantum_numbers for orb in orbitals)
        if len(qn_set) != len(orbitals):
            raise ValueError("Duplicate quantum numbers detected - violates Pauli exclusion principle")

    @property
    def orbital_set(self) -> set:
        """Set of orbitals (for comparison)."""
        return set(self.orbitals)

    @property
    def quantum_number_set(self) -> set:
        """Set of quantum number tuples (n, l, m, spin)."""
        return set(orb.quantum_numbers for orb in self.orbitals)

    @property
    def vec3(self) -> Optional[Coordinate3D]:
        """Shared coordinate if all orbitals use the same one."""
        if not self.orbitals:
            return None
        first = self.orbitals[0].vec3
        if all(orb.vec3 == first for orb in self.orbitals):
            return first
        return None

    @property
    def t(self) -> Optional[Expr]:
        """Shared time parameter if all orbitals use the same one."""
        if not self.orbitals:
            return None
        first = self.orbitals[0].t
        if all(orb.t == first for orb in self.orbitals):
            return first
        return None

    @property
    def is_time_dependent(self) -> bool:
        """True if any orbital has time dependence."""
        return any(orb.is_time_dependent for orb in self.orbitals)

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

    def render(self, normalize: bool = False, show_time: bool = True):
        """
        Render the Slater determinant as LaTeX.

        Args:
            normalize: Include 1/√n! normalization factor
            show_time: Show time dependence if present
        """
        labels = ", ".join(self.ket_labels())

        if normalize:
            n = self.n_electrons
            latex = f"\\frac{{1}}{{\\sqrt{{{n}!}}}} | {labels} \\rangle"
        else:
            latex = f"| {labels} \\rangle"

        # Add time dependence notation
        if show_time and self.is_time_dependent:
            t_latex = self.t.to_latex() if self.t else "t"
            latex = f"\\Psi({t_latex}) = {latex}"

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def conjugate(self) -> "SlaterDeterminant":
        """
        Return the complex conjugate of the Slater determinant.

        For Slater determinants, conjugation creates the bra state ⟨Ψ|
        from the ket state |Ψ⟩. Since the determinant is built from
        orthonormal spin-orbitals, the conjugate is the same determinant
        but interpreted as a bra.

        This is primarily useful for clarity in expressions - the @ operator
        already handles conjugation implicitly when computing ⟨ψ|φ⟩.

        Returns:
            Self (SlaterDeterminant with same orbitals)

        Example:
            psi = qm.slater(orbitals)
            bra_psi = psi.conjugate()  # ⟨ψ|
            # Equivalent to: psi @ phi computes ⟨ψ|φ⟩
        """
        # For orthonormal basis, conjugate is the same determinant
        # The distinction is semantic (bra vs ket)
        return self

    # Alias for conjugate
    conj = conjugate

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
            # Check for MolecularHamiltonian (late import to avoid circular dependency)
            from .hamiltonian import MolecularHamiltonian
            if isinstance(other, MolecularHamiltonian):
                return PartialMatrixElement(self, other)
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

    def __init__(self, bra: SlaterDeterminant, operator: Any):
        """
        Args:
            bra: The bra (left) SlaterDeterminant
            operator: An Operator or MolecularHamiltonian
        """
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

    def __init__(self, bra: SlaterDeterminant, operator: Any, ket: SlaterDeterminant):
        """
        Args:
            bra: The bra (left) SlaterDeterminant
            operator: An Operator or MolecularHamiltonian
            ket: The ket (right) SlaterDeterminant
        """
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
            op_latex = self._get_operator_latex()
            latex = f"\\langle {bra_labels} | {op_latex} | {ket_labels} \\rangle"

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def _is_one_electron_only(self) -> bool:
        """Check if operator only contains one-electron terms."""
        # Check for Operator class
        if hasattr(self.operator, 'operator_type'):
            return self.operator.operator_type == "one_electron"
        # Check for MolecularHamiltonian
        if hasattr(self.operator, 'n_body_max'):
            return self.operator.n_body_max == 1
        # Default to full Hamiltonian
        return False

    def _get_operator_latex(self) -> str:
        """Get LaTeX representation of the operator."""
        if hasattr(self.operator, 'latex'):
            return self.operator.latex
        if hasattr(self.operator, 'to_latex'):
            return self.operator.to_latex()
        return "\\hat{H}"

    def _render_diagonal(self) -> str:
        """Render diagonal matrix element (same determinant)."""
        if self._is_one_electron_only():
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

        if self._is_one_electron_only():
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

        if self._is_one_electron_only():
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


