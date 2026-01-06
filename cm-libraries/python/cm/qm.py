"""
Chemical Machines Quantum Mechanics Module

Provides tools for working with Slater determinants, spin-orbitals,
and quantum mechanical matrix elements using spherical harmonic basis functions.

Example:
    from cm import qm, Math

    # Create coordinates and spin-orbitals with new tuple-based format
    coord = qm.spherical_coord()
    orbitals = qm.basis_orbitals([
        (1, 0, 0, 1),   # 1s up: (n, l, m, spin)
        (1, 0, 0, -1),  # 1s down
        (2, 1, 0, 1),   # 2p up, m=0
    ])

    # Create Slater determinant
    psi = qm.SlaterDeterminant(orbitals)
    psi.render()

    # Time-dependent orbital
    t = Math.var("t")
    orbital_td = qm.SpinOrbital(coord, n=1, l=0, m=0, spin=1, t=t)

    # Inner products with automatic orthogonality
    phi = qm.SlaterDeterminant(qm.basis_orbitals([(1, 0, 0, 1), (1, 0, 0, -1), (2, 1, 1, 1)]))
    overlap = psi @ phi
    overlap.render()  # Renders 0 (orthogonal states)

    # Matrix elements with Hamiltonian
    H = qm.hamiltonian()
    matrix_elem = psi @ H @ phi
    matrix_elem.render()
"""

from typing import List, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from . import views
from .symbols import Expr, Var, Const, _ensure_expr, _get_sympy


# =============================================================================
# COORDINATE TYPES AND 3D COORDINATES
# =============================================================================

class CoordinateType(Enum):
    """
    Coordinate system type for spatial coordinates.

    CARTESIAN: (x, y, z) coordinates
    SPHERICAL: (r, theta, phi) coordinates where theta is polar, phi is azimuthal
    """
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"


class Coordinate3D(Expr):
    """
    A 3D coordinate vector with symbolic components and coordinate type metadata.

    This class represents a position in 3D space using symbolic Expr objects,
    allowing for symbolic manipulation, differentiation, and integration.

    Attributes:
        c1, c2, c3: Symbolic Expr components
        coord_type: CoordinateType (CARTESIAN or SPHERICAL)

    For CARTESIAN: (c1, c2, c3) = (x, y, z)
    For SPHERICAL: (c1, c2, c3) = (r, theta, phi)

    Example:
        from cm.symbols import Math
        from cm import qm

        # Cartesian coordinates
        x, y, z = Math.var("x"), Math.var("y"), Math.var("z")
        r_cart = qm.Coordinate3D(x, y, z, qm.CoordinateType.CARTESIAN)

        # Spherical coordinates
        r, theta, phi = Math.var("r"), Math.var("theta"), Math.var("phi")
        r_sph = qm.Coordinate3D(r, theta, phi, qm.CoordinateType.SPHERICAL)
    """

    def __init__(self, c1, c2, c3, coord_type: CoordinateType = CoordinateType.CARTESIAN):
        super().__init__()
        self._c1 = _ensure_expr(c1)
        self._c2 = _ensure_expr(c2)
        self._c3 = _ensure_expr(c3)
        self._coord_type = coord_type

    # Property accessors
    @property
    def c1(self) -> Expr:
        """First component (x for Cartesian, r for spherical)."""
        return self._c1

    @property
    def c2(self) -> Expr:
        """Second component (y for Cartesian, theta for spherical)."""
        return self._c2

    @property
    def c3(self) -> Expr:
        """Third component (z for Cartesian, phi for spherical)."""
        return self._c3

    @property
    def coord_type(self) -> CoordinateType:
        """Coordinate system type."""
        return self._coord_type

    # Semantic aliases for Cartesian
    @property
    def x(self) -> Expr:
        """Cartesian x (raises if not CARTESIAN)."""
        if self._coord_type != CoordinateType.CARTESIAN:
            raise ValueError("x property only valid for CARTESIAN coordinates")
        return self._c1

    @property
    def y(self) -> Expr:
        """Cartesian y (raises if not CARTESIAN)."""
        if self._coord_type != CoordinateType.CARTESIAN:
            raise ValueError("y property only valid for CARTESIAN coordinates")
        return self._c2

    @property
    def z(self) -> Expr:
        """Cartesian z (raises if not CARTESIAN)."""
        if self._coord_type != CoordinateType.CARTESIAN:
            raise ValueError("z property only valid for CARTESIAN coordinates")
        return self._c3

    # Semantic aliases for Spherical
    @property
    def r(self) -> Expr:
        """Radial coordinate (raises if not SPHERICAL)."""
        if self._coord_type != CoordinateType.SPHERICAL:
            raise ValueError("r property only valid for SPHERICAL coordinates")
        return self._c1

    @property
    def theta(self) -> Expr:
        """Polar angle (raises if not SPHERICAL)."""
        if self._coord_type != CoordinateType.SPHERICAL:
            raise ValueError("theta property only valid for SPHERICAL coordinates")
        return self._c2

    @property
    def phi(self) -> Expr:
        """Azimuthal angle (raises if not SPHERICAL)."""
        if self._coord_type != CoordinateType.SPHERICAL:
            raise ValueError("phi property only valid for SPHERICAL coordinates")
        return self._c3

    @property
    def components(self) -> tuple:
        """Return tuple of (c1, c2, c3)."""
        return (self._c1, self._c2, self._c3)

    # Required Expr methods
    def to_sympy(self):
        """Convert to SymPy Matrix (column vector)."""
        sp = _get_sympy()
        return sp.Matrix([
            self._c1.to_sympy(),
            self._c2.to_sympy(),
            self._c3.to_sympy()
        ])

    def to_latex(self) -> str:
        """Generate LaTeX representation."""
        c1_latex = self._c1.to_latex()
        c2_latex = self._c2.to_latex()
        c3_latex = self._c3.to_latex()

        if self._coord_type == CoordinateType.CARTESIAN:
            return f"({c1_latex}, {c2_latex}, {c3_latex})"
        else:
            return f"({c1_latex}, {c2_latex}, {c3_latex})"

    def _get_free_variables(self) -> Set['Var']:
        """Return set of free variables from all components."""
        result = set()
        result |= self._c1._get_free_variables()
        result |= self._c2._get_free_variables()
        result |= self._c3._get_free_variables()
        return result

    def __eq__(self, other):
        if not isinstance(other, Coordinate3D):
            return False
        return (self._c1.to_latex() == other._c1.to_latex() and
                self._c2.to_latex() == other._c2.to_latex() and
                self._c3.to_latex() == other._c3.to_latex() and
                self._coord_type == other._coord_type)

    def __hash__(self):
        return hash(('Coordinate3D',
                     self._c1.to_latex(),
                     self._c2.to_latex(),
                     self._c3.to_latex(),
                     self._coord_type))

    def __repr__(self):
        return f"Coordinate3D({self._c1}, {self._c2}, {self._c3}, {self._coord_type.value})"


# =============================================================================
# COORDINATE FACTORY FUNCTIONS
# =============================================================================

def coord3d(c1, c2, c3, coord_type: Union[CoordinateType, str] = CoordinateType.CARTESIAN) -> Coordinate3D:
    """
    Create a 3D coordinate vector.

    Args:
        c1, c2, c3: Coordinate components (Expr, int, or float)
        coord_type: CoordinateType enum or string ("cartesian" or "spherical")

    Returns:
        Coordinate3D object

    Example:
        from cm.symbols import Math
        from cm import qm

        r, theta, phi = Math.var("r"), Math.var("theta"), Math.var("phi")
        coord = qm.coord3d(r, theta, phi, "spherical")
    """
    if isinstance(coord_type, str):
        coord_type = CoordinateType(coord_type.lower())
    return Coordinate3D(c1, c2, c3, coord_type)


def spherical_coord(r=None, theta=None, phi=None) -> Coordinate3D:
    """
    Create spherical coordinates with default variable names.

    Args:
        r: Radial coordinate (default: Var("r"))
        theta: Polar angle (default: Var("theta"))
        phi: Azimuthal angle (default: Var("phi"))

    Returns:
        Coordinate3D with SPHERICAL type

    Example:
        coord = qm.spherical_coord()  # Uses r, theta, phi
        coord = qm.spherical_coord(r=Math.var("rho"))  # Custom r
    """
    r = r if r is not None else Var("r")
    theta = theta if theta is not None else Var("theta")
    phi = phi if phi is not None else Var("phi")
    return Coordinate3D(r, theta, phi, CoordinateType.SPHERICAL)


def cartesian_coord(x=None, y=None, z=None) -> Coordinate3D:
    """
    Create Cartesian coordinates with default variable names.

    Args:
        x: x coordinate (default: Var("x"))
        y: y coordinate (default: Var("y"))
        z: z coordinate (default: Var("z"))

    Returns:
        Coordinate3D with CARTESIAN type
    """
    x = x if x is not None else Var("x")
    y = y if y is not None else Var("y")
    z = z if z is not None else Var("z")
    return Coordinate3D(x, y, z, CoordinateType.CARTESIAN)


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
                 t: Optional[Expr] = None):
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

    # Tuple representation (for compatibility and hashing)
    def as_tuple(self) -> tuple:
        """Return tuple representation (vec3, n, l, m, spin, t)."""
        return (self._vec3, self._n, self._l, self._m, self._spin, self._t)

    # Quantum number tuple (for orthogonality checks)
    @property
    def quantum_numbers(self) -> tuple:
        """Return (n, l, m, spin) tuple for orthogonality."""
        return (self._n, self._l, self._m, self._spin)

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
    def label(self) -> str:
        """Human-readable label like '2p↑(m=1)'."""
        n_str = str(self._n) if self._n is not None else ""
        return f"{n_str}{self.l_label}{self.spin_arrow}(m={self._m})"

    @property
    def short_label(self) -> str:
        """Short label like '2p↑' (without m)."""
        n_str = str(self._n) if self._n is not None else ""
        return f"{n_str}{self.l_label}{self.spin_arrow}"

    @property
    def latex_label(self) -> str:
        """LaTeX formatted label."""
        n_str = str(self._n) if self._n is not None else ""
        return f"{n_str}{self.l_label}_{{{self.spin_arrow_latex}, m={self._m}}}"

    @property
    def ket_label(self) -> str:
        """Label for use in ket notation: |n, l, m, σ⟩."""
        if self._n is not None:
            return f"{self._n}, {self._l}, {self._m}, {self.spin_arrow_latex}"
        return f"{self._l}, {self._m}, {self.spin_arrow_latex}"

    # Orthogonality
    def is_orthogonal_to(self, other: "SpinOrbital") -> bool:
        """
        Check orthogonality based on quantum numbers.

        Orbitals are orthogonal if any quantum number differs.
        """
        if not isinstance(other, SpinOrbital):
            return False
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
                ((self._t is None and other._t is None) or
                 (self._t is not None and other._t is not None and
                  self._t.to_latex() == other._t.to_latex())))

    def __hash__(self):
        t_hash = self._t.to_latex() if self._t else None
        return hash(('SpinOrbital', hash(self._vec3), self._n, self._l,
                     self._m, self._spin, t_hash))

    def __repr__(self):
        t_str = f", t={self._t}" if self._t else ""
        return f"SpinOrbital(vec3={self._vec3}, n={self._n}, l={self._l}, m={self._m}, spin={self._spin}{t_str})"


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
