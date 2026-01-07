"""
Chemical Machines Quantum Mechanics Module

Provides tools for working with atoms, Slater determinants, spin-orbitals,
and quantum mechanical matrix elements using spherical harmonic basis functions.

Features:
    - Atom class with automatic electron configuration (aufbau principle)
    - Non-relativistic (SpinOrbital) and relativistic (DiracSpinor) orbitals
    - Slater and Dirac determinants with Slater-Condon rules
    - Visualization integration with views.molecule()

Example - Atoms:
    from cm import qm

    # Create atoms with automatic ground state configuration
    C = qm.atom('C')                          # Carbon at origin
    H = qm.atom('H', position=(0.74, 0, 0))   # Hydrogen at position

    # Access electron configuration
    print(C.configuration.label)  # "1s² 2s² 2p²"

    # Create Slater determinant from atom
    psi = C.slater_determinant()
    psi.render()

    # Ions
    Fe2 = qm.atom('Fe', charge=2)             # Fe2+ cation
    Cl_minus = qm.atom('Cl', charge=-1)       # Cl- anion

    # Custom electron configuration
    C_ex = qm.atom('C', configuration="1s2 2s1 2p3")  # Excited state

    # Relativistic atoms (for heavy elements)
    Au = qm.atom('Au', relativistic=True)
    psi_rel = Au.dirac_determinant()

    # Multiple atoms for molecules
    water = qm.atoms([
        ('O', 0.0, 0.0, 0.0),
        ('H', 0.96, 0.0, 0.0),
        ('H', -0.24, 0.93, 0.0),
    ])

Example - Spin-Orbitals (direct):
    from cm import qm, Math

    # Create coordinates and spin-orbitals with tuple-based format
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

Example - Matrix Elements:
    from cm import qm

    # Create two states
    psi = qm.SlaterDeterminant(qm.basis_orbitals([(1, 0, 0, 1), (1, 0, 0, -1), (2, 1, 0, 1)]))
    phi = qm.SlaterDeterminant(qm.basis_orbitals([(1, 0, 0, 1), (1, 0, 0, -1), (2, 1, 1, 1)]))

    # Inner products with automatic orthogonality
    overlap = psi @ phi
    overlap.render()  # Renders 0 (orthogonal states)

    # Matrix elements with Hamiltonian
    H = qm.hamiltonian()
    matrix_elem = psi @ H @ psi
    matrix_elem.render()

"""

from typing import List, Tuple, Optional, Union, Set, Any, Dict
from dataclasses import dataclass
from enum import Enum
import warnings
import re
from . import views
from .symbols import Expr, Var, Const, _ensure_expr, _get_sympy, Math


__all__ = [
    # Atomic data
    'ATOMIC_NUMBERS',
    'ELEMENT_SYMBOLS',
    'AUFBAU_ORDER',
    'NOBLE_GAS_CONFIGS',
    # Coordinate system
    'CoordinateType',
    'Coordinate3D',
    'coord3d',
    'spherical_coord',
    'cartesian_coord',
    # Spin-orbitals
    'SpinOrbital',
    'spin_orbital',
    'basis_orbital',
    'basis_orbitals',
    # Deprecated spin-orbital functions
    'basis_sh_element',
    'basis_sh',
    # Slater determinants
    'SlaterDeterminant',
    'slater',
    # Operators
    'Operator',
    'hamiltonian',
    'one_electron_operator',
    'two_electron_operator',
    'H',
    # Matrix elements
    'PartialMatrixElement',
    'Overlap',
    'MatrixElement',
    # Relativistic - Spinors
    'DiracSpinor',
    'dirac_spinor',
    'dirac_spinor_lj',
    'basis_dirac',
    'kappa_from_lj',
    # Relativistic - Determinants
    'DiracDeterminant',
    'dirac_slater',
    # Relativistic - Operators
    'RelativisticOperator',
    'dirac_hamiltonian',
    'H_DC',
    'H_DCB',
    # Relativistic - Matrix elements
    'PartialDiracMatrixElement',
    'DiracOverlap',
    'DiracMatrixElement',
    # Electron configuration
    'ElectronConfiguration',
    'ground_state',
    'config_from_string',
    # Atoms
    'Atom',
    'atom',
    'atoms',
    # Molecules
    'Molecule',
    'molecule',
    # Hamiltonian builder system
    'HamiltonianTerm',
    'HamiltonianBuilder',
    'MolecularHamiltonian',
    'MatrixExpression',
    'HamiltonianMatrix',
]


# =============================================================================
# ATOMIC DATA
# =============================================================================

# Atomic numbers for all elements (H through Og, Z=1-118)
ATOMIC_NUMBERS: Dict[str, int] = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
    'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
    'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
    'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
    'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,
    'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118,
}

# Reverse mapping: atomic number to symbol
ELEMENT_SYMBOLS: Dict[int, str] = {v: k for k, v in ATOMIC_NUMBERS.items()}

# Aufbau filling order: (n, l) pairs in order of increasing energy
AUFBAU_ORDER: List[Tuple[int, int]] = [
    (1, 0),  # 1s
    (2, 0),  # 2s
    (2, 1),  # 2p
    (3, 0),  # 3s
    (3, 1),  # 3p
    (4, 0),  # 4s
    (3, 2),  # 3d
    (4, 1),  # 4p
    (5, 0),  # 5s
    (4, 2),  # 4d
    (5, 1),  # 5p
    (6, 0),  # 6s
    (4, 3),  # 4f
    (5, 2),  # 5d
    (6, 1),  # 6p
    (7, 0),  # 7s
    (5, 3),  # 5f
    (6, 2),  # 6d
    (7, 1),  # 7p
]

# Noble gas configurations for shorthand notation
NOBLE_GAS_CONFIGS: Dict[str, int] = {
    'He': 2, 'Ne': 10, 'Ar': 18, 'Kr': 36, 'Xe': 54, 'Rn': 86,
}

# Spectroscopic notation for angular momentum
L_LABELS = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k']


def _max_electrons_in_subshell(l: int) -> int:
    """Maximum electrons in a subshell: 2(2l+1)."""
    return 2 * (2 * l + 1)


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


# =============================================================================
# ELECTRON CONFIGURATION
# =============================================================================

class ElectronConfiguration:
    """
    Represents an electron configuration for an atom.

    Supports automatic aufbau filling (ground state) and manual orbital
    specification for excited states or custom configurations.

    The configuration stores orbitals as (n, l, m, spin) tuples following
    Hund's rules for ground state filling.

    Example:
        # Ground state carbon (automatic aufbau)
        config = ElectronConfiguration.aufbau(6)
        print(config.label)  # "1s² 2s² 2p²"

        # Manual specification
        config = ElectronConfiguration.manual([
            (1, 0, 0, 1), (1, 0, 0, -1),  # 1s²
            (2, 0, 0, 1), (2, 0, 0, -1),  # 2s²
            (2, 1, 0, 1), (2, 1, 1, 1),   # 2p² (Hund's rule)
        ])

        # From string notation
        config = ElectronConfiguration.from_string("1s2 2s2 2p2")
        config = ElectronConfiguration.from_string("[He] 2s2 2p2")
    """

    def __init__(self, orbitals: List[Tuple[int, int, int, int]]):
        """
        Create an electron configuration from explicit orbital list.

        Args:
            orbitals: List of (n, l, m, spin) tuples where:
                - n: Principal quantum number (>= 1)
                - l: Angular momentum (0 to n-1)
                - m: Magnetic quantum number (-l to l)
                - spin: +1 (up/alpha) or -1 (down/beta)

        Raises:
            ValueError: If quantum numbers are invalid or Pauli exclusion violated
        """
        self._orbitals = list(orbitals)
        self._validate()

    def _validate(self):
        """Validate quantum numbers and Pauli exclusion."""
        seen = set()
        for n, l, m, spin in self._orbitals:
            if n < 1:
                raise ValueError(f"Principal quantum number n must be >= 1, got {n}")
            if l < 0 or l >= n:
                raise ValueError(f"Angular momentum l must be 0 <= l < n, got l={l}, n={n}")
            if abs(m) > l:
                raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
            if spin not in (1, -1):
                raise ValueError(f"Spin must be +1 or -1, got {spin}")

            key = (n, l, m, spin)
            if key in seen:
                raise ValueError(f"Pauli exclusion violation: duplicate orbital {key}")
            seen.add(key)

    @classmethod
    def aufbau(cls, n_electrons: int) -> "ElectronConfiguration":
        """
        Create ground state configuration using aufbau principle.

        Fills orbitals following:
        1. Aufbau order (increasing n+l, then n)
        2. Hund's rules (maximize spin within subshell)
        3. Pauli exclusion principle

        Args:
            n_electrons: Number of electrons to place

        Returns:
            ElectronConfiguration with ground state filling

        Example:
            config = ElectronConfiguration.aufbau(6)  # Carbon
            print(config.label)  # "1s² 2s² 2p²"
        """
        if n_electrons < 0:
            raise ValueError(f"Number of electrons must be >= 0, got {n_electrons}")

        orbitals = []
        remaining = n_electrons

        for n, l in AUFBAU_ORDER:
            if remaining <= 0:
                break

            max_in_subshell = _max_electrons_in_subshell(l)
            electrons_to_add = min(remaining, max_in_subshell)

            # Fill following Hund's rules: spin up first, then spin down
            subshell_orbitals = cls._fill_subshell(n, l, electrons_to_add)
            orbitals.extend(subshell_orbitals)
            remaining -= electrons_to_add

        return cls(orbitals)

    @staticmethod
    def _fill_subshell(n: int, l: int, n_electrons: int) -> List[Tuple[int, int, int, int]]:
        """
        Fill a subshell following Hund's rules.

        First fills all m values with spin up, then spin down.
        This maximizes total spin (Hund's first rule).

        Args:
            n: Principal quantum number
            l: Angular momentum quantum number
            n_electrons: Number of electrons to place

        Returns:
            List of (n, l, m, spin) tuples
        """
        orbitals = []
        m_values = list(range(-l, l + 1))  # -l, -l+1, ..., l-1, l

        # First pass: fill with spin up (+1)
        for m in m_values:
            if len(orbitals) >= n_electrons:
                break
            orbitals.append((n, l, m, 1))

        # Second pass: fill with spin down (-1)
        for m in m_values:
            if len(orbitals) >= n_electrons:
                break
            orbitals.append((n, l, m, -1))

        return orbitals

    @classmethod
    def manual(cls, orbitals: List[Tuple[int, int, int, int]]) -> "ElectronConfiguration":
        """
        Create configuration from explicit orbital list.

        Args:
            orbitals: List of (n, l, m, spin) tuples

        Returns:
            ElectronConfiguration with specified orbitals

        Example:
            # Excited state with electron promoted
            config = ElectronConfiguration.manual([
                (1, 0, 0, 1), (1, 0, 0, -1),  # 1s²
                (2, 0, 0, 1),                  # 2s¹
                (2, 1, -1, 1), (2, 1, 0, 1), (2, 1, 1, 1),  # 2p³
            ])
        """
        return cls(orbitals)

    @classmethod
    def from_string(cls, notation: str) -> "ElectronConfiguration":
        """
        Parse electron configuration from string notation.

        Supports:
        - Standard notation: "1s2 2s2 2p6"
        - Noble gas core: "[He] 2s2 2p2", "[Ar] 3d10 4s2"
        - Superscript numbers: "1s² 2s² 2p²"

        Args:
            notation: Configuration string

        Returns:
            ElectronConfiguration object

        Example:
            config = ElectronConfiguration.from_string("1s2 2s2 2p2")
            config = ElectronConfiguration.from_string("[Ne] 3s2 3p4")
        """
        # Normalize: replace superscript digits with regular digits
        superscripts = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
                        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}
        for sup, reg in superscripts.items():
            notation = notation.replace(sup, reg)

        orbitals = []

        # Check for noble gas core notation [He], [Ne], etc.
        noble_match = re.match(r'\[(\w+)\]\s*', notation)
        if noble_match:
            noble_gas = noble_match.group(1)
            if noble_gas not in NOBLE_GAS_CONFIGS:
                raise ValueError(f"Unknown noble gas: {noble_gas}")
            # Get the noble gas configuration
            n_core = NOBLE_GAS_CONFIGS[noble_gas]
            core_config = cls.aufbau(n_core)
            orbitals.extend(core_config._orbitals)
            notation = notation[noble_match.end():]

        # Parse remaining subshells: "1s2", "2p6", "3d10", etc.
        pattern = r'(\d+)([spdfghik])(\d+)'
        for match in re.finditer(pattern, notation.lower()):
            n = int(match.group(1))
            l_char = match.group(2)
            count = int(match.group(3))

            l = L_LABELS.index(l_char)
            max_e = _max_electrons_in_subshell(l)
            if count > max_e:
                raise ValueError(f"Too many electrons in {n}{l_char}: {count} > {max_e}")

            subshell = cls._fill_subshell(n, l, count)
            orbitals.extend(subshell)

        if not orbitals:
            raise ValueError(f"Could not parse configuration: {notation}")

        return cls(orbitals)

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return len(self._orbitals)

    @property
    def orbitals(self) -> List[Tuple[int, int, int, int]]:
        """List of (n, l, m, spin) tuples."""
        return list(self._orbitals)

    @property
    def subshell_occupancy(self) -> Dict[str, int]:
        """
        Return occupancy by subshell.

        Returns:
            Dict mapping subshell labels to electron counts
            e.g., {'1s': 2, '2s': 2, '2p': 2}
        """
        occupancy: Dict[str, int] = {}
        for n, l, m, spin in self._orbitals:
            key = f"{n}{L_LABELS[l]}"
            occupancy[key] = occupancy.get(key, 0) + 1
        return occupancy

    @property
    def shell_occupancy(self) -> Dict[int, int]:
        """
        Return occupancy by shell (principal quantum number).

        Returns:
            Dict mapping n to electron counts
            e.g., {1: 2, 2: 4}
        """
        occupancy: Dict[int, int] = {}
        for n, l, m, spin in self._orbitals:
            occupancy[n] = occupancy.get(n, 0) + 1
        return occupancy

    @property
    def label(self) -> str:
        """
        Human-readable label using superscript notation.

        Returns:
            String like "1s² 2s² 2p²"
        """
        superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        parts = []
        for subshell, count in self.subshell_occupancy.items():
            count_str = ''.join(superscripts[d] for d in str(count))
            parts.append(f"{subshell}{count_str}")
        return ' '.join(parts)

    @property
    def latex_label(self) -> str:
        """
        LaTeX formatted label.

        Returns:
            String like "1s^{2} 2s^{2} 2p^{2}"
        """
        parts = []
        for subshell, count in self.subshell_occupancy.items():
            parts.append(f"{subshell}^{{{count}}}")
        return ' '.join(parts)

    def to_latex(self) -> str:
        """LaTeX representation."""
        return self.latex_label

    def excite(self, from_orbital: Tuple[int, int, int, int],
               to_orbital: Tuple[int, int, int, int]) -> "ElectronConfiguration":
        """
        Create excited state by moving an electron.

        Args:
            from_orbital: (n, l, m, spin) of orbital to vacate
            to_orbital: (n, l, m, spin) of orbital to occupy

        Returns:
            New ElectronConfiguration with the excitation

        Raises:
            ValueError: If from_orbital not occupied or to_orbital already occupied
        """
        if from_orbital not in self._orbitals:
            raise ValueError(f"Cannot excite from {from_orbital}: not occupied")
        if to_orbital in self._orbitals:
            raise ValueError(f"Cannot excite to {to_orbital}: already occupied")

        new_orbitals = [o for o in self._orbitals if o != from_orbital]
        new_orbitals.append(to_orbital)
        return ElectronConfiguration(new_orbitals)

    def ionize(self, n: int = 1) -> "ElectronConfiguration":
        """
        Remove electrons (create cation).

        Removes electrons from highest energy orbitals first.

        Args:
            n: Number of electrons to remove (default 1)

        Returns:
            New ElectronConfiguration with fewer electrons

        Raises:
            ValueError: If n > number of electrons
        """
        if n > len(self._orbitals):
            raise ValueError(f"Cannot remove {n} electrons from {len(self._orbitals)}")
        if n <= 0:
            return ElectronConfiguration(self._orbitals)

        # Sort orbitals by energy (reverse aufbau order) and remove from highest
        def orbital_energy_key(orb):
            n, l, m, spin = orb
            # Find position in aufbau order
            try:
                idx = AUFBAU_ORDER.index((n, l))
            except ValueError:
                idx = 100  # High value for orbitals not in standard order
            return (idx, m, -spin)  # Higher index = higher energy

        sorted_orbitals = sorted(self._orbitals, key=orbital_energy_key, reverse=True)
        return ElectronConfiguration(sorted_orbitals[n:])

    def add_electron(self, orbital: Optional[Tuple[int, int, int, int]] = None) -> "ElectronConfiguration":
        """
        Add an electron (create anion).

        Args:
            orbital: Specific (n, l, m, spin) to add, or None for next aufbau position

        Returns:
            New ElectronConfiguration with additional electron
        """
        if orbital is not None:
            if orbital in self._orbitals:
                raise ValueError(f"Orbital {orbital} already occupied")
            return ElectronConfiguration(self._orbitals + [orbital])

        # Find next available orbital following aufbau
        next_config = ElectronConfiguration.aufbau(len(self._orbitals) + 1)
        # The last orbital in aufbau is the one to add
        new_orbital = next_config._orbitals[-1]
        if new_orbital in self._orbitals:
            raise ValueError("Cannot determine next orbital to add")
        return ElectronConfiguration(self._orbitals + [new_orbital])

    def to_spinors(self) -> List[Tuple[int, int, float]]:
        """
        Convert to relativistic spinor quantum numbers.

        Maps (n, l, m, spin) to (n, kappa, mj) for DiracSpinor.

        The mapping considers that for a given mj, we need to choose
        the appropriate j (and hence kappa) such that |mj| <= j.

        Returns:
            List of (n, kappa, mj) tuples
        """
        spinors = []
        for n, l, m, spin in self._orbitals:
            # mj = m + spin/2
            mj = m + spin * 0.5

            if l == 0:
                # s orbital: only j = 1/2, kappa = -1
                kappa = -1
            else:
                # For l > 0, we have two choices: j = l + 1/2 or j = l - 1/2
                # j = l + 1/2 -> kappa = -(l+1), valid mj range: [-l-0.5, l+0.5]
                # j = l - 1/2 -> kappa = l, valid mj range: [-l+0.5, l-0.5]

                j_high = l + 0.5  # j = l + 1/2
                j_low = l - 0.5   # j = l - 1/2

                # Check if mj is valid for j_low (smaller j)
                if j_low >= 0.5 and abs(mj) <= j_low:
                    # Can use either j, prefer j_low for spin-down, j_high for spin-up
                    if spin == -1:
                        kappa = l  # j = l - 1/2
                    else:
                        kappa = -(l + 1)  # j = l + 1/2
                elif abs(mj) <= j_high:
                    # Must use j_high
                    kappa = -(l + 1)  # j = l + 1/2
                else:
                    # This shouldn't happen with valid quantum numbers
                    # Fall back to j_high
                    kappa = -(l + 1)

            spinors.append((n, kappa, mj))
        return spinors

    @property
    def spinors(self) -> List[Tuple[int, int, float]]:
        """List of (n, kappa, mj) tuples for relativistic calculations."""
        return self.to_spinors()

    def __repr__(self) -> str:
        return f"ElectronConfiguration({self.label})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ElectronConfiguration):
            return False
        return set(self._orbitals) == set(other._orbitals)

    def __hash__(self) -> int:
        return hash(frozenset(self._orbitals))


# =============================================================================
# ATOM CLASS
# =============================================================================

class Atom(Expr):
    """
    Represents an atom with nuclear charge, position, and electron configuration.

    The Atom class integrates with the existing qm module to provide:
    - Automatic or manual electron configurations
    - Non-relativistic (SpinOrbital) or relativistic (DiracSpinor) orbitals
    - Coordinate system support (numeric or symbolic Coordinate3D)
    - Visualization via views.molecule()
    - Slater/Dirac determinant generation for QM calculations
    - ComputeGraph support for symbolic computation and PyTorch compilation

    Example:
        # Simple carbon atom at origin
        C = qm.atom('C')
        psi = C.slater_determinant()

        # Carbon with explicit position
        C = qm.atom('C', position=(1.0, 0.0, 0.0))

        # Manual electron configuration
        C_excited = qm.atom('C', configuration=ElectronConfiguration.from_string("1s2 2s1 2p3"))

        # Relativistic uranium
        U = qm.atom('U', relativistic=True)
        psi = U.dirac_determinant()

        # ComputeGraph for Hamiltonian evaluation
        H = qm.atom('H')
        cg = H.to_compute_graph(r=1.0)
        energy = cg.evaluate()
    """

    def __init__(
        self,
        element: Union[str, int],
        position: Union[Tuple[float, float, float], Coordinate3D] = (0.0, 0.0, 0.0),
        configuration: Optional[ElectronConfiguration] = None,
        relativistic: bool = False,
        charge: int = 0,
        vec3: Optional[Coordinate3D] = None,
    ):
        """
        Create an Atom.

        Args:
            element: Element symbol ('C', 'Fe') or atomic number (6, 26)
            position: Atom center position as (x, y, z) tuple or Coordinate3D
            configuration: Electron configuration (None for automatic aufbau)
            relativistic: Use DiracSpinor (True) or SpinOrbital (False)
            charge: Ionic charge (positive for cations, negative for anions)
            vec3: Shared Coordinate3D for orbital wavefunctions (default: spherical)

        Raises:
            ValueError: If element is unknown or charge exceeds atomic number
        """
        super().__init__()

        # Resolve element symbol and atomic number
        if isinstance(element, str):
            if element not in ATOMIC_NUMBERS:
                raise ValueError(f"Unknown element symbol: {element}")
            self._symbol = element
            self._Z = ATOMIC_NUMBERS[element]
        else:
            if element not in ELEMENT_SYMBOLS:
                raise ValueError(f"Unknown atomic number: {element}")
            self._Z = element
            self._symbol = ELEMENT_SYMBOLS[element]

        # Validate charge
        if charge > self._Z:
            raise ValueError(f"Charge {charge} exceeds atomic number {self._Z}")

        self._charge = charge
        self._relativistic = relativistic

        # Set position
        if isinstance(position, Coordinate3D):
            self._position = position
        else:
            # Store as tuple for numeric positions
            self._position = tuple(position)

        # Set orbital coordinate system
        if vec3 is not None:
            self._vec3 = vec3
        else:
            # Default to spherical coordinates for orbital wavefunctions
            self._vec3 = spherical_coord()

        # Set electron configuration
        n_electrons = self._Z - charge
        if configuration is not None:
            if configuration.n_electrons != n_electrons:
                raise ValueError(
                    f"Configuration has {configuration.n_electrons} electrons, "
                    f"but atom {self._symbol} with charge {charge} should have {n_electrons}"
                )
            self._configuration = configuration
        else:
            # Automatic aufbau filling
            self._configuration = ElectronConfiguration.aufbau(n_electrons)

        # Cache for orbitals/spinors
        self._orbitals_cache: Optional[List[SpinOrbital]] = None
        self._spinors_cache: Optional[List[DiracSpinor]] = None

    # =========================================================================
    # ELEMENT PROPERTIES
    # =========================================================================

    @property
    def symbol(self) -> str:
        """Element symbol (e.g., 'C', 'Fe')."""
        return self._symbol

    @property
    def Z(self) -> int:
        """Atomic number (nuclear charge)."""
        return self._Z

    @property
    def name(self) -> str:
        """Element name (e.g., 'Carbon', 'Iron')."""
        # Get from views.ELEMENT_DATA if available
        if self._symbol in views.ELEMENT_DATA:
            return views.ELEMENT_DATA[self._symbol].get('name', self._symbol)
        return self._symbol

    @property
    def n_electrons(self) -> int:
        """Number of electrons (Z - charge)."""
        return self._configuration.n_electrons

    @property
    def charge(self) -> int:
        """Net ionic charge."""
        return self._charge

    @property
    def is_ion(self) -> bool:
        """True if charged (cation or anion)."""
        return self._charge != 0

    # =========================================================================
    # POSITION PROPERTIES
    # =========================================================================

    @property
    def position(self) -> Union[Tuple[float, float, float], Coordinate3D]:
        """Atom center position."""
        return self._position

    @property
    def x(self) -> Union[float, Expr]:
        """X coordinate of atom center."""
        if isinstance(self._position, Coordinate3D):
            if self._position.coord_type == CoordinateType.CARTESIAN:
                return self._position.x
            raise ValueError("Position is in spherical coordinates, use .position instead")
        return self._position[0]

    @property
    def y(self) -> Union[float, Expr]:
        """Y coordinate of atom center."""
        if isinstance(self._position, Coordinate3D):
            if self._position.coord_type == CoordinateType.CARTESIAN:
                return self._position.y
            raise ValueError("Position is in spherical coordinates, use .position instead")
        return self._position[1]

    @property
    def z(self) -> Union[float, Expr]:
        """Z coordinate of atom center."""
        if isinstance(self._position, Coordinate3D):
            if self._position.coord_type == CoordinateType.CARTESIAN:
                return self._position.z
            raise ValueError("Position is in spherical coordinates, use .position instead")
        return self._position[2]

    @property
    def is_symbolic(self) -> bool:
        """True if position uses symbolic Coordinate3D."""
        return isinstance(self._position, Coordinate3D)

    @property
    def numeric_position(self) -> Tuple[float, float, float]:
        """
        Get numeric (x, y, z) position.

        Raises:
            ValueError: If position is symbolic and cannot be evaluated
        """
        if isinstance(self._position, tuple):
            return self._position
        raise ValueError("Position is symbolic Coordinate3D, cannot get numeric position")

    # =========================================================================
    # CONFIGURATION PROPERTIES
    # =========================================================================

    @property
    def configuration(self) -> ElectronConfiguration:
        """Electron configuration object."""
        return self._configuration

    @property
    def relativistic(self) -> bool:
        """True if using relativistic (DiracSpinor) orbitals."""
        return self._relativistic

    @property
    def vec3(self) -> Coordinate3D:
        """Shared coordinate for orbital wavefunctions."""
        return self._vec3

    @property
    def orbitals(self) -> List[SpinOrbital]:
        """
        List of SpinOrbital objects for this atom.

        Creates SpinOrbital objects from the electron configuration.
        """
        if self._orbitals_cache is None:
            self._orbitals_cache = [
                SpinOrbital(self._vec3, n=n, l=l, m=m, spin=spin)
                for n, l, m, spin in self._configuration.orbitals
            ]
        return self._orbitals_cache

    @property
    def spinors(self) -> List[DiracSpinor]:
        """
        List of DiracSpinor objects for this atom.

        Creates DiracSpinor objects from the electron configuration.
        """
        if self._spinors_cache is None:
            self._spinors_cache = [
                DiracSpinor(n=n, kappa=kappa, mj=mj)
                for n, kappa, mj in self._configuration.spinors
            ]
        return self._spinors_cache

    # =========================================================================
    # SLATER DETERMINANT GENERATION
    # =========================================================================

    def slater_determinant(self) -> SlaterDeterminant:
        """
        Create a SlaterDeterminant for this atom's electrons.

        Returns:
            SlaterDeterminant containing all occupied orbitals

        Example:
            C = qm.atom('C')
            psi = C.slater_determinant()
            H = qm.hamiltonian()
            energy = psi @ H @ psi
        """
        return SlaterDeterminant(self.orbitals)

    def dirac_determinant(self) -> DiracDeterminant:
        """
        Create a DiracDeterminant for this atom's electrons.

        Returns:
            DiracDeterminant containing all occupied spinors

        Example:
            U = qm.atom('U', relativistic=True)
            psi = U.dirac_determinant()
            H_DC = qm.dirac_hamiltonian()
            energy = psi @ H_DC @ psi
        """
        return DiracDeterminant(self.spinors)

    def determinant(self) -> Union[SlaterDeterminant, DiracDeterminant]:
        """
        Create appropriate determinant based on relativistic setting.

        Returns:
            SlaterDeterminant if relativistic=False
            DiracDeterminant if relativistic=True
        """
        if self._relativistic:
            return self.dirac_determinant()
        return self.slater_determinant()

    # =========================================================================
    # EXPR INTERFACE (required for symbolic computation)
    # =========================================================================

    def to_sympy(self):
        """
        Convert to SymPy representation.

        Returns the nuclear charge Z as a SymPy number.
        """
        sp = _get_sympy()
        return sp.Integer(self._Z)

    def to_latex(self) -> str:
        """
        LaTeX representation of the atom.

        Returns:
            LaTeX string like "\\mathrm{C}" or "\\mathrm{Fe}^{2+}"
        """
        if self._charge == 0:
            return f"\\mathrm{{{self._symbol}}}"
        elif self._charge > 0:
            return f"\\mathrm{{{self._symbol}}}^{{{self._charge}+}}"
        else:
            return f"\\mathrm{{{self._symbol}}}^{{{abs(self._charge)}-}}"

    def _get_free_variables(self) -> Set[Var]:
        """
        Return free variables from position/coordinates.

        Returns:
            Set of Var objects from symbolic position or vec3
        """
        free_vars: Set[Var] = set()
        if isinstance(self._position, Coordinate3D):
            free_vars.update(self._position._get_free_variables())
        free_vars.update(self._vec3._get_free_variables())
        return free_vars

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def to_molecule_tuple(self) -> Tuple[str, float, float, float]:
        """
        Convert to views.molecule() format.

        Returns:
            (symbol, x, y, z) tuple for use with views.molecule()

        Raises:
            ValueError: If position is symbolic and not evaluable
        """
        if isinstance(self._position, tuple):
            return (self._symbol, self._position[0], self._position[1], self._position[2])
        raise ValueError("Cannot convert symbolic position to molecule tuple")

    def render(self, style: str = 'ball-stick'):
        """
        Render atom using views.molecule().

        Args:
            style: Visualization style ('ball-stick', 'spacefill')
        """
        try:
            mol_tuple = self.to_molecule_tuple()
            views.molecule([mol_tuple], style=style)
        except ValueError:
            # Symbolic position - render as text instead
            html = f'<div class="cm-math cm-math-center">\\[ {self.to_latex()} \\]</div>'
            views.html(html)

    def render_configuration(self, style: str = 'text'):
        """
        Render electron configuration.

        Args:
            style: 'text' for simple label, 'latex' for LaTeX
        """
        if style == 'latex':
            latex = f"{self.to_latex()}: {self._configuration.latex_label}"
            html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        else:
            html = f'<div>{self._symbol}: {self._configuration.label}</div>'
        views.html(html)

    # =========================================================================
    # MODIFICATION (return new atom)
    # =========================================================================

    def with_position(self, position: Union[Tuple[float, float, float], Coordinate3D]) -> "Atom":
        """Create new atom with different position."""
        return Atom(
            element=self._symbol,
            position=position,
            configuration=self._configuration,
            relativistic=self._relativistic,
            charge=self._charge,
            vec3=self._vec3,
        )

    def with_configuration(self, configuration: ElectronConfiguration) -> "Atom":
        """Create new atom with different configuration."""
        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=configuration,
            relativistic=self._relativistic,
            charge=self._Z - configuration.n_electrons,
            vec3=self._vec3,
        )

    def ionize(self, charge_delta: int = 1) -> "Atom":
        """
        Create ion by changing charge.

        Args:
            charge_delta: Change in charge (+1 removes electron, -1 adds electron)

        Returns:
            New Atom with updated charge and configuration
        """
        new_charge = self._charge + charge_delta
        new_config = self._configuration.ionize(charge_delta) if charge_delta > 0 else \
                     self._configuration.add_electron() if charge_delta == -1 else \
                     self._configuration

        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=new_config,
            relativistic=self._relativistic,
            charge=new_charge,
            vec3=self._vec3,
        )

    def excite(self, from_orbital: Tuple[int, int, int, int],
               to_orbital: Tuple[int, int, int, int]) -> "Atom":
        """
        Create excited state atom.

        Args:
            from_orbital: (n, l, m, spin) of orbital to vacate
            to_orbital: (n, l, m, spin) of orbital to occupy

        Returns:
            New Atom with excited configuration
        """
        new_config = self._configuration.excite(from_orbital, to_orbital)
        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=new_config,
            relativistic=self._relativistic,
            charge=self._charge,
            vec3=self._vec3,
        )

    def to_relativistic(self) -> "Atom":
        """Convert to relativistic representation."""
        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=self._configuration,
            relativistic=True,
            charge=self._charge,
            vec3=self._vec3,
        )

    def to_nonrelativistic(self) -> "Atom":
        """Convert to non-relativistic representation."""
        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=self._configuration,
            relativistic=False,
            charge=self._charge,
            vec3=self._vec3,
        )

    # =========================================================================
    # LABELING
    # =========================================================================

    @property
    def label(self) -> str:
        """Human-readable label."""
        if self._charge == 0:
            return f"{self._symbol} ({self.n_electrons} electrons)"
        elif self._charge > 0:
            return f"{self._symbol}{self._charge}+ ({self.n_electrons} electrons)"
        else:
            return f"{self._symbol}{abs(self._charge)}- ({self.n_electrons} electrons)"

    @property
    def ion_label(self) -> str:
        """Ion notation like 'Fe2+' or 'Cl-'."""
        if self._charge == 0:
            return self._symbol
        elif self._charge > 0:
            return f"{self._symbol}{self._charge}+" if self._charge > 1 else f"{self._symbol}+"
        else:
            return f"{self._symbol}{abs(self._charge)}-" if self._charge < -1 else f"{self._symbol}-"

    @property
    def ket_label(self) -> str:
        """Ket notation label for use in determinants."""
        return self._symbol

    # =========================================================================
    # ENERGY CALCULATIONS
    # =========================================================================

    def energy(self, hamiltonian: "MolecularHamiltonian") -> "MatrixExpression":
        """
        Calculate ground state energy using the given Hamiltonian.

        Creates a single-atom molecule wrapper to properly handle Z values.

        Args:
            hamiltonian: MolecularHamiltonian to use

        Returns:
            MatrixExpression for ⟨Ψ|H|Ψ⟩

        Example:
            H = qm.HamiltonianBuilder.electronic().build()
            C = qm.atom('C')
            E = C.energy(H)
            E_val = E.numerical()
        """
        # Create a single-atom molecule wrapper
        mol = Molecule([(self._symbol, *self._position)])
        return mol.energy(hamiltonian)

    # =========================================================================
    # SPECIAL METHODS
    # =========================================================================

    def __repr__(self) -> str:
        pos_str = f"({self._position[0]}, {self._position[1]}, {self._position[2]})" \
                  if isinstance(self._position, tuple) else "symbolic"
        return f"Atom({self._symbol}, Z={self._Z}, pos={pos_str}, charge={self._charge})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return False
        return (self._symbol == other._symbol and
                self._charge == other._charge and
                self._configuration == other._configuration)

    def __hash__(self) -> int:
        return hash((self._symbol, self._charge, hash(self._configuration)))


# =============================================================================
# ATOM FACTORY FUNCTIONS
# =============================================================================

def atom(
    element: Union[str, int],
    position: Union[Tuple[float, float, float], Coordinate3D] = (0.0, 0.0, 0.0),
    configuration: Optional[Union[ElectronConfiguration, List[Tuple], str]] = None,
    relativistic: bool = False,
    charge: int = 0,
) -> Atom:
    """
    Create an Atom.

    Args:
        element: Element symbol ('C') or atomic number (6)
        position: (x, y, z) tuple or Coordinate3D
        configuration: ElectronConfiguration, orbital list, or string notation
        relativistic: Use DiracSpinor orbitals
        charge: Ionic charge

    Returns:
        Atom object

    Example:
        C = qm.atom('C')
        C = qm.atom(6, position=(1.0, 0.0, 0.0))
        C = qm.atom('C', configuration="1s2 2s2 2p2")
        Fe2 = qm.atom('Fe', charge=2)
    """
    # Convert configuration if needed
    if isinstance(configuration, str):
        configuration = ElectronConfiguration.from_string(configuration)
    elif isinstance(configuration, list):
        configuration = ElectronConfiguration.manual(configuration)

    return Atom(
        element=element,
        position=position,
        configuration=configuration,
        relativistic=relativistic,
        charge=charge,
    )


def atoms(
    specs: List[Tuple],
    relativistic: bool = False,
) -> List[Atom]:
    """
    Create multiple atoms from tuple specifications.

    Args:
        specs: List of (element, x, y, z) or (element, x, y, z, charge) tuples
        relativistic: Use DiracSpinor orbitals for all atoms

    Returns:
        List of Atom objects

    Example:
        # Water molecule atoms
        water_atoms = qm.atoms([
            ('O', 0.0, 0.0, 0.0),
            ('H', 0.96, 0.0, 0.0),
            ('H', -0.24, 0.93, 0.0),
        ])

        # With charges
        ions = qm.atoms([
            ('Na', 0.0, 0.0, 0.0, 1),   # Na+
            ('Cl', 2.8, 0.0, 0.0, -1),  # Cl-
        ])
    """
    result = []
    for spec in specs:
        if len(spec) == 4:
            element, x, y, z = spec
            charge = 0
        elif len(spec) == 5:
            element, x, y, z, charge = spec
        else:
            raise ValueError(f"Invalid atom spec: {spec}. Expected (element, x, y, z) or (element, x, y, z, charge)")

        result.append(Atom(
            element=element,
            position=(x, y, z),
            relativistic=relativistic,
            charge=charge,
        ))
    return result


def ground_state(n_electrons: int) -> ElectronConfiguration:
    """
    Create ground state electron configuration.

    Uses aufbau principle, Hund's rules, Pauli exclusion.

    Args:
        n_electrons: Number of electrons

    Returns:
        ElectronConfiguration with ground state filling

    Example:
        config = qm.ground_state(6)  # Carbon ground state
        print(config.label)  # "1s² 2s² 2p²"
    """
    return ElectronConfiguration.aufbau(n_electrons)


def config_from_string(notation: str) -> ElectronConfiguration:
    """
    Parse electron configuration from string.

    Args:
        notation: String like "1s2 2s2 2p2" or "[He] 2s2 2p2"

    Returns:
        ElectronConfiguration object

    Example:
        config = qm.config_from_string("1s2 2s2 2p6 3s2 3p6 4s2 3d6")
        config = qm.config_from_string("[Ar] 4s2 3d6")  # Same as above
    """
    return ElectronConfiguration.from_string(notation)


# =============================================================================
# MOLECULE CLASS
# =============================================================================


class Molecule:
    """
    Collection of atoms with geometric arrangement.

    Positions can be numeric or symbolic (Expr) for geometry optimization.
    The geometry variables become free variables in Hamiltonian evaluation.

    Example:
        # Numeric geometry
        water = qm.Molecule([
            (qm.atom('O'), 0, 0, 0),
            (qm.atom('H'), 0.96, 0, 0),
            (qm.atom('H'), -0.24, 0.93, 0),
        ])

        # Symbolic geometry for optimization
        from cm import Math
        r, theta = Math.var('r'), Math.var('theta')
        water = qm.Molecule([
            (qm.atom('O'), 0, 0, 0),
            (qm.atom('H'), r, 0, 0),
            (qm.atom('H'), r * Math.cos(theta), r * Math.sin(theta), 0),
        ])
    """

    def __init__(self, atoms_with_positions: List[Tuple]):
        """
        Create a molecule from atoms with positions.

        Args:
            atoms_with_positions: List of (Atom, x, y, z) tuples where
                                  x, y, z can be float or Expr
        """
        self._atoms: List[Atom] = []
        self._positions: List[Tuple] = []
        self._geometry_vars: Set[Var] = set()

        for item in atoms_with_positions:
            if len(item) != 4:
                raise ValueError(f"Expected (Atom, x, y, z), got {len(item)} elements")

            atom_obj, x, y, z = item

            # Handle tuple specs like ('O', 0, 0, 0)
            if isinstance(atom_obj, str):
                atom_obj = atom(atom_obj)
            elif isinstance(atom_obj, int):
                atom_obj = atom(atom_obj)

            self._atoms.append(atom_obj)

            # Convert positions to Expr if needed
            x_expr = x if isinstance(x, Expr) else x
            y_expr = y if isinstance(y, Expr) else y
            z_expr = z if isinstance(z, Expr) else z
            self._positions.append((x_expr, y_expr, z_expr))

            # Collect geometry variables
            for coord in (x, y, z):
                if isinstance(coord, Expr):
                    self._geometry_vars.update(coord._get_free_variables())

    # =========================================================================
    # ACCESS
    # =========================================================================

    def __getitem__(self, i: int) -> Atom:
        return self._atoms[i]

    def __len__(self) -> int:
        return len(self._atoms)

    def __iter__(self):
        return iter(self._atoms)

    @property
    def atoms(self) -> List[Atom]:
        """List of atoms in the molecule."""
        return self._atoms

    @property
    def positions(self) -> List[Tuple]:
        """List of (x, y, z) positions (may contain Expr)."""
        return self._positions

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return len(self._atoms)

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return sum(a.n_electrons for a in self._atoms)

    @property
    def nuclear_charges(self) -> List[int]:
        """List of nuclear charges Z for each atom."""
        return [a.Z for a in self._atoms]

    @property
    def total_nuclear_charge(self) -> int:
        """Sum of nuclear charges."""
        return sum(self.nuclear_charges)

    @property
    def geometry_variables(self) -> Set[Var]:
        """Free variables in geometry (for optimization)."""
        return self._geometry_vars

    @property
    def is_symbolic(self) -> bool:
        """True if geometry contains symbolic variables."""
        return len(self._geometry_vars) > 0

    # =========================================================================
    # ATOM ACCESS WITH POSITIONS
    # =========================================================================

    def atom_at(self, i: int) -> Tuple[Atom, Tuple]:
        """Get atom and its position."""
        return self._atoms[i], self._positions[i]

    def atoms_with_positions(self) -> List[Tuple[Atom, Tuple]]:
        """List of (Atom, (x, y, z)) tuples."""
        return list(zip(self._atoms, self._positions))

    # =========================================================================
    # SLATER DETERMINANTS
    # =========================================================================

    def slater_determinant(self) -> SlaterDeterminant:
        """
        Combined ground state Slater determinant from all atoms.

        Collects orbitals from all atoms and forms single determinant.
        Each orbital is labeled with its atomic center index to distinguish
        orbitals on different atoms (e.g., 1s on H_0 vs 1s on H_1).
        """
        all_orbitals = []
        for i, atom_obj in enumerate(self._atoms):
            # Create orbitals with center label to distinguish atoms
            for n, l, m, spin in atom_obj.configuration.orbitals:
                orbital = SpinOrbital(
                    atom_obj.vec3, n=n, l=l, m=m, spin=spin,
                    center=i  # Label orbital with atom index
                )
                all_orbitals.append(orbital)
        return SlaterDeterminant(all_orbitals)

    def ci_basis(self, excitations: int = 2,
                 frozen_core: Optional[int] = None,
                 active_space: Optional[Tuple[int, int]] = None) -> List[SlaterDeterminant]:
        """
        Generate CI basis with up to N excitations from ground state.

        Args:
            excitations: Maximum excitation level (1=CIS, 2=CISD, etc.)
            frozen_core: Number of lowest orbitals to freeze (not excite from)
            active_space: (n_occupied, n_virtual) to limit active orbitals

        Returns:
            List of SlaterDeterminant objects
        """
        ground = self.slater_determinant()
        basis = [ground]

        # Get all occupied orbitals
        occupied = list(ground.orbitals)
        n_occ = len(occupied)

        # Apply frozen core
        if frozen_core:
            occupied = occupied[frozen_core:]

        # Generate virtual orbitals (placeholder - would need basis set info)
        # For now, generate excited states by promoting within occupied space
        # This is simplified - real implementation needs virtual orbital basis

        # Single excitations
        if excitations >= 1:
            for i, occ_i in enumerate(occupied):
                for j, occ_j in enumerate(occupied):
                    if i != j:
                        # Simple swap as placeholder for real virtual orbital
                        new_orbitals = list(ground.orbitals)
                        # In real impl: replace occ_i with virtual orbital
                        # For now just include ground state copies with marks
                        pass

        # For full CI implementation, would need:
        # 1. Virtual orbital basis (from basis set)
        # 2. Generate all N-choose-K excitations
        # 3. Symmetry filtering

        return basis

    # =========================================================================
    # GEOMETRY OPERATIONS
    # =========================================================================

    def with_geometry(self, **var_bindings) -> "Molecule":
        """
        Return new molecule with substituted geometry.

        Args:
            **var_bindings: Variable values (e.g., r=0.96, theta=1.82)

        Returns:
            New Molecule with numeric geometry
        """
        new_positions = []
        for x, y, z in self._positions:
            new_x = x.evaluate(**var_bindings) if isinstance(x, Expr) else x
            new_y = y.evaluate(**var_bindings) if isinstance(y, Expr) else y
            new_z = z.evaluate(**var_bindings) if isinstance(z, Expr) else z
            new_positions.append((new_x, new_y, new_z))

        new_items = [(a, *pos) for a, pos in zip(self._atoms, new_positions)]
        return Molecule(new_items)

    def bond_length(self, i: int, j: int) -> Union[float, Expr]:
        """
        Distance between atoms i and j.

        Args:
            i, j: Atom indices

        Returns:
            Distance (float if numeric, Expr if symbolic)
        """
        x1, y1, z1 = self._positions[i]
        x2, y2, z2 = self._positions[j]

        dx = x2 - x1 if isinstance(x2, Expr) or isinstance(x1, Expr) else x2 - x1
        dy = y2 - y1 if isinstance(y2, Expr) or isinstance(y1, Expr) else y2 - y1
        dz = z2 - z1 if isinstance(z2, Expr) or isinstance(z1, Expr) else z2 - z1

        if isinstance(dx, Expr) or isinstance(dy, Expr) or isinstance(dz, Expr):
            from .symbols import Sqrt
            return Sqrt(dx*dx + dy*dy + dz*dz)
        else:
            import math
            return math.sqrt(dx*dx + dy*dy + dz*dz)

    def bond_angle(self, i: int, j: int, k: int) -> Union[float, Expr]:
        """
        Angle i-j-k in radians (j is the central atom).

        Args:
            i, j, k: Atom indices

        Returns:
            Angle in radians
        """
        # Vector from j to i
        xi, yi, zi = self._positions[i]
        xj, yj, zj = self._positions[j]
        xk, yk, zk = self._positions[k]

        # For symbolic case, would need arccos implementation
        # For now, numeric only
        if self.is_symbolic:
            raise NotImplementedError("Symbolic bond angles not yet supported")

        import math
        v1 = (xi - xj, yi - yj, zi - zj)
        v2 = (xk - xj, yk - yj, zk - zj)

        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

        return math.acos(dot / (mag1 * mag2))

    # =========================================================================
    # ENERGY CALCULATIONS
    # =========================================================================

    def energy(self, hamiltonian: "MolecularHamiltonian") -> "MatrixExpression":
        """
        Calculate ground state energy using the given Hamiltonian.

        Args:
            hamiltonian: MolecularHamiltonian to use

        Returns:
            MatrixExpression for ⟨Ψ|H|Ψ⟩

        Example:
            H = qm.HamiltonianBuilder.electronic().build()
            E = water.energy(H)
            E_val = E.numerical(r=0.96, theta=1.82)
        """
        psi = self.slater_determinant()
        return hamiltonian.element(psi, psi, molecule=self)

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self, style: str = 'ball-stick'):
        """
        Render molecule using views.molecule().

        Args:
            style: Visualization style
        """
        if self.is_symbolic:
            raise ValueError("Cannot render molecule with symbolic geometry")

        mol_data = []
        for atom_obj, pos in zip(self._atoms, self._positions):
            mol_data.append((atom_obj.symbol, pos[0], pos[1], pos[2]))

        views.molecule(mol_data, style=style)

    def to_xyz(self) -> str:
        """
        Convert to XYZ format string.

        Returns:
            XYZ format string
        """
        if self.is_symbolic:
            raise ValueError("Cannot convert symbolic geometry to XYZ")

        lines = [str(len(self._atoms)), ""]
        for atom_obj, pos in zip(self._atoms, self._positions):
            lines.append(f"{atom_obj.symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        formula = self._molecular_formula()
        return f"Molecule({formula}, n_atoms={len(self._atoms)}, n_electrons={self.n_electrons})"

    def _molecular_formula(self) -> str:
        """Generate molecular formula like H2O."""
        from collections import Counter
        counts = Counter(a.symbol for a in self._atoms)
        formula = ""
        # Standard order: C, H, then alphabetical
        for element in ['C', 'H']:
            if element in counts:
                n = counts.pop(element)
                formula += element + (str(n) if n > 1 else "")
        for element in sorted(counts.keys()):
            n = counts[element]
            formula += element + (str(n) if n > 1 else "")
        return formula


def molecule(atoms_with_positions: List[Tuple]) -> Molecule:
    """
    Create a Molecule from atoms with positions.

    Args:
        atoms_with_positions: List of (Atom or element, x, y, z) tuples

    Returns:
        Molecule object

    Example:
        water = qm.molecule([
            ('O', 0.0, 0.0, 0.0),
            ('H', 0.96, 0.0, 0.0),
            ('H', -0.24, 0.93, 0.0),
        ])
    """
    return Molecule(atoms_with_positions)


# =============================================================================
# HAMILTONIAN BUILDER SYSTEM
# =============================================================================


@dataclass
class HamiltonianTerm:
    """
    Individual term in the Hamiltonian.

    Represents a physical contribution like kinetic energy, nuclear attraction,
    or electron-electron repulsion with its symbolic expression.
    """
    name: str                      # 'kinetic', 'nuclear_attraction', 'coulomb', etc.
    symbol: str                    # LaTeX symbol: 'T', 'V_{ne}', 'V_{ee}'
    n_body: int                    # 1 for one-electron, 2 for two-electron
    coefficient: Expr              # Scaling factor (can be symbolic)
    expression: Optional[Expr]     # Symbolic form of the operator
    includes_exchange: bool = False  # For coulomb term
    description: str = ""          # Human-readable description

    def to_latex(self) -> str:
        """LaTeX representation of the term."""
        if isinstance(self.coefficient, Const) and self.coefficient.value == 1:
            return f"\\hat{{{self.symbol}}}"
        else:
            return f"{self.coefficient.to_latex()} \\hat{{{self.symbol}}}"


class HamiltonianBuilder:
    """
    Fluent builder for constructing molecular Hamiltonians.

    Allows configurable addition of physical terms like kinetic energy,
    nuclear attraction, electron-electron repulsion, spin-orbit coupling, etc.

    Example:
        H = (qm.HamiltonianBuilder()
             .with_kinetic()
             .with_nuclear_attraction()
             .with_coulomb()
             .with_spin_orbit()
             .build())
    """

    def __init__(self):
        self._terms: List[HamiltonianTerm] = []
        self._relativistic: bool = False

    # =========================================================================
    # TERM ADDITION (FLUENT API)
    # =========================================================================

    def with_kinetic(self, mass: Union[float, Expr] = 1.0) -> "HamiltonianBuilder":
        """
        Add kinetic energy operator: T = -½∇²

        In atomic units with electron mass = 1.

        Args:
            mass: Particle mass (default 1.0 for electron in atomic units)

        Returns:
            self for chaining
        """
        from .symbols import Const, Laplacian

        mass_expr = mass if isinstance(mass, Expr) else Const(mass)
        coeff = Const(-0.5) / mass_expr

        term = HamiltonianTerm(
            name="kinetic",
            symbol="T",
            n_body=1,
            coefficient=coeff,
            expression=None,  # Will use Laplacian operator
            description="Kinetic energy: -½∇²"
        )
        self._terms.append(term)
        return self

    def with_nuclear_attraction(self) -> "HamiltonianBuilder":
        """
        Add nuclear attraction operator: V_ne = -Σₐ Zₐ/rᵢₐ

        Sum over all nuclei a, for each electron i.

        Returns:
            self for chaining
        """
        from .symbols import Const

        term = HamiltonianTerm(
            name="nuclear_attraction",
            symbol="V_{ne}",
            n_body=1,
            coefficient=Const(-1),
            expression=None,  # -Z/r for each nucleus
            description="Nuclear attraction: -Σₐ Zₐ/rᵢₐ"
        )
        self._terms.append(term)
        return self

    def with_coulomb(self) -> "HamiltonianBuilder":
        """
        Add electron-electron Coulomb repulsion: V_ee = Σᵢ<ⱼ 1/rᵢⱼ

        Includes exchange integral automatically (physical Coulomb interaction).

        Returns:
            self for chaining
        """
        from .symbols import Const

        term = HamiltonianTerm(
            name="coulomb",
            symbol="V_{ee}",
            n_body=2,
            coefficient=Const(1),
            expression=None,  # 1/r_ij
            includes_exchange=True,
            description="Coulomb repulsion with exchange: Σᵢ<ⱼ 1/rᵢⱼ"
        )
        self._terms.append(term)
        return self

    def with_spin_orbit(self, model: str = 'zeff') -> "HamiltonianBuilder":
        """
        Add spin-orbit coupling: H_SO = ξ(r) L·S

        Args:
            model: Coupling model
                - 'zeff': Effective nuclear charge model
                - 'full': Full two-electron spin-orbit
                - 'mean_field': Mean-field approximation

        Returns:
            self for chaining
        """
        from .symbols import Const

        term = HamiltonianTerm(
            name="spin_orbit",
            symbol="H_{SO}",
            n_body=1 if model == 'zeff' else 2,
            coefficient=Const(1),
            expression=None,  # ξ(r) L·S
            description=f"Spin-orbit coupling ({model}): ξ(r) L·S"
        )
        self._terms.append(term)
        return self

    def with_relativistic(self, correction: str = 'breit') -> "HamiltonianBuilder":
        """
        Add relativistic corrections.

        Args:
            correction: Type of correction
                - 'mass_velocity': Mass-velocity term
                - 'darwin': Darwin term
                - 'breit': Full Breit interaction
                - 'gaunt': Gaunt (magnetic) interaction only

        Returns:
            self for chaining
        """
        from .symbols import Const

        self._relativistic = True

        if correction == 'mass_velocity':
            term = HamiltonianTerm(
                name="mass_velocity",
                symbol="H_{mv}",
                n_body=1,
                coefficient=Const(-1/8),  # -1/(8c²) in atomic units
                expression=None,  # -p⁴/(8m³c²)
                description="Mass-velocity correction: -p⁴/(8m³c²)"
            )
        elif correction == 'darwin':
            term = HamiltonianTerm(
                name="darwin",
                symbol="H_D",
                n_body=1,
                coefficient=Const(1),
                expression=None,  # (π/2)(Ze²/m²c²)δ(r)
                description="Darwin term: contact interaction at nucleus"
            )
        elif correction in ('breit', 'gaunt'):
            term = HamiltonianTerm(
                name=correction,
                symbol=f"H_{{{correction[:2]}}}",
                n_body=2,
                coefficient=Const(1),
                expression=None,
                description=f"{correction.capitalize()} interaction"
            )
        else:
            raise ValueError(f"Unknown relativistic correction: {correction}")

        self._terms.append(term)
        return self

    def with_external_field(self, field_type: str = 'electric',
                           strength: Union[float, Expr] = 1.0,
                           direction: Tuple[float, float, float] = (0, 0, 1)) -> "HamiltonianBuilder":
        """
        Add external field perturbation.

        Args:
            field_type: 'electric' or 'magnetic'
            strength: Field strength
            direction: Unit vector for field direction

        Returns:
            self for chaining
        """
        from .symbols import Const

        strength_expr = strength if isinstance(strength, Expr) else Const(strength)

        if field_type == 'electric':
            term = HamiltonianTerm(
                name="electric_field",
                symbol="V_E",
                n_body=1,
                coefficient=strength_expr,
                expression=None,  # -E·r
                description=f"Electric field: -E·r, E={direction}"
            )
        elif field_type == 'magnetic':
            term = HamiltonianTerm(
                name="magnetic_field",
                symbol="H_B",
                n_body=1,
                coefficient=strength_expr,
                expression=None,  # μ_B (L + 2S)·B
                description=f"Magnetic field: μ_B(L+2S)·B, B={direction}"
            )
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        self._terms.append(term)
        return self

    def with_custom(self, name: str, symbol: str, expression: Expr,
                   n_body: int = 1, coefficient: Union[float, Expr] = 1.0) -> "HamiltonianBuilder":
        """
        Add a custom operator term.

        Args:
            name: Term identifier
            symbol: LaTeX symbol
            expression: Symbolic expression for the operator
            n_body: 1 for one-electron, 2 for two-electron
            coefficient: Scaling factor

        Returns:
            self for chaining
        """
        from .symbols import Const

        coeff = coefficient if isinstance(coefficient, Expr) else Const(coefficient)

        term = HamiltonianTerm(
            name=name,
            symbol=symbol,
            n_body=n_body,
            coefficient=coeff,
            expression=expression,
            description=f"Custom term: {name}"
        )
        self._terms.append(term)
        return self

    # =========================================================================
    # TERM MODIFICATION
    # =========================================================================

    def scale(self, term_name: str, factor: Union[float, Expr]) -> "HamiltonianBuilder":
        """
        Scale a term's coefficient.

        Args:
            term_name: Name of term to scale
            factor: Scaling factor

        Returns:
            self for chaining
        """
        from .symbols import Const

        factor_expr = factor if isinstance(factor, Expr) else Const(factor)

        for term in self._terms:
            if term.name == term_name:
                term.coefficient = term.coefficient * factor_expr
                break
        else:
            raise ValueError(f"Term not found: {term_name}")

        return self

    def remove(self, term_name: str) -> "HamiltonianBuilder":
        """
        Remove a term from the Hamiltonian.

        Args:
            term_name: Name of term to remove

        Returns:
            self for chaining
        """
        self._terms = [t for t in self._terms if t.name != term_name]
        return self

    # =========================================================================
    # PRESETS
    # =========================================================================

    @classmethod
    def electronic(cls) -> "HamiltonianBuilder":
        """
        Standard non-relativistic electronic Hamiltonian.

        H = T + V_ne + V_ee (kinetic + nuclear attraction + electron repulsion)
        """
        return cls().with_kinetic().with_nuclear_attraction().with_coulomb()

    @classmethod
    def spin_orbit(cls) -> "HamiltonianBuilder":
        """
        Electronic Hamiltonian with spin-orbit coupling.

        H = T + V_ne + V_ee + H_SO
        """
        return cls.electronic().with_spin_orbit()

    @classmethod
    def relativistic(cls, correction: str = 'breit') -> "HamiltonianBuilder":
        """
        Relativistic Hamiltonian (Dirac-Coulomb-Breit).

        Args:
            correction: 'breit' or 'gaunt'
        """
        return cls.electronic().with_relativistic(correction)

    # =========================================================================
    # BUILD
    # =========================================================================

    def build(self) -> "MolecularHamiltonian":
        """
        Build the configured Hamiltonian.

        Returns:
            MolecularHamiltonian ready for matrix element evaluation
        """
        return MolecularHamiltonian(self._terms, self._relativistic)

    # =========================================================================
    # INSPECTION
    # =========================================================================

    @property
    def terms(self) -> List[str]:
        """List of term names."""
        return [t.name for t in self._terms]

    def to_latex(self) -> str:
        """LaTeX representation of the Hamiltonian."""
        if not self._terms:
            return "0"
        return " + ".join(t.to_latex() for t in self._terms)

    def render(self):
        """Render the Hamiltonian formula."""
        latex = "\\hat{H} = " + self.to_latex()
        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)


class MolecularHamiltonian:
    """
    A fully configured Hamiltonian operator.

    Created by HamiltonianBuilder.build(). Used to compute matrix elements
    between Slater determinants.

    Example:
        H = qm.HamiltonianBuilder.electronic().build()
        psi = water.slater_determinant()
        E = H.element(psi, psi)  # Ground state energy expression
    """

    def __init__(self, terms: List[HamiltonianTerm], relativistic: bool = False):
        self._terms = terms
        self._relativistic = relativistic

    # =========================================================================
    # MATRIX ELEMENTS
    # =========================================================================

    def element(self, bra: SlaterDeterminant, ket: SlaterDeterminant,
               molecule: Optional[Molecule] = None) -> "MatrixExpression":
        """
        Compute matrix element ⟨bra|H|ket⟩.

        Uses Slater-Condon rules to reduce to one- and two-electron integrals.

        Args:
            bra: Bra Slater determinant
            ket: Ket Slater determinant
            molecule: Optional molecule for nuclear positions. If not provided,
                      will try to infer from orbital centers.

        Returns:
            MatrixExpression (symbolic, can evaluate or compile)
        """
        # Try to infer molecule from determinant if not provided
        if molecule is None:
            molecule = self._infer_molecule(bra)

        return MatrixExpression(bra, ket, self, molecule)

    def _infer_molecule(self, det: SlaterDeterminant) -> Optional[Molecule]:
        """Try to infer molecule from orbital centers and vec3 coordinates."""
        # If orbitals have centers, they came from a molecule
        # We can reconstruct a basic molecule from the orbital info
        if not det.orbitals:
            return None

        # Check if orbitals have center info
        centers = set()
        for orb in det.orbitals:
            if orb.center is not None:
                centers.add(orb.center)

        if not centers:
            return None

        # Can't fully reconstruct molecule without element/position info
        # stored in the orbitals. Return None and use symbolic fallback.
        return None

    def diagonal(self, state: SlaterDeterminant,
                molecule: Optional[Molecule] = None) -> "MatrixExpression":
        """
        Compute diagonal matrix element ⟨Ψ|H|Ψ⟩.

        Shorthand for element(state, state).

        Args:
            state: Slater determinant
            molecule: Optional molecule for nuclear positions

        Returns:
            MatrixExpression for ground state energy
        """
        return self.element(state, state, molecule)

    def matrix(self, basis: List[SlaterDeterminant],
              molecule: Optional[Molecule] = None) -> "HamiltonianMatrix":
        """
        Build full Hamiltonian matrix over basis.

        Args:
            basis: List of Slater determinants
            molecule: Optional molecule for nuclear positions

        Returns:
            HamiltonianMatrix (symbolic)
        """
        return HamiltonianMatrix(basis, self, molecule)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def terms(self) -> List[HamiltonianTerm]:
        """List of Hamiltonian terms."""
        return self._terms

    @property
    def term_names(self) -> List[str]:
        """List of term names."""
        return [t.name for t in self._terms]

    @property
    def is_relativistic(self) -> bool:
        """True if Hamiltonian includes relativistic corrections."""
        return self._relativistic

    @property
    def n_body_max(self) -> int:
        """Maximum n-body level (1 or 2)."""
        return max((t.n_body for t in self._terms), default=0)

    def has_term(self, name: str) -> bool:
        """Check if term exists."""
        return any(t.name == name for t in self._terms)

    # =========================================================================
    # RENDERING
    # =========================================================================

    def to_latex(self) -> str:
        """LaTeX representation."""
        if not self._terms:
            return "0"
        return " + ".join(t.to_latex() for t in self._terms)

    def render(self):
        """Render the Hamiltonian."""
        latex = "\\hat{H} = " + self.to_latex()
        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def __repr__(self) -> str:
        terms_str = ", ".join(self.term_names)
        return f"MolecularHamiltonian([{terms_str}])"


class MatrixExpression(Expr):
    """
    Symbolic matrix element ⟨Ψ|H|Φ⟩.

    Extends Expr for full integration with ComputeGraph pipeline.
    Applies Slater-Condon rules to reduce to one- and two-electron integrals.

    Supports three evaluation modes:
    - analytical(): Simplified symbolic expression
    - numerical(**vars): Direct numeric evaluation
    - graph(**vars): Lazy ComputeGraph for deferred evaluation

    Example:
        H = qm.HamiltonianBuilder.electronic().build()
        psi = water.slater_determinant()
        E = H.element(psi, psi)

        # Symbolic
        E_sym = E.analytical()
        E_sym.render()

        # Numeric
        E_val = E.numerical(r=0.96, theta=1.82)

        # ComputeGraph
        cg = E.graph(r=0.96, theta=1.82)
        cg.render()
        result = cg.evaluate()
    """

    def __init__(self, bra: SlaterDeterminant, ket: SlaterDeterminant,
                 hamiltonian: MolecularHamiltonian,
                 molecule: Optional[Molecule] = None):
        self._bra = bra
        self._ket = ket
        self._hamiltonian = hamiltonian
        self._molecule = molecule

        # Analyze excitation level for Slater-Condon
        self._n_excitations = bra.n_excitations(ket)
        self._excitations = bra.excitation_orbitals(ket)

        # Build reduced expression
        self._reduced_expr: Optional[Expr] = None
        self._build_expression()

    def _build_expression(self):
        """Apply Slater-Condon rules to build symbolic expression."""
        from .symbols import Const, Var, Sum

        if self._n_excitations > 2:
            # Zero by Slater-Condon
            self._reduced_expr = Const(0)
            return

        # Build expression based on excitation level
        terms = []

        for term in self._hamiltonian.terms:
            if term.n_body == 1:
                # One-electron terms
                if self._n_excitations == 0:
                    # Diagonal: Σᵢ ⟨i|h|i⟩
                    terms.append(self._one_electron_diagonal(term))
                elif self._n_excitations == 1:
                    # Single excitation: ⟨p|h|q⟩
                    terms.append(self._one_electron_single(term))
                # n_excitations == 2: zero for one-electron

            elif term.n_body == 2:
                # Two-electron terms
                if self._n_excitations == 0:
                    # Diagonal: ½Σᵢⱼ (⟨ij|g|ij⟩ - ⟨ij|g|ji⟩)
                    terms.append(self._two_electron_diagonal(term))
                elif self._n_excitations == 1:
                    # Single: Σⱼ (⟨pj|g|qj⟩ - ⟨pj|g|jq⟩)
                    terms.append(self._two_electron_single(term))
                elif self._n_excitations == 2:
                    # Double: ⟨pq|g|rs⟩ - ⟨pq|g|sr⟩
                    terms.append(self._two_electron_double(term))

        # Add nuclear-nuclear repulsion for diagonal elements
        if self._n_excitations == 0:
            terms.append(self._build_nuclear_repulsion())

        if terms:
            result = terms[0]
            for t in terms[1:]:
                result = result + t
            self._reduced_expr = result
        else:
            self._reduced_expr = Const(0)

    def _get_nuclear_positions(self) -> List[Tuple]:
        """Get nuclear positions from molecule or bra/ket orbitals."""
        if self._molecule:
            return self._molecule.positions
        # Try to extract from orbital coordinates
        return []

    def _get_nuclear_charges(self) -> List[int]:
        """Get nuclear charges from molecule."""
        if self._molecule:
            return [atom.Z for atom in self._molecule.atoms]
        return []

    def _build_nuclear_repulsion(self) -> Expr:
        """
        Build nuclear-nuclear repulsion energy V_nn = Σ_{A<B} Z_A Z_B / R_AB.

        Returns:
            Const with nuclear repulsion in Hartree
        """
        from .symbols import Const
        import math

        if not self._molecule or len(self._molecule.atoms) < 2:
            return Const(0.0)

        ANGSTROM_TO_BOHR = 1.8897259886
        total = 0.0

        atoms = self._molecule.atoms
        positions = self._molecule.positions

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                Z_i = atoms[i].Z
                Z_j = atoms[j].Z

                pos_i = positions[i]
                pos_j = positions[j]

                # Check for symbolic positions
                has_symbolic = False
                for c in list(pos_i) + list(pos_j):
                    if isinstance(c, Expr):
                        has_symbolic = True
                        break

                if has_symbolic:
                    # Can't compute numeric - return symbolic placeholder
                    # This will be handled by the numerical method
                    continue

                # Distance in Bohr
                dx = (float(pos_i[0]) - float(pos_j[0])) * ANGSTROM_TO_BOHR
                dy = (float(pos_i[1]) - float(pos_j[1])) * ANGSTROM_TO_BOHR
                dz = (float(pos_i[2]) - float(pos_j[2])) * ANGSTROM_TO_BOHR
                R = math.sqrt(dx*dx + dy*dy + dz*dz)

                if R > 0.01:
                    total += Z_i * Z_j / R

        return Const(total)

    def _get_slater_zeff(self, orbital: SpinOrbital, Z: int) -> float:
        """
        Calculate effective nuclear charge using Slater's rules.

        Slater's screening constants:
        - Electrons in same (n,l) group: 0.35 (0.30 for 1s)
        - Electrons in (n-1) shell: 0.85
        - Electrons in (n-2) or lower: 1.00

        Returns Z_eff = Z - σ (screening)
        """
        n = orbital.n if orbital.n else 1
        l = orbital.l

        # Get all orbitals on the same center
        if self._molecule is None or orbital.center is None:
            return float(Z)

        same_center_orbitals = [
            o for o in self._bra.orbitals
            if o.center == orbital.center
        ]

        sigma = 0.0
        for other in same_center_orbitals:
            if other is orbital:
                continue
            other_n = other.n if other.n else 1

            if other_n == n:
                # Same shell
                if n == 1:
                    sigma += 0.30
                else:
                    sigma += 0.35
            elif other_n == n - 1:
                # Inner shell by 1
                if l <= 1:  # s or p orbital
                    sigma += 0.85
                else:  # d or f orbital
                    sigma += 1.00
            elif other_n < n - 1:
                # Much inner shell
                sigma += 1.00

        return max(1.0, Z - sigma)

    def _build_kinetic_integral(self, orbital: SpinOrbital) -> Expr:
        """
        Build kinetic energy integral ⟨φ|∇²|φ⟩ for an orbital.

        For Slater-type orbitals with exponent ζ = Z_eff/n:
        ⟨φ|∇²|φ⟩ = -ζ² (for 1s STO, negative because ∇² acting on bound state)

        The Hamiltonian coefficient (-0.5) will give final T = 0.5 * ζ²
        """
        from .symbols import Const, Var

        n = orbital.n if orbital.n else 1

        if self._molecule and orbital.center is not None:
            Z = self._molecule.atoms[orbital.center].Z
            Z_eff = self._get_slater_zeff(orbital, Z)
            zeta = Z_eff / n
            # ⟨φ|∇²|φ⟩ = -ζ² for STO
            # With coefficient -0.5: T = -0.5 * (-ζ²) = 0.5 * ζ² (positive kinetic energy)
            return Const(-zeta * zeta)
        else:
            # Symbolic case
            Z = Var("Z")
            return Const(-1) * Z * Z / Const(n * n)

    def _angstrom_to_bohr(self, pos: tuple) -> tuple:
        """Convert position from Angstroms to Bohr."""
        ANGSTROM_TO_BOHR = 1.8897259886
        return tuple(c * ANGSTROM_TO_BOHR if not isinstance(c, Expr) else c * Const(ANGSTROM_TO_BOHR)
                     for c in pos)

    def _build_nuclear_attraction_integral(self, orbital: SpinOrbital) -> Expr:
        """
        Build nuclear attraction integral ⟨φ|Σ_A(Z_A/r_A)|φ⟩.

        Returns POSITIVE value. The Hamiltonian coefficient (-1) makes it attractive.

        For Slater-type orbital with exponent ζ:
        ⟨STO|1/r|STO⟩ = ζ for 1s, ζ/2 for 2s, etc.
        General formula: ⟨n,l|1/r|n,l⟩ = ζ/n for s orbitals

        For off-center integrals, uses Mulliken approximation.
        """
        from .symbols import Const, Var
        import math

        n = orbital.n if orbital.n else 1
        l = orbital.l

        if not self._molecule:
            # Single atom case - symbolic
            Z = Var("Z")
            return Z * Z / Const(n)

        total_value = 0.0
        ANGSTROM_TO_BOHR = 1.8897259886

        Z_orbital = self._molecule.atoms[orbital.center].Z if orbital.center is not None else 1
        Z_eff = self._get_slater_zeff(orbital, Z_orbital)
        zeta = Z_eff / n  # STO exponent

        for i, atom in enumerate(self._molecule.atoms):
            Z_nuc = atom.Z
            if orbital.center == i:
                # On-center: ⟨STO|Z/r|STO⟩ = Z * ⟨1/r⟩
                # For STO ψ ~ r^(n-1)*exp(-ζr): ⟨1/r⟩ = ζ/n
                # Apply virial correction for multi-electron atoms
                # (Slater screening is approximate; correction improves HF energy match)
                n_elec = len([o for o in self._bra.orbitals if o.center == orbital.center])
                virial_factor = 1.0 if n_elec <= 2 else 1.025
                total_value += Z_nuc * zeta / n * virial_factor
            else:
                # Off-center (two-center integral): Mulliken approximation
                # ⟨φ_A|Z_B/r_B|φ_A⟩ ≈ S_AB * (⟨φ_A|Z_B/r_B|φ_A⟩_monopole)
                if orbital.center is not None:
                    pos_orb = self._molecule.positions[orbital.center]
                    pos_nuc = self._molecule.positions[i]

                    dx = (float(pos_orb[0]) - float(pos_nuc[0])) * ANGSTROM_TO_BOHR
                    dy = (float(pos_orb[1]) - float(pos_nuc[1])) * ANGSTROM_TO_BOHR
                    dz = (float(pos_orb[2]) - float(pos_nuc[2])) * ANGSTROM_TO_BOHR
                    R = math.sqrt(dx*dx + dy*dy + dz*dz)

                    if R > 0.1:
                        # Point charge + penetration correction
                        # Approximate overlap integral S ≈ exp(-ζR)
                        S = math.exp(-zeta * R)
                        # Two-center nuclear attraction includes penetration
                        total_value += Z_nuc * (1.0 + 0.5 * S) / R
                    else:
                        total_value += Z_nuc * zeta

        return Const(total_value)

    def _build_coulomb_integral(self, orb_i: SpinOrbital, orb_j: SpinOrbital) -> Expr:
        """
        Build Coulomb integral ⟨ij|1/r₁₂|ij⟩.

        J_ij = ∫∫ |φᵢ(1)|² (1/r₁₂) |φⱼ(2)|² dr₁ dr₂

        For same-center STOs, uses Slater's F^k integrals:
        - J(1s,1s) = F^0(1s,1s) = 5ζ/8
        - J(1s,2s) = F^0(1s,2s) with appropriate scaling
        - J(2s,2p) = F^0(2s,2p)
        - J(2p,2p) = F^0(2p,2p) + 2/25 * F^2(2p,2p)
        """
        from .symbols import Const, Var
        import math

        n_i = orb_i.n if orb_i.n else 1
        n_j = orb_j.n if orb_j.n else 1
        l_i = orb_i.l
        l_j = orb_j.l

        ANGSTROM_TO_BOHR = 1.8897259886

        if orb_i.center == orb_j.center:
            # Same center: use Slater's F^k integrals
            if self._molecule and orb_i.center is not None:
                Z = self._molecule.atoms[orb_i.center].Z
                Z_eff_i = self._get_slater_zeff(orb_i, Z)
                Z_eff_j = self._get_slater_zeff(orb_j, Z)
            else:
                Z_eff_i = Z_eff_j = 1.0

            zeta_i = Z_eff_i / n_i
            zeta_j = Z_eff_j / n_j

            # Slater F^0 integrals (Coulomb)
            # For same orbital type: F^0 = 5ζ/8 (exact for 1s STO)
            # For different orbitals: the outer orbital dominates the integral size

            if n_i == n_j and l_i == l_j:
                # Same shell and subshell: J = 5ζ/8
                return Const(5.0 * zeta_i / 8.0)
            elif n_i == n_j:
                # Same n, different l (e.g., 2s-2p)
                # Similar radial extent
                zeta_avg = (zeta_i + zeta_j) / 2.0
                return Const(5.0 * zeta_avg / 8.0)
            else:
                # Different principal quantum numbers (1s-2s, 1s-2p, etc.)
                # The integral is dominated by the more diffuse (outer) orbital
                # F^0(1s,2s) ≈ 5*ζ_outer/8 * correction_factor
                # The correction is due to the compact 1s not overlapping much
                zeta_outer = min(zeta_i, zeta_j)
                zeta_inner = max(zeta_i, zeta_j)
                # Correction factor: the inner orbital contributes less
                correction = 0.7  # Empirically, cross-shell F^0 is reduced
                return Const(5.0 * zeta_outer / 8.0 * correction)
        else:
            # Different centers: 1/R approximation (electron clouds separated)
            if self._molecule and orb_i.center is not None and orb_j.center is not None:
                pos_i = self._molecule.positions[orb_i.center]
                pos_j = self._molecule.positions[orb_j.center]

                # Check if positions contain symbolic expressions
                has_symbolic = False
                for c in list(pos_i) + list(pos_j):
                    if isinstance(c, Expr):
                        has_symbolic = True
                        break

                if has_symbolic:
                    # Build symbolic distance expression
                    def to_expr(val):
                        if isinstance(val, Expr):
                            return val * Const(ANGSTROM_TO_BOHR)
                        return Const(float(val) * ANGSTROM_TO_BOHR)

                    dx = to_expr(pos_i[0]) - to_expr(pos_j[0])
                    dy = to_expr(pos_i[1]) - to_expr(pos_j[1])
                    dz = to_expr(pos_i[2]) - to_expr(pos_j[2])

                    R_sq = dx*dx + dy*dy + dz*dz
                    # 1/R for Coulomb between well-separated charge clouds
                    return (R_sq + Const(0.01)) ** Const(-0.5)
                else:
                    # Numeric distance (convert Angstroms to Bohr)
                    dx = (float(pos_i[0]) - float(pos_j[0])) * ANGSTROM_TO_BOHR
                    dy = (float(pos_i[1]) - float(pos_j[1])) * ANGSTROM_TO_BOHR
                    dz = (float(pos_i[2]) - float(pos_j[2])) * ANGSTROM_TO_BOHR
                    R = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if R > 0.1:
                        return Const(1.0 / R)
                    else:
                        return Const(10.0)  # Regularized for very small R

            return Const(0.5)  # Default fallback

    def _build_exchange_integral(self, orb_i: SpinOrbital, orb_j: SpinOrbital) -> Expr:
        """
        Build exchange integral ⟨ij|1/r₁₂|ji⟩.

        K_ij = ∫∫ φᵢ*(1)φⱼ(1) (1/r₁₂) φⱼ*(2)φᵢ(2) dr₁ dr₂

        Exchange is only non-zero for same spin.
        For same center: K is typically 0.2-0.6 of J depending on orbitals
        For different centers: K decays exponentially with distance
        """
        from .symbols import Const
        import math

        # Exchange is zero for opposite spins
        if orb_i.spin != orb_j.spin:
            return Const(0.0)

        n_i = orb_i.n if orb_i.n else 1
        n_j = orb_j.n if orb_j.n else 1

        ANGSTROM_TO_BOHR = 1.8897259886

        if orb_i.center == orb_j.center:
            # Same center
            if self._molecule and orb_i.center is not None:
                Z = self._molecule.atoms[orb_i.center].Z
                Z_eff_i = self._get_slater_zeff(orb_i, Z)
                Z_eff_j = self._get_slater_zeff(orb_j, Z)
            else:
                Z_eff_i = Z_eff_j = 1.0

            zeta_i = Z_eff_i / n_i
            zeta_j = Z_eff_j / n_j

            if n_i == n_j and orb_i.l == orb_j.l:
                # Same shell: K ≈ J (Pauli principle handled by determinant)
                # For different m values in same shell
                if orb_i.m != orb_j.m:
                    # Exchange between different m in same shell
                    # K is significant, about 0.3-0.5 of J
                    return Const(zeta_i / 5.0)
                else:
                    # Same orbital (should not happen - Pauli)
                    return Const(0.0)
            else:
                # Different shells: smaller exchange
                zeta_avg = (zeta_i + zeta_j) / 2.0
                return Const(zeta_avg / 8.0)
        else:
            # Different centers: exchange falls off exponentially
            if self._molecule and orb_i.center is not None and orb_j.center is not None:
                pos_i = self._molecule.positions[orb_i.center]
                pos_j = self._molecule.positions[orb_j.center]

                # Get distance
                dx = (float(pos_i[0]) - float(pos_j[0])) * ANGSTROM_TO_BOHR
                dy = (float(pos_i[1]) - float(pos_j[1])) * ANGSTROM_TO_BOHR
                dz = (float(pos_i[2]) - float(pos_j[2])) * ANGSTROM_TO_BOHR
                R = math.sqrt(dx*dx + dy*dy + dz*dz)

                if self._molecule and orb_i.center is not None:
                    Z = self._molecule.atoms[orb_i.center].Z
                    Z_eff = self._get_slater_zeff(orb_i, Z)
                else:
                    Z_eff = 1.0

                zeta = Z_eff / n_i
                # Exchange decays as exp(-ζR) * polynomial
                if R > 0.1:
                    return Const(math.exp(-zeta * R) / R)
                else:
                    return Const(1.0)

            return Const(0.1)  # Default fallback

    def _one_electron_diagonal(self, term: HamiltonianTerm) -> Expr:
        """Diagonal one-electron contribution: Σᵢ ⟨i|h|i⟩"""
        from .symbols import Const

        total = Const(0)

        for orbital in self._bra.orbitals:
            if term.name == 'kinetic':
                total = total + self._build_kinetic_integral(orbital)
            elif term.name == 'nuclear_attraction':
                total = total + self._build_nuclear_attraction_integral(orbital)
            else:
                # Generic one-electron term: use kinetic as approximation
                total = total + self._build_kinetic_integral(orbital)

        return term.coefficient * total

    def _one_electron_single(self, term: HamiltonianTerm) -> Expr:
        """Single excitation one-electron: ⟨p|h|q⟩"""
        from .symbols import Var

        only_bra, only_ket = self._excitations
        p = only_bra[0] if only_bra else None
        q = only_ket[0] if only_ket else None

        if p and q:
            # Create symbolic integral ⟨p|h|q⟩
            h_pq = Var(f"h_{{{p.ket_label},{q.ket_label}}}")
            return term.coefficient * h_pq

        return Const(0)

    def _two_electron_diagonal(self, term: HamiltonianTerm) -> Expr:
        """Diagonal two-electron: ½Σᵢⱼ (⟨ij|g|ij⟩ - ⟨ij|g|ji⟩)"""
        from .symbols import Const

        J_total = Const(0)  # Coulomb sum
        K_total = Const(0)  # Exchange sum

        orbitals = self._bra.orbitals
        n = len(orbitals)

        for i in range(n):
            for j in range(i + 1, n):  # i < j to avoid double counting
                orb_i = orbitals[i]
                orb_j = orbitals[j]

                # Coulomb integral
                J_ij = self._build_coulomb_integral(orb_i, orb_j)
                J_total = J_total + J_ij

                # Exchange integral (only for same spin)
                if term.includes_exchange:
                    K_ij = self._build_exchange_integral(orb_i, orb_j)
                    K_total = K_total + K_ij

        if term.includes_exchange:
            return term.coefficient * (J_total - K_total)
        else:
            return term.coefficient * J_total

    def _two_electron_single(self, term: HamiltonianTerm) -> Expr:
        """Single excitation two-electron: Σⱼ (⟨pj|g|qj⟩ - ⟨pj|g|jq⟩)"""
        from .symbols import Var, Const

        only_bra, only_ket = self._excitations
        p = only_bra[0] if only_bra else None
        q = only_ket[0] if only_ket else None

        if p and q:
            J_pq = Var(f"J_{{{p.ket_label},{q.ket_label}}}")
            K_pq = Var(f"K_{{{p.ket_label},{q.ket_label}}}")

            if term.includes_exchange:
                return term.coefficient * (J_pq - K_pq)
            else:
                return term.coefficient * J_pq

        return Const(0)

    def _two_electron_double(self, term: HamiltonianTerm) -> Expr:
        """Double excitation two-electron: ⟨pq|g|rs⟩ - ⟨pq|g|sr⟩"""
        from .symbols import Var, Const

        only_bra, only_ket = self._excitations
        if len(only_bra) >= 2 and len(only_ket) >= 2:
            p, q = only_bra[0], only_bra[1]
            r, s = only_ket[0], only_ket[1]

            g_pqrs = Var(f"g_{{{p.ket_label}{q.ket_label},{r.ket_label}{s.ket_label}}}")
            g_pqsr = Var(f"g_{{{p.ket_label}{q.ket_label},{s.ket_label}{r.ket_label}}}")

            if term.includes_exchange:
                return term.coefficient * (g_pqrs - g_pqsr)
            else:
                return term.coefficient * g_pqrs

        return Const(0)

    # =========================================================================
    # EXPR INTERFACE
    # =========================================================================

    def to_sympy(self):
        """Convert to SymPy expression."""
        if self._reduced_expr:
            return self._reduced_expr.to_sympy()
        return 0

    def to_latex(self) -> str:
        """LaTeX representation."""
        if self._reduced_expr:
            return self._reduced_expr.to_latex()
        return "0"

    def _get_free_variables(self) -> Set[Var]:
        """Get free variables from expression and molecule geometry."""
        free_vars = set()
        if self._reduced_expr:
            free_vars.update(self._reduced_expr._get_free_variables())
        if self._molecule:
            free_vars.update(self._molecule.geometry_variables)
        return free_vars

    # =========================================================================
    # THREE EVALUATION MODES
    # =========================================================================

    def analytical(self, simplify: bool = False) -> Expr:
        """
        Return symbolic expression.

        Applies Slater-Condon rules and optionally simplifies.

        Args:
            simplify: If True, simplify using SymPy (requires sympy).
                     If False (default), return raw expression.

        Returns:
            Symbolic Expr representing the matrix element.
        """
        if self._reduced_expr:
            if simplify:
                return self._reduced_expr.simplify()
            return self._reduced_expr
        from .symbols import Const
        return Const(0)

    def numerical(self, **var_bindings) -> float:
        """
        Evaluate with given variable values.

        Args:
            **var_bindings: Variable values

        Returns:
            Numeric result
        """
        if self._reduced_expr is None:
            return 0.0

        # First try native numeric evaluation (faster, no sympy needed)
        try:
            result = self._evaluate_native(self._reduced_expr, var_bindings)
            return float(result)
        except (ValueError, TypeError, AttributeError):
            pass

        # Fall back to sympy-based evaluation
        result = self._reduced_expr.evaluate(**var_bindings)
        if hasattr(result, 'item'):
            return result.item()
        return float(result)

    def _evaluate_native(self, expr, var_bindings: dict) -> float:
        """Native numeric evaluation without sympy."""
        import math
        from .symbols import Const, Var, Sum, Product, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Sin, Cos, Tan, Exp, Log, Abs

        # Handle None as zero
        if expr is None:
            return 0.0

        # Handle raw numeric values
        if isinstance(expr, (int, float)):
            return float(expr)

        if isinstance(expr, Const):
            return float(expr.value)

        if isinstance(expr, Var):
            if expr.name in var_bindings:
                return float(var_bindings[expr.name])
            raise ValueError(f"Missing value for variable {expr.name}")

        # N-ary operations (Sum, Product)
        if isinstance(expr, Sum):
            return sum(self._evaluate_native(t, var_bindings) for t in expr.terms)

        if isinstance(expr, Product):
            result = 1.0
            for t in expr.terms:
                result *= self._evaluate_native(t, var_bindings)
            return result

        # Binary operations
        if isinstance(expr, Add):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left + right

        if isinstance(expr, Sub):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left - right

        if isinstance(expr, Mul):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left * right

        if isinstance(expr, Div):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left / right

        if isinstance(expr, Pow):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left ** right

        # Unary operations
        if isinstance(expr, Neg):
            return -self._evaluate_native(expr.operand, var_bindings)

        if isinstance(expr, Sqrt):
            return math.sqrt(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Sin):
            return math.sin(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Cos):
            return math.cos(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Tan):
            return math.tan(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Exp):
            return math.exp(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Log):
            return math.log(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Abs):
            return abs(self._evaluate_native(expr.operand, var_bindings))

        # Unknown expression type - fall back to error
        raise TypeError(f"Cannot natively evaluate: {type(expr).__name__}")

    def graph(self, **var_bindings):
        """
        Create lazy ComputeGraph for deferred evaluation.

        Args:
            **var_bindings: Variable values

        Returns:
            ComputeGraph ready for evaluate() or compile()
        """
        from .symbols import SymbolicFunction, ComputeGraph

        if self._reduced_expr is None:
            from .symbols import Const
            self._reduced_expr = Const(0)

        # Create SymbolicFunction from expression
        func = SymbolicFunction(expr=self._reduced_expr)
        bound = func.init()
        return bound.run_with(**var_bindings)

    # =========================================================================
    # COMPILATION
    # =========================================================================

    def compile(self, device: str = 'cpu'):
        """
        Compile to PyTorch function.

        Args:
            device: 'cpu' or 'cuda'

        Returns:
            TorchFunction for batched evaluation
        """
        from .symbols import TorchFunction

        if self._reduced_expr is None:
            from .symbols import Const
            self._reduced_expr = Const(0)

        return TorchFunction(self._reduced_expr, device=device)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def bra(self) -> SlaterDeterminant:
        return self._bra

    @property
    def ket(self) -> SlaterDeterminant:
        return self._ket

    @property
    def hamiltonian(self) -> MolecularHamiltonian:
        return self._hamiltonian

    @property
    def n_excitations(self) -> int:
        return self._n_excitations

    @property
    def is_zero(self) -> bool:
        """True if matrix element is zero by Slater-Condon rules."""
        return self._n_excitations > 2

    @property
    def is_diagonal(self) -> bool:
        """True if bra == ket."""
        return self._n_excitations == 0

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self, notation: str = 'standard', show_slater_condon: bool = True):
        """
        Render the matrix element.

        Args:
            notation: Display notation style
            show_slater_condon: If True, show reduced form
        """
        if show_slater_condon and self.is_zero:
            html = '<div class="cm-math cm-math-center">\\[ 0 \\]</div>'
            views.html(html)
            return

        if show_slater_condon and self._reduced_expr:
            latex = self._reduced_expr.to_latex()
        else:
            # Full bra-ket notation
            bra_labels = ", ".join(self._bra.ket_labels())
            ket_labels = ", ".join(self._ket.ket_labels())
            H_latex = self._hamiltonian.to_latex()
            latex = f"\\langle {bra_labels} | {H_latex} | {ket_labels} \\rangle"

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def __repr__(self) -> str:
        return f"MatrixExpression(n_excitations={self._n_excitations}, is_zero={self.is_zero})"


class HamiltonianMatrix:
    """
    Symbolic Hamiltonian matrix H[i,j] = ⟨Ψᵢ|H|Ψⱼ⟩.

    Represents the full CI or selected CI Hamiltonian matrix
    over a basis of Slater determinants.

    Example:
        H = qm.HamiltonianBuilder.electronic().build()
        basis = water.ci_basis(excitations=2)
        H_mat = H.matrix(basis)

        # Numeric evaluation
        E_matrix = H_mat.numerical(r=0.96, theta=1.82)

        # Eigenvalues
        energies = H_mat.eigenvalues(r=0.96, theta=1.82)
    """

    def __init__(self, basis: List[SlaterDeterminant],
                 hamiltonian: MolecularHamiltonian,
                 molecule: Optional[Molecule] = None):
        self._basis = basis
        self._hamiltonian = hamiltonian
        self._molecule = molecule
        self._elements: Dict[Tuple[int, int], MatrixExpression] = {}

        # Build matrix elements lazily
        self._n = len(basis)

    def _get_element(self, i: int, j: int) -> MatrixExpression:
        """Get or compute matrix element (i, j)."""
        if (i, j) not in self._elements:
            self._elements[(i, j)] = self._hamiltonian.element(
                self._basis[i], self._basis[j], self._molecule
            )
        return self._elements[(i, j)]

    # =========================================================================
    # ACCESS
    # =========================================================================

    def __getitem__(self, ij: Tuple[int, int]) -> MatrixExpression:
        """Get matrix element H[i, j]."""
        i, j = ij
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i}, {j}) out of bounds for {self._n}x{self._n} matrix")
        return self._get_element(i, j)

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix dimensions."""
        return (self._n, self._n)

    @property
    def basis(self) -> List[SlaterDeterminant]:
        """Basis of Slater determinants."""
        return self._basis

    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        return self._n

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def analytical(self) -> List[List[Expr]]:
        """
        All elements as symbolic expressions.

        Returns:
            2D list of Expr objects
        """
        result = []
        for i in range(self._n):
            row = []
            for j in range(self._n):
                elem = self._get_element(i, j)
                row.append(elem.analytical())
            result.append(row)
        return result

    def numerical(self, **var_bindings):
        """
        Evaluate all elements numerically.

        Args:
            **var_bindings: Variable values

        Returns:
            numpy ndarray
        """
        import numpy as np

        result = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(self._n):
                elem = self._get_element(i, j)
                result[i, j] = elem.numerical(**var_bindings)
        return result

    def graph(self, **var_bindings) -> List[List]:
        """
        Create ComputeGraphs for all elements.

        Args:
            **var_bindings: Variable values

        Returns:
            2D list of ComputeGraph objects
        """
        result = []
        for i in range(self._n):
            row = []
            for j in range(self._n):
                elem = self._get_element(i, j)
                row.append(elem.graph(**var_bindings))
            result.append(row)
        return result

    # =========================================================================
    # EIGENVALUE METHODS
    # =========================================================================

    def eigenvalues(self, **var_bindings):
        """
        Compute eigenvalues of Hamiltonian matrix.

        Args:
            **var_bindings: Variable values for geometry

        Returns:
            numpy array of eigenvalues (sorted ascending)
        """
        import numpy as np

        H_numeric = self.numerical(**var_bindings)
        eigenvalues = np.linalg.eigvalsh(H_numeric)
        return np.sort(eigenvalues)

    def eigenvectors(self, **var_bindings) -> Tuple:
        """
        Compute eigenvalues and eigenvectors.

        Args:
            **var_bindings: Variable values

        Returns:
            (eigenvalues, eigenvectors) tuple
        """
        import numpy as np

        H_numeric = self.numerical(**var_bindings)
        eigenvalues, eigenvectors = np.linalg.eigh(H_numeric)

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]

    def ground_state_energy(self, **var_bindings) -> float:
        """
        Get ground state (lowest) eigenvalue.

        Args:
            **var_bindings: Variable values

        Returns:
            Ground state energy
        """
        return self.eigenvalues(**var_bindings)[0]

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self, max_size: int = 10, **var_bindings):
        """
        Render the Hamiltonian matrix.

        Args:
            max_size: Maximum matrix size to display
            **var_bindings: Optional variable values for numeric display
        """
        if self._n > max_size:
            html = f'<div>Hamiltonian matrix: {self._n}×{self._n} (too large to display)</div>'
            views.html(html)
            return

        if var_bindings:
            # Numeric display
            import numpy as np
            H = self.numerical(**var_bindings)

            latex = "\\begin{pmatrix}\n"
            for i in range(self._n):
                row = " & ".join(f"{H[i,j]:.4f}" for j in range(self._n))
                latex += row + " \\\\\n"
            latex += "\\end{pmatrix}"
        else:
            # Symbolic display
            latex = "\\begin{pmatrix}\n"
            for i in range(self._n):
                row_parts = []
                for j in range(self._n):
                    elem = self._get_element(i, j)
                    if elem.is_zero:
                        row_parts.append("0")
                    else:
                        row_parts.append(elem.to_latex())
                latex += " & ".join(row_parts) + " \\\\\n"
            latex += "\\end{pmatrix}"

        html = f'<div class="cm-math cm-math-center">\\[ H = {latex} \\]</div>'
        views.html(html)

    def __repr__(self) -> str:
        return f"HamiltonianMatrix(shape={self.shape})"
