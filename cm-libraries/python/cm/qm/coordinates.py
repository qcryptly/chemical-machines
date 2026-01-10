"""
Chemical Machines QM Coordinates Module

3D coordinate types and coordinate system utilities.
"""

from typing import Optional, List, Union, Tuple, Set, Any
from enum import Enum

from ..symbols import Expr, Var, Const, _ensure_expr

__all__ = [
    'CoordinateType',
    'Coordinate3D',
    'coord3d',
    'spherical_coord',
    'cartesian_coord',
]


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
