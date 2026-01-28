"""
ECP Core Data Structures

Defines the data structures for effective core potentials.

An ECP consists of:
1. A local potential (acts on all electrons)
2. Semi-local potentials for each angular momentum l

The ECP potential is:
    V_ECP(r) = V_local(r) + Σ_l Σ_k A_lk r^(n_lk) exp(-ζ_lk r²) |l><l|

Reference: Hay, Wadt, J. Chem. Phys. 82, 270 (1985)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ECPComponent:
    """
    Single component of an ECP potential.

    Represents: A * r^n * exp(-ζ r²)

    Attributes:
        coefficient: Amplitude A
        r_power: Power of r (n)
        exponent: Gaussian exponent ζ
    """
    coefficient: float
    r_power: int
    exponent: float


@dataclass
class ECPPotential:
    """
    Effective Core Potential for an element.

    Attributes:
        element: Element symbol
        n_core: Number of core electrons replaced
        lmax: Maximum angular momentum in semi-local part
        local: Local potential components (l = lmax + 1)
        semilocal: Semi-local potentials by angular momentum
    """
    element: str
    n_core: int
    lmax: int
    local: List[ECPComponent] = field(default_factory=list)
    semilocal: Dict[int, List[ECPComponent]] = field(default_factory=dict)

    @property
    def n_valence(self) -> int:
        """Number of valence electrons (Z - n_core)."""
        Z = self._get_atomic_number()
        return Z - self.n_core

    def _get_atomic_number(self) -> int:
        """Get atomic number from element symbol."""
        Z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
            'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
        }
        return Z.get(self.element, 1)

    def evaluate_local(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate local potential at radial distances.

        Args:
            r: Radial distances (n_points,)

        Returns:
            V_local(r) values
        """
        V = np.zeros_like(r)
        for comp in self.local:
            V += comp.coefficient * r ** comp.r_power * np.exp(-comp.exponent * r ** 2)
        return V

    def evaluate_semilocal(self, l: int, r: np.ndarray) -> np.ndarray:
        """
        Evaluate semi-local potential for angular momentum l.

        Args:
            l: Angular momentum
            r: Radial distances

        Returns:
            V_l(r) values
        """
        if l not in self.semilocal:
            return np.zeros_like(r)

        V = np.zeros_like(r)
        for comp in self.semilocal[l]:
            V += comp.coefficient * r ** comp.r_power * np.exp(-comp.exponent * r ** 2)
        return V


class ECPBasisSet:
    """
    Basis set with associated ECPs.

    Combines a valence basis set with ECPs for heavy elements.

    Example:
        basis = ECPBasisSet('LANL2DZ')
        basis.build_for_molecule([('Fe', (0, 0, 0)), ('O', (2, 0, 0))])
    """

    def __init__(self, name: str):
        """
        Initialize ECP basis set.

        Args:
            name: Basis/ECP name ('LANL2DZ', 'SDD', etc.)
        """
        self.name = name
        self.ecps = {}  # Element -> ECPPotential
        self.valence_basis = {}  # Element -> basis data
        self._load_data()

    def _load_data(self):
        """Load ECP and valence basis data."""
        if self.name.upper() == 'LANL2DZ':
            from .lanl2dz import LANL2DZ_ECP, LANL2DZ_BASIS
            self.ecps = LANL2DZ_ECP
            self.valence_basis = LANL2DZ_BASIS

    def get_ecp(self, element: str) -> Optional[ECPPotential]:
        """Get ECP for element (None if no ECP needed)."""
        return self.ecps.get(element)

    def get_valence_basis(self, element: str) -> Dict:
        """Get valence basis data for element."""
        return self.valence_basis.get(element, {})

    def has_ecp(self, element: str) -> bool:
        """Check if element uses an ECP."""
        return element in self.ecps

    def n_core_electrons(self, element: str) -> int:
        """Get number of core electrons for element."""
        ecp = self.get_ecp(element)
        return ecp.n_core if ecp else 0


# ECP registry
_ECP_REGISTRY = {}


def register_ecp(name: str, ecps: Dict[str, ECPPotential]):
    """Register an ECP set."""
    _ECP_REGISTRY[name.upper()] = ecps


def get_ecp(name: str, element: str) -> Optional[ECPPotential]:
    """
    Get ECP for an element from a named ECP set.

    Args:
        name: ECP name ('LANL2DZ', 'SDD', etc.)
        element: Element symbol

    Returns:
        ECPPotential or None if element has no ECP
    """
    if name.upper() not in _ECP_REGISTRY:
        # Try to load
        if name.upper() == 'LANL2DZ':
            from .lanl2dz import LANL2DZ_ECP
            register_ecp('LANL2DZ', LANL2DZ_ECP)

    ecps = _ECP_REGISTRY.get(name.upper(), {})
    return ecps.get(element)
