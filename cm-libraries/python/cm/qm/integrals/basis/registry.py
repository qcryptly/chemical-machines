"""
Basis Set Registry

Provides centralized access to all basis sets with aliasing support.
Handles basis set name normalization and variant selection.
"""

from typing import Dict, Optional, Tuple, List
import re


class BasisRegistry:
    """
    Central registry for all basis sets.

    Supports:
    - Name normalization (6-31G* == 6-31G(d) == 631Gs)
    - Variant selection (polarization, diffuse functions)
    - Lazy loading of basis data

    Example:
        data = BasisRegistry.get('6-31G*')
        data = BasisRegistry.get('cc-pVTZ')
        names = BasisRegistry.list_available()
    """

    _basis_sets: Dict[str, Dict] = {}
    _loaded: Dict[str, bool] = {}

    # Canonical names and their aliases
    _aliases: Dict[str, str] = {
        # STO-3G
        'sto3g': 'STO-3G',
        'sto-3g': 'STO-3G',
        'minao': 'STO-3G',

        # 3-21G
        '321g': '3-21G',
        '3-21g': '3-21G',

        # 6-31G family
        '631g': '6-31G',
        '6-31g': '6-31G',
        '631gs': '6-31G*',
        '6-31gs': '6-31G*',
        '631g*': '6-31G*',
        '6-31g*': '6-31G*',
        '631gd': '6-31G*',
        '6-31gd': '6-31G*',
        '6-31g(d)': '6-31G*',
        '631g**': '6-31G**',
        '6-31g**': '6-31G**',
        '631gdp': '6-31G**',
        '6-31gdp': '6-31G**',
        '6-31g(d,p)': '6-31G**',
        '6-31+g': '6-31+G',
        '631+g': '6-31+G',
        '6-31+g*': '6-31+G*',
        '631+gs': '6-31+G*',
        '6-31+g(d)': '6-31+G*',
        '6-31++g': '6-31++G',
        '6-31++g**': '6-31++G**',
        '6-31++g(d,p)': '6-31++G**',

        # 6-311G family
        '6311g': '6-311G',
        '6-311g': '6-311G',
        '6-311g*': '6-311G*',
        '6-311g**': '6-311G**',
        '6-311+g*': '6-311+G*',
        '6-311++g**': '6-311++G**',
        '6-311g(2d,2p)': '6-311G(2d,2p)',
        '6-311+g(2d,2p)': '6-311+G(2d,2p)',

        # Dunning cc-pVXZ
        'ccpvdz': 'cc-pVDZ',
        'cc-pvdz': 'cc-pVDZ',
        'ccpvtz': 'cc-pVTZ',
        'cc-pvtz': 'cc-pVTZ',
        'ccpvqz': 'cc-pVQZ',
        'cc-pvqz': 'cc-pVQZ',

        # Augmented Dunning
        'augccpvdz': 'aug-cc-pVDZ',
        'aug-cc-pvdz': 'aug-cc-pVDZ',
        'augccpvtz': 'aug-cc-pVTZ',
        'aug-cc-pvtz': 'aug-cc-pVTZ',
        'augccpvqz': 'aug-cc-pVQZ',
        'aug-cc-pvqz': 'aug-cc-pVQZ',

        # def2 family
        'def2svp': 'def2-SVP',
        'def2-svp': 'def2-SVP',
        'def2tzvp': 'def2-TZVP',
        'def2-tzvp': 'def2-TZVP',
        'def2qzvp': 'def2-QZVP',
        'def2-qzvp': 'def2-QZVP',
    }

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        """Normalize basis set name to canonical form."""
        # Convert to lowercase and remove spaces
        normalized = name.lower().replace(' ', '').replace('_', '')

        # Check aliases
        if normalized in cls._aliases:
            return cls._aliases[normalized]

        # Return uppercase version if not found in aliases
        return name.upper()

    @classmethod
    def _load_basis(cls, canonical_name: str) -> Dict:
        """Load basis set data lazily."""
        if canonical_name in cls._basis_sets:
            return cls._basis_sets[canonical_name]

        # Import basis data based on name
        if canonical_name == 'STO-3G':
            from .sto3g import STO_3G_DATA
            cls._basis_sets[canonical_name] = STO_3G_DATA

        elif canonical_name == 'cc-pVTZ':
            from .cc_pvtz import CC_PVTZ_DATA
            cls._basis_sets[canonical_name] = CC_PVTZ_DATA

        elif canonical_name == 'cc-pVQZ':
            from .cc_pvqz import CC_PVQZ_DATA
            cls._basis_sets[canonical_name] = CC_PVQZ_DATA

        elif canonical_name == 'cc-pVDZ':
            from .cc_pvdz import CC_PVDZ_DATA
            cls._basis_sets[canonical_name] = CC_PVDZ_DATA

        elif canonical_name in ('6-31G', '6-31G*', '6-31G**',
                                '6-31+G', '6-31+G*', '6-31++G', '6-31++G**'):
            from .pople import get_6_31g_basis
            cls._basis_sets[canonical_name] = get_6_31g_basis(canonical_name)

        elif canonical_name in ('3-21G',):
            from .pople import get_3_21g_basis
            cls._basis_sets[canonical_name] = get_3_21g_basis()

        elif canonical_name in ('6-311G', '6-311G*', '6-311G**',
                                '6-311+G*', '6-311++G**',
                                '6-311G(2d,2p)', '6-311+G(2d,2p)'):
            from .pople import get_6_311g_basis
            cls._basis_sets[canonical_name] = get_6_311g_basis(canonical_name)

        elif canonical_name.startswith('aug-cc-pV'):
            from .dunning import get_aug_cc_pvxz_basis
            cls._basis_sets[canonical_name] = get_aug_cc_pvxz_basis(canonical_name)

        elif canonical_name.startswith('def2-'):
            from .def2 import get_def2_basis
            cls._basis_sets[canonical_name] = get_def2_basis(canonical_name)

        else:
            raise ValueError(f"Unknown basis set: {canonical_name}")

        return cls._basis_sets[canonical_name]

    @classmethod
    def get(cls, name: str) -> Dict:
        """
        Get basis set data by name.

        Args:
            name: Basis set name (aliases accepted)

        Returns:
            Dictionary mapping elements to shell data

        Raises:
            ValueError: If basis set is not found

        Example:
            data = BasisRegistry.get('6-31G*')
            # Returns: {'H': {'1s': [...]}, 'C': {'1s': [...], '2s': [...], ...}}
        """
        canonical = cls._normalize_name(name)
        return cls._load_basis(canonical)

    @classmethod
    def get_canonical_name(cls, name: str) -> str:
        """Get the canonical name for a basis set."""
        return cls._normalize_name(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available basis sets."""
        return [
            'STO-3G',
            '3-21G',
            '6-31G', '6-31G*', '6-31G**',
            '6-31+G', '6-31+G*', '6-31++G', '6-31++G**',
            '6-311G', '6-311G*', '6-311G**',
            '6-311+G*', '6-311++G**',
            '6-311G(2d,2p)', '6-311+G(2d,2p)',
            'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ',
            'aug-cc-pVDZ', 'aug-cc-pVTZ', 'aug-cc-pVQZ',
            'def2-SVP', 'def2-TZVP', 'def2-QZVP',
        ]

    @classmethod
    def supports_element(cls, name: str, element: str) -> bool:
        """Check if a basis set supports a given element."""
        try:
            data = cls.get(name)
            return element in data
        except (ValueError, KeyError):
            return False

    @classmethod
    def get_elements(cls, name: str) -> List[str]:
        """Get list of elements supported by a basis set."""
        data = cls.get(name)
        return list(data.keys())


# Atomic data for all elements H-Ar
ATOMIC_DATA = {
    'H':  {'Z': 1,  'mass': 1.00794,   'bragg_slater': 0.25, 'covalent': 0.31},
    'He': {'Z': 2,  'mass': 4.002602,  'bragg_slater': 0.25, 'covalent': 0.28},
    'Li': {'Z': 3,  'mass': 6.941,     'bragg_slater': 1.45, 'covalent': 1.28},
    'Be': {'Z': 4,  'mass': 9.012182,  'bragg_slater': 1.05, 'covalent': 0.96},
    'B':  {'Z': 5,  'mass': 10.811,    'bragg_slater': 0.85, 'covalent': 0.84},
    'C':  {'Z': 6,  'mass': 12.0107,   'bragg_slater': 0.70, 'covalent': 0.76},
    'N':  {'Z': 7,  'mass': 14.0067,   'bragg_slater': 0.65, 'covalent': 0.71},
    'O':  {'Z': 8,  'mass': 15.9994,   'bragg_slater': 0.60, 'covalent': 0.66},
    'F':  {'Z': 9,  'mass': 18.9984,   'bragg_slater': 0.50, 'covalent': 0.57},
    'Ne': {'Z': 10, 'mass': 20.1797,   'bragg_slater': 0.45, 'covalent': 0.58},
    'Na': {'Z': 11, 'mass': 22.98977,  'bragg_slater': 1.80, 'covalent': 1.66},
    'Mg': {'Z': 12, 'mass': 24.305,    'bragg_slater': 1.50, 'covalent': 1.41},
    'Al': {'Z': 13, 'mass': 26.98154,  'bragg_slater': 1.25, 'covalent': 1.21},
    'Si': {'Z': 14, 'mass': 28.0855,   'bragg_slater': 1.10, 'covalent': 1.11},
    'P':  {'Z': 15, 'mass': 30.97376,  'bragg_slater': 1.00, 'covalent': 1.07},
    'S':  {'Z': 16, 'mass': 32.065,    'bragg_slater': 1.00, 'covalent': 1.05},
    'Cl': {'Z': 17, 'mass': 35.453,    'bragg_slater': 1.00, 'covalent': 1.02},
    'Ar': {'Z': 18, 'mass': 39.948,    'bragg_slater': 0.88, 'covalent': 1.06},
    # Transition metals (for future ECP support)
    'Sc': {'Z': 21, 'mass': 44.95591,  'bragg_slater': 1.60, 'covalent': 1.70},
    'Ti': {'Z': 22, 'mass': 47.867,    'bragg_slater': 1.40, 'covalent': 1.60},
    'V':  {'Z': 23, 'mass': 50.9415,   'bragg_slater': 1.35, 'covalent': 1.53},
    'Cr': {'Z': 24, 'mass': 51.9961,   'bragg_slater': 1.40, 'covalent': 1.39},
    'Mn': {'Z': 25, 'mass': 54.93805,  'bragg_slater': 1.40, 'covalent': 1.39},
    'Fe': {'Z': 26, 'mass': 55.845,    'bragg_slater': 1.40, 'covalent': 1.32},
    'Co': {'Z': 27, 'mass': 58.9332,   'bragg_slater': 1.35, 'covalent': 1.26},
    'Ni': {'Z': 28, 'mass': 58.6934,   'bragg_slater': 1.35, 'covalent': 1.24},
    'Cu': {'Z': 29, 'mass': 63.546,    'bragg_slater': 1.35, 'covalent': 1.32},
    'Zn': {'Z': 30, 'mass': 65.38,     'bragg_slater': 1.35, 'covalent': 1.22},
}


def get_atomic_number(element: str) -> int:
    """Get atomic number for an element symbol."""
    if element not in ATOMIC_DATA:
        raise ValueError(f"Unknown element: {element}")
    return ATOMIC_DATA[element]['Z']


def get_atomic_mass(element: str) -> float:
    """Get atomic mass in amu for an element symbol."""
    if element not in ATOMIC_DATA:
        raise ValueError(f"Unknown element: {element}")
    return ATOMIC_DATA[element]['mass']


def get_bragg_slater_radius(element: str) -> float:
    """Get Bragg-Slater radius in Angstroms for an element."""
    if element not in ATOMIC_DATA:
        raise ValueError(f"Unknown element: {element}")
    return ATOMIC_DATA[element]['bragg_slater']


def get_covalent_radius(element: str) -> float:
    """Get covalent radius in Angstroms for an element."""
    if element not in ATOMIC_DATA:
        raise ValueError(f"Unknown element: {element}")
    return ATOMIC_DATA[element]['covalent']
