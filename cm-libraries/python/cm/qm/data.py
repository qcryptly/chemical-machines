"""
Chemical Machines QM Data Module

Atomic data constants including element symbols, atomic numbers,
and electron configuration utilities.
"""

from typing import Dict, List, Tuple

__all__ = [
    'ATOMIC_NUMBERS',
    'ELEMENT_SYMBOLS',
    'AUFBAU_ORDER',
    'NOBLE_GAS_CONFIGS',
    'L_LABELS',
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
