"""
Pople Basis Sets

Split-valence basis sets developed by John Pople and coworkers.
Commonly used for routine calculations.

Available:
- 3-21G: Split-valence, good for geometry optimization
- 6-31G: Split-valence, standard for many applications
- 6-311G: Triple-split valence, higher accuracy

Variants:
- *: d polarization on heavy atoms
- **: d on heavy atoms + p on hydrogen
- +: Diffuse functions on heavy atoms
- ++: Diffuse functions on all atoms
"""

from .basis_6_31g import get_6_31g_basis, BASIS_6_31G_DATA, POLARIZATION_D, POLARIZATION_P, DIFFUSE_SP
from .basis_3_21g import get_3_21g_basis, BASIS_3_21G_DATA
from .basis_6_311g import get_6_311g_basis, BASIS_6_311G_DATA

__all__ = [
    'get_6_31g_basis',
    'get_3_21g_basis',
    'get_6_311g_basis',
    'BASIS_6_31G_DATA',
    'BASIS_3_21G_DATA',
    'BASIS_6_311G_DATA',
]
