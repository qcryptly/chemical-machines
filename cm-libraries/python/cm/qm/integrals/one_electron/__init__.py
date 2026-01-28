"""
One-Electron Integrals Module

Provides implementations for one-electron molecular integrals:
- Overlap integrals (S)
- Kinetic energy integrals (T)
- Nuclear attraction integrals (V)
"""

from .overlap import (
    overlap_1d,
    overlap_primitive,
    overlap_contracted,
    overlap_integral,
    overlap_matrix,
)
from .kinetic import (
    kinetic_primitive,
    kinetic_contracted,
    kinetic_integral,
    kinetic_matrix,
)
from .nuclear import (
    nuclear_primitive,
    nuclear_contracted,
    nuclear_attraction_integral,
    nuclear_attraction_matrix,
)

__all__ = [
    # Overlap
    'overlap_1d',
    'overlap_primitive',
    'overlap_contracted',
    'overlap_integral',
    'overlap_matrix',
    # Kinetic
    'kinetic_primitive',
    'kinetic_contracted',
    'kinetic_integral',
    'kinetic_matrix',
    # Nuclear
    'nuclear_primitive',
    'nuclear_contracted',
    'nuclear_attraction_integral',
    'nuclear_attraction_matrix',
]
