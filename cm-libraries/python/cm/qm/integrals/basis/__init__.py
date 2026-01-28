"""
Gaussian Basis Set Module

Provides Gaussian-type orbital (GTO) basis sets for quantum chemistry calculations.

Available basis sets:
- STO-3G: Minimal basis (teaching/qualitative)
- 3-21G: Split-valence (geometry optimization)
- 6-31G, 6-31G*, 6-31G**: Standard split-valence with polarization
- 6-311G, 6-311G*, 6-311G**: Triple-split valence
- cc-pVDZ, cc-pVTZ, cc-pVQZ: Correlation-consistent (publication quality)
- aug-cc-pVXZ: Augmented with diffuse functions (coming soon)
- def2-SVP, def2-TZVP, def2-QZVP: Karlsruhe basis sets (coming soon)

Use BasisRegistry.get(name) to retrieve basis set data by name,
or BasisSet(name) to create a basis set object for a calculation.
"""

from .core import (
    GaussianPrimitive,
    ContractedGaussian,
    BasisFunction,
    BasisSet,
)
from .sto3g import STO_3G_DATA
from .cc_pvtz import CC_PVTZ_DATA
from .cc_pvqz import CC_PVQZ_DATA
from .cc_pvdz import CC_PVDZ_DATA

from .registry import (
    BasisRegistry,
    ATOMIC_DATA,
    get_atomic_number,
    get_atomic_mass,
    get_bragg_slater_radius,
    get_covalent_radius,
)

__all__ = [
    # Core classes
    'GaussianPrimitive',
    'ContractedGaussian',
    'BasisFunction',
    'BasisSet',
    # Registry
    'BasisRegistry',
    'ATOMIC_DATA',
    'get_atomic_number',
    'get_atomic_mass',
    'get_bragg_slater_radius',
    'get_covalent_radius',
    # Basis data
    'STO_3G_DATA',
    'CC_PVTZ_DATA',
    'CC_PVQZ_DATA',
    'CC_PVDZ_DATA',
]
