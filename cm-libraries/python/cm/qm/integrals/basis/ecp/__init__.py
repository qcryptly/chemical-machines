"""
Effective Core Potential (ECP) Module

Provides effective core potentials for heavy elements where
all-electron calculations are expensive or require relativistic treatment.

Available ECPs:
- LANL2DZ: Los Alamos National Lab double-zeta (Hay-Wadt)
- SDD: Stuttgart/Dresden ECPs

ECPs replace core electrons with an effective potential:
    V_ECP = V_local + Σ_l V_l |l><l|

where V_local is the local (l-independent) potential and V_l are
semi-local (l-dependent) projectors.
"""

from .core import (
    ECPPotential,
    ECPBasisSet,
    get_ecp,
)
from .lanl2dz import LANL2DZ_ECP

__all__ = [
    'ECPPotential',
    'ECPBasisSet',
    'get_ecp',
    'LANL2DZ_ECP',
]
