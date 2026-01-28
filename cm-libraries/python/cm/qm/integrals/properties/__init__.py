"""
Molecular Properties Module

Provides calculation of molecular properties:
- Hessian (second derivatives)
- Vibrational frequencies
- Thermochemistry (ZPE, enthalpy, entropy, Gibbs energy)
- Dipole moment
- Polarizability
- NMR chemical shifts (future)
"""

from .hessian import (
    compute_hessian,
    HessianResult,
)
from .frequencies import (
    FrequencyResult,
    harmonic_frequencies,
    normal_mode_analysis,
)
from .thermochemistry import (
    ThermochemistryResult,
    thermochemistry,
)
from .dipole import (
    dipole_moment,
    DipoleResult,
)
from .polarizability import (
    PolarizabilityResult,
    PolarizabilityCalculator,
    static_polarizability,
)

__all__ = [
    # Hessian
    'compute_hessian',
    'HessianResult',
    # Frequencies
    'FrequencyResult',
    'harmonic_frequencies',
    'normal_mode_analysis',
    # Thermochemistry
    'ThermochemistryResult',
    'thermochemistry',
    # Dipole
    'dipole_moment',
    'DipoleResult',
    # Polarizability
    'PolarizabilityResult',
    'PolarizabilityCalculator',
    'static_polarizability',
]
