"""
Analytic Gradients Module

Provides analytic energy gradients for geometry optimization:
- HF gradients
- DFT gradients
- MP2 gradients (numerical)

The gradient of the HF energy with respect to nuclear coordinates is:
    dE/dX = dH_core/dX + dG/dX - W·dS/dX + dV_nn/dX

where W is the energy-weighted density matrix.
"""

from .hf_gradient import (
    GradientResult,
    hf_gradient,
    HFGradientCalculator,
    numerical_gradient,
)
from .dft_gradient import (
    dft_gradient,
    DFTGradientCalculator,
)
from .derivative_integrals import (
    overlap_derivative,
    kinetic_derivative,
    nuclear_derivative,
    eri_derivative,
)

__all__ = [
    # Results
    'GradientResult',
    # HF
    'hf_gradient',
    'HFGradientCalculator',
    # DFT
    'dft_gradient',
    'DFTGradientCalculator',
    # Numerical
    'numerical_gradient',
    # Derivative integrals
    'overlap_derivative',
    'kinetic_derivative',
    'nuclear_derivative',
    'eri_derivative',
]
