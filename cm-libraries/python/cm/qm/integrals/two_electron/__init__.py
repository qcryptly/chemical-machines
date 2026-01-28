"""
Two-Electron Integrals Module

Provides implementations for two-electron repulsion integrals (ERI):
- Standard ERI computation
- Optimized ERI with Schwarz screening
- GPU-accelerated ERI via PyTorch
"""

from .eri import (
    eri_primitive,
    eri_contracted,
    electron_repulsion_integral,
    eri_tensor,
    compute_J_matrix,
    compute_K_matrix,
    compute_G_matrix,
)
from .eri_optimized import (
    compute_schwarz_bounds,
    eri_tensor_screened,
    eri_tensor_torch,
    eri_tensor_optimized,
    eri_direct,
)

__all__ = [
    # Standard ERI
    'eri_primitive',
    'eri_contracted',
    'electron_repulsion_integral',
    'eri_tensor',
    'compute_J_matrix',
    'compute_K_matrix',
    'compute_G_matrix',
    # Optimized ERI
    'compute_schwarz_bounds',
    'eri_tensor_screened',
    'eri_tensor_torch',
    'eri_tensor_optimized',
    'eri_direct',
]
