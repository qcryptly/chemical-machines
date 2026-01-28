"""
Time-Dependent DFT (TDDFT) Module

Provides linear-response TDDFT for excited state calculations.

Implements:
- Random Phase Approximation (RPA)
- Tamm-Dancoff Approximation (TDA)
- Full TDDFT response equations

The TDDFT eigenvalue problem:
    [A  B ] [X]   [1  0] [X]
    [B* A*] [Y] = [0 -1] [Y] ω

In TDA (B=0):
    A X = ω X

Reference: Casida, "Time-Dependent Density Functional Response Theory" (1995)
"""

from .linear_response import (
    TDDFTResult,
    tddft,
    TDAResult,
    tda,
    compute_oscillator_strength,
)

__all__ = [
    'TDDFTResult',
    'tddft',
    'TDAResult',
    'tda',
    'compute_oscillator_strength',
]
