"""
Elementary mathematical functions for the linear algebra structure.

Usage:
    from cm.math.struct.lin_alg import fxn
    y = struct.lin_alg.scalar(name="y")
    expr = fxn.sin(y) + fxn.cos(y)
"""

from .ops import (
    sin, cos, tan, exp, log, sqrt, fabs as abs,
    asin, acos, atan, sinh, cosh, tanh,
    factorial, comb, krok_delta,
    pi,
)
from .special import (
    assoc_laguerre, assoc_legendre,
    spherical_harmonic, radial,
)

__all__ = [
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs',
    'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
    'factorial', 'comb', 'krok_delta',
    'pi',
    'assoc_laguerre', 'assoc_legendre',
    'spherical_harmonic', 'radial',
]
