"""
Special functions for cm.math.

Available functions:
    krok_delta(a, b)  - Kronecker delta: 1 if a == b, else 0
"""

from .ops import krok_delta, register_spec_func_ops

__all__ = ['krok_delta']

# Register backends on import
register_spec_func_ops()
