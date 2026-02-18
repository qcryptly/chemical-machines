"""
Chemical Machines Math Package

Structure-based symbolic mathematics library.

Usage:
    from cm.math import struct, numbers

    A = struct.lin_alg.tensor(shape=(3,3), dtype=numbers.float64, name="A")
    B = struct.lin_alg.tensor(shape=(3,3), name="B")
    expr = (A @ B).det() + A.trace()

    expr.to_latex()          # LaTeX string
    expr.evaluate(B=val)     # NumPy eager eval
    expr.to_torch()          # PyTorch compute graph
    expr.render()            # HTML output via views.html()
"""

from . import struct
from . import numbers
from . import operator
from .index import index, IndexedExpression

__all__ = ['struct', 'numbers', 'index', 'IndexedExpression', 'operator']
