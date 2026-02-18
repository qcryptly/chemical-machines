"""
Eager (NumPy) evaluation backend for expressions.

Walks the expression DAG, substitutes variable values, computes results.
Registry maps (structure_name, op_name) -> numpy callable.
"""

from __future__ import annotations
from typing import Dict, Tuple, Callable, Any, Optional

__all__ = ['EagerBackend', 'UnboundVariableError', 'eager_backend']

_np = None


def _get_np():
    global _np
    if _np is None:
        import numpy
        _np = numpy
    return _np


class UnboundVariableError(Exception):
    """Raised when evaluating an expression with unbound variables."""
    pass


class EagerBackend:
    """Recursively evaluates an expression DAG using NumPy."""

    def __init__(self):
        self._registry: Dict[Tuple[str, str], Callable] = {}

    def register(self, structure, op, impl):
        self._registry[(structure, op)] = impl

    def evaluate(self, expr, bindings=None):
        from ..base import Var, ScalarExpr
        bindings = bindings or {}
        np = _get_np()

        # Leaf: Var
        if isinstance(expr, Var):
            if expr.value is not None:
                return expr.value
            name = expr.var_name
            if name in bindings:
                val = bindings[name]
                if hasattr(val, 'detach'):
                    return val.detach().cpu().numpy()
                return val
            raise UnboundVariableError(
                f"Variable '{name}' has no value and was not provided in bindings.\n"
                f"Available bindings: {list(bindings.keys())}"
            )

        # Leaf: ScalarExpr
        if isinstance(expr, ScalarExpr):
            return expr.scalar_value

        # Recursive: evaluate children
        child_values = [self.evaluate(c, bindings) for c in expr.children]

        # Registered implementation
        key = (expr.structure.name, expr.op.name)
        impl = self._registry.get(key)
        if impl is not None:
            return impl(*child_values)

        # Fallback: basic arithmetic
        return self._eval_generic(expr.op.name, child_values)

    def _eval_generic(self, op_name, values):
        np = _get_np()

        if op_name == "add":
            return values[0] + values[1]
        if op_name == "sub":
            return values[0] - values[1]
        if op_name == "mul":
            return values[0] * values[1]
        if op_name == "div":
            return values[0] / values[1]
        if op_name == "pow":
            return np.power(values[0], values[1])
        if op_name == "neg":
            return -values[0]

        raise NotImplementedError(
            f"No eager implementation for operation '{op_name}'"
        )


# Module-level singleton
eager_backend = EagerBackend()
