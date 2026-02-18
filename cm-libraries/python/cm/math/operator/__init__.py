"""
Operator module for cm.math.

Provides operator expressions that transform other expressions when applied.
Operators compose with SymbolicTensor matmul for operator algebra.

Usage:
    from cm.math import operator as op
    y = struct.lin_alg.scalar(name="y")
    d_dy = op.derivative(y)
    d2_dy2 = op.derivative(y, 2)
"""

from __future__ import annotations
from ..struct.base import Expression, Var, ScalarExpr, Operation
from .diff import differentiate, differentiate_n

__all__ = ['derivative', 'OperatorExpr']


class OperatorExpr:
    """An operator expression that transforms other expressions when applied.

    When used as an element in a SymbolicTensor and matmul'd with another
    tensor, the operator is applied to the corresponding elements via
    .apply() instead of standard multiplication.

    Supports adding scalar expressions: (d/dy + y²)(f) = f' + y²·f
    """

    def __init__(self, apply_fn, structure, var=None, order=1):
        self._apply_fn = apply_fn
        self._structure = structure
        self._var = var
        self._order = order
        self._scalar_part = None
        self._coeff = None  # scalar coefficient on the operator

    def apply(self, expr):
        """Apply this operator to an expression.

        For derivative: returns d(expr)/d(var)
        If scalar_part exists: returns apply(expr) + scalar_part * expr
        """
        result = self._apply_fn(expr)
        if self._coeff is not None:
            result = self._coeff * result
        if self._scalar_part is not None:
            result = result + self._scalar_part * expr
        return result

    def __add__(self, other):
        """op + scalar_expr → operator with scalar part."""
        if isinstance(other, OperatorExpr):
            raise NotImplementedError("Adding two operators not yet supported")
        new_op = OperatorExpr(
            self._apply_fn, self._structure,
            var=self._var, order=self._order,
        )
        new_op._coeff = self._coeff
        if self._scalar_part is not None:
            new_op._scalar_part = self._scalar_part + _to_expr(other, self._structure)
        else:
            new_op._scalar_part = _to_expr(other, self._structure)
        return new_op

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        """scalar * op → scaled operator."""
        new_op = OperatorExpr(
            self._apply_fn, self._structure,
            var=self._var, order=self._order,
        )
        coeff = _to_expr(other, self._structure)
        if self._coeff is not None:
            new_op._coeff = self._coeff * coeff
        else:
            new_op._coeff = coeff
        new_op._scalar_part = self._scalar_part
        return new_op

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        var_name = self._var.var_name if self._var else "?"
        parts = [f"d/d{var_name}"]
        if self._order > 1:
            parts = [f"d^{self._order}/d{var_name}^{self._order}"]
        if self._scalar_part is not None:
            parts.append(f"+ {self._scalar_part}")
        return f"OperatorExpr({', '.join(parts)})"


def _to_expr(value, structure):
    """Convert value to Expression if needed."""
    if isinstance(value, Expression):
        return value
    return ScalarExpr(value, structure)


def derivative(var, order=1):
    """Create a derivative operator with respect to var.

    Args:
        var: A Var to differentiate with respect to.
        order: Order of derivative (default 1).

    Returns:
        OperatorExpr that differentiates expressions when applied.
    """
    def apply_fn(expr):
        return differentiate_n(expr, var, order)

    return OperatorExpr(
        apply_fn=apply_fn,
        structure=var.structure,
        var=var,
        order=order,
    )
