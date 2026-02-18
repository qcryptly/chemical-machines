"""
Symbolic differentiation engine for cm.math expressions.

Walks the expression DAG and applies standard calculus rules
(power rule, product rule, chain rule, etc.) to produce a new
expression tree representing the derivative.
"""

from __future__ import annotations
from ..struct.base import Expression, Var, ScalarExpr, _ensure_expression


def differentiate(expr, var):
    """Symbolically differentiate expr with respect to Var var.

    Args:
        expr: An Expression DAG node.
        var: A Var representing the differentiation variable.

    Returns:
        A new Expression representing d(expr)/d(var).
    """
    # Leaf: Var
    if isinstance(expr, Var):
        if expr.var_name == var.var_name:
            return ScalarExpr(1, expr.structure)
        return ScalarExpr(0, expr.structure)

    # Leaf: ScalarExpr
    if isinstance(expr, ScalarExpr):
        return ScalarExpr(0, expr.structure)

    op_name = expr.op.name
    children = expr.children
    s = expr.structure

    # Arithmetic operations
    if op_name == "add":
        return differentiate(children[0], var) + differentiate(children[1], var)

    if op_name == "sub":
        return differentiate(children[0], var) - differentiate(children[1], var)

    if op_name == "mul":
        # Product rule: (fg)' = f'g + fg'
        f, g = children
        return differentiate(f, var) * g + f * differentiate(g, var)

    if op_name == "div":
        # Quotient rule: (f/g)' = (f'g - fg') / g²
        f, g = children
        return (differentiate(f, var) * g - f * differentiate(g, var)) / (g ** 2)

    if op_name == "pow":
        f, n = children
        if isinstance(n, ScalarExpr):
            # Power rule: (f^n)' = n * f^(n-1) * f'
            nval = n.scalar_value
            return ScalarExpr(nval, s) * (f ** ScalarExpr(nval - 1, s)) * differentiate(f, var)
        # General case: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        from ..struct.lin_alg.fxn.ops import log as fxn_log
        return expr * (
            differentiate(n, var) * fxn_log(f)
            + n * differentiate(f, var) / f
        )

    if op_name == "neg":
        return -differentiate(children[0], var)

    # Elementary functions (chain rule)
    if op_name == "sin":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import cos as fxn_cos
        return fxn_cos(f) * differentiate(f, var)

    if op_name == "cos":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import sin as fxn_sin
        return -(fxn_sin(f) * differentiate(f, var))

    if op_name == "tan":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import cos as fxn_cos
        return differentiate(f, var) / (fxn_cos(f) ** 2)

    if op_name == "exp":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import exp as fxn_exp
        return fxn_exp(f) * differentiate(f, var)

    if op_name == "log":
        f = children[0]
        return differentiate(f, var) / f

    if op_name == "sqrt":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import sqrt as fxn_sqrt
        return differentiate(f, var) / (ScalarExpr(2, s) * fxn_sqrt(f))

    if op_name == "abs":
        raise NotImplementedError("Derivative of abs() is not defined at zero")

    if op_name == "asin":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import sqrt as fxn_sqrt
        return differentiate(f, var) / fxn_sqrt(ScalarExpr(1, s) - f ** 2)

    if op_name == "acos":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import sqrt as fxn_sqrt
        return -(differentiate(f, var) / fxn_sqrt(ScalarExpr(1, s) - f ** 2))

    if op_name == "atan":
        f = children[0]
        return differentiate(f, var) / (ScalarExpr(1, s) + f ** 2)

    if op_name == "sinh":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import cosh as fxn_cosh
        return fxn_cosh(f) * differentiate(f, var)

    if op_name == "cosh":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import sinh as fxn_sinh
        return fxn_sinh(f) * differentiate(f, var)

    if op_name == "tanh":
        f = children[0]
        from ..struct.lin_alg.fxn.ops import cosh as fxn_cosh
        return differentiate(f, var) / (fxn_cosh(f) ** 2)

    raise NotImplementedError(
        f"Differentiation not implemented for operation '{op_name}'"
    )


def differentiate_n(expr, var, order=1):
    """Apply differentiate() n times for higher-order derivatives."""
    result = expr
    for _ in range(order):
        result = differentiate(result, var)
    return result
