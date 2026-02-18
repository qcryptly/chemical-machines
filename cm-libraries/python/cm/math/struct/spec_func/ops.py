"""
Special function operations and backend registrations.

Operations: kron_delta (Kronecker delta)
"""

from __future__ import annotations
from ..base import Expression, Var, ScalarExpr, Operation, _ensure_expression


# ---- Operation definitions ----

import numbers

_kron_delta_op = Operation("kron_delta", arity=2, latex_name="delta")


def krok_delta(a, b) -> Expression:
    """
    Kronecker delta: δ(a, b) = 1 if a == b, else 0.

    Returns a symbolic Expression that evaluates to 1 when both
    inputs have equal values, 0 otherwise.

    Args:
        a: An Expression (typically a scalar Var).
        b: An Expression (typically a scalar Var).

    Returns:
        Expression with kron_delta operation.
    """
    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        return ScalarExpr(a) == ScalarExpr(b)
    if not isinstance(a, Expression):
        raise TypeError(f"Expected Expression, got {type(a).__name__}")
    if not isinstance(b, Expression):
        b = _ensure_expression(b, a.structure)

    return Expression(
        op=_kron_delta_op,
        children=[a, b],
        structure=a.structure,
        metadata={'shape': (), 'is_scalar': True},
    )


# ---- Backend registrations ----

def register_spec_func_ops():
    """Register special function operations with all three backends."""
    _register_eager()
    _register_torch()
    _register_latex()


def _register_eager():
    from ..backends.eager_be import eager_backend
    import numpy as np

    S = "linear_algebra"

    def _eager_kron_delta(a, b):
        return np.float64(1.0) if np.array_equal(np.asarray(a), np.asarray(b)) else np.float64(0.0)

    eager_backend.register(S, "kron_delta", _eager_kron_delta)


def _register_torch():
    from ..backends.torch_be import torch_backend

    S = "linear_algebra"

    def _torch_kron_delta(a, b):
        import torch
        if torch.equal(a, b):
            return torch.tensor(1.0, dtype=torch.float64, device=a.device)
        return torch.tensor(0.0, dtype=torch.float64, device=a.device)

    torch_backend.register(S, "kron_delta", _torch_kron_delta)


def _register_latex():
    from ..backends.latex_be import latex_backend

    S = "linear_algebra"

    def _latex_kron_delta(child_a, child_b, **kw):
        return rf"\delta_{{{child_a} {child_b}}}"

    latex_backend.register(S, "kron_delta", _latex_kron_delta)
