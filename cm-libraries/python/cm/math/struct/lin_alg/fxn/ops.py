"""
Elementary mathematical function operations and backend registrations.

Operations: sin, cos, tan, exp, log, sqrt, abs,
            asin, acos, atan, sinh, cosh, tanh
"""

from __future__ import annotations
from ...base import Expression, Operation, _ensure_expression


# ---- Operation definitions ----

_sin_op = Operation("sin", arity=1, latex_name="sin")
_cos_op = Operation("cos", arity=1, latex_name="cos")
_tan_op = Operation("tan", arity=1, latex_name="tan")
_exp_op = Operation("exp", arity=1, latex_name="exp")
_log_op = Operation("log", arity=1, latex_name="ln")
_sqrt_op = Operation("sqrt", arity=1, latex_name="sqrt")
_abs_op = Operation("abs", arity=1, latex_name="abs")
_asin_op = Operation("asin", arity=1, latex_name="arcsin")
_acos_op = Operation("acos", arity=1, latex_name="arccos")
_atan_op = Operation("atan", arity=1, latex_name="arctan")
_sinh_op = Operation("sinh", arity=1, latex_name="sinh")
_cosh_op = Operation("cosh", arity=1, latex_name="cosh")
_tanh_op = Operation("tanh", arity=1, latex_name="tanh")


# ---- Factory functions ----

def _make_fxn(op, expr):
    if not isinstance(expr, Expression):
        expr = _ensure_expression(expr, None)
    return Expression(
        op=op,
        children=[expr],
        structure=expr.structure,
        metadata={'shape': expr.metadata.get('shape', ()), 'is_scalar': expr.metadata.get('is_scalar', True)},
    )


def sin(expr):
    """Sine function."""
    return _make_fxn(_sin_op, expr)

def cos(expr):
    """Cosine function."""
    return _make_fxn(_cos_op, expr)

def tan(expr):
    """Tangent function."""
    return _make_fxn(_tan_op, expr)

def exp(expr):
    """Exponential function."""
    return _make_fxn(_exp_op, expr)

def log(expr):
    """Natural logarithm."""
    return _make_fxn(_log_op, expr)

def sqrt(expr):
    """Square root."""
    return _make_fxn(_sqrt_op, expr)

def fabs(expr):
    """Absolute value."""
    return _make_fxn(_abs_op, expr)

def asin(expr):
    """Inverse sine."""
    return _make_fxn(_asin_op, expr)

def acos(expr):
    """Inverse cosine."""
    return _make_fxn(_acos_op, expr)

def atan(expr):
    """Inverse tangent."""
    return _make_fxn(_atan_op, expr)

def sinh(expr):
    """Hyperbolic sine."""
    return _make_fxn(_sinh_op, expr)

def cosh(expr):
    """Hyperbolic cosine."""
    return _make_fxn(_cosh_op, expr)

def tanh(expr):
    """Hyperbolic tangent."""
    return _make_fxn(_tanh_op, expr)


# ---- Backend registrations ----

def register_fxn_ops():
    """Register all fxn operations with all three backends."""
    _register_eager()
    _register_torch()
    _register_latex()


def _register_eager():
    from ...backends.eager_be import eager_backend
    import numpy as np

    S = "linear_algebra"
    eager_backend.register(S, "sin", lambda a: np.sin(a))
    eager_backend.register(S, "cos", lambda a: np.cos(a))
    eager_backend.register(S, "tan", lambda a: np.tan(a))
    eager_backend.register(S, "exp", lambda a: np.exp(a))
    eager_backend.register(S, "log", lambda a: np.log(a))
    eager_backend.register(S, "sqrt", lambda a: np.sqrt(a))
    eager_backend.register(S, "abs", lambda a: np.abs(a))
    eager_backend.register(S, "asin", lambda a: np.arcsin(a))
    eager_backend.register(S, "acos", lambda a: np.arccos(a))
    eager_backend.register(S, "atan", lambda a: np.arctan(a))
    eager_backend.register(S, "sinh", lambda a: np.sinh(a))
    eager_backend.register(S, "cosh", lambda a: np.cosh(a))
    eager_backend.register(S, "tanh", lambda a: np.tanh(a))


def _register_torch():
    from ...backends.torch_be import torch_backend

    S = "linear_algebra"

    def _make_torch(name):
        def impl(a):
            import torch
            return getattr(torch, name)(a)
        return impl

    torch_backend.register(S, "sin", _make_torch("sin"))
    torch_backend.register(S, "cos", _make_torch("cos"))
    torch_backend.register(S, "tan", _make_torch("tan"))
    torch_backend.register(S, "exp", _make_torch("exp"))
    torch_backend.register(S, "log", _make_torch("log"))
    torch_backend.register(S, "sqrt", _make_torch("sqrt"))
    torch_backend.register(S, "abs", _make_torch("abs"))
    torch_backend.register(S, "asin", _make_torch("asin"))
    torch_backend.register(S, "acos", _make_torch("acos"))
    torch_backend.register(S, "atan", _make_torch("atan"))
    torch_backend.register(S, "sinh", _make_torch("sinh"))
    torch_backend.register(S, "cosh", _make_torch("cosh"))
    torch_backend.register(S, "tanh", _make_torch("tanh"))


def _register_latex():
    from ...backends.latex_be import latex_backend

    S = "linear_algebra"

    def _make_latex(latex_name):
        def impl(child, **kw):
            return rf"\{latex_name}\left({child}\right)"
        return impl

    latex_backend.register(S, "sin", _make_latex("sin"))
    latex_backend.register(S, "cos", _make_latex("cos"))
    latex_backend.register(S, "tan", _make_latex("tan"))
    latex_backend.register(S, "exp", _make_latex("exp"))
    latex_backend.register(S, "log", _make_latex("ln"))
    latex_backend.register(S, "sqrt", lambda child, **kw: rf"\sqrt{{{child}}}")
    latex_backend.register(S, "abs", lambda child, **kw: rf"\left|{child}\right|")
    latex_backend.register(S, "asin", _make_latex("arcsin"))
    latex_backend.register(S, "acos", _make_latex("arccos"))
    latex_backend.register(S, "atan", _make_latex("arctan"))
    latex_backend.register(S, "sinh", _make_latex("sinh"))
    latex_backend.register(S, "cosh", _make_latex("cosh"))
    latex_backend.register(S, "tanh", _make_latex("tanh"))


# Auto-register on import
register_fxn_ops()
