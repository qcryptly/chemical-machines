"""
Linear algebra structure for cm.math.

Factory functions for creating tensor expressions:
    tensor(shape, dtype, name, value)
    vector(dim, dtype, name, value)
    matrix(rows, cols, dtype, name, value)
    scalar(value, dtype, name)

Usage:
    from cm.math import struct, numbers
    A = struct.lin_alg.tensor(shape=(3,3), dtype=numbers.float64, name="A")
    B = struct.lin_alg.tensor(shape=(3,3), name="B")
    expr = (A @ B).det() + A.trace()
"""

from __future__ import annotations
import numpy as np

from ..base import Structure, Signature, Expression, Var, ScalarExpr
from .ops import register_lin_alg_ops, LIN_ALG_OPS
from .axioms import LIN_ALG_AXIOMS

__all__ = ['tensor', 'vector', 'matrix', 'scalar', 'diff', 'jacobian', 'hessian', 'LINEAR_ALGEBRA']

# Build the structure
_signature = Signature()
for _op in LIN_ALG_OPS:
    _signature.add(_op)

LINEAR_ALGEBRA = Structure(
    name="linear_algebra",
    signature=_signature,
    axioms=LIN_ALG_AXIOMS,
)


def tensor(shape=None, dtype=None, name=None, value=None):
    """
    Create a tensor expression in the linear algebra structure.

    Args:
        shape: Tensor shape tuple, e.g. (3,3). Can also pass a numpy array
               or nested list as the first arg for immediate value binding.
        dtype: A cm.math.numbers Dtype (e.g. numbers.float64).
        name: Variable name for LaTeX rendering. Auto-generated if None.
        value: Concrete numpy array or scalar. If provided, tensor is eager.

    Returns:
        A SymbolicTensor with concrete values if an array is passed as
        shape or value; otherwise a SymbolicTensor for element assignment,
        or a Var if a concrete value is provided with no shape.
    """
    # Detect array-like first arg: treat as immediate value binding
    if shape is not None and not isinstance(shape, tuple):
        arr = np.asarray(shape)
        if arr.ndim > 0:
            if name is None:
                Var._var_counter += 1
                name = f"T_{Var._var_counter}"
            from ...tensor import SymbolicTensor
            st = SymbolicTensor(
                shape=tuple(arr.shape),
                structure=LINEAR_ALGEBRA,
                name=name,
                dtype=dtype,
            )
            st.value = arr
            return st

    if name is None:
        Var._var_counter += 1
        name = f"T_{Var._var_counter}"

    # If shape given but no value, return a SymbolicTensor for element assignment
    if shape is not None and value is None:
        from ...tensor import SymbolicTensor
        return SymbolicTensor(
            shape=shape,
            structure=LINEAR_ALGEBRA,
            name=name,
            dtype=dtype,
        )

    # value kwarg with array-like: return SymbolicTensor with concrete values
    if value is not None:
        arr = np.asarray(value)
        if arr.ndim > 0:
            from ...tensor import SymbolicTensor
            st = SymbolicTensor(
                shape=tuple(arr.shape),
                structure=LINEAR_ALGEBRA,
                name=name,
                dtype=dtype,
            )
            st.value = arr
            return st

    actual_shape = shape
    if value is not None and actual_shape is None:
        if hasattr(value, 'shape'):
            actual_shape = tuple(value.shape)
        elif isinstance(value, (int, float, complex)):
            actual_shape = ()

    if value is not None and not isinstance(value, np.ndarray):
        np_dtype = dtype.np_dtype if dtype else None
        value = np.asarray(value, dtype=np_dtype)

    return Var(
        name=name,
        structure=LINEAR_ALGEBRA,
        value=value,
        shape=actual_shape,
        dtype=dtype,
        is_tensor=True,
    )


def vector(dim=None, dtype=None, name=None, value=None):
    """Create a vector (rank-1 tensor)."""
    shape = (dim,) if dim is not None else None
    if name is None:
        Var._var_counter += 1
        name = f"v_{Var._var_counter}"
    return tensor(shape=shape, dtype=dtype, name=name, value=value)


def matrix(rows=None, cols=None, dtype=None, name=None, value=None):
    """Create a matrix (rank-2 tensor)."""
    shape = None
    if rows is not None and cols is not None:
        shape = (rows, cols)
    if name is None:
        Var._var_counter += 1
        name = f"M_{Var._var_counter}"
    return tensor(shape=shape, dtype=dtype, name=name, value=value)


def scalar(value=None, dtype=None, name=None):
    """Create a scalar expression."""
    # If value is a string, treat it as a name (convenience shorthand)
    if isinstance(value, str):
        name = value
        value = None
    if value is not None and name is None:
        return ScalarExpr(value, LINEAR_ALGEBRA)
    if name is None:
        Var._var_counter += 1
        name = f"s_{Var._var_counter}"
    return Var(
        name=name,
        structure=LINEAR_ALGEBRA,
        value=value,
        shape=(),
        dtype=dtype,
        is_tensor=False,
    )


def diff(expr, vars=None, order=1):
    """Compute the Nth-order differential of an expression or symbolic tensor.

    Args:
        expr: A scalar Expression or SymbolicTensor.
        vars: Tuple of Var nodes to differentiate with respect to.
               If None, auto-collects free variables (sorted by name).
        order: Derivative order. 1 = Jacobian/gradient, 2 = Hessian, etc.

    Returns:
        SymbolicTensor with shape input_shape + (n_vars,) * order.
    """
    import itertools
    from ...tensor import SymbolicTensor
    from ...operator.diff import differentiate

    # Handle scalar Expression input
    if isinstance(expr, Expression):
        if vars is None:
            vars = tuple(sorted(expr._get_free_variables(), key=lambda v: v.name))
        n = len(vars)
        out_shape = (n,) * order
        result = SymbolicTensor(shape=out_shape, structure=LINEAR_ALGEBRA)
        for idx in itertools.product(range(n), repeat=order):
            d = expr
            for dim in idx:
                d = differentiate(d, vars[dim])
            result._elements[idx] = d
        result._is_composite = True
        return result

    # Handle SymbolicTensor input
    if isinstance(expr, SymbolicTensor):
        if vars is None:
            vars = expr.free_vars()
        n = len(vars)
        in_shape = expr.shape
        out_shape = in_shape + (n,) * order
        result = SymbolicTensor(shape=out_shape, structure=expr._structure)
        for in_idx in itertools.product(*(range(s) for s in in_shape)):
            elem = expr._get_element(*in_idx)
            for var_idx in itertools.product(range(n), repeat=order):
                d = elem
                for dim in var_idx:
                    d = differentiate(d, vars[dim])
                result._elements[in_idx + var_idx] = d
            result._is_composite = True
        return result

    raise TypeError(f"diff() expects Expression or SymbolicTensor, got {type(expr).__name__}")


def jacobian(expr, vars=None):
    """First-order differential (Jacobian). Alias for diff(expr, vars, order=1)."""
    return diff(expr, vars=vars, order=1)


def hessian(expr, vars=None):
    """Second-order differential (Hessian). Alias for diff(expr, vars, order=2)."""
    return diff(expr, vars=vars, order=2)


# Register backends on import
register_lin_alg_ops()

# Sub-modules
from . import fxn
