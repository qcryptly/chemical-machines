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

__all__ = ['tensor', 'vector', 'matrix', 'scalar', 'LINEAR_ALGEBRA']

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
        shape: Tensor shape tuple, e.g. (3,3). None for lazy/abstract.
        dtype: A cm.math.numbers Dtype (e.g. numbers.float64).
        name: Variable name for LaTeX rendering. Auto-generated if None.
        value: Concrete numpy array or scalar. If provided, tensor is eager.

    Returns:
        A SymbolicTensor if shape is given with no value (supports element
        assignment), or a Var if a concrete value is provided.
    """
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
        var_name=name,
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
    if value is not None and name is None:
        return ScalarExpr(value, LINEAR_ALGEBRA)
    if name is None:
        Var._var_counter += 1
        name = f"s_{Var._var_counter}"
    return Var(
        var_name=name,
        structure=LINEAR_ALGEBRA,
        value=value,
        shape=(),
        dtype=dtype,
        is_tensor=False,
    )


# Register backends on import
register_lin_alg_ops()

# Sub-modules
from . import fxn
