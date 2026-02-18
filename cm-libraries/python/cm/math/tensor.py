"""
SymbolicTensor — a tensor whose elements are symbolic Expressions.

Supports element assignment, symbolic determinant, trace, matmul,
and evaluation to numpy arrays.

Usage:
    from cm.math import struct
    y = struct.lin_alg.scalar(name="y")
    x = struct.lin_alg.tensor(shape=(2,2))
    x[0][0] = y ** 2 + 5
    x[0][1] = y ** 3
    x[1][0] = y
    x[1][1] = y ** 2
    print(x.det().bind(y=3).evaluate())
"""

from __future__ import annotations
import inspect
from typing import Optional, Dict, Tuple, Any

from .struct.base import (
    Expression, Var, ScalarExpr, _ensure_expression, _resolve_kwargs,
)

__all__ = ['SymbolicTensor', 'SymbolicTensorSlice']


class SymbolicTensorSlice:
    """View into a SymbolicTensor for chained indexing (x[0][1] = expr)."""

    def __init__(self, tensor, prefix):
        self._tensor = tensor
        self._prefix = prefix

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        full = self._prefix + key
        self._tensor[full] = value

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        full = self._prefix + key
        if len(full) < len(self._tensor._shape):
            return SymbolicTensorSlice(self._tensor, full)
        return self._tensor[full]

    def __repr__(self):
        return f"SymbolicTensorSlice(prefix={self._prefix})"


class SymbolicTensor:
    """A tensor with symbolic expression elements.

    Elements are stored sparsely in a dict keyed by index tuples.
    Unset elements default to ScalarExpr(0).
    """

    def __init__(self, shape, structure, name=None, dtype=None):
        self._shape = shape
        self._structure = structure
        self._name = name
        self._dtype = dtype
        self._elements: Dict[Tuple[int, ...], Expression] = {}
        self._bindings: Dict[str, Any] = {}
        self._is_composite = False

    @property
    def shape(self):
        return self._shape

    @property
    def structure(self):
        return self._structure

    @property
    def metadata(self):
        return {
            'shape': self._shape,
            'name': self._name,
            'is_tensor': True,
            'dtype': self._dtype,
        }

    # ---- Element access ----

    def _get_element(self, *idx):
        """Get element expression at index, defaulting to zero."""
        return self._elements.get(idx, ScalarExpr(0, self._structure))

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        from .operator import OperatorExpr
        if not isinstance(value, (Expression, OperatorExpr)):
            value = _ensure_expression(value, self._structure)
        self._elements[key] = value
        self._is_composite = True

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if self._is_composite:
            if len(key) < len(self._shape):
                return SymbolicTensorSlice(self, key)
            return self._get_element(*key)
        # Fall back: partial index returns slice
        if len(key) < len(self._shape):
            return SymbolicTensorSlice(self, key)
        return self._get_element(*key)

    # ---- Symbolic linear algebra ----

    def det(self):
        """Symbolic determinant via cofactor expansion."""
        if len(self._shape) != 2 or self._shape[0] != self._shape[1]:
            raise ValueError(f"det() requires square matrix, got shape {self._shape}")
        n = self._shape[0]
        rows = list(range(n))
        cols = list(range(n))
        return self._cofactor_det(rows, cols)

    def _cofactor_det(self, rows, cols):
        if len(rows) == 1:
            return self._get_element(rows[0], cols[0])
        if len(rows) == 2:
            a = self._get_element(rows[0], cols[0])
            b = self._get_element(rows[0], cols[1])
            c = self._get_element(rows[1], cols[0])
            d = self._get_element(rows[1], cols[1])
            return a * d - b * c
        result = None
        for j_idx, j in enumerate(cols):
            minor_cols = cols[:j_idx] + cols[j_idx + 1:]
            cofactor = self._cofactor_det(rows[1:], minor_cols)
            term = self._get_element(rows[0], j) * cofactor
            if j_idx % 2 == 0:
                result = term if result is None else result + term
            else:
                result = result - term if result is not None else -term
        return result

    def trace(self):
        """Symbolic trace (sum of diagonal elements)."""
        if len(self._shape) != 2 or self._shape[0] != self._shape[1]:
            raise ValueError(f"trace() requires square matrix, got shape {self._shape}")
        n = self._shape[0]
        result = self._get_element(0, 0)
        for i in range(1, n):
            result = result + self._get_element(i, i)
        return result

    def __matmul__(self, other):
        """Symbolic matrix multiplication."""
        if not isinstance(other, SymbolicTensor):
            raise TypeError(f"Cannot matmul SymbolicTensor with {type(other).__name__}")
        if self._shape[1] != other._shape[0]:
            raise ValueError(
                f"Shape mismatch for matmul: {self._shape} @ {other._shape}"
            )
        rows = self._shape[0]
        inner = self._shape[1]
        cols = other._shape[1]
        result = SymbolicTensor(
            shape=(rows, cols),
            structure=self._structure,
        )
        for i in range(rows):
            for j in range(cols):
                total = None
                for k in range(inner):
                    a = self._get_element(i, k)
                    b = other._get_element(k, j)
                    # Check for operator expressions
                    from .operator import OperatorExpr
                    if isinstance(a, OperatorExpr):
                        term = a.apply(b)
                    else:
                        term = a * b
                    total = term if total is None else total + term
                result._elements[(i, j)] = total
        result._is_composite = True
        return result

    # ---- Binding and evaluation ----

    def bind(self, value_or_dict=None, **kwargs):
        """Bind values to free variables in element expressions.

        Accepts dict with Var keys, or kwargs (resolved via caller scope).
        """
        if value_or_dict is not None and isinstance(value_or_dict, dict):
            for key, val in value_or_dict.items():
                if isinstance(key, Var):
                    self._bindings[key.var_name] = val
                else:
                    self._bindings[key] = val
        self._bindings.update(_resolve_kwargs(kwargs))
        return self

    def evaluate(self, bindings_dict=None, **kwargs):
        """Evaluate all elements, return numpy array."""
        import numpy as np

        merged = {**self._bindings}
        if bindings_dict is not None:
            for key, val in bindings_dict.items():
                if isinstance(key, Var):
                    merged[key.var_name] = val
                else:
                    merged[key] = val
        merged.update(_resolve_kwargs(kwargs))

        result = np.empty(self._shape, dtype=np.float64)
        for idx in _iter_indices(self._shape):
            expr = self._elements.get(idx, ScalarExpr(0, self._structure))
            result[idx] = expr.evaluate(**merged)
        return result

    # ---- LaTeX and rendering ----

    def to_latex(self):
        """Render as pmatrix LaTeX."""
        if len(self._shape) != 2:
            raise NotImplementedError("to_latex only supports 2D tensors")
        rows = []
        for i in range(self._shape[0]):
            row_parts = []
            for j in range(self._shape[1]):
                elem = self._get_element(i, j)
                row_parts.append(elem.to_latex())
            rows.append(" & ".join(row_parts))
        body = r" \\ ".join(rows)
        return rf"\begin{{pmatrix}} {body} \end{{pmatrix}}"

    def render(self, display=True, justify="center"):
        """Render as MathJax HTML output."""
        from .. import views
        latex_str = self.to_latex()
        ds, de = (r"\[", r"\]") if display else (r"\(", r"\)")
        html = (f'<div class="cm-math cm-math-{justify}" '
                f'style="line-height: 1.5;">{ds} {latex_str} {de}</div>')
        views.html(html)

    def __repr__(self):
        name = self._name or "unnamed"
        n_set = len(self._elements)
        return f"SymbolicTensor(shape={self._shape}, name={name!r}, elements={n_set})"


def _iter_indices(shape):
    """Iterate over all index tuples for a given shape."""
    if len(shape) == 0:
        yield ()
        return
    import itertools
    for idx in itertools.product(*(range(s) for s in shape)):
        yield idx
