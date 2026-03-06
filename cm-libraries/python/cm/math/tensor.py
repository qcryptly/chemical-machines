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
    def value(self):
        """Get concrete numpy array if all elements are ScalarExpr, else None."""
        import numpy as np
        # Determine dtype from first element
        dt = np.float64
        for expr in self._elements.values():
            if isinstance(expr, ScalarExpr) and isinstance(expr.scalar_value, complex):
                dt = np.complex128
                break
        result = np.empty(self._shape, dtype=dt)
        for idx in _iter_indices(self._shape):
            expr = self._elements.get(idx, ScalarExpr(0, self._structure))
            if isinstance(expr, ScalarExpr):
                result[idx] = expr.scalar_value
            else:
                return None
        return result

    @value.setter
    def value(self, arr):
        """Set concrete values from a numpy array or nested list."""
        import numpy as np
        arr = np.asarray(arr)
        if arr.shape != self._shape:
            raise ValueError(f"Shape mismatch: expected {self._shape}, got {arr.shape}")
        coerce = complex if np.issubdtype(arr.dtype, np.complexfloating) else float
        for idx in _iter_indices(self._shape):
            self._elements[idx] = ScalarExpr(coerce(arr[idx]), self._structure)
        self._is_composite = True

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

    # ---- Scalar broadcasting ----

    def __mul__(self, other):
        """Scalar * tensor broadcast: multiply each element by a scalar expression."""
        if isinstance(other, SymbolicTensor):
            raise TypeError("Use @ for matrix multiplication; element-wise mul between tensors not yet supported")
        if not isinstance(other, Expression):
            other = _ensure_expression(other, self._structure)
        result = SymbolicTensor(shape=self._shape, structure=self._structure)
        for idx in _iter_indices(self._shape):
            result._elements[idx] = self._get_element(*idx) * other
        result._is_composite = True
        return result

    def __rmul__(self, other):
        """tensor * scalar broadcast (commutative)."""
        if not isinstance(other, Expression):
            other = _ensure_expression(other, self._structure)
        result = SymbolicTensor(shape=self._shape, structure=self._structure)
        for idx in _iter_indices(self._shape):
            result._elements[idx] = other * self._get_element(*idx)
        result._is_composite = True
        return result

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

    def transpose(self, *axes):
        """Permute tensor indices.

        With no args: reverses all axes (standard transpose for 2D).
        With a permutation tuple: reorders axes accordingly.

        Examples:
            A.transpose()       # (m, n) -> (n, m)
            A.transpose(1, 0)   # same as above
            T.transpose(2, 0, 1) # (a, b, c) -> (c, a, b)

        Returns a new SymbolicTensor with permuted shape and elements.
        """
        ndim = len(self._shape)
        if not axes:
            perm = tuple(reversed(range(ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        else:
            perm = tuple(axes)
        if sorted(perm) != list(range(ndim)):
            raise ValueError(
                f"Invalid axis permutation {perm} for {ndim}D tensor"
            )
        new_shape = tuple(self._shape[p] for p in perm)
        result = SymbolicTensor(
            shape=new_shape, structure=self._structure, name=self._name,
            dtype=self._dtype,
        )
        for old_idx, expr in self._elements.items():
            new_idx = tuple(old_idx[p] for p in perm)
            result._elements[new_idx] = expr
        result._is_composite = self._is_composite
        result._bindings = dict(self._bindings)
        return result

    @property
    def T(self):
        """Transpose (reverses axes). Shorthand for .transpose()."""
        return self.transpose()

    def free_vars(self):
        """Return all free Var nodes across element expressions as a sorted tuple."""
        all_vars = set()
        for expr in self._elements.values():
            all_vars |= expr._get_free_variables()
        return tuple(sorted(all_vars, key=lambda v: v.name))

    def var_map(self):
        """Return a dict mapping var name -> Var for all free variables."""
        return {v.name: v for v in self.free_vars()}

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

    def bind_indices(self, *args, **kwargs):
        """Bind index ranges for iteration during evaluate().

        Positional args specify ranges per dimension:
            x.bind_indices(index.range(2), index.range(2))

        Keyword args bind named index variables (same as Expression):
            x.bind_indices(i=0, j=index.range(3))

        Returns self for chaining.
        """
        self._dim_ranges = args if args else None
        self._index_kwargs = kwargs
        return self

    def bind(self, value_or_dict=None, **kwargs):
        """Bind values to free variables in element expressions.

        Accepts:
            - A numpy array or nested list: sets concrete element values (like .value = ...)
            - A dict with Var keys: binds variables by Var reference
            - Keyword args: binds variables by name (resolved via caller scope)
        """
        import numpy as np
        if value_or_dict is not None:
            if isinstance(value_or_dict, dict):
                for key, val in value_or_dict.items():
                    if isinstance(key, Var):
                        self._bindings[key.name] = val
                    else:
                        self._bindings[key] = val
            elif isinstance(value_or_dict, (np.ndarray, list, tuple)):
                self.value = value_or_dict
                return self
            else:
                self._bindings[str(value_or_dict)] = value_or_dict
        self._bindings.update(_resolve_kwargs(kwargs))
        return self

    def unbind(self, *vars):
        """Remove variable bindings.

        With no args: clears all bindings.
        With Var args: removes only those variables.
        With string args: removes by name.
        """
        if not vars:
            self._bindings.clear()
        else:
            for v in vars:
                name = v.name if isinstance(v, Var) else str(v)
                self._bindings.pop(name, None)
        return self

    def evaluate(self, bindings_dict=None, **kwargs):
        """Evaluate all elements, return numpy array."""
        import numpy as np
        import itertools

        merged = {**self._bindings}
        if bindings_dict is not None:
            for key, val in bindings_dict.items():
                if isinstance(key, Var):
                    merged[key.name] = val
                else:
                    merged[key] = val
        merged.update(_resolve_kwargs(kwargs))

        # Use dimension ranges from bind_indices if set, else full shape
        dim_ranges = getattr(self, '_dim_ranges', None)
        if dim_ranges:
            ranges = [list(r) for r in dim_ranges]
            out_shape = tuple(len(r) for r in ranges)
            result = np.empty(out_shape, dtype=np.float64)
            for out_idx in itertools.product(*(range(s) for s in out_shape)):
                src_idx = tuple(ranges[d][out_idx[d]] for d in range(len(out_shape)))
                expr = self._elements.get(src_idx, ScalarExpr(0, self._structure))
                result[out_idx] = expr.evaluate(**merged)
        else:
            result = np.empty(self._shape, dtype=np.float64)
            for idx in _iter_indices(self._shape):
                expr = self._elements.get(idx, ScalarExpr(0, self._structure))
                result[idx] = expr.evaluate(**merged)
        return result

    # ---- LaTeX and rendering ----

    def to_latex(self):
        """Render as pmatrix LaTeX. Applies any stored bindings as substitutions."""
        if len(self._shape) != 2:
            raise NotImplementedError("to_latex only supports 2D tensors")
        rows = []
        for i in range(self._shape[0]):
            row_parts = []
            for j in range(self._shape[1]):
                elem = self._get_element(i, j)
                if self._bindings:
                    elem = elem.substitute(**self._bindings)
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
