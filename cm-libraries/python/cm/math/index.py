"""
Indexable expression support for cm.math.

Provides:
    index       - Symbolic index class (also a Var subclass)
    IndexRange  - Range specification for index iteration
    BoundIndexExpression - Expression with bound index ranges
    IndexResult - Single result from range evaluation

Usage:
    from cm.math import index
    i = index("i")
    j = index("j")
    z = krok_delta(x[i,j], y[i,j])
    result = z.bind_indices(i=0, j=index.range(0,2,1))
    result.evaluate()
"""

from __future__ import annotations
import itertools
import math
from typing import Optional, Dict, Any, List

from .struct.base import (
    Expression, Var, ScalarExpr, Operation, _ensure_expression,
)

__all__ = [
    'index', 'IndexedExpression',
    'IndexRange', 'BoundIndexExpression', 'IndexResult',
]


# ---- The index_access Operation (variadic) ----

_index_access_op = Operation("index_access", arity=-1)


# ---- IndexRange ----

class IndexRange:
    """
    A range specification for index iteration.
    Same API as np.arange: IndexRange(start, stop, step).
    """

    def __init__(self, start_or_stop, stop=None, step=1):
        if stop is None:
            self.start = 0
            self.stop = start_or_stop
        else:
            self.start = start_or_stop
            self.stop = stop
        self.step = step

    def __iter__(self):
        val = self.start
        while val < self.stop:
            yield val
            val += self.step

    def __len__(self):
        return max(0, math.ceil((self.stop - self.start) / self.step))

    def __repr__(self):
        return f"IndexRange({self.start}, {self.stop}, {self.step})"


# ---- IndexResult ----

class IndexResult:
    """A single evaluation result for one combination of index values."""

    def __init__(self, indices: Dict[str, int], value: Any):
        self.indices = indices
        self.value = value

    def __repr__(self):
        idx_str = ", ".join(f"{k}={v}" for k, v in self.indices.items())
        return f"IndexResult({idx_str}) -> {self.value}"


# ---- BoundIndexExpression ----

class BoundIndexExpression:
    """
    An expression with indices bound to concrete values or ranges.

    Created by Expression.bind_indices(). Calling .evaluate() iterates
    over all IndexRange bindings (cartesian product) and returns a list
    of IndexResult objects.
    """

    def __init__(self, expr: Expression, index_bindings: Dict[str, Any]):
        self._expr = expr
        self._index_bindings = index_bindings

    def evaluate(self, bindings_dict=None, **kwargs):
        # Resolve Var-keyed bindings to string keys
        resolved_kwargs = {}
        if bindings_dict is not None:
            for key, val in bindings_dict.items():
                if isinstance(key, Var):
                    resolved_kwargs[key.var_name] = val
                else:
                    resolved_kwargs[key] = val
        resolved_kwargs.update(kwargs)

        concrete = {}
        ranges = {}

        for name, val in self._index_bindings.items():
            if isinstance(val, IndexRange):
                ranges[name] = val
            else:
                concrete[name] = val

        if not ranges:
            merged = {**concrete, **resolved_kwargs}
            return self._expr.evaluate(**merged)

        range_names = list(ranges.keys())
        range_iters = [list(ranges[n]) for n in range_names]

        results = []
        for combo in itertools.product(*range_iters):
            idx_map = dict(zip(range_names, combo))
            merged = {**concrete, **idx_map, **resolved_kwargs}
            val = self._expr.evaluate(**merged)
            all_indices = {**{k: concrete[k] for k in concrete}, **idx_map}
            results.append(IndexResult(indices=all_indices, value=val))

        return results

    def __repr__(self):
        parts = [f"{k}={v}" for k, v in self._index_bindings.items()]
        return f"BoundIndexExpression({', '.join(parts)})"


# ---- IndexedExpression ----

class IndexedExpression(Expression):
    """
    An expression with first-class index information.

    Returned by tensor[i, j] or tensor[0][i]. Carries the base tensor
    and its indices as queryable attributes, while remaining a full
    Expression that composes with arithmetic, LaTeX, evaluate, etc.

    Attributes:
        base:     The root tensor expression being indexed.
        indices:  Tuple of index nodes (mix of index vars, ScalarExpr, Var).

    Properties:
        free_indices:     Symbolic index vars (unbound).
        concrete_indices: Concrete integer indices (ScalarExpr).
        var_indices:      Non-index Var nodes used as indices.

    Example:
        i, j = index("i"), index("j")
        A = struct.lin_alg.matrix(rows=3, cols=3, name="A")

        expr = A[i, j]
        expr.base          # Var("A")
        expr.indices        # (index('i'), index('j'))
        expr.free_indices   # (index('i'), index('j'))

        expr2 = A[0, j]
        expr2.concrete_indices  # (ScalarExpr(0),)
        expr2.free_indices      # (index('j'),)
    """

    def __init__(self, base, indices, structure, metadata=None):
        children = [base] + list(indices)
        super().__init__(
            op=_index_access_op,
            children=children,
            structure=structure,
            metadata=metadata,
        )
        self.base = base
        self.indices = tuple(indices)

    @property
    def free_indices(self):
        """Symbolic index variables (unbound)."""
        return tuple(
            idx for idx in self.indices
            if isinstance(idx, Var) and idx.metadata.get('is_index')
        )

    @property
    def concrete_indices(self):
        """Concrete (integer) index positions."""
        return tuple(
            idx for idx in self.indices
            if isinstance(idx, ScalarExpr)
        )

    @property
    def var_indices(self):
        """Non-index Var nodes used as indices (e.g., a scalar variable)."""
        return tuple(
            idx for idx in self.indices
            if isinstance(idx, Var) and not idx.metadata.get('is_index')
        )

    def substitute_indices(self, **kwargs):
        """
        Replace symbolic indices with concrete values, returning a new
        IndexedExpression (or evaluating immediately if fully concrete).

            A[i, j].substitute_indices(i=0)  ->  A[0, j]  (IndexedExpression)
            A[i, j].substitute_indices(i=0, j=1)  ->  A[0, 1]  (IndexedExpression)
        """
        new_indices = []
        for idx in self.indices:
            if (isinstance(idx, Var) and idx.metadata.get('is_index')
                    and idx.var_name in kwargs):
                new_indices.append(
                    ScalarExpr(kwargs[idx.var_name], self.structure)
                )
            else:
                new_indices.append(idx)

        return _make_index_access(self.base, tuple(new_indices))

    def _substitute(self, bindings):
        """Substitute into base and indices, rebuild IndexedExpression."""
        new_base = self.base._substitute(bindings)
        new_indices = tuple(idx._substitute(bindings) for idx in self.indices)
        return _make_index_access(new_base, new_indices)

    def __repr__(self):
        base_name = getattr(self.base, 'var_name', str(self.base))
        idx_strs = []
        for idx in self.indices:
            if isinstance(idx, ScalarExpr):
                idx_strs.append(str(idx.scalar_value))
            elif isinstance(idx, Var):
                idx_strs.append(idx.var_name)
            else:
                idx_strs.append(repr(idx))
        return f"{base_name}[{', '.join(idx_strs)}]"


# ---- index class (Var subclass) ----

class index(Var):
    """
    A symbolic index variable for tensor indexing.

    Usage:
        i = index("i")          # named symbolic index
        i = index()             # auto-named symbolic index
        index.range(0, 10, 1)   # range specification (same as np.arange)

    An index is a Var with value=None, so it appears as a free variable
    in expressions and can be bound via bind_indices() or evaluate().
    """

    _index_counter: int = 0

    def __init__(self, name: Optional[str] = None):
        if name is None:
            index._index_counter += 1
            name = f"_idx_{index._index_counter}"

        from .struct.lin_alg import LINEAR_ALGEBRA

        super().__init__(
            var_name=name,
            structure=LINEAR_ALGEBRA,
            value=None,
            shape=(),
            dtype=None,
            is_tensor=False,
        )
        self.metadata['is_index'] = True

    @staticmethod
    def range(start_or_stop, stop=None, step=1):
        """Create an IndexRange (same API as np.arange)."""
        return IndexRange(start_or_stop, stop, step)

    def _get_free_variables(self):
        return {self}

    def __repr__(self):
        return f"index({self.var_name!r})"


# ---- Helper for Expression.__getitem__ ----

def _make_index_access(tensor_expr, indices_tuple):
    """
    Create an IndexedExpression from tensor[i, j, ...].

    Chained indexing is flattened: A[i][j] produces the same expression
    as A[i, j], so that LaTeX rendering and evaluation stay simple.

    Args:
        tensor_expr: The tensor Expression being indexed.
        indices_tuple: Tuple of index expressions (index, int, ScalarExpr).

    Returns:
        IndexedExpression with base, indices, and proper shape metadata.
    """
    # Flatten chained index_access: if tensor_expr is already an
    # IndexedExpression, unwrap it and prepend its indices.
    if isinstance(tensor_expr, IndexedExpression):
        return _make_index_access(
            tensor_expr.base,
            tensor_expr.indices + indices_tuple,
        )

    # Validate: tensor must have shape metadata to be indexable
    tensor_shape = tensor_expr.metadata.get('shape')
    if tensor_shape is not None and len(tensor_shape) == 0:
        raise IndexError(
            f"Cannot index a scalar expression"
            f" (shape {tensor_shape})"
        )

    # Normalize indices to Expression nodes
    normalized = []
    for idx in indices_tuple:
        if isinstance(idx, Expression):
            normalized.append(idx)
        elif isinstance(idx, (int, float)):
            normalized.append(ScalarExpr(idx, tensor_expr.structure))
        else:
            normalized.append(_ensure_expression(idx, tensor_expr.structure))

    # Compute result shape
    n_indices = len(normalized)
    result_meta = {}
    if tensor_shape is not None:
        if n_indices > len(tensor_shape):
            raise IndexError(
                f"Too many indices ({n_indices}) for tensor of"
                f" rank {len(tensor_shape)} (shape {tensor_shape})"
            )
        if n_indices == len(tensor_shape):
            result_meta = {'shape': (), 'is_scalar': True}
        else:
            result_meta = {
                'shape': tensor_shape[n_indices:],
                'is_tensor': True,
            }

    return IndexedExpression(
        base=tensor_expr,
        indices=normalized,
        structure=tensor_expr.structure,
        metadata=result_meta,
    )


# ---- Backend registrations ----

def _register_index_backends():
    _register_eager()
    _register_torch()
    _register_latex()


def _register_eager():
    from .struct.backends.eager_be import eager_backend
    import numpy as np

    S = "linear_algebra"

    def _eager_index_access(tensor, *indices):
        idx_tuple = tuple(int(i) for i in indices)
        return tensor[idx_tuple]

    eager_backend.register(S, "index_access", _eager_index_access)


def _register_torch():
    from .struct.backends.torch_be import torch_backend

    S = "linear_algebra"

    def _torch_index_access(tensor, *indices):
        import torch
        idx_tuple = tuple(
            int(i.item()) if isinstance(i, torch.Tensor) else int(i)
            for i in indices
        )
        return tensor[idx_tuple]

    torch_backend.register(S, "index_access", _torch_index_access)


def _register_latex():
    from .struct.backends.latex_be import latex_backend

    S = "linear_algebra"

    def _latex_index_access(*children, **kw):
        tensor_latex = children[0]
        index_latexes = children[1:]
        subscript = ", ".join(str(i) for i in index_latexes)
        # Wrap base in braces if it already contains a subscript,
        # e.g. T_1 -> {T_1}_{i,j} not T_1_{i,j}
        if '_' in tensor_latex:
            return f"{{{tensor_latex}}}_{{{subscript}}}"
        return f"{tensor_latex}_{{{subscript}}}"

    latex_backend.register(S, "index_access", _latex_index_access)


# Auto-register on import
_register_index_backends()
