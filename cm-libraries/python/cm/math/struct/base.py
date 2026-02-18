"""
Core base classes for the structure-based expression system.

Defines: Operation, Signature, Structure, Expression, Var, ScalarExpr
"""

from __future__ import annotations
import inspect
from typing import Optional, List, Dict, Set, Tuple, Any, Union

__all__ = [
    'Operation', 'Signature', 'Structure',
    'Expression', 'Var', 'ScalarExpr',
]

# Lazy imports
_np = None


def _get_np():
    global _np
    if _np is None:
        import numpy
        _np = numpy
    return _np


def _resolve_kwargs(kwargs, stack_offset=2):
    """Resolve kwargs where keys may refer to Var objects in the caller's scope.

    When a user writes ``expr.bind(x=value)``, Python passes ``{"x": value}``.
    If the caller has a local variable ``x`` that is a ``Var``, we use its
    ``var_name`` (which may be an auto-generated name like ``T_1``) as the
    binding key instead of the literal string ``"x"``.
    """
    frame = inspect.currentframe()
    try:
        caller = frame
        for _ in range(stack_offset):
            caller = caller.f_back
        caller_locals = caller.f_locals
    finally:
        del frame

    resolved = {}
    for key, val in kwargs.items():
        caller_var = caller_locals.get(key)
        if isinstance(caller_var, Var):
            resolved[caller_var.var_name] = val
        else:
            resolved[key] = val
    return resolved


class Operation:
    """A named operation within a structure."""

    def __init__(self, name, arity, signature=None, result_shape_fn=None,
                 latex_name=None):
        self.name = name
        self.arity = arity
        self.signature = signature
        self.result_shape_fn = result_shape_fn
        self.latex_name = latex_name

    def __repr__(self):
        return f"Operation({self.name!r}, arity={self.arity})"

    def __eq__(self, other):
        return isinstance(other, Operation) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Signature:
    """The full set of operations defined on a structure."""

    def __init__(self, ops=None):
        self.ops: Dict[str, Operation] = ops or {}

    def add(self, op):
        self.ops[op.name] = op

    def get(self, name):
        return self.ops.get(name)

    def has(self, name):
        return name in self.ops

    def __contains__(self, name):
        return name in self.ops


class Structure:
    """
    Base class for all mathematical structures.
    Defines a name, signature (set of operations), and axioms.
    """

    def __init__(self, name, signature, axioms=None):
        self.name = name
        self.signature = signature
        self.axioms = axioms or []

    def has_op(self, op_name):
        return self.signature.has(op_name)

    def get_op(self, op_name):
        op = self.signature.get(op_name)
        if op is None:
            raise ValueError(
                f"Operation '{op_name}' is not defined on structure '{self.name}'"
            )
        return op

    def __repr__(self):
        return f"Structure({self.name!r})"


class Expression:
    """
    A node in the symbolic expression DAG.

    Every operation on a structure instance returns a new Expression.
    Unifying type across all four interfaces (torch, eager, LaTeX, composition).
    """

    def __init__(self, op, children, structure, metadata=None):
        self.op = op
        self.children = children
        self.structure = structure
        self.metadata = metadata or {}
        self._bindings = {}
        self._index_bindings = {}
        from .constraints import ConstraintSet
        self.constraints = ConstraintSet()

    # ---- Variable binding ----

    def bind(self, bindings_dict=None, **kwargs):
        """
        Bind values to free variables in this expression.

        Can be called incrementally. Returns self for chaining.
        Bound values are used as defaults by evaluate() and to_torch().

        Accepts either a dict mapping Var objects to values, or keyword
        arguments where keys match variable names:

            w.bind({x: np.array([1,2])})    # Var object as key
            w.bind(x=np.array([1,2]))       # name string as key (resolves via caller scope)
        """
        if bindings_dict is not None:
            for key, val in bindings_dict.items():
                if isinstance(key, Var):
                    self._bindings[key.var_name] = val
                else:
                    self._bindings[key] = val
        self._bindings.update(_resolve_kwargs(kwargs))
        return self

    # ---- The four interfaces ----

    def to_latex(self):
        from .backends.latex_be import latex_backend
        return latex_backend.render(self)

    def evaluate(self, bindings_dict=None, **kwargs):
        from .backends.eager_be import eager_backend
        merged = {**self._bindings}
        if bindings_dict is not None:
            for key, val in bindings_dict.items():
                if isinstance(key, Var):
                    merged[key.var_name] = val
                else:
                    merged[key] = val
        merged.update(_resolve_kwargs(kwargs))

        if self._index_bindings:
            from ..index import IndexRange, IndexResult
            import itertools

            concrete = {}
            ranges = {}
            for name, val in self._index_bindings.items():
                if isinstance(val, IndexRange):
                    ranges[name] = val
                else:
                    concrete[name] = val
            merged.update(concrete)

            if ranges:
                range_names = list(ranges.keys())
                range_iters = [list(ranges[n]) for n in range_names]
                results = []
                for combo in itertools.product(*range_iters):
                    idx_map = dict(zip(range_names, combo))
                    eval_merged = {**merged, **idx_map}
                    val = eager_backend.evaluate(self, eval_merged)
                    all_indices = {**concrete, **idx_map}
                    results.append(IndexResult(indices=all_indices, value=val))
                return results

        return eager_backend.evaluate(self, merged)

    def to_torch(self, device="cpu"):
        from .backends.torch_be import torch_backend
        return torch_backend.compile(self, device=device, bindings=self._bindings)

    def render(self, display=True, justify="center"):
        from ... import views
        latex_str = self.to_latex()
        ds, de = (r"\[", r"\]") if display else (r"\(", r"\)")
        html = (f'<div class="cm-math cm-math-{justify}" '
                f'style="line-height: 1.5;">{ds} {latex_str} {de}</div>')
        views.html(html)

    def _repr_latex_(self):
        return f"$${self.to_latex()}$$"

    # ---- Metadata accessors ----

    @property
    def shape(self):
        return self.metadata.get('shape')

    @property
    def dtype(self):
        return self.metadata.get('dtype')

    @property
    def name(self):
        return self.metadata.get('name')

    # ---- Free variable collection ----

    def _get_free_variables(self):
        result = set()
        for child in self.children:
            result |= child._get_free_variables()
        return result

    # ---- Operator overloading ----

    def __eq__ (self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("eq", self, other)

    def __add__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("add", self, other)

    def __radd__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("add", other, self)

    def __sub__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("sub", self, other)

    def __rsub__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("sub", other, self)

    def __mul__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("mul", self, other)

    def __rmul__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("mul", other, self)

    def __matmul__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("matmul", self, other)

    def __truediv__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("div", self, other)

    def __rtruediv__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("div", other, self)

    def __pow__(self, other):
        other = _ensure_expression(other, self.structure)
        return _make_binop("pow", self, other)

    def __neg__(self):
        return _make_unaryop("neg", self)

    # ---- Indexing support ----

    def __getitem__(self, key):
        """Index into this expression: x[i, j] or x[0, 1]."""
        from ..index import _make_index_access
        if not isinstance(key, tuple):
            key = (key,)
        return _make_index_access(self, key)

    def bind_indices(self, index_map=None, **kwargs):
        """
        Bind index variables to concrete values or ranges.

        Stores index bindings on self. Returns self for chaining.
        When evaluate() is called, any IndexRange bindings produce a
        cartesian-product iteration returning a list of IndexResult.

        Two forms:
            z.bind_indices(i=0, j=index.range(0,2))   # kwargs (names must match index var_names)
            z.bind_indices({i: 0, j: index.range(0,2)}) # dict (keys are index objects)
        """
        if index_map is not None:
            for idx_obj, val in index_map.items():
                self._index_bindings[idx_obj.var_name] = val
        for k, v in kwargs.items():
            self._index_bindings[k] = v

        return self

    # ---- Substitution (partial evaluation) ----

    def substitute(self, **kwargs):
        """
        Create a new expression tree with variables replaced by concrete values.

        Unlike bind() which stores values for later, substitute() rewrites
        the tree — bound vars become concrete Var nodes (literals).

            expr = A[i, j] + B[i, j]
            partial = expr.substitute(A=np.array([[1,2],[3,4]]))
            # A is now a concrete literal, B is still symbolic
            partial.to_latex()       # shows concrete A, symbolic B
            partial.evaluate(B=..., i=0, j=1)  # only B needed
        """
        return self._substitute(kwargs)

    def _substitute(self, bindings):
        """Recursive substitution — override in subclasses."""
        new_children = [c._substitute(bindings) for c in self.children]
        new_expr = Expression(self.op, new_children, self.structure,
                              dict(self.metadata))
        new_expr._bindings = dict(self._bindings)
        return new_expr

    # ---- Convenience methods for lin_alg ops ----

    def det(self):
        return _make_unaryop("det", self)

    def trace(self):
        return _make_unaryop("trace", self)

    def transpose(self):
        return _make_unaryop("transpose", self)

    def inverse(self):
        return _make_unaryop("inverse", self)

    def eigenvalues(self):
        return _make_unaryop("eigenvalues", self)

    def norm(self):
        return _make_unaryop("norm", self)

    def __repr__(self):
        name = self.metadata.get('name', '')
        if name:
            return f"Expression({self.op.name!r}, name={name!r})"
        return f"Expression(op={self.op.name!r})"


class Var(Expression):
    """
    A leaf Expression -- symbolic variable (concrete or abstract).

    If value is provided, it's concrete (eager).
    If value is None, it's abstract (lazy) and must be bound at evaluate() time.
    """

    _GREEK_MAP = {
        'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
        'delta': r'\delta', 'epsilon': r'\epsilon', 'varepsilon': r'\varepsilon',
        'zeta': r'\zeta', 'eta': r'\eta', 'theta': r'\theta',
        'vartheta': r'\vartheta', 'iota': r'\iota', 'kappa': r'\kappa',
        'lambda': r'\lambda', 'mu': r'\mu', 'nu': r'\nu',
        'xi': r'\xi', 'pi': r'\pi', 'rho': r'\rho',
        'sigma': r'\sigma', 'tau': r'\tau', 'tao': r'\tau',
        'upsilon': r'\upsilon', 'phi': r'\phi', 'varphi': r'\varphi',
        'chi': r'\chi', 'psi': r'\psi', 'omega': r'\omega',
        'Gamma': r'\Gamma', 'Delta': r'\Delta', 'Theta': r'\Theta',
        'Lambda': r'\Lambda', 'Xi': r'\Xi', 'Pi': r'\Pi',
        'Sigma': r'\Sigma', 'Upsilon': r'\Upsilon', 'Phi': r'\Phi',
        'Psi': r'\Psi', 'Omega': r'\Omega',
    }

    _var_counter: int = 0

    def __init__(self, var_name, structure, value=None, shape=None, dtype=None,
                 is_tensor=False):
        var_op = Operation("var", arity=0)
        metadata = {'name': var_name}
        if shape is not None:
            metadata['shape'] = shape
        if dtype is not None:
            metadata['dtype'] = dtype
        metadata['is_tensor'] = is_tensor

        super().__init__(op=var_op, children=[], structure=structure,
                         metadata=metadata)
        self.var_name = var_name
        self.value = value

    def _get_free_variables(self):
        if self.value is not None:
            return set()
        return {self}

    def bind(self, value_or_dict=None, **kwargs):
        """Bind a value directly to this variable, or pass through to Expression.bind().

            x.bind(np.array([1,2]))       # direct value binding
            x.bind({y: np.array([3,4])})  # dict form (passed to Expression.bind)
            x.bind(y=np.array([3,4]))     # kwargs form (passed to Expression.bind)
        """
        if value_or_dict is not None and not isinstance(value_or_dict, dict):
            np = _get_np()
            if not isinstance(value_or_dict, np.ndarray):
                value_or_dict = np.asarray(value_or_dict)
            self.value = value_or_dict
            if self.metadata.get('shape') is None:
                self.metadata['shape'] = tuple(value_or_dict.shape)
            return self
        return super().bind(value_or_dict, **kwargs)

    def realize(self, shape=None, dtype=None, value=None):
        if shape is not None:
            self.metadata['shape'] = shape
        if dtype is not None:
            self.metadata['dtype'] = dtype
        if value is not None:
            self.value = value
            if shape is None and hasattr(value, 'shape'):
                self.metadata['shape'] = tuple(value.shape)
        self.constraints.resolve(self.metadata)
        return self
    
    def __getitem__(self, key):
        """Indexing into a Var creates an IndexAccess expression."""
        from ..index import _make_index_access
        if not isinstance(key, tuple):
            key = (key,)
        return _make_index_access(self, key)

    def _substitute(self, bindings):
        """If this var is in bindings, return a concrete copy."""
        if self.var_name in bindings:
            np = _get_np()
            val = bindings[self.var_name]
            if not isinstance(val, np.ndarray):
                val = np.asarray(val)
            return Var(
                self.var_name, self.structure,
                value=val,
                shape=tuple(val.shape),
                dtype=self.metadata.get('dtype'),
                is_tensor=self.metadata.get('is_tensor', False),
            )
        return self

    def __eq__(self, other):
        if isinstance(other, Var):
            return self.var_name == other.var_name
        return False

    def __hash__(self):
        return hash(('Var', self.var_name))

    def __repr__(self):
        return f"Var({self.var_name!r})"


class ScalarExpr(Expression):
    """A numeric constant expression."""

    def __init__(self, value, structure):
        scalar_op = Operation("scalar", arity=0)
        metadata = {'value': value, 'shape': (), 'is_scalar': True}
        super().__init__(op=scalar_op, children=[], structure=structure,
                         metadata=metadata)
        self.scalar_value = value

    def _get_free_variables(self):
        return set()

    def _substitute(self, _bindings):
        return self

    def __repr__(self):
        return f"ScalarExpr({self.scalar_value})"


# ---- Helper functions ----

def _ensure_expression(value, structure):
    """Convert scalars/arrays to Expression, pass through Expressions."""
    if isinstance(value, Expression):
        return value
    if isinstance(value, (int, float, complex)):
        return ScalarExpr(value, structure)
    np = _get_np()
    if isinstance(value, np.ndarray):
        Var._var_counter += 1
        return Var(f"_const_{Var._var_counter}", structure, value=value,
                   shape=tuple(value.shape))
    raise TypeError(f"Cannot convert {type(value).__name__} to Expression")


def _make_binop(op_name, left, right):
    """Create a binary operation Expression."""
    structure = left.structure
    op = structure.get_op(op_name)
    metadata = {}
    if op.result_shape_fn:
        metadata = op.result_shape_fn(left.metadata, right.metadata)
    return Expression(op, [left, right], structure, metadata)


def _make_unaryop(op_name, operand):
    """Create a unary operation Expression."""
    structure = operand.structure
    op = structure.get_op(op_name)
    metadata = {}
    if op.result_shape_fn:
        metadata = op.result_shape_fn(operand.metadata)
    return Expression(op, [operand], structure, metadata)
