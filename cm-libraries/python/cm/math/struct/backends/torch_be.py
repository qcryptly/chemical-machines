"""
PyTorch compute graph backend for expressions.

Compiles an expression DAG into a callable TorchGraph.
Registry maps (structure_name, op_name) -> torch callable.
"""

from __future__ import annotations
from typing import Dict, Tuple, Callable, Any, List

__all__ = ['TorchBackend', 'TorchGraph', 'torch_backend']

_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class TorchGraph:
    """
    A compiled PyTorch compute graph from an Expression DAG.

    Callable: accepts keyword arguments mapping var names to tensors.
    Returns a tensor result on the specified device.
    """

    def __init__(self, expr, device, backend, bindings=None):
        self._expr = expr
        self._device = device
        self._backend = backend
        self._pre_bindings = bindings or {}
        self._input_vars = sorted(
            v.var_name for v in expr._get_free_variables()
            if v.var_name not in self._pre_bindings
        )

    def __call__(self, **kwargs):
        torch = _get_torch()
        merged = {**self._pre_bindings, **kwargs}
        all_vars = sorted(v.var_name for v in self._expr._get_free_variables())
        missing = set(all_vars) - set(merged.keys())
        if missing:
            raise ValueError(f"Missing input variables: {missing}")

        bindings = {}
        for name, val in merged.items():
            if isinstance(val, torch.Tensor):
                bindings[name] = val.to(self._device)
            else:
                bindings[name] = torch.tensor(val, device=self._device,
                                              dtype=torch.float64)

        result = self._backend._eval_torch(self._expr, bindings)
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, device=self._device,
                                  dtype=torch.float64)
        return result

    @property
    def input_vars(self):
        return self._input_vars.copy()

    @property
    def device(self):
        return self._device

    def __repr__(self):
        return f"TorchGraph(inputs={self._input_vars}, device={self._device!r})"


class TorchBackend:
    """Walks the expression DAG and emits PyTorch operations."""

    def __init__(self):
        self._registry: Dict[Tuple[str, str], Callable] = {}

    def register(self, structure, op, impl):
        self._registry[(structure, op)] = impl

    def compile(self, expr, device="cpu", bindings=None):
        return TorchGraph(expr, device, self, bindings=bindings)

    def _eval_torch(self, expr, bindings):
        from ..base import Var, ScalarExpr
        torch = _get_torch()

        # Infer device from bindings
        device = "cpu"
        for v in bindings.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break

        # Leaf: Var
        if isinstance(expr, Var):
            if expr.value is not None:
                val = expr.value
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float64)
                return val.to(device)
            name = expr.var_name
            if name in bindings:
                return bindings[name]
            raise ValueError(f"Unbound variable '{name}' in torch evaluation")

        # Leaf: ScalarExpr
        if isinstance(expr, ScalarExpr):
            return torch.tensor(expr.scalar_value, device=device,
                                dtype=torch.float64)

        # Recursive: evaluate children
        child_values = [self._eval_torch(c, bindings) for c in expr.children]

        # Registered implementation
        key = (expr.structure.name, expr.op.name)
        impl = self._registry.get(key)
        if impl is not None:
            return impl(*child_values)

        # Fallback: basic arithmetic
        return self._eval_generic(expr.op.name, child_values)

    def _eval_generic(self, op_name, values):
        torch = _get_torch()
        if op_name == "add":
            return values[0] + values[1]
        if op_name == "sub":
            return values[0] - values[1]
        if op_name == "mul":
            return values[0] * values[1]
        if op_name == "div":
            return values[0] / values[1]
        if op_name == "pow":
            return torch.pow(values[0], values[1])
        if op_name == "neg":
            return -values[0]
        raise NotImplementedError(
            f"No torch implementation for operation '{op_name}'"
        )


# Module-level singleton
torch_backend = TorchBackend()
