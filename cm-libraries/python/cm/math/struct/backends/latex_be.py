"""
LaTeX rendering backend for expressions.

Walks the expression DAG and emits LaTeX strings.
Registry maps (structure_name, op_name) -> renderer callable.
"""

from __future__ import annotations
from typing import Dict, Tuple, Callable

__all__ = ['LatexBackend', 'latex_backend']

# Precedence for parenthesization (higher = binds tighter)
_PRECEDENCE = {
    'add': 1, 'sub': 1,
    'mul': 2, 'div': 2, 'matmul': 2,
    'pow': 3,
    'neg': 4,
}


class LatexBackend:
    """Walks the expression DAG and emits LaTeX strings."""

    def __init__(self):
        self._registry: Dict[Tuple[str, str], Callable] = {}

    def register(self, structure, op, renderer):
        self._registry[(structure, op)] = renderer

    def render(self, expr, **options):
        from ..base import Var, ScalarExpr

        if isinstance(expr, Var):
            return _render_var(expr)

        if isinstance(expr, ScalarExpr):
            return _render_scalar(expr)

        # Registered renderer
        key = (expr.structure.name, expr.op.name)
        renderer = self._registry.get(key)
        if renderer is not None:
            child_latexes = [self.render(c) for c in expr.children]
            return renderer(*child_latexes, expr=expr, backend=self)

        # Fallback: generic arithmetic
        return self._render_generic(expr)

    def _wrap_if_lower(self, child, parent_prec):
        child_latex = self.render(child)
        child_prec = _PRECEDENCE.get(child.op.name, 100)
        if child_prec < parent_prec and child.op.name in _PRECEDENCE:
            return rf"\left({child_latex}\right)"
        return child_latex

    def _render_generic(self, expr):
        op = expr.op.name
        ch = expr.children

        if op == "add" and len(ch) == 2:
            return f"{self.render(ch[0])} + {self.render(ch[1])}"

        if op == "sub" and len(ch) == 2:
            left = self.render(ch[0])
            right = self.render(ch[1])
            if ch[1].op.name in ('add', 'sub'):
                right = rf"\left({right}\right)"
            return f"{left} - {right}"

        if op == "mul" and len(ch) == 2:
            from ..base import ScalarExpr, Var
            prec = _PRECEDENCE['mul']
            left = self._wrap_if_lower(ch[0], prec)
            right = self._wrap_if_lower(ch[1], prec)
            # Smart mul: omit \cdot for scalar * tensor/var patterns
            if isinstance(ch[0], ScalarExpr):
                return f"{left} {right}"
            return rf"{left} \cdot {right}"

        if op == "div" and len(ch) == 2:
            return rf"\frac{{{self.render(ch[0])}}}{{{self.render(ch[1])}}}"

        if op == "pow" and len(ch) == 2:
            base = self.render(ch[0])
            exp = self.render(ch[1])
            if ch[0].op.name in _PRECEDENCE and _PRECEDENCE.get(ch[0].op.name, 100) < _PRECEDENCE['pow']:
                base = rf"\left({base}\right)"
            return f"{base}^{{{exp}}}"

        if op == "neg" and len(ch) == 1:
            inner = self.render(ch[0])
            if ch[0].op.name in ('add', 'sub'):
                return rf"-\left({inner}\right)"
            return f"-{inner}"

        # Unknown op: render as function call
        args = ", ".join(self.render(c) for c in ch)
        return rf"\mathrm{{{op}}}\left({args}\right)"


def _render_var(var):
    """Render a Var to LaTeX. Replicates cm.symbols Var.to_latex() exactly."""
    from ..base import Var
    name = var.var_name
    is_tensor = var.metadata.get('is_tensor', False)

    # Check for Greek letters
    if name in Var._GREEK_MAP:
        base = Var._GREEK_MAP[name]
        if is_tensor:
            return rf"\boldsymbol{{{base}}}"
        return base

    # Handle subscripts: x_0 -> x_0, x_10 -> x_{10}
    if '_' in name:
        parts = name.split('_', 1)
        base = parts[0]
        subscript = parts[1]
        if base in Var._GREEK_MAP:
            base = Var._GREEK_MAP[base]
        elif is_tensor:
            base = rf"\mathbf{{{base}}}"
        if len(subscript) > 1:
            return f"{base}_{{{subscript}}}"
        return f"{base}_{subscript}"

    if is_tensor:
        return rf"\mathbf{{{name}}}"
    return name


def _render_scalar(scalar):
    """Render a ScalarExpr. Replicates cm.symbols Const.to_latex() exactly."""
    val = scalar.scalar_value
    if isinstance(val, complex):
        real, imag = val.real, val.imag
        if abs(real) < 1e-10:
            if abs(imag - 1.0) < 1e-10:
                return "i"
            if abs(imag + 1.0) < 1e-10:
                return "-i"
            return f"{imag:.6g}i"
        if abs(imag) < 1e-10:
            if abs(real - int(real)) < 1e-10:
                return str(int(real))
            return f"{real:.6g}"
        sign = "+" if imag > 0 else "-"
        return f"({real:.6g} {sign} {abs(imag):.6g}i)"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if abs(val - int(val)) < 1e-10:
            return str(int(val))
        return f"{val:.6g}"
    return str(val)


# Module-level singleton
latex_backend = LatexBackend()
