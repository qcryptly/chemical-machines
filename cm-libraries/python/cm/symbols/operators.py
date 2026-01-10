"""
Chemical Machines Symbols Operators Module

Differential operators for symbolic computation.
Includes gradient, Laplacian, and partial derivatives.
"""

from typing import Optional, List, Union, Dict, Set, Tuple, Any

from .core import Expr, Var, _ensure_expr, _get_sympy

__all__ = [
    'DifferentialOperator',
    'ComposedOperator',
    'AppliedOperator',
    'PartialDerivative',
    'Gradient',
    'Divergence',
    'Curl',
    'Laplacian',
]


# =============================================================================
# DIFFERENTIAL OPERATORS
# =============================================================================


class DifferentialOperator(Expr):
    """
    Base class for differential operators.

    Differential operators act on functions/expressions and can be
    composed, applied, and rendered symbolically.
    """

    def __init__(self):
        pass

    def __call__(self, expr: Expr) -> Expr:
        """Apply operator to an expression."""
        raise NotImplementedError("Subclasses must implement __call__")

    def __mul__(self, other):
        """Compose operators or scale by constant."""
        if isinstance(other, DifferentialOperator):
            return ComposedOperator(self, other)
        elif isinstance(other, Expr):
            return AppliedOperator(self, other)
        else:
            return AppliedOperator(self, _ensure_expr(other))

    def __rmul__(self, other):
        """Scalar multiplication from left."""
        return ScaledOperator(_ensure_expr(other), self)

    def __add__(self, other):
        """Add operators."""
        if isinstance(other, DifferentialOperator):
            return SumOperator(self, other)
        return NotImplemented

    def _get_free_variables(self) -> Set['Var']:
        return set()


class PartialDerivative(DifferentialOperator):
    """
    Partial derivative operator ∂/∂x.

    Example:
        x = Math.var("x")
        d_dx = PartialDerivative(x)
        f = x**2
        result = d_dx(f)  # 2x
    """

    def __init__(self, var: 'Var', order: int = 1):
        super().__init__()
        self._var = var
        self._order = order

    @property
    def var(self) -> 'Var':
        return self._var

    @property
    def order(self) -> int:
        return self._order

    def __call__(self, expr: Expr) -> Expr:
        """Apply partial derivative to expression."""
        result = expr
        for _ in range(self._order):
            result = result.diff(self._var)
        return result

    def to_sympy(self):
        import sympy as sp
        # Return a symbolic representation
        return sp.Derivative(sp.Function('f')(self._var.to_sympy()),
                            self._var.to_sympy(), self._order)

    def to_latex(self) -> str:
        var_latex = self._var.to_latex()
        if self._order == 1:
            return f"\\frac{{\\partial}}{{\\partial {var_latex}}}"
        else:
            return f"\\frac{{\\partial^{self._order}}}{{\\partial {var_latex}^{self._order}}}"


class Gradient(DifferentialOperator):
    """
    Gradient operator ∇ in specified coordinate system.

    In Cartesian: ∇ = (∂/∂x, ∂/∂y, ∂/∂z)
    In spherical: ∇ = (∂/∂r, (1/r)∂/∂θ, (1/r sin θ)∂/∂φ)

    Example:
        x, y, z = Math.var("x"), Math.var("y"), Math.var("z")
        grad = Gradient([x, y, z], coord_system='cartesian')
        f = x**2 + y**2
        grad_f = grad(f)  # Returns vector expression
    """

    def __init__(self, coords: List['Var'], coord_system: str = 'cartesian'):
        super().__init__()
        self._coords = coords
        self._coord_system = coord_system

    @property
    def coords(self) -> List['Var']:
        return self._coords

    @property
    def coord_system(self) -> str:
        return self._coord_system

    def __call__(self, expr: Expr) -> List[Expr]:
        """Apply gradient to expression, returns vector of partial derivatives."""
        if self._coord_system == 'cartesian':
            return [expr.diff(c) for c in self._coords]
        elif self._coord_system == 'spherical':
            r, theta, phi = self._coords
            return [
                expr.diff(r),
                (Const(1) / r) * expr.diff(theta),
                (Const(1) / (r * Sin(theta))) * expr.diff(phi)
            ]
        else:
            raise ValueError(f"Unknown coordinate system: {self._coord_system}")

    def to_latex(self) -> str:
        return r"\nabla"

    def to_sympy(self):
        import sympy as sp
        # SymPy vector module would be needed for full support
        return sp.Symbol('nabla')


class Laplacian(DifferentialOperator):
    """
    Laplacian operator ∇² in specified coordinate system.

    In Cartesian: ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²

    In spherical: ∇² = (1/r²)∂/∂r(r²∂/∂r) + (1/r²sinθ)∂/∂θ(sinθ∂/∂θ)
                       + (1/r²sin²θ)∂²/∂φ²

    Example:
        r, theta, phi = Math.var("r"), Math.var("theta"), Math.var("phi")
        laplacian = Laplacian([r, theta, phi], coord_system='spherical')
        f = r**2
        result = laplacian(f)  # 6
    """

    def __init__(self, coords: List['Var'], coord_system: str = 'cartesian'):
        super().__init__()
        self._coords = coords
        self._coord_system = coord_system

    @property
    def coords(self) -> List['Var']:
        return self._coords

    @property
    def coord_system(self) -> str:
        return self._coord_system

    def __call__(self, expr: Expr) -> Expr:
        """Apply Laplacian to expression."""
        if self._coord_system == 'cartesian':
            # ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
            result = Const(0)
            for c in self._coords:
                result = result + expr.diff(c).diff(c)
            return result

        elif self._coord_system == 'spherical':
            if len(self._coords) != 3:
                raise ValueError("Spherical Laplacian requires 3 coordinates (r, θ, φ)")
            r, theta, phi = self._coords

            # Radial part: (1/r²)∂/∂r(r²∂f/∂r)
            df_dr = expr.diff(r)
            radial = (Const(1) / (r * r)) * (Const(2) * r * df_dr + r * r * df_dr.diff(r))

            # Angular part θ: (1/r²sinθ)∂/∂θ(sinθ∂f/∂θ)
            df_dtheta = expr.diff(theta)
            angular_theta = (Const(1) / (r * r * Sin(theta))) * (
                Cos(theta) * df_dtheta + Sin(theta) * df_dtheta.diff(theta)
            )

            # Angular part φ: (1/r²sin²θ)∂²f/∂φ²
            angular_phi = (Const(1) / (r * r * Sin(theta) * Sin(theta))) * expr.diff(phi).diff(phi)

            return radial + angular_theta + angular_phi

        else:
            raise ValueError(f"Unknown coordinate system: {self._coord_system}")

    def to_latex(self) -> str:
        return r"\nabla^2"

    def to_sympy(self):
        import sympy as sp
        return sp.Symbol('nabla^2')


class ScaledOperator(DifferentialOperator):
    """Operator scaled by a constant or expression."""

    def __init__(self, scalar: Expr, operator: DifferentialOperator):
        super().__init__()
        self._scalar = scalar
        self._operator = operator

    def __call__(self, expr: Expr) -> Expr:
        return self._scalar * self._operator(expr)

    def to_latex(self) -> str:
        return f"{self._scalar.to_latex()} {self._operator.to_latex()}"

    def to_sympy(self):
        return self._scalar.to_sympy() * self._operator.to_sympy()


class ComposedOperator(DifferentialOperator):
    """Composition of two operators: A ∘ B means apply B then A."""

    def __init__(self, first: DifferentialOperator, second: DifferentialOperator):
        super().__init__()
        self._first = first
        self._second = second

    def __call__(self, expr: Expr) -> Expr:
        return self._first(self._second(expr))

    def to_latex(self) -> str:
        return f"{self._first.to_latex()} {self._second.to_latex()}"

    def to_sympy(self):
        return self._first.to_sympy() * self._second.to_sympy()


class SumOperator(DifferentialOperator):
    """Sum of two operators."""

    def __init__(self, op1: DifferentialOperator, op2: DifferentialOperator):
        super().__init__()
        self._op1 = op1
        self._op2 = op2

    def __call__(self, expr: Expr) -> Expr:
        return self._op1(expr) + self._op2(expr)

    def to_latex(self) -> str:
        return f"\\left({self._op1.to_latex()} + {self._op2.to_latex()}\\right)"

    def to_sympy(self):
        return self._op1.to_sympy() + self._op2.to_sympy()


class AppliedOperator(Expr):
    """Result of applying an operator to an expression (lazy)."""

    def __init__(self, operator: DifferentialOperator, expr: Expr):
        self._operator = operator
        self._expr = expr

    def evaluate(self, **kwargs) -> Any:
        """Evaluate by first applying operator, then evaluating result."""
        result = self._operator(self._expr)
        return result.evaluate(**kwargs)

    def to_sympy(self):
        result = self._operator(self._expr)
        return result.to_sympy()

    def to_latex(self) -> str:
        return f"{self._operator.to_latex()} {self._expr.to_latex()}"

    def _get_free_variables(self) -> Set['Var']:
        return self._expr._get_free_variables()


