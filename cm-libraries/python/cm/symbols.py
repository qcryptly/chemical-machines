"""
Chemical Machines Symbols Module

A module for rendering LaTeX math expressions with support for different notation styles.

Notation Styles:
    - standard: Default LaTeX math notation
    - physicist: Physics notation (hbar, vectors with arrows, etc.)
    - chemist: Chemistry notation (reaction arrows, chemical formulas)
    - braket: Dirac bra-ket notation for quantum mechanics
    - engineering: Engineering notation (j for imaginary, etc.)

Slater Determinants:
    The module provides SlaterState for representing many-electron wavefunctions
    as Slater determinants using a simple 1D list of occupied spin-orbitals.

    Symmetry classes for automatic term zeroing:
    - Symmetry.NONE: Keep all terms
    - Symmetry.SPIN: α/β spin orthogonality
    - Symmetry.SPATIAL: Different spatial orbitals orthogonal
    - Symmetry.ORTHONORMAL: Full orthonormality ⟨φᵢ|φⱼ⟩ = δᵢⱼ

Usage:
    from cm.symbols import latex, Math, set_notation, SlaterState, SpinOrbital, Symmetry

    # Simple LaTeX rendering
    latex(r"E = mc^2")

    # Using the Math builder
    m = Math()
    m.frac("a", "b").plus().sqrt("c")
    m.render()

    # Change notation style
    set_notation("physicist")
    latex(r"\\hbar \\omega")

    # Bra-ket notation
    set_notation("braket")
    m = Math()
    m.bra("psi").ket("phi")
    m.render()

    # Slater determinants with 1D state vector
    state = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
    m = Math()
    m.slater_bra_state(state)  # ⟨1s↑, 1s↓, 2s↑|
    m.render()

    # Inner products with symmetry-based simplification
    psi = SlaterState.from_labels(["1s↑", "1s↓"])
    phi = SlaterState.from_labels(["1s↑", "2s↑"])
    m = Math()
    m.slater_overlap(psi, phi)  # Renders: 0 (orthogonal)
    m.render()

    # Slater-Condon rules
    m = Math()
    m.slater_condon_rule(psi, phi, operator_type="one_electron")
    m.render()
"""

from typing import Optional, List, Union, Dict, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from . import views

# Lazy imports for heavy dependencies
_sympy = None
_torch = None

def _get_sympy():
    """Lazy load sympy."""
    global _sympy
    if _sympy is None:
        import sympy as sp
        _sympy = sp
    return _sympy

def _get_torch():
    """Lazy load torch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# =============================================================================
# SYMBOLIC COMPUTATION LIBRARY
# =============================================================================
# Expression tree (AST) based symbolic computation with SymPy backend for
# symbolic operations and PyTorch backend for numerical evaluation.
# =============================================================================

class EvaluationError(Exception):
    """Raised when an expression cannot be numerically evaluated."""
    pass


class Expr(ABC):
    """
    Abstract base class for symbolic expressions.

    Expressions form a tree structure that can be:
    - Rendered to LaTeX via .render()
    - Converted to SymPy for symbolic manipulation via .to_sympy()
    - Numerically evaluated via .evaluate()

    Supports operator overloading: +, -, *, /, **, unary -
    """

    # Global counter for auto-naming variables
    _var_counter: int = 0

    def __init__(self):
        self._sympy_cache: Any = None

    @abstractmethod
    def to_sympy(self):
        """Convert to SymPy expression."""
        pass

    @abstractmethod
    def to_latex(self) -> str:
        """Generate LaTeX representation."""
        pass

    @abstractmethod
    def _get_free_variables(self) -> Set['Var']:
        """Return set of free (unbound) variables in this expression."""
        pass

    def render(self, display: bool = True, justify: str = "center"):
        """
        Render expression to MathJax via views.html().

        Args:
            display: If True, use display math mode (centered block)
            justify: Alignment ("left", "center", "right")
        """
        latex_str = self.to_latex()
        delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
        html = f'<div class="cm-math cm-math-{justify}" style="line-height: 1.5;">{delim_start} {latex_str} {delim_end}</div>'
        views.html(html)

    def simplify(self) -> 'Expr':
        """Simplify expression using SymPy."""
        sp = _get_sympy()
        simplified = sp.simplify(self.to_sympy())
        return SympyWrapper(simplified)

    def expand(self) -> 'Expr':
        """Expand expression using SymPy."""
        sp = _get_sympy()
        expanded = sp.expand(self.to_sympy())
        return SympyWrapper(expanded)

    def integrate(self, var: 'Var', bounds: Optional[List] = None) -> 'Integral':
        """
        Integrate expression with respect to variable.

        Args:
            var: Variable to integrate over
            bounds: Optional [lower, upper] bounds for definite integral

        Returns:
            Integral expression

        Example:
            x = Math.var("x")
            expr = x ** 2
            expr.integrate(x, bounds=[0, 1])  # Definite integral
            expr.integrate(x)  # Indefinite integral
        """
        return Integral(self, var, bounds)

    def diff(self, var: 'Var', order: int = 1) -> 'Derivative':
        """
        Differentiate expression with respect to variable.

        Args:
            var: Variable to differentiate with respect to
            order: Order of derivative (default 1)

        Returns:
            Derivative expression
        """
        return Derivative(self, var, order)

    def sum(self, var: 'Var', lower=None, upper=None, bounds=None) -> 'Sum':
        """
        Sum expression with respect to variable.

        Args:
            var: Summation variable
            lower: Lower bound (alternative to bounds)
            upper: Upper bound (alternative to bounds)
            bounds: Tuple of (lower, upper) bounds

        Returns:
            Sum expression

        Example:
            i = Math.var("i")
            n = Math.var("n")
            expr = i ** 2
            expr.sum(i, 1, n)           # Positional bounds
            expr.sum(i, bounds=(1, n))  # Named bounds tuple
        """
        if bounds is not None:
            lower, upper = bounds
        return Sum(self, var, lower, upper)

    def prod(self, var: 'Var', lower=None, upper=None, bounds=None) -> 'Product':
        """
        Product expression with respect to variable.

        Args:
            var: Product variable
            lower: Lower bound (alternative to bounds)
            upper: Upper bound (alternative to bounds)
            bounds: Tuple of (lower, upper) bounds

        Returns:
            Product expression

        Example:
            i = Math.var("i")
            expr = i
            expr.prod(i, 1, 5)  # = 1 * 2 * 3 * 4 * 5 = 120
        """
        if bounds is not None:
            lower, upper = bounds
        return Product(self, var, lower, upper)

    def evaluate(self, **kwargs) -> Any:
        """
        Numerically evaluate expression with given variable values.

        Args:
            **kwargs: Variable name -> value mappings

        Returns:
            Numeric result (torch.Tensor or scalar)

        Raises:
            EvaluationError: If expression cannot be evaluated (e.g., missing variables)

        Example:
            x = Math.var("x")
            y = Math.var("y")
            expr = x**2 + y**2
            expr.evaluate(x=3, y=4)  # Returns tensor(25)
        """
        free_vars = self._get_free_variables()
        var_names = {v.name for v in free_vars}
        missing = var_names - set(kwargs.keys())

        if missing:
            raise EvaluationError(
                f"Cannot evaluate: missing values for variables {missing}\n"
                f"Expression: {self.to_latex()}"
            )

        sp = _get_sympy()
        sympy_expr = self.to_sympy()

        # Substitute values into the expression
        subs_dict = {sp.Symbol(name): value for name, value in kwargs.items()}
        result_expr = sympy_expr.subs(subs_dict)

        # Try to evaluate numerically
        try:
            # First try: get numeric value from SymPy
            numeric_result = float(result_expr.evalf())

            # Try to return as torch tensor if available
            try:
                torch = _get_torch()
                return torch.tensor(numeric_result)
            except ImportError:
                return numeric_result

        except (TypeError, ValueError) as e:
            # If evalf fails, try lambdify with numpy
            try:
                sorted_names = sorted(kwargs.keys())
                var_symbols = [sp.Symbol(n) for n in sorted_names]
                values = [kwargs[n] for n in sorted_names]

                f = sp.lambdify(var_symbols, sympy_expr, modules=['numpy'])
                result = f(*values)

                try:
                    torch = _get_torch()
                    if not isinstance(result, torch.Tensor):
                        result = torch.tensor(float(result))
                    return result
                except ImportError:
                    return float(result)

            except Exception as e2:
                raise EvaluationError(f"Evaluation failed: {e2}\nExpression: {self.to_latex()}")

    def subs(self, substitutions: Dict['Var', 'Expr']) -> 'Expr':
        """
        Substitute variables with expressions.

        Args:
            substitutions: Dict mapping Var -> Expr replacements

        Returns:
            New expression with substitutions applied
        """
        sp = _get_sympy()
        sympy_expr = self.to_sympy()

        sympy_subs = {}
        for var, expr in substitutions.items():
            sympy_subs[var.to_sympy()] = expr.to_sympy()

        result = sympy_expr.subs(sympy_subs)
        return SympyWrapper(result)

    # =========================================================================
    # Operator Overloads
    # =========================================================================

    def __add__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Add(self, _ensure_expr(other))

    def __radd__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Add(_ensure_expr(other), self)

    def __sub__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Sub(self, _ensure_expr(other))

    def __rsub__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Sub(_ensure_expr(other), self)

    def __mul__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Mul(self, _ensure_expr(other))

    def __rmul__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Mul(_ensure_expr(other), self)

    def __truediv__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Div(self, _ensure_expr(other))

    def __rtruediv__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Div(_ensure_expr(other), self)

    def __pow__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Pow(self, _ensure_expr(other))

    def __rpow__(self, other: Union['Expr', int, float]) -> 'Expr':
        return Pow(_ensure_expr(other), self)

    def __neg__(self) -> 'Expr':
        return Neg(self)

    def __pos__(self) -> 'Expr':
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_latex()})"


def _ensure_expr(value) -> 'Expr':
    """Convert value to Expr if needed."""
    if isinstance(value, Expr):
        return value
    elif isinstance(value, (int, float)):
        return Const(value)
    else:
        raise TypeError(f"Cannot convert {type(value).__name__} to Expr")


# =============================================================================
# Variable and Constant Classes
# =============================================================================

class Var(Expr):
    """
    A symbolic variable.

    Supports auto-naming and Greek letter recognition.

    Example:
        x = Var()           # Auto-named x_0
        y = Var("y")        # Named y
        t = Var("tau")      # Greek letter tau
        theta = Var("theta")  # Greek letter theta
    """

    # Greek letter mapping
    _GREEK_MAP = {
        'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
        'delta': r'\delta', 'epsilon': r'\epsilon', 'varepsilon': r'\varepsilon',
        'zeta': r'\zeta', 'eta': r'\eta', 'theta': r'\theta',
        'vartheta': r'\vartheta', 'iota': r'\iota', 'kappa': r'\kappa',
        'lambda': r'\lambda', 'mu': r'\mu', 'nu': r'\nu',
        'xi': r'\xi', 'pi': r'\pi', 'rho': r'\rho',
        'sigma': r'\sigma', 'tau': r'\tau', 'tao': r'\tau',  # Common misspelling
        'upsilon': r'\upsilon', 'phi': r'\phi', 'varphi': r'\varphi',
        'chi': r'\chi', 'psi': r'\psi', 'omega': r'\omega',
        # Uppercase
        'Gamma': r'\Gamma', 'Delta': r'\Delta', 'Theta': r'\Theta',
        'Lambda': r'\Lambda', 'Xi': r'\Xi', 'Pi': r'\Pi',
        'Sigma': r'\Sigma', 'Upsilon': r'\Upsilon', 'Phi': r'\Phi',
        'Psi': r'\Psi', 'Omega': r'\Omega',
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        if name is None:
            name = f"x_{Expr._var_counter}"
            Expr._var_counter += 1
        self.name = name
        self._sympy_symbol = None  # Lazy create

    def to_sympy(self):
        if self._sympy_symbol is None:
            sp = _get_sympy()
            self._sympy_symbol = sp.Symbol(self.name)
        return self._sympy_symbol

    def to_latex(self) -> str:
        # Check for Greek letters
        if self.name in self._GREEK_MAP:
            return self._GREEK_MAP[self.name]

        # Handle subscripts: x_0 -> x_0, x_10 -> x_{10}
        if '_' in self.name:
            parts = self.name.split('_', 1)
            base = parts[0]
            subscript = parts[1]
            # Check if base is Greek
            if base in self._GREEK_MAP:
                base = self._GREEK_MAP[base]
            if len(subscript) > 1:
                return f"{base}_{{{subscript}}}"
            return f"{base}_{subscript}"

        return self.name

    def _get_free_variables(self) -> Set['Var']:
        return {self}

    def __eq__(self, other):
        if isinstance(other, Var):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(('Var', self.name))

    def __repr__(self):
        return f"Var('{self.name}')"


class Const(Expr):
    """A numeric constant."""

    def __init__(self, value: Union[int, float]):
        super().__init__()
        self.value = value

    def to_sympy(self):
        sp = _get_sympy()
        return sp.Number(self.value)

    def to_latex(self) -> str:
        if isinstance(self.value, int):
            return str(self.value)
        # Format float nicely
        if self.value == int(self.value):
            return str(int(self.value))
        return f"{self.value:.6g}"

    def _get_free_variables(self) -> Set['Var']:
        return set()

    def __repr__(self):
        return f"Const({self.value})"


class SymbolicConst(Expr):
    """A named symbolic constant (pi, e, infinity)."""

    _LATEX_MAP = {
        'pi': r'\pi',
        'e': r'e',
        'inf': r'\infty',
        'oo': r'\infty',
        'I': r'i',
    }

    def __init__(self, name: str, sympy_value=None):
        super().__init__()
        self.name = name
        self._sympy_value = sympy_value

    def to_sympy(self):
        if self._sympy_value is not None:
            return self._sympy_value
        sp = _get_sympy()
        if self.name == 'pi':
            return sp.pi
        elif self.name == 'e':
            return sp.E
        elif self.name in ('inf', 'oo'):
            return sp.oo
        elif self.name == 'I':
            return sp.I
        return sp.Symbol(self.name)

    def to_latex(self) -> str:
        return self._LATEX_MAP.get(self.name, self.name)

    def _get_free_variables(self) -> Set['Var']:
        return set()

    def __repr__(self):
        return f"SymbolicConst('{self.name}')"


class SympyWrapper(Expr):
    """Wrapper for SymPy expressions (result of simplify, etc.)."""

    def __init__(self, sympy_expr):
        super().__init__()
        self._sympy_expr = sympy_expr

    def to_sympy(self):
        return self._sympy_expr

    def to_latex(self) -> str:
        sp = _get_sympy()
        return sp.latex(self._sympy_expr)

    def _get_free_variables(self) -> Set['Var']:
        sp = _get_sympy()
        free_symbols = self._sympy_expr.free_symbols
        return {Var(str(s)) for s in free_symbols}


# =============================================================================
# Binary Operations
# =============================================================================

class BinOp(Expr):
    """Base class for binary operations."""

    _latex_op: str = ""
    _precedence: int = 0

    def __init__(self, left: Expr, right: Expr):
        super().__init__()
        self.left = left
        self.right = right

    def _get_free_variables(self) -> Set['Var']:
        return self.left._get_free_variables() | self.right._get_free_variables()

    @abstractmethod
    def _sympy_op(self, left, right):
        pass

    def to_sympy(self):
        return self._sympy_op(self.left.to_sympy(), self.right.to_sympy())

    def _wrap_if_lower_precedence(self, expr: Expr, latex: str) -> str:
        """Wrap in parentheses if expr has lower precedence."""
        if isinstance(expr, BinOp) and expr._precedence < self._precedence:
            return f"\\left({latex}\\right)"
        return latex

    def to_latex(self) -> str:
        left_latex = self.left.to_latex()
        right_latex = self.right.to_latex()
        return f"{left_latex} {self._latex_op} {right_latex}"


class Add(BinOp):
    """Addition: a + b"""
    _latex_op = "+"
    _precedence = 1

    def _sympy_op(self, left, right):
        return left + right


class Sub(BinOp):
    """Subtraction: a - b"""
    _latex_op = "-"
    _precedence = 1

    def _sympy_op(self, left, right):
        return left - right

    def to_latex(self) -> str:
        left_latex = self.left.to_latex()
        right_latex = self.right.to_latex()
        # Wrap right side if it's addition/subtraction to avoid ambiguity
        if isinstance(self.right, (Add, Sub)):
            right_latex = f"\\left({right_latex}\\right)"
        return f"{left_latex} - {right_latex}"


class Mul(BinOp):
    """Multiplication: a * b"""
    _latex_op = r"\cdot"
    _precedence = 2

    def _sympy_op(self, left, right):
        return left * right

    def to_latex(self) -> str:
        left_latex = self._wrap_if_lower_precedence(self.left, self.left.to_latex())
        right_latex = self._wrap_if_lower_precedence(self.right, self.right.to_latex())

        # Smart multiplication: omit dot for coefficient * variable patterns
        if isinstance(self.left, Const) and isinstance(self.right, (Var, Pow, Mul)):
            return f"{left_latex} {right_latex}"

        return f"{left_latex} {self._latex_op} {right_latex}"


class Div(BinOp):
    """Division: a / b (renders as fraction)"""
    _latex_op = "/"
    _precedence = 2

    def _sympy_op(self, left, right):
        return left / right

    def to_latex(self) -> str:
        left_latex = self.left.to_latex()
        right_latex = self.right.to_latex()
        return f"\\frac{{{left_latex}}}{{{right_latex}}}"


class Pow(BinOp):
    """Exponentiation: a ** b"""
    _latex_op = "^"
    _precedence = 3

    def _sympy_op(self, left, right):
        return left ** right

    def to_latex(self) -> str:
        left_latex = self.left.to_latex()
        right_latex = self.right.to_latex()

        # Wrap base in parens if it's a binary operation
        if isinstance(self.left, BinOp):
            left_latex = f"\\left({left_latex}\\right)"

        return f"{left_latex}^{{{right_latex}}}"


# =============================================================================
# Unary Operations
# =============================================================================

class UnaryOp(Expr):
    """Base class for unary operations."""

    def __init__(self, operand: Expr):
        super().__init__()
        self.operand = operand

    def _get_free_variables(self) -> Set['Var']:
        return self.operand._get_free_variables()


class Neg(UnaryOp):
    """Negation: -a"""

    def to_sympy(self):
        return -self.operand.to_sympy()

    def to_latex(self) -> str:
        operand_latex = self.operand.to_latex()
        if isinstance(self.operand, BinOp):
            return f"-\\left({operand_latex}\\right)"
        return f"-{operand_latex}"


class Sqrt(UnaryOp):
    """Square root: sqrt(a)"""

    def to_sympy(self):
        sp = _get_sympy()
        return sp.sqrt(self.operand.to_sympy())

    def to_latex(self) -> str:
        return f"\\sqrt{{{self.operand.to_latex()}}}"


class Sin(UnaryOp):
    """Sine function"""

    def to_sympy(self):
        sp = _get_sympy()
        return sp.sin(self.operand.to_sympy())

    def to_latex(self) -> str:
        return f"\\sin\\left({self.operand.to_latex()}\\right)"


class Cos(UnaryOp):
    """Cosine function"""

    def to_sympy(self):
        sp = _get_sympy()
        return sp.cos(self.operand.to_sympy())

    def to_latex(self) -> str:
        return f"\\cos\\left({self.operand.to_latex()}\\right)"


class Tan(UnaryOp):
    """Tangent function"""

    def to_sympy(self):
        sp = _get_sympy()
        return sp.tan(self.operand.to_sympy())

    def to_latex(self) -> str:
        return f"\\tan\\left({self.operand.to_latex()}\\right)"


class Exp(UnaryOp):
    """Exponential function: e^x"""

    def to_sympy(self):
        sp = _get_sympy()
        return sp.exp(self.operand.to_sympy())

    def to_latex(self) -> str:
        return f"e^{{{self.operand.to_latex()}}}"


class Log(UnaryOp):
    """Natural logarithm"""

    def to_sympy(self):
        sp = _get_sympy()
        return sp.log(self.operand.to_sympy())

    def to_latex(self) -> str:
        return f"\\ln\\left({self.operand.to_latex()}\\right)"


class Abs(UnaryOp):
    """Absolute value"""

    def to_sympy(self):
        sp = _get_sympy()
        return sp.Abs(self.operand.to_sympy())

    def to_latex(self) -> str:
        return f"\\left|{self.operand.to_latex()}\\right|"


# =============================================================================
# Calculus Operations
# =============================================================================

class Integral(Expr):
    """
    Represents an integral expression.

    Supports both definite and indefinite integrals, with symbolic bounds.

    Example:
        x = Var("x")
        t = Var("t")
        expr = x ** 2

        # Indefinite integral
        expr.integrate(x)

        # Definite integral
        expr.integrate(x, bounds=[0, 1])

        # Symbolic bounds
        expr.integrate(x, bounds=[0, t])
    """

    def __init__(self, integrand: Expr, var: Var, bounds: Optional[List] = None):
        super().__init__()
        self.integrand = integrand
        self.var = var
        self.bounds = bounds  # [lower, upper] or None

    @property
    def is_definite(self) -> bool:
        """True if this is a definite integral (has bounds)."""
        return self.bounds is not None

    def integrate(self, var: Var, bounds: Optional[List] = None) -> 'Integral':
        """Chain another integration."""
        return Integral(self, var, bounds)

    def to_sympy(self):
        sp = _get_sympy()
        integrand_sympy = self.integrand.to_sympy()
        var_sympy = self.var.to_sympy()

        if self.bounds:
            lower, upper = self.bounds
            # Convert bounds to SymPy if they're Expr
            if isinstance(lower, Expr):
                lower = lower.to_sympy()
            elif lower == float('-inf'):
                lower = -sp.oo
            elif lower == float('inf'):
                lower = sp.oo

            if isinstance(upper, Expr):
                upper = upper.to_sympy()
            elif upper == float('-inf'):
                upper = -sp.oo
            elif upper == float('inf'):
                upper = sp.oo

            return sp.integrate(integrand_sympy, (var_sympy, lower, upper))
        else:
            return sp.integrate(integrand_sympy, var_sympy)

    def to_latex(self) -> str:
        integrand_latex = self.integrand.to_latex()
        var_latex = self.var.to_latex()

        if self.bounds:
            lower, upper = self.bounds

            # Convert bounds to LaTeX
            def bound_to_latex(b):
                if isinstance(b, Expr):
                    return b.to_latex()
                elif b == float('-inf'):
                    return r'-\infty'
                elif b == float('inf'):
                    return r'\infty'
                return str(b)

            lower_latex = bound_to_latex(lower)
            upper_latex = bound_to_latex(upper)

            return f"\\int_{{{lower_latex}}}^{{{upper_latex}}} {integrand_latex} \\, d{var_latex}"
        else:
            return f"\\int {integrand_latex} \\, d{var_latex}"

    def _get_free_variables(self) -> Set['Var']:
        # The integration variable is bound, not free
        free_vars = self.integrand._get_free_variables() - {self.var}

        # Add variables from symbolic bounds
        if self.bounds:
            for b in self.bounds:
                if isinstance(b, Expr):
                    free_vars |= b._get_free_variables()

        return free_vars

    def evaluate(self, **kwargs) -> Any:
        """
        Numerically evaluate the integral.

        Raises:
            EvaluationError: If integral is indefinite or has unresolved symbolic bounds
        """
        if not self.is_definite:
            raise EvaluationError(
                "Cannot numerically evaluate indefinite integral.\n"
                "Provide bounds with .integrate(var, bounds=[lower, upper])\n"
                f"Expression: {self.to_latex()}"
            )

        # Check if bounds contain unresolved symbolic values
        if self.bounds:
            for b in self.bounds:
                if isinstance(b, Expr):
                    bound_vars = b._get_free_variables()
                    missing = {v.name for v in bound_vars} - set(kwargs.keys())
                    if missing:
                        raise EvaluationError(
                            f"Cannot evaluate: bound contains unresolved variables {missing}\n"
                            f"Expression: {self.to_latex()}"
                        )

        # Use parent evaluate method
        return super().evaluate(**kwargs)


class Derivative(Expr):
    """
    Represents a derivative expression.

    Example:
        x = Var("x")
        expr = x ** 2
        expr.diff(x)      # First derivative
        expr.diff(x, 2)   # Second derivative
    """

    def __init__(self, expr: Expr, var: Var, order: int = 1):
        super().__init__()
        self.expr = expr
        self.var = var
        self.order = order

    def diff(self, var: Var, order: int = 1) -> 'Derivative':
        """Chain another differentiation."""
        return Derivative(self, var, order)

    def to_sympy(self):
        sp = _get_sympy()
        return sp.diff(self.expr.to_sympy(), self.var.to_sympy(), self.order)

    def to_latex(self) -> str:
        expr_latex = self.expr.to_latex()
        var_latex = self.var.to_latex()

        if self.order == 1:
            return f"\\frac{{d}}{{d{var_latex}}} \\left({expr_latex}\\right)"
        else:
            return f"\\frac{{d^{{{self.order}}}}}{{d{var_latex}^{{{self.order}}}}} \\left({expr_latex}\\right)"

    def _get_free_variables(self) -> Set['Var']:
        return self.expr._get_free_variables()


# =============================================================================
# Summation and Product Operations
# =============================================================================

class Sum(Expr):
    """
    Represents a discrete summation expression.

    Supports both indefinite sums (no bounds) and definite sums (with bounds).

    Example:
        i = Var("i")
        n = Var("n")
        expr = i ** 2

        # Indefinite sum
        Math.sum(expr, i)

        # Definite sum - multiple syntax options
        Math.sum(expr, i, 1, n)
        Math.sum(expr, i, bounds=(1, n))
        expr.sum(i, 1, n)
        expr.sum(i, bounds=(1, n))
    """

    def __init__(self, summand: Expr, var: Var,
                 lower: Optional[Union[Expr, int]] = None,
                 upper: Optional[Union[Expr, int]] = None):
        super().__init__()
        self.summand = summand
        self.var = var
        self.lower = _ensure_expr(lower) if lower is not None else None
        self.upper = _ensure_expr(upper) if upper is not None else None

    @property
    def is_definite(self) -> bool:
        """True if this is a definite sum (has both bounds)."""
        return self.lower is not None and self.upper is not None

    def sum(self, var: Var, lower=None, upper=None, bounds=None) -> 'Sum':
        """Chain another summation."""
        if bounds is not None:
            lower, upper = bounds
        return Sum(self, var, lower, upper)

    def prod(self, var: Var, lower=None, upper=None, bounds=None) -> 'Product':
        """Chain a product onto this sum."""
        if bounds is not None:
            lower, upper = bounds
        return Product(self, var, lower, upper)

    def to_sympy(self):
        sp = _get_sympy()
        summand_sympy = self.summand.to_sympy()
        var_sympy = self.var.to_sympy()

        if self.is_definite:
            lower_sympy = self.lower.to_sympy()
            upper_sympy = self.upper.to_sympy()
            return sp.Sum(summand_sympy, (var_sympy, lower_sympy, upper_sympy))
        else:
            # Indefinite sum - return unevaluated
            return sp.Sum(summand_sympy, var_sympy)

    def to_latex(self) -> str:
        summand_latex = self.summand.to_latex()
        var_latex = self.var.to_latex()

        if self.is_definite:
            lower_latex = self.lower.to_latex()
            upper_latex = self.upper.to_latex()
            return f"\\sum_{{{var_latex}={lower_latex}}}^{{{upper_latex}}} {summand_latex}"
        else:
            return f"\\sum_{{{var_latex}}} {summand_latex}"

    def _get_free_variables(self) -> Set['Var']:
        # The summation variable is bound, not free
        free_vars = self.summand._get_free_variables() - {self.var}

        # Add variables from symbolic bounds
        if self.lower is not None:
            free_vars |= self.lower._get_free_variables()
        if self.upper is not None:
            free_vars |= self.upper._get_free_variables()

        return free_vars

    def evaluate(self, **kwargs) -> Any:
        """
        Numerically evaluate the sum.

        Raises:
            EvaluationError: If sum is indefinite or has unresolved symbolic bounds
        """
        if not self.is_definite:
            raise EvaluationError(
                "Cannot numerically evaluate indefinite sum.\n"
                "Provide bounds with .sum(var, lower, upper)\n"
                f"Expression: {self.to_latex()}"
            )

        # Use SymPy's doit() to compute the sum
        sympy_sum = self.to_sympy()

        # Try to compute the sum symbolically first
        computed = sympy_sum.doit()

        # Wrap in SympyWrapper and evaluate
        wrapper = SympyWrapper(computed)
        return wrapper.evaluate(**kwargs)


class Product(Expr):
    """
    Represents a discrete product expression.

    Supports both indefinite products (no bounds) and definite products (with bounds).

    Example:
        i = Var("i")
        n = Var("n")
        expr = i

        # Definite product (factorial)
        Math.prod(expr, i, 1, 5)  # = 1 * 2 * 3 * 4 * 5 = 120
        Math.prod(expr, i, bounds=(1, n))
    """

    def __init__(self, factor: Expr, var: Var,
                 lower: Optional[Union[Expr, int]] = None,
                 upper: Optional[Union[Expr, int]] = None):
        super().__init__()
        self.factor = factor
        self.var = var
        self.lower = _ensure_expr(lower) if lower is not None else None
        self.upper = _ensure_expr(upper) if upper is not None else None

    @property
    def is_definite(self) -> bool:
        """True if this is a definite product (has both bounds)."""
        return self.lower is not None and self.upper is not None

    def sum(self, var: Var, lower=None, upper=None, bounds=None) -> 'Sum':
        """Chain a summation onto this product."""
        if bounds is not None:
            lower, upper = bounds
        return Sum(self, var, lower, upper)

    def prod(self, var: Var, lower=None, upper=None, bounds=None) -> 'Product':
        """Chain another product."""
        if bounds is not None:
            lower, upper = bounds
        return Product(self, var, lower, upper)

    def to_sympy(self):
        sp = _get_sympy()
        factor_sympy = self.factor.to_sympy()
        var_sympy = self.var.to_sympy()

        if self.is_definite:
            lower_sympy = self.lower.to_sympy()
            upper_sympy = self.upper.to_sympy()
            return sp.Product(factor_sympy, (var_sympy, lower_sympy, upper_sympy))
        else:
            # Indefinite product - return unevaluated
            return sp.Product(factor_sympy, var_sympy)

    def to_latex(self) -> str:
        factor_latex = self.factor.to_latex()
        var_latex = self.var.to_latex()

        if self.is_definite:
            lower_latex = self.lower.to_latex()
            upper_latex = self.upper.to_latex()
            return f"\\prod_{{{var_latex}={lower_latex}}}^{{{upper_latex}}} {factor_latex}"
        else:
            return f"\\prod_{{{var_latex}}} {factor_latex}"

    def _get_free_variables(self) -> Set['Var']:
        # The product variable is bound, not free
        free_vars = self.factor._get_free_variables() - {self.var}

        # Add variables from symbolic bounds
        if self.lower is not None:
            free_vars |= self.lower._get_free_variables()
        if self.upper is not None:
            free_vars |= self.upper._get_free_variables()

        return free_vars

    def evaluate(self, **kwargs) -> Any:
        """
        Numerically evaluate the product.

        Raises:
            EvaluationError: If product is indefinite or has unresolved symbolic bounds
        """
        if not self.is_definite:
            raise EvaluationError(
                "Cannot numerically evaluate indefinite product.\n"
                "Provide bounds with .prod(var, lower, upper)\n"
                f"Expression: {self.to_latex()}"
            )

        # Use SymPy's doit() to compute the product
        sympy_prod = self.to_sympy()

        # Try to compute the product symbolically first
        computed = sympy_prod.doit()

        # Wrap in SympyWrapper and evaluate
        wrapper = SympyWrapper(computed)
        return wrapper.evaluate(**kwargs)


# =============================================================================
# Math Factory Class
# =============================================================================

class Math:
    """
    Factory for creating symbolic expressions.

    This is the main entry point for the symbolic computation library.

    Example:
        from cm.symbols import Math

        x = Math.var("x")
        y = Math.var("y")

        expr = x**2 + y**2
        result = expr.integrate(x, bounds=[0, 1])
        result.render()

        value = result.evaluate(y=2)  # Evaluates the integral
    """

    @staticmethod
    def var(name: str = None) -> Var:
        """
        Create a symbolic variable.

        Args:
            name: Variable name. If None, auto-generates (x_0, x_1, etc.)
                  Greek names like 'tau', 'theta' render as Greek letters.

        Example:
            x = Math.var()         # Auto-named x_0
            y = Math.var("y")      # Named y
            t = Math.var("tau")    # Renders as Greek tau
        """
        return Var(name)

    @staticmethod
    def const(value: Union[int, float]) -> Const:
        """Create a numeric constant."""
        return Const(value)

    @staticmethod
    def pi() -> SymbolicConst:
        """Create the constant pi."""
        sp = _get_sympy()
        return SymbolicConst('pi', sp.pi)

    @staticmethod
    def e() -> SymbolicConst:
        """Create Euler's number e."""
        sp = _get_sympy()
        return SymbolicConst('e', sp.E)

    @staticmethod
    def inf() -> SymbolicConst:
        """Create positive infinity."""
        sp = _get_sympy()
        return SymbolicConst('inf', sp.oo)

    @staticmethod
    def I() -> SymbolicConst:
        """Create the imaginary unit i."""
        sp = _get_sympy()
        return SymbolicConst('I', sp.I)

    # Mathematical functions
    @staticmethod
    def sqrt(expr: Union[Expr, int, float]) -> Sqrt:
        """Square root function."""
        return Sqrt(_ensure_expr(expr))

    @staticmethod
    def sin(expr: Union[Expr, int, float]) -> Sin:
        """Sine function."""
        return Sin(_ensure_expr(expr))

    @staticmethod
    def cos(expr: Union[Expr, int, float]) -> Cos:
        """Cosine function."""
        return Cos(_ensure_expr(expr))

    @staticmethod
    def tan(expr: Union[Expr, int, float]) -> Tan:
        """Tangent function."""
        return Tan(_ensure_expr(expr))

    @staticmethod
    def exp(expr: Union[Expr, int, float]) -> Exp:
        """Exponential function e^x."""
        return Exp(_ensure_expr(expr))

    @staticmethod
    def log(expr: Union[Expr, int, float]) -> Log:
        """Natural logarithm."""
        return Log(_ensure_expr(expr))

    @staticmethod
    def abs(expr: Union[Expr, int, float]) -> Abs:
        """Absolute value."""
        return Abs(_ensure_expr(expr))

    @staticmethod
    def expr(sympy_expr) -> SympyWrapper:
        """Wrap a SymPy expression."""
        return SympyWrapper(sympy_expr)

    # Summation and Product functions
    @staticmethod
    def sum(expr: Union[Expr, int, float], var: Var,
            lower=None, upper=None, bounds=None) -> Sum:
        """
        Create a summation expression.

        Args:
            expr: Expression to sum
            var: Summation variable
            lower: Lower bound (or use bounds parameter)
            upper: Upper bound (or use bounds parameter)
            bounds: Tuple of (lower, upper)

        Example:
            i = Math.var("i")
            n = Math.var("n")
            Math.sum(i**2, i, 1, n)
            Math.sum(i**2, i, bounds=(1, n))
        """
        if bounds is not None:
            lower, upper = bounds
        return Sum(_ensure_expr(expr), var, lower, upper)

    @staticmethod
    def prod(expr: Union[Expr, int, float], var: Var,
             lower=None, upper=None, bounds=None) -> Product:
        """
        Create a product expression.

        Args:
            expr: Expression to multiply
            var: Product variable
            lower: Lower bound (or use bounds parameter)
            upper: Upper bound (or use bounds parameter)
            bounds: Tuple of (lower, upper)

        Example:
            i = Math.var("i")
            Math.prod(i, i, 1, 5)  # = 120 (5!)
            Math.prod(i, i, bounds=(1, 5))
        """
        if bounds is not None:
            lower, upper = bounds
        return Product(_ensure_expr(expr), var, lower, upper)

    # Function composition API
    @staticmethod
    def function(expr: Optional[Expr] = None,
                 name: Optional[str] = None,
                 hyperparams: Optional[Dict[str, Any]] = None) -> 'SymbolicFunction':
        """
        Create a symbolic function with typed hyperparameters.

        Args:
            expr: Optional defining expression
            name: Optional function name
            hyperparams: Dict mapping parameter names to types (Scalar, ExprType, BoundsType)

        Example:
            a, b, x = Math.var("a"), Math.var("b"), Math.var("x")
            f = Math.function(a * Math.exp(b * x), hyperparams={"a": Scalar, "b": Scalar})
            f.save("MyExponential")

            f_inst = f.init(a=10, b=0.5)
            result = f_inst.run(x=2)
        """
        return SymbolicFunction(expr=expr, name=name, hyperparams=hyperparams)

    @staticmethod
    def get_function(name: str) -> 'SymbolicFunction':
        """
        Retrieve a saved function from the registry.

        Args:
            name: Name of the registered function

        Raises:
            KeyError: If function not found
        """
        func = _get_registered_function(name)
        if func is None:
            available = _list_registered_functions()
            raise KeyError(f"Function '{name}' not found in registry. "
                          f"Available: {available}")
        return func

    @staticmethod
    def list_functions() -> List[str]:
        """List all registered function names."""
        return _list_registered_functions()

    # =========================================================================
    # Special Functions - Factory Methods
    # =========================================================================

    # Gamma and related functions
    @staticmethod
    def gamma(z) -> 'Gamma':
        """Gamma function Γ(z)."""
        return Gamma(z)

    @staticmethod
    def loggamma(z) -> 'LogGamma':
        """Log-gamma function ln(Γ(z))."""
        return LogGamma(z)

    @staticmethod
    def digamma(z) -> 'Digamma':
        """Digamma function ψ(z) = d/dz ln(Γ(z))."""
        return Digamma(z)

    @staticmethod
    def beta(a, b) -> 'Beta':
        """Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)."""
        return Beta(a, b)

    @staticmethod
    def factorial(n) -> 'Factorial':
        """Factorial function n!"""
        return Factorial(n)

    @staticmethod
    def factorial2(n) -> 'DoubleFactorial':
        """Double factorial n!! = n(n-2)(n-4)..."""
        return DoubleFactorial(n)

    @staticmethod
    def binomial(n, k) -> 'Binomial':
        """Binomial coefficient C(n, k)."""
        return Binomial(n, k)

    # Error functions
    @staticmethod
    def erf(z) -> 'Erf':
        """Error function erf(z)."""
        return Erf(z)

    @staticmethod
    def erfc(z) -> 'Erfc':
        """Complementary error function erfc(z) = 1 - erf(z)."""
        return Erfc(z)

    @staticmethod
    def erfi(z) -> 'Erfi':
        """Imaginary error function erfi(z) = -i·erf(iz)."""
        return Erfi(z)

    # Bessel functions
    @staticmethod
    def besselj(nu, z) -> 'BesselJ':
        """Bessel function of the first kind J_ν(z)."""
        return BesselJ(nu, z)

    @staticmethod
    def bessely(nu, z) -> 'BesselY':
        """Bessel function of the second kind Y_ν(z)."""
        return BesselY(nu, z)

    @staticmethod
    def besseli(nu, z) -> 'BesselI':
        """Modified Bessel function of the first kind I_ν(z)."""
        return BesselI(nu, z)

    @staticmethod
    def besselk(nu, z) -> 'BesselK':
        """Modified Bessel function of the second kind K_ν(z)."""
        return BesselK(nu, z)

    @staticmethod
    def jn(n, z) -> 'SphericalBesselJ':
        """Spherical Bessel function of the first kind j_n(z)."""
        return SphericalBesselJ(n, z)

    @staticmethod
    def yn(n, z) -> 'SphericalBesselY':
        """Spherical Bessel function of the second kind y_n(z)."""
        return SphericalBesselY(n, z)

    @staticmethod
    def hankel1(nu, z) -> 'Hankel1':
        """Hankel function of the first kind H^(1)_ν(z)."""
        return Hankel1(nu, z)

    @staticmethod
    def hankel2(nu, z) -> 'Hankel2':
        """Hankel function of the second kind H^(2)_ν(z)."""
        return Hankel2(nu, z)

    # Airy functions
    @staticmethod
    def airyai(z) -> 'AiryAi':
        """Airy function Ai(z)."""
        return AiryAi(z)

    @staticmethod
    def airybi(z) -> 'AiryBi':
        """Airy function Bi(z)."""
        return AiryBi(z)

    @staticmethod
    def airyaiprime(z) -> 'AiryAiPrime':
        """Derivative of Airy function Ai'(z)."""
        return AiryAiPrime(z)

    @staticmethod
    def airybiprime(z) -> 'AiryBiPrime':
        """Derivative of Airy function Bi'(z)."""
        return AiryBiPrime(z)

    # Orthogonal polynomials
    @staticmethod
    def legendre(n, x) -> 'Legendre':
        """Legendre polynomial P_n(x)."""
        return Legendre(n, x)

    @staticmethod
    def assoc_legendre(n, m, x) -> 'AssocLegendre':
        """Associated Legendre function P_n^m(x)."""
        return AssocLegendre(n, m, x)

    @staticmethod
    def hermite(n, x) -> 'Hermite':
        """Hermite polynomial H_n(x) (physicist's convention)."""
        return Hermite(n, x)

    @staticmethod
    def hermite_prob(n, x) -> 'HermiteProb':
        """Probabilist's Hermite polynomial He_n(x)."""
        return HermiteProb(n, x)

    @staticmethod
    def laguerre(n, x) -> 'Laguerre':
        """Laguerre polynomial L_n(x)."""
        return Laguerre(n, x)

    @staticmethod
    def assoc_laguerre(n, alpha, x) -> 'AssocLaguerre':
        """Associated Laguerre polynomial L_n^(α)(x)."""
        return AssocLaguerre(n, alpha, x)

    @staticmethod
    def chebyshevt(n, x) -> 'Chebyshev1':
        """Chebyshev polynomial of the first kind T_n(x)."""
        return Chebyshev1(n, x)

    @staticmethod
    def chebyshevu(n, x) -> 'Chebyshev2':
        """Chebyshev polynomial of the second kind U_n(x)."""
        return Chebyshev2(n, x)

    @staticmethod
    def gegenbauer(n, alpha, x) -> 'Gegenbauer':
        """Gegenbauer (ultraspherical) polynomial C_n^(α)(x)."""
        return Gegenbauer(n, alpha, x)

    @staticmethod
    def jacobi(n, alpha, beta, x) -> 'Jacobi':
        """Jacobi polynomial P_n^(α,β)(x)."""
        return Jacobi(n, alpha, beta, x)

    # Spherical harmonics
    @staticmethod
    def Ylm(l, m, theta, phi) -> 'SphericalHarmonic':
        """Spherical harmonic Y_l^m(θ, φ)."""
        return SphericalHarmonic(l, m, theta, phi)

    @staticmethod
    def Ylm_real(l, m, theta, phi) -> 'RealSphericalHarmonic':
        """Real spherical harmonic Y_{lm}(θ, φ)."""
        return RealSphericalHarmonic(l, m, theta, phi)

    # Hypergeometric functions
    @staticmethod
    def hyper2f1(a, b, c, z) -> 'Hypergeometric2F1':
        """Gauss hypergeometric function ₂F₁(a, b; c; z)."""
        return Hypergeometric2F1(a, b, c, z)

    @staticmethod
    def hyper1f1(a, b, z) -> 'Hypergeometric1F1':
        """Confluent hypergeometric function ₁F₁(a; b; z)."""
        return Hypergeometric1F1(a, b, z)

    @staticmethod
    def hyper0f1(b, z) -> 'Hypergeometric0F1':
        """Confluent hypergeometric limit function ₀F₁(; b; z)."""
        return Hypergeometric0F1(b, z)

    @staticmethod
    def hyperpfq(a_list: List, b_list: List, z) -> 'HypergeometricPFQ':
        """Generalized hypergeometric function ₚFq."""
        return HypergeometricPFQ(a_list, b_list, z)

    # Elliptic integrals
    @staticmethod
    def elliptic_k(m) -> 'EllipticK':
        """Complete elliptic integral of the first kind K(m)."""
        return EllipticK(m)

    @staticmethod
    def elliptic_e(m) -> 'EllipticE':
        """Complete elliptic integral of the second kind E(m)."""
        return EllipticE(m)

    @staticmethod
    def elliptic_pi(n, m) -> 'EllipticPi':
        """Complete elliptic integral of the third kind Π(n, m)."""
        return EllipticPi(n, m)

    # Other special functions
    @staticmethod
    def zeta(s) -> 'Zeta':
        """Riemann zeta function ζ(s)."""
        return Zeta(s)

    @staticmethod
    def polylog(s, z) -> 'PolyLog':
        """Polylogarithm Li_s(z)."""
        return PolyLog(s, z)

    @staticmethod
    def dirac(x) -> 'DiracDelta':
        """Dirac delta function δ(x)."""
        return DiracDelta(x)

    @staticmethod
    def heaviside(x) -> 'Heaviside':
        """Heaviside step function θ(x)."""
        return Heaviside(x)

    @staticmethod
    def kronecker(i, j) -> 'KroneckerDelta':
        """Kronecker delta δ_{ij}."""
        return KroneckerDelta(i, j)

    @staticmethod
    def levi_civita(*indices) -> 'LeviCivita':
        """Levi-Civita symbol ε_{i,j,k,...}."""
        return LeviCivita(*indices)


# Convenience: expose inf for bounds
inf = float('inf')


# =============================================================================
# Hyperparameter Type System
# =============================================================================

class ParamType(ABC):
    """
    Base class for hyperparameter types.

    Used to define and validate hyperparameters in SymbolicFunction.
    """

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Return True if value is valid for this type."""
        pass

    @abstractmethod
    def coerce(self, value: Any) -> Any:
        """Coerce value to appropriate type."""
        pass


class Scalar(ParamType):
    """
    Numeric scalar type (int or float).

    Example:
        func = Math.function(a * x, hyperparams={"a": Scalar})
        inst = func.init(a=10)  # Validates that a is numeric
    """

    def validate(self, value: Any) -> bool:
        return isinstance(value, (int, float))

    def coerce(self, value: Any) -> Union[int, float]:
        if isinstance(value, (int, float)):
            return value
        return float(value)


class ExprType(ParamType):
    """
    Expression type - accepts Expr or converts numbers to Const.

    Example:
        func = Math.function(a * x, hyperparams={"a": ExprType})
        inst = func.init(a=x**2)  # Can pass an expression
    """

    def validate(self, value: Any) -> bool:
        return isinstance(value, (Expr, int, float))

    def coerce(self, value: Any) -> Expr:
        return _ensure_expr(value)


class BoundsType(ParamType):
    """
    Bounds tuple type (lower, upper).

    Example:
        func = Math.function(expr.sum(i, bounds=b), hyperparams={"b": BoundsType})
        inst = func.init(b=(0, 10))
    """

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return False
        return all(isinstance(v, (int, float, Expr)) for v in value)

    def coerce(self, value: Any) -> Tuple[Expr, Expr]:
        lower, upper = value
        return (_ensure_expr(lower), _ensure_expr(upper))


@dataclass
class ParamSpec:
    """Specification for a function hyperparameter."""
    name: str
    param_type: ParamType
    default: Optional[Any] = None
    description: str = ""


# =============================================================================
# Function Registry
# =============================================================================

_FUNCTION_REGISTRY: Dict[str, 'SymbolicFunction'] = {}


def _register_function(name: str, func: 'SymbolicFunction') -> None:
    """Register a function in the global registry."""
    _FUNCTION_REGISTRY[name] = func


def _get_registered_function(name: str) -> Optional['SymbolicFunction']:
    """Retrieve a function from the registry."""
    return _FUNCTION_REGISTRY.get(name)


def _list_registered_functions() -> List[str]:
    """List all registered function names."""
    return list(_FUNCTION_REGISTRY.keys())


# =============================================================================
# SymbolicFunction and Related Classes
# =============================================================================

class SymbolicFunction:
    """
    A user-defined symbolic function with typed hyperparameters.

    Functions are defined with hyperparameters (fixed at instantiation)
    and free variables (bound at evaluation time).

    Example:
        # Define function with hyperparameters
        a = Math.var("a")
        b = Math.var("b")
        x = Math.var("x")

        func = Math.function(
            a * Math.exp(b * x),
            hyperparams={"a": Scalar, "b": Scalar}
        )

        # Save for later retrieval
        func.save("MyExponential")

        # Instantiate with hyperparameter values
        inst = func.init(a=10, b=0.5)

        # Evaluate
        result = inst.run(x=2)  # Eager evaluation
        cg = inst.run_with(x=2)  # Lazy evaluation (compute graph)
    """

    def __init__(self,
                 expr: Optional[Expr] = None,
                 name: Optional[str] = None,
                 hyperparams: Optional[Dict[str, Union[ParamType, type]]] = None):
        self.name = name
        self._hyperparams: Dict[str, ParamSpec] = {}
        self._expr: Optional[Expr] = None
        self._hyperparam_vars: Dict[str, Var] = {}

        # Parse hyperparams specification
        if hyperparams:
            for param_name, param_type in hyperparams.items():
                # Allow passing class or instance
                if isinstance(param_type, type) and issubclass(param_type, ParamType):
                    param_type = param_type()
                elif not isinstance(param_type, ParamType):
                    raise TypeError(f"Invalid param type for '{param_name}': expected ParamType subclass")
                self._hyperparams[param_name] = ParamSpec(param_name, param_type)
                self._hyperparam_vars[param_name] = Var(param_name)

        if expr is not None:
            self.define(expr)

    def define(self, expr: Expr) -> 'SymbolicFunction':
        """Define the function expression."""
        self._expr = expr
        return self

    @property
    def free_variables(self) -> Set[Var]:
        """Return free variables (not hyperparameters)."""
        if self._expr is None:
            return set()
        all_vars = self._expr._get_free_variables()
        hyperparam_names = set(self._hyperparam_vars.keys())
        return {v for v in all_vars if v.name not in hyperparam_names}

    @property
    def expression(self) -> Optional[Expr]:
        """Return the defining expression."""
        return self._expr

    @property
    def hyperparam_names(self) -> List[str]:
        """Return list of hyperparameter names."""
        return list(self._hyperparams.keys())

    def init(self, **kwargs) -> 'BoundFunction':
        """
        Bind hyperparameters and return a BoundFunction.

        Args:
            **kwargs: Hyperparameter name -> value mappings

        Returns:
            BoundFunction with hyperparameters bound

        Raises:
            ValueError: If required hyperparameters are missing
            TypeError: If hyperparameter value has wrong type
        """
        # Validate all required hyperparams provided
        missing = set(self._hyperparams.keys()) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing hyperparameters: {missing}")

        # Validate and coerce values
        bound_values = {}
        for name, value in kwargs.items():
            if name not in self._hyperparams:
                raise ValueError(f"Unknown hyperparameter: {name}. "
                               f"Available: {list(self._hyperparams.keys())}")
            spec = self._hyperparams[name]
            if not spec.param_type.validate(value):
                raise TypeError(
                    f"Invalid type for '{name}': got {type(value).__name__}, "
                    f"expected {spec.param_type.__class__.__name__}"
                )
            bound_values[name] = spec.param_type.coerce(value)

        return BoundFunction(self, bound_values)

    def save(self, name: str) -> 'SymbolicFunction':
        """
        Save function to the in-memory registry.

        Args:
            name: Name to register the function under

        Returns:
            Self for chaining
        """
        self.name = name
        _register_function(name, self)
        return self

    def to_latex(self) -> str:
        """Return LaTeX representation of the function definition."""
        if self._expr is None:
            return f"\\text{{{self.name or 'f'}}}(\\ldots)"
        return self._expr.to_latex()

    def render(self, display: bool = True, justify: str = "center"):
        """Render the function definition."""
        if self._expr:
            self._expr.render(display=display, justify=justify)

    def __repr__(self):
        params = ", ".join(self._hyperparams.keys())
        return f"SymbolicFunction(name={self.name!r}, hyperparams=[{params}])"


class BoundFunction:
    """
    A function with hyperparameters bound to specific values.

    This is the result of calling func.init(a=10, b=20).
    """

    def __init__(self, func: SymbolicFunction, hyperparam_values: Dict[str, Any]):
        self._func = func
        self._hyperparam_values = hyperparam_values
        self._bound_expr: Optional[Expr] = None

        # Create bound expression by substituting hyperparameters
        if func.expression is not None:
            subs = {}
            for name, value in hyperparam_values.items():
                var = func._hyperparam_vars[name]
                if isinstance(value, Expr):
                    subs[var] = value
                else:
                    subs[var] = Const(value)
            self._bound_expr = func.expression.subs(subs)

    @property
    def free_variables(self) -> Set[Var]:
        """Return remaining free variables after hyperparameter binding."""
        if self._bound_expr is None:
            return set()
        return self._bound_expr._get_free_variables()

    @property
    def hyperparam_values(self) -> Dict[str, Any]:
        """Return the bound hyperparameter values."""
        return self._hyperparam_values.copy()

    def run_with(self, **kwargs) -> 'ComputeGraph':
        """
        Create a compute graph with variable bindings (lazy evaluation).

        Args:
            **kwargs: Variable name -> value mappings

        Returns:
            ComputeGraph object (not yet evaluated)
        """
        return ComputeGraph(self, kwargs)

    def run(self, **kwargs) -> Any:
        """
        Eagerly evaluate the function with given variable values.

        Args:
            **kwargs: Variable name -> value mappings

        Returns:
            Numeric result
        """
        if self._bound_expr is None:
            raise EvaluationError("Function has no expression defined")
        return self._bound_expr.evaluate(**kwargs)

    def to_latex(self) -> str:
        """Return LaTeX with hyperparameter values shown."""
        if self._bound_expr is None:
            return ""
        return self._bound_expr.to_latex()

    def render(self, display: bool = True, justify: str = "center"):
        """
        Render with 'Expression where param=value' format.
        """
        if self._bound_expr is None:
            return

        # Build "where" clause
        param_strs = []
        for name, value in self._hyperparam_values.items():
            if isinstance(value, Expr):
                param_strs.append(f"{name}={value.to_latex()}")
            else:
                param_strs.append(f"{name}={value}")

        where_clause = ", ".join(param_strs)
        full_latex = f"{self._bound_expr.to_latex()} \\quad \\text{{where }} {where_clause}"

        delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
        html = f'<div class="cm-math cm-math-{justify}" style="line-height: 1.5;">{delim_start} {full_latex} {delim_end}</div>'
        views.html(html)

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self._hyperparam_values.items())
        return f"BoundFunction({params})"


class ComputeGraph:
    """
    A lazy computation graph with all bindings specified.

    This is the result of bound_func.run_with(x=5, y=10).
    The expression is not evaluated until .evaluate() or .result is accessed.
    """

    def __init__(self, bound_func: BoundFunction, var_bindings: Dict[str, Any]):
        self._bound_func = bound_func
        self._var_bindings = var_bindings
        self._notation_style: Optional[str] = None
        self._result_cache: Optional[Any] = None

    def notation(self, style: str) -> 'ComputeGraph':
        """
        Set notation style for rendering.

        Args:
            style: One of 'standard', 'physicist', 'chemist', 'braket', 'engineering'

        Returns:
            Self for chaining
        """
        self._notation_style = style
        return self

    def evaluate(self) -> Any:
        """Execute the computation and return result."""
        if self._result_cache is None:
            if self._bound_func._bound_expr is None:
                raise EvaluationError("No expression to evaluate")
            self._result_cache = self._bound_func._bound_expr.evaluate(**self._var_bindings)
        return self._result_cache

    @property
    def result(self) -> Any:
        """Alias for evaluate()."""
        return self.evaluate()

    @property
    def var_bindings(self) -> Dict[str, Any]:
        """Return the variable bindings."""
        return self._var_bindings.copy()

    def to_latex(self, show_substituted: bool = True) -> str:
        """
        Generate LaTeX representation.

        Args:
            show_substituted: If True, show numeric values substituted
        """
        if self._bound_func._bound_expr is None:
            return ""

        if show_substituted:
            # Substitute values into expression for display
            subs = {}
            for name, value in self._var_bindings.items():
                var = Var(name)
                if isinstance(value, Expr):
                    subs[var] = value
                else:
                    subs[var] = Const(value)
            substituted = self._bound_func._bound_expr.subs(subs)
            return substituted.to_latex()
        else:
            return self._bound_func._bound_expr.to_latex()

    def render(self, display: bool = True, justify: str = "center"):
        """
        Render the compute graph with substituted values.
        """
        # Temporarily switch notation if specified
        old_notation = None
        if self._notation_style:
            old_notation = get_notation()
            set_notation(self._notation_style)

        try:
            latex_str = self.to_latex(show_substituted=True)
            delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
            html = f'<div class="cm-math cm-math-{justify}" style="line-height: 1.5;">{delim_start} {latex_str} {delim_end}</div>'
            views.html(html)
        finally:
            # Restore notation
            if old_notation:
                set_notation(old_notation)

    def compile(self, backend: str = 'torch', device: str = 'cpu') -> 'TorchFunction':
        """
        Compile the compute graph to a PyTorch function for GPU acceleration.

        Args:
            backend: Compilation backend ('torch' supported)
            device: PyTorch device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            TorchFunction that can be called with tensor inputs

        Example:
            cg = inst.run_with(x=2)
            torch_fn = cg.compile(device='cuda')
            result = torch_fn(x=torch.tensor([1.0, 2.0, 3.0]))
        """
        if backend != 'torch':
            raise ValueError(f"Unsupported backend: {backend}. Currently only 'torch' is supported.")

        if self._bound_func._bound_expr is None:
            raise EvaluationError("No expression to compile")

        return TorchFunction(self._bound_func._bound_expr, device=device)

    def to_torch(self, device: str = 'cpu') -> 'TorchFunction':
        """
        Shorthand for compile(backend='torch', device=device).

        Args:
            device: PyTorch device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            TorchFunction
        """
        return self.compile(backend='torch', device=device)

    def __repr__(self):
        bindings = ", ".join(f"{k}={v}" for k, v in self._var_bindings.items())
        return f"ComputeGraph({bindings})"


class TorchFunction:
    """
    A compiled PyTorch function from a symbolic expression.

    Supports:
    - GPU acceleration via device placement
    - Automatic differentiation via autograd
    - Batched evaluation with tensor inputs
    - JIT compilation via torch.compile (optional)

    Example:
        # Create and compile
        a, x = Math.var("a"), Math.var("x")
        func = Math.function(a * Math.exp(x), hyperparams={"a": Scalar})
        inst = func.init(a=2.0)
        torch_fn = inst.run_with(x=0).to_torch(device='cuda')

        # Evaluate with tensors
        x_vals = torch.linspace(0, 1, 100, device='cuda')
        y_vals = torch_fn(x=x_vals)

        # Compute gradients
        x_vals.requires_grad = True
        y = torch_fn(x=x_vals)
        dy_dx = torch.autograd.grad(y.sum(), x_vals)[0]
    """

    def __init__(self, expr: Expr, device: str = 'cpu', use_jit: bool = False):
        """
        Compile a symbolic expression to PyTorch.

        Args:
            expr: The symbolic expression to compile
            device: PyTorch device
            use_jit: Whether to apply torch.compile() for optimization
        """
        self._expr = expr
        self._device = device
        self._use_jit = use_jit
        self._compiled_fn: Optional[Callable] = None
        self._input_vars: List[str] = []

        # Compile on initialization
        self._compile()

    def _compile(self):
        """Compile the expression to a PyTorch function using SymPy's lambdify."""
        sp = _get_sympy()
        torch = _get_torch()

        # Get free variables and sort them for consistent ordering
        free_vars = self._expr._get_free_variables()
        self._input_vars = sorted([v.name for v in free_vars])

        if not self._input_vars:
            # No free variables - just evaluate to a constant
            sympy_expr = self._expr.to_sympy()
            const_val = float(sympy_expr.evalf())
            device = self._device  # Capture for closure
            self._compiled_fn = lambda **_: torch.tensor(const_val, device=device)
            return

        # Create SymPy symbols in sorted order
        sympy_symbols = [sp.Symbol(name) for name in self._input_vars]
        sympy_expr = self._expr.to_sympy()

        # Use lambdify with torch module for GPU support
        # Note: SymPy's torch module maps to PyTorch functions
        try:
            self._compiled_fn = sp.lambdify(
                sympy_symbols,
                sympy_expr,
                modules=[_torch_module_mapping(), 'numpy']
            )
        except Exception as e:
            raise EvaluationError(f"Failed to compile expression to PyTorch: {e}")

        # Optionally apply torch.compile for JIT optimization
        if self._use_jit and hasattr(torch, 'compile'):
            self._compiled_fn = torch.compile(self._compiled_fn)

    def __call__(self, **kwargs) -> Any:
        """
        Evaluate the compiled function with PyTorch tensors.

        Args:
            **kwargs: Variable name -> tensor/scalar mappings

        Returns:
            PyTorch tensor result
        """
        torch = _get_torch()

        # Validate inputs
        missing = set(self._input_vars) - set(kwargs.keys())
        if missing:
            raise EvaluationError(f"Missing input variables: {missing}")

        # Convert inputs to tensors on the correct device
        tensor_inputs = []
        for var_name in self._input_vars:
            val = kwargs[var_name]
            if isinstance(val, torch.Tensor):
                tensor_inputs.append(val.to(self._device))
            else:
                tensor_inputs.append(torch.tensor(val, device=self._device, dtype=torch.float32))

        # Call the compiled function
        result = self._compiled_fn(*tensor_inputs)

        # Ensure result is a tensor on the correct device
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, device=self._device, dtype=torch.float32)

        return result

    def grad(self, output_var: Optional[str] = None) -> 'TorchGradFunction':
        """
        Create a function that computes gradients.

        Args:
            output_var: Variable to differentiate with respect to (None = all)

        Returns:
            TorchGradFunction for computing gradients

        Example:
            torch_fn = cg.to_torch()
            grad_fn = torch_fn.grad()
            x = torch.tensor([1.0, 2.0], requires_grad=True)
            grads = grad_fn(x=x)  # Returns dict of gradients
        """
        return TorchGradFunction(self, output_var)

    @property
    def device(self) -> str:
        """Return the device this function runs on."""
        return self._device

    @property
    def input_vars(self) -> List[str]:
        """Return the list of input variable names."""
        return self._input_vars.copy()

    def to(self, device: str) -> 'TorchFunction':
        """
        Move the function to a different device.

        Args:
            device: New device ('cpu', 'cuda', etc.)

        Returns:
            New TorchFunction on the specified device
        """
        return TorchFunction(self._expr, device=device, use_jit=self._use_jit)

    def cuda(self) -> 'TorchFunction':
        """Move to CUDA device."""
        return self.to('cuda')

    def cpu(self) -> 'TorchFunction':
        """Move to CPU."""
        return self.to('cpu')

    def __repr__(self):
        return f"TorchFunction(inputs={self._input_vars}, device='{self._device}')"


class TorchGradFunction:
    """
    A function that computes gradients of a TorchFunction.

    Uses PyTorch autograd for automatic differentiation.
    """

    def __init__(self, torch_fn: TorchFunction, wrt: Optional[str] = None):
        """
        Args:
            torch_fn: The TorchFunction to differentiate
            wrt: Variable to differentiate with respect to (None = all inputs)
        """
        self._torch_fn = torch_fn
        self._wrt = wrt

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Compute gradients at the given input values.

        Args:
            **kwargs: Variable name -> tensor mappings (requires_grad should be True)

        Returns:
            Dict mapping variable names to their gradients
        """
        torch = _get_torch()

        # Ensure inputs require grad
        inputs = {}
        for name in self._torch_fn.input_vars:
            if name not in kwargs:
                raise EvaluationError(f"Missing input: {name}")
            val = kwargs[name]
            if isinstance(val, torch.Tensor):
                if not val.requires_grad:
                    val = val.detach().requires_grad_(True)
                inputs[name] = val.to(self._torch_fn.device)
            else:
                inputs[name] = torch.tensor(
                    val,
                    device=self._torch_fn.device,
                    dtype=torch.float32,
                    requires_grad=True
                )

        # Forward pass
        output = self._torch_fn(**inputs)

        # Compute gradients
        if self._wrt:
            # Gradient with respect to specific variable
            if self._wrt not in inputs:
                raise EvaluationError(f"Variable '{self._wrt}' not in inputs")
            grad = torch.autograd.grad(
                output.sum(),
                inputs[self._wrt],
                create_graph=True
            )[0]
            return {self._wrt: grad}
        else:
            # Gradients with respect to all inputs
            grads = torch.autograd.grad(
                output.sum(),
                list(inputs.values()),
                create_graph=True
            )
            return dict(zip(inputs.keys(), grads))

    def __repr__(self):
        wrt_str = f"wrt='{self._wrt}'" if self._wrt else "wrt=all"
        return f"TorchGradFunction({wrt_str})"


def _torch_module_mapping() -> Dict[str, Callable]:
    """
    Create a mapping of mathematical functions to their PyTorch equivalents.

    This is used by SymPy's lambdify to generate PyTorch-compatible code.
    """
    torch = _get_torch()

    return {
        'sin': torch.sin,
        'cos': torch.cos,
        'tan': torch.tan,
        'exp': torch.exp,
        'log': torch.log,
        'sqrt': torch.sqrt,
        'abs': torch.abs,
        'Abs': torch.abs,
        'sign': torch.sign,
        'floor': torch.floor,
        'ceiling': torch.ceil,
        'asin': torch.asin,
        'acos': torch.acos,
        'atan': torch.atan,
        'atan2': torch.atan2,
        'sinh': torch.sinh,
        'cosh': torch.cosh,
        'tanh': torch.tanh,
        'asinh': torch.asinh,
        'acosh': torch.acosh,
        'atanh': torch.atanh,
        'erf': torch.erf,
        'erfc': torch.erfc,
        'gamma': torch.lgamma,  # Note: torch has lgamma, not gamma
        'pi': torch.pi,
        'E': torch.e,
        'I': 1j,  # Complex imaginary unit
        'oo': float('inf'),
        'zoo': complex('inf'),
        'nan': float('nan'),
        # Power and exponential
        'Pow': torch.pow,
        'exp2': lambda x: torch.pow(2, x),
        'log2': torch.log2,
        'log10': torch.log10,
        # Min/max
        'Min': torch.minimum,
        'Max': torch.maximum,
        # Rounding
        'round': torch.round,
        'trunc': torch.trunc,
    }


# =============================================================================
# SPECIAL FUNCTIONS
# =============================================================================
# Mathematical special functions commonly used in physics and engineering.
# These include Bessel functions, orthogonal polynomials, spherical harmonics,
# hypergeometric functions, and more.
#
# All special functions:
# - Extend Expr for seamless integration with the symbolic computation library
# - Support conversion to SymPy via .to_sympy()
# - Support LaTeX rendering via .to_latex() and .render()
# - Support numerical evaluation via .evaluate()
# =============================================================================


class SpecialFunction(Expr):
    """
    Base class for special mathematical functions.

    Special functions have:
    - A name (for display and SymPy conversion)
    - Multiple arguments (unlike UnaryOp which has one)
    - Custom LaTeX formatting
    """

    # Override in subclasses
    _name: str = ""
    _latex_name: str = ""
    _sympy_func: str = ""

    def __init__(self, *args):
        super().__init__()
        self.args = tuple(_ensure_expr(a) for a in args)

    def _get_free_variables(self) -> Set['Var']:
        result = set()
        for arg in self.args:
            result |= arg._get_free_variables()
        return result

    def to_sympy(self):
        """Convert to SymPy. Override in subclasses for custom behavior."""
        sp = _get_sympy()
        sympy_func = getattr(sp, self._sympy_func, None)
        if sympy_func is None:
            raise NotImplementedError(f"SymPy function {self._sympy_func} not found")
        sympy_args = [arg.to_sympy() for arg in self.args]
        return sympy_func(*sympy_args)

    def to_latex(self) -> str:
        """Generate LaTeX. Override in subclasses for custom formatting."""
        args_latex = ", ".join(arg.to_latex() for arg in self.args)
        return f"{self._latex_name}\\left({args_latex}\\right)"

    def __repr__(self):
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.__class__.__name__}({args_str})"


# =============================================================================
# Gamma and Related Functions
# =============================================================================

class Gamma(SpecialFunction):
    """
    Gamma function Γ(z).

    The gamma function generalizes the factorial: Γ(n) = (n-1)!

    Example:
        z = Math.var("z")
        Math.gamma(z).render()  # Renders Γ(z)
        Math.gamma(5).evaluate()  # Returns 24 (= 4!)
    """
    _name = "gamma"
    _latex_name = r"\Gamma"
    _sympy_func = "gamma"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\Gamma\\left({self.z.to_latex()}\\right)"


class LogGamma(SpecialFunction):
    """
    Log-gamma function ln(Γ(z)).

    More numerically stable than log(gamma(z)) for large z.

    Example:
        Math.loggamma(100).evaluate()  # ln(99!) without overflow
    """
    _name = "loggamma"
    _latex_name = r"\ln\Gamma"
    _sympy_func = "loggamma"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\ln\\Gamma\\left({self.z.to_latex()}\\right)"


class Digamma(SpecialFunction):
    """
    Digamma function ψ(z) = d/dz ln(Γ(z)).

    Also known as psi function.
    """
    _name = "digamma"
    _latex_name = r"\psi"
    _sympy_func = "digamma"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\psi\\left({self.z.to_latex()}\\right)"


class Beta(SpecialFunction):
    """
    Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).

    Example:
        a, b = Math.var("a"), Math.var("b")
        Math.beta(a, b).render()
    """
    _name = "beta"
    _latex_name = r"\mathrm{B}"
    _sympy_func = "beta"

    def __init__(self, a, b):
        super().__init__(a, b)

    @property
    def a(self) -> Expr:
        return self.args[0]

    @property
    def b(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        return f"\\mathrm{{B}}\\left({self.a.to_latex()}, {self.b.to_latex()}\\right)"


class Factorial(SpecialFunction):
    """
    Factorial function n!

    Example:
        n = Math.var("n")
        Math.factorial(n).render()  # n!
        Math.factorial(5).evaluate()  # 120
    """
    _name = "factorial"
    _latex_name = ""
    _sympy_func = "factorial"

    def __init__(self, n):
        super().__init__(n)

    @property
    def n(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        # Wrap in parens if complex expression
        if isinstance(self.n, (BinOp, UnaryOp)):
            n_latex = f"\\left({n_latex}\\right)"
        return f"{n_latex}!"


class DoubleFactorial(SpecialFunction):
    """
    Double factorial n!! = n(n-2)(n-4)...

    Example:
        Math.factorial2(7).evaluate()  # 7*5*3*1 = 105
    """
    _name = "factorial2"
    _latex_name = ""
    _sympy_func = "factorial2"

    def __init__(self, n):
        super().__init__(n)

    @property
    def n(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        if isinstance(self.n, (BinOp, UnaryOp)):
            n_latex = f"\\left({n_latex}\\right)"
        return f"{n_latex}!!"


class Binomial(SpecialFunction):
    """
    Binomial coefficient C(n, k) = n! / (k!(n-k)!)

    Example:
        Math.binomial(10, 3).evaluate()  # 120
    """
    _name = "binomial"
    _latex_name = ""
    _sympy_func = "binomial"

    def __init__(self, n, k):
        super().__init__(n, k)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def k(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        return f"\\binom{{{self.n.to_latex()}}}{{{self.k.to_latex()}}}"


# =============================================================================
# Error Functions
# =============================================================================

class Erf(SpecialFunction):
    """
    Error function erf(z) = (2/√π) ∫₀ᶻ e^(-t²) dt

    Example:
        x = Math.var("x")
        Math.erf(x).render()
    """
    _name = "erf"
    _latex_name = r"\mathrm{erf}"
    _sympy_func = "erf"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\mathrm{{erf}}\\left({self.z.to_latex()}\\right)"


class Erfc(SpecialFunction):
    """
    Complementary error function erfc(z) = 1 - erf(z)

    Example:
        Math.erfc(1).evaluate()  # ≈ 0.1573
    """
    _name = "erfc"
    _latex_name = r"\mathrm{erfc}"
    _sympy_func = "erfc"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\mathrm{{erfc}}\\left({self.z.to_latex()}\\right)"


class Erfi(SpecialFunction):
    """
    Imaginary error function erfi(z) = -i·erf(iz)

    Example:
        Math.erfi(1).evaluate()  # ≈ 1.6504
    """
    _name = "erfi"
    _latex_name = r"\mathrm{erfi}"
    _sympy_func = "erfi"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\mathrm{{erfi}}\\left({self.z.to_latex()}\\right)"


# =============================================================================
# Bessel Functions
# =============================================================================

class BesselJ(SpecialFunction):
    """
    Bessel function of the first kind J_ν(z).

    Solves Bessel's differential equation: z²y'' + zy' + (z² - ν²)y = 0

    Example:
        nu, z = Math.var("nu"), Math.var("z")
        Math.besselj(nu, z).render()  # J_ν(z)
        Math.besselj(0, 2.4048).evaluate()  # ≈ 0 (first zero of J₀)
    """
    _name = "besselj"
    _latex_name = "J"
    _sympy_func = "besselj"

    def __init__(self, nu, z):
        super().__init__(nu, z)

    @property
    def nu(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        nu_latex = self.nu.to_latex()
        z_latex = self.z.to_latex()
        return f"J_{{{nu_latex}}}\\left({z_latex}\\right)"


class BesselY(SpecialFunction):
    """
    Bessel function of the second kind Y_ν(z).

    Also called Neumann function N_ν(z) or Weber function.

    Example:
        Math.bessely(0, 1).evaluate()  # ≈ 0.0883
    """
    _name = "bessely"
    _latex_name = "Y"
    _sympy_func = "bessely"

    def __init__(self, nu, z):
        super().__init__(nu, z)

    @property
    def nu(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        nu_latex = self.nu.to_latex()
        z_latex = self.z.to_latex()
        return f"Y_{{{nu_latex}}}\\left({z_latex}\\right)"


class BesselI(SpecialFunction):
    """
    Modified Bessel function of the first kind I_ν(z).

    Exponentially growing solution to the modified Bessel equation.

    Example:
        Math.besseli(0, 1).evaluate()  # ≈ 1.2661
    """
    _name = "besseli"
    _latex_name = "I"
    _sympy_func = "besseli"

    def __init__(self, nu, z):
        super().__init__(nu, z)

    @property
    def nu(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        nu_latex = self.nu.to_latex()
        z_latex = self.z.to_latex()
        return f"I_{{{nu_latex}}}\\left({z_latex}\\right)"


class BesselK(SpecialFunction):
    """
    Modified Bessel function of the second kind K_ν(z).

    Exponentially decaying solution to the modified Bessel equation.
    Also called MacDonald function.

    Example:
        Math.besselk(0, 1).evaluate()  # ≈ 0.4210
    """
    _name = "besselk"
    _latex_name = "K"
    _sympy_func = "besselk"

    def __init__(self, nu, z):
        super().__init__(nu, z)

    @property
    def nu(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        nu_latex = self.nu.to_latex()
        z_latex = self.z.to_latex()
        return f"K_{{{nu_latex}}}\\left({z_latex}\\right)"


class SphericalBesselJ(SpecialFunction):
    """
    Spherical Bessel function of the first kind j_n(z).

    j_n(z) = √(π/(2z)) J_{n+1/2}(z)

    Used in solutions to the Helmholtz equation in spherical coordinates.

    Example:
        Math.jn(0, Math.pi()).evaluate()  # = 0 (j₀(π) = sin(π)/π = 0)
    """
    _name = "jn"
    _latex_name = "j"
    _sympy_func = "jn"

    def __init__(self, n, z):
        super().__init__(n, z)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        z_latex = self.z.to_latex()
        return f"j_{{{n_latex}}}\\left({z_latex}\\right)"


class SphericalBesselY(SpecialFunction):
    """
    Spherical Bessel function of the second kind y_n(z).

    y_n(z) = √(π/(2z)) Y_{n+1/2}(z)

    Also called spherical Neumann function.

    Example:
        Math.yn(0, 1).evaluate()  # = -cos(1)/1 ≈ -0.5403
    """
    _name = "yn"
    _latex_name = "y"
    _sympy_func = "yn"

    def __init__(self, n, z):
        super().__init__(n, z)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        z_latex = self.z.to_latex()
        return f"y_{{{n_latex}}}\\left({z_latex}\\right)"


class Hankel1(SpecialFunction):
    """
    Hankel function of the first kind H^(1)_ν(z) = J_ν(z) + iY_ν(z).

    Represents outgoing cylindrical waves.

    Example:
        Math.hankel1(0, 1).evaluate()  # Complex result
    """
    _name = "hankel1"
    _latex_name = r"H^{(1)}"
    _sympy_func = "hankel1"

    def __init__(self, nu, z):
        super().__init__(nu, z)

    @property
    def nu(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        nu_latex = self.nu.to_latex()
        z_latex = self.z.to_latex()
        return f"H^{{(1)}}_{{{nu_latex}}}\\left({z_latex}\\right)"


class Hankel2(SpecialFunction):
    """
    Hankel function of the second kind H^(2)_ν(z) = J_ν(z) - iY_ν(z).

    Represents incoming cylindrical waves.
    """
    _name = "hankel2"
    _latex_name = r"H^{(2)}"
    _sympy_func = "hankel2"

    def __init__(self, nu, z):
        super().__init__(nu, z)

    @property
    def nu(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        nu_latex = self.nu.to_latex()
        z_latex = self.z.to_latex()
        return f"H^{{(2)}}_{{{nu_latex}}}\\left({z_latex}\\right)"


# =============================================================================
# Airy Functions
# =============================================================================

class AiryAi(SpecialFunction):
    """
    Airy function Ai(z).

    Solution to y'' - xy = 0 that decays as x → +∞.
    Used in WKB approximation and quantum tunneling.

    Example:
        Math.airyai(0).evaluate()  # ≈ 0.3550
    """
    _name = "airyai"
    _latex_name = r"\mathrm{Ai}"
    _sympy_func = "airyai"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\mathrm{{Ai}}\\left({self.z.to_latex()}\\right)"


class AiryBi(SpecialFunction):
    """
    Airy function Bi(z).

    Solution to y'' - xy = 0 that grows as x → +∞.

    Example:
        Math.airybi(0).evaluate()  # ≈ 0.6149
    """
    _name = "airybi"
    _latex_name = r"\mathrm{Bi}"
    _sympy_func = "airybi"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\mathrm{{Bi}}\\left({self.z.to_latex()}\\right)"


class AiryAiPrime(SpecialFunction):
    """
    Derivative of Airy function Ai'(z).

    Example:
        Math.airyaiprime(0).evaluate()  # ≈ -0.2588
    """
    _name = "airyaiprime"
    _latex_name = r"\mathrm{Ai}'"
    _sympy_func = "airyaiprime"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\mathrm{{Ai}}'\\left({self.z.to_latex()}\\right)"


class AiryBiPrime(SpecialFunction):
    """
    Derivative of Airy function Bi'(z).

    Example:
        Math.airybiprime(0).evaluate()  # ≈ 0.4483
    """
    _name = "airybiprime"
    _latex_name = r"\mathrm{Bi}'"
    _sympy_func = "airybiprime"

    def __init__(self, z):
        super().__init__(z)

    @property
    def z(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\mathrm{{Bi}}'\\left({self.z.to_latex()}\\right)"


# =============================================================================
# Orthogonal Polynomials
# =============================================================================

class Legendre(SpecialFunction):
    """
    Legendre polynomial P_n(x).

    Orthogonal polynomials on [-1, 1] with weight function w(x) = 1.
    Used in multipole expansions and solutions to Laplace's equation.

    Example:
        n, x = Math.var("n"), Math.var("x")
        Math.legendre(n, x).render()  # P_n(x)
        Math.legendre(2, 0.5).evaluate()  # P₂(0.5) = (3*0.25 - 1)/2 = -0.125
    """
    _name = "legendre"
    _latex_name = "P"
    _sympy_func = "legendre"

    def __init__(self, n, x):
        super().__init__(n, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def x(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        x_latex = self.x.to_latex()
        return f"P_{{{n_latex}}}\\left({x_latex}\\right)"


class AssocLegendre(SpecialFunction):
    """
    Associated Legendre function P_l^m(x).

    P_l^m(x) = (-1)^m (1-x²)^(m/2) d^m/dx^m P_l(x)

    Used in spherical harmonics: Y_l^m(θ,φ) ∝ P_l^m(cos θ) e^(imφ)

    Example:
        l, m, x = Math.var("l"), Math.var("m"), Math.var("x")
        Math.assoc_legendre(l, m, x).render()  # P_l^m(x)
    """
    _name = "assoc_legendre"
    _latex_name = "P"
    _sympy_func = "assoc_legendre"

    def __init__(self, n, m, x):
        super().__init__(n, m, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def m(self) -> Expr:
        return self.args[1]

    @property
    def x(self) -> Expr:
        return self.args[2]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        m_latex = self.m.to_latex()
        x_latex = self.x.to_latex()
        return f"P_{{{n_latex}}}^{{{m_latex}}}\\left({x_latex}\\right)"


class Hermite(SpecialFunction):
    """
    Hermite polynomial H_n(x) (physicist's convention).

    Orthogonal polynomials with weight function w(x) = e^(-x²).
    Used in quantum harmonic oscillator wavefunctions.

    H_0(x) = 1
    H_1(x) = 2x
    H_2(x) = 4x² - 2

    Example:
        n, x = Math.var("n"), Math.var("x")
        Math.hermite(n, x).render()  # H_n(x)
        Math.hermite(2, 1).evaluate()  # H₂(1) = 4 - 2 = 2
    """
    _name = "hermite"
    _latex_name = "H"
    _sympy_func = "hermite"

    def __init__(self, n, x):
        super().__init__(n, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def x(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        x_latex = self.x.to_latex()
        return f"H_{{{n_latex}}}\\left({x_latex}\\right)"


class HermiteProb(SpecialFunction):
    """
    Probabilist's Hermite polynomial He_n(x).

    Orthogonal with weight function w(x) = e^(-x²/2).
    Related to physicist's: H_n(x) = 2^(n/2) He_n(√2 x)

    He_0(x) = 1
    He_1(x) = x
    He_2(x) = x² - 1

    Example:
        Math.hermite_prob(2, 1).evaluate()  # He₂(1) = 0
    """
    _name = "hermite_prob"
    _latex_name = r"\mathrm{He}"

    def __init__(self, n, x):
        super().__init__(n, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def x(self) -> Expr:
        return self.args[1]

    def to_sympy(self):
        """Convert using: He_n(x) = 2^(-n/2) H_n(x/√2)"""
        sp = _get_sympy()
        n_sympy = self.n.to_sympy()
        x_sympy = self.x.to_sympy()
        # Use SymPy's hermite_prob if available, otherwise convert
        if hasattr(sp, 'hermite_prob'):
            return sp.hermite_prob(n_sympy, x_sympy)
        # Fall back to definition via regular Hermite
        return sp.hermite(n_sympy, x_sympy / sp.sqrt(2)) / (2 ** (n_sympy / 2))

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        x_latex = self.x.to_latex()
        return f"\\mathrm{{He}}_{{{n_latex}}}\\left({x_latex}\\right)"


class Laguerre(SpecialFunction):
    """
    Laguerre polynomial L_n(x).

    Orthogonal on [0, ∞) with weight function w(x) = e^(-x).

    L_0(x) = 1
    L_1(x) = 1 - x
    L_2(x) = (2 - 4x + x²)/2

    Example:
        n, x = Math.var("n"), Math.var("x")
        Math.laguerre(n, x).render()  # L_n(x)
    """
    _name = "laguerre"
    _latex_name = "L"
    _sympy_func = "laguerre"

    def __init__(self, n, x):
        super().__init__(n, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def x(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        x_latex = self.x.to_latex()
        return f"L_{{{n_latex}}}\\left({x_latex}\\right)"


class AssocLaguerre(SpecialFunction):
    """
    Associated (generalized) Laguerre polynomial L_n^(α)(x).

    Orthogonal on [0, ∞) with weight function w(x) = x^α e^(-x).

    Used in hydrogen atom radial wavefunctions:
    R_nl(r) ∝ L_{n-l-1}^(2l+1)(2r/na₀) e^(-r/na₀)

    Example:
        n, alpha, x = Math.var("n"), Math.var("alpha"), Math.var("x")
        Math.assoc_laguerre(n, alpha, x).render()  # L_n^(α)(x)
    """
    _name = "assoc_laguerre"
    _latex_name = "L"
    _sympy_func = "assoc_laguerre"

    def __init__(self, n, alpha, x):
        super().__init__(n, alpha, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def alpha(self) -> Expr:
        return self.args[1]

    @property
    def x(self) -> Expr:
        return self.args[2]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        alpha_latex = self.alpha.to_latex()
        x_latex = self.x.to_latex()
        return f"L_{{{n_latex}}}^{{({alpha_latex})}}\\left({x_latex}\\right)"


class Chebyshev1(SpecialFunction):
    """
    Chebyshev polynomial of the first kind T_n(x).

    Orthogonal on [-1, 1] with weight (1-x²)^(-1/2).
    T_n(cos θ) = cos(nθ)

    Example:
        Math.chebyshevt(3, 0.5).evaluate()  # T₃(0.5) = -1
    """
    _name = "chebyshevt"
    _latex_name = "T"
    _sympy_func = "chebyshevt"

    def __init__(self, n, x):
        super().__init__(n, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def x(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        x_latex = self.x.to_latex()
        return f"T_{{{n_latex}}}\\left({x_latex}\\right)"


class Chebyshev2(SpecialFunction):
    """
    Chebyshev polynomial of the second kind U_n(x).

    Orthogonal on [-1, 1] with weight (1-x²)^(1/2).
    U_n(cos θ) = sin((n+1)θ)/sin(θ)

    Example:
        Math.chebyshevu(2, 0.5).evaluate()  # U₂(0.5) = 0
    """
    _name = "chebyshevu"
    _latex_name = "U"
    _sympy_func = "chebyshevu"

    def __init__(self, n, x):
        super().__init__(n, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def x(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        x_latex = self.x.to_latex()
        return f"U_{{{n_latex}}}\\left({x_latex}\\right)"


class Gegenbauer(SpecialFunction):
    """
    Gegenbauer (ultraspherical) polynomial C_n^(α)(x).

    Generalizes Legendre (α=1/2) and Chebyshev (α→0, α=1) polynomials.

    Example:
        Math.gegenbauer(2, 0.5, 0.5).evaluate()  # C_2^(0.5)(0.5) = P_2(0.5)
    """
    _name = "gegenbauer"
    _latex_name = "C"
    _sympy_func = "gegenbauer"

    def __init__(self, n, alpha, x):
        super().__init__(n, alpha, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def alpha(self) -> Expr:
        return self.args[1]

    @property
    def x(self) -> Expr:
        return self.args[2]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        alpha_latex = self.alpha.to_latex()
        x_latex = self.x.to_latex()
        return f"C_{{{n_latex}}}^{{({alpha_latex})}}\\left({x_latex}\\right)"


class Jacobi(SpecialFunction):
    """
    Jacobi polynomial P_n^(α,β)(x).

    Most general classical orthogonal polynomial family.
    Orthogonal on [-1, 1] with weight (1-x)^α (1+x)^β.

    Example:
        Math.jacobi(2, 1, 2, 0.5).evaluate()
    """
    _name = "jacobi"
    _latex_name = "P"
    _sympy_func = "jacobi"

    def __init__(self, n, alpha, beta, x):
        super().__init__(n, alpha, beta, x)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def alpha(self) -> Expr:
        return self.args[1]

    @property
    def beta(self) -> Expr:
        return self.args[2]

    @property
    def x(self) -> Expr:
        return self.args[3]

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        alpha_latex = self.alpha.to_latex()
        beta_latex = self.beta.to_latex()
        x_latex = self.x.to_latex()
        return f"P_{{{n_latex}}}^{{({alpha_latex},{beta_latex})}}\\left({x_latex}\\right)"


# =============================================================================
# Spherical Harmonics
# =============================================================================

class SphericalHarmonic(SpecialFunction):
    """
    Spherical harmonic Y_l^m(θ, φ).

    Eigenfunctions of the angular momentum operators L² and L_z.
    Y_l^m(θ, φ) = N_l^m P_l^|m|(cos θ) e^(imφ)

    Used in:
    - Hydrogen atom angular wavefunctions
    - Multipole expansions
    - Angular momentum in quantum mechanics

    Example:
        l, m = Math.var("l"), Math.var("m")
        theta, phi = Math.var("theta"), Math.var("phi")
        Math.Ylm(l, m, theta, phi).render()  # Y_l^m(θ, φ)
    """
    _name = "Ylm"
    _latex_name = "Y"
    _sympy_func = "Ynm"

    def __init__(self, l, m, theta, phi):
        super().__init__(l, m, theta, phi)

    @property
    def l(self) -> Expr:
        return self.args[0]

    @property
    def m(self) -> Expr:
        return self.args[1]

    @property
    def theta(self) -> Expr:
        return self.args[2]

    @property
    def phi(self) -> Expr:
        return self.args[3]

    def to_latex(self) -> str:
        l_latex = self.l.to_latex()
        m_latex = self.m.to_latex()
        theta_latex = self.theta.to_latex()
        phi_latex = self.phi.to_latex()
        return f"Y_{{{l_latex}}}^{{{m_latex}}}\\left({theta_latex}, {phi_latex}\\right)"


class RealSphericalHarmonic(SpecialFunction):
    """
    Real spherical harmonic Y_{lm}(θ, φ).

    Real-valued combinations of complex spherical harmonics.
    Used when dealing with real-valued functions and potentials.

    For m > 0:  Y_{l,m} = (Y_l^m + Y_l^{-m}) / √2
    For m < 0:  Y_{l,m} = i(Y_l^m - Y_l^{-m}) / √2
    For m = 0:  Y_{l,0} = Y_l^0
    """
    _name = "Ylm_real"
    _latex_name = "Y"

    def __init__(self, l, m, theta, phi):
        super().__init__(l, m, theta, phi)

    @property
    def l(self) -> Expr:
        return self.args[0]

    @property
    def m(self) -> Expr:
        return self.args[1]

    @property
    def theta(self) -> Expr:
        return self.args[2]

    @property
    def phi(self) -> Expr:
        return self.args[3]

    def to_sympy(self):
        sp = _get_sympy()
        l_sympy = self.l.to_sympy()
        m_sympy = self.m.to_sympy()
        theta_sympy = self.theta.to_sympy()
        phi_sympy = self.phi.to_sympy()
        return sp.Znm(l_sympy, m_sympy, theta_sympy, phi_sympy)

    def to_latex(self) -> str:
        l_latex = self.l.to_latex()
        m_latex = self.m.to_latex()
        theta_latex = self.theta.to_latex()
        phi_latex = self.phi.to_latex()
        return f"Y_{{{l_latex},{m_latex}}}\\left({theta_latex}, {phi_latex}\\right)"


# =============================================================================
# Hypergeometric Functions
# =============================================================================

class Hypergeometric2F1(SpecialFunction):
    """
    Gauss hypergeometric function ₂F₁(a, b; c; z).

    The most important hypergeometric function, appearing in many special functions:
    - Legendre: P_n(x) = ₂F₁(-n, n+1; 1; (1-x)/2)
    - Many others as special cases

    Example:
        a, b, c, z = Math.var("a"), Math.var("b"), Math.var("c"), Math.var("z")
        Math.hyper2f1(a, b, c, z).render()  # ₂F₁(a,b;c;z)
    """
    _name = "hyper2f1"
    _latex_name = r"{}_{2}F_{1}"
    _sympy_func = "hyper"

    def __init__(self, a, b, c, z):
        super().__init__(a, b, c, z)

    @property
    def a(self) -> Expr:
        return self.args[0]

    @property
    def b(self) -> Expr:
        return self.args[1]

    @property
    def c(self) -> Expr:
        return self.args[2]

    @property
    def z(self) -> Expr:
        return self.args[3]

    def to_sympy(self):
        sp = _get_sympy()
        a_sympy = self.a.to_sympy()
        b_sympy = self.b.to_sympy()
        c_sympy = self.c.to_sympy()
        z_sympy = self.z.to_sympy()
        return sp.hyper([a_sympy, b_sympy], [c_sympy], z_sympy)

    def to_latex(self) -> str:
        a_latex = self.a.to_latex()
        b_latex = self.b.to_latex()
        c_latex = self.c.to_latex()
        z_latex = self.z.to_latex()
        return f"{{}}_{2}F_{{1}}\\left({a_latex}, {b_latex}; {c_latex}; {z_latex}\\right)"


class Hypergeometric1F1(SpecialFunction):
    """
    Confluent hypergeometric function ₁F₁(a; b; z).

    Also known as Kummer's function M(a, b, z).
    Appears in many quantum mechanical systems:
    - Hydrogen atom: Laguerre polynomials are special cases
    - Harmonic oscillator: Hermite via ₁F₁

    Example:
        Math.hyper1f1(1, 2, 1).evaluate()  # = (e-1)
    """
    _name = "hyper1f1"
    _latex_name = r"{}_{1}F_{1}"
    _sympy_func = "hyper"

    def __init__(self, a, b, z):
        super().__init__(a, b, z)

    @property
    def a(self) -> Expr:
        return self.args[0]

    @property
    def b(self) -> Expr:
        return self.args[1]

    @property
    def z(self) -> Expr:
        return self.args[2]

    def to_sympy(self):
        sp = _get_sympy()
        a_sympy = self.a.to_sympy()
        b_sympy = self.b.to_sympy()
        z_sympy = self.z.to_sympy()
        return sp.hyper([a_sympy], [b_sympy], z_sympy)

    def to_latex(self) -> str:
        a_latex = self.a.to_latex()
        b_latex = self.b.to_latex()
        z_latex = self.z.to_latex()
        return f"{{}}_{1}F_{{1}}\\left({a_latex}; {b_latex}; {z_latex}\\right)"


class Hypergeometric0F1(SpecialFunction):
    """
    Confluent hypergeometric limit function ₀F₁(; b; z).

    Related to Bessel functions:
    J_ν(z) = (z/2)^ν / Γ(ν+1) · ₀F₁(; ν+1; -z²/4)

    Example:
        Math.hyper0f1(1, -1).evaluate()  # = J_0(2)
    """
    _name = "hyper0f1"
    _latex_name = r"{}_{0}F_{1}"
    _sympy_func = "hyper"

    def __init__(self, b, z):
        super().__init__(b, z)

    @property
    def b(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_sympy(self):
        sp = _get_sympy()
        b_sympy = self.b.to_sympy()
        z_sympy = self.z.to_sympy()
        return sp.hyper([], [b_sympy], z_sympy)

    def to_latex(self) -> str:
        b_latex = self.b.to_latex()
        z_latex = self.z.to_latex()
        return f"{{}}_{0}F_{{1}}\\left(; {b_latex}; {z_latex}\\right)"


class HypergeometricPFQ(SpecialFunction):
    """
    Generalized hypergeometric function ₚFq(a₁,...,aₚ; b₁,...,bq; z).

    The most general hypergeometric function.

    Example:
        # Create ₃F₂(1,2,3; 4,5; z)
        a_list = [Math.const(1), Math.const(2), Math.const(3)]
        b_list = [Math.const(4), Math.const(5)]
        z = Math.var("z")
        Math.hyperpfq(a_list, b_list, z).render()
    """
    _name = "hyperpfq"
    _latex_name = ""
    _sympy_func = "hyper"

    def __init__(self, a_list: List, b_list: List, z):
        # Store separately - not as single args tuple
        super().__init__()
        self._a_list = [_ensure_expr(a) for a in a_list]
        self._b_list = [_ensure_expr(b) for b in b_list]
        self._z = _ensure_expr(z)
        self.args = (*self._a_list, *self._b_list, self._z)

    @property
    def a_list(self) -> List[Expr]:
        return self._a_list

    @property
    def b_list(self) -> List[Expr]:
        return self._b_list

    @property
    def z(self) -> Expr:
        return self._z

    def _get_free_variables(self) -> Set['Var']:
        result = set()
        for a in self._a_list:
            result |= a._get_free_variables()
        for b in self._b_list:
            result |= b._get_free_variables()
        result |= self._z._get_free_variables()
        return result

    def to_sympy(self):
        sp = _get_sympy()
        a_sympy = [a.to_sympy() for a in self._a_list]
        b_sympy = [b.to_sympy() for b in self._b_list]
        z_sympy = self._z.to_sympy()
        return sp.hyper(a_sympy, b_sympy, z_sympy)

    def to_latex(self) -> str:
        p = len(self._a_list)
        q = len(self._b_list)
        a_latex = ", ".join(a.to_latex() for a in self._a_list)
        b_latex = ", ".join(b.to_latex() for b in self._b_list)
        z_latex = self._z.to_latex()
        # Use string concatenation to avoid f-string brace escaping issues
        return "{}_{{{}}}F_{{{}}}\\left({}; {}; {}\\right)".format(p, q, a_latex, b_latex, z_latex)


# =============================================================================
# Elliptic Integrals
# =============================================================================

class EllipticK(SpecialFunction):
    """
    Complete elliptic integral of the first kind K(m).

    K(m) = ∫₀^(π/2) dθ / √(1 - m·sin²θ)

    Note: Uses parameter m = k² convention.

    Example:
        Math.elliptic_k(0.5).evaluate()  # ≈ 1.8541
    """
    _name = "elliptic_k"
    _latex_name = "K"
    _sympy_func = "elliptic_k"

    def __init__(self, m):
        super().__init__(m)

    @property
    def m(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"K\\left({self.m.to_latex()}\\right)"


class EllipticE(SpecialFunction):
    """
    Complete elliptic integral of the second kind E(m).

    E(m) = ∫₀^(π/2) √(1 - m·sin²θ) dθ

    Note: Uses parameter m = k² convention.

    Example:
        Math.elliptic_e(0.5).evaluate()  # ≈ 1.3506
    """
    _name = "elliptic_e"
    _latex_name = "E"
    _sympy_func = "elliptic_e"

    def __init__(self, m):
        super().__init__(m)

    @property
    def m(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"E\\left({self.m.to_latex()}\\right)"


class EllipticPi(SpecialFunction):
    """
    Complete elliptic integral of the third kind Π(n, m).

    Π(n, m) = ∫₀^(π/2) dθ / ((1 - n·sin²θ)√(1 - m·sin²θ))

    Example:
        Math.elliptic_pi(0.5, 0.5).evaluate()
    """
    _name = "elliptic_pi"
    _latex_name = r"\Pi"
    _sympy_func = "elliptic_pi"

    def __init__(self, n, m):
        super().__init__(n, m)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def m(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        return f"\\Pi\\left({self.n.to_latex()}, {self.m.to_latex()}\\right)"


# =============================================================================
# Other Important Functions
# =============================================================================

class Zeta(SpecialFunction):
    """
    Riemann zeta function ζ(s).

    ζ(s) = Σ_{n=1}^∞ 1/n^s

    Example:
        Math.zeta(2).evaluate()  # = π²/6 ≈ 1.6449
    """
    _name = "zeta"
    _latex_name = r"\zeta"
    _sympy_func = "zeta"

    def __init__(self, s):
        super().__init__(s)

    @property
    def s(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\zeta\\left({self.s.to_latex()}\\right)"


class PolyLog(SpecialFunction):
    """
    Polylogarithm Li_s(z).

    Li_s(z) = Σ_{k=1}^∞ z^k / k^s

    Example:
        Math.polylog(2, 0.5).evaluate()  # Li₂(0.5) ≈ 0.5822
    """
    _name = "polylog"
    _latex_name = r"\mathrm{Li}"
    _sympy_func = "polylog"

    def __init__(self, s, z):
        super().__init__(s, z)

    @property
    def s(self) -> Expr:
        return self.args[0]

    @property
    def z(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        s_latex = self.s.to_latex()
        z_latex = self.z.to_latex()
        return f"\\mathrm{{Li}}_{{{s_latex}}}\\left({z_latex}\\right)"


class DiracDelta(SpecialFunction):
    """
    Dirac delta function δ(x).

    The distributional identity:
    ∫ f(x)δ(x-a)dx = f(a)

    Example:
        x = Math.var("x")
        Math.dirac(x).render()  # δ(x)
    """
    _name = "dirac"
    _latex_name = r"\delta"
    _sympy_func = "DiracDelta"

    def __init__(self, x):
        super().__init__(x)

    @property
    def x(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\delta\\left({self.x.to_latex()}\\right)"


class Heaviside(SpecialFunction):
    """
    Heaviside step function θ(x) or H(x).

    H(x) = 0 for x < 0
    H(x) = 1 for x > 0

    Example:
        Math.heaviside(x).render()  # θ(x)
    """
    _name = "heaviside"
    _latex_name = r"\theta"
    _sympy_func = "Heaviside"

    def __init__(self, x):
        super().__init__(x)

    @property
    def x(self) -> Expr:
        return self.args[0]

    def to_latex(self) -> str:
        return f"\\theta\\left({self.x.to_latex()}\\right)"


class KroneckerDelta(SpecialFunction):
    """
    Kronecker delta δ_{ij}.

    δ_{ij} = 1 if i = j
    δ_{ij} = 0 if i ≠ j

    Example:
        i, j = Math.var("i"), Math.var("j")
        Math.kronecker(i, j).render()  # δ_{i,j}
    """
    _name = "kronecker"
    _latex_name = r"\delta"
    _sympy_func = "KroneckerDelta"

    def __init__(self, i, j):
        super().__init__(i, j)

    @property
    def i(self) -> Expr:
        return self.args[0]

    @property
    def j(self) -> Expr:
        return self.args[1]

    def to_latex(self) -> str:
        i_latex = self.i.to_latex()
        j_latex = self.j.to_latex()
        return f"\\delta_{{{i_latex},{j_latex}}}"


class LeviCivita(SpecialFunction):
    """
    Levi-Civita symbol ε_{ijk}.

    Totally antisymmetric tensor:
    ε_{123} = ε_{231} = ε_{312} = 1
    ε_{321} = ε_{213} = ε_{132} = -1
    All others = 0

    Example:
        i, j, k = Math.var("i"), Math.var("j"), Math.var("k")
        Math.levi_civita(i, j, k).render()  # ε_{i,j,k}
    """
    _name = "levi_civita"
    _latex_name = r"\varepsilon"
    _sympy_func = "LeviCivita"

    def __init__(self, *indices):
        super().__init__(*indices)

    @property
    def indices(self) -> Tuple[Expr, ...]:
        return self.args

    def to_latex(self) -> str:
        indices_latex = ",".join(idx.to_latex() for idx in self.indices)
        return f"\\varepsilon_{{{indices_latex}}}"


# =============================================================================
# LEGACY SUPPORT - Original Classes Below
# =============================================================================

class Symmetry(Enum):
    """Symmetry types for spin-orbital basis functions."""
    NONE = "none"           # No symmetry (keep all terms)
    SPIN = "spin"           # Spin symmetry (α/β orthogonal)
    SPATIAL = "spatial"     # Spatial symmetry (different spatial orbitals orthogonal)
    ORTHONORMAL = "orthonormal"  # Full orthonormality (⟨φᵢ|φⱼ⟩ = δᵢⱼ)


@dataclass
class SpinOrbital:
    """
    Represents a spin-orbital: a spatial orbital combined with spin.

    Attributes:
        label: Orbital label (e.g., "1s", "2p", "φ₁")
        spin: Spin state ("α", "β", "↑", "↓", or None for spinless)
        spatial_quantum_numbers: Optional dict of quantum numbers (n, l, m, etc.)
        symmetry_group: Optional symmetry group label for point group symmetry
    """
    label: str
    spin: Optional[str] = None
    spatial_quantum_numbers: Optional[Dict[str, int]] = None
    symmetry_group: Optional[str] = None

    def __post_init__(self):
        # Normalize spin notation
        if self.spin in ("↑", "up", "alpha"):
            self.spin = "α"
        elif self.spin in ("↓", "down", "beta"):
            self.spin = "β"

    @property
    def full_label(self) -> str:
        """Return full label including spin if present."""
        if self.spin:
            spin_symbol = "↑" if self.spin == "α" else "↓"
            return f"{self.label}{spin_symbol}"
        return self.label

    @property
    def latex_label(self) -> str:
        """Return LaTeX-formatted label."""
        if self.spin:
            spin_symbol = r"\uparrow" if self.spin == "α" else r"\downarrow"
            return f"{self.label}_{{{spin_symbol}}}"
        return self.label

    def is_orthogonal_to(self, other: "SpinOrbital", symmetry: Symmetry) -> bool:
        """
        Check if this orbital is orthogonal to another under given symmetry.

        Args:
            other: Another SpinOrbital
            symmetry: Symmetry type to apply

        Returns:
            True if orbitals are orthogonal, False otherwise
        """
        if symmetry == Symmetry.NONE:
            return False

        if symmetry == Symmetry.SPIN:
            # Different spins are orthogonal
            if self.spin and other.spin and self.spin != other.spin:
                return True
            return False

        if symmetry == Symmetry.SPATIAL:
            # Different spatial orbitals are orthogonal
            if self.label != other.label:
                return True
            return False

        if symmetry == Symmetry.ORTHONORMAL:
            # Full orthonormality: both spin and spatial must match
            return self.label != other.label or self.spin != other.spin

        return False

    def __eq__(self, other):
        if isinstance(other, SpinOrbital):
            return self.label == other.label and self.spin == other.spin
        return False

    def __hash__(self):
        return hash((self.label, self.spin))

    def __str__(self):
        return self.full_label


@dataclass
class SlaterState:
    """
    Represents a Slater determinant state as a list of occupied spin-orbitals.

    This is the primary interface for working with Slater determinants.
    Instead of specifying a full matrix, you provide a 1D list of occupied
    orbitals, and the class handles the determinant structure automatically.

    The electron coordinates are implicitly coupled with the spin-orbitals:
    electron 1 is associated with the first orbital, electron 2 with the second, etc.

    Attributes:
        orbitals: List of occupied spin-orbitals
        symmetry: Symmetry type for orthogonality rules
        basis_name: Optional name for the basis set

    Example:
        # Create a 3-electron state
        state = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])

        # Or with explicit SpinOrbital objects
        orbs = [SpinOrbital("1s", "α"), SpinOrbital("1s", "β"), SpinOrbital("2s", "α")]
        state = SlaterState(orbs, symmetry=Symmetry.ORTHONORMAL)

        # Use with Math class
        m = Math()
        m.slater_bra(state)  # ⟨1s↑, 1s↓, 2s↑|
    """
    orbitals: List[SpinOrbital]
    symmetry: Symmetry = Symmetry.ORTHONORMAL
    basis_name: Optional[str] = None

    @classmethod
    def from_labels(cls, labels: List[str], symmetry: Symmetry = Symmetry.ORTHONORMAL,
                    basis_name: Optional[str] = None) -> "SlaterState":
        """
        Create a SlaterState from string labels.

        Parses spin notation from labels:
        - "1s↑" or "1sα" or "1s_up" -> SpinOrbital("1s", "α")
        - "2p↓" or "2pβ" or "2p_down" -> SpinOrbital("2p", "β")
        - "φ1" -> SpinOrbital("φ1", None) (no spin)

        Args:
            labels: List of orbital labels with optional spin notation
            symmetry: Symmetry type for orthogonality rules
            basis_name: Optional name for the basis set
        """
        orbitals = []
        for label in labels:
            # Parse spin from label
            spin = None
            base_label = label

            for spin_marker, spin_val in [
                ("↑", "α"), ("↓", "β"),
                ("α", "α"), ("β", "β"),
                ("_up", "α"), ("_down", "β"),
                ("_alpha", "α"), ("_beta", "β"),
            ]:
                if spin_marker in label:
                    spin = spin_val
                    base_label = label.replace(spin_marker, "")
                    break

            orbitals.append(SpinOrbital(base_label, spin))

        return cls(orbitals, symmetry, basis_name)

    @classmethod
    def from_occupation(cls, occupation: Dict[str, int],
                        symmetry: Symmetry = Symmetry.ORTHONORMAL) -> "SlaterState":
        """
        Create a SlaterState from occupation numbers.

        Args:
            occupation: Dict mapping orbital labels to occupation (0, 1, or 2)
            symmetry: Symmetry type

        Example:
            state = SlaterState.from_occupation({"1s": 2, "2s": 1})
            # Creates: |1s↑, 1s↓, 2s↑⟩
        """
        orbitals = []
        for label, n in occupation.items():
            if n >= 1:
                orbitals.append(SpinOrbital(label, "α"))
            if n >= 2:
                orbitals.append(SpinOrbital(label, "β"))
        return cls(orbitals, symmetry)

    @property
    def n_electrons(self) -> int:
        """Number of electrons in this state."""
        return len(self.orbitals)

    @property
    def labels(self) -> List[str]:
        """List of full orbital labels."""
        return [orb.full_label for orb in self.orbitals]

    @property
    def latex_labels(self) -> List[str]:
        """List of LaTeX-formatted orbital labels."""
        return [orb.latex_label for orb in self.orbitals]

    def overlap_with(self, other: "SlaterState") -> Tuple[bool, Optional[int]]:
        """
        Compute overlap with another Slater state under current symmetry.

        For orthonormal orbitals:
        - ⟨Ψ|Φ⟩ = 0 if different occupied orbitals
        - ⟨Ψ|Φ⟩ = (-1)^P if same orbitals (P = permutation parity)

        Returns:
            (is_nonzero, sign) where sign is +1, -1, or None if zero
        """
        if self.n_electrons != other.n_electrons:
            return (False, None)

        if self.symmetry == Symmetry.NONE:
            return (True, None)  # Can't determine without explicit computation

        # Check if same set of orbitals (for orthonormal case)
        if self.symmetry == Symmetry.ORTHONORMAL:
            self_set = set(self.orbitals)
            other_set = set(other.orbitals)

            if self_set != other_set:
                return (False, None)

            # Compute permutation parity
            # Find the permutation that maps other's order to self's order
            parity = self._compute_permutation_parity(other)
            return (True, parity)

        return (True, None)

    def _compute_permutation_parity(self, other: "SlaterState") -> int:
        """Compute the parity of permutation between two states with same orbitals."""
        # Build index map
        other_indices = {orb: i for i, orb in enumerate(other.orbitals)}
        perm = [other_indices[orb] for orb in self.orbitals]

        # Count inversions
        inversions = 0
        n = len(perm)
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inversions += 1

        return 1 if inversions % 2 == 0 else -1

    def to_matrix(self) -> List[List[str]]:
        """
        Convert to symbolic matrix representation for determinant expansion.

        Returns n×n matrix where element [i][j] represents orbital j evaluated
        at electron i's coordinates: φⱼ(rᵢ)
        """
        n = self.n_electrons
        matrix = []
        for i in range(n):
            row = []
            for j, orb in enumerate(self.orbitals):
                # Element represents φⱼ(rᵢ) - orbital j at electron i's position
                row.append(orb.latex_label)
            matrix.append(row)
        return matrix

    def __str__(self):
        return f"|{', '.join(self.labels)}⟩"

    def __repr__(self):
        return f"SlaterState({self.labels}, symmetry={self.symmetry.value})"

# Current notation style
_notation_style = "standard"

# Current line height (default: normal)
_line_height = "normal"


def set_line_height(height: str):
    """
    Set the line height for math rendering.

    Args:
        height: CSS line-height value (e.g., "1", "1.5", "2", "normal", "1.2em")

    Example:
        set_line_height("1.5")
        set_line_height("2")
        set_line_height("normal")
    """
    global _line_height
    _line_height = height


def get_line_height() -> str:
    """Get the current line height setting."""
    return _line_height

# Notation-specific LaTeX preambles/macros
_NOTATION_MACROS = {
    "standard": "",
    "physicist": r"""
        \newcommand{\vect}[1]{\vec{#1}}
        \newcommand{\grad}{\nabla}
        \newcommand{\curl}{\nabla \times}
        \newcommand{\divg}{\nabla \cdot}
        \newcommand{\lapl}{\nabla^2}
        \newcommand{\ddt}{\frac{d}{dt}}
        \newcommand{\ddx}{\frac{d}{dx}}
        \newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
    """,
    "chemist": r"""
        \newcommand{\ce}[1]{\mathrm{#1}}
        \newcommand{\rightmark}{\rightarrow}
        \newcommand{\leftmark}{\leftarrow}
        \newcommand{\eqmark}{\rightleftharpoons}
        \newcommand{\yields}{\rightarrow}
        \newcommand{\equilibrium}{\rightleftharpoons}
        \newcommand{\gas}{\uparrow}
        \newcommand{\precipitate}{\downarrow}
        \newcommand{\aq}{_{(aq)}}
        \newcommand{\solid}{_{(s)}}
        \newcommand{\liquid}{_{(l)}}
        \newcommand{\gasphase}{_{(g)}}
    """,
    "braket": r"""
        \newcommand{\bra}[1]{\langle #1 |}
        \newcommand{\ket}[1]{| #1 \rangle}
        \newcommand{\braket}[2]{\langle #1 | #2 \rangle}
        \newcommand{\expval}[1]{\langle #1 \rangle}
        \newcommand{\matelem}[3]{\langle #1 | #2 | #3 \rangle}
        \newcommand{\op}[1]{\hat{#1}}
        \newcommand{\comm}[2]{[#1, #2]}
        \newcommand{\anticomm}[2]{\{#1, #2\}}
        \newcommand{\dagger}{\dagger}
    """,
    "engineering": r"""
        \newcommand{\j}{\mathrm{j}}
        \newcommand{\ohm}{\Omega}
        \newcommand{\simark}[1]{\,\mathrm{#1}}
        \newcommand{\phasor}[1]{\tilde{#1}}
        \newcommand{\magnitude}[1]{|#1|}
        \newcommand{\phase}[1]{\angle #1}
        \newcommand{\conj}[1]{#1^*}
        \newcommand{\re}{\mathrm{Re}}
        \newcommand{\im}{\mathrm{Im}}
    """
}


def set_notation(style: str):
    """
    Set the notation style for math rendering.

    Args:
        style: One of 'standard', 'physicist', 'chemist', 'braket', 'engineering'

    Example:
        set_notation("physicist")
        set_notation("braket")
    """
    global _notation_style
    if style not in _NOTATION_MACROS:
        raise ValueError(f"Unknown notation style: {style}. "
                        f"Available: {list(_NOTATION_MACROS.keys())}")
    _notation_style = style


def get_notation() -> str:
    """Get the current notation style."""
    return _notation_style


def _get_macros() -> str:
    """Get the LaTeX macros for the current notation style."""
    return _NOTATION_MACROS.get(_notation_style, "")


def _get_line_height_style() -> str:
    """Get inline style for line height if not default."""
    if _line_height != "normal":
        return f' style="line-height: {_line_height};"'
    return ""


def latex(expression: str, display: bool = True, label: Optional[str] = None,
          justify: str = "center"):
    """
    Render a LaTeX math expression.

    Args:
        expression: LaTeX math expression (without delimiters)
        display: If True, render as display math (centered, block). If False, inline.
        label: Optional label/caption for the expression
        justify: Alignment - 'left', 'center', or 'right'

    Example:
        latex(r"E = mc^2")
        latex(r"\\int_0^\\infty e^{-x} dx = 1", display=True)
        latex(r"x^2", display=False)  # Inline math
        latex(r"F = ma", justify="left")
    """
    if display:
        delim_start, delim_end = r"\[", r"\]"
    else:
        delim_start, delim_end = r"\(", r"\)"

    justify_class = f"cm-math-{justify}" if justify in ("left", "center", "right") else "cm-math-center"
    line_height_style = _get_line_height_style()
    html_content = f'<div class="cm-math {justify_class}"{line_height_style}>{delim_start}{expression}{delim_end}</div>'

    if label:
        html_content = f'<div class="cm-math-labeled"{line_height_style}><span class="cm-math-label">{label}</span>{html_content}</div>'

    views.html(html_content)


def equation(expression: str, number: Optional[Union[int, str]] = None):
    """
    Render a numbered equation.

    Args:
        expression: LaTeX math expression
        number: Optional equation number or label

    Example:
        equation(r"F = ma", number=1)
        equation(r"E = mc^2", number="2.1")
    """
    line_height_style = _get_line_height_style()
    if number is not None:
        html_content = f'''
        <div class="cm-equation"{line_height_style}>
            <span class="cm-equation-content">\\[{expression}\\]</span>
            <span class="cm-equation-number">({number})</span>
        </div>
        '''
    else:
        html_content = f'<div class="cm-equation"{line_height_style}>\\[{expression}\\]</div>'

    views.html(html_content)


def _render_scrollable(expression: str, display: bool = True, label: Optional[str] = None,
                       justify: str = "center"):
    """
    Render a LaTeX expression in a scrollable container for long equations.

    This wraps the math output in a div with horizontal scrolling enabled,
    useful for very long equations that would otherwise overflow.
    """
    if display:
        delim_start, delim_end = r"\[", r"\]"
    else:
        delim_start, delim_end = r"\(", r"\)"

    justify_class = f"cm-math-{justify}" if justify in ("left", "center", "right") else "cm-math-center"
    line_height_style = _get_line_height_style()

    # Wrap in scrollable container
    html_content = f'<div class="cm-math-scroll"><div class="cm-math {justify_class}"{line_height_style}>{delim_start}{expression}{delim_end}</div></div>'

    if label:
        html_content = f'<div class="cm-math-labeled"{line_height_style}><span class="cm-math-label">{label}</span>{html_content}</div>'

    views.html(html_content)


def align(*equations: str):
    """
    Render aligned equations (useful for multi-step derivations).

    Args:
        *equations: LaTeX expressions with & for alignment points

    Example:
        align(
            r"x &= a + b",
            r"&= c + d",
            r"&= e"
        )
    """
    aligned = r" \\ ".join(equations)
    line_height_style = _get_line_height_style()
    html_content = f'<div class="cm-math"{line_height_style}>\\[\\begin{{aligned}}{aligned}\\end{{aligned}}\\]</div>'
    views.html(html_content)


def matrix(data: List[List], style: str = "pmatrix"):
    """
    Render a matrix.

    Args:
        data: 2D list of matrix elements
        style: Matrix style - 'pmatrix' (parentheses), 'bmatrix' (brackets),
               'vmatrix' (vertical bars), 'Vmatrix' (double bars), 'matrix' (none)

    Example:
        matrix([[1, 2], [3, 4]])
        matrix([[1, 0], [0, 1]], style="bmatrix")
    """
    rows = []
    for row in data:
        row_str = " & ".join(str(x) for x in row)
        rows.append(row_str)

    matrix_content = r" \\ ".join(rows)
    line_height_style = _get_line_height_style()
    html_content = f'<div class="cm-math"{line_height_style}>\\[\\begin{{{style}}}{matrix_content}\\end{{{style}}}\\]</div>'
    views.html(html_content)


def bullets(*expressions: str, display: bool = True):
    """
    Render a bulleted list of LaTeX expressions.

    Args:
        *expressions: LaTeX expressions for each bullet point
        display: If True, use display math. If False, inline math.

    Example:
        bullets(
            r"x^2 + y^2 = r^2",
            r"e^{i\\pi} + 1 = 0",
            r"\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}"
        )
    """
    delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
    list_items = "\n".join(f'<li>{delim_start}{expr}{delim_end}</li>' for expr in expressions)
    line_height_style = _get_line_height_style()
    html_content = f'<ul class="cm-math-list bulleted"{line_height_style}>{list_items}</ul>'
    views.html(html_content)


def numbered(*expressions: str, start: int = 1, display: bool = True):
    """
    Render a numbered list of LaTeX expressions.

    Args:
        *expressions: LaTeX expressions for each numbered item
        start: Starting number (default: 1)
        display: If True, use display math. If False, inline math.

    Example:
        numbered(
            r"F = ma",
            r"E = mc^2",
            r"p = mv"
        )
    """
    delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
    list_items = "\n".join(f'<li>{delim_start}{expr}{delim_end}</li>' for expr in expressions)
    line_height_style = _get_line_height_style()
    html_content = f'<ol class="cm-math-list numbered" start="{start}"{line_height_style}>{list_items}</ol>'
    views.html(html_content)


def items(*expressions: str, display: bool = True):
    """
    Render a plain list of LaTeX expressions (no bullets or numbers).

    Args:
        *expressions: LaTeX expressions for each item
        display: If True, use display math. If False, inline math.

    Example:
        items(
            r"\\text{First equation: } x = 1",
            r"\\text{Second equation: } y = 2"
        )
    """
    delim_start, delim_end = (r"\[", r"\]") if display else (r"\(", r"\)")
    item_list = "\n".join(f'<li>{delim_start}{expr}{delim_end}</li>' for expr in expressions)
    line_height_style = _get_line_height_style()
    html_content = f'<ul class="cm-math-list none"{line_height_style}>{item_list}</ul>'
    views.html(html_content)


class MathBuilder:
    """
    A builder class for constructing LaTeX expressions programmatically.

    NOTE: This is the legacy API. For new code, use the Math factory:
        from cm.symbols import Math
        x = Math.var("x")
        expr = x**2 + 1
        expr.render()

    Legacy Example:
        m = MathBuilder()
        m.frac("a", "b").plus().sqrt("c").equals().text("result")
        m.render()

        # Bra-ket notation (with braket style)
        m = MathBuilder()
        m.bra("psi").ket("phi")
        m.render()

        # Operator overloads for inner products
        m_g = MathBuilder()
        m_g.determinant_bra(sm_g).equals().bra('\\phi_1')

        m_1 = MathBuilder()
        m_1.determinant_ket(sm_1).equals().ket('\\phi_2')

        m_s = m_g @ m_1  # Inner product of both sides
        m_s.render()
    """

    def __init__(self):
        self._parts: List[str] = []
        # Track equation structure for operator overloads
        self._lhs_parts: List[str] = []  # Left-hand side of equation
        self._rhs_parts: List[str] = []  # Right-hand side of equation
        self._has_equals: bool = False
        # Track determinant matrices for inner product computation
        self._lhs_det_matrix = None  # Determinant matrix on LHS
        self._rhs_det_matrix = None  # Determinant matrix on RHS
        self._lhs_det_type: str = None  # 'bra' or 'ket'
        self._rhs_det_type: str = None  # 'bra' or 'ket'
        # Track simple bra/ket labels
        self._lhs_bra_label = None
        self._rhs_bra_label = None
        self._lhs_ket_label = None
        self._rhs_ket_label = None
        # Track if this is an operator (for matrix element notation)
        self._is_operator: bool = False
        self._lhs_operator_str: str = None  # Operator expression for LHS
        self._rhs_operator_str: str = None  # Operator expression for RHS
        # Track pending operator for chained @ operations (bra @ op @ ket)
        self._pending_operator: "Math" = None
        # Track SlaterState objects for inner products
        self._lhs_slater_state: "SlaterState" = None
        self._rhs_slater_state: "SlaterState" = None
        self._lhs_slater_type: str = None  # 'bra' or 'ket'
        self._rhs_slater_type: str = None  # 'bra' or 'ket'

    def _append(self, content: str) -> "MathBuilder":
        self._parts.append(content)
        # Track LHS vs RHS for operator overloads
        if self._has_equals:
            self._rhs_parts.append(content)
        else:
            self._lhs_parts.append(content)
        return self

    def raw(self, latex: str) -> "MathBuilder":
        """Add raw LaTeX content."""
        return self._append(latex)

    def text(self, content: str) -> "MathBuilder":
        """Add text (non-italic) content."""
        return self._append(f"\\text{{{content}}}")

    def var(self, name: str) -> "MathBuilder":
        """Add a variable."""
        return self._append(name)

    # Basic operations
    def plus(self) -> "MathBuilder":
        return self._append(" + ")

    def minus(self) -> "MathBuilder":
        return self._append(" - ")

    def times(self) -> "MathBuilder":
        return self._append(" \\times ")

    def cdot(self) -> "MathBuilder":
        return self._append(" \\cdot ")

    def div(self) -> "MathBuilder":
        return self._append(" \\div ")

    def equals(self) -> "MathBuilder":
        self._has_equals = True
        return self._append(" = ")

    def approx(self) -> "MathBuilder":
        return self._append(" \\approx ")

    def neq(self) -> "MathBuilder":
        return self._append(" \\neq ")

    def lt(self) -> "MathBuilder":
        return self._append(" < ")

    def gt(self) -> "MathBuilder":
        return self._append(" > ")

    def leq(self) -> "MathBuilder":
        return self._append(" \\leq ")

    def geq(self) -> "MathBuilder":
        return self._append(" \\geq ")

    # Fractions and roots
    def frac(self, num: str, denom: str) -> "MathBuilder":
        """Add a fraction."""
        return self._append(f"\\frac{{{num}}}{{{denom}}}")

    def sqrt(self, content: str, n: Optional[str] = None) -> "MathBuilder":
        """Add a square root or nth root."""
        if n:
            return self._append(f"\\sqrt[{n}]{{{content}}}")
        return self._append(f"\\sqrt{{{content}}}")

    # Subscripts and superscripts
    def sub(self, content: str) -> "MathBuilder":
        """Add a subscript."""
        return self._append(f"_{{{content}}}")

    def sup(self, content: str) -> "MathBuilder":
        """Add a superscript."""
        return self._append(f"^{{{content}}}")

    def subsup(self, sub: str, sup: str) -> "MathBuilder":
        """Add both subscript and superscript."""
        return self._append(f"_{{{sub}}}^{{{sup}}}")

    # Greek letters
    def alpha(self) -> "MathBuilder": return self._append("\\alpha")
    def beta(self) -> "MathBuilder": return self._append("\\beta")
    def gamma(self) -> "MathBuilder": return self._append("\\gamma")
    def delta(self) -> "MathBuilder": return self._append("\\delta")
    def epsilon(self) -> "MathBuilder": return self._append("\\epsilon")
    def zeta(self) -> "MathBuilder": return self._append("\\zeta")
    def eta(self) -> "MathBuilder": return self._append("\\eta")
    def theta(self) -> "MathBuilder": return self._append("\\theta")
    def iota(self) -> "MathBuilder": return self._append("\\iota")
    def kappa(self) -> "MathBuilder": return self._append("\\kappa")
    def lambda_(self) -> "MathBuilder": return self._append("\\lambda")
    def mu(self) -> "MathBuilder": return self._append("\\mu")
    def nu(self) -> "MathBuilder": return self._append("\\nu")
    def xi(self) -> "MathBuilder": return self._append("\\xi")
    def pi(self) -> "MathBuilder": return self._append("\\pi")
    def rho(self) -> "MathBuilder": return self._append("\\rho")
    def sigma(self) -> "MathBuilder": return self._append("\\sigma")
    def tau(self) -> "MathBuilder": return self._append("\\tau")
    def upsilon(self) -> "MathBuilder": return self._append("\\upsilon")
    def phi(self) -> "MathBuilder": return self._append("\\phi")
    def chi(self) -> "MathBuilder": return self._append("\\chi")
    def psi(self) -> "MathBuilder": return self._append("\\psi")
    def omega(self) -> "MathBuilder": return self._append("\\omega")

    # Capital Greek
    def Gamma(self) -> "MathBuilder": return self._append("\\Gamma")
    def Delta(self) -> "MathBuilder": return self._append("\\Delta")
    def Theta(self) -> "MathBuilder": return self._append("\\Theta")
    def Lambda(self) -> "MathBuilder": return self._append("\\Lambda")
    def Xi(self) -> "MathBuilder": return self._append("\\Xi")
    def Pi(self) -> "MathBuilder": return self._append("\\Pi")
    def Sigma(self) -> "MathBuilder": return self._append("\\Sigma")
    def Phi(self) -> "MathBuilder": return self._append("\\Phi")
    def Psi(self) -> "MathBuilder": return self._append("\\Psi")
    def Omega(self) -> "MathBuilder": return self._append("\\Omega")

    # Calculus
    def integral(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "MathBuilder":
        """Add an integral sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\int_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\int_{{{lower}}}")
        return self._append("\\int")

    def sum(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "MathBuilder":
        """Add a summation sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\sum_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\sum_{{{lower}}}")
        return self._append("\\sum")

    def prod(self, lower: Optional[str] = None, upper: Optional[str] = None) -> "MathBuilder":
        """Add a product sign with optional limits."""
        if lower is not None and upper is not None:
            return self._append(f"\\prod_{{{lower}}}^{{{upper}}}")
        elif lower is not None:
            return self._append(f"\\prod_{{{lower}}}")
        return self._append("\\prod")

    def lim(self, var: str, to: str) -> "MathBuilder":
        """Add a limit."""
        return self._append(f"\\lim_{{{var} \\to {to}}}")

    def deriv(self, func: str = "", var: str = "x") -> "MathBuilder":
        """Add a derivative."""
        if func:
            return self._append(f"\\frac{{d{func}}}{{d{var}}}")
        return self._append(f"\\frac{{d}}{{d{var}}}")

    def partial(self, func: str = "", var: str = "x") -> "MathBuilder":
        """Add a partial derivative."""
        if func:
            return self._append(f"\\frac{{\\partial {func}}}{{\\partial {var}}}")
        return self._append(f"\\frac{{\\partial}}{{\\partial {var}}}")

    def nabla(self) -> "MathBuilder":
        return self._append("\\nabla")

    # Brackets and grouping
    def paren(self, content: str) -> "MathBuilder":
        """Add parentheses."""
        return self._append(f"\\left({content}\\right)")

    def bracket(self, content: str) -> "MathBuilder":
        """Add square brackets."""
        return self._append(f"\\left[{content}\\right]")

    def brace(self, content: str) -> "MathBuilder":
        """Add curly braces."""
        return self._append(f"\\left\\{{{content}\\right\\}}")

    def abs(self, content: str) -> "MathBuilder":
        """Add absolute value bars."""
        return self._append(f"\\left|{content}\\right|")

    def norm(self, content: str) -> "MathBuilder":
        """Add norm double bars."""
        return self._append(f"\\left\\|{content}\\right\\|")

    # Quantum mechanics / Bra-ket
    def bra(self, content) -> "MathBuilder":
        """Add a bra <content|. Content can be a string or list of quantum numbers."""
        original_content = content
        if isinstance(content, (list, tuple)):
            content = ", ".join(str(c) for c in content)
        # Track for operator overloads
        if self._has_equals:
            self._rhs_bra_label = original_content
        else:
            self._lhs_bra_label = original_content
        return self._append(f"\\langle {content} |")

    def ket(self, content) -> "MathBuilder":
        """Add a ket |content>. Content can be a string or list of quantum numbers."""
        original_content = content
        if isinstance(content, (list, tuple)):
            content = ", ".join(str(c) for c in content)
        # Track for operator overloads
        if self._has_equals:
            self._rhs_ket_label = original_content
        else:
            self._lhs_ket_label = original_content
        return self._append(f"| {content} \\rangle")

    def braket(self, bra, ket) -> "MathBuilder":
        """Add a braket <bra|ket>. Arguments can be strings or lists of quantum numbers."""
        if isinstance(bra, (list, tuple)):
            bra = ", ".join(str(c) for c in bra)
        if isinstance(ket, (list, tuple)):
            ket = ", ".join(str(c) for c in ket)
        return self._append(f"\\langle {bra} | {ket} \\rangle")

    def expval(self, operator) -> "MathBuilder":
        """Add an expectation value <operator>."""
        if isinstance(operator, (list, tuple)):
            operator = ", ".join(str(c) for c in operator)
        return self._append(f"\\langle {operator} \\rangle")

    def matelem(self, bra, op, ket) -> "MathBuilder":
        """Add a matrix element <bra|op|ket>."""
        if isinstance(bra, (list, tuple)):
            bra = ", ".join(str(c) for c in bra)
        if isinstance(ket, (list, tuple)):
            ket = ", ".join(str(c) for c in ket)
        return self._append(f"\\langle {bra} | {op} | {ket} \\rangle")

    def op(self, name: str) -> "MathBuilder":
        """Add an operator with hat."""
        return self._append(f"\\hat{{{name}}}")

    def dagger(self) -> "MathBuilder":
        """Add a dagger superscript."""
        return self._append("^\\dagger")

    def comm(self, a: str, b: str) -> "MathBuilder":
        """Add a commutator [a, b]."""
        return self._append(f"[{a}, {b}]")

    # Physics
    def vec(self, content: str) -> "MathBuilder":
        """Add a vector with arrow."""
        return self._append(f"\\vec{{{content}}}")

    def hbar(self) -> "MathBuilder":
        return self._append("\\hbar")

    def infty(self) -> "MathBuilder":
        return self._append("\\infty")

    # Chemistry
    def ce(self, formula: str) -> "MathBuilder":
        """Add a chemical formula (upright text)."""
        return self._append(f"\\mathrm{{{formula}}}")

    def yields(self) -> "MathBuilder":
        """Add a reaction arrow."""
        return self._append(" \\rightarrow ")

    def equilibrium(self) -> "MathBuilder":
        """Add an equilibrium arrow."""
        return self._append(" \\rightleftharpoons ")

    # Special functions
    def sin(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\sin{{{arg}}}" if arg else "\\sin")

    def cos(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\cos{{{arg}}}" if arg else "\\cos")

    def tan(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\tan{{{arg}}}" if arg else "\\tan")

    def ln(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\ln{{{arg}}}" if arg else "\\ln")

    def log(self, arg: str = "", base: Optional[str] = None) -> "MathBuilder":
        if base:
            return self._append(f"\\log_{{{base}}}{{{arg}}}" if arg else f"\\log_{{{base}}}")
        return self._append(f"\\log{{{arg}}}" if arg else "\\log")

    def exp(self, arg: str = "") -> "MathBuilder":
        return self._append(f"\\exp{{{arg}}}" if arg else "\\exp")

    # Symbolic Determinant Expansion
    @staticmethod
    def _symbolic_determinant(m) -> List[tuple]:
        """
        Compute symbolic determinant expansion as list of (sign, [elements]) tuples.
        Uses cofactor expansion along the first row.
        """
        import numpy as np
        m = np.asarray(m)

        if m.shape[0] == 1:
            return [(1, [m[0, 0]])]

        if m.shape[0] == 2:
            return [(1, [m[0, 0], m[1, 1]]), (-1, [m[0, 1], m[1, 0]])]

        def distribute(sign, element, terms):
            """Distribute an element and sign across symbolic terms."""
            return [(sign * s, [element] + elems) for s, elems in terms]

        def interleave(lists):
            """Interleave terms from multiple cofactor expansions."""
            result = []
            max_len = max(len(lst) for lst in lists)
            for i in range(max_len):
                for lst in lists:
                    if i < len(lst):
                        result.append(lst[i])
            return result

        cofactors = []
        for col in range(m.shape[1]):
            sign = (-1) ** col
            minor = np.delete(np.delete(m, 0, axis=0), col, axis=1)
            sub_terms = Math._symbolic_determinant(minor)
            cofactors.append(distribute(sign, m[0, col], sub_terms))

        return interleave(cofactors)

    def determinant_bra(self, matrix) -> "MathBuilder":
        """
        Render symbolic determinant expansion using bra notation ⟨a,b,c|.

        Each term in the determinant expansion is rendered as a bra with
        the product of elements shown as comma-separated values.

        Args:
            matrix: 2D array-like (numpy array or nested list)

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_bra(y)
            m.render()
        """
        import numpy as np
        matrix = np.asarray(matrix)
        # Track for operator overloads
        if self._has_equals:
            self._rhs_det_matrix = matrix
            self._rhs_det_type = 'bra'
        else:
            self._lhs_det_matrix = matrix
            self._lhs_det_type = 'bra'
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'bra')

    def determinant_ket(self, matrix) -> "MathBuilder":
        """
        Render symbolic determinant expansion using ket notation |a,b,c⟩.

        Each term in the determinant expansion is rendered as a ket with
        the product of elements shown as comma-separated values.

        Args:
            matrix: 2D array-like (numpy array or nested list)

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_ket(y)
            m.render()
        """
        import numpy as np
        matrix = np.asarray(matrix)
        # Track for operator overloads
        if self._has_equals:
            self._rhs_det_matrix = matrix
            self._rhs_det_type = 'ket'
        else:
            self._lhs_det_matrix = matrix
            self._lhs_det_type = 'ket'
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'ket')

    def determinant_braket(self, matrix, bra_label: str = "\\psi") -> "MathBuilder":
        """
        Render symbolic determinant expansion using braket notation ⟨ψ|a,b,c⟩.

        Each term in the determinant expansion is rendered as a braket with
        a fixed bra label and the product elements as the ket.

        Args:
            matrix: 2D array-like (numpy array or nested list)
            bra_label: Label for the bra side (default: ψ)

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_braket(y, bra_label="\\phi")
            m.render()
        """
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'braket', bra_label=bra_label)

    def determinant_product(self, matrix) -> "MathBuilder":
        """
        Render symbolic determinant expansion as product notation (a·b·c).

        Each term in the determinant expansion is rendered as a product
        of elements using cdot.

        Args:
            matrix: 2D array-like (numpy array or nested list)

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_product(y)
            m.render()
        """
        terms = self._symbolic_determinant(matrix)
        return self._render_symbolic_terms(terms, 'product')

    def determinant_subscript(self, matrix, var: str = "a") -> "MathBuilder":
        """
        Render symbolic determinant expansion using subscript notation (a_{ij}).

        Each element is rendered with row/column subscripts.

        Args:
            matrix: 2D array-like (numpy array or nested list)
            var: Variable name for elements (default: 'a')

        Example:
            m = Math()
            y = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
            m.determinant_subscript(y, var="a")
            m.render()
        """
        import numpy as np
        matrix = np.asarray(matrix)
        n = matrix.shape[0]

        # Create symbolic matrix with subscripts
        sym_matrix = [[f"{var}_{{{i+1}{j+1}}}" for j in range(n)] for i in range(n)]
        terms = self._symbolic_determinant(sym_matrix)
        return self._render_symbolic_terms(terms, 'product')

    def _render_symbolic_terms(self, terms: List[tuple], notation: str,
                                bra_label: str = None) -> "MathBuilder":
        """
        Internal method to render symbolic determinant terms in various notations.

        Args:
            terms: List of (sign, [elements]) tuples
            notation: One of 'bra', 'ket', 'braket', 'product'
            bra_label: Label for bra side (only used with 'braket' notation)
        """
        first = True

        for sign, elements in terms:
            # Handle sign
            if sign > 0:
                if not first:
                    self.plus()
            else:
                self.minus()
            first = False

            # Render elements in chosen notation
            if notation == 'bra':
                self.bra(elements)
            elif notation == 'ket':
                self.ket(elements)
            elif notation == 'braket':
                self.braket(bra_label, elements)
            elif notation == 'product':
                # Render as product with cdots
                elem_strs = [str(e) for e in elements]
                product_str = " \\cdot ".join(elem_strs)
                self._append(f"({product_str})")

        return self

    def slater_determinant(self, orbitals: List[str], normalize: bool = True) -> "MathBuilder":
        """
        Render a Slater determinant in standard physics notation.

        A Slater determinant represents an antisymmetrized product of
        single-particle wavefunctions (orbitals).

        Args:
            orbitals: List of orbital labels (e.g., ['1s', '2s', '2p'])
            normalize: Include normalization factor 1/√n!

        Example:
            m = Math()
            m.slater_determinant(['\\phi_1', '\\phi_2', '\\phi_3'])
            m.render()
        """
        n = len(orbitals)

        if normalize:
            self._append(f"\\frac{{1}}{{\\sqrt{{{n}!}}}}")

        # Build determinant matrix representation
        self._append("\\begin{vmatrix}")

        for i in range(n):
            row_parts = []
            for orb in orbitals:
                row_parts.append(f"{orb}(\\mathbf{{r}}_{{{i+1}}})")
            self._append(" & ".join(row_parts))
            if i < n - 1:
                self._append(" \\\\ ")

        self._append("\\end{vmatrix}")

        return self

    def slater_ket(self, orbitals: List[str], normalize: bool = True) -> "MathBuilder":
        """
        Render a Slater determinant in occupation number (ket) notation.

        Args:
            orbitals: List of orbital labels
            normalize: Include normalization factor

        Example:
            m = Math()
            m.slater_ket(['1s↑', '1s↓', '2s↑'])
            m.render()
            # Renders: |1s↑, 1s↓, 2s↑⟩
        """
        n = len(orbitals)

        if normalize:
            self._append(f"\\frac{{1}}{{\\sqrt{{{n}!}}}}")

        self.ket(orbitals)
        return self

    def slater_bra_state(self, state: "SlaterState", normalize: bool = False) -> "MathBuilder":
        """
        Render a SlaterState as a bra ⟨orbitals|.

        This method works with the new SlaterState class that represents
        a Slater determinant as a 1D list of occupied spin-orbitals.

        Args:
            state: SlaterState object
            normalize: Include normalization factor 1/√n!

        Example:
            state = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
            m = Math()
            m.slater_bra_state(state)
            m.render()
            # Renders: ⟨1s↑, 1s↓, 2s↑|
        """
        n = state.n_electrons

        if normalize:
            self._append(f"\\frac{{1}}{{\\sqrt{{{n}!}}}}")

        # Track the SlaterState for operator overloads
        if self._has_equals:
            self._rhs_slater_state = state
            self._rhs_slater_type = 'bra'
        else:
            self._lhs_slater_state = state
            self._lhs_slater_type = 'bra'

        self.bra(state.latex_labels)
        return self

    def slater_ket_state(self, state: "SlaterState", normalize: bool = False) -> "MathBuilder":
        """
        Render a SlaterState as a ket |orbitals⟩.

        This method works with the new SlaterState class that represents
        a Slater determinant as a 1D list of occupied spin-orbitals.

        Args:
            state: SlaterState object
            normalize: Include normalization factor 1/√n!

        Example:
            state = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
            m = Math()
            m.slater_ket_state(state)
            m.render()
            # Renders: |1s↑, 1s↓, 2s↑⟩
        """
        n = state.n_electrons

        if normalize:
            self._append(f"\\frac{{1}}{{\\sqrt{{{n}!}}}}")

        # Track the SlaterState for operator overloads
        if self._has_equals:
            self._rhs_slater_state = state
            self._rhs_slater_type = 'ket'
        else:
            self._lhs_slater_state = state
            self._lhs_slater_type = 'ket'

        self.ket(state.latex_labels)
        return self

    def slater_matrix_element(self, bra_state: "SlaterState", operator: str,
                               ket_state: "SlaterState",
                               apply_symmetry: bool = True) -> "MathBuilder":
        """
        Render a matrix element ⟨bra|op|ket⟩ between two SlaterStates.

        When apply_symmetry is True and states have orthonormal symmetry,
        uses Slater-Condon rules to simplify:
        - If states differ by more than 2 orbitals: result is 0
        - If states are identical: sum of one-electron terms + two-electron terms
        - If states differ by 1 orbital: one-electron + two-electron terms
        - If states differ by 2 orbitals: only two-electron terms

        Args:
            bra_state: SlaterState for bra
            operator: LaTeX string for operator (e.g., "\\hat{H}")
            ket_state: SlaterState for ket
            apply_symmetry: Use symmetry to simplify (default True)

        Example:
            psi = SlaterState.from_labels(["1s↑", "1s↓"])
            phi = SlaterState.from_labels(["1s↑", "2s↑"])
            m = Math()
            m.slater_matrix_element(psi, "\\hat{H}", phi)
            m.render()
        """
        # Render the full matrix element notation
        bra_str = ", ".join(bra_state.latex_labels)
        ket_str = ", ".join(ket_state.latex_labels)
        self._append(f"\\langle {bra_str} | {operator} | {ket_str} \\rangle")

        return self

    def slater_overlap(self, bra_state: "SlaterState", ket_state: "SlaterState",
                        simplify: bool = True) -> "MathBuilder":
        """
        Render and optionally simplify the overlap ⟨bra|ket⟩ between two SlaterStates.

        When simplify is True and states have orthonormal symmetry:
        - Returns 0 if different sets of orbitals
        - Returns ±1 based on permutation parity if same orbitals

        Args:
            bra_state: SlaterState for bra
            ket_state: SlaterState for ket
            simplify: Apply symmetry simplifications (default True)

        Example:
            psi = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
            phi = SlaterState.from_labels(["1s↑", "1s↓", "2s↑"])
            m = Math()
            m.slater_overlap(psi, phi)
            m.render()
            # Renders: 1 (same orbitals)
        """
        if simplify and bra_state.symmetry == Symmetry.ORTHONORMAL:
            is_nonzero, sign = bra_state.overlap_with(ket_state)

            if not is_nonzero:
                self._append("0")
            elif sign is not None:
                self._append(str(sign))
            else:
                # Render full overlap
                bra_str = ", ".join(bra_state.latex_labels)
                ket_str = ", ".join(ket_state.latex_labels)
                self._append(f"\\langle {bra_str} | {ket_str} \\rangle")
        else:
            # Render without simplification
            bra_str = ", ".join(bra_state.latex_labels)
            ket_str = ", ".join(ket_state.latex_labels)
            self._append(f"\\langle {bra_str} | {ket_str} \\rangle")

        return self

    def slater_condon_rule(self, bra_state: "SlaterState", ket_state: "SlaterState",
                           operator_type: str = "one_electron") -> "MathBuilder":
        """
        Apply Slater-Condon rules to determine which terms survive.

        This analyzes the difference between two Slater states and renders
        the appropriate Slater-Condon expression.

        Args:
            bra_state: SlaterState for bra
            ket_state: SlaterState for ket
            operator_type: "one_electron" for ĥ(i), "two_electron" for ĝ(i,j)

        Returns:
            Math object with the Slater-Condon expression
        """
        bra_set = set(bra_state.orbitals)
        ket_set = set(ket_state.orbitals)

        # Find differing orbitals
        only_in_bra = bra_set - ket_set
        only_in_ket = ket_set - bra_set
        n_diff = len(only_in_bra)

        if n_diff > 2:
            # More than 2 orbitals differ: matrix element is zero
            self._append("0")
            return self

        if n_diff == 0:
            # Same orbitals (diagonal case)
            if operator_type == "one_electron":
                # Sum over all occupied orbitals
                self._append("\\sum_i ")
                self._append("\\langle i | \\hat{h} | i \\rangle")
            else:
                # Two-electron: sum over pairs
                self._append("\\frac{1}{2} \\sum_{i \\neq j} ")
                self._append("\\left[ \\langle ij | \\hat{g} | ij \\rangle - \\langle ij | \\hat{g} | ji \\rangle \\right]")

        elif n_diff == 1:
            # Single excitation
            orb_bra = list(only_in_bra)[0]
            orb_ket = list(only_in_ket)[0]

            if operator_type == "one_electron":
                self._append(f"\\langle {orb_bra.latex_label} | \\hat{{h}} | {orb_ket.latex_label} \\rangle")
            else:
                # Two-electron with common orbitals
                common = bra_set & ket_set
                self._append("\\sum_j ")
                self._append(f"\\left[ \\langle {orb_bra.latex_label} j | \\hat{{g}} | {orb_ket.latex_label} j \\rangle ")
                self._append(f"- \\langle {orb_bra.latex_label} j | \\hat{{g}} | j {orb_ket.latex_label} \\rangle \\right]")

        else:  # n_diff == 2
            # Double excitation - only two-electron operator contributes
            if operator_type == "one_electron":
                self._append("0")
            else:
                orbs_bra = list(only_in_bra)
                orbs_ket = list(only_in_ket)
                b1, b2 = orbs_bra[0].latex_label, orbs_bra[1].latex_label
                k1, k2 = orbs_ket[0].latex_label, orbs_ket[1].latex_label

                self._append(f"\\langle {b1} {b2} | \\hat{{g}} | {k1} {k2} \\rangle ")
                self._append(f"- \\langle {b1} {b2} | \\hat{{g}} | {k2} {k1} \\rangle")

        return self

    # Inner Products of Determinants
    @staticmethod
    def _compute_inner_product_terms(bra_terms: List[tuple], ket_terms: List[tuple],
                                      orthogonal: bool = False,
                                      orthogonal_states: Optional[List[str]] = None) -> List[tuple]:
        """
        Compute inner product terms between two symbolic determinant expansions.

        When orthogonality is specified, applies Kronecker delta: ⟨φᵢ|φⱼ⟩ = δᵢⱼ
        Non-matching overlaps evaluate to zero.

        Args:
            bra_terms: Symbolic determinant expansion for bra (from _symbolic_determinant)
            ket_terms: Symbolic determinant expansion for ket (from _symbolic_determinant)
            orthogonal: If True, all states are orthonormal (⟨i|j⟩ = δᵢⱼ)
            orthogonal_states: Optional list of states that are orthogonal to each other.
                              If provided, only these states follow orthogonality rules.

        Returns:
            List of (sign, bra_elements, ket_elements) tuples representing surviving terms
        """
        result = []

        for bra_sign, bra_elems in bra_terms:
            for ket_sign, ket_elems in ket_terms:
                combined_sign = bra_sign * ket_sign

                if orthogonal:
                    # Check if all elements match (Kronecker delta condition)
                    # For orthonormal states: ⟨a,b,c|d,e,f⟩ = δ_ad·δ_be·δ_cf
                    if len(bra_elems) != len(ket_elems):
                        continue

                    # Check pairwise orthogonality
                    all_match = True
                    for b, k in zip(bra_elems, ket_elems):
                        b_str, k_str = str(b), str(k)
                        if orthogonal_states is not None:
                            # Only apply orthogonality to specified states
                            if b_str in orthogonal_states and k_str in orthogonal_states:
                                if b_str != k_str:
                                    all_match = False
                                    break
                        else:
                            # All states are orthogonal
                            if b_str != k_str:
                                all_match = False
                                break

                    if all_match:
                        result.append((combined_sign, bra_elems, ket_elems))
                else:
                    # No orthogonality - keep all terms
                    result.append((combined_sign, bra_elems, ket_elems))

        return result

    def determinant_inner_product(self, bra_matrix, ket_matrix,
                                   orthogonal: bool = False,
                                   orthogonal_states: Optional[List[str]] = None,
                                   show_zeros: bool = False) -> "MathBuilder":
        """
        Render the inner product of two symbolic determinant expansions.

        When orthogonality is enabled, applies the Kronecker delta condition:
        ⟨φᵢ|φⱼ⟩ = δᵢⱼ (non-matching overlaps evaluate to zero).

        Args:
            bra_matrix: 2D array-like for the bra determinant
            ket_matrix: 2D array-like for the ket determinant
            orthogonal: If True, all states are orthonormal
            orthogonal_states: Optional list of states that are mutually orthogonal
            show_zeros: If True, show all terms with zeros; if False, only show surviving terms

        Example:
            m = Math()
            bra = np.array([['a', 'b'], ['c', 'd']])
            ket = np.array([['a', 'b'], ['c', 'd']])
            m.determinant_inner_product(bra, ket, orthogonal=True)
            m.render()
        """
        bra_terms = self._symbolic_determinant(bra_matrix)
        ket_terms = self._symbolic_determinant(ket_matrix)

        inner_terms = self._compute_inner_product_terms(
            bra_terms, ket_terms, orthogonal, orthogonal_states
        )

        if not inner_terms:
            self._append("0")
            return self

        first = True
        for sign, bra_elems, ket_elems in inner_terms:
            # Handle sign
            if sign > 0:
                if not first:
                    self.plus()
            else:
                self.minus()
            first = False

            # Render as ⟨bra|ket⟩ for each pair
            if orthogonal:
                # When orthogonal and matching, render as 1 (or simplified form)
                # Show the overlaps that survived
                overlaps = []
                for b, k in zip(bra_elems, ket_elems):
                    overlaps.append(f"\\langle {b} | {k} \\rangle")
                self._append("(" + " ".join(overlaps) + ")")
            else:
                # Show full inner product notation
                overlaps = []
                for b, k in zip(bra_elems, ket_elems):
                    overlaps.append(f"\\langle {b} | {k} \\rangle")
                self._append("(" + " ".join(overlaps) + ")")

        return self

    def determinant_inner_product_simplified(self, bra_matrix, ket_matrix,
                                              orthogonal: bool = True) -> "MathBuilder":
        """
        Render the simplified inner product assuming orthonormal states.

        When states are orthonormal, ⟨φᵢ|φⱼ⟩ = δᵢⱼ, so matching terms become 1.
        This method shows the final simplified result.

        Args:
            bra_matrix: 2D array-like for the bra determinant
            ket_matrix: 2D array-like for the ket determinant
            orthogonal: If True (default), simplify matching terms to 1

        Example:
            m = Math()
            bra = np.array([['a', 'b'], ['c', 'd']])
            ket = np.array([['a', 'b'], ['c', 'd']])
            m.determinant_inner_product_simplified(bra, ket)
            m.render()
            # With orthogonality: terms where all bra elements match ket elements survive
        """
        bra_terms = self._symbolic_determinant(bra_matrix)
        ket_terms = self._symbolic_determinant(ket_matrix)

        inner_terms = self._compute_inner_product_terms(
            bra_terms, ket_terms, orthogonal=orthogonal
        )

        if not inner_terms:
            self._append("0")
            return self

        # Count surviving terms
        positive_count = sum(1 for sign, _, _ in inner_terms if sign > 0)
        negative_count = sum(1 for sign, _, _ in inner_terms if sign < 0)

        net_result = positive_count - negative_count

        if net_result == 0:
            self._append("0")
        elif net_result > 0:
            self._append(str(net_result))
        else:
            self._append(str(net_result))

        return self

    def slater_inner_product(self, bra_orbitals: List[str], ket_orbitals: List[str],
                              orthogonal: bool = True,
                              normalize: bool = True) -> "MathBuilder":
        """
        Render the inner product of two Slater determinants.

        For orthonormal orbitals, ⟨Ψ|Φ⟩ = δ_{Ψ,Φ} (determinants are orthogonal
        unless they have the same set of occupied orbitals).

        Args:
            bra_orbitals: List of orbital labels for bra Slater determinant
            ket_orbitals: List of orbital labels for ket Slater determinant
            orthogonal: If True, orbitals are orthonormal
            normalize: Include normalization factors

        Example:
            m = Math()
            m.slater_inner_product(['a', 'b', 'c'], ['a', 'b', 'c'], orthogonal=True)
            m.render()
        """
        n_bra = len(bra_orbitals)
        n_ket = len(ket_orbitals)

        if normalize:
            # Normalization: 1/sqrt(n!) for each determinant
            self._append(f"\\frac{{1}}{{{n_bra}!}}")

        # Create symbolic matrices for the Slater determinants
        # Each row i represents electron i, each column j represents orbital j
        import numpy as np
        bra_matrix = np.array([[orb for orb in bra_orbitals] for _ in range(n_bra)])
        ket_matrix = np.array([[orb for orb in ket_orbitals] for _ in range(n_ket)])

        # For proper Slater determinant inner product, we need permutation matching
        # The inner product expands to n! terms, with orthogonality reducing this

        if orthogonal:
            # For orthonormal orbitals: ⟨Ψ|Φ⟩ = 1 if same orbitals, 0 otherwise
            # (accounting for antisymmetry and normalization)
            bra_set = set(str(o) for o in bra_orbitals)
            ket_set = set(str(o) for o in ket_orbitals)

            if bra_set == ket_set:
                # Same occupied orbitals - result is 1 (after normalization)
                if not normalize:
                    import math
                    self._append(f"{math.factorial(n_bra)}")
                else:
                    self._append("1")
            else:
                self._append("0")
        else:
            # Show the full expansion without orthogonality simplification
            self._append("\\sum_{P} (-1)^P ")
            overlaps = []
            for i, (b, k) in enumerate(zip(bra_orbitals, ket_orbitals)):
                overlaps.append(f"\\langle {b} | {k} \\rangle")
            self._append(" ".join(overlaps))

        return self

    def determinant_overlap_expansion(self, bra_matrix, ket_matrix,
                                       notation: str = 'braket') -> "MathBuilder":
        """
        Render the full overlap expansion of two determinants without simplification.

        Shows all pairwise inner products between bra and ket determinant terms.

        Args:
            bra_matrix: 2D array-like for the bra determinant
            ket_matrix: 2D array-like for the ket determinant
            notation: 'braket' for ⟨a|b⟩ notation, 'product' for (a*·b) notation

        Example:
            m = Math()
            bra = np.array([['a', 'b'], ['c', 'd']])
            ket = np.array([['e', 'f'], ['g', 'h']])
            m.determinant_overlap_expansion(bra, ket)
            m.render()
        """
        bra_terms = self._symbolic_determinant(bra_matrix)
        ket_terms = self._symbolic_determinant(ket_matrix)

        # Show bra expansion
        self._append("\\Bigl(")
        first = True
        for sign, elements in bra_terms:
            if sign > 0:
                if not first:
                    self._append(" + ")
            else:
                self._append(" - ")
            first = False
            self.bra(elements)
        self._append("\\Bigr)")

        self._append("\\Bigl(")
        first = True
        for sign, elements in ket_terms:
            if sign > 0:
                if not first:
                    self._append(" + ")
            else:
                self._append(" - ")
            first = False
            self.ket(elements)
        self._append("\\Bigr)")

        return self

    # Spacing
    def space(self) -> "MathBuilder":
        return self._append("\\ ")

    def quad(self) -> "MathBuilder":
        return self._append("\\quad")

    def qquad(self) -> "MathBuilder":
        return self._append("\\qquad")

    # Line breaks for multi-line equations
    def newline(self) -> "MathBuilder":
        """Add a line break (\\\\) for use in aligned environments."""
        return self._append(" \\\\ ")

    def br(self) -> "MathBuilder":
        """Alias for newline() - add a line break."""
        return self.newline()

    def align_eq(self) -> "MathBuilder":
        """Add alignment marker (&) followed by equals sign for aligned environments."""
        return self._append(" &= ")

    def align_mark(self) -> "MathBuilder":
        """Add alignment marker (&) for aligned environments."""
        return self._append(" & ")

    # Build and render
    def build(self) -> str:
        """Build and return the LaTeX string."""
        return "".join(self._parts)

    def render(self, display: bool = True, label: Optional[str] = None,
               justify: str = "center", multiline: bool = False, scrollable: bool = False):
        """
        Render the built expression to HTML output.

        Args:
            display: If True, render as display math (block). If False, inline.
            label: Optional label/caption for the expression.
            justify: Alignment - 'left', 'center', or 'right'.
            multiline: If True, wrap in aligned environment for line breaks.
            scrollable: If True, wrap in scrollable container for long equations.
        """
        expr = self.build()

        # Wrap in aligned environment for multi-line equations
        if multiline or " \\\\ " in expr:
            expr = f"\\begin{{aligned}} {expr} \\end{{aligned}}"

        # Use scrollable container if requested
        if scrollable:
            _render_scrollable(expr, display=display, label=label, justify=justify)
        else:
            latex(expr, display=display, label=label, justify=justify)

    def clear(self) -> "MathBuilder":
        """Clear the builder."""
        self._parts = []
        return self

    def __str__(self) -> str:
        return self.build()

    # Operator overloads for inner products
    def __matmul__(self, other: "Math") -> "MathBuilder":
        """
        Matrix multiplication operator (@) for computing inner products.

        When two Math objects are combined with @, it computes the inner product
        of both the LHS and RHS of their equations separately.

        For bra @ ket operations:
        - LHS: ⟨bra_lhs| @ |ket_lhs⟩ = ⟨bra_lhs|ket_lhs⟩
        - RHS: ⟨bra_rhs| @ |ket_rhs⟩ = ⟨bra_rhs|ket_rhs⟩

        For bra @ op @ ket operations (matrix elements):
        - LHS: ⟨bra_lhs|op_lhs|ket_lhs⟩
        - RHS: ⟨bra_rhs|op_rhs|ket_rhs⟩

        Example:
            m_g = Math()
            m_g.determinant_bra(sm_g).equals().bra('\\phi_1')

            m_1 = Math()
            m_1.determinant_ket(sm_1).equals().ket('\\phi_2')

            m_s = m_g @ m_1  # Inner product of both sides
            m_s.render()

            # With operator:
            my_op = Math().var("h")
            m_o = m_g @ my_op @ m_1  # Matrix element ⟨...|h|...⟩
            m_o.render()
        """
        if not isinstance(other, Math):
            raise TypeError(f"unsupported operand type(s) for @: 'Math' and '{type(other).__name__}'")

        # Check what kind of objects we're dealing with
        self_is_bra = (self._lhs_det_type == 'bra' or self._lhs_bra_label is not None or
                       self._lhs_slater_type == 'bra')
        self_has_pending_op = self._pending_operator is not None
        other_is_ket = (other._lhs_det_type == 'ket' or other._lhs_ket_label is not None or
                        other._lhs_slater_type == 'ket')
        other_is_bra = (other._lhs_det_type == 'bra' or other._lhs_bra_label is not None or
                        other._lhs_slater_type == 'bra')

        # Case 1: bra @ op (store operator for later, return intermediate result)
        # The "other" is an operator (not a bra or ket)
        if self_is_bra and not other_is_ket and not other_is_bra:
            result = Math()
            # Copy bra information to result
            result._lhs_det_matrix = self._lhs_det_matrix
            result._lhs_det_type = self._lhs_det_type
            result._lhs_bra_label = self._lhs_bra_label
            result._rhs_det_matrix = self._rhs_det_matrix
            result._rhs_det_type = self._rhs_det_type
            result._rhs_bra_label = self._rhs_bra_label
            result._has_equals = self._has_equals
            # Copy SlaterState info
            result._lhs_slater_state = self._lhs_slater_state
            result._lhs_slater_type = self._lhs_slater_type
            result._rhs_slater_state = self._rhs_slater_state
            result._rhs_slater_type = self._rhs_slater_type
            # Store operator for when we encounter the ket
            result._pending_operator = other
            return result

        # Case 2: (bra @ op) @ ket - we have a pending operator
        if self_has_pending_op and other_is_ket:
            result = Math()
            op = self._pending_operator

            # Get operator strings for LHS and RHS
            # If operator doesn't have equals, use same expression for both sides
            lhs_op_str = "".join(op._lhs_parts) if op._lhs_parts else op.build()
            if op._has_equals:
                rhs_op_str = "".join(op._rhs_parts) if op._rhs_parts else op.build()
            else:
                rhs_op_str = lhs_op_str

            # Compute LHS matrix element
            lhs_computed = self._compute_side_matrix_element(
                self._lhs_det_matrix, self._lhs_det_type, self._lhs_bra_label,
                lhs_op_str,
                other._lhs_det_matrix, other._lhs_det_type, other._lhs_ket_label
            )
            result._append(lhs_computed)

            # If both have equations, compute RHS matrix element
            if self._has_equals and other._has_equals:
                result._has_equals = True
                result._append(" = ")

                rhs_computed = self._compute_side_matrix_element(
                    self._rhs_det_matrix, self._rhs_det_type, self._rhs_bra_label,
                    rhs_op_str,
                    other._rhs_det_matrix, other._rhs_det_type, other._rhs_ket_label
                )
                result._append(rhs_computed)
            elif self._has_equals or other._has_equals:
                # One has equals - use operator's expression for RHS
                result._has_equals = True
                result._append(" = ")

                # Use RHS if available, otherwise LHS
                bra_matrix = self._rhs_det_matrix if self._has_equals else self._lhs_det_matrix
                bra_type = self._rhs_det_type if self._has_equals else self._lhs_det_type
                bra_label = self._rhs_bra_label if self._has_equals else self._lhs_bra_label
                ket_matrix = other._rhs_det_matrix if other._has_equals else other._lhs_det_matrix
                ket_type = other._rhs_det_type if other._has_equals else other._lhs_det_type
                ket_label = other._rhs_ket_label if other._has_equals else other._lhs_ket_label

                rhs_computed = self._compute_side_matrix_element(
                    bra_matrix, bra_type, bra_label,
                    rhs_op_str,
                    ket_matrix, ket_type, ket_label
                )
                result._append(rhs_computed)

            return result

        # Case 3: Simple bra @ ket (no operator)
        if self_is_bra and other_is_ket:
            result = Math()

            # Check if we're using SlaterState objects
            if self._lhs_slater_state is not None and other._lhs_slater_state is not None:
                # Use SlaterState-aware inner product
                lhs_computed = self._compute_slater_inner_product(
                    self._lhs_slater_state, other._lhs_slater_state
                )
                result._append(lhs_computed)

                # If both have equations, compute RHS inner product
                if self._has_equals and other._has_equals:
                    result._has_equals = True
                    result._append(" = ")

                    rhs_computed = self._compute_slater_inner_product(
                        self._rhs_slater_state, other._rhs_slater_state
                    )
                    result._append(rhs_computed)
            else:
                # Fall back to original matrix-based inner product
                lhs_computed = self._compute_side_inner_product(
                    self._lhs_det_matrix, self._lhs_det_type, self._lhs_bra_label, self._lhs_ket_label,
                    other._lhs_det_matrix, other._lhs_det_type, other._lhs_bra_label, other._lhs_ket_label
                )
                result._append(lhs_computed)

                # If both have equations, compute RHS inner product
                if self._has_equals and other._has_equals:
                    result._has_equals = True
                    result._append(" = ")

                    rhs_computed = self._compute_side_inner_product(
                        self._rhs_det_matrix, self._rhs_det_type, self._rhs_bra_label, self._rhs_ket_label,
                        other._rhs_det_matrix, other._rhs_det_type, other._rhs_bra_label, other._rhs_ket_label
                    )
                    result._append(rhs_computed)

            return result

        # Invalid combination
        if not self_is_bra and not self_has_pending_op:
            raise ValueError("Left operand of @ must contain bra notation (use determinant_bra or bra)")
        raise ValueError("Right operand of @ must contain ket notation or be an operator expression")

    def _compute_slater_inner_product(self, bra_state: "SlaterState",
                                       ket_state: "SlaterState") -> str:
        """
        Compute the inner product ⟨bra|ket⟩ between two SlaterState objects.

        Uses the symmetry settings on the bra_state to determine simplification.
        For orthonormal states, applies Kronecker delta rules.
        """
        if bra_state.symmetry == Symmetry.ORTHONORMAL:
            is_nonzero, sign = bra_state.overlap_with(ket_state)

            if not is_nonzero:
                return "0"
            elif sign is not None:
                return str(sign)

        # Render full overlap notation
        bra_str = ", ".join(bra_state.latex_labels)
        ket_str = ", ".join(ket_state.latex_labels)
        return f"\\langle {bra_str} | {ket_str} \\rangle"

    def _compute_side_inner_product(self, bra_matrix, bra_det_type, bra_label, bra_ket_label,
                                     ket_matrix, ket_det_type, ket_bra_label, ket_label) -> str:
        """
        Compute the inner product for one side (LHS or RHS) of the equation.

        Handles combinations of:
        - determinant_bra @ determinant_ket -> combined braket notation ⟨a,b,c|d,e,f⟩
        - bra @ ket -> simple braket
        - determinant_bra @ ket -> simplified notation
        - bra @ determinant_ket -> simplified notation
        """
        import numpy as np

        # Case 1: Both are determinants
        if bra_matrix is not None and ket_matrix is not None:
            if bra_det_type != 'bra':
                raise ValueError("Left side must be bra notation for inner product")
            if ket_det_type != 'ket':
                raise ValueError("Right side must be ket notation for inner product")

            # Check shape compatibility
            bra_shape = np.asarray(bra_matrix).shape
            ket_shape = np.asarray(ket_matrix).shape
            if bra_shape != ket_shape:
                raise ValueError(f"Shape mismatch: bra has shape {bra_shape}, ket has shape {ket_shape}")

            # Compute inner product terms
            bra_terms = self._symbolic_determinant(bra_matrix)
            ket_terms = self._symbolic_determinant(ket_matrix)
            inner_terms = self._compute_inner_product_terms(bra_terms, ket_terms, orthogonal=False)

            # Build result string using combined braket notation ⟨a,b,c|d,e,f⟩
            parts = []
            first = True
            for sign, bra_elems, ket_elems in inner_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False

                # Combined braket: ⟨bra_elements|ket_elements⟩
                bra_str = ", ".join(str(b) for b in bra_elems)
                ket_str = ", ".join(str(k) for k in ket_elems)
                parts.append(f"\\langle {bra_str} | {ket_str} \\rangle")

            return "".join(parts) if parts else "0"

        # Case 2: Simple bra @ simple ket
        elif bra_label is not None and ket_label is not None:
            bra_str = bra_label
            ket_str = ket_label
            if isinstance(bra_str, (list, tuple)):
                bra_str = ", ".join(str(c) for c in bra_str)
            if isinstance(ket_str, (list, tuple)):
                ket_str = ", ".join(str(c) for c in ket_str)
            return f"\\langle {bra_str} | {ket_str} \\rangle"

        # Case 3: determinant_bra @ simple ket
        elif bra_matrix is not None and ket_label is not None:
            ket_str = ket_label
            if isinstance(ket_str, (list, tuple)):
                ket_str = ", ".join(str(c) for c in ket_str)

            # Render determinant bra terms contracted with ket
            bra_terms = self._symbolic_determinant(bra_matrix)
            parts = []
            first = True
            for sign, elements in bra_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False
                elem_str = ", ".join(str(e) for e in elements)
                parts.append(f"\\langle {elem_str} | {ket_str} \\rangle")

            return "".join(parts)

        # Case 4: simple bra @ determinant_ket
        elif bra_label is not None and ket_matrix is not None:
            bra_str = bra_label
            if isinstance(bra_str, (list, tuple)):
                bra_str = ", ".join(str(c) for c in bra_str)

            # Render bra contracted with determinant ket terms
            ket_terms = self._symbolic_determinant(ket_matrix)
            parts = []
            first = True
            for sign, elements in ket_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False
                elem_str = ", ".join(str(e) for e in elements)
                parts.append(f"\\langle {bra_str} | {elem_str} \\rangle")

            return "".join(parts)

        else:
            raise ValueError("Cannot compute inner product: incompatible operands")

    def _compute_side_matrix_element(self, bra_matrix, bra_det_type, bra_label,
                                      operator_str: str,
                                      ket_matrix, ket_det_type, ket_label) -> str:
        """
        Compute the matrix element ⟨bra|op|ket⟩ for one side of the equation.

        Handles combinations of:
        - determinant_bra @ op @ determinant_ket -> ⟨a,b,c|op|d,e,f⟩ terms
        - bra @ op @ ket -> simple ⟨bra|op|ket⟩
        - determinant_bra @ op @ ket -> determinant terms with operator
        - bra @ op @ determinant_ket -> bra with determinant ket terms
        """
        import numpy as np

        # Case 1: Both are determinants
        if bra_matrix is not None and ket_matrix is not None:
            if bra_det_type != 'bra':
                raise ValueError("Left side must be bra notation for matrix element")
            if ket_det_type != 'ket':
                raise ValueError("Right side must be ket notation for matrix element")

            # Check shape compatibility
            bra_shape = np.asarray(bra_matrix).shape
            ket_shape = np.asarray(ket_matrix).shape
            if bra_shape != ket_shape:
                raise ValueError(f"Shape mismatch: bra has shape {bra_shape}, ket has shape {ket_shape}")

            # Compute matrix element terms
            bra_terms = self._symbolic_determinant(bra_matrix)
            ket_terms = self._symbolic_determinant(ket_matrix)
            inner_terms = self._compute_inner_product_terms(bra_terms, ket_terms, orthogonal=False)

            # Build result string using matrix element notation ⟨a,b,c|op|d,e,f⟩
            parts = []
            first = True
            for sign, bra_elems, ket_elems in inner_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False

                # Matrix element: ⟨bra_elements|operator|ket_elements⟩
                bra_str = ", ".join(str(b) for b in bra_elems)
                ket_str = ", ".join(str(k) for k in ket_elems)
                parts.append(f"\\langle {bra_str} | {operator_str} | {ket_str} \\rangle")

            return "".join(parts) if parts else "0"

        # Case 2: Simple bra @ op @ simple ket
        elif bra_label is not None and ket_label is not None:
            bra_str = bra_label
            ket_str = ket_label
            if isinstance(bra_str, (list, tuple)):
                bra_str = ", ".join(str(c) for c in bra_str)
            if isinstance(ket_str, (list, tuple)):
                ket_str = ", ".join(str(c) for c in ket_str)
            return f"\\langle {bra_str} | {operator_str} | {ket_str} \\rangle"

        # Case 3: determinant_bra @ op @ simple ket
        elif bra_matrix is not None and ket_label is not None:
            ket_str = ket_label
            if isinstance(ket_str, (list, tuple)):
                ket_str = ", ".join(str(c) for c in ket_str)

            bra_terms = self._symbolic_determinant(bra_matrix)
            parts = []
            first = True
            for sign, elements in bra_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False
                elem_str = ", ".join(str(e) for e in elements)
                parts.append(f"\\langle {elem_str} | {operator_str} | {ket_str} \\rangle")

            return "".join(parts)

        # Case 4: simple bra @ op @ determinant_ket
        elif bra_label is not None and ket_matrix is not None:
            bra_str = bra_label
            if isinstance(bra_str, (list, tuple)):
                bra_str = ", ".join(str(c) for c in bra_str)

            ket_terms = self._symbolic_determinant(ket_matrix)
            parts = []
            first = True
            for sign, elements in ket_terms:
                if sign > 0:
                    if not first:
                        parts.append(" + ")
                else:
                    parts.append(" - ")
                first = False
                elem_str = ", ".join(str(e) for e in elements)
                parts.append(f"\\langle {bra_str} | {operator_str} | {elem_str} \\rangle")

            return "".join(parts)

        else:
            raise ValueError("Cannot compute matrix element: incompatible operands")

    def __rmatmul__(self, other):
        """Right matrix multiplication - not typically used but included for completeness."""
        raise TypeError("Right @ operation not supported. Use bra @ ket order.")

    def __add__(self, other: "Math") -> "MathBuilder":
        """
        Addition operator (+) for combining Math expressions.

        Concatenates two Math expressions with a plus sign between them.

        Example:
            m1 = Math().var('a')
            m2 = Math().var('b')
            m3 = m1 + m2  # Results in "a + b"
        """
        if not isinstance(other, Math):
            raise TypeError(f"unsupported operand type(s) for +: 'Math' and '{type(other).__name__}'")

        result = Math()
        result._parts = self._parts.copy()
        result._parts.append(" + ")
        result._parts.extend(other._parts)
        return result

    def __sub__(self, other: "Math") -> "MathBuilder":
        """
        Subtraction operator (-) for combining Math expressions.

        Concatenates two Math expressions with a minus sign between them.

        Example:
            m1 = Math().var('a')
            m2 = Math().var('b')
            m3 = m1 - m2  # Results in "a - b"
        """
        if not isinstance(other, Math):
            raise TypeError(f"unsupported operand type(s) for -: 'Math' and '{type(other).__name__}'")

        result = Math()
        result._parts = self._parts.copy()
        result._parts.append(" - ")
        result._parts.extend(other._parts)
        return result

    def __mul__(self, other: "Math") -> "MathBuilder":
        """
        Multiplication operator (*) for combining Math expressions.

        Concatenates two Math expressions with a cdot between them.

        Example:
            m1 = Math().var('a')
            m2 = Math().var('b')
            m3 = m1 * m2  # Results in "a \\cdot b"
        """
        if not isinstance(other, Math):
            raise TypeError(f"unsupported operand type(s) for *: 'Math' and '{type(other).__name__}'")

        result = Math()
        result._parts = self._parts.copy()
        result._parts.append(" \\cdot ")
        result._parts.extend(other._parts)
        return result


# Convenience functions for common expressions
def fraction(num: str, denom: str):
    """Render a simple fraction."""
    latex(f"\\frac{{{num}}}{{{denom}}}")


def sqrt(content: str, n: Optional[str] = None):
    """Render a square root or nth root."""
    if n:
        latex(f"\\sqrt[{n}]{{{content}}}")
    else:
        latex(f"\\sqrt{{{content}}}")


# Chemical notation helpers
def chemical(formula: str):
    """
    Render a chemical formula.

    Example:
        chemical("H2O")
        chemical("2H2 + O2 -> 2H2O")
    """
    # Simple conversion: numbers after letters become subscripts
    import re
    # Convert trailing numbers to subscripts
    result = re.sub(r'([A-Za-z])(\d+)', r'\1_{\2}', formula)
    # Convert arrows
    result = result.replace('->', r'\rightarrow')
    result = result.replace('<->', r'\rightleftharpoons')
    result = result.replace('<=>', r'\rightleftharpoons')

    latex(f"\\mathrm{{{result}}}")


def reaction(reactants: str, products: str, reversible: bool = False):
    """
    Render a chemical reaction.

    Example:
        reaction("2H2 + O2", "2H2O")
        reaction("N2 + 3H2", "2NH3", reversible=True)
    """
    arrow = r"\rightleftharpoons" if reversible else r"\rightarrow"
    chemical(f"{reactants} {arrow} {products}")
