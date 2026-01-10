"""
Chemical Machines Symbols Core Module

Core expression tree classes for symbolic computation.
"""

from typing import Optional, List, Union, Dict, Set, Tuple, Callable, Any
from abc import ABC, abstractmethod

__all__ = [
    # Core classes
    'EvaluationError',
    'Expr',
    'Var',
    'Const',
    'SymbolicConst',
    'SympyWrapper',
    # Binary operations
    'BinOp',
    'Add',
    'Sub',
    'Mul',
    'Div',
    'Pow',
    # Unary operations
    'UnaryOp',
    'Neg',
    'Sqrt',
    'Sin',
    'Cos',
    'Tan',
    'Exp',
    'Log',
    'Abs',
    # Calculus
    'Integral',
    'Derivative',
    # Summation/Product
    'Sum',
    'Product',
    # Utilities
    '_ensure_expr',
    '_get_sympy',
    '_get_torch',
]

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


