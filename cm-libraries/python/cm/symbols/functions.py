"""
Chemical Machines Symbols Functions Module

Math factory class and function composition utilities.
"""

from typing import Optional, List, Union, Dict, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .core import (
    Expr, Var, Const, SymbolicConst, SympyWrapper,
    Add, Sub, Mul, Div, Pow, Neg,
    Sqrt, Sin, Cos, Tan, Exp, Log, Abs,
    Integral, Derivative, Sum, Product,
    _ensure_expr, _get_sympy, _get_torch,
)

__all__ = [
    'Math',
    'Scalar',
    'ExprType',
    'BoundsType',
    'ParamType',
    'ParamSpec',
    'SymbolicFunction',
    'BoundFunction',
    'ComputeGraph',
    'TorchFunction',
    'TorchGradFunction',
]


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


