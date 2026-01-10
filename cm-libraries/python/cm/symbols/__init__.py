"""
Chemical Machines Symbols Package

A package for symbolic computation with LaTeX rendering support.

This package provides:
- Core expression classes (Expr, Var, Const, operators)
- Special mathematical functions (Bessel, Gamma, polynomials, etc.)
- Differential operators (gradient, Laplacian)
- Atomic/molecular orbital functions (hydrogen, Slater, Gaussian)
- LaTeX rendering and display utilities

Usage:
    from cm.symbols import Math, Var, Const, latex

    # Create symbolic expressions
    x = Math.var("x")
    expr = x**2 + 1
    expr.render()

    # Use special functions
    from cm.symbols import BesselJ
    nu = Math.var("nu")
    z = Math.var("z")
    bessel = BesselJ(nu, z)
    bessel.render()
"""

# Core expression classes
from .core import (
    EvaluationError,
    Expr,
    Var,
    Const,
    SymbolicConst,
    SympyWrapper,
    BinOp,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    UnaryOp,
    Neg,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Abs,
    Integral,
    Derivative,
    Sum,
    Product,
    _ensure_expr,
    _get_sympy,
    _get_torch,
)

# Math factory and function composition
from .functions import (
    Math,
    Scalar,
    ExprType,
    BoundsType,
    ParamType,
    ParamSpec,
    SymbolicFunction,
    BoundFunction,
    ComputeGraph,
    TorchFunction,
    TorchGradFunction,
)

# Special functions
from .special import (
    SpecialFunction,
    # Gamma and related
    Gamma,
    LogGamma,
    Digamma,
    Beta,
    Factorial,
    DoubleFactorial,
    Binomial,
    # Error functions
    Erf,
    Erfc,
    Erfi,
    # Bessel functions
    BesselJ,
    BesselY,
    BesselI,
    BesselK,
    SphericalBesselJ,
    SphericalBesselY,
    Hankel1,
    Hankel2,
    # Airy functions
    AiryAi,
    AiryBi,
    AiryAiPrime,
    AiryBiPrime,
    # Orthogonal polynomials
    Legendre,
    AssocLegendre,
    Hermite,
    HermiteProb,
    Laguerre,
    AssocLaguerre,
    Chebyshev1,
    Chebyshev2,
    Gegenbauer,
    Jacobi,
    # Spherical harmonics
    SphericalHarmonic,
    RealSphericalHarmonic,
    # Hypergeometric functions
    Hypergeometric2F1,
    Hypergeometric1F1,
    Hypergeometric0F1,
    HypergeometricPFQ,
    # Elliptic integrals
    EllipticK,
    EllipticE,
    EllipticPi,
    # Other special functions
    Zeta,
    PolyLog,
    DiracDelta,
    Heaviside,
    KroneckerDelta,
    LeviCivita,
    # Angular momentum coupling
    ClebschGordan,
    Wigner3j,
    Wigner6j,
    Wigner9j,
)

# Differential operators
from .operators import (
    DifferentialOperator,
    PartialDerivative,
    Gradient,
    Laplacian,
)

# Orbital functions
from .orbitals import (
    HydrogenRadial,
    HydrogenOrbital,
    SlaterTypeOrbital,
    GaussianTypeOrbital,
    ContractedGTO,
)

# Display and rendering
from .display import (
    latex,
    equation,
    align,
    matrix,
    bullets,
    numbered,
    items,
    set_notation,
    set_line_height,
    chemical,
    reaction,
    fraction,
    sqrt,
)

__all__ = [
    # Core expression classes
    'Math',
    'Expr',
    'Var',
    'Const',
    'Sum',
    'Product',
    # Hyperparameter types
    'Scalar',
    'ExprType',
    'BoundsType',
    # Function composition
    'SymbolicFunction',
    'BoundFunction',
    'ComputeGraph',
    # PyTorch compilation
    'TorchFunction',
    'TorchGradFunction',
    # Special functions - Base class
    'SpecialFunction',
    # Gamma and related
    'Gamma',
    'LogGamma',
    'Digamma',
    'Beta',
    'Factorial',
    'DoubleFactorial',
    'Binomial',
    # Error functions
    'Erf',
    'Erfc',
    'Erfi',
    # Bessel functions
    'BesselJ',
    'BesselY',
    'BesselI',
    'BesselK',
    'SphericalBesselJ',
    'SphericalBesselY',
    'Hankel1',
    'Hankel2',
    # Airy functions
    'AiryAi',
    'AiryBi',
    'AiryAiPrime',
    'AiryBiPrime',
    # Orthogonal polynomials
    'Legendre',
    'AssocLegendre',
    'Hermite',
    'HermiteProb',
    'Laguerre',
    'AssocLaguerre',
    'Chebyshev1',
    'Chebyshev2',
    'Gegenbauer',
    'Jacobi',
    # Spherical harmonics
    'SphericalHarmonic',
    'RealSphericalHarmonic',
    # Hypergeometric functions
    'Hypergeometric2F1',
    'Hypergeometric1F1',
    'Hypergeometric0F1',
    'HypergeometricPFQ',
    # Elliptic integrals
    'EllipticK',
    'EllipticE',
    'EllipticPi',
    # Other special functions
    'Zeta',
    'PolyLog',
    'DiracDelta',
    'Heaviside',
    'KroneckerDelta',
    'LeviCivita',
    # Angular momentum coupling
    'ClebschGordan',
    'Wigner3j',
    'Wigner6j',
    'Wigner9j',
    # Differential operators
    'DifferentialOperator',
    'PartialDerivative',
    'Gradient',
    'Laplacian',
    # Hydrogen-like wavefunctions
    'HydrogenRadial',
    'HydrogenOrbital',
    # Basis functions
    'SlaterTypeOrbital',
    'GaussianTypeOrbital',
    'ContractedGTO',
    # LaTeX rendering
    'latex',
    'equation',
    'align',
    'matrix',
    'bullets',
    'numbered',
    'items',
    'set_notation',
    'set_line_height',
    # Chemistry helpers
    'chemical',
    'reaction',
    'fraction',
    'sqrt',
]
