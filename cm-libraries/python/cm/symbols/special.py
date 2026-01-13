"""
Chemical Machines Symbols Special Functions Module

Mathematical special functions commonly used in physics and engineering.
Includes Bessel functions, orthogonal polynomials, spherical harmonics,
hypergeometric functions, and more.
"""

from typing import Optional, List, Union, Dict, Set, Tuple

from .core import (
    Expr, Var, Const, BinOp, UnaryOp,
    _ensure_expr, _get_sympy,
)

__all__ = [
    'SpecialFunction',
    # Gamma and related
    'Gamma', 'LogGamma', 'Digamma', 'Beta',
    'Factorial', 'DoubleFactorial', 'Binomial',
    # Error functions
    'Erf', 'Erfc', 'Erfi',
    # Bessel functions
    'BesselJ', 'BesselY', 'BesselI', 'BesselK',
    'SphericalBesselJ', 'SphericalBesselY',
    'Hankel1', 'Hankel2',
    # Airy functions
    'AiryAi', 'AiryBi', 'AiryAiPrime', 'AiryBiPrime',
    # Orthogonal polynomials
    'Legendre', 'AssocLegendre',
    'Hermite', 'HermiteProb',
    'Laguerre', 'AssocLaguerre',
    'Chebyshev1', 'Chebyshev2',
    'Gegenbauer', 'Jacobi',
    # Spherical harmonics
    'SphericalHarmonic', 'RealSphericalHarmonic',
    # Hypergeometric functions
    'Hypergeometric2F1', 'Hypergeometric1F1',
    'Hypergeometric0F1', 'HypergeometricPFQ',
    # Elliptic integrals
    'EllipticK', 'EllipticE', 'EllipticPi',
    # Other special functions
    'Zeta', 'PolyLog',
    'DiracDelta', 'Heaviside',
    'KroneckerDelta', 'LeviCivita',
    # Angular momentum coupling
    'ClebschGordan', 'Wigner3j', 'Wigner6j', 'Wigner9j',
]


# =============================================================================
# SPECIAL FUNCTIONS
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

    def expand(self) -> Optional[Expr]:
        """
        Expand to explicit formula when l and m are concrete integers.

        Returns:
            Explicit expression, or None if l,m are symbolic
        """
        # Check if l and m are concrete values
        if not isinstance(self.l, Const) or not isinstance(self.m, Const):
            return None

        l_val = int(self.l.value)
        m_val = int(self.m.value)

        # Import needed functions
        from .core import Const as C
        from . import Sin, Cos, Exp

        # Get theta and phi variables
        theta = self.theta
        phi = self.phi

        # Compute normalization constant
        from math import factorial, pi, sqrt

        # N_l^m = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
        abs_m = abs(m_val)
        norm_sq = (2*l_val + 1) / (4*pi) * factorial(l_val - abs_m) / factorial(l_val + abs_m)
        N = C(sqrt(norm_sq))

        # Phase factor for m < 0 (Condon-Shortley convention)
        if m_val < 0:
            phase = C((-1)**abs_m)
        else:
            phase = C(1)

        # Associated Legendre polynomial P_l^|m|(cos θ)
        # We'll use explicit formulas for low l,m values
        cos_theta = Cos(theta)
        sin_theta = Sin(theta)

        # Build P_l^|m|(x) where x = cos(θ)
        P_lm = self._legendre_poly(l_val, abs_m, cos_theta, sin_theta)

        # Azimuthal part: exp(i*m*φ)
        if m_val != 0:
            # Use SymbolicConst for the imaginary unit
            from .core import SymbolicConst
            I = SymbolicConst("I")  # imaginary unit
            exp_part = Exp(I * C(m_val) * phi)
        else:
            exp_part = C(1)

        # Combine: N * P_l^|m|(cos θ) * exp(i*m*φ) * phase
        result = phase * N * P_lm * exp_part
        return result

    def _legendre_poly(self, l: int, m: int, cos_theta: Expr, sin_theta: Expr) -> Expr:
        """
        Generate associated Legendre polynomial P_l^m(cos θ).

        Uses explicit formulas for l <= 3.
        """
        from .core import Const as C
        from . import Pow

        x = cos_theta  # cos(θ)
        sx = sin_theta  # sin(θ)

        # P_l^m(x) formulas
        if l == 0 and m == 0:
            return C(1)

        elif l == 1:
            if m == 0:
                return x
            elif m == 1:
                return -sx

        elif l == 2:
            if m == 0:
                # (3x² - 1)/2
                return (C(3) * Pow(x, C(2)) - C(1)) * C(0.5)
            elif m == 1:
                # -3x*sin(θ)
                return C(-3) * x * sx
            elif m == 2:
                # 3*sin²(θ)
                return C(3) * Pow(sx, C(2))

        elif l == 3:
            if m == 0:
                # (5x³ - 3x)/2
                return (C(5) * Pow(x, C(3)) - C(3) * x) * C(0.5)
            elif m == 1:
                # -3(5x² - 1)*sin(θ)/2
                return C(-1.5) * (C(5) * Pow(x, C(2)) - C(1)) * sx
            elif m == 2:
                # 15x*sin²(θ)
                return C(15) * x * Pow(sx, C(2))
            elif m == 3:
                # -15*sin³(θ)
                return C(-15) * Pow(sx, C(3))

        # For higher l, fall back to SymPy (won't expand)
        return None

    def conjugate(self) -> Expr:
        """
        Return complex conjugate of spherical harmonic.

        For Y_l^m, the conjugate is: Y_l^m* = (-1)^m * Y_l^{-m}

        However, when l,m are concrete, we expand and conjugate the explicit formula.
        """
        # If l,m are concrete, expand and conjugate the result
        expanded = self.expand()
        if expanded is not None:
            # Use the base class conjugate on the expanded form
            return expanded.conjugate()

        # For symbolic l,m, use the mathematical formula:
        # Y_l^m* = (-1)^m * Y_l^{-m}
        # But since we don't support negative m in our expansion,
        # we'll use SymPy's conjugate and wrap it
        from .core import SympyWrapper, _get_sympy
        sp = _get_sympy()
        sympy_ylm = self.to_sympy()
        conjugated = sp.conjugate(sympy_ylm)
        return SympyWrapper(conjugated)

    def to_sympy(self):
        """Convert to SymPy, with expansion if possible."""
        # Try to expand first
        expanded = self.expand()
        if expanded is not None:
            return expanded.to_sympy()

        # Fall back to Ynm
        sp = _get_sympy()
        sympy_args = [arg.to_sympy() for arg in self.args]
        return sp.Ynm(*sympy_args)


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
# ANGULAR MOMENTUM COUPLING COEFFICIENTS
# =============================================================================


class ClebschGordan(SpecialFunction):
    """
    Clebsch-Gordan coefficient ⟨j₁ m₁ j₂ m₂ | J M⟩.

    Coupling coefficients for angular momentum addition:
    |J M⟩ = Σ_{m₁,m₂} ⟨j₁ m₁ j₂ m₂ | J M⟩ |j₁ m₁⟩ |j₂ m₂⟩

    Selection rules:
    - |j₁ - j₂| ≤ J ≤ j₁ + j₂
    - m₁ + m₂ = M
    - |m₁| ≤ j₁, |m₂| ≤ j₂, |M| ≤ J

    Example:
        from cm import ClebschGordan
        # ⟨1/2, 1/2, 1/2, -1/2 | 1, 0⟩
        cg = ClebschGordan(Const(0.5), Const(0.5), Const(0.5), Const(-0.5), Const(1), Const(0))
        cg.render()
    """
    _name = "clebsch_gordan"
    _latex_name = "C"
    _sympy_func = "CG"

    def __init__(self, j1: Expr, m1: Expr, j2: Expr, m2: Expr, J: Expr, M: Expr):
        super().__init__(j1, m1, j2, m2, J, M)

    @property
    def j1(self) -> Expr:
        return self.args[0]

    @property
    def m1(self) -> Expr:
        return self.args[1]

    @property
    def j2(self) -> Expr:
        return self.args[2]

    @property
    def m2(self) -> Expr:
        return self.args[3]

    @property
    def J(self) -> Expr:
        return self.args[4]

    @property
    def M(self) -> Expr:
        return self.args[5]

    def to_sympy(self):
        import sympy as sp
        from sympy.physics.quantum.cg import CG
        return CG(
            self.j1.to_sympy(), self.m1.to_sympy(),
            self.j2.to_sympy(), self.m2.to_sympy(),
            self.J.to_sympy(), self.M.to_sympy()
        ).doit()

    def to_latex(self) -> str:
        return (f"\\langle {self.j1.to_latex()}, {self.m1.to_latex()}, "
                f"{self.j2.to_latex()}, {self.m2.to_latex()} | "
                f"{self.J.to_latex()}, {self.M.to_latex()} \\rangle")


class Wigner3j(SpecialFunction):
    """
    Wigner 3-j symbol.

    Related to Clebsch-Gordan coefficients:
    ⎛ j₁  j₂  j₃ ⎞
    ⎝ m₁  m₂  m₃ ⎠ = (-1)^(j₁-j₂-m₃) / √(2j₃+1) × ⟨j₁ m₁ j₂ m₂ | j₃ -m₃⟩

    Selection rules:
    - m₁ + m₂ + m₃ = 0
    - Triangle condition: |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂

    Example:
        w3j = Wigner3j(Const(1), Const(1), Const(2), Const(0), Const(0), Const(0))
        w3j.render()
    """
    _name = "wigner_3j"
    _latex_name = "3j"
    _sympy_func = "Wigner3j"

    def __init__(self, j1: Expr, j2: Expr, j3: Expr, m1: Expr, m2: Expr, m3: Expr):
        super().__init__(j1, j2, j3, m1, m2, m3)

    @property
    def j1(self) -> Expr:
        return self.args[0]

    @property
    def j2(self) -> Expr:
        return self.args[1]

    @property
    def j3(self) -> Expr:
        return self.args[2]

    @property
    def m1(self) -> Expr:
        return self.args[3]

    @property
    def m2(self) -> Expr:
        return self.args[4]

    @property
    def m3(self) -> Expr:
        return self.args[5]

    def to_sympy(self):
        import sympy as sp
        from sympy.physics.wigner import wigner_3j
        return wigner_3j(
            self.j1.to_sympy(), self.j2.to_sympy(), self.j3.to_sympy(),
            self.m1.to_sympy(), self.m2.to_sympy(), self.m3.to_sympy()
        )

    def to_latex(self) -> str:
        return (f"\\begin{{pmatrix}} {self.j1.to_latex()} & {self.j2.to_latex()} & {self.j3.to_latex()} \\\\ "
                f"{self.m1.to_latex()} & {self.m2.to_latex()} & {self.m3.to_latex()} \\end{{pmatrix}}")


class Wigner6j(SpecialFunction):
    """
    Wigner 6-j symbol (Racah W coefficient).

    Recoupling coefficient for three angular momenta:
    ⎧ j₁  j₂  j₃ ⎫
    ⎩ j₄  j₅  j₆ ⎭

    Used in angular momentum recoupling and reduced matrix elements.

    Example:
        w6j = Wigner6j(Const(1), Const(1), Const(2), Const(1), Const(2), Const(1))
        w6j.render()
    """
    _name = "wigner_6j"
    _latex_name = "6j"
    _sympy_func = "Wigner6j"

    def __init__(self, j1: Expr, j2: Expr, j3: Expr, j4: Expr, j5: Expr, j6: Expr):
        super().__init__(j1, j2, j3, j4, j5, j6)

    @property
    def j1(self) -> Expr:
        return self.args[0]

    @property
    def j2(self) -> Expr:
        return self.args[1]

    @property
    def j3(self) -> Expr:
        return self.args[2]

    @property
    def j4(self) -> Expr:
        return self.args[3]

    @property
    def j5(self) -> Expr:
        return self.args[4]

    @property
    def j6(self) -> Expr:
        return self.args[5]

    def to_sympy(self):
        import sympy as sp
        from sympy.physics.wigner import wigner_6j
        return wigner_6j(
            self.j1.to_sympy(), self.j2.to_sympy(), self.j3.to_sympy(),
            self.j4.to_sympy(), self.j5.to_sympy(), self.j6.to_sympy()
        )

    def to_latex(self) -> str:
        return (f"\\begin{{Bmatrix}} {self.j1.to_latex()} & {self.j2.to_latex()} & {self.j3.to_latex()} \\\\ "
                f"{self.j4.to_latex()} & {self.j5.to_latex()} & {self.j6.to_latex()} \\end{{Bmatrix}}")


class Wigner9j(SpecialFunction):
    """
    Wigner 9-j symbol.

    Recoupling coefficient for four angular momenta:
    ⎧ j₁  j₂  j₃ ⎫
    ⎨ j₄  j₅  j₆ ⎬
    ⎩ j₇  j₈  j₉ ⎭

    Example:
        w9j = Wigner9j(*[Const(1) for _ in range(9)])
        w9j.render()
    """
    _name = "wigner_9j"
    _latex_name = "9j"
    _sympy_func = "Wigner9j"

    def __init__(self, j1: Expr, j2: Expr, j3: Expr,
                 j4: Expr, j5: Expr, j6: Expr,
                 j7: Expr, j8: Expr, j9: Expr):
        super().__init__(j1, j2, j3, j4, j5, j6, j7, j8, j9)

    def to_sympy(self):
        import sympy as sp
        from sympy.physics.wigner import wigner_9j
        return wigner_9j(*[arg.to_sympy() for arg in self.args])

    def to_latex(self) -> str:
        j = [arg.to_latex() for arg in self.args]
        return (f"\\begin{{Bmatrix}} {j[0]} & {j[1]} & {j[2]} \\\\ "
                f"{j[3]} & {j[4]} & {j[5]} \\\\ "
                f"{j[6]} & {j[7]} & {j[8]} \\end{{Bmatrix}}")


