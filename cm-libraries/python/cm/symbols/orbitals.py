"""
Chemical Machines Symbols Orbitals Module

Atomic and molecular orbital functions including:
- Hydrogen-like wavefunctions
- Slater-type orbitals (STO)
- Gaussian-type orbitals (GTO)
"""

from typing import Optional, List, Union, Dict, Set, Tuple

from .core import Expr, Var, Const, _ensure_expr, _get_sympy
from .special import SpecialFunction, AssocLaguerre, SphericalHarmonic

__all__ = [
    'HydrogenRadial',
    'HydrogenOrbital',
    'SlaterTypeOrbital',
    'GaussianTypeOrbital',
    'ContractedGTO',
]


# =============================================================================
# HYDROGEN-LIKE WAVEFUNCTIONS
# =============================================================================


class HydrogenRadial(SpecialFunction):
    """
    Hydrogen-like radial wavefunction R_nl(r).

    R_nl(r) = N_nl × (2Zr/na₀)^l × e^(-Zr/na₀) × L_{n-l-1}^{2l+1}(2Zr/na₀)

    where:
    - N_nl is the normalization constant
    - Z is the nuclear charge
    - a₀ is the Bohr radius
    - L_n^α is the associated Laguerre polynomial

    Example:
        r = Math.var("r")
        R_10 = HydrogenRadial(n=1, l=0, r=r, Z=1)
        R_10.render()  # Displays 1s radial function
    """
    _name = "hydrogen_radial"
    _latex_name = "R"
    _sympy_func = None  # Custom implementation

    def __init__(self, n: Union[int, Expr], l: Union[int, Expr],
                 r: Expr, Z: Union[int, float, Expr] = 1,
                 a0: Union[float, Expr] = 1.0):
        """
        Args:
            n: Principal quantum number (n ≥ 1)
            l: Angular momentum quantum number (0 ≤ l < n)
            r: Radial coordinate
            Z: Nuclear charge (default 1 for hydrogen)
            a0: Bohr radius (default 1.0 for atomic units)
        """
        n_expr = _ensure_expr(n)
        l_expr = _ensure_expr(l)
        Z_expr = _ensure_expr(Z)
        a0_expr = _ensure_expr(a0)
        super().__init__(n_expr, l_expr, r, Z_expr, a0_expr)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def l(self) -> Expr:
        return self.args[1]

    @property
    def r(self) -> Expr:
        return self.args[2]

    @property
    def Z(self) -> Expr:
        return self.args[3]

    @property
    def a0(self) -> Expr:
        return self.args[4]

    def to_sympy(self):
        import sympy as sp
        from sympy import sqrt, factorial, exp
        from sympy.functions.special.polynomials import assoc_laguerre

        n = self.n.to_sympy()
        l = self.l.to_sympy()
        r = self.r.to_sympy()
        Z = self.Z.to_sympy()
        a0 = self.a0.to_sympy()

        # Dimensionless variable
        rho = 2 * Z * r / (n * a0)

        # Normalization constant
        norm = sqrt(
            ((2 * Z) / (n * a0))**3 *
            factorial(n - l - 1) / (2 * n * factorial(n + l))
        )

        # Radial wavefunction
        return norm * exp(-rho / 2) * rho**l * assoc_laguerre(n - l - 1, 2*l + 1, rho)

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        l_latex = self.l.to_latex()
        r_latex = self.r.to_latex()
        return f"R_{{{n_latex}{l_latex}}}\\left({r_latex}\\right)"


class HydrogenOrbital(SpecialFunction):
    """
    Complete hydrogen-like orbital ψ_nlm(r, θ, φ).

    ψ_nlm(r, θ, φ) = R_nl(r) × Y_l^m(θ, φ)

    where R_nl is the radial wavefunction and Y_l^m is the spherical harmonic.

    Example:
        r, theta, phi = Math.var("r"), Math.var("theta"), Math.var("phi")
        psi_210 = HydrogenOrbital(n=2, l=1, m=0, r=r, theta=theta, phi=phi)
        psi_210.render()  # Displays 2p_z orbital
    """
    _name = "hydrogen_orbital"
    _latex_name = r"\psi"
    _sympy_func = None

    def __init__(self, n: Union[int, Expr], l: Union[int, Expr], m: Union[int, Expr],
                 r: Expr, theta: Expr, phi: Expr,
                 Z: Union[int, float, Expr] = 1, a0: Union[float, Expr] = 1.0):
        n_expr = _ensure_expr(n)
        l_expr = _ensure_expr(l)
        m_expr = _ensure_expr(m)
        Z_expr = _ensure_expr(Z)
        a0_expr = _ensure_expr(a0)
        super().__init__(n_expr, l_expr, m_expr, r, theta, phi, Z_expr, a0_expr)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def l(self) -> Expr:
        return self.args[1]

    @property
    def m(self) -> Expr:
        return self.args[2]

    @property
    def r(self) -> Expr:
        return self.args[3]

    @property
    def theta(self) -> Expr:
        return self.args[4]

    @property
    def phi(self) -> Expr:
        return self.args[5]

    @property
    def Z(self) -> Expr:
        return self.args[6]

    @property
    def a0(self) -> Expr:
        return self.args[7]

    def radial_part(self) -> HydrogenRadial:
        """Return just the radial part R_nl(r)."""
        return HydrogenRadial(self.n, self.l, self.r, self.Z, self.a0)

    def angular_part(self) -> SphericalHarmonic:
        """Return just the angular part Y_l^m(θ, φ)."""
        return SphericalHarmonic(self.l, self.m, self.theta, self.phi)

    def to_sympy(self):
        radial = self.radial_part().to_sympy()
        angular = self.angular_part().to_sympy()
        return radial * angular

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        l_latex = self.l.to_latex()
        m_latex = self.m.to_latex()
        return f"\\psi_{{{n_latex}{l_latex}{m_latex}}}"


# =============================================================================
# SLATER-TYPE AND GAUSSIAN-TYPE ORBITALS
# =============================================================================


class SlaterTypeOrbital(SpecialFunction):
    """
    Slater-type orbital (STO).

    φ_STO(r, θ, φ) = N × r^(n-1) × e^(-ζr) × Y_l^m(θ, φ)

    where:
    - N is the normalization constant
    - ζ (zeta) is the orbital exponent
    - n is the principal quantum number
    - Y_l^m is the spherical harmonic

    STOs have correct nuclear cusp behavior but difficult two-center integrals.

    Example:
        r, theta, phi = Math.var("r"), Math.var("theta"), Math.var("phi")
        sto = SlaterTypeOrbital(n=1, l=0, m=0, zeta=1.0, r=r, theta=theta, phi=phi)
        sto.render()
    """
    _name = "slater_orbital"
    _latex_name = r"\chi"
    _sympy_func = None

    def __init__(self, n: Union[int, Expr], l: Union[int, Expr], m: Union[int, Expr],
                 zeta: Union[float, Expr], r: Expr, theta: Expr, phi: Expr):
        n_expr = _ensure_expr(n)
        l_expr = _ensure_expr(l)
        m_expr = _ensure_expr(m)
        zeta_expr = _ensure_expr(zeta)
        super().__init__(n_expr, l_expr, m_expr, zeta_expr, r, theta, phi)

    @property
    def n(self) -> Expr:
        return self.args[0]

    @property
    def l(self) -> Expr:
        return self.args[1]

    @property
    def m(self) -> Expr:
        return self.args[2]

    @property
    def zeta(self) -> Expr:
        return self.args[3]

    @property
    def r(self) -> Expr:
        return self.args[4]

    @property
    def theta(self) -> Expr:
        return self.args[5]

    @property
    def phi(self) -> Expr:
        return self.args[6]

    def to_sympy(self):
        import sympy as sp
        from sympy import sqrt, factorial, exp, pi

        n = self.n.to_sympy()
        l = self.l.to_sympy()
        m = self.m.to_sympy()
        zeta = self.zeta.to_sympy()
        r = self.r.to_sympy()
        theta = self.theta.to_sympy()
        phi = self.phi.to_sympy()

        # Normalization: N = (2ζ)^(n+1/2) / sqrt((2n)!)
        norm = (2 * zeta)**(n + sp.Rational(1, 2)) / sqrt(factorial(2 * n))

        # Radial part
        radial = r**(n - 1) * exp(-zeta * r)

        # Angular part (spherical harmonic)
        angular = sp.Ynm(l, m, theta, phi)

        return norm * radial * angular

    def to_latex(self) -> str:
        n_latex = self.n.to_latex()
        l_latex = self.l.to_latex()
        m_latex = self.m.to_latex()
        zeta_latex = self.zeta.to_latex()
        return f"\\chi_{{{n_latex}{l_latex}{m_latex}}}^{{\\zeta={zeta_latex}}}"


class GaussianTypeOrbital(SpecialFunction):
    """
    Gaussian-type orbital (GTO).

    Cartesian form: φ_GTO = N × x^i × y^j × z^k × e^(-αr²)

    where:
    - N is the normalization constant
    - α (alpha) is the orbital exponent
    - (i, j, k) are Cartesian angular momenta with l = i + j + k

    GTOs allow analytical two-center integrals but lack nuclear cusp.
    Typically used in contracted form (linear combinations).

    Example:
        x, y, z = Math.var("x"), Math.var("y"), Math.var("z")
        # s-type GTO (l=0)
        gto_s = GaussianTypeOrbital(i=0, j=0, k=0, alpha=1.0, x=x, y=y, z=z)
        # p_x GTO (l=1)
        gto_px = GaussianTypeOrbital(i=1, j=0, k=0, alpha=1.0, x=x, y=y, z=z)
    """
    _name = "gaussian_orbital"
    _latex_name = r"g"
    _sympy_func = None

    def __init__(self, i: Union[int, Expr], j: Union[int, Expr], k: Union[int, Expr],
                 alpha: Union[float, Expr], x: Expr, y: Expr, z: Expr):
        i_expr = _ensure_expr(i)
        j_expr = _ensure_expr(j)
        k_expr = _ensure_expr(k)
        alpha_expr = _ensure_expr(alpha)
        super().__init__(i_expr, j_expr, k_expr, alpha_expr, x, y, z)

    @property
    def i(self) -> Expr:
        return self.args[0]

    @property
    def j(self) -> Expr:
        return self.args[1]

    @property
    def k(self) -> Expr:
        return self.args[2]

    @property
    def alpha(self) -> Expr:
        return self.args[3]

    @property
    def x(self) -> Expr:
        return self.args[4]

    @property
    def y(self) -> Expr:
        return self.args[5]

    @property
    def z(self) -> Expr:
        return self.args[6]

    @property
    def l(self) -> Expr:
        """Total angular momentum l = i + j + k."""
        return self.i + self.j + self.k

    def to_sympy(self):
        import sympy as sp
        from sympy import sqrt, pi, exp, factorial2

        i = self.i.to_sympy()
        j = self.j.to_sympy()
        k = self.k.to_sympy()
        alpha = self.alpha.to_sympy()
        x = self.x.to_sympy()
        y = self.y.to_sympy()
        z = self.z.to_sympy()

        r_sq = x**2 + y**2 + z**2

        # Normalization constant for Cartesian GTO
        # N = (2α/π)^(3/4) × (4α)^((i+j+k)/2) / sqrt((2i-1)!!(2j-1)!!(2k-1)!!)
        norm = ((2 * alpha / pi)**(sp.Rational(3, 4)) *
                (4 * alpha)**((i + j + k) / 2) /
                sqrt(factorial2(2*i - 1) * factorial2(2*j - 1) * factorial2(2*k - 1)))

        return norm * x**i * y**j * z**k * exp(-alpha * r_sq)

    def to_latex(self) -> str:
        i_latex = self.i.to_latex()
        j_latex = self.j.to_latex()
        k_latex = self.k.to_latex()
        alpha_latex = self.alpha.to_latex()
        return f"g_{{{i_latex}{j_latex}{k_latex}}}^{{\\alpha={alpha_latex}}}"


class ContractedGTO(Expr):
    """
    Contracted Gaussian-type orbital.

    φ_CGTO = Σᵢ cᵢ × φ_GTO(αᵢ)

    A linear combination of primitive GTOs with different exponents
    but the same angular momentum.

    Example:
        x, y, z = Math.var("x"), Math.var("y"), Math.var("z")
        # STO-3G style contraction for 1s
        cgto = ContractedGTO(
            i=0, j=0, k=0,
            exponents=[3.42525, 0.62391, 0.16885],
            coefficients=[0.15433, 0.53533, 0.44463],
            x=x, y=y, z=z
        )
    """

    def __init__(self, i: int, j: int, k: int,
                 exponents: List[float], coefficients: List[float],
                 x: Expr, y: Expr, z: Expr):
        if len(exponents) != len(coefficients):
            raise ValueError("Number of exponents must match number of coefficients")

        self._i = i
        self._j = j
        self._k = k
        self._exponents = exponents
        self._coefficients = coefficients
        self._x = x
        self._y = y
        self._z = z

        # Build primitive GTOs
        self._primitives = [
            GaussianTypeOrbital(i, j, k, alpha, x, y, z)
            for alpha in exponents
        ]

    @property
    def n_primitives(self) -> int:
        return len(self._exponents)

    @property
    def primitives(self) -> List[GaussianTypeOrbital]:
        return self._primitives

    @property
    def coefficients(self) -> List[float]:
        return self._coefficients

    def to_sympy(self):
        import sympy as sp
        result = sp.Integer(0)
        for coeff, prim in zip(self._coefficients, self._primitives):
            result = result + coeff * prim.to_sympy()
        return result

    def to_latex(self) -> str:
        terms = []
        for coeff, prim in zip(self._coefficients, self._primitives):
            terms.append(f"{coeff:.4f} \\cdot {prim.to_latex()}")
        return " + ".join(terms)

    def _get_free_variables(self) -> Set['Var']:
        return self._x._get_free_variables() | self._y._get_free_variables() | self._z._get_free_variables()


