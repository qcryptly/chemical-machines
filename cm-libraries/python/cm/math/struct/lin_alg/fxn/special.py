"""
Special mathematical functions built from cm.math primitives.

Functions:
    assoc_laguerre(n, k, x)              - Associated Laguerre polynomial L_n^k(x)
    assoc_legendre(l, m, x)              - Associated Legendre polynomial P_l^m(x)
    spherical_harmonic(l, m, theta, phi) - Spherical harmonic Y_l^m(θ, φ)
    radial(n, l, Z, r, a0)              - Hydrogen-like radial wavefunction R_{nl}(r)
"""

from __future__ import annotations
import math as _math

from ...base import Expression, ScalarExpr, _ensure_expression
from .ops import sin, cos, exp, sqrt


def assoc_laguerre(n, k, x):
    """Associated (generalized) Laguerre polynomial L_n^k(x).

    L_n^k(x) = sum_{m=0}^{n} (-1)^m * C(n+k, n-m) / m! * x^m

    Args:
        n: Non-negative integer (degree).
        k: Non-negative integer (order).
        x: Symbolic Expression or numeric value.

    Returns:
        Symbolic expression for L_n^k(x).
    """
    if not isinstance(x, Expression):
        x = _ensure_expression(x, None)

    result = None
    for m in range(n + 1):
        coeff = ((-1)**m * _math.comb(n + k, n - m)
                 / _math.factorial(m))
        if abs(coeff) < 1e-15:
            continue
        if m == 0:
            term = _ensure_expression(coeff, x.structure)
        else:
            term = coeff * x**m
        result = term if result is None else result + term
    return result if result is not None else ScalarExpr(0, x.structure)


def assoc_legendre(l, m, x):
    """Associated Legendre polynomial P_l^m(x) (Condon-Shortley phase).

    P_l^m(x) = (-1)^m (1-x^2)^{m/2} d^m/dx^m [P_l(x)]

    Args:
        l: Non-negative integer (degree).
        m: Integer with |m| <= l (order).
        x: Symbolic Expression or numeric value.

    Returns:
        Symbolic expression for P_l^m(x).
    """
    if not isinstance(x, Expression):
        x = _ensure_expression(x, None)

    abs_m = abs(m)
    if abs_m > l:
        return ScalarExpr(0, x.structure)

    # d^|m|/dx^|m| [P_l(x)] via explicit Legendre polynomial coefficients.
    # P_l(x) = sum_k (-1)^k (2l-2k)! / (2^l k! (l-k)! (l-2k)!) x^{l-2k}
    deriv_sum = None
    for k in range(l // 2 + 1):
        p = l - 2 * k
        if p < abs_m:
            continue
        c_k = ((-1)**k * _math.factorial(2 * l - 2 * k)
               / (2**l * _math.factorial(k)
                  * _math.factorial(l - k)
                  * _math.factorial(l - 2 * k)))
        deriv_coeff = c_k * _math.factorial(p) / _math.factorial(p - abs_m)
        new_power = p - abs_m
        if abs(deriv_coeff) < 1e-15:
            continue
        if new_power == 0:
            term = _ensure_expression(deriv_coeff, x.structure)
        else:
            term = deriv_coeff * x**new_power
        deriv_sum = term if deriv_sum is None else deriv_sum + term

    if deriv_sum is None:
        return ScalarExpr(0, x.structure)

    # Multiply by (-1)^|m| * (1 - x^2)^{|m|/2}
    if abs_m == 0:
        result = deriv_sum
    else:
        phase = (-1)**abs_m
        one_minus_x2 = 1 - x**2
        if abs_m % 2 == 0:
            result = phase * one_minus_x2**(abs_m // 2) * deriv_sum
        else:
            half = (abs_m - 1) // 2
            sqrt_factor = sqrt(one_minus_x2)
            if half == 0:
                result = phase * sqrt_factor * deriv_sum
            else:
                result = phase * one_minus_x2**half * sqrt_factor * deriv_sum

    # Negative m: P_l^{-|m|}(x) = (-1)^|m| (l-|m|)!/(l+|m|)! P_l^{|m|}(x)
    if m < 0:
        factor = ((-1)**abs_m
                  * _math.factorial(l - abs_m)
                  / _math.factorial(l + abs_m))
        result = factor * result

    return result


def spherical_harmonic(l, m, theta, phi):
    """Complex spherical harmonic Y_l^m(theta, phi).

    Uses the Condon-Shortley phase convention:
        Y_l^m = sqrt((2l+1)/(4pi) (l-|m|)!/(l+|m|)!) P_l^|m|(cos theta) e^{i m phi}

    Args:
        l:     Non-negative integer (degree).
        m:     Integer with |m| <= l (order).
        theta: Polar angle (symbolic Expression or numeric).
        phi:   Azimuthal angle (symbolic Expression or numeric).

    Returns:
        Symbolic expression for Y_l^m (complex-valued when m != 0).
    """
    if not isinstance(theta, Expression):
        theta = _ensure_expression(theta, None)
    if not isinstance(phi, Expression):
        phi = _ensure_expression(phi, theta.structure)

    abs_m = abs(m)

    # Normalization constant
    norm = _math.sqrt(
        (2 * l + 1) / (4 * _math.pi)
        * _math.factorial(l - abs_m)
        / _math.factorial(l + abs_m)
    )

    # Build P_l^|m|(cos theta) directly with sin/cos for clean expressions
    cos_theta = cos(theta)
    sin_theta = sin(theta)

    # d^|m|/dx^|m| [P_l(x)] evaluated at x = cos(theta)
    deriv_at_cos = None
    for k in range(l // 2 + 1):
        p = l - 2 * k
        if p < abs_m:
            continue
        c_k = ((-1)**k * _math.factorial(2 * l - 2 * k)
               / (2**l * _math.factorial(k)
                  * _math.factorial(l - k)
                  * _math.factorial(l - 2 * k)))
        deriv_coeff = c_k * _math.factorial(p) / _math.factorial(p - abs_m)
        new_power = p - abs_m
        if abs(deriv_coeff) < 1e-15:
            continue
        if new_power == 0:
            term = _ensure_expression(deriv_coeff, theta.structure)
        else:
            term = deriv_coeff * cos_theta**new_power
        deriv_at_cos = term if deriv_at_cos is None else deriv_at_cos + term

    if deriv_at_cos is None:
        return ScalarExpr(0, theta.structure)

    # P_l^|m|(cos theta) = (-1)^|m| sin^|m|(theta) * deriv_at_cos
    if abs_m == 0:
        p_lm = deriv_at_cos
    else:
        phase = (-1)**abs_m
        p_lm = phase * sin_theta**abs_m * deriv_at_cos

    # Assemble Y_l^m
    if m == 0:
        return norm * p_lm
    elif m > 0:
        real_part = norm * p_lm * cos(m * phi)
        imag_part = norm * p_lm * sin(m * phi)
        return real_part + 1j * imag_part
    else:
        # Y_l^{-|m|} = (-1)^|m| conj(Y_l^|m|)
        sign = (-1)**abs_m
        real_part = sign * norm * p_lm * cos(abs_m * phi)
        imag_part = sign * norm * p_lm * sin(abs_m * phi)
        return real_part + (-1j) * imag_part


def radial(n, l, Z, r, a0):
    """Hydrogen-like radial wavefunction R_{nl}(r).

    R_{nl}(r) = N * e^{-Zr/(n a0)} * rho^l * L_{n-l-1}^{2l+1}(rho)
    where rho = 2Zr/(n a0) and N is the normalization constant.

    Args:
        n:  Principal quantum number (positive integer, n >= 1).
        l:  Angular momentum quantum number (0 <= l < n).
        Z:  Nuclear charge (symbolic Expression or numeric).
        r:  Radial distance (symbolic Expression or numeric).
        a0: Bohr radius (symbolic Expression or numeric).

    Returns:
        Symbolic expression for R_{nl}(r).
    """
    # Find structure from any Expression argument
    struct = None
    for v in [Z, r, a0]:
        if isinstance(v, Expression):
            struct = v.structure
            break
    if not isinstance(r, Expression):
        r = _ensure_expression(r, struct)
    if not isinstance(Z, Expression):
        Z = _ensure_expression(Z, r.structure)
    if not isinstance(a0, Expression):
        a0 = _ensure_expression(a0, r.structure)

    # rho = 2Zr / (n * a0)
    rho = (2 * Z * r) / (n * a0)

    # Pure numeric part of normalization:
    # sqrt((2/n)^3 * (n-l-1)! / (2n * (n+l)!))
    norm_num = _math.sqrt(
        (2.0 / n)**3
        * _math.factorial(n - l - 1)
        / (2.0 * n * _math.factorial(n + l))
    )

    # Symbolic (Z/a0)^{3/2}
    z_a0 = Z / a0
    norm = norm_num * z_a0 * sqrt(z_a0)

    # Exponential decay: e^{-Zr/(n*a0)}
    exp_part = exp((-1 * Z * r) / (n * a0))

    # Associated Laguerre: L_{n-l-1}^{2l+1}(rho)
    lag_part = assoc_laguerre(n - l - 1, 2 * l + 1, rho)

    # Assemble: N * e^{...} * rho^l * L(rho)
    result = norm * exp_part
    if l > 0:
        result = result * rho**l
    result = result * lag_part
    return result
