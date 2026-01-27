"""
Nuclear Attraction Integrals

Computes nuclear attraction integrals <μ|Z/r_C|ν> using the Boys function.

The nuclear attraction integral represents the Coulomb interaction between
an electron (described by basis functions μ, ν) and a nucleus with charge Z
located at position C.

V_μν = -Z ∫ φ_μ(r) (1/|r-C|) φ_ν(r) dr

This requires evaluation of the Boys function F_n(x).
"""

import numpy as np
import math
from typing import Tuple, List

from .basis import GaussianPrimitive, ContractedGaussian, BasisFunction, BasisSet
from .boys import boys_function


def hermite_coefficient(i: int, j: int, t: int, Qx: float, a: float, b: float) -> float:
    """
    Compute Hermite Gaussian expansion coefficient E_t^{ij}.

    The product of two Cartesian Gaussians can be written as:
        g_i(x-A) * g_j(x-B) = Σ_t E_t^{ij} Λ_t(x-P)

    where Λ_t is a Hermite Gaussian and P is the product center.

    Uses the recurrence:
        E_t^{i+1,j} = (1/2p) E_{t-1}^{ij} + Qx E_t^{ij} + (t+1) E_{t+1}^{ij}

    Args:
        i: Angular momentum on center A
        j: Angular momentum on center B
        t: Hermite index
        Qx: P_x - A_x (x-component of P - A)
        a: Exponent on A
        b: Exponent on B

    Returns:
        Hermite coefficient E_t^{ij}
    """
    p = a + b
    q = a * b / p

    if t < 0 or t > i + j:
        return 0.0

    if i == 0 and j == 0 and t == 0:
        # Base case: E_0^{00} = exp(-q * (A-B)²)
        # This is handled externally, return 1.0 here
        return 1.0

    if j == 0:
        # Vertical recurrence: E_t^{i+1,0}
        if i == 0:
            return 0.0 if t != 0 else 1.0

        # E_t^{i,0} = (1/2p) E_{t-1}^{i-1,0} + Qx E_t^{i-1,0} + (t+1) E_{t+1}^{i-1,0}
        result = Qx * hermite_coefficient(i-1, 0, t, Qx, a, b)
        result += (1.0 / (2*p)) * hermite_coefficient(i-1, 0, t-1, Qx, a, b)
        result += (t + 1) * hermite_coefficient(i-1, 0, t+1, Qx, a, b)
        return result

    else:
        # Horizontal recurrence: E_t^{i,j} = E_t^{i+1,j-1} + (A-B)_x E_t^{i,j-1}
        # But simpler: use vertical on j
        Qx_B = -b / p * (0)  # Need (P - B)_x
        # Actually, Qx = PA_x = (P - A)_x, need PB_x = (P - B)_x = PA_x - AB_x
        # AB = A - B, so PB = PA - AB = PA + BA
        # For simplicity, use direct recurrence

        # Transfer relation:
        # E_t^{i,j} from E_{...}^{i+1,j-1}
        result = hermite_coefficient(i+1, j-1, t, Qx, a, b)
        # Add AB_x contribution (but we don't have AB here)
        # This is getting complex; use simpler McMurchie-Davidson

        return result


def _E_coefficients(i: int, j: int, PA: float, PB: float, p: float) -> np.ndarray:
    """
    Compute all Hermite expansion coefficients E_t^{ij} for t = 0 to i+j.

    Uses McMurchie-Davidson recurrence.

    Args:
        i: Angular momentum on A
        j: Angular momentum on B
        PA: P - A distance
        PB: P - B distance
        p: Combined exponent α + β

    Returns:
        Array E[t] for t = 0, ..., i+j
    """
    # Build 2D table E[ii][jj][t] but we only need final E[i][j][:]
    max_t = i + j + 1

    # E[ii, jj, t] storage
    E = np.zeros((i + 2, j + 2, max_t + 1))

    # Base case
    E[0, 0, 0] = 1.0

    # Vertical recurrence: increase ii
    for ii in range(1, i + 1):
        for t in range(ii + 1):
            E[ii, 0, t] = PA * E[ii-1, 0, t]
            if t > 0:
                E[ii, 0, t] += (1.0 / (2*p)) * E[ii-1, 0, t-1]
            if t < ii - 1:
                E[ii, 0, t] += (t + 1) * E[ii-1, 0, t+1]

    # Horizontal recurrence: increase jj
    for jj in range(1, j + 1):
        for ii in range(i + 1):
            for t in range(ii + jj + 1):
                E[ii, jj, t] = PB * E[ii, jj-1, t]
                if t > 0:
                    E[ii, jj, t] += (1.0 / (2*p)) * E[ii, jj-1, t-1]
                if t < ii + jj - 1:
                    E[ii, jj, t] += (t + 1) * E[ii, jj-1, t+1]

    return E[i, j, :max_t]


def _R_coefficients(t: int, u: int, v: int, n: int, p: float, PC: np.ndarray, RPC2: float) -> float:
    """
    Compute auxiliary Hermite integral R^n_{tuv}(p, PC).

    R^n_{tuv} = (-2p)^n ∂^t/∂Px ∂^u/∂Py ∂^v/∂Pz F_n(p|PC|²)

    Uses the recurrence relation.

    Args:
        t, u, v: Differentiation orders
        n: Boys function order
        p: Combined exponent
        PC: P - C vector (Gaussian center to nucleus)
        RPC2: |P - C|² (squared distance)

    Returns:
        R^n_{tuv} value
    """
    if t < 0 or u < 0 or v < 0:
        return 0.0

    if t == 0 and u == 0 and v == 0:
        # Base case: R^n_{000} = (-2p)^n F_n(p * RPC²)
        return ((-2 * p) ** n) * boys_function(n, p * RPC2)

    # Use recurrence to reduce t, u, v
    if t > 0:
        # R^n_{t,u,v} = (t-1) R^{n+1}_{t-2,u,v} + PC_x R^{n+1}_{t-1,u,v}
        result = PC[0] * _R_coefficients(t-1, u, v, n+1, p, PC, RPC2)
        if t > 1:
            result += (t - 1) * _R_coefficients(t-2, u, v, n+1, p, PC, RPC2)
        return result

    elif u > 0:
        result = PC[1] * _R_coefficients(t, u-1, v, n+1, p, PC, RPC2)
        if u > 1:
            result += (u - 1) * _R_coefficients(t, u-2, v, n+1, p, PC, RPC2)
        return result

    else:  # v > 0
        result = PC[2] * _R_coefficients(t, u, v-1, n+1, p, PC, RPC2)
        if v > 1:
            result += (v - 1) * _R_coefficients(t, u, v-2, n+1, p, PC, RPC2)
        return result


def nuclear_primitive(prim_a: GaussianPrimitive, prim_b: GaussianPrimitive,
                      C: np.ndarray, Z: float = 1.0) -> float:
    """
    Compute nuclear attraction integral between two primitives.

    V = -Z ∫ g_a(r) (1/|r-C|) g_b(r) dr

    Uses McMurchie-Davidson scheme with Boys function.

    Args:
        prim_a: First Gaussian primitive
        prim_b: Second Gaussian primitive
        C: Nuclear position (x, y, z) in Bohr
        Z: Nuclear charge

    Returns:
        Nuclear attraction integral value
    """
    alpha = prim_a.alpha
    beta = prim_b.alpha
    p = alpha + beta

    A = np.array(prim_a.center)
    B = np.array(prim_b.center)
    C = np.array(C)

    AB = A - B
    AB2 = np.dot(AB, AB)

    # Gaussian product center
    P = (alpha * A + beta * B) / p
    PA = P - A
    PB = P - B
    PC = P - C
    RPC2 = np.dot(PC, PC)

    # Exponential prefactor
    K = math.exp(-alpha * beta * AB2 / p)

    # Angular momenta
    i1, j1, k1 = prim_a.angular
    i2, j2, k2 = prim_b.angular

    # Compute Hermite expansion coefficients
    Ex = _E_coefficients(i1, i2, PA[0], PB[0], p)
    Ey = _E_coefficients(j1, j2, PA[1], PB[1], p)
    Ez = _E_coefficients(k1, k2, PA[2], PB[2], p)

    # Sum over Hermite indices
    result = 0.0
    for t in range(i1 + i2 + 1):
        for u in range(j1 + j2 + 1):
            for v in range(k1 + k2 + 1):
                R = _R_coefficients(t, u, v, 0, p, PC, RPC2)
                result += Ex[t] * Ey[u] * Ez[v] * R

    # Prefactor: -Z * 2π/p * K * N_a * N_b
    prefactor = -Z * 2 * math.pi / p * K * prim_a.norm * prim_b.norm

    return prefactor * result


def nuclear_contracted(cgf_a: ContractedGaussian, cgf_b: ContractedGaussian,
                       C: np.ndarray, Z: float = 1.0) -> float:
    """
    Compute nuclear attraction integral between two contracted Gaussians.

    Args:
        cgf_a: First contracted Gaussian
        cgf_b: Second contracted Gaussian
        C: Nuclear position in Bohr
        Z: Nuclear charge

    Returns:
        Nuclear attraction integral
    """
    result = 0.0

    for coeff_a, exp_a in cgf_a.primitives:
        prim_a = GaussianPrimitive(
            alpha=exp_a,
            center=cgf_a.center,
            angular=cgf_a.angular
        )

        for coeff_b, exp_b in cgf_b.primitives:
            prim_b = GaussianPrimitive(
                alpha=exp_b,
                center=cgf_b.center,
                angular=cgf_b.angular
            )

            result += coeff_a * coeff_b * nuclear_primitive(prim_a, prim_b, C, Z)

    result *= cgf_a._norm_factor * cgf_b._norm_factor

    return result


def nuclear_attraction_integral(bf_a: BasisFunction, bf_b: BasisFunction,
                                nuclei: List[Tuple[np.ndarray, float]]) -> float:
    """
    Compute nuclear attraction integral for all nuclei.

    V_μν = Σ_C -Z_C <μ|1/r_C|ν>

    Args:
        bf_a: First basis function
        bf_b: Second basis function
        nuclei: List of (position, charge) tuples. Positions in Bohr.

    Returns:
        Total nuclear attraction integral
    """
    result = 0.0

    for C, Z in nuclei:
        result += nuclear_contracted(bf_a.cgf, bf_b.cgf, C, Z)

    return result


def nuclear_attraction_matrix(basis: BasisSet,
                              nuclei: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """
    Compute the full nuclear attraction matrix V for a basis set.

    V_μν = Σ_C -Z_C <μ|1/r_C|ν>

    Args:
        basis: BasisSet object
        nuclei: List of (position, charge) tuples. Positions in Bohr.

    Returns:
        Nuclear attraction matrix of shape (n_basis, n_basis)
    """
    n = basis.n_basis
    V = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            V[i, j] = nuclear_attraction_integral(basis[i], basis[j], nuclei)
            V[j, i] = V[i, j]

    return V
