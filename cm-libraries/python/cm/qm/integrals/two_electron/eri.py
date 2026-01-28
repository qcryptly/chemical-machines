"""
Two-Electron Repulsion Integrals (ERI)

Computes electron repulsion integrals (μν|λσ) using the Boys function
and McMurchie-Davidson/Obara-Saika scheme.

The ERI represents the Coulomb repulsion between two electrons:
    (μν|λσ) = ∫∫ φ_μ(r₁) φ_ν(r₁) (1/r₁₂) φ_λ(r₂) φ_σ(r₂) dr₁ dr₂

This is the most computationally expensive part of ab initio calculations,
scaling as O(N⁴) with the number of basis functions.
"""

import numpy as np
import math
from typing import Tuple, List

from ..basis import GaussianPrimitive, ContractedGaussian, BasisFunction, BasisSet
from ..utils.boys import boys_function
from ..one_electron.nuclear import _E_coefficients


def _R_coefficients_eri(t: int, u: int, v: int, n: int, p: float,
                        alpha: float, PQ: np.ndarray, RPQ2: float) -> float:
    """
    Compute auxiliary Hermite integral R^n_{tuv} for ERI.

    R^n_{tuv} appears in the ERI evaluation and involves the Boys function.

    Args:
        t, u, v: Differentiation orders
        n: Boys function order
        p: Combined exponent (p + q for ERI)
        alpha: Prefactor for recursion
        PQ: P - Q vector (between Gaussian product centers)
        RPQ2: |P - Q|² (squared distance)

    Returns:
        R^n_{tuv} value
    """
    if t < 0 or u < 0 or v < 0:
        return 0.0

    if t == 0 and u == 0 and v == 0:
        # Base case
        return ((-2 * alpha) ** n) * boys_function(n, alpha * RPQ2)

    # Recurrence relations (same structure as nuclear attraction)
    if t > 0:
        result = PQ[0] * _R_coefficients_eri(t-1, u, v, n+1, p, alpha, PQ, RPQ2)
        if t > 1:
            result += (t - 1) * _R_coefficients_eri(t-2, u, v, n+1, p, alpha, PQ, RPQ2)
        return result

    elif u > 0:
        result = PQ[1] * _R_coefficients_eri(t, u-1, v, n+1, p, alpha, PQ, RPQ2)
        if u > 1:
            result += (u - 1) * _R_coefficients_eri(t, u-2, v, n+1, p, alpha, PQ, RPQ2)
        return result

    else:  # v > 0
        result = PQ[2] * _R_coefficients_eri(t, u, v-1, n+1, p, alpha, PQ, RPQ2)
        if v > 1:
            result += (v - 1) * _R_coefficients_eri(t, u, v-2, n+1, p, alpha, PQ, RPQ2)
        return result


def eri_primitive(prim_a: GaussianPrimitive, prim_b: GaussianPrimitive,
                  prim_c: GaussianPrimitive, prim_d: GaussianPrimitive) -> float:
    """
    Compute two-electron repulsion integral between four primitives.

    (ab|cd) = ∫∫ g_a(r₁) g_b(r₁) (1/r₁₂) g_c(r₂) g_d(r₂) dr₁ dr₂

    Uses the McMurchie-Davidson scheme:
    1. Form Gaussian products: g_a * g_b -> centered at P
                               g_c * g_d -> centered at Q
    2. Expand in Hermite Gaussians
    3. Evaluate using Boys function

    Args:
        prim_a, prim_b: Primitives for electron 1
        prim_c, prim_d: Primitives for electron 2

    Returns:
        Two-electron integral value
    """
    # Exponents
    a = prim_a.alpha
    b = prim_b.alpha
    c = prim_c.alpha
    d = prim_d.alpha

    p = a + b  # Combined exponent for (ab)
    q = c + d  # Combined exponent for (cd)
    alpha = p * q / (p + q)  # Reduced exponent

    # Centers
    A = np.array(prim_a.center)
    B = np.array(prim_b.center)
    C = np.array(prim_c.center)
    D = np.array(prim_d.center)

    # Distances
    AB = A - B
    CD = C - D
    AB2 = np.dot(AB, AB)
    CD2 = np.dot(CD, CD)

    # Gaussian product centers
    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    PQ = P - Q
    RPQ2 = np.dot(PQ, PQ)

    PA = P - A
    PB = P - B
    QC = Q - C
    QD = Q - D

    # Exponential prefactors
    K_AB = math.exp(-a * b * AB2 / p)
    K_CD = math.exp(-c * d * CD2 / q)

    # Angular momenta
    i1, j1, k1 = prim_a.angular
    i2, j2, k2 = prim_b.angular
    i3, j3, k3 = prim_c.angular
    i4, j4, k4 = prim_d.angular

    # Hermite expansion coefficients for (ab) pair
    Ex_ab = _E_coefficients(i1, i2, PA[0], PB[0], p)
    Ey_ab = _E_coefficients(j1, j2, PA[1], PB[1], p)
    Ez_ab = _E_coefficients(k1, k2, PA[2], PB[2], p)

    # Hermite expansion coefficients for (cd) pair
    Ex_cd = _E_coefficients(i3, i4, QC[0], QD[0], q)
    Ey_cd = _E_coefficients(j3, j4, QC[1], QD[1], q)
    Ez_cd = _E_coefficients(k3, k4, QC[2], QD[2], q)

    # Sum over all Hermite indices
    result = 0.0

    for t1 in range(i1 + i2 + 1):
        for u1 in range(j1 + j2 + 1):
            for v1 in range(k1 + k2 + 1):
                for t2 in range(i3 + i4 + 1):
                    for u2 in range(j3 + j4 + 1):
                        for v2 in range(k3 + k4 + 1):
                            # Combined Hermite index
                            t = t1 + t2
                            u = u1 + u2
                            v = v1 + v2

                            # Sign factor: (-1)^(t2+u2+v2)
                            sign = (-1) ** (t2 + u2 + v2)

                            # Hermite coefficients product
                            E_prod = (Ex_ab[t1] * Ey_ab[u1] * Ez_ab[v1] *
                                     Ex_cd[t2] * Ey_cd[u2] * Ez_cd[v2])

                            # R integral
                            R = _R_coefficients_eri(t, u, v, 0, p + q, alpha, PQ, RPQ2)

                            result += sign * E_prod * R

    # Prefactor: 2π^(5/2) / (pq√(p+q)) * K_AB * K_CD * normalization
    prefactor = (2 * math.pi ** 2.5 / (p * q * math.sqrt(p + q)) *
                K_AB * K_CD *
                prim_a.norm * prim_b.norm * prim_c.norm * prim_d.norm)

    return prefactor * result


def eri_contracted(cgf_a: ContractedGaussian, cgf_b: ContractedGaussian,
                   cgf_c: ContractedGaussian, cgf_d: ContractedGaussian) -> float:
    """
    Compute two-electron integral between four contracted Gaussians.

    (ab|cd) = Σ_{pqrs} c_p c_q c_r c_s (p q | r s)

    Args:
        cgf_a, cgf_b: Contracted Gaussians for electron 1
        cgf_c, cgf_d: Contracted Gaussians for electron 2

    Returns:
        Two-electron integral
    """
    result = 0.0

    for coeff_a, exp_a in cgf_a.primitives:
        prim_a = GaussianPrimitive(exp_a, cgf_a.center, cgf_a.angular)

        for coeff_b, exp_b in cgf_b.primitives:
            prim_b = GaussianPrimitive(exp_b, cgf_b.center, cgf_b.angular)

            for coeff_c, exp_c in cgf_c.primitives:
                prim_c = GaussianPrimitive(exp_c, cgf_c.center, cgf_c.angular)

                for coeff_d, exp_d in cgf_d.primitives:
                    prim_d = GaussianPrimitive(exp_d, cgf_d.center, cgf_d.angular)

                    contrib = eri_primitive(prim_a, prim_b, prim_c, prim_d)
                    result += coeff_a * coeff_b * coeff_c * coeff_d * contrib

    # Apply contraction normalization
    result *= (cgf_a._norm_factor * cgf_b._norm_factor *
              cgf_c._norm_factor * cgf_d._norm_factor)

    return result


def electron_repulsion_integral(bf_a: BasisFunction, bf_b: BasisFunction,
                                 bf_c: BasisFunction, bf_d: BasisFunction) -> float:
    """
    Compute two-electron repulsion integral between four basis functions.

    (μν|λσ) = ∫∫ φ_μ(r₁) φ_ν(r₁) (1/r₁₂) φ_λ(r₂) φ_σ(r₂) dr₁ dr₂

    Args:
        bf_a, bf_b: Basis functions for electron 1 (bra)
        bf_c, bf_d: Basis functions for electron 2 (ket)

    Returns:
        Two-electron integral (μν|λσ)
    """
    return eri_contracted(bf_a.cgf, bf_b.cgf, bf_c.cgf, bf_d.cgf)


def eri_tensor(basis: BasisSet) -> np.ndarray:
    """
    Compute the full two-electron integral tensor.

    G_μνλσ = (μν|λσ)

    This is an O(N⁴) operation and can be very expensive.
    For large basis sets, consider computing on-the-fly or using
    density fitting approximations.

    Args:
        basis: BasisSet object

    Returns:
        4D tensor of shape (n_basis, n_basis, n_basis, n_basis)
    """
    n = basis.n_basis
    G = np.zeros((n, n, n, n))

    # Use 8-fold permutational symmetry
    # (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ)
    #         = (λσ|μν) = (σλ|μν) = (λσ|νμ) = (σλ|νμ)

    for i in range(n):
        for j in range(i + 1):
            for k in range(n):
                for l in range(k + 1):
                    # Only compute if (ij) >= (kl) in some ordering
                    ij = i * (i + 1) // 2 + j
                    kl = k * (k + 1) // 2 + l

                    if ij >= kl:
                        val = electron_repulsion_integral(
                            basis[i], basis[j], basis[k], basis[l]
                        )

                        # Apply 8-fold symmetry
                        G[i, j, k, l] = val
                        G[j, i, k, l] = val
                        G[i, j, l, k] = val
                        G[j, i, l, k] = val
                        G[k, l, i, j] = val
                        G[l, k, i, j] = val
                        G[k, l, j, i] = val
                        G[l, k, j, i] = val

    return G


def compute_J_matrix(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Compute the Coulomb matrix J from density matrix and ERIs.

    J_μν = Σ_{λσ} P_λσ (μν|λσ)

    Args:
        P: Density matrix of shape (n, n)
        G: ERI tensor of shape (n, n, n, n)

    Returns:
        Coulomb matrix J of shape (n, n)
    """
    return np.einsum('ls,ijls->ij', P, G)


def compute_K_matrix(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Compute the exchange matrix K from density matrix and ERIs.

    K_μν = Σ_{λσ} P_λσ (μλ|νσ)

    Args:
        P: Density matrix of shape (n, n)
        G: ERI tensor of shape (n, n, n, n)

    Returns:
        Exchange matrix K of shape (n, n)
    """
    return np.einsum('ls,iljs->ij', P, G)


def compute_G_matrix(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Compute the two-electron part of Fock matrix.

    G_μν = J_μν - 0.5 * K_μν = Σ_{λσ} P_λσ [(μν|λσ) - 0.5(μλ|νσ)]

    Args:
        P: Density matrix
        G: ERI tensor

    Returns:
        Two-electron Fock contribution
    """
    J = compute_J_matrix(P, G)
    K = compute_K_matrix(P, G)
    return J - 0.5 * K
