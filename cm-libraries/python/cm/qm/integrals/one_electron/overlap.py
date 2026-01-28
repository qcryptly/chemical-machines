"""
Overlap Integrals

Computes overlap integrals <μ|ν> between Gaussian basis functions
using the Obara-Saika recurrence relations.

The overlap integral between two Gaussian primitives centered at A and B:
    S = ∫ g_A(r) g_B(r) dr

For Cartesian Gaussians, this factors into a product of 1D integrals.
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from functools import lru_cache

from ..basis import GaussianPrimitive, ContractedGaussian, BasisFunction, BasisSet


def overlap_1d(i: int, j: int, PA: float, alpha: float, beta: float) -> float:
    """
    Compute 1D overlap integral using Obara-Saika recurrence.

    S_ij = ∫ (x-A)^i (x-B)^j exp(-α(x-A)²) exp(-β(x-B)²) dx

    The recurrence relations are:
        S_{i+1,j} = PA * S_{ij} + (1/2p) * (i * S_{i-1,j} + j * S_{i,j-1})
        S_{i,j+1} = PB * S_{ij} + (1/2p) * (i * S_{i-1,j} + j * S_{i,j-1})

    where:
        p = α + β
        P = (α*A + β*B) / p  (Gaussian product center)
        PA = P - A
        PB = P - B

    Args:
        i: Angular momentum on center A
        j: Angular momentum on center B
        PA: P - A (distance from product center to A)
        alpha: Exponent on center A
        beta: Exponent on center B

    Returns:
        Value of the 1D overlap integral (without prefactors)
    """
    p = alpha + beta

    # Build table of S_ij values using recurrence
    # We need S[0..i, 0..j]
    S = np.zeros((i + 1, j + 1))

    # Base case: S_00 = sqrt(π/p)
    S[0, 0] = math.sqrt(math.pi / p)

    # PB = PA - (A - B) but we compute it from the relation PA + PB = 0
    # Actually: PA = P - A, PB = P - B, where P = (αA + βB)/p
    # So PB = P - B = (αA + βB)/p - B = α(A-B)/p = -α * AB / p
    # And PA = P - A = (αA + βB)/p - A = -β(A-B)/p = β * AB / p
    # The relation is: PA = β/(α+β) * AB, PB = -α/(α+β) * AB
    # Given PA, we have AB = PA * p / β (if β > 0)
    # Then PB = -α * AB / p = -α * PA / β

    if abs(beta) > 1e-15:
        PB = -alpha * PA / beta
    else:
        PB = 0.0

    one_over_2p = 0.5 / p

    # Vertical recurrence: increase i
    for ii in range(1, i + 1):
        S[ii, 0] = PA * S[ii - 1, 0]
        if ii > 1:
            S[ii, 0] += (ii - 1) * one_over_2p * S[ii - 2, 0]

    # Horizontal recurrence: increase j
    for jj in range(1, j + 1):
        for ii in range(i + 1):
            S[ii, jj] = PB * S[ii, jj - 1]
            if ii > 0:
                S[ii, jj] += ii * one_over_2p * S[ii - 1, jj - 1]
            if jj > 1:
                S[ii, jj] += (jj - 1) * one_over_2p * S[ii, jj - 2]

    return S[i, j]


def overlap_primitive(prim_a: GaussianPrimitive, prim_b: GaussianPrimitive) -> float:
    """
    Compute overlap integral between two Gaussian primitives.

    S_ab = N_a N_b ∫ g_a(r) g_b(r) dr

    where N_a, N_b are normalization constants.

    Args:
        prim_a: First Gaussian primitive
        prim_b: Second Gaussian primitive

    Returns:
        Overlap integral value
    """
    alpha = prim_a.alpha
    beta = prim_b.alpha
    p = alpha + beta

    A = np.array(prim_a.center)
    B = np.array(prim_b.center)
    AB = A - B
    AB2 = np.dot(AB, AB)

    # Gaussian product center
    P = (alpha * A + beta * B) / p
    PA = P - A

    # Exponential prefactor from Gaussian product theorem
    K = math.exp(-alpha * beta * AB2 / p)

    # Compute 1D overlaps
    i1, j1, k1 = prim_a.angular
    i2, j2, k2 = prim_b.angular

    Sx = overlap_1d(i1, i2, PA[0], alpha, beta)
    Sy = overlap_1d(j1, j2, PA[1], alpha, beta)
    Sz = overlap_1d(k1, k2, PA[2], alpha, beta)

    return prim_a.norm * prim_b.norm * K * Sx * Sy * Sz


def overlap_contracted(cgf_a: ContractedGaussian, cgf_b: ContractedGaussian) -> float:
    """
    Compute overlap integral between two contracted Gaussians.

    S_ab = Σ_pq c_p c_q S_pq

    where S_pq is the primitive overlap.

    Args:
        cgf_a: First contracted Gaussian
        cgf_b: Second contracted Gaussian

    Returns:
        Overlap integral value
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

            result += coeff_a * coeff_b * overlap_primitive(prim_a, prim_b)

    # Apply contraction normalization factors
    result *= cgf_a._norm_factor * cgf_b._norm_factor

    return result


def overlap_integral(bf_a: BasisFunction, bf_b: BasisFunction) -> float:
    """
    Compute overlap integral between two basis functions.

    This is the main user-facing function.

    Args:
        bf_a: First basis function
        bf_b: Second basis function

    Returns:
        Overlap integral <μ|ν>
    """
    return overlap_contracted(bf_a.cgf, bf_b.cgf)


def overlap_matrix(basis: BasisSet) -> np.ndarray:
    """
    Compute the full overlap matrix S for a basis set.

    S_μν = <μ|ν>

    Args:
        basis: BasisSet object with all basis functions

    Returns:
        Overlap matrix of shape (n_basis, n_basis)
    """
    n = basis.n_basis
    S = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            S[i, j] = overlap_integral(basis[i], basis[j])
            S[j, i] = S[i, j]  # Symmetric

    return S
