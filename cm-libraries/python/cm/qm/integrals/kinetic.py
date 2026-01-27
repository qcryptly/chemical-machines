"""
Kinetic Energy Integrals

Computes kinetic energy integrals <μ|-½∇²|ν> between Gaussian basis functions.

The kinetic energy operator is:
    T = -½∇² = -½(∂²/∂x² + ∂²/∂y² + ∂²/∂z²)

For Cartesian Gaussians, the kinetic integral can be computed from
overlap integrals using:
    T_ij = -½ [∂²S/∂A_x² + ∂²S/∂A_y² + ∂²S/∂A_z²]

Or equivalently using the recurrence relation approach.
"""

import numpy as np
import math
from typing import Tuple

from .basis import GaussianPrimitive, ContractedGaussian, BasisFunction, BasisSet
from .overlap import overlap_1d


def kinetic_primitive(prim_a: GaussianPrimitive, prim_b: GaussianPrimitive) -> float:
    """
    Compute kinetic energy integral between two Gaussian primitives.

    T_ab = <a|-½∇²|b>

    Using the Obara-Saika scheme for kinetic integrals.

    For the second derivative: <i|d²/dx²|j> = j(j-1)S_{i,j-2} - 2β(2j+1)S_{i,j} + 4β²S_{i,j+2}

    The kinetic energy (with -½ factor) becomes:
    T_ij = -½ <i|d²/dx²|j> = β(2j+1)S_{i,j} - 2β²S_{i,j+2} - ½j(j-1)S_{i,j-2}

    For j=0 (s-orbitals): T_00 = β·S_00 - 2β²·S_02

    The 3D integral factorizes as:
    T = T_x·S_y·S_z + S_x·T_y·S_z + S_x·S_y·T_z

    Args:
        prim_a: First Gaussian primitive
        prim_b: Second Gaussian primitive

    Returns:
        Kinetic energy integral value
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

    # Exponential prefactor
    K = math.exp(-alpha * beta * AB2 / p)

    i1, j1, k1 = prim_a.angular
    i2, j2, k2 = prim_b.angular

    # Compute overlap integrals for each direction
    Sx = overlap_1d(i1, i2, PA[0], alpha, beta)
    Sy = overlap_1d(j1, j2, PA[1], alpha, beta)
    Sz = overlap_1d(k1, k2, PA[2], alpha, beta)

    def kinetic_1d(l1, l2, PA_x, a, b):
        """
        Compute 1D kinetic contribution T_ij = -½ <l1|d²/dx²|l2>.

        Using the formula derived from:
        <l1|d²/dx²|l2> = l2(l2-1)S_{l1,l2-2} - 2β(2l2+1)S_{l1,l2} + 4β²S_{l1,l2+2}

        So: T = -½ * [above] = β(2l2+1)S - 2β²S_{l2+2} - ½l2(l2-1)S_{l2-2}
        """
        S_l1l2 = overlap_1d(l1, l2, PA_x, a, b)
        S_l1l2p2 = overlap_1d(l1, l2 + 2, PA_x, a, b)

        # T = β(2l2+1)S - 2β²S_{l2+2} - ½l2(l2-1)S_{l2-2}
        result = b * (2 * l2 + 1) * S_l1l2 - 2 * b * b * S_l1l2p2

        if l2 >= 2:
            S_l1l2m2 = overlap_1d(l1, l2 - 2, PA_x, a, b)
            result -= 0.5 * l2 * (l2 - 1) * S_l1l2m2

        return result

    Tx = kinetic_1d(i1, i2, PA[0], alpha, beta)
    Ty = kinetic_1d(j1, j2, PA[1], alpha, beta)
    Tz = kinetic_1d(k1, k2, PA[2], alpha, beta)

    # Total kinetic: T = Tx·Sy·Sz + Sx·Ty·Sz + Sx·Sy·Tz
    # Note: The kinetic_1d function already includes the -½ factor
    T = Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz

    return prim_a.norm * prim_b.norm * K * T


def kinetic_contracted(cgf_a: ContractedGaussian, cgf_b: ContractedGaussian) -> float:
    """
    Compute kinetic energy integral between two contracted Gaussians.

    T_ab = Σ_pq c_p c_q T_pq

    Args:
        cgf_a: First contracted Gaussian
        cgf_b: Second contracted Gaussian

    Returns:
        Kinetic energy integral value
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

            result += coeff_a * coeff_b * kinetic_primitive(prim_a, prim_b)

    result *= cgf_a._norm_factor * cgf_b._norm_factor

    return result


def kinetic_integral(bf_a: BasisFunction, bf_b: BasisFunction) -> float:
    """
    Compute kinetic energy integral between two basis functions.

    Args:
        bf_a: First basis function
        bf_b: Second basis function

    Returns:
        Kinetic energy integral <μ|-½∇²|ν>
    """
    return kinetic_contracted(bf_a.cgf, bf_b.cgf)


def kinetic_matrix(basis: BasisSet) -> np.ndarray:
    """
    Compute the full kinetic energy matrix T for a basis set.

    T_μν = <μ|-½∇²|ν>

    Args:
        basis: BasisSet object

    Returns:
        Kinetic energy matrix of shape (n_basis, n_basis)
    """
    n = basis.n_basis
    T = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            T[i, j] = kinetic_integral(basis[i], basis[j])
            T[j, i] = T[i, j]

    return T
