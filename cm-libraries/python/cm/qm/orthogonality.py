"""
Orthogonality calculations for quantum mechanical functions.

This module provides utilities for computing orthogonality integrals
of special functions like spherical harmonics.
"""

from typing import Union, Tuple
import numpy as np
from scipy import integrate
from scipy.special import sph_harm_y


def spherical_harmonic_orthogonality(
    l: int, m: int, l_p: int, m_p: int,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, np.pi), (0, 2*np.pi))
) -> float:
    """
    Compute orthogonality integral for spherical harmonics.

    Evaluates:
        ∫∫ Y_l^m*(θ,φ) Y_l'^m'(θ,φ) sin(θ) dθ dφ

    This should return 1 for l==l' and m==m' (orthonormal), 0 otherwise.

    Args:
        l: First spherical harmonic degree
        m: First spherical harmonic order
        l_p: Second spherical harmonic degree
        m_p: Second spherical harmonic order
        bounds: Integration bounds as ((theta_min, theta_max), (phi_min, phi_max))

    Returns:
        The orthogonality integral value (1 if orthonormal, 0 if orthogonal)

    Example:
        >>> # Diagonal element (should be 1)
        >>> result = spherical_harmonic_orthogonality(1, 0, 1, 0)
        >>> print(f"{result:.6f}")  # ~1.000000

        >>> # Off-diagonal element (should be 0)
        >>> result = spherical_harmonic_orthogonality(1, 0, 2, 0)
        >>> print(f"{result:.6f}")  # ~0.000000
    """
    theta_bounds, phi_bounds = bounds

    def integrand(phi, theta):
        """Orthogonality integrand: Y_l^m* Y_l'^m' sin(theta)."""
        # sph_harm_y takes (l, m, theta, phi)
        Y1 = sph_harm_y(l, m, theta, phi)
        Y2 = sph_harm_y(l_p, m_p, theta, phi)
        # Include sin(theta) from spherical integration measure
        return np.conj(Y1) * Y2 * np.sin(theta)

    # Integrate over phi first, then theta
    result, error = integrate.dblquad(
        lambda phi, theta: integrand(phi, theta).real,
        theta_bounds[0], theta_bounds[1],  # theta bounds
        phi_bounds[0], phi_bounds[1]        # phi bounds
    )

    return result


__all__ = ['spherical_harmonic_orthogonality']
