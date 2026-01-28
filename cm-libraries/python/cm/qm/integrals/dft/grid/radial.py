"""
Radial Integration Grids

Implements Gauss-Chebyshev radial quadrature with Becke mapping
for atomic integration in DFT calculations.

Reference:
- Becke, J. Chem. Phys. 88, 2547 (1988)
- Mura & Knowles, J. Chem. Phys. 104, 9848 (1996)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Bragg-Slater radii in Bohr (atomic units)
# Used to scale radial grids for each element
BRAGG_SLATER_RADII_BOHR = {
    'H': 0.472,  'He': 0.472,
    'Li': 2.740, 'Be': 1.984, 'B': 1.606, 'C': 1.323, 'N': 1.228, 'O': 1.134, 'F': 0.945, 'Ne': 0.850,
    'Na': 3.402, 'Mg': 2.835, 'Al': 2.362, 'Si': 2.079, 'P': 1.890, 'S': 1.890, 'Cl': 1.890, 'Ar': 1.663,
    'K': 4.063,  'Ca': 3.496, 'Sc': 3.024, 'Ti': 2.646, 'V': 2.551, 'Cr': 2.646, 'Mn': 2.646,
    'Fe': 2.646, 'Co': 2.551, 'Ni': 2.551, 'Cu': 2.551, 'Zn': 2.551,
}


@dataclass
class RadialGrid:
    """
    Radial integration grid for atomic calculations.

    Attributes:
        points: Radial points in Bohr
        weights: Quadrature weights (includes r² Jacobian)
        n_points: Number of radial points
        element: Element symbol this grid is for
        r_bragg: Bragg-Slater radius used for scaling
    """
    points: np.ndarray
    weights: np.ndarray
    n_points: int
    element: str
    r_bragg: float

    @classmethod
    def for_element(cls, element: str, n_points: int = 75) -> 'RadialGrid':
        """
        Create a radial grid appropriate for a given element.

        Args:
            element: Element symbol
            n_points: Number of radial points

        Returns:
            RadialGrid scaled for the element
        """
        r_bragg = BRAGG_SLATER_RADII_BOHR.get(element, 1.5)
        points, weights = gauss_chebyshev_grid(n_points, r_bragg)
        return cls(
            points=points,
            weights=weights,
            n_points=n_points,
            element=element,
            r_bragg=r_bragg
        )


def gauss_chebyshev_grid(n_points: int, r_bragg: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gauss-Chebyshev type 2 radial grid with Becke mapping.

    Uses the transformation:
        r = R * (1 + x) / (1 - x)

    where x are Chebyshev nodes in [-1, 1] and R is the Bragg-Slater radius.

    Args:
        n_points: Number of radial points
        r_bragg: Bragg-Slater radius for scaling (in Bohr)

    Returns:
        Tuple of (radial_points, weights) where weights include r² Jacobian
    """
    # Chebyshev nodes: x_i = cos(π * i / (n+1)) for i = 1..n
    i = np.arange(1, n_points + 1)
    x = np.cos(np.pi * i / (n_points + 1))

    # Becke mapping: r = R * (1 + x) / (1 - x)
    r = r_bragg * (1 + x) / (1 - x)

    # Gauss-Chebyshev weights: w_i = π/(n+1) * sin²(πi/(n+1))
    sin_term = np.sin(np.pi * i / (n_points + 1)) ** 2
    w_chebyshev = np.pi / (n_points + 1) * sin_term

    # Jacobian of Becke mapping: dr/dx = 2R / (1-x)²
    jacobian = 2 * r_bragg / (1 - x) ** 2

    # Total weights including r² for spherical integration
    # ∫ f(r) 4πr² dr = Σ w_i * f(r_i) * 4π
    # We include r² in weights, caller adds 4π if needed
    weights = w_chebyshev * jacobian * r ** 2

    return r, weights


def mura_knowles_grid(n_points: int, r_bragg: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Mura-Knowles radial grid (Euler-Maclaurin based).

    Alternative to Gauss-Chebyshev, sometimes more efficient for
    slowly decaying integrands.

    Reference:
        Mura & Knowles, J. Chem. Phys. 104, 9848 (1996)

    Args:
        n_points: Number of radial points
        r_bragg: Bragg-Slater radius for scaling

    Returns:
        Tuple of (radial_points, weights)
    """
    # Mura-Knowles transformation: r = R * x² / (1 - x)³
    # where x = i / (n + 1)
    i = np.arange(1, n_points + 1)
    x = i / (n_points + 1)

    # Radial points
    r = r_bragg * x ** 2 / (1 - x) ** 3

    # Jacobian: dr/dx = R * x * (2 - 5x + 3x²) / (1 - x)⁴
    # Simplified: dr/dx = R * (2x(1-x) + 3x²) / (1-x)⁴
    jacobian = r_bragg * x * (2 + x) / (1 - x) ** 4

    # Weights (Euler-Maclaurin, simplified)
    dx = 1.0 / (n_points + 1)
    weights = dx * jacobian * r ** 2

    return r, weights


def treutler_ahlrichs_grid(n_points: int, r_bragg: float = 1.0,
                           alpha: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Treutler-Ahlrichs M4 radial grid.

    Uses logarithmic mapping which is often more efficient for
    molecular calculations.

    Reference:
        Treutler & Ahlrichs, J. Chem. Phys. 102, 346 (1995)

    Args:
        n_points: Number of radial points
        r_bragg: Bragg-Slater radius for scaling
        alpha: Mapping parameter (default 0.6)

    Returns:
        Tuple of (radial_points, weights)
    """
    # Gauss-Chebyshev nodes
    i = np.arange(1, n_points + 1)
    x = np.cos(np.pi * i / (n_points + 1))

    # Treutler-Ahlrichs M4 mapping
    # r = -R/ln(2) * (1+x)^alpha * ln((1-x)/2)
    ln_term = np.log((1 - x) / 2)
    r = -r_bragg / np.log(2) * (1 + x) ** alpha * ln_term

    # Weights
    sin_term = np.sin(np.pi * i / (n_points + 1)) ** 2
    w_base = np.pi / (n_points + 1) * sin_term

    # Jacobian (simplified form)
    # dr/dx = R/ln(2) * (1+x)^(α-1) * [α*ln((1-x)/2) + (1+x)/(1-x)]
    jacobian = r_bragg / np.log(2) * (1 + x) ** (alpha - 1) * (
        alpha * ln_term + (1 + x) / (1 - x)
    )

    weights = w_base * np.abs(jacobian) * r ** 2

    return r, weights
