"""
Hessian (Second Derivative) Calculation

Computes the Hessian matrix (matrix of second derivatives of energy
with respect to nuclear coordinates).

H_ij = ∂²E / ∂X_i ∂X_j

Methods:
- Numerical: Finite difference of gradients (most general)
- Analytic: Direct computation (more accurate, method-specific)
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..methods.hf import HFResult, hartree_fock
from ..methods.dft import DFTResult, kohn_sham
from ..gradients import hf_gradient, dft_gradient


ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR

# Atomic masses in amu
ATOMIC_MASSES = {
    'H': 1.00783, 'He': 4.00260,
    'Li': 6.94100, 'Be': 9.01218, 'B': 10.81100, 'C': 12.00000,
    'N': 14.00307, 'O': 15.99491, 'F': 18.99840, 'Ne': 20.17970,
    'Na': 22.98977, 'Mg': 24.30500, 'Al': 26.98154, 'Si': 28.08550,
    'P': 30.97376, 'S': 32.06500, 'Cl': 34.96885, 'Ar': 39.94800,
}


@dataclass
class HessianResult:
    """
    Result of Hessian calculation.

    Attributes:
        hessian: Hessian matrix in Cartesian coordinates (3N x 3N)
        hessian_mw: Mass-weighted Hessian
        atoms: Molecule geometry
        masses: Atomic masses used
        eigenvalues: Eigenvalues of mass-weighted Hessian
        eigenvectors: Normal mode eigenvectors
    """
    hessian: np.ndarray
    hessian_mw: np.ndarray
    atoms: List[Tuple[str, Tuple[float, float, float]]]
    masses: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def compute_hessian(atoms: List[Tuple[str, Tuple[float, float, float]]],
                    method: str = 'HF',
                    basis: str = 'STO-3G',
                    step: float = 0.005,
                    verbose: bool = False) -> HessianResult:
    """
    Compute Hessian matrix by finite difference of gradients.

    H_ij ≈ [g_i(x + h·e_j) - g_i(x - h·e_j)] / (2h)

    Args:
        atoms: Molecule geometry
        method: Electronic structure method
        basis: Basis set name
        step: Finite difference step size (Bohr)
        verbose: Print progress

    Returns:
        HessianResult with Hessian matrix and eigenanalysis
    """
    n_atoms = len(atoms)
    n_coords = 3 * n_atoms

    # Get atomic masses
    masses = np.array([ATOMIC_MASSES.get(a[0], 1.0) for a in atoms])
    mass_weights = np.repeat(masses, 3)

    # Convert to coordinates in Bohr
    elements = [a[0] for a in atoms]
    coords = np.array([a[1] for a in atoms]) * ANGSTROM_TO_BOHR
    x0 = coords.flatten()

    is_dft = method.upper() not in ('HF', 'RHF', 'UHF')

    if verbose:
        print(f"Computing Hessian by finite difference")
        print(f"Method: {method}/{basis}")
        print(f"Step size: {step} Bohr")
        print(f"Total gradient evaluations: {2 * n_coords}")

    # Initialize Hessian
    H = np.zeros((n_coords, n_coords))

    for i in range(n_coords):
        if verbose and i % 3 == 0:
            print(f"  Processing coordinate {i+1}/{n_coords}...")

        # Forward displacement
        x_plus = x0.copy()
        x_plus[i] += step
        atoms_plus = _coords_to_atoms(elements, x_plus)

        if is_dft:
            result_plus = kohn_sham(atoms_plus, functional=method, basis=basis)
            grad_plus = dft_gradient(dft_result=result_plus).gradient.flatten()
        else:
            result_plus = hartree_fock(atoms_plus, basis=basis)
            grad_plus = hf_gradient(hf_result=result_plus).gradient.flatten()

        # Backward displacement
        x_minus = x0.copy()
        x_minus[i] -= step
        atoms_minus = _coords_to_atoms(elements, x_minus)

        if is_dft:
            result_minus = kohn_sham(atoms_minus, functional=method, basis=basis)
            grad_minus = dft_gradient(dft_result=result_minus).gradient.flatten()
        else:
            result_minus = hartree_fock(atoms_minus, basis=basis)
            grad_minus = hf_gradient(hf_result=result_minus).gradient.flatten()

        # Central difference
        H[:, i] = (grad_plus - grad_minus) / (2 * step)

    # Symmetrize
    H = 0.5 * (H + H.T)

    # Mass-weight the Hessian
    # H_mw = M^(-1/2) H M^(-1/2)
    mass_sqrt_inv = 1.0 / np.sqrt(mass_weights)
    H_mw = np.outer(mass_sqrt_inv, mass_sqrt_inv) * H

    # Diagonalize mass-weighted Hessian
    eigenvalues, eigenvectors = np.linalg.eigh(H_mw)

    if verbose:
        print(f"\nHessian computed successfully")
        print(f"Lowest eigenvalue: {eigenvalues[0]:.6f}")
        n_negative = np.sum(eigenvalues < -1e-6)
        print(f"Negative eigenvalues: {n_negative}")

    return HessianResult(
        hessian=H,
        hessian_mw=H_mw,
        atoms=atoms,
        masses=masses,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors
    )


def _coords_to_atoms(elements: List[str],
                     x: np.ndarray) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Convert flat coordinate array to atoms list."""
    coords = x.reshape(-1, 3) * BOHR_TO_ANGSTROM
    return [(e, tuple(c)) for e, c in zip(elements, coords)]
