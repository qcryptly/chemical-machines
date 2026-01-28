"""
Hartree-Fock Analytic Gradients

Computes the gradient of the HF energy with respect to nuclear coordinates.

The HF gradient is:
    dE/dX = Tr[P·dH/dX] + 1/2 Tr[P·dG/dX·P] - Tr[W·dS/dX] + dV_nn/dX

where:
    P = density matrix
    H = core Hamiltonian (T + V)
    G = two-electron integrals
    W = energy-weighted density matrix
    S = overlap matrix
    V_nn = nuclear repulsion

Reference: Pulay, Mol. Phys. 17, 197 (1969)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..methods.hf import HFResult, hartree_fock
from ..basis import BasisSet
from .derivative_integrals import (
    overlap_derivative,
    kinetic_derivative,
    nuclear_derivative,
    eri_derivative,
    nuclear_repulsion_gradient,
)


ANGSTROM_TO_BOHR = 1.8897259886


@dataclass
class GradientResult:
    """
    Result of gradient calculation.

    Attributes:
        gradient: Nuclear gradient (n_atoms, 3) in Hartree/Bohr
        gradient_norm: RMS gradient
        max_gradient: Maximum gradient component
        atoms: Atom list
    """
    gradient: np.ndarray
    gradient_norm: float
    max_gradient: float
    atoms: List[Tuple[str, Tuple[float, float, float]]]

    def to_angstrom(self) -> np.ndarray:
        """Convert gradient to Hartree/Angstrom."""
        return self.gradient * ANGSTROM_TO_BOHR


class HFGradientCalculator:
    """
    Calculator for HF analytic gradients.

    Uses the Pulay force formulation:
        dE/dX = dE_nn/dX + Tr[P(dT/dX + dV/dX)]
                + 1/2 Tr[P·dG/dX·P] - Tr[W·dS/dX]
    """

    def __init__(self, hf_result: HFResult):
        """
        Initialize gradient calculator.

        Args:
            hf_result: Converged HF result
        """
        self.hf = hf_result
        self.P = hf_result.density
        self.C = hf_result.mo_coefficients
        self.eps = hf_result.orbital_energies
        self.n_occ = hf_result.n_electrons // 2

    def compute(self, verbose: bool = False) -> GradientResult:
        """
        Compute HF gradient.

        Returns:
            GradientResult with gradient array
        """
        atoms = self.hf.atoms
        n_atoms = len(atoms)

        # Build basis and nuclei
        basis = BasisSet('STO-3G')  # Should match HF calculation
        basis.build_for_molecule(atoms)

        nuclei = []
        for element, pos in atoms:
            Z = self._element_to_Z(element)
            pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR
            nuclei.append((pos_bohr, float(Z)))

        # Compute energy-weighted density matrix
        W = self._compute_weighted_density()

        if verbose:
            print("Computing HF gradient...")

        gradient = np.zeros((n_atoms, 3))

        # Nuclear repulsion gradient
        grad_Vnn = nuclear_repulsion_gradient(nuclei)
        gradient += grad_Vnn

        if verbose:
            print("  Nuclear repulsion gradient computed")

        # One-electron gradient contributions
        for A in range(n_atoms):
            for x in range(3):
                # dT/dX contribution
                dT = kinetic_derivative(basis, A, x)
                gradient[A, x] += np.sum(self.P * dT)

                # dV/dX contribution
                dV = nuclear_derivative(basis, nuclei, A, x)
                gradient[A, x] += np.sum(self.P * dV)

                # -W·dS/dX contribution (Pulay force)
                dS = overlap_derivative(basis, A, x)
                gradient[A, x] -= np.sum(W * dS)

        if verbose:
            print("  One-electron gradients computed")

        # Two-electron gradient contribution
        # This is expensive: dG/dX contracts with density matrix
        for A in range(n_atoms):
            for x in range(3):
                dG = eri_derivative(basis, A, x)

                # Coulomb contribution: Tr[P·dJ/dX]
                # dJ_μν = Σ_λσ (dμν|λσ) P_λσ
                dJ = np.einsum('mnls,ls->mn', dG, self.P)
                gradient[A, x] += 0.5 * np.sum(self.P * dJ)

                # Exchange contribution: -0.5 Tr[P·dK/dX]
                # dK_μν = Σ_λσ (dμλ|νσ) P_λσ
                dK = np.einsum('mlns,ls->mn', dG, self.P)
                gradient[A, x] -= 0.25 * np.sum(self.P * dK)

        if verbose:
            print("  Two-electron gradients computed")

        # Compute gradient statistics
        grad_norm = np.sqrt(np.mean(gradient ** 2))
        max_grad = np.max(np.abs(gradient))

        if verbose:
            print(f"\nGradient norm: {grad_norm:.6f} Hartree/Bohr")
            print(f"Max gradient: {max_grad:.6f} Hartree/Bohr")

        return GradientResult(
            gradient=gradient,
            gradient_norm=grad_norm,
            max_gradient=max_grad,
            atoms=atoms
        )

    def _compute_weighted_density(self) -> np.ndarray:
        """
        Compute energy-weighted density matrix.

        W_μν = 2 Σ_i^occ ε_i C_μi C_νi

        This appears in the Pulay force term for non-orthogonal basis.
        """
        C_occ = self.C[:, :self.n_occ]
        eps_occ = self.eps[:self.n_occ]

        W = 2.0 * np.einsum('mi,i,ni->mn', C_occ, eps_occ, C_occ)

        return W

    def _element_to_Z(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        elements = {
            'H': 1, 'He': 2,
            'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        }
        return elements.get(element, 1)


def hf_gradient(hf_result: HFResult = None,
                atoms: List[Tuple[str, Tuple[float, float, float]]] = None,
                basis: str = 'STO-3G',
                verbose: bool = False) -> GradientResult:
    """
    Compute HF gradient.

    Can use existing HF result or run HF first.

    Args:
        hf_result: Pre-computed HF result (optional)
        atoms: Molecule geometry (required if no hf_result)
        basis: Basis set name
        verbose: Print progress

    Returns:
        GradientResult with gradient array

    Example:
        # From HF result
        hf = hartree_fock(atoms)
        grad = hf_gradient(hf_result=hf)

        # Direct
        grad = hf_gradient(atoms=[('H', (0,0,0)), ('H', (0.74,0,0))])
    """
    if hf_result is None:
        if atoms is None:
            raise ValueError("Either hf_result or atoms required")
        hf_result = hartree_fock(atoms, basis=basis, verbose=verbose)

    calc = HFGradientCalculator(hf_result)
    return calc.compute(verbose=verbose)


def numerical_gradient(atoms: List[Tuple[str, Tuple[float, float, float]]],
                       energy_func,
                       step: float = 0.001) -> np.ndarray:
    """
    Compute gradient numerically by finite difference.

    Useful for testing analytic gradients.

    Args:
        atoms: Molecule geometry
        energy_func: Function that returns energy given atoms
        step: Finite difference step in Angstrom

    Returns:
        Numerical gradient (n_atoms, 3) in Hartree/Angstrom
    """
    n_atoms = len(atoms)
    gradient = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        element, pos = atoms[i]
        pos = list(pos)

        for j in range(3):
            # Forward
            pos[j] += step
            atoms_plus = atoms[:i] + [(element, tuple(pos))] + atoms[i + 1:]
            E_plus = energy_func(atoms_plus)

            # Backward
            pos[j] -= 2 * step
            atoms_minus = atoms[:i] + [(element, tuple(pos))] + atoms[i + 1:]
            E_minus = energy_func(atoms_minus)

            # Reset
            pos[j] += step

            gradient[i, j] = (E_plus - E_minus) / (2 * step)

    return gradient
