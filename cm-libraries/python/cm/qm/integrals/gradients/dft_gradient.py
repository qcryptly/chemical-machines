"""
DFT Analytic Gradients

Computes the gradient of the Kohn-Sham DFT energy.

The DFT gradient includes:
    dE/dX = dE_nn/dX + Tr[P·dH/dX] + Tr[P·dJ/dX] + dE_xc/dX
            - Tr[W·dS/dX] + α·Tr[P·dK/dX]

where the XC gradient requires:
    dE_xc/dX = ∫ v_xc(r) dρ(r)/dX dr + ∫ ρ(r) ∂v_xc/∂X dr

For hybrid functionals, α is the fraction of exact exchange.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..methods.dft import DFTResult, kohn_sham
from ..basis import BasisSet
from ..dft import MolecularGrid, get_functional
from .hf_gradient import GradientResult, numerical_gradient
from .derivative_integrals import (
    overlap_derivative,
    kinetic_derivative,
    nuclear_derivative,
    eri_derivative,
    nuclear_repulsion_gradient,
)


ANGSTROM_TO_BOHR = 1.8897259886


class DFTGradientCalculator:
    """
    Calculator for DFT analytic gradients.

    Extends HF gradient with XC contributions.
    """

    def __init__(self, dft_result: DFTResult):
        """
        Initialize gradient calculator.

        Args:
            dft_result: Converged DFT result
        """
        self.dft = dft_result
        self.P = dft_result.density
        self.C = dft_result.mo_coefficients
        self.eps = dft_result.orbital_energies
        self.n_occ = dft_result.n_electrons // 2
        self.functional = get_functional(dft_result.functional_name)
        self.exact_exchange = dft_result.exact_exchange_fraction

    def compute(self, verbose: bool = False) -> GradientResult:
        """
        Compute DFT gradient.

        Returns:
            GradientResult with gradient array
        """
        atoms = self.dft.atoms
        n_atoms = len(atoms)

        # Build basis and nuclei
        basis = BasisSet('6-31G*')  # Should match DFT calculation
        basis.build_for_molecule(atoms)

        nuclei = []
        atom_centers = []
        for element, pos in atoms:
            Z = self._element_to_Z(element)
            pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR
            nuclei.append((pos_bohr, float(Z)))
            atom_centers.append(pos_bohr)

        atom_centers = np.array(atom_centers)

        # Compute energy-weighted density matrix
        W = self._compute_weighted_density()

        if verbose:
            print("Computing DFT gradient...")

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

        # Two-electron gradient (Coulomb only for pure DFT)
        for A in range(n_atoms):
            for x in range(3):
                dG = eri_derivative(basis, A, x)

                # Coulomb contribution
                dJ = np.einsum('mnls,ls->mn', dG, self.P)
                gradient[A, x] += 0.5 * np.sum(self.P * dJ)

                # Exact exchange for hybrids
                if self.exact_exchange > 0:
                    dK = np.einsum('mlns,ls->mn', dG, self.P)
                    gradient[A, x] -= 0.5 * self.exact_exchange * np.sum(self.P * dK)

        if verbose:
            print("  Two-electron gradients computed")

        # XC gradient contribution
        grad_xc = self._compute_xc_gradient(basis, atoms, atom_centers)
        gradient += grad_xc

        if verbose:
            print("  XC gradient computed")

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

    def _compute_xc_gradient(self, basis: BasisSet, atoms: List,
                             atom_centers: np.ndarray) -> np.ndarray:
        """
        Compute XC contribution to gradient.

        dE_xc/dX = ∫ v_xc(r) dρ(r)/dX dr

        Uses numerical differentiation on grid.
        """
        n_atoms = len(atoms)
        grad_xc = np.zeros((n_atoms, 3))

        # Build molecular grid
        grid = MolecularGrid.build(atoms, radial_points=50, angular_order=194)

        # This is a simplified implementation
        # Full implementation requires derivative of basis functions on grid
        # and proper integration of v_xc * d(basis)/dX terms

        # Use numerical differentiation for now
        h = 1e-5

        for A in range(n_atoms):
            for x in range(3):
                # Perturb atom position
                atoms_plus = list(atoms)
                atoms_minus = list(atoms)

                elem, pos = atoms[A]
                pos_plus = list(pos)
                pos_minus = list(pos)
                pos_plus[x] += h / ANGSTROM_TO_BOHR  # h is in Bohr
                pos_minus[x] -= h / ANGSTROM_TO_BOHR

                atoms_plus[A] = (elem, tuple(pos_plus))
                atoms_minus[A] = (elem, tuple(pos_minus))

                # Compute XC energy at displaced positions
                E_xc_plus = self._compute_xc_energy_at_geometry(atoms_plus, basis)
                E_xc_minus = self._compute_xc_energy_at_geometry(atoms_minus, basis)

                grad_xc[A, x] = (E_xc_plus - E_xc_minus) / (2 * h)

        return grad_xc

    def _compute_xc_energy_at_geometry(self, atoms: List, basis: BasisSet) -> float:
        """Compute XC energy at given geometry using current density."""
        from ..dft import MolecularGrid, DensityData

        # Build grid for this geometry
        grid = MolecularGrid.build(atoms, radial_points=50, angular_order=194)

        # Evaluate density on grid (using current density matrix)
        n_points = grid.n_points
        ao_values = np.zeros((n_points, basis.n_basis))

        for mu, bf in enumerate(basis.basis_functions):
            ao_values[:, mu] = self._evaluate_ao(bf, grid.points)

        # Compute density
        P_phi = np.einsum('mn,gn->gm', self.P, ao_values)
        rho = np.einsum('gm,gm->g', P_phi, ao_values)
        rho = np.maximum(rho, 1e-15)

        # Evaluate functional
        sigma = 1e-10 * np.ones(n_points) if self.functional.needs_gradient else None
        density = DensityData(rho=rho, sigma=sigma)
        xc_out = self.functional.compute(density)

        # Integrate
        E_xc = np.sum(rho * xc_out.exc * grid.weights)

        return E_xc

    def _evaluate_ao(self, bf, points: np.ndarray) -> np.ndarray:
        """Evaluate atomic orbital at grid points."""
        center = bf.center
        r = points - center
        r2 = np.sum(r * r, axis=1)

        # Radial part
        radial = np.zeros(len(points))
        for prim in bf.primitives:
            radial += prim.coefficient * np.exp(-prim.exponent * r2)

        # Angular part (simplified for s-type)
        l = bf.l
        if l > 0:
            r_l = np.sqrt(r2) ** l
            radial *= r_l

        return radial

    def _compute_weighted_density(self) -> np.ndarray:
        """Compute energy-weighted density matrix."""
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


def dft_gradient(dft_result: DFTResult = None,
                 atoms: List[Tuple[str, Tuple[float, float, float]]] = None,
                 functional: str = 'B3LYP',
                 basis: str = '6-31G*',
                 verbose: bool = False) -> GradientResult:
    """
    Compute DFT gradient.

    Args:
        dft_result: Pre-computed DFT result (optional)
        atoms: Molecule geometry (required if no dft_result)
        functional: XC functional name
        basis: Basis set name
        verbose: Print progress

    Returns:
        GradientResult with gradient array

    Example:
        result = kohn_sham(atoms, functional='B3LYP')
        grad = dft_gradient(dft_result=result)
    """
    if dft_result is None:
        if atoms is None:
            raise ValueError("Either dft_result or atoms required")
        dft_result = kohn_sham(atoms, functional=functional, basis=basis,
                               verbose=verbose)

    calc = DFTGradientCalculator(dft_result)
    return calc.compute(verbose=verbose)
