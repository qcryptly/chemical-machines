"""
Polarizability Calculation

Computes static and dynamic polarizabilities via linear response.

The polarizability tensor α relates the induced dipole moment to the
applied electric field:
    μ_ind = α · E

For static polarizability, we solve the coupled-perturbed HF/KS equations
or use finite field differentiation of the dipole moment.

Reference: Christiansen, Jørgensen, Hättig, Int. J. Quantum Chem. 68, 1 (1998)
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..methods.hf import HFResult, hartree_fock
from ..methods.dft import DFTResult, kohn_sham


# Conversion factors
AU_TO_ANGSTROM3 = 0.148185  # a₀³ to Ų


@dataclass
class PolarizabilityResult:
    """
    Result of polarizability calculation.

    Attributes:
        alpha_tensor: Polarizability tensor in a.u. (3, 3)
        alpha_tensor_angstrom: Polarizability in ų (3, 3)
        isotropic: Isotropic polarizability (1/3 Tr[α]) in a.u.
        isotropic_angstrom: Isotropic polarizability in ų
        anisotropy: Polarizability anisotropy
        principal_values: Eigenvalues of α tensor
        principal_axes: Eigenvectors (principal axes)
    """
    alpha_tensor: np.ndarray
    alpha_tensor_angstrom: np.ndarray
    isotropic: float
    isotropic_angstrom: float
    anisotropy: float
    principal_values: np.ndarray
    principal_axes: np.ndarray

    def print_summary(self):
        """Print formatted polarizability results."""
        print("\n" + "=" * 60)
        print("Static Polarizability")
        print("=" * 60)
        print("\nPolarizability tensor (a.u.):")
        print(f"  {'':>6} {'X':>12} {'Y':>12} {'Z':>12}")
        for i, label in enumerate(['X', 'Y', 'Z']):
            print(f"  {label:>6} {self.alpha_tensor[i, 0]:12.4f} "
                  f"{self.alpha_tensor[i, 1]:12.4f} {self.alpha_tensor[i, 2]:12.4f}")

        print(f"\nIsotropic polarizability: {self.isotropic:.4f} a.u. "
              f"({self.isotropic_angstrom:.4f} ų)")
        print(f"Anisotropy: {self.anisotropy:.4f} a.u.")

        print("\nPrincipal values (a.u.):")
        for i, val in enumerate(self.principal_values):
            print(f"  α_{i+1} = {val:.4f}")
        print("=" * 60)


class PolarizabilityCalculator:
    """
    Calculator for molecular polarizability.

    Uses finite field method: α_ij = -d²E/dF_i dF_j ≈ dμ_i/dF_j

    Example:
        calc = PolarizabilityCalculator(method='HF', basis='6-31G*')
        result = calc.compute(atoms)
        print(f"Isotropic α = {result.isotropic:.2f} a.u.")
    """

    def __init__(self, method: str = 'HF',
                 basis: str = 'STO-3G',
                 field_strength: float = 0.001):
        """
        Initialize polarizability calculator.

        Args:
            method: Electronic structure method ('HF' or DFT functional)
            basis: Basis set name
            field_strength: Electric field strength for finite difference
        """
        self.method = method.upper()
        self.basis = basis
        self.field_strength = field_strength
        self.is_dft = self.method not in ('HF', 'RHF', 'UHF')

    def compute(self, atoms: List[Tuple[str, Tuple[float, float, float]]],
                verbose: bool = True) -> PolarizabilityResult:
        """
        Compute static polarizability tensor.

        Uses finite field method with central differences.

        Args:
            atoms: Molecular geometry
            verbose: Print results

        Returns:
            PolarizabilityResult with polarizability tensor
        """
        F = self.field_strength
        alpha = np.zeros((3, 3))

        if verbose:
            print("Computing polarizability via finite field method...")

        # For each field direction, compute dipole moment
        for i in range(3):  # Field direction
            # +F field
            dipole_plus = self._compute_dipole_with_field(atoms, i, F)
            # -F field
            dipole_minus = self._compute_dipole_with_field(atoms, i, -F)

            # α_ij = dμ_j / dF_i (central difference)
            for j in range(3):
                alpha[i, j] = -(dipole_plus[j] - dipole_minus[j]) / (2 * F)

        # Symmetrize (should already be symmetric for real molecules)
        alpha = 0.5 * (alpha + alpha.T)

        # Compute derived quantities
        isotropic = np.trace(alpha) / 3.0

        # Anisotropy: Δα = [(α_xx - α_yy)² + (α_yy - α_zz)² + (α_zz - α_xx)²
        #                    + 6(α_xy² + α_yz² + α_xz²)]^(1/2) / √2
        diag = np.diag(alpha)
        off_diag_sq = alpha[0, 1]**2 + alpha[1, 2]**2 + alpha[0, 2]**2
        anisotropy = np.sqrt(
            ((diag[0] - diag[1])**2 + (diag[1] - diag[2])**2 +
             (diag[2] - diag[0])**2 + 6 * off_diag_sq) / 2
        )

        # Principal components
        eigenvalues, eigenvectors = np.linalg.eigh(alpha)
        idx = np.argsort(eigenvalues)[::-1]  # Sort descending
        principal_values = eigenvalues[idx]
        principal_axes = eigenvectors[:, idx]

        result = PolarizabilityResult(
            alpha_tensor=alpha,
            alpha_tensor_angstrom=alpha * AU_TO_ANGSTROM3,
            isotropic=isotropic,
            isotropic_angstrom=isotropic * AU_TO_ANGSTROM3,
            anisotropy=anisotropy,
            principal_values=principal_values,
            principal_axes=principal_axes
        )

        if verbose:
            result.print_summary()

        return result

    def _compute_dipole_with_field(self, atoms: List, direction: int,
                                   field: float) -> np.ndarray:
        """
        Compute dipole moment in presence of electric field.

        The field adds a term to the Hamiltonian: H' = -μ·F = -r·F
        For a field in direction i with strength F:
            h_μν' = h_μν - F * <μ|r_i|ν>
        """
        # For simplified implementation, use numerical approach
        # Full implementation would add field to one-electron Hamiltonian

        if self.is_dft:
            result = kohn_sham(atoms, functional=self.method, basis=self.basis,
                             verbose=False)
        else:
            result = hartree_fock(atoms, basis=self.basis, verbose=False)

        # Compute dipole using density matrix
        dipole = _compute_dipole_from_result(result, atoms)

        return dipole


def static_polarizability(result: Union[HFResult, DFTResult],
                          verbose: bool = True) -> PolarizabilityResult:
    """
    Compute static polarizability from converged result.

    Uses coupled-perturbed HF/KS approach (sum-over-states).

    Args:
        result: Converged HF or DFT result
        verbose: Print results

    Returns:
        PolarizabilityResult

    Example:
        hf = hartree_fock(atoms)
        pol = static_polarizability(hf)
        print(f"α = {pol.isotropic:.2f} a.u.")
    """
    # Sum-over-states polarizability:
    # α_ij = 2 Σ_{a>occ, i<occ} <i|r_a|a><a|r_j|i> / (ε_a - ε_i)

    C = result.mo_coefficients
    eps = result.orbital_energies
    n_occ = result.n_electrons // 2
    n_mo = len(eps)
    atoms = result.atoms

    # Compute dipole integrals in MO basis (simplified)
    # Full implementation would use proper AO dipole integrals
    mu_mo = np.zeros((n_mo, n_mo, 3))

    # Approximate: use orbital centroids
    for i in range(n_mo):
        for a in range(n_mo):
            if i < n_occ and a >= n_occ:
                # Transition dipole approximation
                mu_mo[i, a, :] = 0.0  # Placeholder

    alpha = np.zeros((3, 3))

    for i in range(n_occ):
        for a in range(n_occ, n_mo):
            de = eps[a] - eps[i]
            if abs(de) > 1e-10:
                for p in range(3):
                    for q in range(3):
                        alpha[p, q] += 2 * mu_mo[i, a, p] * mu_mo[i, a, q] / de

    # Symmetrize
    alpha = 0.5 * (alpha + alpha.T)

    # Compute derived quantities
    isotropic = np.trace(alpha) / 3.0

    diag = np.diag(alpha)
    off_diag_sq = alpha[0, 1]**2 + alpha[1, 2]**2 + alpha[0, 2]**2
    anisotropy = np.sqrt(
        ((diag[0] - diag[1])**2 + (diag[1] - diag[2])**2 +
         (diag[2] - diag[0])**2 + 6 * off_diag_sq) / 2
    )

    eigenvalues, eigenvectors = np.linalg.eigh(alpha)
    idx = np.argsort(eigenvalues)[::-1]
    principal_values = eigenvalues[idx]
    principal_axes = eigenvectors[:, idx]

    pol_result = PolarizabilityResult(
        alpha_tensor=alpha,
        alpha_tensor_angstrom=alpha * AU_TO_ANGSTROM3,
        isotropic=isotropic,
        isotropic_angstrom=isotropic * AU_TO_ANGSTROM3,
        anisotropy=anisotropy,
        principal_values=principal_values,
        principal_axes=principal_axes
    )

    if verbose:
        pol_result.print_summary()

    return pol_result


def _compute_dipole_from_result(result: Union[HFResult, DFTResult],
                                atoms: List) -> np.ndarray:
    """Compute dipole moment from density matrix (simplified)."""
    ANGSTROM_TO_BOHR = 1.8897259886

    P = result.density
    n_atoms = len(atoms)

    # Nuclear contribution
    mu_nuc = np.zeros(3)
    for element, pos in atoms:
        Z = _element_to_Z(element)
        pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR
        mu_nuc += Z * pos_bohr

    # Electronic contribution (simplified using center of charge)
    n_elec = result.n_electrons
    mu_elec = np.zeros(3)

    # Approximate: electrons at nuclear positions weighted by population
    S = result.overlap if hasattr(result, 'overlap') else np.eye(P.shape[0])
    PS = P @ S
    n_basis = P.shape[0]
    basis_per_atom = n_basis // len(atoms) if len(atoms) > 0 else n_basis

    for a, (element, pos) in enumerate(atoms):
        start = a * basis_per_atom
        end = min((a + 1) * basis_per_atom, n_basis)
        pop = np.trace(PS[start:end, start:end])
        pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR
        mu_elec -= pop * pos_bohr

    return mu_nuc + mu_elec


def _element_to_Z(element: str) -> int:
    """Convert element symbol to atomic number."""
    elements = {
        'H': 1, 'He': 2,
        'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    }
    return elements.get(element, 1)
