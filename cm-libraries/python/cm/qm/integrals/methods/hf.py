"""
Hartree-Fock Solver

Implements the Restricted Hartree-Fock (RHF) method for closed-shell molecules
using proper Gaussian basis integrals.

The HF energy is:
    E_HF = Σ_i h_ii + ½ Σ_{ij} (J_ij - K_ij) + V_nn

where:
    h = T + V_ne (one-electron integrals)
    J = Coulomb integrals
    K = Exchange integrals
    V_nn = Nuclear repulsion
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..basis import BasisSet, BasisFunction
from ..one_electron import overlap_matrix, kinetic_matrix, nuclear_attraction_matrix
from ..two_electron import eri_tensor, compute_J_matrix, compute_K_matrix


@dataclass
class HFResult:
    """
    Results from a Hartree-Fock calculation.

    Attributes:
        energy: Total HF energy (Hartree)
        orbital_energies: MO energies (eigenvalues of Fock matrix)
        mo_coefficients: MO coefficients (columns are MOs in AO basis)
        density: Density matrix
        converged: Whether SCF converged
        n_iterations: Number of SCF iterations

        # Integral matrices
        S: Overlap matrix
        T: Kinetic energy matrix
        V: Nuclear attraction matrix
        H_core: Core Hamiltonian (T + V)
        G: Two-electron integrals (4D tensor)

        # Energy components
        E_kinetic: Kinetic energy
        E_nuclear_attraction: Electron-nuclear attraction energy
        E_coulomb: Coulomb repulsion energy
        E_exchange: Exchange energy
        E_nuclear_repulsion: Nuclear-nuclear repulsion

        # Molecule info (for visualization)
        atoms: List of (element, (x, y, z)) in Angstroms
        n_electrons: Total number of electrons
    """
    energy: float
    orbital_energies: np.ndarray
    mo_coefficients: np.ndarray
    density: np.ndarray
    converged: bool
    n_iterations: int

    # Matrices
    S: np.ndarray
    T: np.ndarray
    V: np.ndarray
    H_core: np.ndarray
    G: np.ndarray

    # Energy components
    E_kinetic: float = 0.0
    E_nuclear_attraction: float = 0.0
    E_coulomb: float = 0.0
    E_exchange: float = 0.0
    E_nuclear_repulsion: float = 0.0

    # Molecule info
    atoms: List[Tuple[str, Tuple[float, float, float]]] = None
    n_electrons: int = 0

    # Basis set info (for downstream properties)
    basis: object = None
    basis_name: str = ''


class HartreeFockSolver:
    """
    Restricted Hartree-Fock solver for closed-shell molecules.

    Uses the SCF (Self-Consistent Field) procedure:
    1. Build one-electron integrals (S, T, V)
    2. Build two-electron integrals (ERI tensor)
    3. Guess initial density matrix
    4. Iterate until convergence:
       - Build Fock matrix: F = H_core + G(P)
       - Solve Roothaan equations: FC = SCε
       - Build new density matrix
       - Check convergence

    Example:
        solver = HartreeFockSolver()
        result = solver.solve(
            atoms=[('H', (0, 0, 0)), ('H', (0.74, 0, 0))],
            n_electrons=2
        )
        print(f"HF Energy: {result.energy:.6f} Hartree")
    """

    def __init__(self, basis_name: str = 'STO-3G',
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-8,
                 diis: bool = True):
        """
        Initialize the HF solver.

        Args:
            basis_name: Basis set to use ('STO-3G')
            max_iterations: Maximum SCF iterations
            convergence_threshold: Energy convergence threshold (Hartree)
            diis: Use DIIS acceleration (not yet implemented)
        """
        self.basis_name = basis_name
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_diis = diis

    def solve(self, atoms: List[Tuple[str, Tuple[float, float, float]]],
              n_electrons: int,
              charge: int = 0,
              verbose: bool = False) -> HFResult:
        """
        Perform Hartree-Fock calculation.

        Args:
            atoms: List of (element_symbol, (x, y, z)) tuples.
                   Positions in Angstroms.
            n_electrons: Total number of electrons
            charge: Molecular charge (default 0)
            verbose: Print iteration info

        Returns:
            HFResult with energy and wavefunction information
        """
        ANGSTROM_TO_BOHR = 1.8897259886

        # Build basis set
        basis = BasisSet(self.basis_name)
        basis.build_for_molecule(atoms)
        n_basis = basis.n_basis

        if verbose:
            print(f"Basis set: {self.basis_name}")
            print(f"Number of basis functions: {n_basis}")
            print(f"Number of electrons: {n_electrons}")

        # Number of occupied orbitals (RHF: doubly occupied)
        n_occ = n_electrons // 2
        if n_electrons % 2 != 0:
            raise ValueError("RHF requires even number of electrons. Use UHF for odd electrons.")

        # Build nuclear positions and charges for integrals
        nuclei = []
        for element, pos in atoms:
            # Nuclear charge from element symbol
            Z = self._element_to_Z(element)
            # Position in Bohr
            pos_bohr = (
                pos[0] * ANGSTROM_TO_BOHR,
                pos[1] * ANGSTROM_TO_BOHR,
                pos[2] * ANGSTROM_TO_BOHR
            )
            nuclei.append((np.array(pos_bohr), float(Z)))

        # =========================================================
        # Step 1: Build one-electron integrals
        # =========================================================
        if verbose:
            print("\nBuilding one-electron integrals...")

        S = overlap_matrix(basis)
        T = kinetic_matrix(basis)
        V = nuclear_attraction_matrix(basis, nuclei)
        H_core = T + V

        if verbose:
            print(f"  Overlap matrix built")
            print(f"  Kinetic matrix built")
            print(f"  Nuclear attraction matrix built")

        # =========================================================
        # Step 2: Build two-electron integrals
        # =========================================================
        if verbose:
            print("\nBuilding two-electron integrals...")

        G_tensor = eri_tensor(basis)

        if verbose:
            print(f"  ERI tensor built: {G_tensor.shape}")

        # =========================================================
        # Step 3: Nuclear repulsion energy
        # =========================================================
        E_nuc = self._nuclear_repulsion(nuclei)

        if verbose:
            print(f"\nNuclear repulsion energy: {E_nuc:.6f} Hartree")

        # =========================================================
        # Step 4: Orthogonalization matrix
        # =========================================================
        # Symmetric orthogonalization: X = S^(-1/2)
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        X = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        # =========================================================
        # Step 5: Initial guess (core Hamiltonian guess)
        # =========================================================
        # Diagonalize H_core to get initial MO coefficients
        F_prime = X.T @ H_core @ X
        eps_init, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime

        # Initial density matrix: P = 2 * Σ_i^occ |φ_i><φ_i|
        P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

        # =========================================================
        # Step 6: SCF iterations
        # =========================================================
        E_old = 0.0
        converged = False

        if verbose:
            print("\nStarting SCF iterations:")
            print("-" * 50)

        for iteration in range(self.max_iterations):
            # Build two-electron part of Fock matrix
            J = compute_J_matrix(P, G_tensor)
            K = compute_K_matrix(P, G_tensor)
            G = J - 0.5 * K

            # Fock matrix
            F = H_core + G

            # Calculate electronic energy
            # E_elec = 0.5 * Tr[P(H_core + F)]
            E_elec = 0.5 * np.sum(P * (H_core + F))
            E_total = E_elec + E_nuc

            # Check convergence
            dE = abs(E_total - E_old)
            if verbose:
                print(f"  Iter {iteration+1:3d}: E = {E_total:16.10f}, dE = {dE:12.2e}")

            if dE < self.convergence_threshold:
                converged = True
                break

            E_old = E_total

            # Diagonalize Fock matrix
            F_prime = X.T @ F @ X
            eps, C_prime = np.linalg.eigh(F_prime)
            C = X @ C_prime

            # New density matrix
            P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

        if verbose:
            print("-" * 50)
            if converged:
                print(f"SCF converged in {iteration+1} iterations")
            else:
                print(f"SCF did not converge after {self.max_iterations} iterations")
            print(f"Final HF energy: {E_total:.10f} Hartree")

        # =========================================================
        # Step 7: Calculate energy components
        # =========================================================
        E_kinetic = np.sum(P * T)
        E_nuclear_attraction = np.sum(P * V)
        E_coulomb = 0.5 * np.sum(P * J)
        E_exchange = -0.25 * np.sum(P * K)

        return HFResult(
            energy=E_total,
            orbital_energies=eps,
            mo_coefficients=C,
            density=P,
            converged=converged,
            n_iterations=iteration + 1,
            S=S,
            T=T,
            V=V,
            H_core=H_core,
            G=G_tensor,
            E_kinetic=E_kinetic,
            E_nuclear_attraction=E_nuclear_attraction,
            E_coulomb=E_coulomb,
            E_exchange=E_exchange,
            E_nuclear_repulsion=E_nuc,
            atoms=atoms,
            n_electrons=n_electrons,
            basis=basis,
            basis_name=self.basis_name
        )

    def _nuclear_repulsion(self, nuclei: List[Tuple[np.ndarray, float]]) -> float:
        """
        Calculate nuclear-nuclear repulsion energy.

        V_nn = Σ_{A<B} Z_A * Z_B / R_AB

        Args:
            nuclei: List of (position, charge) tuples. Positions in Bohr.

        Returns:
            Nuclear repulsion energy in Hartree
        """
        E_nuc = 0.0
        n_nuclei = len(nuclei)

        for A in range(n_nuclei):
            for B in range(A + 1, n_nuclei):
                pos_A, Z_A = nuclei[A]
                pos_B, Z_B = nuclei[B]

                R_AB = np.linalg.norm(pos_A - pos_B)
                if R_AB > 1e-10:
                    E_nuc += Z_A * Z_B / R_AB

        return E_nuc

    def _element_to_Z(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        elements = {
            'H': 1, 'He': 2,
            'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        }
        return elements.get(element, 1)


def hartree_fock(atoms: List[Tuple[str, Tuple[float, float, float]]],
                 n_electrons: int = None,
                 charge: int = 0,
                 basis: str = 'STO-3G',
                 verbose: bool = False) -> HFResult:
    """
    Convenience function for Hartree-Fock calculation.

    Args:
        atoms: List of (element_symbol, (x, y, z)) tuples.
               Positions in Angstroms.
        n_electrons: Number of electrons (default: sum of atomic numbers - charge)
        charge: Molecular charge
        basis: Basis set name
        verbose: Print iteration info

    Returns:
        HFResult with energy and wavefunction

    Example:
        # H2 molecule
        result = hartree_fock([
            ('H', (0, 0, 0)),
            ('H', (0.74, 0, 0))
        ])
        print(f"H2 energy: {result.energy:.6f} Hartree")

        # Water molecule
        result = hartree_fock([
            ('O', (0, 0, 0)),
            ('H', (0.96, 0, 0)),
            ('H', (-0.24, 0.93, 0))
        ], verbose=True)
    """
    # Auto-determine electron count
    if n_electrons is None:
        element_Z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10
        }
        n_electrons = sum(element_Z.get(el, 1) for el, _ in atoms) - charge

    solver = HartreeFockSolver(basis_name=basis)
    return solver.solve(atoms, n_electrons, charge, verbose)
