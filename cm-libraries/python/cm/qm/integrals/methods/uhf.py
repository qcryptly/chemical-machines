"""
Unrestricted Hartree-Fock (UHF) Solver

Implements UHF for open-shell molecules with separate α and β spin orbitals.

The UHF energy is:
    E_UHF = Σ_i (h_αα + h_ββ) + ½(J_αα + J_ββ + J_αβ + J_βα)
            - ½(K_αα + K_ββ) + V_nn

where α and β electrons have separate orbital sets.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..basis import BasisSet, BasisFunction
from ..one_electron import overlap_matrix, kinetic_matrix, nuclear_attraction_matrix
from ..two_electron import eri_tensor, compute_J_matrix, compute_K_matrix


ANGSTROM_TO_BOHR = 1.8897259886


@dataclass
class UHFResult:
    """
    Results from an Unrestricted Hartree-Fock calculation.

    Attributes:
        energy: Total UHF energy (Hartree)
        orbital_energies_alpha: Alpha MO energies
        orbital_energies_beta: Beta MO energies
        mo_coefficients_alpha: Alpha MO coefficients
        mo_coefficients_beta: Beta MO coefficients
        density_alpha: Alpha density matrix
        density_beta: Beta density matrix
        density_total: Total density matrix (P_α + P_β)
        converged: Whether SCF converged
        n_iterations: Number of SCF iterations

        # Spin expectation value
        S2: <S²> expectation value
        S2_exact: Exact <S²> for given multiplicity
        multiplicity: 2S+1 spin multiplicity

        # Integral matrices
        S: Overlap matrix
        T: Kinetic energy matrix
        V: Nuclear attraction matrix
        H_core: Core Hamiltonian (T + V)

        # Energy components
        E_kinetic: Kinetic energy
        E_nuclear_attraction: Electron-nuclear attraction energy
        E_coulomb: Coulomb repulsion energy
        E_exchange: Exchange energy
        E_nuclear_repulsion: Nuclear-nuclear repulsion

        # Molecule info
        atoms: List of (element, (x, y, z)) in Angstroms
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons
    """
    energy: float
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    mo_coefficients_alpha: np.ndarray
    mo_coefficients_beta: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    density_total: np.ndarray
    converged: bool
    n_iterations: int

    # Spin
    S2: float
    S2_exact: float
    multiplicity: int

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
    n_alpha: int = 0
    n_beta: int = 0


class UnrestrictedHartreeFockSolver:
    """
    Unrestricted Hartree-Fock solver for open-shell molecules.

    Uses separate α and β spin orbitals, allowing for proper treatment
    of radicals, excited states, and bond breaking.

    The SCF procedure:
    1. Build one-electron integrals (S, T, V)
    2. Build two-electron integrals (ERI tensor)
    3. Guess initial density matrices (from core Hamiltonian)
    4. Iterate until convergence:
       - Build Fock matrices: F_α = H + J[P_total] - K[P_α]
                              F_β = H + J[P_total] - K[P_β]
       - Solve eigenvalue problems: F_α C_α = S C_α ε_α
       - Build new density matrices
       - Check convergence

    Example:
        solver = UnrestrictedHartreeFockSolver()
        result = solver.solve(
            atoms=[('O', (0, 0, 0))],
            n_alpha=5, n_beta=3,  # Triplet oxygen
            multiplicity=3
        )
    """

    def __init__(self, basis_name: str = 'STO-3G',
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-8):
        """
        Initialize the UHF solver.

        Args:
            basis_name: Basis set name
            max_iterations: Maximum SCF iterations
            convergence_threshold: Energy convergence threshold (Hartree)
        """
        self.basis_name = basis_name
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def solve(self, atoms: List[Tuple[str, Tuple[float, float, float]]],
              n_alpha: int = None,
              n_beta: int = None,
              multiplicity: int = None,
              charge: int = 0,
              verbose: bool = False) -> UHFResult:
        """
        Perform UHF calculation.

        Args:
            atoms: List of (element_symbol, (x, y, z)) tuples (positions in Angstroms)
            n_alpha: Number of alpha electrons
            n_beta: Number of beta electrons
            multiplicity: Spin multiplicity (2S+1), alternative to specifying n_alpha/n_beta
            charge: Molecular charge
            verbose: Print iteration info

        Returns:
            UHFResult with energy and wavefunction information
        """
        # Determine electron counts
        total_electrons = sum(self._element_to_Z(el) for el, _ in atoms) - charge

        if multiplicity is not None:
            # Derive n_alpha, n_beta from multiplicity
            # multiplicity = n_alpha - n_beta + 1
            # n_alpha + n_beta = total_electrons
            n_unpaired = multiplicity - 1
            n_paired = total_electrons - n_unpaired
            if n_paired % 2 != 0:
                raise ValueError(f"Invalid multiplicity {multiplicity} for {total_electrons} electrons")
            n_beta = n_paired // 2
            n_alpha = n_beta + n_unpaired
        elif n_alpha is None or n_beta is None:
            # Default to closed-shell
            if total_electrons % 2 != 0:
                n_alpha = total_electrons // 2 + 1
                n_beta = total_electrons // 2
            else:
                n_alpha = n_beta = total_electrons // 2

        if n_alpha + n_beta != total_electrons:
            raise ValueError(f"n_alpha ({n_alpha}) + n_beta ({n_beta}) != total electrons ({total_electrons})")

        actual_multiplicity = n_alpha - n_beta + 1

        # Build basis set
        basis = BasisSet(self.basis_name)
        basis.build_for_molecule(atoms)
        n_basis = basis.n_basis

        if verbose:
            print(f"UHF Calculation")
            print(f"Basis set: {self.basis_name}")
            print(f"Number of basis functions: {n_basis}")
            print(f"Alpha electrons: {n_alpha}, Beta electrons: {n_beta}")
            print(f"Multiplicity: {actual_multiplicity}")

        # Build nuclear positions
        nuclei = []
        for element, pos in atoms:
            Z = self._element_to_Z(element)
            pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR
            nuclei.append((pos_bohr, float(Z)))

        # =========================================================
        # Step 1: Build one-electron integrals
        # =========================================================
        if verbose:
            print("\nBuilding integrals...")

        S = overlap_matrix(basis)
        T = kinetic_matrix(basis)
        V = nuclear_attraction_matrix(basis, nuclei)
        H_core = T + V

        # =========================================================
        # Step 2: Build two-electron integrals
        # =========================================================
        G_tensor = eri_tensor(basis)

        # =========================================================
        # Step 3: Nuclear repulsion energy
        # =========================================================
        E_nuc = self._nuclear_repulsion(nuclei)

        # =========================================================
        # Step 4: Orthogonalization matrix
        # =========================================================
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        X = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        # =========================================================
        # Step 5: Initial guess
        # =========================================================
        F_prime = X.T @ H_core @ X
        eps, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime

        # Initial density matrices
        P_alpha = C[:, :n_alpha] @ C[:, :n_alpha].T
        P_beta = C[:, :n_beta] @ C[:, :n_beta].T

        # =========================================================
        # Step 6: SCF iterations
        # =========================================================
        E_old = 0.0
        converged = False

        if verbose:
            print("\nStarting UHF SCF iterations:")
            print("-" * 55)
            print(f"{'Iter':>4} {'E_total':>16} {'dE':>12} {'<S²>':>10}")
            print("-" * 55)

        for iteration in range(self.max_iterations):
            P_total = P_alpha + P_beta

            # Build Coulomb matrix (same for both spins)
            J = compute_J_matrix(P_total, G_tensor)

            # Build exchange matrices (different for each spin)
            K_alpha = compute_K_matrix(P_alpha, G_tensor)
            K_beta = compute_K_matrix(P_beta, G_tensor)

            # Fock matrices
            F_alpha = H_core + J - K_alpha
            F_beta = H_core + J - K_beta

            # Calculate energy
            E_elec = 0.5 * (np.sum(P_alpha * (H_core + F_alpha)) +
                           np.sum(P_beta * (H_core + F_beta)))
            E_total = E_elec + E_nuc

            # Calculate <S²>
            S2 = self._compute_S2(P_alpha, P_beta, S, n_alpha, n_beta)
            S2_exact = 0.25 * (n_alpha - n_beta) * (n_alpha - n_beta + 2)

            # Check convergence
            dE = abs(E_total - E_old)

            if verbose:
                print(f"{iteration+1:4d} {E_total:16.10f} {dE:12.2e} {S2:10.4f}")

            if dE < self.convergence_threshold:
                converged = True
                break

            E_old = E_total

            # Diagonalize Fock matrices
            F_alpha_prime = X.T @ F_alpha @ X
            eps_alpha, C_alpha_prime = np.linalg.eigh(F_alpha_prime)
            C_alpha = X @ C_alpha_prime

            F_beta_prime = X.T @ F_beta @ X
            eps_beta, C_beta_prime = np.linalg.eigh(F_beta_prime)
            C_beta = X @ C_beta_prime

            # New density matrices
            P_alpha = C_alpha[:, :n_alpha] @ C_alpha[:, :n_alpha].T
            P_beta = C_beta[:, :n_beta] @ C_beta[:, :n_beta].T

        if verbose:
            print("-" * 55)
            if converged:
                print(f"SCF converged in {iteration+1} iterations")
            else:
                print(f"SCF did not converge after {self.max_iterations} iterations")
            print(f"\nFinal UHF energy: {E_total:.10f} Hartree")
            print(f"<S²>: {S2:.4f} (exact: {S2_exact:.4f})")
            if abs(S2 - S2_exact) > 0.1:
                print(f"  Warning: Significant spin contamination detected!")

        # =========================================================
        # Step 7: Calculate energy components
        # =========================================================
        P_total = P_alpha + P_beta
        E_kinetic = np.sum(P_total * T)
        E_nuclear_attraction = np.sum(P_total * V)
        E_coulomb = 0.5 * np.sum(P_total * J)
        E_exchange = -0.5 * (np.sum(P_alpha * K_alpha) + np.sum(P_beta * K_beta))

        return UHFResult(
            energy=E_total,
            orbital_energies_alpha=eps_alpha,
            orbital_energies_beta=eps_beta,
            mo_coefficients_alpha=C_alpha,
            mo_coefficients_beta=C_beta,
            density_alpha=P_alpha,
            density_beta=P_beta,
            density_total=P_total,
            converged=converged,
            n_iterations=iteration + 1,
            S2=S2,
            S2_exact=S2_exact,
            multiplicity=actual_multiplicity,
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
            n_alpha=n_alpha,
            n_beta=n_beta
        )

    def _compute_S2(self, P_alpha: np.ndarray, P_beta: np.ndarray,
                    S: np.ndarray, n_alpha: int, n_beta: int) -> float:
        """
        Compute <S²> expectation value.

        <S²> = S_exact + N_β - Σ_{ij} |<φ_i^α|φ_j^β>|²

        where S_exact = (N_α - N_β)/2 * ((N_α - N_β)/2 + 1)
        """
        S_exact = 0.25 * (n_alpha - n_beta) * (n_alpha - n_beta + 2)

        # Overlap between alpha and beta orbitals
        # <S²> = S_exact + n_beta - Tr[P_alpha @ S @ P_beta @ S]
        contamination = np.trace(P_alpha @ S @ P_beta @ S)

        return S_exact + n_beta - contamination

    def _nuclear_repulsion(self, nuclei: List[Tuple[np.ndarray, float]]) -> float:
        """Calculate nuclear repulsion energy."""
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


def uhf(atoms: List[Tuple[str, Tuple[float, float, float]]],
        n_alpha: int = None,
        n_beta: int = None,
        multiplicity: int = None,
        charge: int = 0,
        basis: str = 'STO-3G',
        verbose: bool = False) -> UHFResult:
    """
    Convenience function for UHF calculation.

    Args:
        atoms: List of (element_symbol, (x, y, z)) tuples (positions in Angstroms)
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons
        multiplicity: Spin multiplicity (2S+1)
        charge: Molecular charge
        basis: Basis set name
        verbose: Print iteration info

    Returns:
        UHFResult with energy and wavefunction

    Example:
        # Triplet oxygen atom
        result = uhf([('O', (0, 0, 0))], multiplicity=3)

        # Hydrogen radical
        result = uhf([('H', (0, 0, 0))], multiplicity=2)
    """
    solver = UnrestrictedHartreeFockSolver(basis_name=basis)
    return solver.solve(atoms, n_alpha, n_beta, multiplicity, charge, verbose)
