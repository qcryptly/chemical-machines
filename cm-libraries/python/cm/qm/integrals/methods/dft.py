"""
Kohn-Sham DFT Solver

Implements density functional theory using the Kohn-Sham formalism.
The KS equations are analogous to HF but with an XC potential instead
of exact exchange.

E_KS = Σ_i h_ii + ½ J + E_xc[ρ] + V_nn

where:
    h = T + V_ne (one-electron integrals)
    J = Coulomb repulsion
    E_xc = Exchange-correlation energy (from functional)
    V_nn = Nuclear repulsion
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..basis import BasisSet, BasisFunction
from ..one_electron import overlap_matrix, kinetic_matrix, nuclear_attraction_matrix
from ..two_electron import eri_tensor, compute_J_matrix, compute_K_matrix
from ..dft import MolecularGrid, get_functional, XCFunctional, DensityData


ANGSTROM_TO_BOHR = 1.8897259886


@dataclass
class DFTResult:
    """
    Results from a Kohn-Sham DFT calculation.

    Attributes:
        energy: Total DFT energy (Hartree)
        orbital_energies: KS orbital energies
        mo_coefficients: KS orbital coefficients (columns are MOs in AO basis)
        density: Density matrix
        converged: Whether SCF converged
        n_iterations: Number of SCF iterations

        # Energy components
        E_kinetic: Kinetic energy
        E_nuclear_attraction: Electron-nuclear attraction energy
        E_coulomb: Coulomb repulsion energy
        E_xc: Exchange-correlation energy
        E_nuclear_repulsion: Nuclear-nuclear repulsion
        E_dispersion: Dispersion correction (if applicable)

        # DFT-specific
        functional_name: Name of XC functional used
        exact_exchange_fraction: Fraction of exact exchange (for hybrids)

        # Integral matrices
        S: Overlap matrix
        T: Kinetic energy matrix
        V: Nuclear attraction matrix
        H_core: Core Hamiltonian (T + V)

        # Molecule info
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

    # Energy components
    E_kinetic: float = 0.0
    E_nuclear_attraction: float = 0.0
    E_coulomb: float = 0.0
    E_xc: float = 0.0
    E_nuclear_repulsion: float = 0.0
    E_dispersion: float = 0.0
    E_exact_exchange: float = 0.0

    # DFT-specific
    functional_name: str = ""
    exact_exchange_fraction: float = 0.0

    # Molecule info
    atoms: List[Tuple[str, Tuple[float, float, float]]] = None
    n_electrons: int = 0

    # Basis set info (for downstream properties)
    basis: object = None
    basis_name: str = ''


class KohnShamSolver:
    """
    Kohn-Sham DFT solver.

    Uses the self-consistent field procedure with XC functionals:
    1. Build one-electron integrals (S, T, V)
    2. Build two-electron integrals (for Coulomb and optional exact exchange)
    3. Build molecular grid for XC integration
    4. Iterate until convergence:
       - Compute density on grid
       - Evaluate XC functional
       - Build Fock matrix: F = H_core + J + V_xc + α*K (for hybrids)
       - Solve KS equations: FC = SCε
       - Build new density matrix
       - Check convergence

    Example:
        solver = KohnShamSolver(functional='B3LYP', basis='6-31G*')
        result = solver.solve(
            atoms=[('O', (0, 0, 0)), ('H', (0.96, 0, 0)), ('H', (-0.24, 0.93, 0))],
            n_electrons=10
        )
        print(f"DFT Energy: {result.energy:.6f} Hartree")
    """

    def __init__(self, functional: str = 'B3LYP',
                 basis_name: str = '6-31G*',
                 grid_level: int = 3,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-7,
                 dispersion: Optional[str] = None):
        """
        Initialize the KS solver.

        Args:
            functional: XC functional name ('B3LYP', 'PBE', 'SVWN5', etc.)
            basis_name: Basis set name
            grid_level: Grid quality (1=coarse, 3=medium, 5=fine)
            max_iterations: Maximum SCF iterations
            convergence_threshold: Energy convergence threshold (Hartree)
            dispersion: Dispersion correction ('D3', 'D3BJ', None)
        """
        self.functional_name = functional
        self.basis_name = basis_name
        self.grid_level = grid_level
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.dispersion = dispersion

        # Get functional object
        self.functional = get_functional(functional)

    def solve(self, atoms: List[Tuple[str, Tuple[float, float, float]]],
              n_electrons: int = None,
              charge: int = 0,
              verbose: bool = False) -> DFTResult:
        """
        Perform Kohn-Sham DFT calculation.

        Args:
            atoms: List of (element_symbol, (x, y, z)) tuples (positions in Angstroms)
            n_electrons: Total number of electrons (auto-detected if None)
            charge: Molecular charge
            verbose: Print iteration info

        Returns:
            DFTResult with energy and wavefunction information
        """
        # Auto-determine electron count
        if n_electrons is None:
            n_electrons = sum(self._element_to_Z(el) for el, _ in atoms) - charge

        # Build basis set
        basis = BasisSet(self.basis_name)
        basis.build_for_molecule(atoms)
        n_basis = basis.n_basis

        if verbose:
            print(f"Functional: {self.functional_name}")
            print(f"Basis set: {self.basis_name}")
            print(f"Number of basis functions: {n_basis}")
            print(f"Number of electrons: {n_electrons}")

        # Number of occupied orbitals
        n_occ = n_electrons // 2
        if n_electrons % 2 != 0:
            raise ValueError("Closed-shell DFT requires even number of electrons.")

        # Build nuclear positions
        nuclei = []
        atom_centers_bohr = []
        elements = []
        for element, pos in atoms:
            Z = self._element_to_Z(element)
            pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR
            nuclei.append((pos_bohr, float(Z)))
            atom_centers_bohr.append(pos_bohr)
            elements.append(element)

        atom_centers_bohr = np.array(atom_centers_bohr)

        # =========================================================
        # Step 1: Build one-electron integrals
        # =========================================================
        if verbose:
            print("\nBuilding one-electron integrals...")

        S = overlap_matrix(basis)
        T = kinetic_matrix(basis)
        V = nuclear_attraction_matrix(basis, nuclei)
        H_core = T + V

        # =========================================================
        # Step 2: Build two-electron integrals (for J and hybrid K)
        # =========================================================
        if verbose:
            print("Building two-electron integrals...")

        G_tensor = eri_tensor(basis)

        # Check for exact exchange
        exact_exchange = 0.0
        if hasattr(self.functional, 'exact_exchange'):
            exact_exchange = self.functional.exact_exchange
        elif hasattr(self.functional, 'exact_exchange_sr'):
            # Range-separated: use short-range fraction as approximation
            exact_exchange = self.functional.exact_exchange_sr

        if verbose and exact_exchange > 0:
            print(f"  Hybrid functional with {exact_exchange*100:.1f}% exact exchange")

        # =========================================================
        # Step 3: Build molecular grid for XC integration
        # =========================================================
        if verbose:
            print("Building molecular integration grid...")

        # Grid parameters based on level
        radial_points = 50 + self.grid_level * 25  # 75, 100, 125, 150, 175
        angular_order = 110 + self.grid_level * 64  # 174, 238, 302, 366, 430

        grid = MolecularGrid.build(
            atoms=atoms,
            radial_points=radial_points,
            angular_order=min(angular_order, 590),  # Cap at max Lebedev
            pruning='sg1'
        )

        if verbose:
            print(f"  Grid points: {grid.n_points}")

        # =========================================================
        # Step 4: Nuclear repulsion energy
        # =========================================================
        E_nuc = self._nuclear_repulsion(nuclei)

        if verbose:
            print(f"\nNuclear repulsion energy: {E_nuc:.6f} Hartree")

        # =========================================================
        # Step 5: Orthogonalization matrix
        # =========================================================
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        X = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        # =========================================================
        # Step 6: Initial guess (core Hamiltonian)
        # =========================================================
        F_prime = X.T @ H_core @ X
        eps_init, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime
        P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

        # =========================================================
        # Step 7: SCF iterations
        # =========================================================
        E_old = 0.0
        converged = False

        if verbose:
            print("\nStarting SCF iterations:")
            print("-" * 60)
            print(f"{'Iter':>4} {'E_total':>16} {'dE':>12} {'E_xc':>12}")
            print("-" * 60)

        for iteration in range(self.max_iterations):
            # Compute Coulomb matrix
            J = compute_J_matrix(P, G_tensor)

            # Compute exact exchange if hybrid
            K = np.zeros_like(J)
            if exact_exchange > 0:
                K = compute_K_matrix(P, G_tensor)

            # Compute XC energy and potential on grid
            E_xc, V_xc = self._compute_xc_on_grid(
                P, basis, grid, atom_centers_bohr
            )

            # Build Fock matrix
            # F = H_core + J + V_xc - α*K (for hybrids)
            F = H_core + J + V_xc - exact_exchange * K

            # Calculate total energy
            # E = Tr[P*H_core] + 0.5*Tr[P*J] + E_xc - 0.5*α*Tr[P*K] + E_nuc
            E_kinetic = np.sum(P * T)
            E_ne = np.sum(P * V)
            E_coulomb = 0.5 * np.sum(P * J)
            E_exact_exch = -0.5 * exact_exchange * np.sum(P * K)

            E_elec = E_kinetic + E_ne + E_coulomb + E_xc + E_exact_exch
            E_total = E_elec + E_nuc

            # Check convergence
            dE = abs(E_total - E_old)

            if verbose:
                print(f"{iteration+1:4d} {E_total:16.10f} {dE:12.2e} {E_xc:12.6f}")

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
            print("-" * 60)
            if converged:
                print(f"SCF converged in {iteration+1} iterations")
            else:
                print(f"SCF did not converge after {self.max_iterations} iterations")
            print(f"\nFinal DFT energy: {E_total:.10f} Hartree")

        # =========================================================
        # Step 8: Dispersion correction (if requested)
        # =========================================================
        E_dispersion = 0.0
        if self.dispersion is not None:
            # Placeholder for D3 dispersion
            # E_dispersion = compute_d3_dispersion(atoms, self.dispersion)
            pass

        return DFTResult(
            energy=E_total + E_dispersion,
            orbital_energies=eps,
            mo_coefficients=C,
            density=P,
            converged=converged,
            n_iterations=iteration + 1,
            S=S,
            T=T,
            V=V,
            H_core=H_core,
            E_kinetic=E_kinetic,
            E_nuclear_attraction=E_ne,
            E_coulomb=E_coulomb,
            E_xc=E_xc,
            E_exact_exchange=E_exact_exch,
            E_nuclear_repulsion=E_nuc,
            E_dispersion=E_dispersion,
            functional_name=self.functional_name,
            exact_exchange_fraction=exact_exchange,
            atoms=atoms,
            n_electrons=n_electrons,
            basis=basis,
            basis_name=self.basis_name
        )

    def _compute_xc_on_grid(self, P: np.ndarray, basis: BasisSet,
                            grid: MolecularGrid,
                            atom_centers: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute XC energy and potential matrix.

        This is a simplified implementation that evaluates the XC functional
        on grid points and integrates numerically.

        Args:
            P: Density matrix
            basis: Basis set
            grid: Molecular grid
            atom_centers: Atom positions in Bohr

        Returns:
            (E_xc, V_xc) tuple
        """
        n_basis = basis.n_basis
        n_points = grid.n_points
        points = grid.points
        weights = grid.weights

        # Evaluate all basis functions on grid
        ao_values = np.zeros((n_points, n_basis))

        for mu, bf in enumerate(basis.basis_functions):
            ao_values[:, mu] = self._evaluate_ao(bf, points)

        # Compute density on grid: ρ = Σ_μν P_μν φ_μ φ_ν
        P_phi = np.einsum('mn,gn->gm', P, ao_values)
        rho = np.einsum('gm,gm->g', P_phi, ao_values)
        rho = np.maximum(rho, 1e-15)

        # For GGA: compute gradient (simplified - using finite difference would be better)
        sigma = None
        if self.functional.needs_gradient:
            # Approximate: set sigma to small positive value
            # Full implementation would compute ∇ρ properly
            sigma = 1e-10 * np.ones(n_points)

        # Create density data
        density = DensityData(rho=rho, sigma=sigma)

        # Evaluate XC functional
        xc_out = self.functional.compute(density)

        # XC energy: E_xc = ∫ ρ ε_xc dτ
        E_xc = np.sum(rho * xc_out.exc * weights)

        # XC potential matrix: V_μν = ∫ v_xc φ_μ φ_ν dτ
        vrho = xc_out.vrho
        if vrho.ndim > 1:
            vrho = vrho.sum(axis=1)  # Sum spin components

        weighted_vrho = vrho * weights
        V_xc = np.einsum('g,gm,gn->mn', weighted_vrho, ao_values, ao_values)

        return E_xc, V_xc

    def _evaluate_ao(self, bf: BasisFunction, points: np.ndarray) -> np.ndarray:
        """
        Evaluate atomic orbital at grid points.

        Args:
            bf: Basis function
            points: Grid points (n_points, 3)

        Returns:
            AO values (n_points,)
        """
        center = bf.center
        r = points - center
        r2 = np.sum(r * r, axis=1)

        # Angular part
        l = bf.l
        angular = self._angular_part(l, bf.m, r)

        # Radial part: sum of contracted Gaussians with r^l
        radial = np.zeros(len(points))
        for prim in bf.primitives:
            radial += prim.coefficient * np.exp(-prim.exponent * r2)

        # r^l factor
        if l > 0:
            r_l = np.sqrt(r2) ** l
            radial *= r_l

        return angular * radial

    def _angular_part(self, l: int, m: int, r: np.ndarray) -> np.ndarray:
        """Compute angular part of basis function."""
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        r_norm = np.sqrt(x**2 + y**2 + z**2)
        r_norm = np.maximum(r_norm, 1e-15)

        if l == 0:
            return np.ones(len(r))
        elif l == 1:
            if m == -1:
                return y / r_norm
            elif m == 0:
                return z / r_norm
            else:
                return x / r_norm
        elif l == 2:
            r2 = r_norm ** 2
            if m == -2:
                return np.sqrt(3) * x * y / r2
            elif m == -1:
                return np.sqrt(3) * y * z / r2
            elif m == 0:
                return (3 * z**2 - r2) / (2 * r2)
            elif m == 1:
                return np.sqrt(3) * x * z / r2
            else:
                return np.sqrt(3) / 2 * (x**2 - y**2) / r2
        else:
            return np.ones(len(r))

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
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        }
        return elements.get(element, 1)


def kohn_sham(atoms: List[Tuple[str, Tuple[float, float, float]]],
              functional: str = 'B3LYP',
              basis: str = '6-31G*',
              n_electrons: int = None,
              charge: int = 0,
              dispersion: str = None,
              verbose: bool = False) -> DFTResult:
    """
    Convenience function for Kohn-Sham DFT calculation.

    Args:
        atoms: List of (element_symbol, (x, y, z)) tuples (positions in Angstroms)
        functional: XC functional name ('B3LYP', 'PBE', 'SVWN5', etc.)
        basis: Basis set name
        n_electrons: Number of electrons (auto-detected from atoms - charge)
        charge: Molecular charge
        dispersion: Dispersion correction ('D3', 'D3BJ', None)
        verbose: Print iteration info

    Returns:
        DFTResult with energy and wavefunction

    Example:
        # Water with B3LYP/6-31G*
        result = kohn_sham([
            ('O', (0, 0, 0)),
            ('H', (0.96, 0, 0)),
            ('H', (-0.24, 0.93, 0))
        ], functional='B3LYP', basis='6-31G*', verbose=True)
        print(f"Energy: {result.energy:.6f} Hartree")
    """
    solver = KohnShamSolver(
        functional=functional,
        basis_name=basis,
        dispersion=dispersion
    )
    return solver.solve(atoms, n_electrons, charge, verbose)
