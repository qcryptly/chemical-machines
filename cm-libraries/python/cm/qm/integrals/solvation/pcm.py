"""
Polarizable Continuum Model (PCM)

Implements the IEF-PCM (Integral Equation Formalism PCM) for implicit solvation.

The solute is placed in a molecular-shaped cavity surrounded by a dielectric
continuum representing the solvent. The electrostatic interaction between the
solute charge distribution and the induced polarization of the solvent is
computed by solving the Poisson equation.

Key equations:
    - Apparent surface charge: Kq = -Rρ
    - K = (ε+1)/(ε-1) S + D
    - Solvation energy: G_solv = (1/2) ∫ ρ(r) V_reaction(r) dr

Reference: Tomasi, Mennucci, Cammi, Chem. Rev. 105, 2999 (2005)
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..methods.hf import HFResult
from ..methods.dft import DFTResult


# Physical constants
ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR

# Dielectric constants of common solvents
SOLVENT_EPSILON = {
    'water': 78.39,
    'methanol': 32.63,
    'ethanol': 24.55,
    'acetonitrile': 35.94,
    'dmso': 46.70,
    'chloroform': 4.81,
    'dichloromethane': 8.93,
    'thf': 7.43,
    'toluene': 2.38,
    'hexane': 1.88,
    'benzene': 2.27,
    'acetone': 20.56,
    'dmf': 36.71,
    'diethylether': 4.24,
}

# Van der Waals radii for cavity construction (Bondi radii in Angstrom)
VDW_RADII = {
    'H': 1.20, 'He': 1.40,
    'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    'K': 2.75, 'Ca': 2.31,
    'Fe': 2.00, 'Co': 2.00, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39,
}


@dataclass
class CavityPoint:
    """A point on the molecular cavity surface."""
    position: np.ndarray  # (3,) position in Bohr
    normal: np.ndarray    # (3,) outward normal vector
    area: float           # Surface element area in Bohr²
    atom_index: int       # Index of atom this tesserae belongs to


@dataclass
class MolecularCavity:
    """
    Molecular cavity for PCM calculations.

    Attributes:
        points: List of cavity surface points (tesserae)
        n_points: Number of surface points
        total_area: Total cavity surface area in Bohr²
    """
    points: List[CavityPoint]
    n_points: int
    total_area: float

    def get_positions(self) -> np.ndarray:
        """Return all tesserae positions as (N, 3) array."""
        return np.array([p.position for p in self.points])

    def get_areas(self) -> np.ndarray:
        """Return all tesserae areas as (N,) array."""
        return np.array([p.area for p in self.points])

    def get_normals(self) -> np.ndarray:
        """Return all tesserae normals as (N, 3) array."""
        return np.array([p.normal for p in self.points])


@dataclass
class PCMResult:
    """
    Result of PCM solvation calculation.

    Attributes:
        solvation_energy: Electrostatic solvation free energy (Hartree)
        solvation_energy_kcal: Solvation energy in kcal/mol
        surface_charges: Apparent surface charges on tesserae
        reaction_field_energy: Interaction energy with reaction field
        cavity: The molecular cavity used
        epsilon: Dielectric constant used
        solvent: Solvent name if applicable
    """
    solvation_energy: float
    solvation_energy_kcal: float
    surface_charges: np.ndarray
    reaction_field_energy: float
    cavity: MolecularCavity
    epsilon: float
    solvent: Optional[str] = None

    def print_summary(self):
        """Print formatted PCM results."""
        print("\n" + "=" * 50)
        print("PCM Solvation Results")
        print("=" * 50)
        if self.solvent:
            print(f"Solvent: {self.solvent} (ε = {self.epsilon:.2f})")
        else:
            print(f"Dielectric constant: {self.epsilon:.2f}")
        print(f"Cavity surface points: {self.cavity.n_points}")
        print(f"Cavity surface area: {self.cavity.total_area * BOHR_TO_ANGSTROM**2:.2f} Ų")
        print(f"\nSolvation free energy: {self.solvation_energy:.6f} Hartree")
        print(f"                       {self.solvation_energy_kcal:.2f} kcal/mol")
        print("=" * 50)


class PCMSolver:
    """
    Polarizable Continuum Model solver.

    Implements IEF-PCM for computing solvation free energies.

    Example:
        pcm = PCMSolver(solvent='water')
        result = pcm.compute(hf_result, atoms)
        result.print_summary()
    """

    def __init__(self,
                 epsilon: float = None,
                 solvent: str = None,
                 scaling_factor: float = 1.2,
                 n_points_per_atom: int = 60):
        """
        Initialize PCM solver.

        Args:
            epsilon: Dielectric constant (overrides solvent)
            solvent: Solvent name (e.g., 'water', 'methanol')
            scaling_factor: Scale factor for atomic radii (default 1.2)
            n_points_per_atom: Approximate surface points per atom
        """
        if solvent is not None:
            solvent_lower = solvent.lower()
            if solvent_lower not in SOLVENT_EPSILON:
                raise ValueError(f"Unknown solvent: {solvent}. "
                               f"Available: {list(SOLVENT_EPSILON.keys())}")
            self.epsilon = SOLVENT_EPSILON[solvent_lower]
            self.solvent = solvent_lower
        elif epsilon is not None:
            self.epsilon = epsilon
            self.solvent = None
        else:
            # Default to water
            self.epsilon = SOLVENT_EPSILON['water']
            self.solvent = 'water'

        self.scaling_factor = scaling_factor
        self.n_points_per_atom = n_points_per_atom

    def compute(self,
                result: Union[HFResult, DFTResult],
                verbose: bool = True) -> PCMResult:
        """
        Compute PCM solvation energy.

        Args:
            result: Converged HF or DFT result
            verbose: Print summary

        Returns:
            PCMResult with solvation energies
        """
        atoms = result.atoms
        P = result.density

        # Build molecular cavity
        cavity = build_cavity(atoms, self.scaling_factor, self.n_points_per_atom)

        # Compute electrostatic potential from solute at cavity points
        V_solute = self._compute_solute_potential(P, result, cavity)

        # Build PCM matrices
        S, D = self._build_pcm_matrices(cavity)

        # Build K matrix: K = (ε+1)/(ε-1) * S + D
        f_eps = (self.epsilon + 1) / (self.epsilon - 1)
        K = f_eps * S + D

        # Solve for apparent surface charges: K * q = -R * ρ
        # For point charges, R * ρ ≈ V_solute (the electrostatic potential)
        b = -V_solute

        # Solve linear system
        try:
            q = np.linalg.solve(K, b)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            q = np.linalg.lstsq(K, b, rcond=None)[0]

        # Compute solvation energy
        # G_solv = (1/2) * Σ_i q_i * V_solute(r_i)
        G_solv = 0.5 * np.dot(q, V_solute)

        result_pcm = PCMResult(
            solvation_energy=G_solv,
            solvation_energy_kcal=G_solv * 627.5095,  # Hartree to kcal/mol
            surface_charges=q,
            reaction_field_energy=G_solv,
            cavity=cavity,
            epsilon=self.epsilon,
            solvent=self.solvent
        )

        if verbose:
            result_pcm.print_summary()

        return result_pcm

    def _compute_solute_potential(self, P: np.ndarray,
                                   result: Union[HFResult, DFTResult],
                                   cavity: MolecularCavity) -> np.ndarray:
        """
        Compute electrostatic potential from solute at cavity surface points.

        V(r) = V_nuc(r) + V_elec(r)
        """
        atoms = result.atoms
        positions = cavity.get_positions()
        n_points = cavity.n_points

        V = np.zeros(n_points)

        # Nuclear contribution
        for element, pos in atoms:
            Z = _element_to_Z(element)
            pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR

            for i, r in enumerate(positions):
                dist = np.linalg.norm(r - pos_bohr)
                if dist > 1e-10:
                    V[i] += Z / dist

        # Electronic contribution (using density matrix)
        # V_elec(r) = -∫ ρ(r') / |r - r'| dr'
        # This requires evaluating basis functions at cavity points
        # Simplified: use Mulliken charges as point charges

        # Get approximate atomic charges from Mulliken analysis
        S = result.overlap if hasattr(result, 'overlap') else np.eye(P.shape[0])
        PS = P @ S
        n_basis = P.shape[0]

        # Distribute electrons to atoms (simplified)
        atomic_pops = []
        basis_per_atom = n_basis // len(atoms) if len(atoms) > 0 else n_basis

        for a, (element, pos) in enumerate(atoms):
            start = a * basis_per_atom
            end = min((a + 1) * basis_per_atom, n_basis)
            pop = np.trace(PS[start:end, start:end])
            atomic_pops.append(pop)

        # Electronic contribution from Mulliken charges
        for a, (element, pos) in enumerate(atoms):
            if a < len(atomic_pops):
                q_elec = atomic_pops[a]  # Number of electrons on atom
                pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR

                for i, r in enumerate(positions):
                    dist = np.linalg.norm(r - pos_bohr)
                    if dist > 1e-10:
                        V[i] -= q_elec / dist

        return V

    def _build_pcm_matrices(self, cavity: MolecularCavity) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build PCM S and D matrices.

        S_ij = ∫∫ 1/|s_i - s_j| ds_i ds_j  (Coulomb)
        D_ij = ∫∫ n_j · (s_i - s_j) / |s_i - s_j|³ ds_i ds_j  (dipole layer)
        """
        n = cavity.n_points
        positions = cavity.get_positions()
        areas = cavity.get_areas()
        normals = cavity.get_normals()

        S = np.zeros((n, n))
        D = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements (self-interaction)
                    # Use analytical formulas for sphere elements
                    S[i, i] = 1.0694 * np.sqrt(4 * np.pi / areas[i])
                    D[i, i] = -0.5 * S[i, i]  # Factor of -1/2 for self-term
                else:
                    # Off-diagonal elements
                    r_ij = positions[i] - positions[j]
                    dist = np.linalg.norm(r_ij)

                    if dist > 1e-10:
                        # Coulomb interaction
                        S[i, j] = areas[j] / dist

                        # Dipole layer (derivative of Coulomb)
                        D[i, j] = areas[j] * np.dot(normals[j], r_ij) / (dist ** 3)

        return S, D

    def compute_pcm_fock_contribution(self,
                                      result: Union[HFResult, DFTResult],
                                      cavity: MolecularCavity,
                                      surface_charges: np.ndarray) -> np.ndarray:
        """
        Compute PCM contribution to Fock matrix for SCF.

        V_PCM_μν = Σ_k q_k <μ|1/|r-r_k||ν>

        This allows self-consistent treatment of solvation.
        """
        # Simplified: return zero matrix
        # Full implementation would compute integrals with cavity charges
        n_basis = result.density.shape[0]
        return np.zeros((n_basis, n_basis))


def build_cavity(atoms: List[Tuple[str, Tuple[float, float, float]]],
                 scaling_factor: float = 1.2,
                 n_points_per_atom: int = 60) -> MolecularCavity:
    """
    Build molecular cavity using overlapping spheres (SAS).

    Uses Lebedev grid points on atomic spheres, removing points
    inside other spheres.

    Args:
        atoms: List of (element, (x, y, z)) in Angstrom
        scaling_factor: Scale factor for vdW radii
        n_points_per_atom: Target points per atom

    Returns:
        MolecularCavity with tesserae
    """
    # Get Lebedev points for a unit sphere
    unit_points, unit_weights = _lebedev_grid(n_points_per_atom)

    cavity_points = []
    total_area = 0.0

    for atom_idx, (element, pos) in enumerate(atoms):
        # Get atomic radius
        radius = VDW_RADII.get(element, 1.70) * scaling_factor
        radius_bohr = radius * ANGSTROM_TO_BOHR
        center = np.array(pos) * ANGSTROM_TO_BOHR

        # Generate points on this atomic sphere
        for pt, wt in zip(unit_points, unit_weights):
            point = center + radius_bohr * pt

            # Check if point is inside any other atomic sphere
            inside_other = False
            for other_idx, (other_elem, other_pos) in enumerate(atoms):
                if other_idx == atom_idx:
                    continue
                other_radius = VDW_RADII.get(other_elem, 1.70) * scaling_factor
                other_radius_bohr = other_radius * ANGSTROM_TO_BOHR
                other_center = np.array(other_pos) * ANGSTROM_TO_BOHR

                dist = np.linalg.norm(point - other_center)
                if dist < other_radius_bohr - 0.01:  # Small tolerance
                    inside_other = True
                    break

            if not inside_other:
                # This point is on the molecular surface
                normal = pt  # Outward normal for sphere
                area = wt * 4 * np.pi * radius_bohr ** 2

                cavity_points.append(CavityPoint(
                    position=point,
                    normal=normal,
                    area=area,
                    atom_index=atom_idx
                ))
                total_area += area

    return MolecularCavity(
        points=cavity_points,
        n_points=len(cavity_points),
        total_area=total_area
    )


def compute_solvation_energy(result: Union[HFResult, DFTResult],
                             solvent: str = 'water',
                             verbose: bool = True) -> PCMResult:
    """
    Compute solvation free energy using PCM.

    Convenience function for PCMSolver.

    Args:
        result: HF or DFT result
        solvent: Solvent name
        verbose: Print results

    Returns:
        PCMResult with solvation energies

    Example:
        hf = hartree_fock(atoms)
        solv = compute_solvation_energy(hf, solvent='water')
        print(f"Solvation energy: {solv.solvation_energy_kcal:.2f} kcal/mol")
    """
    solver = PCMSolver(solvent=solvent)
    return solver.compute(result, verbose=verbose)


def _lebedev_grid(n_target: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lebedev grid points on unit sphere.

    Returns approximate grid with n_target points.
    Uses simplified octahedral grid for now.
    """
    # Simplified: use Fibonacci spiral for approximately uniform points
    n = max(n_target, 6)

    indices = np.arange(n)
    phi = np.pi * (3 - np.sqrt(5))  # Golden angle

    y = 1 - (indices / (n - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * indices

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    points = np.column_stack([x, y, z])
    weights = np.ones(n) / n  # Equal weights (approximate)

    return points, weights


def _element_to_Z(element: str) -> int:
    """Convert element symbol to atomic number."""
    elements = {
        'H': 1, 'He': 2,
        'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    }
    return elements.get(element, 1)
