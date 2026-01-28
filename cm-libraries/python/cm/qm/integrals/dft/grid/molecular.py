"""
Molecular Integration Grids

Combines atomic grids into molecular grids using Becke partitioning.

Reference:
- Becke, J. Chem. Phys. 88, 2547 (1988)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .radial import RadialGrid, BRAGG_SLATER_RADII_BOHR
from .angular import LebedevGrid, get_lebedev_grid


ANGSTROM_TO_BOHR = 1.8897259886


@dataclass
class AtomicGrid:
    """
    Integration grid centered on a single atom.

    Attributes:
        points: Grid points in Bohr (n, 3)
        weights: Quadrature weights (n,)
        center: Atom position in Bohr (3,)
        element: Element symbol
        atom_index: Index of this atom in the molecule
    """
    points: np.ndarray
    weights: np.ndarray
    center: np.ndarray
    element: str
    atom_index: int

    @classmethod
    def build(cls, element: str, center: np.ndarray, atom_index: int,
              radial_points: int = 75, angular_order: int = 302) -> 'AtomicGrid':
        """
        Build atomic grid for a single atom.

        Args:
            element: Element symbol
            center: Atom position in Bohr
            atom_index: Index of this atom
            radial_points: Number of radial points
            angular_order: Lebedev angular grid order

        Returns:
            AtomicGrid for this atom
        """
        # Get radial and angular grids
        radial = RadialGrid.for_element(element, radial_points)
        angular = get_lebedev_grid(angular_order)

        # Build full 3D grid
        n_total = radial.n_points * angular.n_points
        points = np.zeros((n_total, 3))
        weights = np.zeros(n_total)

        idx = 0
        for i_r, (r, w_r) in enumerate(zip(radial.points, radial.weights)):
            for i_a, (ang_pt, w_a) in enumerate(zip(angular.points, angular.weights)):
                # Point in 3D: center + r * angular_direction
                points[idx] = center + r * ang_pt
                # Weight: radial_weight * angular_weight * 4π (sphere factor)
                weights[idx] = w_r * w_a * 4 * np.pi
                idx += 1

        return cls(
            points=points,
            weights=weights,
            center=center,
            element=element,
            atom_index=atom_index
        )


@dataclass
class MolecularGrid:
    """
    Molecular integration grid with Becke partitioning.

    Combines atomic grids into a single molecular grid where each
    point has a weight determined by Becke's fuzzy cell partitioning.

    Attributes:
        points: All grid points (N, 3) in Bohr
        weights: Combined weights including Becke partitioning (N,)
        atom_indices: Which atom each point belongs to (N,)
        n_points: Total number of grid points
        atomic_grids: List of underlying atomic grids
    """
    points: np.ndarray
    weights: np.ndarray
    atom_indices: np.ndarray
    n_points: int
    atomic_grids: List[AtomicGrid]

    @classmethod
    def build(cls, atoms: List[Tuple[str, Tuple[float, float, float]]],
              radial_points: int = 75, angular_order: int = 302,
              pruning: str = 'sg1') -> 'MolecularGrid':
        """
        Build molecular grid with Becke partitioning.

        Args:
            atoms: List of (element, (x, y, z)) in Angstroms
            radial_points: Number of radial points per atom
            angular_order: Lebedev angular grid order
            pruning: Grid pruning scheme ('none', 'sg0', 'sg1')

        Returns:
            MolecularGrid for the molecule
        """
        n_atoms = len(atoms)

        # Convert atom positions to Bohr
        atom_centers = []
        elements = []
        for elem, pos in atoms:
            center = np.array(pos) * ANGSTROM_TO_BOHR
            atom_centers.append(center)
            elements.append(elem)

        atom_centers = np.array(atom_centers)

        # Build atomic grids
        atomic_grids = []
        for i, (elem, center) in enumerate(zip(elements, atom_centers)):
            # Apply pruning if requested
            n_ang = angular_order
            if pruning == 'sg1':
                n_ang = _sg1_angular_order(elem, radial_points)
            elif pruning == 'sg0':
                n_ang = min(angular_order, 194)

            grid = AtomicGrid.build(elem, center, i, radial_points, n_ang)
            atomic_grids.append(grid)

        # Combine grids and apply Becke partitioning
        all_points = []
        all_weights = []
        all_atom_indices = []

        for grid in atomic_grids:
            # Compute Becke weights for this atom's grid points
            becke_weights = _becke_weights(
                grid.points, grid.atom_index, atom_centers, elements
            )

            # Combined weights
            combined_weights = grid.weights * becke_weights

            all_points.append(grid.points)
            all_weights.append(combined_weights)
            all_atom_indices.append(np.full(len(grid.points), grid.atom_index))

        points = np.vstack(all_points)
        weights = np.concatenate(all_weights)
        atom_indices = np.concatenate(all_atom_indices)

        return cls(
            points=points,
            weights=weights,
            atom_indices=atom_indices,
            n_points=len(weights),
            atomic_grids=atomic_grids
        )


def _becke_weights(points: np.ndarray, atom_idx: int,
                   atom_centers: np.ndarray, elements: List[str]) -> np.ndarray:
    """
    Compute Becke partition weights for grid points.

    Uses the standard Becke partitioning with atomic size adjustments
    based on Bragg-Slater radii.

    Args:
        points: Grid points (n, 3)
        atom_idx: Index of the atom these points belong to
        atom_centers: All atom positions (n_atoms, 3)
        elements: Element symbols for all atoms

    Returns:
        Becke weights for each point (n,)
    """
    n_points = len(points)
    n_atoms = len(atom_centers)

    if n_atoms == 1:
        return np.ones(n_points)

    # Get Bragg-Slater radii for atomic size adjustments
    radii = np.array([BRAGG_SLATER_RADII_BOHR.get(e, 1.5) for e in elements])

    # Compute distances from each point to each atom
    # dist[i, j] = distance from point i to atom j
    dist = np.zeros((n_points, n_atoms))
    for j in range(n_atoms):
        diff = points - atom_centers[j]
        dist[:, j] = np.linalg.norm(diff, axis=1)

    # Compute cell functions P_A(r) for each atom
    # P_A(r) = Π_{B≠A} s(μ_AB)
    # where μ_AB = (r_A - r_B) / R_AB is the elliptical coordinate
    # and s(μ) is a smoothed step function

    cell_functions = np.ones((n_points, n_atoms))

    for A in range(n_atoms):
        for B in range(n_atoms):
            if A == B:
                continue

            # Distance between atoms A and B
            R_AB = np.linalg.norm(atom_centers[A] - atom_centers[B])
            if R_AB < 1e-10:
                continue

            # Elliptical coordinate: μ_AB = (r_A - r_B) / R_AB
            mu = (dist[:, A] - dist[:, B]) / R_AB

            # Atomic size adjustment
            chi = radii[A] / radii[B]
            u = (chi - 1) / (chi + 1)
            a = u / (u * u - 1)
            a = np.clip(a, -0.5, 0.5)

            # Adjusted μ
            nu = mu + a * (1 - mu * mu)

            # Becke's switching function (3 iterations)
            s = _becke_switch(nu)

            cell_functions[:, A] *= s

    # Normalize: w_A = P_A / Σ_B P_B
    total = np.sum(cell_functions, axis=1)
    total = np.maximum(total, 1e-20)  # Avoid division by zero

    weights = cell_functions[:, atom_idx] / total

    return weights


def _becke_switch(mu: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    Becke's smoothed step function.

    s(μ) = 1/2 (1 - f(f(f(μ))))
    where f(x) = 3/2 x - 1/2 x³

    Args:
        mu: Elliptical coordinate values
        iterations: Number of switching function iterations (default 3)

    Returns:
        Step function values between 0 and 1
    """
    p = mu
    for _ in range(iterations):
        p = 1.5 * p - 0.5 * p ** 3

    return 0.5 * (1 - p)


def _sg1_angular_order(element: str, n_radial: int) -> int:
    """
    Get angular grid order for SG-1 pruning scheme.

    SG-1 uses fewer angular points near the nucleus and far from the atom.

    Returns:
        Lebedev grid order
    """
    # Simplified SG-1: use 302 for most regions
    # A full implementation would vary angular points with radial distance
    return 302
