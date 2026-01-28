"""
Internal Coordinates

Provides transformations between Cartesian and internal coordinates
for more efficient geometry optimization.

Internal coordinates include:
- Bond lengths (stretches)
- Bond angles (bends)
- Dihedral angles (torsions)
- Out-of-plane angles (improper torsions)

The Wilson B-matrix relates internal and Cartesian coordinates:
    q = B·x  (internal = B × Cartesian)
    dq = B·dx

Reference: Wilson, Decius, Cross, "Molecular Vibrations" (1955)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InternalCoord:
    """Base class for internal coordinates."""
    atoms: Tuple[int, ...]  # Atom indices involved
    value: float = 0.0      # Current value


@dataclass
class BondStretch(InternalCoord):
    """Bond length between two atoms."""
    pass


@dataclass
class BondAngle(InternalCoord):
    """Angle between three atoms (A-B-C)."""
    pass


@dataclass
class DihedralAngle(InternalCoord):
    """Dihedral angle between four atoms (A-B-C-D)."""
    pass


def bond_length(coords: np.ndarray, i: int, j: int) -> float:
    """
    Compute bond length between atoms i and j.

    Args:
        coords: Cartesian coordinates (n_atoms, 3)
        i, j: Atom indices

    Returns:
        Bond length
    """
    return np.linalg.norm(coords[i] - coords[j])


def bond_angle(coords: np.ndarray, i: int, j: int, k: int) -> float:
    """
    Compute bond angle A-B-C (angle at B).

    Args:
        coords: Cartesian coordinates (n_atoms, 3)
        i, j, k: Atom indices (angle at j)

    Returns:
        Angle in radians
    """
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cos_angle = np.dot(v1, v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


def dihedral_angle(coords: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """
    Compute dihedral angle A-B-C-D.

    The dihedral is the angle between planes ABC and BCD.

    Args:
        coords: Cartesian coordinates (n_atoms, 3)
        i, j, k, l: Atom indices

    Returns:
        Dihedral angle in radians (-π to π)
    """
    b1 = coords[j] - coords[i]
    b2 = coords[k] - coords[j]
    b3 = coords[l] - coords[k]

    # Normal vectors to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # Angle between normals
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Sign from cross product
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    sign = np.sign(np.dot(m1, n2))

    return sign * np.arccos(cos_angle)


class InternalCoordinates:
    """
    Internal coordinate system for a molecule.

    Automatically generates internal coordinates from connectivity
    and provides transformations to/from Cartesian.
    """

    def __init__(self, coords: np.ndarray, connectivity: List[Tuple[int, int]] = None):
        """
        Initialize internal coordinates.

        Args:
            coords: Cartesian coordinates (n_atoms, 3)
            connectivity: List of bonded atom pairs (auto-detected if None)
        """
        self.n_atoms = len(coords)
        self.coords = coords.copy()

        # Detect or use provided connectivity
        if connectivity is None:
            connectivity = self._detect_connectivity(coords)
        self.connectivity = connectivity

        # Generate internal coordinates
        self.internals = self._generate_internals()

    def _detect_connectivity(self, coords: np.ndarray,
                             threshold: float = 3.0) -> List[Tuple[int, int]]:
        """Detect bonds based on distance threshold (in Bohr)."""
        connectivity = []
        n = len(coords)

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < threshold:
                    connectivity.append((i, j))

        return connectivity

    def _generate_internals(self) -> List[InternalCoord]:
        """Generate complete set of internal coordinates."""
        internals = []

        # Bond stretches
        for i, j in self.connectivity:
            r = bond_length(self.coords, i, j)
            internals.append(BondStretch(atoms=(i, j), value=r))

        # Bond angles (for each pair of bonds sharing an atom)
        for j in range(self.n_atoms):
            # Find all atoms bonded to j
            bonded = []
            for i, k in self.connectivity:
                if i == j:
                    bonded.append(k)
                elif k == j:
                    bonded.append(i)

            # Create angles for all pairs
            for idx1 in range(len(bonded)):
                for idx2 in range(idx1 + 1, len(bonded)):
                    i, k = bonded[idx1], bonded[idx2]
                    theta = bond_angle(self.coords, i, j, k)
                    internals.append(BondAngle(atoms=(i, j, k), value=theta))

        # Dihedral angles (for each pair of connected bonds)
        for bond1 in self.connectivity:
            for bond2 in self.connectivity:
                if bond1 >= bond2:
                    continue

                # Check if bonds share an atom
                shared = set(bond1) & set(bond2)
                if len(shared) == 1:
                    j = shared.pop()
                    # Get other atoms
                    i = bond1[0] if bond1[1] == j else bond1[1]
                    k = bond2[0] if bond2[1] == j else bond2[1]

                    # Find atoms bonded to k (but not j)
                    for bond3 in self.connectivity:
                        if k in bond3 and j not in bond3:
                            l = bond3[0] if bond3[1] == k else bond3[1]
                            phi = dihedral_angle(self.coords, i, j, k, l)
                            internals.append(DihedralAngle(
                                atoms=(i, j, k, l), value=phi
                            ))
                            break

        return internals

    def wilson_b_matrix(self) -> np.ndarray:
        """
        Compute Wilson B-matrix.

        B_ij = ∂q_i / ∂x_j

        where q are internal coordinates and x are Cartesian.

        Returns:
            B-matrix (n_internal, 3*n_atoms)
        """
        n_int = len(self.internals)
        n_cart = 3 * self.n_atoms

        B = np.zeros((n_int, n_cart))

        for i, internal in enumerate(self.internals):
            if isinstance(internal, BondStretch):
                j, k = internal.atoms
                B[i] = self._bond_b_row(j, k)
            elif isinstance(internal, BondAngle):
                j, k, l = internal.atoms
                B[i] = self._angle_b_row(j, k, l)
            elif isinstance(internal, DihedralAngle):
                j, k, l, m = internal.atoms
                B[i] = self._dihedral_b_row(j, k, l, m)

        return B

    def _bond_b_row(self, i: int, j: int) -> np.ndarray:
        """B-matrix row for bond stretch."""
        row = np.zeros(3 * self.n_atoms)

        r_ij = self.coords[j] - self.coords[i]
        r = np.linalg.norm(r_ij)
        e_ij = r_ij / r

        # ∂r/∂x_i = -e_ij, ∂r/∂x_j = e_ij
        row[3*i:3*i+3] = -e_ij
        row[3*j:3*j+3] = e_ij

        return row

    def _angle_b_row(self, i: int, j: int, k: int) -> np.ndarray:
        """B-matrix row for bond angle."""
        row = np.zeros(3 * self.n_atoms)

        r_ji = self.coords[i] - self.coords[j]
        r_jk = self.coords[k] - self.coords[j]

        r1 = np.linalg.norm(r_ji)
        r2 = np.linalg.norm(r_jk)

        e1 = r_ji / r1
        e2 = r_jk / r2

        cos_theta = np.dot(e1, e2)
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        if sin_theta < 1e-10:
            return row  # Linear angle, undefined

        # Derivatives
        d_i = (cos_theta * e1 - e2) / (r1 * sin_theta)
        d_k = (cos_theta * e2 - e1) / (r2 * sin_theta)
        d_j = -d_i - d_k

        row[3*i:3*i+3] = d_i
        row[3*j:3*j+3] = d_j
        row[3*k:3*k+3] = d_k

        return row

    def _dihedral_b_row(self, i: int, j: int, k: int, l: int) -> np.ndarray:
        """B-matrix row for dihedral angle."""
        row = np.zeros(3 * self.n_atoms)

        # Vectors along bonds
        b1 = self.coords[j] - self.coords[i]
        b2 = self.coords[k] - self.coords[j]
        b3 = self.coords[l] - self.coords[k]

        # Cross products
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        b2_norm = np.linalg.norm(b2)

        if n1_norm < 1e-10 or n2_norm < 1e-10:
            return row  # Undefined dihedral

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        e2 = b2 / b2_norm

        # Derivatives (simplified)
        # Full derivation is complex - using numerical for robustness
        h = 1e-7
        coords_orig = self.coords.copy()

        for atom_idx, offset in [(i, 0), (j, 1), (k, 2), (l, 3)]:
            for xyz in range(3):
                self.coords[atom_idx, xyz] += h
                phi_plus = dihedral_angle(self.coords, i, j, k, l)
                self.coords[atom_idx, xyz] -= 2 * h
                phi_minus = dihedral_angle(self.coords, i, j, k, l)
                self.coords[atom_idx, xyz] += h

                row[3*atom_idx + xyz] = (phi_plus - phi_minus) / (2 * h)

        self.coords = coords_orig

        return row

    def cartesian_to_internal(self, x: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian coordinates to internal.

        Args:
            x: Cartesian coordinates (n_atoms, 3) or (3*n_atoms,)

        Returns:
            Internal coordinates (n_internal,)
        """
        if x.ndim == 1:
            x = x.reshape(-1, 3)

        self.coords = x.copy()
        self.internals = self._generate_internals()

        return np.array([ic.value for ic in self.internals])

    def internal_to_cartesian(self, q: np.ndarray,
                              max_iter: int = 50) -> np.ndarray:
        """
        Convert internal coordinates to Cartesian.

        Uses iterative back-transformation.

        Args:
            q: Target internal coordinates
            max_iter: Maximum iterations

        Returns:
            Cartesian coordinates (n_atoms, 3)
        """
        x = self.coords.copy()

        for _ in range(max_iter):
            # Current internal coords
            q_current = self.cartesian_to_internal(x)

            # Difference
            dq = q - q_current

            if np.max(np.abs(dq)) < 1e-10:
                break

            # B-matrix and pseudo-inverse
            B = self.wilson_b_matrix()
            B_inv = np.linalg.pinv(B)

            # Update Cartesian
            dx = B_inv @ dq
            x += dx.reshape(-1, 3)

        return x

    def gradient_transform(self, g_cart: np.ndarray) -> np.ndarray:
        """
        Transform gradient from Cartesian to internal coordinates.

        g_int = (B·Bᵀ)^(-1) · B · g_cart

        Args:
            g_cart: Cartesian gradient (3*n_atoms,)

        Returns:
            Internal gradient (n_internal,)
        """
        B = self.wilson_b_matrix()
        G = B @ B.T
        G_inv = np.linalg.pinv(G)

        return G_inv @ B @ g_cart

    def hessian_transform(self, H_cart: np.ndarray) -> np.ndarray:
        """
        Transform Hessian from Cartesian to internal coordinates.

        H_int = (B·Bᵀ)^(-1) · B · H_cart · Bᵀ · (B·Bᵀ)^(-1)

        Note: This ignores the derivative of B terms.

        Args:
            H_cart: Cartesian Hessian (3*n_atoms, 3*n_atoms)

        Returns:
            Internal Hessian (n_internal, n_internal)
        """
        B = self.wilson_b_matrix()
        G = B @ B.T
        G_inv = np.linalg.pinv(G)

        return G_inv @ B @ H_cart @ B.T @ G_inv
