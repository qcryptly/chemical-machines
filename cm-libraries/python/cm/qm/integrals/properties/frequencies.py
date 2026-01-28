"""
Vibrational Frequency Analysis

Computes harmonic vibrational frequencies from the Hessian matrix.

The frequencies are obtained by:
1. Mass-weighting the Hessian
2. Projecting out translations and rotations
3. Diagonalizing to get normal modes
4. Converting eigenvalues to frequencies

ν_i = (1/2π) √(λ_i)  in atomic units
ν_i [cm⁻¹] = 5140.48 × √(λ_i [Hartree/(amu·Bohr²)])
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .hessian import compute_hessian, HessianResult, ATOMIC_MASSES


# Conversion factors
HARTREE_TO_WAVENUMBER = 219474.63  # Hartree to cm⁻¹
AMU_TO_ME = 1822.888  # amu to electron mass
BOHR_TO_ANGSTROM = 0.529177

# Combined conversion for frequency: sqrt(Hartree/(amu*Bohr²)) to cm⁻¹
FREQ_CONVERSION = 5140.48


@dataclass
class FrequencyResult:
    """
    Result of vibrational frequency analysis.

    Attributes:
        frequencies: Vibrational frequencies in cm⁻¹ (n_vib,)
        normal_modes: Normal mode displacement vectors (3N, n_vib)
        ir_intensities: IR intensities in km/mol (n_vib,)
        reduced_masses: Reduced masses in amu (n_vib,)
        force_constants: Force constants in mDyne/Å (n_vib,)
        zpe: Zero-point energy in Hartree
        n_imaginary: Number of imaginary frequencies

        # Raw data
        hessian_result: Underlying Hessian calculation
        eigenvalues: Raw eigenvalues
    """
    frequencies: np.ndarray
    normal_modes: np.ndarray
    ir_intensities: np.ndarray
    reduced_masses: np.ndarray
    force_constants: np.ndarray
    zpe: float
    n_imaginary: int

    hessian_result: HessianResult = None
    eigenvalues: np.ndarray = None


def harmonic_frequencies(atoms: List[Tuple[str, Tuple[float, float, float]]] = None,
                         hessian_result: HessianResult = None,
                         method: str = 'HF',
                         basis: str = 'STO-3G',
                         project_trans_rot: bool = True,
                         verbose: bool = False) -> FrequencyResult:
    """
    Compute harmonic vibrational frequencies.

    Args:
        atoms: Molecule geometry (required if no hessian_result)
        hessian_result: Pre-computed Hessian (optional)
        method: Electronic structure method
        basis: Basis set name
        project_trans_rot: Project out translations and rotations
        verbose: Print frequency table

    Returns:
        FrequencyResult with frequencies and normal modes

    Example:
        # Compute frequencies for water
        freq = harmonic_frequencies(
            atoms=[('O', (0, 0, 0)), ('H', (0.96, 0, 0)), ('H', (-0.24, 0.93, 0))],
            method='B3LYP',
            basis='6-31G*'
        )
        print(f"Frequencies: {freq.frequencies}")
        print(f"ZPE: {freq.zpe * 627.5:.2f} kcal/mol")
    """
    # Compute Hessian if needed
    if hessian_result is None:
        if atoms is None:
            raise ValueError("Either atoms or hessian_result required")
        hessian_result = compute_hessian(atoms, method, basis, verbose=verbose)
    else:
        atoms = hessian_result.atoms

    n_atoms = len(atoms)
    n_coords = 3 * n_atoms
    masses = hessian_result.masses

    # Number of vibrational modes (3N - 6 for nonlinear, 3N - 5 for linear)
    is_linear = _is_linear(atoms)
    n_trans_rot = 5 if is_linear else 6
    n_vib = n_coords - n_trans_rot

    # Get mass-weighted Hessian
    H_mw = hessian_result.hessian_mw

    if project_trans_rot:
        # Project out translations and rotations
        H_mw = _project_trans_rot(H_mw, atoms, masses)

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H_mw)

    # Sort by eigenvalue (skip first n_trans_rot which should be ~0)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip translation/rotation modes
    vib_eigenvalues = eigenvalues[n_trans_rot:]
    vib_eigenvectors = eigenvectors[:, n_trans_rot:]

    # Convert eigenvalues to frequencies
    frequencies = np.zeros(n_vib)
    n_imaginary = 0

    for i, ev in enumerate(vib_eigenvalues):
        if ev < 0:
            # Imaginary frequency (transition state or saddle point)
            frequencies[i] = -FREQ_CONVERSION * np.sqrt(abs(ev))
            n_imaginary += 1
        else:
            frequencies[i] = FREQ_CONVERSION * np.sqrt(ev)

    # Calculate reduced masses
    # μ_i = 1 / Σ_j (L_ij)² where L is mass-weighted eigenvector
    reduced_masses = np.zeros(n_vib)
    for i in range(n_vib):
        L = vib_eigenvectors[:, i]
        reduced_masses[i] = 1.0 / np.sum(L ** 2)

    # Force constants: k = μ ω² = μ (2πν)²
    # In mDyne/Å: k = 4π² c² ν² μ × (conversion factors)
    # Simplified: k ≈ 0.01 × (ν/100)² × μ for ν in cm⁻¹, μ in amu
    force_constants = np.zeros(n_vib)
    for i in range(n_vib):
        if frequencies[i] > 0:
            force_constants[i] = 5.891e-7 * frequencies[i] ** 2 * reduced_masses[i]

    # IR intensities (require dipole derivatives - placeholder)
    ir_intensities = np.zeros(n_vib)

    # Zero-point energy
    # ZPE = (1/2) Σ hν = (1/2) Σ ℏω
    # In Hartree: ZPE = (1/2) Σ ν[cm⁻¹] / HARTREE_TO_WAVENUMBER
    real_freqs = frequencies[frequencies > 0]
    zpe = 0.5 * np.sum(real_freqs) / HARTREE_TO_WAVENUMBER

    if verbose:
        print("\nVibrational Frequencies")
        print("-" * 60)
        print(f"{'Mode':>4} {'Freq (cm⁻¹)':>12} {'Red. Mass':>10} {'Force Const':>12}")
        print("-" * 60)
        for i in range(n_vib):
            freq_str = f"{frequencies[i]:12.2f}" if frequencies[i] > 0 else f"{frequencies[i]:11.2f}i"
            print(f"{i+1:4d} {freq_str} {reduced_masses[i]:10.4f} {force_constants[i]:12.4f}")
        print("-" * 60)
        print(f"Zero-point energy: {zpe:.6f} Hartree ({zpe * 627.5:.2f} kcal/mol)")
        if n_imaginary > 0:
            print(f"Warning: {n_imaginary} imaginary frequency(ies) found")

    return FrequencyResult(
        frequencies=frequencies,
        normal_modes=vib_eigenvectors,
        ir_intensities=ir_intensities,
        reduced_masses=reduced_masses,
        force_constants=force_constants,
        zpe=zpe,
        n_imaginary=n_imaginary,
        hessian_result=hessian_result,
        eigenvalues=vib_eigenvalues
    )


def normal_mode_analysis(freq_result: FrequencyResult,
                         mode_idx: int) -> Tuple[np.ndarray, float]:
    """
    Get displacement vectors for a specific normal mode.

    Args:
        freq_result: Frequency calculation result
        mode_idx: Index of normal mode (0-based)

    Returns:
        (displacements, frequency) tuple
        displacements: (n_atoms, 3) displacement vectors
    """
    n_atoms = len(freq_result.hessian_result.atoms)
    mode = freq_result.normal_modes[:, mode_idx]

    # Un-mass-weight the mode
    masses = freq_result.hessian_result.masses
    mass_weights = np.repeat(masses, 3)
    mode_cart = mode / np.sqrt(mass_weights)

    # Normalize
    mode_cart = mode_cart / np.linalg.norm(mode_cart)

    displacements = mode_cart.reshape(n_atoms, 3)
    frequency = freq_result.frequencies[mode_idx]

    return displacements, frequency


def _is_linear(atoms: List[Tuple[str, Tuple[float, float, float]]],
               threshold: float = 1e-6) -> bool:
    """Check if molecule is linear."""
    if len(atoms) <= 2:
        return True

    coords = np.array([a[1] for a in atoms])

    # Check if all atoms are collinear
    v1 = coords[1] - coords[0]
    v1 = v1 / np.linalg.norm(v1)

    for i in range(2, len(atoms)):
        v2 = coords[i] - coords[0]
        v2 = v2 / np.linalg.norm(v2)

        # Check if parallel (cross product ~ 0)
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) > threshold:
            return False

    return True


def _project_trans_rot(H_mw: np.ndarray,
                       atoms: List[Tuple[str, Tuple[float, float, float]]],
                       masses: np.ndarray) -> np.ndarray:
    """
    Project out translations and rotations from mass-weighted Hessian.

    Uses Sayvetz conditions to construct projection matrix.
    """
    n_atoms = len(atoms)
    n_coords = 3 * n_atoms

    coords = np.array([a[1] for a in atoms])
    mass_weights = np.repeat(masses, 3)

    # Center of mass
    total_mass = np.sum(masses)
    com = np.sum(masses[:, np.newaxis] * coords, axis=0) / total_mass

    # Shift to center of mass
    coords_com = coords - com

    # Build translation vectors (3)
    D_trans = np.zeros((n_coords, 3))
    for i in range(n_atoms):
        for j in range(3):
            D_trans[3*i + j, j] = np.sqrt(masses[i])

    # Build rotation vectors (3 for nonlinear, 2 for linear)
    D_rot = np.zeros((n_coords, 3))
    for i in range(n_atoms):
        r = coords_com[i]
        m_sqrt = np.sqrt(masses[i])

        # Rotation about x: (0, z, -y)
        D_rot[3*i + 1, 0] = m_sqrt * r[2]
        D_rot[3*i + 2, 0] = -m_sqrt * r[1]

        # Rotation about y: (-z, 0, x)
        D_rot[3*i + 0, 1] = -m_sqrt * r[2]
        D_rot[3*i + 2, 1] = m_sqrt * r[0]

        # Rotation about z: (y, -x, 0)
        D_rot[3*i + 0, 2] = m_sqrt * r[1]
        D_rot[3*i + 1, 2] = -m_sqrt * r[0]

    # Combine and orthonormalize
    D = np.hstack([D_trans, D_rot])

    # Gram-Schmidt orthonormalization
    Q, R = np.linalg.qr(D)

    # Keep only linearly independent vectors
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    Q = Q[:, :rank]

    # Projection matrix: P = I - Q Q^T
    P = np.eye(n_coords) - Q @ Q.T

    # Project Hessian
    H_proj = P @ H_mw @ P

    return H_proj
