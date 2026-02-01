"""
Dipole Moment Calculation

Computes the electric dipole moment from the electron density.

The dipole moment has electronic and nuclear contributions:
    μ = μ_elec + μ_nuc
    μ_elec = -∫ ρ(r) r dr = -Tr[P D]
    μ_nuc = Σ_A Z_A R_A

where D_μν = <μ|r|ν> is the dipole integral matrix.
"""

import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass

from ..methods.hf import HFResult
from ..methods.dft import DFTResult
from ..basis import BasisSet


ANGSTROM_TO_BOHR = 1.8897259886
DEBYE_PER_AU = 2.541746  # 1 a.u. = 2.541746 Debye


@dataclass
class DipoleResult:
    """
    Result of dipole moment calculation.

    Attributes:
        dipole: Dipole moment vector in Debye (3,)
        dipole_au: Dipole moment in atomic units (3,)
        magnitude: Dipole magnitude in Debye
        magnitude_au: Dipole magnitude in atomic units

        # Components
        dipole_electronic: Electronic contribution
        dipole_nuclear: Nuclear contribution
    """
    dipole: np.ndarray
    dipole_au: np.ndarray
    magnitude: float
    magnitude_au: float

    dipole_electronic: np.ndarray = None
    dipole_nuclear: np.ndarray = None


def dipole_moment(result: Union[HFResult, DFTResult],
                  verbose: bool = False) -> DipoleResult:
    """
    Compute electric dipole moment.

    Args:
        result: HF or DFT result with density matrix
        verbose: Print dipole components

    Returns:
        DipoleResult with dipole moment

    Example:
        hf = hartree_fock(atoms)
        dipole = dipole_moment(hf)
        print(f"Dipole moment: {dipole.magnitude:.4f} Debye")
    """
    atoms = result.atoms
    P = result.density

    # Build basis for dipole integrals
    # Note: This is simplified - full implementation uses analytic integrals
    n_atoms = len(atoms)

    # Nuclear contribution
    mu_nuc = np.zeros(3)
    for element, pos in atoms:
        Z = _element_to_Z(element)
        pos_bohr = np.array(pos) * ANGSTROM_TO_BOHR
        mu_nuc += Z * pos_bohr

    # Electronic contribution: μ_elec = -Tr[P D]
    # D_μν = <μ|r|ν> are dipole integrals
    # Simplified: use expectation value of position operator
    if hasattr(result, 'basis') and result.basis is not None:
        basis = result.basis
    elif hasattr(result, 'basis_name') and result.basis_name:
        basis = BasisSet(result.basis_name)
        basis.build_for_molecule(atoms)
    else:
        basis = BasisSet('STO-3G')
        basis.build_for_molecule(atoms)

    D_x, D_y, D_z = _compute_dipole_integrals(basis)

    mu_elec = np.zeros(3)
    mu_elec[0] = -np.sum(P * D_x)
    mu_elec[1] = -np.sum(P * D_y)
    mu_elec[2] = -np.sum(P * D_z)

    # Total dipole
    mu_au = mu_elec + mu_nuc
    mu_debye = mu_au * DEBYE_PER_AU

    magnitude_au = np.linalg.norm(mu_au)
    magnitude = magnitude_au * DEBYE_PER_AU

    if verbose:
        print("\nDipole Moment Analysis")
        print("-" * 40)
        print(f"Nuclear contribution (a.u.):")
        print(f"  X: {mu_nuc[0]:10.6f}")
        print(f"  Y: {mu_nuc[1]:10.6f}")
        print(f"  Z: {mu_nuc[2]:10.6f}")
        print(f"\nElectronic contribution (a.u.):")
        print(f"  X: {mu_elec[0]:10.6f}")
        print(f"  Y: {mu_elec[1]:10.6f}")
        print(f"  Z: {mu_elec[2]:10.6f}")
        print(f"\nTotal dipole moment:")
        print(f"  X: {mu_debye[0]:10.6f} Debye")
        print(f"  Y: {mu_debye[1]:10.6f} Debye")
        print(f"  Z: {mu_debye[2]:10.6f} Debye")
        print(f"  |μ|: {magnitude:10.6f} Debye")

    return DipoleResult(
        dipole=mu_debye,
        dipole_au=mu_au,
        magnitude=magnitude,
        magnitude_au=magnitude_au,
        dipole_electronic=mu_elec,
        dipole_nuclear=mu_nuc
    )


def _compute_dipole_integrals(basis: BasisSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dipole integral matrices.

    D_μν^x = <μ|x|ν>
    D_μν^y = <μ|y|ν>
    D_μν^z = <μ|z|ν>

    Uses numerical integration (full implementation would be analytic).
    """
    n_basis = basis.n_basis
    D_x = np.zeros((n_basis, n_basis))
    D_y = np.zeros((n_basis, n_basis))
    D_z = np.zeros((n_basis, n_basis))

    # Use quadrature for dipole integrals
    for mu, bf_mu in enumerate(basis.basis_functions):
        for nu, bf_nu in enumerate(basis.basis_functions):
            if mu > nu:
                D_x[mu, nu] = D_x[nu, mu]
                D_y[mu, nu] = D_y[nu, mu]
                D_z[mu, nu] = D_z[nu, mu]
                continue

            dx, dy, dz = _dipole_integral_element(bf_mu, bf_nu)
            D_x[mu, nu] = dx
            D_y[mu, nu] = dy
            D_z[mu, nu] = dz

            if mu != nu:
                D_x[nu, mu] = dx
                D_y[nu, mu] = dy
                D_z[nu, mu] = dz

    return D_x, D_y, D_z


def _dipole_integral_element(bf_a, bf_b) -> Tuple[float, float, float]:
    """
    Compute dipole integral between two basis functions.

    <a|r|b> = <a|r_A|b> + R_A <a|b>

    where R_A is the center of function a.
    Uses Obara-Saika-style recursion.
    """
    A = bf_a.center
    B = bf_b.center
    la, ma, na = bf_a.angular_momentum
    lb, mb, nb = bf_b.angular_momentum

    dx = dy = dz = 0.0

    for prim_a in bf_a.primitives:
        for prim_b in bf_b.primitives:
            alpha = prim_a.exponent
            beta = prim_b.exponent
            ca = prim_a.coefficient
            cb = prim_b.coefficient

            gamma = alpha + beta
            P = (alpha * A + beta * B) / gamma

            # Pre-exponential
            AB2 = np.sum((A - B) ** 2)
            K = np.exp(-alpha * beta / gamma * AB2)
            prefactor = ca * cb * K * (np.pi / gamma) ** 1.5

            # Base overlap
            Sx = _overlap_1d_cached(la, lb, A[0], B[0], P[0], gamma)
            Sy = _overlap_1d_cached(ma, mb, A[1], B[1], P[1], gamma)
            Sz = _overlap_1d_cached(na, nb, A[2], B[2], P[2], gamma)

            # Dipole integrals: <a|x|b> = <a+1_x|b>/2α + <a-1_x|b>/2α + Ax<a|b>
            # Using recurrence: <a|x|b> = P_x <a|b> + la/(2γ) <a-1|b> + lb/(2γ) <a|b-1>

            # X component
            Sx1 = _overlap_1d_cached(la + 1, lb, A[0], B[0], P[0], gamma)
            dx += prefactor * Sx1 * Sy * Sz

            # Y component
            Sy1 = _overlap_1d_cached(ma + 1, mb, A[1], B[1], P[1], gamma)
            dy += prefactor * Sx * Sy1 * Sz

            # Z component
            Sz1 = _overlap_1d_cached(na + 1, nb, A[2], B[2], P[2], gamma)
            dz += prefactor * Sx * Sy * Sz1

    return dx, dy, dz


def _overlap_1d_cached(i: int, j: int, Ax: float, Bx: float, Px: float,
                       gamma: float, cache: dict = None) -> float:
    """1D overlap with optional caching."""
    if i < 0 or j < 0:
        return 0.0
    if i == 0 and j == 0:
        return 1.0

    PA = Px - Ax
    PB = Px - Bx

    if i > 0:
        return PA * _overlap_1d_cached(i - 1, j, Ax, Bx, Px, gamma) + \
               (i - 1) / (2 * gamma) * _overlap_1d_cached(i - 2, j, Ax, Bx, Px, gamma) + \
               j / (2 * gamma) * _overlap_1d_cached(i - 1, j - 1, Ax, Bx, Px, gamma)
    else:
        return PB * _overlap_1d_cached(i, j - 1, Ax, Bx, Px, gamma) + \
               i / (2 * gamma) * _overlap_1d_cached(i - 1, j - 1, Ax, Bx, Px, gamma) + \
               (j - 1) / (2 * gamma) * _overlap_1d_cached(i, j - 2, Ax, Bx, Px, gamma)


def _element_to_Z(element: str) -> int:
    """Convert element symbol to atomic number."""
    elements = {
        'H': 1, 'He': 2,
        'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    }
    return elements.get(element, 1)
