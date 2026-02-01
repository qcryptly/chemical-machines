"""
Møller-Plesset Perturbation Theory (MP2)

Second-order perturbation theory correction to HF energy.

The MP2 correlation energy is:
    E_MP2 = Σ_{ijab} |<ij||ab>|² / (ε_i + ε_j - ε_a - ε_b)

where:
    i, j are occupied orbitals
    a, b are virtual orbitals
    <ij||ab> = <ij|ab> - <ij|ba> are antisymmetrized two-electron integrals
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .hf import HFResult, HartreeFockSolver, hartree_fock
from ..basis import BasisSet


@dataclass
class MP2Result:
    """
    Results from an MP2 calculation.

    Attributes:
        energy_hf: HF reference energy
        energy_mp2: MP2 correlation energy
        energy_total: Total energy (HF + MP2)
        energy_scs: Spin-component scaled MP2 energy (optional)

        # Amplitudes
        t2_amplitudes: Double excitation amplitudes (nocc, nocc, nvirt, nvirt)

        # Orbital info
        n_occ: Number of occupied orbitals
        n_virt: Number of virtual orbitals

        # Reference calculation
        hf_result: Underlying HF result
    """
    energy_hf: float
    energy_mp2: float
    energy_total: float
    energy_scs: Optional[float] = None

    t2_amplitudes: np.ndarray = None

    n_occ: int = 0
    n_virt: int = 0

    hf_result: HFResult = None


def transform_eri_to_mo(C: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Transform ERI tensor from AO to MO basis.

    (pq|rs)_MO = Σ_μνλσ C_μp C_νq (μν|λσ)_AO C_λr C_σs

    Args:
        C: MO coefficient matrix (n_ao, n_mo)
        G: ERI tensor in AO basis (n_ao, n_ao, n_ao, n_ao)

    Returns:
        ERI tensor in MO basis
    """
    n_ao = G.shape[0]
    n_mo = C.shape[1]

    # Four-index transformation (N^5 scaling)
    # Transform indices one at a time

    # First half-transformation: (μν|λσ) -> (pν|λσ)
    G1 = np.einsum('mp,mnls->pnls', C, G)

    # Second: (pν|λσ) -> (pq|λσ)
    G2 = np.einsum('nq,pnls->pqls', C, G1)

    # Third: (pq|λσ) -> (pq|rσ)
    G3 = np.einsum('lr,pqls->pqrs', C, G2)

    # Fourth: (pq|rσ) -> (pq|rs)
    G_mo = np.einsum('ss2,pqrs->pqrs2', C, G3)

    # Handle the last index properly
    G_mo = np.einsum('lr,pqls->pqrs', C, G3)

    return G_mo


def compute_mp2_energy(orbital_energies: np.ndarray,
                       mo_eri: np.ndarray,
                       n_occ: int) -> Tuple[float, np.ndarray]:
    """
    Compute MP2 correlation energy.

    Args:
        orbital_energies: MO orbital energies
        mo_eri: Two-electron integrals in MO basis
        n_occ: Number of occupied orbitals

    Returns:
        (E_mp2, t2_amplitudes) tuple
    """
    n_mo = len(orbital_energies)
    n_virt = n_mo - n_occ

    eps_occ = orbital_energies[:n_occ]
    eps_virt = orbital_energies[n_occ:]

    # Extract occupied-virtual blocks
    # (ia|jb) where i,j are occupied and a,b are virtual
    g_oovv = mo_eri[:n_occ, :n_occ, n_occ:, n_occ:]

    # Build energy denominators
    # D_ijab = ε_i + ε_j - ε_a - ε_b
    e_i = eps_occ[:, np.newaxis, np.newaxis, np.newaxis]
    e_j = eps_occ[np.newaxis, :, np.newaxis, np.newaxis]
    e_a = eps_virt[np.newaxis, np.newaxis, :, np.newaxis]
    e_b = eps_virt[np.newaxis, np.newaxis, np.newaxis, :]

    D = e_i + e_j - e_a - e_b

    # MP2 amplitudes: t2_ijab = <ij|ab> / D_ijab
    t2 = g_oovv / D

    # MP2 energy
    # E_MP2 = Σ_ijab t_ijab * (2<ij|ab> - <ij|ba>)
    # For closed-shell: E_MP2 = Σ_ijab (2*<ij|ab> - <ij|ba>) * <ij|ab> / D

    # Using physicist's notation: <ij|ab> = (ia|jb) in chemist's notation
    g_iajb = mo_eri[:n_occ, n_occ:, :n_occ, n_occ:]  # (i,a,j,b)

    # Direct term: 2 * Σ_ijab |<ij|ab>|² / D
    # Exchange term: -Σ_ijab <ij|ab><ij|ba> / D

    # Transform to physicist notation for clarity
    # g_oovv[i,j,a,b] in chemist = (ij|ab)
    # For spin-adapted closed-shell MP2:
    # E_MP2 = Σ_ijab (ij|ab) * [2(ij|ab) - (ij|ba)] / D

    g_ijab = g_oovv  # (ij|ab)
    g_ijba = np.swapaxes(g_oovv, 2, 3)  # (ij|ba)

    E_mp2 = np.sum(g_ijab * (2 * g_ijab - g_ijba) / D)

    return E_mp2, t2


def compute_scs_mp2_energy(orbital_energies: np.ndarray,
                           mo_eri: np.ndarray,
                           n_occ: int,
                           c_os: float = 6/5,
                           c_ss: float = 1/3) -> float:
    """
    Compute spin-component scaled (SCS) MP2 energy.

    SCS-MP2 scales the opposite-spin and same-spin components:
    E_SCS-MP2 = c_os * E_os + c_ss * E_ss

    Default parameters: c_os = 6/5, c_ss = 1/3 (Grimme's SCS-MP2)

    Args:
        orbital_energies: MO orbital energies
        mo_eri: Two-electron integrals in MO basis
        n_occ: Number of occupied orbitals
        c_os: Opposite-spin scaling factor
        c_ss: Same-spin scaling factor

    Returns:
        SCS-MP2 correlation energy
    """
    n_mo = len(orbital_energies)

    eps_occ = orbital_energies[:n_occ]
    eps_virt = orbital_energies[n_occ:]

    g_oovv = mo_eri[:n_occ, :n_occ, n_occ:, n_occ:]

    # Energy denominators
    e_i = eps_occ[:, np.newaxis, np.newaxis, np.newaxis]
    e_j = eps_occ[np.newaxis, :, np.newaxis, np.newaxis]
    e_a = eps_virt[np.newaxis, np.newaxis, :, np.newaxis]
    e_b = eps_virt[np.newaxis, np.newaxis, np.newaxis, :]

    D = e_i + e_j - e_a - e_b

    g_ijab = g_oovv
    g_ijba = np.swapaxes(g_oovv, 2, 3)

    # Opposite-spin: (ij|ab)² / D
    E_os = np.sum(g_ijab * g_ijab / D)

    # Same-spin: (ij|ab) * [(ij|ab) - (ij|ba)] / D
    E_ss = np.sum(g_ijab * (g_ijab - g_ijba) / D)

    return c_os * E_os + c_ss * E_ss


def mp2(hf_result: HFResult = None,
        atoms: List[Tuple[str, Tuple[float, float, float]]] = None,
        basis: str = 'STO-3G',
        frozen_core: bool = False,
        scs: bool = False,
        verbose: bool = False) -> MP2Result:
    """
    Perform MP2 calculation.

    Can either take an existing HF result or run HF first.

    Args:
        hf_result: Pre-computed HF result (optional)
        atoms: Molecule geometry (required if hf_result not provided)
        basis: Basis set name
        frozen_core: Freeze core orbitals (not yet implemented)
        scs: Also compute SCS-MP2 energy
        verbose: Print info

    Returns:
        MP2Result with energies and amplitudes

    Example:
        # From HF result
        hf = hartree_fock(atoms, basis='cc-pVDZ')
        mp2_result = mp2(hf_result=hf)

        # Direct calculation
        mp2_result = mp2(atoms=[('H', (0,0,0)), ('H', (0.74,0,0))], basis='cc-pVDZ')
    """
    # Run HF if needed
    if hf_result is None:
        if atoms is None:
            raise ValueError("Either hf_result or atoms must be provided")
        hf_result = hartree_fock(atoms, basis=basis, verbose=verbose)

    if verbose:
        print(f"\nMP2 Calculation")
        print(f"HF energy: {hf_result.energy:.10f} Hartree")

    # Get orbital info
    n_occ = hf_result.n_electrons // 2
    n_basis = len(hf_result.orbital_energies)
    n_virt = n_basis - n_occ

    if verbose:
        print(f"Occupied orbitals: {n_occ}")
        print(f"Virtual orbitals: {n_virt}")

    # Transform integrals to MO basis
    C = hf_result.mo_coefficients
    G_ao = hf_result.G

    if verbose:
        print("Transforming integrals to MO basis...")

    # Transform ERI: (μν|λσ) -> (pq|rs)
    # This is the expensive step: O(N^5)
    mo_eri = np.einsum('mp,nq,mnls,lr,su->pqru',
                       C, C, G_ao, C, C, optimize=True)

    # Compute MP2 energy
    if verbose:
        print("Computing MP2 correlation energy...")

    E_mp2, t2 = compute_mp2_energy(
        hf_result.orbital_energies,
        mo_eri,
        n_occ
    )

    E_total = hf_result.energy + E_mp2

    if verbose:
        print(f"\nMP2 correlation energy: {E_mp2:.10f} Hartree")
        print(f"Total MP2 energy: {E_total:.10f} Hartree")

    # SCS-MP2 if requested
    E_scs = None
    if scs:
        E_scs_corr = compute_scs_mp2_energy(
            hf_result.orbital_energies,
            mo_eri,
            n_occ
        )
        E_scs = hf_result.energy + E_scs_corr

        if verbose:
            print(f"SCS-MP2 correlation energy: {E_scs_corr:.10f} Hartree")
            print(f"Total SCS-MP2 energy: {E_scs:.10f} Hartree")

    return MP2Result(
        energy_hf=hf_result.energy,
        energy_mp2=E_mp2,
        energy_total=E_total,
        energy_scs=E_scs,
        t2_amplitudes=t2,
        n_occ=n_occ,
        n_virt=n_virt,
        hf_result=hf_result
    )
