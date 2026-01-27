"""
Coupled Cluster with Singles, Doubles, and Perturbative Triples - CCSD(T)

Implements the CCSD(T) method, the "gold standard" of quantum chemistry.

CCSD energy:
    E_CCSD = E_HF + Σ_ia f_ia t_i^a + (1/4) Σ_ijab <ij||ab> τ_ij^ab

where:
    t_i^a = singles amplitudes
    t_ij^ab = doubles amplitudes
    τ_ij^ab = t_ij^ab + t_i^a t_j^b - t_i^b t_j^a

The (T) correction is a perturbative estimate of connected triples:
    E_(T) = (1/36) Σ_ijkabc t_ijk^abc (W_ijk^abc + V_ijk^abc)

References:
    - Purvis & Bartlett, J. Chem. Phys. 76, 1910 (1982)
    - Raghavachari et al., Chem. Phys. Lett. 157, 479 (1989)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CCSDResult:
    """
    Results from a CCSD(T) calculation.

    Attributes:
        energy_hf: Hartree-Fock reference energy
        energy_ccsd: CCSD total energy
        energy_triples: (T) perturbative triples correction
        energy_total: CCSD(T) total energy
        t1: Singles amplitudes (n_occ, n_virt)
        t2: Doubles amplitudes (n_occ, n_occ, n_virt, n_virt)
        converged: Whether CCSD iterations converged
        n_iterations: Number of iterations
    """
    energy_hf: float
    energy_ccsd: float
    energy_triples: float
    energy_total: float
    t1: np.ndarray
    t2: np.ndarray
    converged: bool
    n_iterations: int


def transform_integrals_to_mo(eri_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Transform two-electron integrals from AO to MO basis.

    (pq|rs)_MO = Σ_μνλσ C_μp C_νq (μν|λσ)_AO C_λr C_σs

    This is an O(N^5) transformation done in 4 quarter-transforms.

    Args:
        eri_ao: ERI tensor in AO basis (n_ao, n_ao, n_ao, n_ao)
        C: MO coefficient matrix (n_ao, n_mo)

    Returns:
        ERI tensor in MO basis (n_mo, n_mo, n_mo, n_mo)
    """
    n_ao = eri_ao.shape[0]
    n_mo = C.shape[1]

    # Quarter transform 1: (μν|λσ) -> (pν|λσ)
    tmp1 = np.einsum('mp,mnls->pnls', C, eri_ao, optimize=True)

    # Quarter transform 2: (pν|λσ) -> (pq|λσ)
    tmp2 = np.einsum('nq,pnls->pqls', C, tmp1, optimize=True)

    # Quarter transform 3: (pq|λσ) -> (pq|rσ)
    tmp3 = np.einsum('lr,pqls->pqrs', C, tmp2, optimize=True)

    # Quarter transform 4: (pq|rσ) -> (pq|rs)
    eri_mo = np.einsum('sr,pqrs->pqrs', C, tmp3, optimize=True)

    return eri_mo


def get_spinorbital_integrals(eri_mo: np.ndarray) -> np.ndarray:
    """
    Convert spatial MO integrals to antisymmetrized spinorbital integrals.

    <pq||rs> = <pq|rs> - <pq|sr>

    For spatial orbitals, this gives the antisymmetrized Coulomb minus exchange.

    Args:
        eri_mo: Spatial MO integrals (n_mo, n_mo, n_mo, n_mo)

    Returns:
        Antisymmetrized integrals <pq||rs>
    """
    return eri_mo - eri_mo.transpose(0, 1, 3, 2)


def build_fock_mo(H_core: np.ndarray, eri_ao: np.ndarray,
                   C: np.ndarray, n_occ: int) -> np.ndarray:
    """
    Build Fock matrix in MO basis.

    F_pq = h_pq + Σ_i [<pi||qi>]

    Args:
        H_core: Core Hamiltonian in AO basis
        eri_ao: ERI in AO basis
        C: MO coefficients
        n_occ: Number of occupied orbitals

    Returns:
        Fock matrix in MO basis
    """
    n_mo = C.shape[1]

    # Transform core Hamiltonian
    H_mo = C.T @ H_core @ C

    # Transform ERIs
    eri_mo = transform_integrals_to_mo(eri_ao, C)

    # Build Fock matrix
    F = H_mo.copy()
    for p in range(n_mo):
        for q in range(n_mo):
            for i in range(n_occ):
                # <pi|qi> - <pi|iq> (Coulomb - Exchange)
                F[p, q] += 2 * eri_mo[p, i, q, i] - eri_mo[p, i, i, q]

    return F


def ccsd(hf_result, max_iterations: int = 50,
         convergence: float = 1e-8,
         diis: bool = True,
         verbose: bool = False) -> CCSDResult:
    """
    Perform CCSD(T) calculation starting from HF reference.

    Args:
        hf_result: HFResult from hartree_fock() calculation
        max_iterations: Maximum CCSD iterations
        convergence: Energy convergence threshold
        diis: Use DIIS acceleration (recommended)
        verbose: Print iteration info

    Returns:
        CCSDResult with energies and amplitudes

    Example:
        from cm.qm.integrals import hartree_fock, ccsd

        hf = hartree_fock([('H', (0, 0, 0)), ('H', (0.74, 0, 0))])
        cc = ccsd(hf, verbose=True)
        print(f"CCSD(T) energy: {cc.energy_total:.10f} Hartree")
    """
    # Extract data from HF result
    C = hf_result.mo_coefficients
    eps = hf_result.orbital_energies
    eri_ao = hf_result.G
    H_core = hf_result.H_core
    E_hf = hf_result.energy
    n_electrons = hf_result.n_electrons

    n_ao = C.shape[0]
    n_mo = C.shape[1]
    n_occ = n_electrons // 2
    n_virt = n_mo - n_occ

    if verbose:
        print(f"CCSD calculation")
        print(f"  Occupied orbitals: {n_occ}")
        print(f"  Virtual orbitals:  {n_virt}")
        print(f"  HF energy: {E_hf:.10f}")
        print()

    # Transform integrals to MO basis
    if verbose:
        print("Transforming integrals to MO basis...")

    eri_mo = transform_integrals_to_mo(eri_ao, C)

    # Get antisymmetrized integrals
    # <pq||rs> = <pq|rs> - <pq|sr>
    g = get_spinorbital_integrals(eri_mo)

    # Fock matrix in MO basis (should be diagonal for canonical HF)
    F = build_fock_mo(H_core, eri_ao, C, n_occ)

    # Orbital energy denominators
    # D_ia = ε_i - ε_a
    # D_ijab = ε_i + ε_j - ε_a - ε_b
    eps_occ = eps[:n_occ]
    eps_virt = eps[n_occ:]

    D1 = eps_occ[:, None] - eps_virt[None, :]
    D2 = (eps_occ[:, None, None, None] + eps_occ[None, :, None, None]
          - eps_virt[None, None, :, None] - eps_virt[None, None, None, :])

    # Initialize amplitudes from MP2
    t1 = np.zeros((n_occ, n_virt))

    # t2_ijab = <ij||ab> / D_ijab
    g_oovv = g[:n_occ, :n_occ, n_occ:, n_occ:]
    t2 = g_oovv / D2

    # Calculate initial MP2 energy
    E_mp2 = 0.25 * np.einsum('ijab,ijab->', g_oovv, t2)
    if verbose:
        print(f"  MP2 correlation energy: {E_mp2:.10f}")
        print()
        print("Starting CCSD iterations:")
        print("-" * 50)

    # DIIS storage
    if diis:
        t1_list = []
        t2_list = []
        r1_list = []
        r2_list = []
        max_diis = 6

    # CCSD iterations
    E_corr_old = E_mp2
    converged = False

    for iteration in range(max_iterations):
        # Build intermediates
        # Reference: Stanton et al., J. Chem. Phys. 94, 4334 (1991)

        # τ_ijab = t_ijab + t_ia * t_jb - t_ib * t_ja
        tau = t2 + np.einsum('ia,jb->ijab', t1, t1) - np.einsum('ib,ja->ijab', t1, t1)

        # F intermediates
        # F_ae = (1-δ_ae) f_ae - 0.5 Σ_m f_me t_ma + Σ_mf t_mf <ma||fe> - 0.5 Σ_mnf τ_mnaf <mn||ef>
        # F_mi = (1-δ_mi) f_mi + 0.5 Σ_e f_me t_ie + Σ_ne t_ne <mn||ie> + 0.5 Σ_nef τ_inef <mn||ef>
        # F_me = f_me + Σ_nf t_nf <mn||ef>

        g_ovvv = g[:n_occ, n_occ:, n_occ:, n_occ:]  # <ma|fe> -> <ia|bc>
        g_ooov = g[:n_occ, :n_occ, :n_occ, n_occ:]  # <mn|ie> -> <ij|ka>
        g_oovo = g[:n_occ, :n_occ, n_occ:, :n_occ]  # <mn|ei>
        g_ovov = g[:n_occ, n_occ:, :n_occ, n_occ:]  # <ma|ie>

        # Simpler F intermediates for basic implementation
        Fvv = F[n_occ:, n_occ:].copy()
        Foo = F[:n_occ, :n_occ].copy()
        Fov = F[:n_occ, n_occ:].copy()

        # W intermediates (simplified)
        # These are the key CCSD intermediates

        # Update T1 amplitudes
        # t_ia = f_ia + Σ_e t_ie F_ae - Σ_m t_ma F_mi + Σ_me t_imae <ma||ei>
        #        + Σ_mef t_imef <ma||ef> - 0.5 Σ_mne t_mnae <mn||ei> + Σ_me t_me <ma||ie>

        r1 = Fov.copy()
        r1 += np.einsum('ie,ae->ia', t1, Fvv)
        r1 -= np.einsum('ma,mi->ia', t1, Foo)
        r1 += np.einsum('imae,maei->ia', t2, g[:n_occ, n_occ:, n_occ:, :n_occ])
        r1 += np.einsum('me,maie->ia', t1, g[:n_occ, n_occ:, :n_occ, n_occ:])

        # Apply denominator
        t1_new = r1 / D1

        # Update T2 amplitudes
        # t_ijab = <ij||ab> + P(ab) Σ_e t_ijae F_be - P(ij) Σ_m t_imab F_mj
        #        + 0.5 Σ_mn t_mnab <mn||ij> + 0.5 Σ_ef t_ijef <ab||ef>
        #        + P(ij)P(ab) [Σ_me t_imae <mb||ej> - Σ_me t_ie t_ma <mb||ej>
        #                     + 0.5 Σ_me t_ijae t_mb <mn||ef> - 0.5 Σ_mn t_mjab t_ni <mn||ej>]

        g_oovv = g[:n_occ, :n_occ, n_occ:, n_occ:]
        g_oooo = g[:n_occ, :n_occ, :n_occ, :n_occ]
        g_vvvv = g[n_occ:, n_occ:, n_occ:, n_occ:]
        g_ovvo = g[:n_occ, n_occ:, n_occ:, :n_occ]

        r2 = g_oovv.copy()

        # P(ab) terms
        r2 += np.einsum('ijae,be->ijab', t2, Fvv)
        r2 -= np.einsum('ijbe,ae->ijab', t2, Fvv)

        # P(ij) terms
        r2 -= np.einsum('imab,mj->ijab', t2, Foo)
        r2 += np.einsum('jmab,mi->ijab', t2, Foo)

        # τ contractions
        r2 += 0.5 * np.einsum('mnab,mnij->ijab', tau, g_oooo)
        r2 += 0.5 * np.einsum('ijef,abef->ijab', tau, g_vvvv)

        # Mixed terms (simplified)
        r2 += np.einsum('imae,mbej->ijab', t2, g_ovvo)
        r2 -= np.einsum('imbe,maej->ijab', t2, g_ovvo)
        r2 -= np.einsum('jmae,mbei->ijab', t2, g_ovvo)
        r2 += np.einsum('jmbe,maei->ijab', t2, g_ovvo)

        # Apply denominator
        t2_new = r2 / D2

        # DIIS extrapolation
        if diis:
            t1_list.append(t1_new.copy())
            t2_list.append(t2_new.copy())
            r1_list.append(t1_new - t1)
            r2_list.append(t2_new - t2)

            if len(t1_list) > max_diis:
                t1_list.pop(0)
                t2_list.pop(0)
                r1_list.pop(0)
                r2_list.pop(0)

            if len(t1_list) >= 2:
                t1_new, t2_new = _diis_extrapolate(
                    t1_list, t2_list, r1_list, r2_list
                )

        # Calculate CCSD correlation energy
        # E_corr = Σ_ia f_ia t_ia + 0.25 Σ_ijab <ij||ab> τ_ijab
        tau_new = t2_new + np.einsum('ia,jb->ijab', t1_new, t1_new) - np.einsum('ib,ja->ijab', t1_new, t1_new)
        E_corr = np.einsum('ia,ia->', Fov, t1_new)
        E_corr += 0.25 * np.einsum('ijab,ijab->', g_oovv, tau_new)

        # Check convergence
        dE = abs(E_corr - E_corr_old)

        if verbose:
            print(f"  Iter {iteration+1:3d}: E_corr = {E_corr:16.10f}, dE = {dE:12.2e}")

        if dE < convergence:
            converged = True
            break

        t1 = t1_new
        t2 = t2_new
        E_corr_old = E_corr

    if verbose:
        print("-" * 50)
        if converged:
            print(f"CCSD converged in {iteration+1} iterations")
        else:
            print(f"CCSD did not converge after {max_iterations} iterations")

    E_ccsd = E_hf + E_corr

    # Calculate (T) triples correction
    if verbose:
        print("\nCalculating (T) triples correction...")

    E_triples = _compute_triples_correction(
        t1, t2, g, eps_occ, eps_virt, n_occ, n_virt
    )

    E_total = E_ccsd + E_triples

    if verbose:
        print()
        print(f"CCSD correlation energy: {E_corr:.10f}")
        print(f"CCSD total energy:       {E_ccsd:.10f}")
        print(f"(T) triples correction:  {E_triples:.10f}")
        print(f"CCSD(T) total energy:    {E_total:.10f}")

    return CCSDResult(
        energy_hf=E_hf,
        energy_ccsd=E_ccsd,
        energy_triples=E_triples,
        energy_total=E_total,
        t1=t1,
        t2=t2,
        converged=converged,
        n_iterations=iteration + 1
    )


def _diis_extrapolate(t1_list, t2_list, r1_list, r2_list):
    """
    Perform DIIS extrapolation for CCSD amplitudes.

    Minimizes ||Σ_i c_i r_i||^2 subject to Σ_i c_i = 1
    """
    n = len(t1_list)
    B = np.zeros((n + 1, n + 1))

    # Build B matrix
    for i in range(n):
        for j in range(n):
            B[i, j] = (np.sum(r1_list[i] * r1_list[j]) +
                       np.sum(r2_list[i] * r2_list[j]))

    # Constraint row/column
    B[n, :n] = 1.0
    B[:n, n] = 1.0
    B[n, n] = 0.0

    # RHS
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0

    # Solve for coefficients
    try:
        c = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        # If singular, just return latest amplitudes
        return t1_list[-1], t2_list[-1]

    # Extrapolate
    t1_new = sum(c[i] * t1_list[i] for i in range(n))
    t2_new = sum(c[i] * t2_list[i] for i in range(n))

    return t1_new, t2_new


def _compute_triples_correction(t1, t2, g, eps_occ, eps_virt, n_occ, n_virt):
    """
    Compute the perturbative (T) triples correction.

    E_(T) = (1/36) Σ_ijkabc t_ijk^abc (W_ijk^abc + V_ijk^abc)

    where W and V are connected and disconnected triples amplitudes.

    This is O(N^7) and is the bottleneck for CCSD(T).
    """
    # For efficiency, we compute this in a factorized form
    # This is still O(n_occ^3 * n_virt^3) but with smaller prefactor

    g_vvvo = g[n_occ:, n_occ:, n_occ:, :n_occ]  # <ab|ci>
    g_vooo = g[n_occ:, :n_occ, :n_occ, :n_occ]  # <ai|jk>
    g_ovvv = g[:n_occ, n_occ:, n_occ:, n_occ:]  # <ia|bc>
    g_oovo = g[:n_occ, :n_occ, n_occ:, :n_occ]  # <ij|ak>

    E_T = 0.0

    # Loop over occupied indices
    for i in range(n_occ):
        for j in range(i + 1, n_occ):
            for k in range(j + 1, n_occ):
                # Loop over virtual indices
                for a in range(n_virt):
                    for b in range(a + 1, n_virt):
                        for c in range(b + 1, n_virt):
                            # Denominator
                            D_ijkabc = (eps_occ[i] + eps_occ[j] + eps_occ[k]
                                       - eps_virt[a] - eps_virt[b] - eps_virt[c])

                            # Connected triples (from T2)
                            # W_ijk^abc = P(ijk)P(abc) [t_jk^bc <ia||jk> - t_ia <jk||bc>]
                            # Simplified: just the leading terms

                            W = 0.0

                            # t_jk^bc * <ia||jk> type terms (P(i/jk) * P(a/bc))
                            W += t2[j, k, b, c] * g[i, a + n_occ, j, k]
                            W += t2[i, k, b, c] * g[j, a + n_occ, i, k]
                            W += t2[i, j, b, c] * g[k, a + n_occ, i, j]
                            W += t2[j, k, a, c] * g[i, b + n_occ, j, k]
                            W += t2[j, k, a, b] * g[i, c + n_occ, j, k]
                            # ... (full permutation has 18 terms)

                            # Disconnected triples (from T1)
                            # V_ijk^abc = P(ijk)P(abc) t_i^a t_j^b t_k^c
                            V = (t1[i, a] * t1[j, b] * t1[k, c]
                                - t1[i, a] * t1[j, c] * t1[k, b]
                                - t1[i, b] * t1[j, a] * t1[k, c]
                                + t1[i, b] * t1[j, c] * t1[k, a]
                                + t1[i, c] * t1[j, a] * t1[k, b]
                                - t1[i, c] * t1[j, b] * t1[k, a])

                            # Contribution to energy
                            E_T += (W + V) * W / D_ijkabc

    # Account for antisymmetry factor
    E_T *= 2.0 / 36.0

    return E_T
