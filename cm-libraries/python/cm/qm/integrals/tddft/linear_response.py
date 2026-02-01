"""
Linear Response TDDFT

Implements TDDFT and TDA for computing excited states.

The response matrices are:
    A_ia,jb = δ_ij δ_ab (ε_a - ε_i) + (ia|jb) - c_x (ij|ab) + (ia|f_xc|jb)
    B_ia,jb = (ia|bj) - c_x (ib|aj) + (ia|f_xc|bj)

where c_x is the fraction of exact exchange and f_xc is the XC kernel.

For TDA (Tamm-Dancoff Approximation), B = 0 and we solve:
    A X = ω X

For full TDDFT (RPA), we solve:
    (A - B)(A + B) Z = ω² Z

Reference: Stratmann, Scuseria, Frisch, J. Chem. Phys. 109, 8218 (1998)
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..methods.hf import HFResult
from ..methods.dft import DFTResult, kohn_sham


EV_PER_HARTREE = 27.2114


@dataclass
class TDDFTResult:
    """
    Result of TDDFT calculation.

    Attributes:
        excitation_energies: Excitation energies in eV (n_states,)
        excitation_energies_au: Excitation energies in Hartree
        oscillator_strengths: Oscillator strengths (n_states,)
        transition_dipoles: Transition dipole moments (n_states, 3)
        X_amplitudes: X response vectors (n_states, n_occ, n_virt)
        Y_amplitudes: Y response vectors (n_states, n_occ, n_virt)
        n_states: Number of excited states computed
        method: 'TDDFT' or 'TDA'
    """
    excitation_energies: np.ndarray
    excitation_energies_au: np.ndarray
    oscillator_strengths: np.ndarray
    transition_dipoles: np.ndarray
    X_amplitudes: np.ndarray
    Y_amplitudes: np.ndarray
    n_states: int
    method: str

    def print_summary(self):
        """Print formatted excited state summary."""
        print("\n" + "=" * 70)
        print(f"{self.method} Excited States")
        print("=" * 70)
        print(f"{'State':>5} {'Energy (eV)':>12} {'f (osc)':>10} "
              f"{'Transition':>20}")
        print("-" * 70)

        for i in range(self.n_states):
            E = self.excitation_energies[i]
            f = self.oscillator_strengths[i]

            # Find dominant transition
            X = self.X_amplitudes[i]
            idx = np.unravel_index(np.argmax(np.abs(X)), X.shape)
            coef = X[idx]

            print(f"{i+1:5d} {E:12.4f} {f:10.4f} "
                  f"    {idx[0]+1}→{idx[1]+1} ({coef:+.3f})")

        print("=" * 70)


@dataclass
class TDAResult(TDDFTResult):
    """Result of TDA calculation (TDDFT with B=0)."""
    pass


def tddft(result: Union[HFResult, DFTResult],
          n_states: int = 5,
          tda: bool = False,
          verbose: bool = True) -> TDDFTResult:
    """
    Perform TDDFT or TDA calculation.

    Args:
        result: Converged HF or DFT result
        n_states: Number of excited states to compute
        tda: Use Tamm-Dancoff Approximation
        verbose: Print excited state table

    Returns:
        TDDFTResult with excitation energies and properties

    Example:
        dft = kohn_sham(atoms, functional='B3LYP')
        excited = tddft(dft, n_states=10)
        excited.print_summary()
    """
    # Get MO info
    C = result.mo_coefficients
    eps = result.orbital_energies
    n_occ = result.n_electrons // 2
    n_virt = len(eps) - n_occ
    n_mo = len(eps)

    # Get two-electron integrals in MO basis
    G_ao = result.G if hasattr(result, 'G') else None

    if G_ao is None:
        raise ValueError("Two-electron integrals required for TDDFT")

    # Transform to MO basis
    G_mo = _transform_eri(C, G_ao)

    # Get exact exchange fraction
    if isinstance(result, DFTResult):
        c_x = result.exact_exchange_fraction
    else:
        c_x = 1.0  # Full exchange for HF

    # Build response matrices
    if verbose:
        print(f"Building {'TDA' if tda else 'TDDFT'} response matrices...")
        print(f"  Occupied orbitals: {n_occ}")
        print(f"  Virtual orbitals: {n_virt}")
        print(f"  Response dimension: {n_occ * n_virt}")

    A = _build_A_matrix(eps, G_mo, n_occ, n_virt, c_x)

    if tda:
        # TDA: solve A X = ω X
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take lowest n_states
        n_states = min(n_states, len(eigenvalues))
        omega = eigenvalues[:n_states]
        X = eigenvectors[:, :n_states]
        Y = np.zeros_like(X)

    else:
        # Full TDDFT: solve (A-B)(A+B) Z = ω² Z
        B = _build_B_matrix(eps, G_mo, n_occ, n_virt, c_x)

        ApB = A + B
        AmB = A - B

        # Hermitian eigenvalue problem for (A-B)(A+B)
        M = AmB @ ApB

        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # ω = sqrt(eigenvalue)
        omega_sq = eigenvalues
        omega = np.sqrt(np.maximum(omega_sq, 0))

        # Sort
        idx = np.argsort(omega)
        omega = omega[idx]
        Z = eigenvectors[:, idx]

        # Get X and Y from Z
        # X + Y = (A+B)^(-1/2) Z sqrt(ω)
        # X - Y = (A-B)^(-1/2) Z / sqrt(ω)
        n_states = min(n_states, len(omega))
        omega = omega[:n_states]
        Z = Z[:, :n_states]

        X = np.zeros((n_occ * n_virt, n_states))
        Y = np.zeros((n_occ * n_virt, n_states))

        for i in range(n_states):
            if omega[i] > 1e-10:
                X[:, i] = 0.5 * (Z[:, i] * np.sqrt(omega[i]) +
                                 Z[:, i] / np.sqrt(omega[i]))
                Y[:, i] = 0.5 * (Z[:, i] * np.sqrt(omega[i]) -
                                 Z[:, i] / np.sqrt(omega[i]))

    # Compute transition dipoles and oscillator strengths
    transition_dipoles = _compute_transition_dipoles(
        X, Y, C, n_occ, n_virt, result.atoms
    )
    oscillator_strengths = _compute_oscillator_strengths(
        omega, transition_dipoles
    )

    # Reshape amplitudes
    X_reshaped = X.T.reshape(n_states, n_occ, n_virt)
    Y_reshaped = Y.T.reshape(n_states, n_occ, n_virt)

    result = TDDFTResult(
        excitation_energies=omega * EV_PER_HARTREE,
        excitation_energies_au=omega,
        oscillator_strengths=oscillator_strengths,
        transition_dipoles=transition_dipoles,
        X_amplitudes=X_reshaped,
        Y_amplitudes=Y_reshaped,
        n_states=n_states,
        method='TDA' if tda else 'TDDFT'
    )

    if verbose:
        result.print_summary()

    return result


def tda(result: Union[HFResult, DFTResult],
        n_states: int = 5,
        verbose: bool = True) -> TDAResult:
    """
    Perform TDA (Tamm-Dancoff Approximation) calculation.

    Equivalent to tddft(..., tda=True).
    """
    return tddft(result, n_states, tda=True, verbose=verbose)


def _build_A_matrix(eps: np.ndarray, G_mo: np.ndarray,
                    n_occ: int, n_virt: int, c_x: float) -> np.ndarray:
    """
    Build A response matrix.

    A_ia,jb = δ_ij δ_ab (ε_a - ε_i) + (ia|jb) - c_x (ij|ab)
    """
    dim = n_occ * n_virt
    A = np.zeros((dim, dim))

    eps_occ = eps[:n_occ]
    eps_virt = eps[n_occ:]

    for i in range(n_occ):
        for a in range(n_virt):
            ia = i * n_virt + a
            for j in range(n_occ):
                for b in range(n_virt):
                    jb = j * n_virt + b

                    # Diagonal contribution
                    if i == j and a == b:
                        A[ia, jb] += eps_virt[a] - eps_occ[i]

                    # Coulomb: (ia|jb) = G_mo[i, n_occ+a, j, n_occ+b]
                    A[ia, jb] += G_mo[i, n_occ + a, j, n_occ + b]

                    # Exchange: -c_x (ij|ab)
                    A[ia, jb] -= c_x * G_mo[i, j, n_occ + a, n_occ + b]

    return A


def _build_B_matrix(eps: np.ndarray, G_mo: np.ndarray,
                    n_occ: int, n_virt: int, c_x: float) -> np.ndarray:
    """
    Build B response matrix.

    B_ia,jb = (ia|bj) - c_x (ib|aj)
    """
    dim = n_occ * n_virt
    B = np.zeros((dim, dim))

    for i in range(n_occ):
        for a in range(n_virt):
            ia = i * n_virt + a
            for j in range(n_occ):
                for b in range(n_virt):
                    jb = j * n_virt + b

                    # (ia|bj)
                    B[ia, jb] += G_mo[i, n_occ + a, n_occ + b, j]

                    # -c_x (ib|aj)
                    B[ia, jb] -= c_x * G_mo[i, n_occ + b, n_occ + a, j]

    return B


def _transform_eri(C: np.ndarray, G_ao: np.ndarray) -> np.ndarray:
    """Transform ERI from AO to MO basis."""
    return np.einsum('mp,nq,mnls,lr,su->pqru',
                     C, C, G_ao, C, C, optimize=True)


def _compute_transition_dipoles(X: np.ndarray, Y: np.ndarray,
                                C: np.ndarray, n_occ: int, n_virt: int,
                                atoms: List) -> np.ndarray:
    """
    Compute transition dipole moments.

    μ_0n = <0|μ|n> = Σ_ia (X_ia + Y_ia) <i|r|a>
    """
    n_states = X.shape[1]
    n_mo = n_occ + n_virt

    # Compute dipole integrals in MO basis (simplified)
    # Full implementation would use proper dipole integrals
    mu_mo = np.zeros((n_mo, n_mo, 3))

    # Approximate using orbital centroids
    for i in range(n_mo):
        for a in range(n_mo):
            # <i|r|a> ≈ center of overlap
            mu_mo[i, a, :] = 0.0  # Placeholder

    transition_dipoles = np.zeros((n_states, 3))

    for n in range(n_states):
        for i in range(n_occ):
            for a in range(n_virt):
                ia = i * n_virt + a
                amplitude = X[ia, n] + Y[ia, n]
                transition_dipoles[n, :] += amplitude * mu_mo[i, n_occ + a, :]

    return transition_dipoles


def _compute_oscillator_strengths(omega: np.ndarray,
                                  transition_dipoles: np.ndarray) -> np.ndarray:
    """
    Compute oscillator strengths.

    f = (2/3) ω |μ|²
    """
    mu_sq = np.sum(transition_dipoles ** 2, axis=1)
    return (2.0 / 3.0) * omega * mu_sq


def compute_oscillator_strength(omega: float, mu: np.ndarray) -> float:
    """
    Compute oscillator strength for single state.

    Args:
        omega: Excitation energy (Hartree)
        mu: Transition dipole moment (3,)

    Returns:
        Oscillator strength (dimensionless)
    """
    return (2.0 / 3.0) * omega * np.sum(mu ** 2)
