"""
Derivative Integrals for Analytic Gradients

Computes derivatives of molecular integrals with respect to nuclear coordinates.

For a Gaussian centered at A with exponent α:
    ∂/∂Ax exp(-α|r-A|²) = 2α(x-Ax) exp(-α|r-A|²)

This leads to recurrence relations for derivative integrals.
"""

import numpy as np
from typing import List, Tuple
from ..basis import BasisSet, BasisFunction


def overlap_derivative(basis: BasisSet, atom_idx: int,
                       coord_idx: int) -> np.ndarray:
    """
    Compute derivative of overlap matrix with respect to nuclear coordinate.

    dS_μν/dX_A = Σ primitives [derivative terms]

    For primitives centered on atom A:
        ∂φ_μ/∂X_A = -2α(x-Ax)φ_μ (for s-type)

    Args:
        basis: Basis set
        atom_idx: Index of atom to differentiate with respect to
        coord_idx: Coordinate index (0=x, 1=y, 2=z)

    Returns:
        Derivative matrix dS/dX (n_basis, n_basis)
    """
    n_basis = basis.n_basis
    dS = np.zeros((n_basis, n_basis))

    for mu, bf_mu in enumerate(basis.basis_functions):
        for nu, bf_nu in enumerate(basis.basis_functions):
            # Only non-zero if μ or ν is centered on atom_idx
            if bf_mu.atom_index == atom_idx or bf_nu.atom_index == atom_idx:
                dS[mu, nu] = _overlap_deriv_element(
                    bf_mu, bf_nu, atom_idx, coord_idx
                )

    return dS


def _overlap_deriv_element(bf_a: BasisFunction, bf_b: BasisFunction,
                           atom_idx: int, coord_idx: int) -> float:
    """Compute single element of overlap derivative."""
    result = 0.0

    A = bf_a.center
    B = bf_b.center
    la, ma, na = bf_a.angular_momentum
    lb, mb, nb = bf_b.angular_momentum

    for prim_a in bf_a.primitives:
        for prim_b in bf_b.primitives:
            alpha = prim_a.exponent
            beta = prim_b.exponent
            ca = prim_a.coefficient
            cb = prim_b.coefficient

            # Gaussian product
            gamma = alpha + beta
            P = (alpha * A + beta * B) / gamma

            # Pre-exponential factor
            AB2 = np.sum((A - B) ** 2)
            K = np.exp(-alpha * beta / gamma * AB2)

            # Base overlap
            S_base = _overlap_1d(la, lb, A[0], B[0], P[0], gamma)
            S_base *= _overlap_1d(ma, mb, A[1], B[1], P[1], gamma)
            S_base *= _overlap_1d(na, nb, A[2], B[2], P[2], gamma)
            S_base *= K * (np.pi / gamma) ** 1.5

            # Derivative contribution
            if bf_a.atom_index == atom_idx:
                # ∂/∂A contribution
                deriv = _overlap_deriv_1d(
                    la if coord_idx == 0 else (ma if coord_idx == 1 else na),
                    lb if coord_idx == 0 else (mb if coord_idx == 1 else nb),
                    A[coord_idx], B[coord_idx], P[coord_idx],
                    alpha, gamma
                )
                # Multiply by other dimensions
                if coord_idx == 0:
                    deriv *= _overlap_1d(ma, mb, A[1], B[1], P[1], gamma)
                    deriv *= _overlap_1d(na, nb, A[2], B[2], P[2], gamma)
                elif coord_idx == 1:
                    deriv *= _overlap_1d(la, lb, A[0], B[0], P[0], gamma)
                    deriv *= _overlap_1d(na, nb, A[2], B[2], P[2], gamma)
                else:
                    deriv *= _overlap_1d(la, lb, A[0], B[0], P[0], gamma)
                    deriv *= _overlap_1d(ma, mb, A[1], B[1], P[1], gamma)

                deriv *= K * (np.pi / gamma) ** 1.5
                result += ca * cb * deriv

            if bf_b.atom_index == atom_idx and bf_b.atom_index != bf_a.atom_index:
                # ∂/∂B contribution (similar structure)
                deriv = _overlap_deriv_1d(
                    lb if coord_idx == 0 else (mb if coord_idx == 1 else nb),
                    la if coord_idx == 0 else (ma if coord_idx == 1 else na),
                    B[coord_idx], A[coord_idx], P[coord_idx],
                    beta, gamma
                )
                if coord_idx == 0:
                    deriv *= _overlap_1d(mb, ma, B[1], A[1], P[1], gamma)
                    deriv *= _overlap_1d(nb, na, B[2], A[2], P[2], gamma)
                elif coord_idx == 1:
                    deriv *= _overlap_1d(lb, la, B[0], A[0], P[0], gamma)
                    deriv *= _overlap_1d(nb, na, B[2], A[2], P[2], gamma)
                else:
                    deriv *= _overlap_1d(lb, la, B[0], A[0], P[0], gamma)
                    deriv *= _overlap_1d(mb, ma, B[1], A[1], P[1], gamma)

                deriv *= K * (np.pi / gamma) ** 1.5
                result += ca * cb * deriv

    return result


def _overlap_1d(i: int, j: int, Ax: float, Bx: float, Px: float,
                gamma: float) -> float:
    """
    1D overlap integral using Obara-Saika recursion.

    S_ij = (2γ)^(-1/2) * Σ_k E_ij^k * (2k-1)!! / (2γ)^k
    """
    if i < 0 or j < 0:
        return 0.0
    if i == 0 and j == 0:
        return 1.0

    PA = Px - Ax
    PB = Px - Bx

    if i > 0:
        return PA * _overlap_1d(i - 1, j, Ax, Bx, Px, gamma) + \
               (i - 1) / (2 * gamma) * _overlap_1d(i - 2, j, Ax, Bx, Px, gamma) + \
               j / (2 * gamma) * _overlap_1d(i - 1, j - 1, Ax, Bx, Px, gamma)
    else:
        return PB * _overlap_1d(i, j - 1, Ax, Bx, Px, gamma) + \
               i / (2 * gamma) * _overlap_1d(i - 1, j - 1, Ax, Bx, Px, gamma) + \
               (j - 1) / (2 * gamma) * _overlap_1d(i, j - 2, Ax, Bx, Px, gamma)


def _overlap_deriv_1d(i: int, j: int, Ax: float, Bx: float, Px: float,
                      alpha: float, gamma: float) -> float:
    """
    Derivative of 1D overlap with respect to center A.

    ∂S_ij/∂Ax = 2α S_{i+1,j} - i S_{i-1,j}
    """
    term1 = 2 * alpha * _overlap_1d(i + 1, j, Ax, Bx, Px, gamma)
    term2 = -i * _overlap_1d(i - 1, j, Ax, Bx, Px, gamma) if i > 0 else 0.0
    return term1 + term2


def kinetic_derivative(basis: BasisSet, atom_idx: int,
                       coord_idx: int) -> np.ndarray:
    """
    Compute derivative of kinetic energy matrix.

    dT_μν/dX = -1/2 d<μ|∇²|ν>/dX

    Uses the relation:
        <μ|∇²|ν> = <μ|∂²/∂x² + ∂²/∂y² + ∂²/∂z²|ν>

    Args:
        basis: Basis set
        atom_idx: Atom index
        coord_idx: Coordinate (0=x, 1=y, 2=z)

    Returns:
        Derivative matrix dT/dX
    """
    n_basis = basis.n_basis
    dT = np.zeros((n_basis, n_basis))

    for mu, bf_mu in enumerate(basis.basis_functions):
        for nu, bf_nu in enumerate(basis.basis_functions):
            if bf_mu.atom_index == atom_idx or bf_nu.atom_index == atom_idx:
                dT[mu, nu] = _kinetic_deriv_element(
                    bf_mu, bf_nu, atom_idx, coord_idx
                )

    return dT


def _kinetic_deriv_element(bf_a: BasisFunction, bf_b: BasisFunction,
                           atom_idx: int, coord_idx: int) -> float:
    """Compute single kinetic derivative element."""
    # Use numerical differentiation for simplicity
    # Full analytic implementation would use recurrence relations
    h = 1e-5
    result = 0.0

    # Temporarily shift center
    original_center = None
    if bf_a.atom_index == atom_idx:
        original_center = bf_a.center.copy()
        bf_a.center[coord_idx] += h
        T_plus = _kinetic_element(bf_a, bf_b)
        bf_a.center[coord_idx] -= 2 * h
        T_minus = _kinetic_element(bf_a, bf_b)
        bf_a.center[coord_idx] = original_center[coord_idx]
        result += (T_plus - T_minus) / (2 * h)

    if bf_b.atom_index == atom_idx and bf_b.atom_index != bf_a.atom_index:
        original_center = bf_b.center.copy()
        bf_b.center[coord_idx] += h
        T_plus = _kinetic_element(bf_a, bf_b)
        bf_b.center[coord_idx] -= 2 * h
        T_minus = _kinetic_element(bf_a, bf_b)
        bf_b.center[coord_idx] = original_center[coord_idx]
        result += (T_plus - T_minus) / (2 * h)

    return result


def _kinetic_element(bf_a: BasisFunction, bf_b: BasisFunction) -> float:
    """Compute kinetic energy integral element."""
    from ..one_electron import kinetic_integral
    return kinetic_integral(bf_a, bf_b)


def nuclear_derivative(basis: BasisSet, nuclei: List[Tuple[np.ndarray, float]],
                       atom_idx: int, coord_idx: int) -> np.ndarray:
    """
    Compute derivative of nuclear attraction matrix.

    dV_μν/dX includes:
    1. Derivative of basis functions (if centered on atom)
    2. Derivative of nuclear position (Hellmann-Feynman term)

    Args:
        basis: Basis set
        nuclei: List of (position, charge) for all nuclei
        atom_idx: Atom index for differentiation
        coord_idx: Coordinate (0=x, 1=y, 2=z)

    Returns:
        Derivative matrix dV/dX
    """
    n_basis = basis.n_basis
    dV = np.zeros((n_basis, n_basis))

    h = 1e-5

    for mu, bf_mu in enumerate(basis.basis_functions):
        for nu, bf_nu in enumerate(basis.basis_functions):
            # Derivative with respect to basis function centers
            if bf_mu.atom_index == atom_idx or bf_nu.atom_index == atom_idx:
                dV[mu, nu] = _nuclear_deriv_basis(
                    bf_mu, bf_nu, nuclei, atom_idx, coord_idx, h
                )

            # Hellmann-Feynman term (derivative of nuclear position)
            dV[mu, nu] += _nuclear_deriv_nucleus(
                bf_mu, bf_nu, nuclei, atom_idx, coord_idx, h
            )

    return dV


def _nuclear_deriv_basis(bf_a: BasisFunction, bf_b: BasisFunction,
                         nuclei: List, atom_idx: int, coord_idx: int,
                         h: float) -> float:
    """Derivative with respect to basis function center."""
    from ..one_electron import nuclear_attraction_integral

    result = 0.0

    if bf_a.atom_index == atom_idx:
        original = bf_a.center[coord_idx]
        bf_a.center[coord_idx] = original + h
        V_plus = sum(nuclear_attraction_integral(bf_a, bf_b, pos, Z)
                     for pos, Z in nuclei)
        bf_a.center[coord_idx] = original - h
        V_minus = sum(nuclear_attraction_integral(bf_a, bf_b, pos, Z)
                      for pos, Z in nuclei)
        bf_a.center[coord_idx] = original
        result += (V_plus - V_minus) / (2 * h)

    if bf_b.atom_index == atom_idx and bf_b.atom_index != bf_a.atom_index:
        original = bf_b.center[coord_idx]
        bf_b.center[coord_idx] = original + h
        V_plus = sum(nuclear_attraction_integral(bf_a, bf_b, pos, Z)
                     for pos, Z in nuclei)
        bf_b.center[coord_idx] = original - h
        V_minus = sum(nuclear_attraction_integral(bf_a, bf_b, pos, Z)
                      for pos, Z in nuclei)
        bf_b.center[coord_idx] = original
        result += (V_plus - V_minus) / (2 * h)

    return result


def _nuclear_deriv_nucleus(bf_a: BasisFunction, bf_b: BasisFunction,
                           nuclei: List, atom_idx: int, coord_idx: int,
                           h: float) -> float:
    """Hellmann-Feynman derivative with respect to nuclear position."""
    from ..one_electron import nuclear_attraction_integral

    if atom_idx >= len(nuclei):
        return 0.0

    pos, Z = nuclei[atom_idx]
    original = pos[coord_idx]

    pos[coord_idx] = original + h
    V_plus = nuclear_attraction_integral(bf_a, bf_b, pos, Z)

    pos[coord_idx] = original - h
    V_minus = nuclear_attraction_integral(bf_a, bf_b, pos, Z)

    pos[coord_idx] = original

    return (V_plus - V_minus) / (2 * h)


def eri_derivative(basis: BasisSet, atom_idx: int,
                   coord_idx: int) -> np.ndarray:
    """
    Compute derivative of ERI tensor with respect to nuclear coordinate.

    d(μν|λσ)/dX

    This is the most expensive part of gradient calculation.
    For efficiency, uses numerical differentiation.

    Args:
        basis: Basis set
        atom_idx: Atom index
        coord_idx: Coordinate (0=x, 1=y, 2=z)

    Returns:
        Derivative tensor d(μν|λσ)/dX
    """
    from ..two_electron import electron_repulsion_integral

    n_basis = basis.n_basis
    dG = np.zeros((n_basis, n_basis, n_basis, n_basis))
    h = 1e-5

    bfs = basis.basis_functions

    for mu in range(n_basis):
        for nu in range(mu + 1):
            for lam in range(n_basis):
                for sig in range(lam + 1):
                    # Check if any basis function is on the atom
                    on_atom = (bfs[mu].atom_index == atom_idx or
                               bfs[nu].atom_index == atom_idx or
                               bfs[lam].atom_index == atom_idx or
                               bfs[sig].atom_index == atom_idx)

                    if not on_atom:
                        continue

                    deriv = _eri_deriv_element(
                        bfs[mu], bfs[nu], bfs[lam], bfs[sig],
                        atom_idx, coord_idx, h
                    )

                    # Use 8-fold symmetry
                    dG[mu, nu, lam, sig] = deriv
                    dG[nu, mu, lam, sig] = deriv
                    dG[mu, nu, sig, lam] = deriv
                    dG[nu, mu, sig, lam] = deriv
                    dG[lam, sig, mu, nu] = deriv
                    dG[sig, lam, mu, nu] = deriv
                    dG[lam, sig, nu, mu] = deriv
                    dG[sig, lam, nu, mu] = deriv

    return dG


def _eri_deriv_element(bf_a, bf_b, bf_c, bf_d,
                       atom_idx: int, coord_idx: int, h: float) -> float:
    """Compute derivative of single ERI element."""
    from ..two_electron import electron_repulsion_integral

    result = 0.0

    for bf in [bf_a, bf_b, bf_c, bf_d]:
        if bf.atom_index == atom_idx:
            original = bf.center[coord_idx]

            bf.center[coord_idx] = original + h
            eri_plus = electron_repulsion_integral(bf_a, bf_b, bf_c, bf_d)

            bf.center[coord_idx] = original - h
            eri_minus = electron_repulsion_integral(bf_a, bf_b, bf_c, bf_d)

            bf.center[coord_idx] = original

            result += (eri_plus - eri_minus) / (2 * h)

    return result


def nuclear_repulsion_gradient(nuclei: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """
    Compute gradient of nuclear repulsion energy.

    dV_nn/dX_A = Σ_{B≠A} Z_A Z_B (X_A - X_B) / |R_A - R_B|³

    Args:
        nuclei: List of (position, charge) tuples

    Returns:
        Gradient array (n_atoms, 3)
    """
    n_atoms = len(nuclei)
    grad = np.zeros((n_atoms, 3))

    for A in range(n_atoms):
        pos_A, Z_A = nuclei[A]
        for B in range(n_atoms):
            if A == B:
                continue
            pos_B, Z_B = nuclei[B]

            R_AB = pos_A - pos_B
            r = np.linalg.norm(R_AB)

            if r > 1e-10:
                grad[A] += Z_A * Z_B * R_AB / (r ** 3)

    return grad
