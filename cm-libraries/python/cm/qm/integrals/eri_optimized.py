"""
Optimized Electron Repulsion Integral (ERI) Computation

Implements several optimizations:
1. Schwarz screening: skip integrals below threshold
2. 8-fold permutation symmetry: only compute unique integrals
3. PyTorch/CUDA acceleration for parallelism

The ERI tensor has the symmetries:
    (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
            = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)

With screening and symmetry, we can reduce the effective N^4 to ~N^2.5 or better.
"""

import numpy as np
from typing import Optional, Tuple
import math

# Try to import PyTorch for GPU acceleration
try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False


def compute_schwarz_bounds(basis) -> np.ndarray:
    """
    Compute Schwarz upper bounds for integral screening.

    Q_ij = sqrt((ij|ij))

    The Schwarz inequality gives:
        |(ij|kl)| <= Q_ij * Q_kl

    This allows us to skip integrals that will be below threshold.

    Args:
        basis: BasisSet with built functions

    Returns:
        Q matrix (n_basis, n_basis) of Schwarz bounds
    """
    from .eri import electron_repulsion_integral

    n = basis.n_basis
    Q = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # (ij|ij) diagonal integral
            val = electron_repulsion_integral(
                basis[i], basis[j], basis[i], basis[j]
            )
            Q[i, j] = math.sqrt(max(0, val))
            Q[j, i] = Q[i, j]

    return Q


def eri_tensor_screened(basis, threshold: float = 1e-10,
                        verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Compute ERI tensor with Schwarz screening.

    Args:
        basis: BasisSet with built functions
        threshold: Screening threshold (default 1e-10)
        verbose: Print screening statistics

    Returns:
        Tuple of (ERI tensor, statistics dict)
    """
    from .eri import electron_repulsion_integral

    n = basis.n_basis
    G = np.zeros((n, n, n, n))

    # Compute Schwarz bounds
    Q = compute_schwarz_bounds(basis)
    Q_max = Q.max()

    # Statistics
    total_integrals = 0
    screened_integrals = 0
    computed_integrals = 0

    # Loop with 8-fold symmetry
    for i in range(n):
        for j in range(i + 1):
            Q_ij = Q[i, j]
            if Q_ij * Q_max < threshold:
                # Skip entire ij shell
                screened_integrals += (n * (n + 1)) // 2
                continue

            for k in range(n):
                for l in range(k + 1):
                    # Skip if kl > ij (symmetry)
                    ij = i * (i + 1) // 2 + j
                    kl = k * (k + 1) // 2 + l
                    if kl > ij:
                        continue

                    total_integrals += 1

                    # Schwarz screening
                    Q_kl = Q[k, l]
                    if Q_ij * Q_kl < threshold:
                        screened_integrals += 1
                        continue

                    # Compute integral
                    computed_integrals += 1
                    val = electron_repulsion_integral(
                        basis[i], basis[j], basis[k], basis[l]
                    )

                    # Apply 8-fold symmetry
                    G[i, j, k, l] = val
                    G[j, i, k, l] = val
                    G[i, j, l, k] = val
                    G[j, i, l, k] = val
                    G[k, l, i, j] = val
                    G[l, k, i, j] = val
                    G[k, l, j, i] = val
                    G[l, k, j, i] = val

    stats = {
        'total_unique': total_integrals,
        'screened': screened_integrals,
        'computed': computed_integrals,
        'screening_efficiency': 1.0 - computed_integrals / max(1, total_integrals),
        'threshold': threshold
    }

    if verbose:
        print(f"ERI Screening Statistics:")
        print(f"  Total unique integrals: {total_integrals}")
        print(f"  Screened (skipped):     {screened_integrals}")
        print(f"  Actually computed:      {computed_integrals}")
        print(f"  Screening efficiency:   {stats['screening_efficiency']*100:.1f}%")

    return G, stats


def eri_tensor_torch(basis, device: str = 'cuda',
                     threshold: float = 1e-10,
                     batch_size: int = 1024) -> np.ndarray:
    """
    Compute ERI tensor using PyTorch for GPU acceleration.

    This parallelizes the primitive Gaussian products using tensor operations.

    Args:
        basis: BasisSet with built functions
        device: 'cuda' or 'cpu'
        threshold: Screening threshold
        batch_size: Batch size for GPU computation

    Returns:
        ERI tensor (n_basis, n_basis, n_basis, n_basis)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available. Install with: pip install torch")

    if device == 'cuda' and not HAS_CUDA:
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'

    n = basis.n_basis
    G = torch.zeros((n, n, n, n), device=device, dtype=torch.float64)

    # Extract primitive data for all basis functions
    # This allows batch processing on GPU
    prim_data = _extract_primitive_data(basis, device)

    # Compute Schwarz bounds on GPU
    Q = _compute_schwarz_bounds_torch(prim_data, n, device)

    # Generate integral batches
    integral_indices = []
    for i in range(n):
        for j in range(i + 1):
            for k in range(n):
                for l in range(k + 1):
                    ij = i * (i + 1) // 2 + j
                    kl = k * (k + 1) // 2 + l
                    if kl <= ij:
                        # Schwarz check
                        if Q[i, j] * Q[k, l] >= threshold:
                            integral_indices.append((i, j, k, l))

    # Process in batches
    n_integrals = len(integral_indices)

    for batch_start in range(0, n_integrals, batch_size):
        batch_end = min(batch_start + batch_size, n_integrals)
        batch_indices = integral_indices[batch_start:batch_end]

        # Compute batch of integrals
        batch_values = _compute_eri_batch_torch(
            prim_data, batch_indices, device
        )

        # Store with symmetry
        for idx, (i, j, k, l) in enumerate(batch_indices):
            val = batch_values[idx]
            G[i, j, k, l] = val
            G[j, i, k, l] = val
            G[i, j, l, k] = val
            G[j, i, l, k] = val
            G[k, l, i, j] = val
            G[l, k, i, j] = val
            G[k, l, j, i] = val
            G[l, k, j, i] = val

    return G.cpu().numpy()


def _extract_primitive_data(basis, device):
    """
    Extract primitive Gaussian data into tensors for GPU computation.
    """
    data = {
        'centers': [],
        'exponents': [],
        'coefficients': [],
        'angular': [],
        'prim_ranges': [],  # (start, end) for each basis function
    }

    prim_idx = 0
    for i, bf in enumerate(basis.functions):
        cgf = bf.cgf
        n_prim = cgf.n_primitives
        start = prim_idx

        for coeff, exp in cgf.primitives:
            data['centers'].append(list(cgf.center))
            data['exponents'].append(exp)
            data['coefficients'].append(coeff)
            data['angular'].append(list(cgf.angular))
            prim_idx += 1

        data['prim_ranges'].append((start, prim_idx))

    # Convert to tensors
    data['centers'] = torch.tensor(data['centers'], device=device, dtype=torch.float64)
    data['exponents'] = torch.tensor(data['exponents'], device=device, dtype=torch.float64)
    data['coefficients'] = torch.tensor(data['coefficients'], device=device, dtype=torch.float64)
    data['angular'] = torch.tensor(data['angular'], device=device, dtype=torch.int64)
    data['prim_ranges'] = data['prim_ranges']

    return data


def _compute_schwarz_bounds_torch(prim_data, n, device):
    """
    Compute Schwarz bounds using PyTorch.
    """
    Q = torch.zeros((n, n), device=device, dtype=torch.float64)

    for i in range(n):
        for j in range(i + 1):
            val = _compute_single_eri_torch(prim_data, i, j, i, j, device)
            Q[i, j] = torch.sqrt(torch.clamp(val, min=0))
            Q[j, i] = Q[i, j]

    return Q


def _compute_eri_batch_torch(prim_data, indices, device):
    """
    Compute a batch of ERIs using PyTorch.

    For simplicity, this loops but could be further parallelized.
    """
    results = []
    for i, j, k, l in indices:
        val = _compute_single_eri_torch(prim_data, i, j, k, l, device)
        results.append(val)
    return torch.stack(results)


def _compute_single_eri_torch(prim_data, i, j, k, l, device):
    """
    Compute a single ERI using primitive data.

    (ij|kl) = Σ_pqrs c_p c_q c_r c_s (p_i q_j | r_k s_l)

    For s-type Gaussians, the primitive integral is:
        (ab|cd) = 2π^(5/2) / (pq * sqrt(p+q)) * exp(-αβρ^2/p) * exp(-γδσ^2/q) * F_0(T)

    where:
        p = α + β, q = γ + δ
        T = pq/(p+q) * |P - Q|^2
        P = (αA + βB)/p, Q = (γC + δD)/q
    """
    ranges = prim_data['prim_ranges']
    centers = prim_data['centers']
    exponents = prim_data['exponents']
    coeffs = prim_data['coefficients']
    angular = prim_data['angular']

    pi_5_2 = math.pi ** 2.5

    # Get ranges for each basis function
    i_start, i_end = ranges[i]
    j_start, j_end = ranges[j]
    k_start, k_end = ranges[k]
    l_start, l_end = ranges[l]

    result = torch.tensor(0.0, device=device, dtype=torch.float64)

    # Loop over primitives (could be parallelized further)
    for pi in range(i_start, i_end):
        for pj in range(j_start, j_end):
            for pk in range(k_start, k_end):
                for pl in range(l_start, l_end):
                    # Get primitive data
                    A = centers[pi]
                    B = centers[pj]
                    C = centers[pk]
                    D = centers[pl]

                    alpha = exponents[pi]
                    beta = exponents[pj]
                    gamma = exponents[pk]
                    delta = exponents[pl]

                    c_i = coeffs[pi]
                    c_j = coeffs[pj]
                    c_k = coeffs[pk]
                    c_l = coeffs[pl]

                    ang_i = angular[pi]
                    ang_j = angular[pj]
                    ang_k = angular[pk]
                    ang_l = angular[pl]

                    # Check if s-type (simplified)
                    L_total = ang_i.sum() + ang_j.sum() + ang_k.sum() + ang_l.sum()

                    if L_total == 0:
                        # s-type integral
                        p = alpha + beta
                        q = gamma + delta
                        rho = p * q / (p + q)

                        # Gaussian product centers
                        P = (alpha * A + beta * B) / p
                        Q = (gamma * C + delta * D) / q

                        # Squared distances
                        AB2 = ((A - B) ** 2).sum()
                        CD2 = ((C - D) ** 2).sum()
                        PQ2 = ((P - Q) ** 2).sum()

                        # Pre-exponential factors
                        K_AB = torch.exp(-alpha * beta / p * AB2)
                        K_CD = torch.exp(-gamma * delta / q * CD2)

                        # Boys function argument
                        T = rho * PQ2

                        # Boys function F_0(T) = sqrt(π/T) * erf(sqrt(T)) / 2 for T > 0
                        # or 1 for T = 0
                        if T < 1e-10:
                            F0 = torch.tensor(1.0, device=device, dtype=torch.float64)
                        else:
                            sqrt_T = torch.sqrt(T)
                            F0 = torch.sqrt(torch.tensor(math.pi, device=device)) / (2 * sqrt_T) * torch.erf(sqrt_T)

                        # Primitive integral
                        val = (2 * pi_5_2 / (p * q * torch.sqrt(p + q))
                               * K_AB * K_CD * F0)

                        # Normalization
                        norm_i = (2 * alpha / math.pi) ** 0.75
                        norm_j = (2 * beta / math.pi) ** 0.75
                        norm_k = (2 * gamma / math.pi) ** 0.75
                        norm_l = (2 * delta / math.pi) ** 0.75

                        result += c_i * c_j * c_k * c_l * norm_i * norm_j * norm_k * norm_l * val

                    else:
                        # Higher angular momentum - use CPU fallback
                        # For a full implementation, we'd need McMurchie-Davidson on GPU
                        from .eri import _primitive_eri
                        from .basis import GaussianPrimitive

                        prim_i = GaussianPrimitive(alpha.item(), tuple(A.cpu().numpy()), tuple(ang_i.cpu().numpy()))
                        prim_j = GaussianPrimitive(beta.item(), tuple(B.cpu().numpy()), tuple(ang_j.cpu().numpy()))
                        prim_k = GaussianPrimitive(gamma.item(), tuple(C.cpu().numpy()), tuple(ang_k.cpu().numpy()))
                        prim_l = GaussianPrimitive(delta.item(), tuple(D.cpu().numpy()), tuple(ang_l.cpu().numpy()))

                        val = _primitive_eri(prim_i, prim_j, prim_k, prim_l)
                        result += c_i.item() * c_j.item() * c_k.item() * c_l.item() * val

    return result


def eri_direct(basis, D: np.ndarray, threshold: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct SCF: compute J and K matrices without storing full ERI tensor.

    This is memory-efficient for large systems:
        J_μν = Σ_λσ D_λσ (μν|λσ)
        K_μν = Σ_λσ D_λσ (μλ|νσ)

    Args:
        basis: BasisSet
        D: Density matrix (n_basis, n_basis)
        threshold: Screening threshold

    Returns:
        Tuple of (J matrix, K matrix)
    """
    from .eri import electron_repulsion_integral

    n = basis.n_basis
    J = np.zeros((n, n))
    K = np.zeros((n, n))

    # Schwarz bounds
    Q = compute_schwarz_bounds(basis)
    D_max = np.abs(D).max()

    # Loop with screening
    for mu in range(n):
        for nu in range(mu + 1):
            Q_mn = Q[mu, nu]

            for lam in range(n):
                for sig in range(lam + 1):
                    Q_ls = Q[lam, sig]

                    # Combined screening with density
                    bound = Q_mn * Q_ls * max(abs(D[lam, sig]), abs(D[mu, lam]), abs(D[mu, sig]))
                    if bound < threshold:
                        continue

                    # Compute integral
                    val = electron_repulsion_integral(
                        basis[mu], basis[nu], basis[lam], basis[sig]
                    )

                    # Coulomb contribution
                    J[mu, nu] += D[lam, sig] * val * (2 if lam != sig else 1)
                    if mu != nu:
                        J[nu, mu] += D[lam, sig] * val * (2 if lam != sig else 1)

                    # Exchange contribution (more complex due to index permutations)
                    K[mu, lam] += D[nu, sig] * val * 0.5
                    K[mu, sig] += D[nu, lam] * val * 0.5
                    if mu != nu:
                        K[nu, lam] += D[mu, sig] * val * 0.5
                        K[nu, sig] += D[mu, lam] * val * 0.5

    # Symmetrize
    J = 0.5 * (J + J.T)
    K = 0.5 * (K + K.T)

    return J, K


# Convenience function to use best available method
def eri_tensor_optimized(basis, threshold: float = 1e-10,
                         use_gpu: bool = True,
                         verbose: bool = False) -> np.ndarray:
    """
    Compute ERI tensor using best available optimization.

    Automatically selects:
    1. GPU (if available and use_gpu=True)
    2. Screened CPU (with symmetry exploitation)

    Args:
        basis: BasisSet
        threshold: Screening threshold
        use_gpu: Whether to try GPU acceleration
        verbose: Print info

    Returns:
        ERI tensor
    """
    n = basis.n_basis

    if verbose:
        print(f"Computing ERI tensor for {n} basis functions")
        print(f"  Full tensor size: {n**4} = {n**4 * 8 / 1e6:.1f} MB")

    if use_gpu and HAS_TORCH and HAS_CUDA:
        if verbose:
            print("  Using GPU acceleration (PyTorch CUDA)")
        return eri_tensor_torch(basis, device='cuda', threshold=threshold)
    else:
        if verbose:
            if use_gpu and not HAS_CUDA:
                print("  GPU requested but not available, using CPU")
            print("  Using screened CPU computation with 8-fold symmetry")
        G, stats = eri_tensor_screened(basis, threshold=threshold, verbose=verbose)
        return G
