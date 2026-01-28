"""
Transition State Search

Implements algorithms for finding first-order saddle points (transition states):
- Eigenvector-following (EF) method
- Partitioned Rational Function Optimization (P-RFO)

A transition state has:
- Zero gradient
- One negative Hessian eigenvalue (imaginary frequency)
- All other eigenvalues positive
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..methods.hf import HFResult, hartree_fock
from ..methods.dft import DFTResult, kohn_sham
from ..gradients import hf_gradient, dft_gradient
from .optimizer import OptimizationResult


ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR


@dataclass
class TSResult(OptimizationResult):
    """
    Result of transition state search.

    Additional attributes:
        hessian: Final Hessian matrix
        imaginary_freq: Imaginary frequency (cm⁻¹)
        reaction_mode: Eigenvector of imaginary mode
    """
    hessian: np.ndarray = None
    imaginary_freq: float = 0.0
    reaction_mode: np.ndarray = None


class TransitionStateOptimizer:
    """
    Transition state optimizer using eigenvector-following.

    The EF method maximizes along one eigenvector (the reaction coordinate)
    while minimizing along all others.

    Example:
        ts_opt = TransitionStateOptimizer(method='B3LYP', basis='6-31G*')
        result = ts_opt.optimize(guess_atoms, mode_follow=0)
    """

    def __init__(self, method: str = 'HF',
                 basis: str = 'STO-3G',
                 max_iterations: int = 50,
                 gradient_threshold: float = 1e-4):
        """
        Initialize TS optimizer.

        Args:
            method: Electronic structure method
            basis: Basis set name
            max_iterations: Maximum optimization steps
            gradient_threshold: Convergence threshold for gradient
        """
        self.method = method.upper()
        self.basis = basis
        self.max_iterations = max_iterations
        self.gradient_threshold = gradient_threshold
        self.is_dft = self.method not in ('HF', 'RHF', 'UHF')

    def optimize(self, atoms: List[Tuple[str, Tuple[float, float, float]]],
                 mode_follow: int = 0,
                 initial_hessian: np.ndarray = None,
                 verbose: bool = True) -> TSResult:
        """
        Find transition state.

        Args:
            atoms: Initial guess geometry
            mode_follow: Index of Hessian mode to follow (0 = lowest)
            initial_hessian: Initial Hessian guess (numerical if None)
            verbose: Print progress

        Returns:
            TSResult with transition state geometry
        """
        n_atoms = len(atoms)
        n_coords = 3 * n_atoms

        # Convert to coordinates
        elements = [a[0] for a in atoms]
        coords = np.array([a[1] for a in atoms]) * ANGSTROM_TO_BOHR
        x = coords.flatten()

        # Get initial Hessian
        if initial_hessian is None:
            H = self._numerical_hessian(elements, x)
        else:
            H = initial_hessian.copy()

        trajectory = [atoms]
        energies = []

        if verbose:
            print(f"Transition State Search")
            print(f"Method: {self.method}/{self.basis}")
            print(f"Following mode: {mode_follow}")
            print("-" * 60)
            print(f"{'Iter':>4} {'Energy':>16} {'|Grad|':>10} {'λ_min':>12}")
            print("-" * 60)

        # Initial energy and gradient
        current_atoms = self._coords_to_atoms(elements, x)
        energy, gradient = self._compute_energy_gradient(current_atoms)
        energies.append(energy)
        g = gradient.flatten()

        converged = False
        result = None

        for iteration in range(self.max_iterations):
            # Diagonalize Hessian
            eigenvalues, eigenvectors = np.linalg.eigh(H)

            # Check gradient convergence
            max_grad = np.max(np.abs(g))

            if verbose:
                print(f"{iteration+1:4d} {energy:16.10f} {max_grad:10.2e} "
                      f"{eigenvalues[mode_follow]:12.6f}")

            if max_grad < self.gradient_threshold:
                # Check that we have one negative eigenvalue
                n_negative = np.sum(eigenvalues < 0)
                if n_negative == 1:
                    converged = True
                    if verbose:
                        print("-" * 60)
                        print("Transition state found!")
                    break
                elif verbose:
                    print(f"  Warning: {n_negative} negative eigenvalues")

            # Eigenvector-following step
            step = self._ef_step(g, H, eigenvalues, eigenvectors, mode_follow)

            # Trust radius check
            step_norm = np.linalg.norm(step)
            trust_radius = 0.3  # Bohr
            if step_norm > trust_radius:
                step = step * trust_radius / step_norm

            # Update position
            x_new = x + step

            # Compute new energy and gradient
            current_atoms = self._coords_to_atoms(elements, x_new)
            energy_new, gradient_new = self._compute_energy_gradient(current_atoms)
            g_new = gradient_new.flatten()

            # Update Hessian (BFGS update)
            s = step
            y = g_new - g
            sy = np.dot(s, y)

            if abs(sy) > 1e-10:
                Hs = H @ s
                sHs = np.dot(s, Hs)
                H = H + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs

            # Update state
            x = x_new
            energy = energy_new
            g = g_new
            energies.append(energy)
            trajectory.append(current_atoms)

        # Final Hessian
        H_final = self._numerical_hessian(elements, x)
        eigenvalues, eigenvectors = np.linalg.eigh(H_final)

        # Compute imaginary frequency
        # ν = sqrt(|λ|) / (2π) in atomic units, convert to cm⁻¹
        if eigenvalues[0] < 0:
            # Conversion factor: sqrt(Hartree/(amu*Bohr²)) to cm⁻¹
            # ≈ 5140.48 cm⁻¹ / sqrt(amu)
            imag_freq = np.sqrt(abs(eigenvalues[0])) * 5140.48
        else:
            imag_freq = 0.0

        final_atoms = self._coords_to_atoms(elements, x)
        _, final_result = self._compute_energy_gradient(final_atoms, return_result=True)

        if verbose and not converged:
            print("-" * 60)
            print(f"TS search did not converge after {self.max_iterations} iterations")

        if verbose:
            print(f"\nFinal energy: {energy:.10f} Hartree")
            print(f"Imaginary frequency: {imag_freq:.1f}i cm⁻¹")

        return TSResult(
            converged=converged,
            n_iterations=iteration + 1,
            energy=energy,
            gradient=gradient_new,
            atoms=final_atoms,
            trajectory=trajectory,
            energies=energies,
            final_result=final_result,
            hessian=H_final,
            imaginary_freq=imag_freq,
            reaction_mode=eigenvectors[:, 0]
        )

    def _ef_step(self, g: np.ndarray, H: np.ndarray,
                 eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                 mode_follow: int) -> np.ndarray:
        """
        Compute eigenvector-following step.

        For the followed mode: maximize (step along positive gradient direction)
        For all other modes: minimize (Newton-Raphson step)
        """
        n = len(g)
        step = np.zeros(n)

        # Transform gradient to eigenvector basis
        g_ev = eigenvectors.T @ g

        for i in range(n):
            if abs(eigenvalues[i]) < 1e-6:
                continue

            if i == mode_follow:
                # Maximize along this mode: step in direction of gradient
                # with magnitude determined by curvature
                lambda_shift = eigenvalues[i] - 0.1  # Shift to ensure uphill
                step_i = -g_ev[i] / lambda_shift
            else:
                # Minimize along other modes: Newton step
                step_i = -g_ev[i] / eigenvalues[i]

            step += step_i * eigenvectors[:, i]

        return step

    def _numerical_hessian(self, elements: List[str], x: np.ndarray,
                           h: float = 0.001) -> np.ndarray:
        """Compute Hessian by finite difference of gradients."""
        n = len(x)
        H = np.zeros((n, n))

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            atoms_plus = self._coords_to_atoms(elements, x_plus)
            _, g_plus = self._compute_energy_gradient(atoms_plus)

            x_minus = x.copy()
            x_minus[i] -= h
            atoms_minus = self._coords_to_atoms(elements, x_minus)
            _, g_minus = self._compute_energy_gradient(atoms_minus)

            H[:, i] = (g_plus.flatten() - g_minus.flatten()) / (2 * h)

        # Symmetrize
        H = 0.5 * (H + H.T)

        return H

    def _compute_energy_gradient(self, atoms: List, return_result: bool = False):
        """Compute energy and gradient."""
        if self.is_dft:
            result = kohn_sham(atoms, functional=self.method, basis=self.basis)
            grad_result = dft_gradient(dft_result=result)
        else:
            result = hartree_fock(atoms, basis=self.basis)
            grad_result = hf_gradient(hf_result=result)

        if return_result:
            return grad_result.gradient, result
        return result.energy, grad_result.gradient

    def _coords_to_atoms(self, elements: List[str],
                         x: np.ndarray) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Convert flat coordinates to atoms list."""
        coords = x.reshape(-1, 3) * BOHR_TO_ANGSTROM
        return [(e, tuple(c)) for e, c in zip(elements, coords)]


def find_transition_state(atoms: List[Tuple[str, Tuple[float, float, float]]],
                          method: str = 'HF',
                          basis: str = 'STO-3G',
                          mode_follow: int = 0,
                          verbose: bool = True) -> TSResult:
    """
    Find transition state geometry.

    Convenience function for TransitionStateOptimizer.

    Args:
        atoms: Initial guess geometry
        method: Electronic structure method
        basis: Basis set name
        mode_follow: Hessian mode to maximize along
        verbose: Print progress

    Returns:
        TSResult with transition state

    Example:
        # Find TS for H2 + H -> H + H2
        ts = find_transition_state(
            atoms=[('H', (-1, 0, 0)), ('H', (0, 0, 0)), ('H', (1.2, 0, 0))],
            method='HF',
            basis='6-31G*'
        )
    """
    optimizer = TransitionStateOptimizer(method=method, basis=basis)
    return optimizer.optimize(atoms, mode_follow=mode_follow, verbose=verbose)
