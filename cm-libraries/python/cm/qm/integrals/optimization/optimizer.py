"""
Geometry Optimizer

Implements BFGS and other optimization algorithms for molecular geometry.

The BFGS algorithm updates an approximate Hessian:
    H_{k+1} = H_k + (y·yᵀ)/(yᵀ·s) - (H·s·sᵀ·H)/(sᵀ·H·s)

where s = x_{k+1} - x_k and y = g_{k+1} - g_k.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

from ..methods.hf import HFResult, hartree_fock
from ..methods.dft import DFTResult, kohn_sham
from ..gradients import hf_gradient, dft_gradient, GradientResult


ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR


class OptimizationMethod(Enum):
    """Optimization algorithm."""
    BFGS = "bfgs"
    STEEPEST_DESCENT = "sd"
    CONJUGATE_GRADIENT = "cg"


@dataclass
class ConvergenceCriteria:
    """Convergence criteria for optimization."""
    energy: float = 1e-6       # Energy change threshold (Hartree)
    gradient: float = 3e-4     # Max gradient threshold (Hartree/Bohr)
    rms_gradient: float = 1e-4 # RMS gradient threshold
    displacement: float = 1e-3  # Max displacement threshold (Bohr)
    rms_displacement: float = 6e-4  # RMS displacement threshold


@dataclass
class OptimizationResult:
    """
    Result of geometry optimization.

    Attributes:
        converged: Whether optimization converged
        n_iterations: Number of iterations
        energy: Final energy (Hartree)
        gradient: Final gradient
        atoms: Optimized geometry
        trajectory: List of geometries during optimization
        energies: List of energies during optimization
        final_result: Final electronic structure result
    """
    converged: bool
    n_iterations: int
    energy: float
    gradient: np.ndarray
    atoms: List[Tuple[str, Tuple[float, float, float]]]
    trajectory: List[List[Tuple[str, Tuple[float, float, float]]]]
    energies: List[float]
    final_result: Union[HFResult, DFTResult]


class GeometryOptimizer:
    """
    Geometry optimizer using quasi-Newton methods.

    Supports HF and DFT calculations with BFGS optimization.

    Example:
        optimizer = GeometryOptimizer(method='B3LYP', basis='6-31G*')
        result = optimizer.optimize(atoms)
        print(f"Optimized energy: {result.energy}")
    """

    def __init__(self, method: str = 'HF',
                 basis: str = 'STO-3G',
                 algorithm: str = 'BFGS',
                 max_iterations: int = 100,
                 convergence: ConvergenceCriteria = None):
        """
        Initialize optimizer.

        Args:
            method: Electronic structure method ('HF', 'B3LYP', 'PBE', etc.)
            basis: Basis set name
            algorithm: Optimization algorithm ('BFGS', 'SD', 'CG')
            max_iterations: Maximum optimization steps
            convergence: Convergence criteria
        """
        self.method = method.upper()
        self.basis = basis
        self.algorithm = OptimizationMethod(algorithm.lower())
        self.max_iterations = max_iterations
        self.convergence = convergence or ConvergenceCriteria()

        # Determine if DFT or HF
        self.is_dft = self.method not in ('HF', 'RHF', 'UHF')

    def optimize(self, atoms: List[Tuple[str, Tuple[float, float, float]]],
                 verbose: bool = True) -> OptimizationResult:
        """
        Optimize molecular geometry.

        Args:
            atoms: Initial geometry (element, (x, y, z)) in Angstroms
            verbose: Print optimization progress

        Returns:
            OptimizationResult with optimized geometry
        """
        n_atoms = len(atoms)

        # Convert to flat coordinate array (in Bohr)
        elements = [a[0] for a in atoms]
        coords = np.array([a[1] for a in atoms]) * ANGSTROM_TO_BOHR
        x = coords.flatten()

        # Initialize Hessian (identity or better estimate)
        H = np.eye(3 * n_atoms) * 0.5  # Initial Hessian guess

        # Storage
        trajectory = [atoms]
        energies = []

        if verbose:
            print(f"Geometry Optimization")
            print(f"Method: {self.method}/{self.basis}")
            print(f"Algorithm: {self.algorithm.value.upper()}")
            print("-" * 60)
            print(f"{'Iter':>4} {'Energy':>16} {'dE':>12} {'|Grad|':>10} {'|Step|':>10}")
            print("-" * 60)

        # Initial energy and gradient
        current_atoms = self._coords_to_atoms(elements, x)
        energy, gradient = self._compute_energy_gradient(current_atoms)
        energies.append(energy)
        g = gradient.flatten()

        converged = False
        result = None

        for iteration in range(self.max_iterations):
            # Check convergence
            max_grad = np.max(np.abs(g))
            rms_grad = np.sqrt(np.mean(g ** 2))

            if max_grad < self.convergence.gradient and \
               rms_grad < self.convergence.rms_gradient:
                converged = True
                if verbose:
                    print("-" * 60)
                    print("Optimization converged!")
                break

            # Compute search direction
            if self.algorithm == OptimizationMethod.BFGS:
                # BFGS direction: p = -H^(-1) g
                try:
                    p = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    p = -g  # Fall back to steepest descent
            elif self.algorithm == OptimizationMethod.STEEPEST_DESCENT:
                p = -g
            else:  # Conjugate gradient
                if iteration == 0:
                    p = -g
                else:
                    beta = np.dot(g, g) / np.dot(g_old, g_old)
                    p = -g + beta * p_old

            # Line search (simple backtracking)
            alpha = self._line_search(elements, x, p, energy, g)

            # Update position
            s = alpha * p
            x_new = x + s

            # Compute new energy and gradient
            current_atoms = self._coords_to_atoms(elements, x_new)
            energy_new, gradient_new = self._compute_energy_gradient(current_atoms)
            g_new = gradient_new.flatten()

            # Print progress
            dE = energy_new - energy
            step_size = np.linalg.norm(s)

            if verbose:
                print(f"{iteration+1:4d} {energy_new:16.10f} {dE:12.2e} "
                      f"{max_grad:10.2e} {step_size:10.2e}")

            # Check energy convergence
            if abs(dE) < self.convergence.energy:
                if max_grad < self.convergence.gradient:
                    converged = True
                    if verbose:
                        print("-" * 60)
                        print("Optimization converged!")
                    break

            # Update Hessian (BFGS)
            if self.algorithm == OptimizationMethod.BFGS:
                y = g_new - g
                sy = np.dot(s, y)

                if sy > 1e-10:  # Curvature condition
                    Hs = H @ s
                    sHs = np.dot(s, Hs)

                    H = H + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs

            # Store for conjugate gradient
            g_old = g.copy()
            p_old = p.copy()

            # Update state
            x = x_new
            energy = energy_new
            g = g_new
            energies.append(energy)
            trajectory.append(current_atoms)

        # Final result
        final_atoms = self._coords_to_atoms(elements, x)
        _, final_result = self._compute_energy_gradient(final_atoms, return_result=True)

        if verbose and not converged:
            print("-" * 60)
            print(f"Optimization did not converge after {self.max_iterations} iterations")

        if verbose:
            print(f"\nFinal energy: {energy:.10f} Hartree")

        return OptimizationResult(
            converged=converged,
            n_iterations=iteration + 1,
            energy=energy,
            gradient=gradient_new,
            atoms=final_atoms,
            trajectory=trajectory,
            energies=energies,
            final_result=final_result
        )

    def _compute_energy_gradient(self, atoms: List, return_result: bool = False):
        """Compute energy and gradient at given geometry."""
        if self.is_dft:
            result = kohn_sham(atoms, functional=self.method, basis=self.basis)
            grad_result = dft_gradient(dft_result=result)
        else:
            result = hartree_fock(atoms, basis=self.basis)
            grad_result = hf_gradient(hf_result=result)

        if return_result:
            return grad_result.gradient, result
        return result.energy, grad_result.gradient

    def _line_search(self, elements: List[str], x: np.ndarray,
                     p: np.ndarray, f0: float, g0: np.ndarray,
                     alpha_max: float = 1.0) -> float:
        """
        Simple backtracking line search.

        Armijo condition: f(x + αp) <= f(x) + c₁ α ∇f·p
        """
        c1 = 1e-4
        rho = 0.5
        alpha = alpha_max

        g0_p = np.dot(g0, p)

        for _ in range(20):
            x_new = x + alpha * p
            atoms = self._coords_to_atoms(elements, x_new)

            if self.is_dft:
                result = kohn_sham(atoms, functional=self.method, basis=self.basis)
            else:
                result = hartree_fock(atoms, basis=self.basis)

            f_new = result.energy

            # Armijo condition
            if f_new <= f0 + c1 * alpha * g0_p:
                return alpha

            alpha *= rho

        return alpha

    def _coords_to_atoms(self, elements: List[str],
                         x: np.ndarray) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Convert flat coordinate array to atoms list."""
        coords = x.reshape(-1, 3) * BOHR_TO_ANGSTROM
        return [(e, tuple(c)) for e, c in zip(elements, coords)]


def optimize_geometry(atoms: List[Tuple[str, Tuple[float, float, float]]],
                      method: str = 'HF',
                      basis: str = 'STO-3G',
                      verbose: bool = True) -> OptimizationResult:
    """
    Optimize molecular geometry.

    Convenience function for GeometryOptimizer.

    Args:
        atoms: Initial geometry (element, (x, y, z)) in Angstroms
        method: Electronic structure method
        basis: Basis set name
        verbose: Print progress

    Returns:
        OptimizationResult with optimized geometry

    Example:
        # Optimize water with B3LYP
        result = optimize_geometry(
            atoms=[('O', (0, 0, 0)), ('H', (1.0, 0, 0)), ('H', (-0.3, 0.95, 0))],
            method='B3LYP',
            basis='6-31G*'
        )
        print(f"Optimized energy: {result.energy}")
        print(f"Final geometry: {result.atoms}")
    """
    optimizer = GeometryOptimizer(method=method, basis=basis)
    return optimizer.optimize(atoms, verbose=verbose)
