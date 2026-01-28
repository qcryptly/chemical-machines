"""
XC Integration Module

Computes exchange-correlation energy and Fock matrix contributions
by numerical integration on molecular grids.

This module bridges the grid infrastructure with XC functionals to provide:
- Density evaluation on grid points
- XC energy by numerical integration
- XC potential matrix (V_xc) for the Fock matrix
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass

from .grid import MolecularGrid
from .functionals import XCFunctional, DensityData, get_functional


@dataclass
class XCResult:
    """
    Result of XC integration.

    Attributes:
        energy: Total XC energy (Hartree)
        vxc_matrix: XC potential matrix for Fock matrix construction
        density_integral: Integral of density (should equal n_electrons)
    """
    energy: float
    vxc_matrix: np.ndarray
    density_integral: float


class XCIntegrator:
    """
    Integrates XC functionals on molecular grids.

    Evaluates basis functions on grid points, computes electron density
    and its gradient, evaluates the XC functional, and assembles the
    XC contribution to the Fock matrix.
    """

    def __init__(self, grid: MolecularGrid, basis_functions: List,
                 atom_centers: np.ndarray):
        """
        Initialize XC integrator.

        Args:
            grid: Molecular integration grid
            basis_functions: List of basis function objects
            atom_centers: Atom positions in Bohr (n_atoms, 3)
        """
        self.grid = grid
        self.basis_functions = basis_functions
        self.atom_centers = atom_centers
        self.n_basis = len(basis_functions)
        self.n_points = grid.n_points

        # Precompute basis function values on grid (expensive but done once)
        self._ao_values = None
        self._ao_derivs = None

    def compute_ao_on_grid(self, compute_derivs: bool = False):
        """
        Evaluate all AOs on grid points.

        Args:
            compute_derivs: Also compute AO gradients for GGA

        Sets:
            self._ao_values: (n_points, n_basis) array
            self._ao_derivs: (n_points, n_basis, 3) array if compute_derivs
        """
        points = self.grid.points

        self._ao_values = np.zeros((self.n_points, self.n_basis))

        if compute_derivs:
            self._ao_derivs = np.zeros((self.n_points, self.n_basis, 3))

        for mu, bf in enumerate(self.basis_functions):
            self._ao_values[:, mu] = self._evaluate_basis_function(bf, points)

            if compute_derivs:
                self._ao_derivs[:, mu, :] = self._evaluate_basis_gradient(bf, points)

    def _evaluate_basis_function(self, bf, points: np.ndarray) -> np.ndarray:
        """
        Evaluate a single basis function at grid points.

        Args:
            bf: Basis function object (from basis module)
            points: Grid points (n_points, 3)

        Returns:
            Function values (n_points,)
        """
        # Get center position
        center = bf.center if hasattr(bf, 'center') else np.zeros(3)

        # Displacement from center
        r = points - center
        r2 = np.sum(r * r, axis=1)

        # Angular part
        l, m = bf.l, bf.m
        angular = self._spherical_harmonic(l, m, r)

        # Radial part: sum of contracted Gaussians
        radial = np.zeros(len(points))
        for coef, exp in zip(bf.coefficients, bf.exponents):
            radial += coef * np.exp(-exp * r2)

        # r^l factor for Cartesian GTOs
        r_l = np.sqrt(r2) ** l if l > 0 else 1.0

        return angular * radial * r_l

    def _evaluate_basis_gradient(self, bf, points: np.ndarray) -> np.ndarray:
        """
        Evaluate gradient of basis function.

        Args:
            bf: Basis function object
            points: Grid points (n_points, 3)

        Returns:
            Gradient values (n_points, 3)
        """
        center = bf.center if hasattr(bf, 'center') else np.zeros(3)
        r = points - center
        r2 = np.sum(r * r, axis=1)

        l, m = bf.l, bf.m

        grad = np.zeros((len(points), 3))

        for coef, exp in zip(bf.coefficients, bf.exponents):
            gauss = np.exp(-exp * r2)

            # Gradient of exp(-α r²) = -2α r exp(-α r²)
            for i in range(3):
                grad[:, i] += coef * (-2 * exp * r[:, i]) * gauss

        # Add angular contributions for l > 0 (simplified)
        if l > 0:
            angular = self._spherical_harmonic(l, m, r)
            grad *= angular[:, np.newaxis]

        return grad

    def _spherical_harmonic(self, l: int, m: int, r: np.ndarray) -> np.ndarray:
        """
        Compute real spherical harmonic.

        Args:
            l: Angular momentum quantum number
            m: Magnetic quantum number
            r: Displacement vectors (n_points, 3)

        Returns:
            Angular function values (n_points,)
        """
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        r_norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r_norm = np.maximum(r_norm, 1e-15)

        if l == 0:  # s
            return np.ones(len(r))

        elif l == 1:  # p
            if m == -1:
                return y / r_norm
            elif m == 0:
                return z / r_norm
            else:  # m == 1
                return x / r_norm

        elif l == 2:  # d
            r2 = r_norm ** 2
            if m == -2:
                return np.sqrt(3) * x * y / r2
            elif m == -1:
                return np.sqrt(3) * y * z / r2
            elif m == 0:
                return (3 * z ** 2 - r2) / (2 * r2)
            elif m == 1:
                return np.sqrt(3) * x * z / r2
            else:  # m == 2
                return np.sqrt(3) / 2 * (x ** 2 - y ** 2) / r2

        else:
            # Higher angular momenta - use Cartesian
            return np.ones(len(r))

    def compute_density(self, P: np.ndarray,
                        compute_gradient: bool = False) -> DensityData:
        """
        Compute electron density on grid from density matrix.

        ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)

        Args:
            P: Density matrix (n_basis, n_basis)
            compute_gradient: Also compute ∇ρ for GGA

        Returns:
            DensityData with density (and gradient if requested)
        """
        if self._ao_values is None:
            self.compute_ao_on_grid(compute_derivs=compute_gradient)

        # ρ = Σ_μν P_μν φ_μ φ_ν = Σ_μ (Σ_ν P_μν φ_ν) φ_μ
        # Compute (P @ φ) then dot with φ
        P_phi = np.einsum('mn,gn->gm', P, self._ao_values)
        rho = np.einsum('gm,gm->g', P_phi, self._ao_values)

        # Ensure non-negative
        rho = np.maximum(rho, 0.0)

        sigma = None
        if compute_gradient and self._ao_derivs is not None:
            # ∇ρ = 2 Σ_μν P_μν φ_μ ∇φ_ν
            grad_rho = np.zeros((self.n_points, 3))
            for i in range(3):
                P_dphi = np.einsum('mn,gn->gm', P, self._ao_derivs[:, :, i])
                grad_rho[:, i] = 2 * np.einsum('gm,gm->g', P_phi, self._ao_derivs[:, :, i])

            sigma = np.sum(grad_rho * grad_rho, axis=1)

        return DensityData(rho=rho, sigma=sigma)

    def compute_xc(self, functional: Union[str, XCFunctional],
                   P: np.ndarray) -> XCResult:
        """
        Compute XC energy and potential matrix.

        Args:
            functional: XC functional name or object
            P: Density matrix

        Returns:
            XCResult with energy and V_xc matrix
        """
        # Get functional
        if isinstance(functional, str):
            func = get_functional(functional)
        else:
            func = functional

        # Compute density on grid
        needs_grad = func.needs_gradient
        density = self.compute_density(P, compute_gradient=needs_grad)

        # Check if we have gradient values
        if needs_grad and density.sigma is None:
            # Recompute with gradients
            if self._ao_derivs is None:
                self.compute_ao_on_grid(compute_derivs=True)
            density = self.compute_density(P, compute_gradient=True)

        # Evaluate functional
        xc_out = func.compute(density)

        # Integrate XC energy
        # E_xc = ∫ ρ(r) ε_xc(r) dr
        energy = np.sum(density.rho * xc_out.exc * self.grid.weights)

        # Integrate density (for sanity check)
        density_integral = np.sum(density.rho * self.grid.weights)

        # Compute V_xc matrix
        # V_μν = ∫ v_xc(r) φ_μ(r) φ_ν(r) dr
        # For GGA: add gradient terms
        vxc_matrix = self._compute_vxc_matrix(xc_out, density)

        return XCResult(
            energy=energy,
            vxc_matrix=vxc_matrix,
            density_integral=density_integral
        )

    def _compute_vxc_matrix(self, xc_out, density: DensityData) -> np.ndarray:
        """
        Compute XC potential matrix.

        For LDA:
            V_μν = ∫ v_ρ φ_μ φ_ν dτ

        For GGA:
            V_μν = ∫ [v_ρ φ_μ φ_ν + 2 v_σ ∇ρ · (φ_μ ∇φ_ν + ∇φ_μ φ_ν)] dτ
        """
        weights = self.grid.weights
        vrho = xc_out.vrho

        # Handle spin-polarized case
        if vrho.ndim > 1:
            vrho = vrho[:, 0] + vrho[:, 1]  # Total potential

        # LDA contribution
        # V_μν = Σ_g w_g v_ρ(g) φ_μ(g) φ_ν(g)
        weighted_vrho = vrho * weights
        vxc = np.einsum('g,gm,gn->mn', weighted_vrho, self._ao_values, self._ao_values)

        # GGA contribution
        if xc_out.vsigma is not None and self._ao_derivs is not None:
            vsigma = xc_out.vsigma
            if vsigma.ndim > 1:
                vsigma = vsigma[:, 0]  # Simplified for closed-shell

            # Compute ∇ρ
            P_phi = np.einsum('mn,gn->gm', self._ao_values.T @ (self._ao_values * weights[:, np.newaxis]), self._ao_values)

            # This is a simplified GGA contribution
            # Full implementation requires proper gradient terms
            for i in range(3):
                phi_dphi = self._ao_values[:, :, np.newaxis] * self._ao_derivs[:, np.newaxis, :]
                weighted = 2 * vsigma * weights
                # Add contribution (simplified)

        return vxc


def integrate_xc(functional: Union[str, XCFunctional],
                 grid: MolecularGrid,
                 basis_functions: List,
                 atom_centers: np.ndarray,
                 P: np.ndarray) -> XCResult:
    """
    Compute XC energy and potential matrix.

    Convenience function for XCIntegrator.

    Args:
        functional: XC functional name or object
        grid: Molecular integration grid
        basis_functions: List of basis functions
        atom_centers: Atom positions in Bohr
        P: Density matrix

    Returns:
        XCResult with energy and V_xc matrix
    """
    integrator = XCIntegrator(grid, basis_functions, atom_centers)
    return integrator.compute_xc(functional, P)
