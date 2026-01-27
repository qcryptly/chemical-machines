"""
Quantum Chemistry Integral Evaluation Module

This module provides proper evaluation of molecular integrals using
Gaussian-type orbitals (GTOs) with the McMurchie-Davidson scheme.

Implements:
- Overlap integrals (S)
- Kinetic energy integrals (T)
- Nuclear attraction integrals (V) via Boys function
- Two-electron repulsion integrals (ERI) via Boys function
- Hartree-Fock solver (RHF)
- Molecular orbital visualization (isosurfaces)
"""

from .basis import GaussianPrimitive, ContractedGaussian, BasisSet, BasisFunction
from .boys import boys_function
from .overlap import overlap_integral, overlap_matrix
from .kinetic import kinetic_integral, kinetic_matrix
from .nuclear import nuclear_attraction_integral, nuclear_attraction_matrix
from .eri import electron_repulsion_integral, eri_tensor
from .hf import HartreeFockSolver, HFResult, hartree_fock
from .orbital import (
    OrbitalGrid,
    create_orbital_grid,
    evaluate_orbital_on_grid,
    extract_orbital_isosurface,
    marching_cubes,
)

__all__ = [
    # Basis functions
    'GaussianPrimitive',
    'ContractedGaussian',
    'BasisSet',
    'BasisFunction',
    # Boys function
    'boys_function',
    # One-electron integrals
    'overlap_integral',
    'overlap_matrix',
    'kinetic_integral',
    'kinetic_matrix',
    'nuclear_attraction_integral',
    'nuclear_attraction_matrix',
    # Two-electron integrals
    'electron_repulsion_integral',
    'eri_tensor',
    # Hartree-Fock
    'HartreeFockSolver',
    'HFResult',
    'hartree_fock',
    # Orbital visualization
    'OrbitalGrid',
    'create_orbital_grid',
    'evaluate_orbital_on_grid',
    'extract_orbital_isosurface',
    'marching_cubes',
]
