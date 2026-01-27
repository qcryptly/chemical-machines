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
- CCSD(T) coupled cluster
- Molecular orbital visualization (isosurfaces)

Basis sets:
- STO-3G: Minimal basis (teaching/qualitative)
- cc-pVTZ: Triple-zeta (publication quality)
- cc-pVQZ: Quadruple-zeta (benchmark accuracy)
"""

from .basis import GaussianPrimitive, ContractedGaussian, BasisSet, BasisFunction
from .boys import boys_function
from .overlap import overlap_integral, overlap_matrix
from .kinetic import kinetic_integral, kinetic_matrix
from .nuclear import nuclear_attraction_integral, nuclear_attraction_matrix
from .eri import electron_repulsion_integral, eri_tensor
from .eri_optimized import (
    eri_tensor_screened,
    eri_tensor_optimized,
    eri_direct,
    compute_schwarz_bounds,
)
from .hf import HartreeFockSolver, HFResult, hartree_fock
from .ccsd import CCSDResult, ccsd, transform_integrals_to_mo
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
    'eri_tensor_screened',
    'eri_tensor_optimized',
    'eri_direct',
    'compute_schwarz_bounds',
    # Hartree-Fock
    'HartreeFockSolver',
    'HFResult',
    'hartree_fock',
    # Coupled Cluster
    'CCSDResult',
    'ccsd',
    'transform_integrals_to_mo',
    # Orbital visualization
    'OrbitalGrid',
    'create_orbital_grid',
    'evaluate_orbital_on_grid',
    'extract_orbital_isosurface',
    'marching_cubes',
]
