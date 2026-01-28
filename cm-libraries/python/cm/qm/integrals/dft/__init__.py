"""
Density Functional Theory Module

Provides DFT capabilities for molecular calculations:
- Numerical integration grids (Lebedev angular, Gauss-Chebyshev radial)
- Exchange-correlation functionals (LDA, GGA, meta-GGA, hybrid, range-separated)
- Dispersion corrections (DFT-D3)
- Kohn-Sham solver

Example:
    from cm.qm.integrals import kohn_sham

    result = kohn_sham(
        atoms=[('O', (0, 0, 0)), ('H', (0.96, 0, 0)), ('H', (-0.24, 0.93, 0))],
        functional='B3LYP',
        basis='6-31G*',
        dispersion='D3BJ'
    )
    print(f"Energy: {result.energy:.10f} Hartree")
"""

from .grid import MolecularGrid, AtomicGrid, RadialGrid, LebedevGrid, get_lebedev_grid
from .functionals import (
    XCFunctional,
    FunctionalType,
    DensityData,
    XCOutput,
    FunctionalRegistry,
    get_functional,
    list_functionals,
    # LDA
    SVWN5, svwn5,
    # GGA
    BLYP, PBE, blyp, pbe,
    # Hybrid
    B3LYP, PBE0, b3lyp, pbe0,
    # Range-separated
    CAMB3LYP, wB97XD, cam_b3lyp, wb97xd,
)

__all__ = [
    # Grid
    'MolecularGrid',
    'AtomicGrid',
    'RadialGrid',
    'LebedevGrid',
    'get_lebedev_grid',
    # Functionals base
    'XCFunctional',
    'FunctionalType',
    'DensityData',
    'XCOutput',
    'FunctionalRegistry',
    'get_functional',
    'list_functionals',
    # LDA
    'SVWN5', 'svwn5',
    # GGA
    'BLYP', 'PBE', 'blyp', 'pbe',
    # Hybrid
    'B3LYP', 'PBE0', 'b3lyp', 'pbe0',
    # Range-separated
    'CAMB3LYP', 'wB97XD', 'cam_b3lyp', 'wb97xd',
]
