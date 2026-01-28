"""
Quantum Chemistry Integral Evaluation Module

This module provides comprehensive quantum chemistry capabilities using
Gaussian-type orbitals (GTOs) with the McMurchie-Davidson scheme.

Implements:
- Overlap integrals (S)
- Kinetic energy integrals (T)
- Nuclear attraction integrals (V) via Boys function
- Two-electron repulsion integrals (ERI) via Boys function
- Hartree-Fock solver (RHF, UHF)
- Kohn-Sham DFT with various functionals
- MP2 perturbation theory
- CCSD(T) coupled cluster
- Analytic gradients (HF, DFT)
- Geometry optimization (BFGS, internal coordinates)
- Transition state search (eigenvector-following)
- Frequency analysis (harmonic)
- Thermochemistry (ZPE, enthalpy, entropy, Gibbs energy)
- TDDFT excited states (linear response, TDA)
- Molecular properties (dipole, polarizability)
- Implicit solvation (PCM)
- Effective core potentials (LANL2DZ)
- Molecular orbital visualization (isosurfaces)

Basis sets:
- STO-3G: Minimal basis (teaching/qualitative)
- 3-21G: Split-valence (geometry optimization)
- 6-31G, 6-31G*, 6-31G**: Standard split-valence with polarization
- cc-pVDZ, cc-pVTZ, cc-pVQZ: Correlation-consistent (publication quality)

DFT Functionals:
- LDA: SVWN5
- GGA: BLYP, PBE
- Hybrid: B3LYP, PBE0, M06, M06-2X
- Range-separated: CAM-B3LYP, ωB97X-D, ωB97M-V

Directory structure:
- basis/: Basis set definitions and ECP data
- one_electron/: Overlap, kinetic, and nuclear attraction integrals
- two_electron/: Electron repulsion integrals (ERI)
- utils/: Helper functions (Boys function)
- methods/: Quantum chemistry methods (HF, UHF, DFT, MP2, CCSD)
- dft/: DFT grids and functionals
- gradients/: Analytic gradient calculations
- optimization/: Geometry optimization and TS search
- properties/: Hessian, frequencies, thermochemistry, properties
- tddft/: Time-dependent DFT for excited states
- solvation/: Implicit solvation models (PCM)
- visualization/: Molecular orbital visualization
"""

from .basis import (
    GaussianPrimitive,
    ContractedGaussian,
    BasisSet,
    BasisFunction,
    BasisRegistry,
    get_atomic_number,
    get_atomic_mass,
)
from .utils import boys_function
from .one_electron import (
    overlap_integral,
    overlap_matrix,
    kinetic_integral,
    kinetic_matrix,
    nuclear_attraction_integral,
    nuclear_attraction_matrix,
)
from .two_electron import (
    electron_repulsion_integral,
    eri_tensor,
    eri_tensor_screened,
    eri_tensor_optimized,
    eri_direct,
    compute_schwarz_bounds,
    compute_J_matrix,
    compute_K_matrix,
)
from .methods import (
    # HF
    HartreeFockSolver,
    HFResult,
    hartree_fock,
    # UHF
    UnrestrictedHartreeFockSolver,
    UHFResult,
    uhf,
    # DFT
    KohnShamSolver,
    DFTResult,
    kohn_sham,
    # MP2
    MP2Result,
    mp2,
    # CCSD
    CCSDResult,
    ccsd,
    transform_integrals_to_mo,
)
from .dft import (
    # Grid
    MolecularGrid,
    AtomicGrid,
    RadialGrid,
    LebedevGrid,
    # Functionals
    XCFunctional,
    FunctionalType,
    DensityData,
    get_functional,
    list_functionals,
    # Specific functionals
    SVWN5, BLYP, PBE, B3LYP, PBE0, CAMB3LYP,
)
from .visualization import (
    OrbitalGrid,
    create_orbital_grid,
    evaluate_orbital_on_grid,
    extract_orbital_isosurface,
    marching_cubes,
)
from .gradients import (
    GradientResult,
    hf_gradient,
    dft_gradient,
    numerical_gradient,
)
from .optimization import (
    OptimizationResult,
    GeometryOptimizer,
    optimize_geometry,
    InternalCoordinates,
    TSResult,
    TransitionStateOptimizer,
    find_transition_state,
)
from .properties import (
    HessianResult,
    compute_hessian,
    FrequencyResult,
    harmonic_frequencies,
    ThermochemistryResult,
    thermochemistry,
    DipoleResult,
    dipole_moment,
    PolarizabilityResult,
    static_polarizability,
)
from .tddft import (
    TDDFTResult,
    TDAResult,
    tddft,
    tda,
)
from .solvation import (
    PCMResult,
    PCMSolver,
    compute_solvation_energy,
    build_cavity,
)

__all__ = [
    # Basis functions
    'GaussianPrimitive',
    'ContractedGaussian',
    'BasisSet',
    'BasisFunction',
    'BasisRegistry',
    'get_atomic_number',
    'get_atomic_mass',

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
    'compute_J_matrix',
    'compute_K_matrix',

    # Hartree-Fock (RHF)
    'HartreeFockSolver',
    'HFResult',
    'hartree_fock',

    # Unrestricted Hartree-Fock
    'UnrestrictedHartreeFockSolver',
    'UHFResult',
    'uhf',

    # Kohn-Sham DFT
    'KohnShamSolver',
    'DFTResult',
    'kohn_sham',

    # MP2
    'MP2Result',
    'mp2',

    # Coupled Cluster
    'CCSDResult',
    'ccsd',
    'transform_integrals_to_mo',

    # DFT Grid
    'MolecularGrid',
    'AtomicGrid',
    'RadialGrid',
    'LebedevGrid',

    # DFT Functionals
    'XCFunctional',
    'FunctionalType',
    'DensityData',
    'get_functional',
    'list_functionals',
    'SVWN5',
    'BLYP',
    'PBE',
    'B3LYP',
    'PBE0',
    'CAMB3LYP',

    # Orbital visualization
    'OrbitalGrid',
    'create_orbital_grid',
    'evaluate_orbital_on_grid',
    'extract_orbital_isosurface',
    'marching_cubes',

    # Gradients
    'GradientResult',
    'hf_gradient',
    'dft_gradient',
    'numerical_gradient',

    # Geometry optimization
    'OptimizationResult',
    'GeometryOptimizer',
    'optimize_geometry',
    'InternalCoordinates',
    'TSResult',
    'TransitionStateOptimizer',
    'find_transition_state',

    # Molecular properties
    'HessianResult',
    'compute_hessian',
    'FrequencyResult',
    'harmonic_frequencies',
    'ThermochemistryResult',
    'thermochemistry',
    'DipoleResult',
    'dipole_moment',
    'PolarizabilityResult',
    'static_polarizability',

    # Excited states (TDDFT)
    'TDDFTResult',
    'TDAResult',
    'tddft',
    'tda',

    # Solvation
    'PCMResult',
    'PCMSolver',
    'compute_solvation_energy',
    'build_cavity',
]
