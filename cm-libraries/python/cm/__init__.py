"""
Chemical Machines (cm) Library

A library for Chemical Machines workspace functionality.

Modules:
    views: HTML output rendering for cells and workspaces
    symbols: LaTeX math and symbol rendering with notation styles
    qm: Quantum mechanics - atoms, Slater determinants, spin-orbitals, matrix elements
    data: Benchmark molecular databases (NIST, PubChem, QM9)

Quick Start - Atoms:
    from cm import qm

    # Create an atom with automatic electron configuration
    C = qm.atom('C')
    print(C.configuration.label)  # "1s² 2s² 2p²"

    # Create Slater determinant and compute matrix elements
    psi = C.slater_determinant()
    H = qm.hamiltonian()
    energy = psi @ H @ psi

    # Relativistic calculations for heavy atoms
    Au = qm.atom('Au', relativistic=True)
    psi_rel = Au.dirac_determinant()

See qm module docstring for more examples.
"""

from . import views
from . import symbols
from . import qm
from . import data

# Re-export commonly used classes from qm for convenience
from .qm import (
    # Coordinate system
    CoordinateType,
    Coordinate3D,
    coord3d,
    spherical_coord,
    cartesian_coord,
    # Spin-orbitals
    SpinOrbital,
    spin_orbital,
    basis_orbital,
    basis_orbitals,
    # Slater determinants
    SlaterDeterminant,
    slater,
    # Operators
    Operator,
    hamiltonian,
    one_electron_operator,
    two_electron_operator,
    # Matrix elements
    Overlap,
    MatrixElement,
    # Relativistic
    DiracSpinor,
    DiracDeterminant,
    RelativisticOperator,
    dirac_spinor,
    dirac_spinor_lj,
    basis_dirac,
    dirac_slater,
    dirac_hamiltonian,
    kappa_from_lj,
    # Atoms and electron configurations
    ATOMIC_NUMBERS,
    ELEMENT_SYMBOLS,
    AUFBAU_ORDER,
    ElectronConfiguration,
    Atom,
    atom,
    atoms,
    ground_state,
    config_from_string,
    # Molecules
    Molecule,
    molecule,
    # Hamiltonian builder system
    HamiltonianTerm,
    HamiltonianBuilder,
    MolecularHamiltonian,
    MatrixExpression,
    HamiltonianMatrix,
)

# Re-export commonly used classes from symbols for convenience
from .symbols import (
    # Core expression classes
    Math,
    Expr,
    Var,
    Const,
    Sum,
    Product,
    # Hyperparameter types
    Scalar,
    ExprType,
    BoundsType,
    # Function composition
    SymbolicFunction,
    BoundFunction,
    ComputeGraph,
    # PyTorch compilation
    TorchFunction,
    TorchGradFunction,
    # Special functions - Base class
    SpecialFunction,
    # Gamma and related
    Gamma,
    LogGamma,
    Digamma,
    Beta,
    Factorial,
    DoubleFactorial,
    Binomial,
    # Error functions
    Erf,
    Erfc,
    Erfi,
    # Bessel functions
    BesselJ,
    BesselY,
    BesselI,
    BesselK,
    SphericalBesselJ,
    SphericalBesselY,
    Hankel1,
    Hankel2,
    # Airy functions
    AiryAi,
    AiryBi,
    AiryAiPrime,
    AiryBiPrime,
    # Orthogonal polynomials
    Legendre,
    AssocLegendre,
    Hermite,
    HermiteProb,
    Laguerre,
    AssocLaguerre,
    Chebyshev1,
    Chebyshev2,
    Gegenbauer,
    Jacobi,
    # Spherical harmonics
    SphericalHarmonic,
    RealSphericalHarmonic,
    # Hypergeometric functions
    Hypergeometric2F1,
    Hypergeometric1F1,
    Hypergeometric0F1,
    HypergeometricPFQ,
    # Elliptic integrals
    EllipticK,
    EllipticE,
    EllipticPi,
    # Other special functions
    Zeta,
    PolyLog,
    DiracDelta,
    Heaviside,
    KroneckerDelta,
    LeviCivita,
    # Angular momentum coupling
    ClebschGordan,
    Wigner3j,
    Wigner6j,
    Wigner9j,
    # Differential operators
    DifferentialOperator,
    PartialDerivative,
    Gradient,
    Laplacian,
    # Hydrogen-like wavefunctions
    HydrogenRadial,
    HydrogenOrbital,
    # Basis functions
    SlaterTypeOrbital,
    GaussianTypeOrbital,
    ContractedGTO,
)

__all__ = [
    'views',
    'symbols',
    'qm',
    'data',
    # QM - Coordinate system
    'CoordinateType',
    'Coordinate3D',
    'coord3d',
    'spherical_coord',
    'cartesian_coord',
    # QM - Spin-orbitals
    'SpinOrbital',
    'spin_orbital',
    'basis_orbital',
    'basis_orbitals',
    # QM - Slater determinants
    'SlaterDeterminant',
    'slater',
    # QM - Operators
    'Operator',
    'hamiltonian',
    'one_electron_operator',
    'two_electron_operator',
    # QM - Matrix elements
    'Overlap',
    'MatrixElement',
    # QM - Relativistic
    'DiracSpinor',
    'DiracDeterminant',
    'RelativisticOperator',
    'dirac_spinor',
    'dirac_spinor_lj',
    'basis_dirac',
    'dirac_slater',
    'dirac_hamiltonian',
    'kappa_from_lj',
    # QM - Atoms and electron configurations
    'ATOMIC_NUMBERS',
    'ELEMENT_SYMBOLS',
    'AUFBAU_ORDER',
    'ElectronConfiguration',
    'Atom',
    'atom',
    'atoms',
    'ground_state',
    'config_from_string',
    # QM - Molecules
    'Molecule',
    'molecule',
    # QM - Hamiltonian builder system
    'HamiltonianTerm',
    'HamiltonianBuilder',
    'MolecularHamiltonian',
    'MatrixExpression',
    'HamiltonianMatrix',
    # Core expression classes
    'Math',
    'Expr',
    'Var',
    'Const',
    'Sum',
    'Product',
    # Hyperparameter types
    'Scalar',
    'ExprType',
    'BoundsType',
    # Function composition
    'SymbolicFunction',
    'BoundFunction',
    'ComputeGraph',
    # PyTorch compilation
    'TorchFunction',
    'TorchGradFunction',
    # Special functions - Base class
    'SpecialFunction',
    # Gamma and related
    'Gamma',
    'LogGamma',
    'Digamma',
    'Beta',
    'Factorial',
    'DoubleFactorial',
    'Binomial',
    # Error functions
    'Erf',
    'Erfc',
    'Erfi',
    # Bessel functions
    'BesselJ',
    'BesselY',
    'BesselI',
    'BesselK',
    'SphericalBesselJ',
    'SphericalBesselY',
    'Hankel1',
    'Hankel2',
    # Airy functions
    'AiryAi',
    'AiryBi',
    'AiryAiPrime',
    'AiryBiPrime',
    # Orthogonal polynomials
    'Legendre',
    'AssocLegendre',
    'Hermite',
    'HermiteProb',
    'Laguerre',
    'AssocLaguerre',
    'Chebyshev1',
    'Chebyshev2',
    'Gegenbauer',
    'Jacobi',
    # Spherical harmonics
    'SphericalHarmonic',
    'RealSphericalHarmonic',
    # Hypergeometric functions
    'Hypergeometric2F1',
    'Hypergeometric1F1',
    'Hypergeometric0F1',
    'HypergeometricPFQ',
    # Elliptic integrals
    'EllipticK',
    'EllipticE',
    'EllipticPi',
    # Other special functions
    'Zeta',
    'PolyLog',
    'DiracDelta',
    'Heaviside',
    'KroneckerDelta',
    'LeviCivita',
    # Angular momentum coupling
    'ClebschGordan',
    'Wigner3j',
    'Wigner6j',
    'Wigner9j',
    # Differential operators
    'DifferentialOperator',
    'PartialDerivative',
    'Gradient',
    'Laplacian',
    # Hydrogen-like wavefunctions
    'HydrogenRadial',
    'HydrogenOrbital',
    # Basis functions
    'SlaterTypeOrbital',
    'GaussianTypeOrbital',
    'ContractedGTO',
]
