"""
Chemical Machines Quantum Mechanics Package

Provides tools for working with atoms, Slater determinants, spin-orbitals,
and quantum mechanical matrix elements using spherical harmonic basis functions.

Features:
    - Atom class with automatic electron configuration (aufbau principle)
    - Non-relativistic (SpinOrbital) and relativistic (DiracSpinor) orbitals
    - Slater and Dirac determinants with Slater-Condon rules
    - Hamiltonian builder system for molecular calculations

Example:
    from cm import qm

    # Create atoms with automatic ground state configuration
    C = qm.atom('C')
    print(C.configuration.label)  # "1s² 2s² 2p²"

    # Create Slater determinant
    psi = C.slater_determinant()
    psi.render()
"""

# Atomic data
from .data import (
    ATOMIC_NUMBERS,
    ELEMENT_SYMBOLS,
    AUFBAU_ORDER,
)

# Coordinate system
from .coordinates import (
    CoordinateType,
    Coordinate3D,
    coord3d,
    spherical_coord,
    cartesian_coord,
)

# Spin-orbitals and determinants
from .spinorbitals import (
    SpinOrbital,
    SlaterDeterminant,
    Operator,
    Overlap,
    MatrixElement,
    spin_orbital,
    basis_orbital,
    basis_orbitals,
    slater,
    hamiltonian,
    one_electron_operator,
    two_electron_operator,
)

# Relativistic
from .relativistic import (
    DiracSpinor,
    DiracDeterminant,
    RelativisticOperator,
    dirac_spinor,
    dirac_spinor_lj,
    basis_dirac,
    dirac_slater,
    dirac_hamiltonian,
    kappa_from_lj,
)

# Atoms and electron configurations
from .atoms import (
    ElectronConfiguration,
    Atom,
    atom,
    atoms,
    ground_state,
    config_from_string,
)

# Molecules
from .molecules import (
    Molecule,
    molecule,
)

# Hamiltonian builder system
from .hamiltonian import (
    HamiltonianTerm,
    HamiltonianBuilder,
    MolecularHamiltonian,
    MatrixExpression,
    HamiltonianMatrix,
)

# Orthogonality utilities
from .orthogonality import (
    spherical_harmonic_orthogonality,
)

__all__ = [
    # Coordinate system
    'CoordinateType',
    'Coordinate3D',
    'coord3d',
    'spherical_coord',
    'cartesian_coord',
    # Spin-orbitals
    'SpinOrbital',
    'spin_orbital',
    'basis_orbital',
    'basis_orbitals',
    # Slater determinants
    'SlaterDeterminant',
    'slater',
    # Operators
    'Operator',
    'hamiltonian',
    'one_electron_operator',
    'two_electron_operator',
    # Matrix elements
    'Overlap',
    'MatrixElement',
    # Relativistic
    'DiracSpinor',
    'DiracDeterminant',
    'RelativisticOperator',
    'dirac_spinor',
    'dirac_spinor_lj',
    'basis_dirac',
    'dirac_slater',
    'dirac_hamiltonian',
    'kappa_from_lj',
    # Atoms and electron configurations
    'ATOMIC_NUMBERS',
    'ELEMENT_SYMBOLS',
    'AUFBAU_ORDER',
    'ElectronConfiguration',
    'Atom',
    'atom',
    'atoms',
    'ground_state',
    'config_from_string',
    # Molecules
    'Molecule',
    'molecule',
    # Hamiltonian builder system
    'HamiltonianTerm',
    'HamiltonianBuilder',
    'MolecularHamiltonian',
    'MatrixExpression',
    'HamiltonianMatrix',
    # Orthogonality utilities
    'spherical_harmonic_orthogonality',
]
