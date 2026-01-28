"""
Quantum Chemistry Methods Module

Provides implementations for electronic structure methods:
- Hartree-Fock (RHF, UHF)
- Kohn-Sham DFT
- MP2 Perturbation Theory
- CCSD(T) Coupled Cluster
"""

from .hf import (
    HFResult,
    HartreeFockSolver,
    hartree_fock,
)
from .uhf import (
    UHFResult,
    UnrestrictedHartreeFockSolver,
    uhf,
)
from .dft import (
    DFTResult,
    KohnShamSolver,
    kohn_sham,
)
from .mp2 import (
    MP2Result,
    mp2,
    compute_mp2_energy,
    compute_scs_mp2_energy,
)
from .ccsd import (
    CCSDResult,
    ccsd,
    transform_integrals_to_mo,
    get_spinorbital_integrals,
    build_fock_mo,
)

__all__ = [
    # Hartree-Fock
    'HFResult',
    'HartreeFockSolver',
    'hartree_fock',
    # UHF
    'UHFResult',
    'UnrestrictedHartreeFockSolver',
    'uhf',
    # DFT
    'DFTResult',
    'KohnShamSolver',
    'kohn_sham',
    # MP2
    'MP2Result',
    'mp2',
    'compute_mp2_energy',
    'compute_scs_mp2_energy',
    # Coupled Cluster
    'CCSDResult',
    'ccsd',
    'transform_integrals_to_mo',
    'get_spinorbital_integrals',
    'build_fock_mo',
]
