"""
Chemical Machines Data Package

Provides access to benchmark molecular databases for comparing
ab initio calculations with experimental and reference data.

Supported databases:
- NIST CCCBDB: Experimental geometries, energies, vibrational frequencies
- PubChem: Molecular properties, 3D structures, computed properties
- QM9: Pre-computed DFT results (134k molecules, B3LYP/6-31G(2df,p))

Usage:
    from cm.data import benchmark

    # Search for molecule
    results = benchmark.search("water")
    results = benchmark.search(formula="H2O")
    results = benchmark.search(cas="7732-18-5")

    # Get specific data
    mol = benchmark.get("7732-18-5")  # By CAS number
    mol.geometry          # XYZ coordinates
    mol.properties        # Experimental/computed properties

    # Compare with computed values
    from cm.qm.integrals import hartree_fock
    hf = hartree_fock([('O', (0, 0, 0)), ('H', (0.96, 0, 0)), ('H', (-0.24, 0.93, 0))])
    comparison = benchmark.compare(hf, "7732-18-5")
    comparison.render()   # Display comparison table
"""

from .benchmark import (
    # Data classes
    BenchmarkMolecule,
    BenchmarkProperty,
    ComparisonResult,
    PropertyComparison,
    MoleculeStatus,
    # Functions
    search,
    get,
    compare,
    sync,
    sync_status,
    wait_for_sync,
    stats,
    status,
    # Exceptions
    BenchmarkError,
    ServiceUnavailableError,
    APIError,
    JobError,
    IndexingInProgressError,
    MoleculeNotFoundError,
    NoIndexError,
)

__all__ = [
    # Data classes
    'BenchmarkMolecule',
    'BenchmarkProperty',
    'ComparisonResult',
    'PropertyComparison',
    'MoleculeStatus',
    # Functions
    'search',
    'get',
    'compare',
    'sync',
    'sync_status',
    'wait_for_sync',
    'stats',
    'status',
    # Exceptions
    'BenchmarkError',
    'ServiceUnavailableError',
    'APIError',
    'JobError',
    'IndexingInProgressError',
    'MoleculeNotFoundError',
    'NoIndexError',
]
