"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add cm-libraries to path
cm_lib_path = Path(__file__).parent.parent
sys.path.insert(0, str(cm_lib_path))


@pytest.fixture
def sample_molecule_h2():
    """Simple H2 molecule for testing."""
    from cm.qm import Molecule, atom
    from cm.symbols import Var

    r = Var("r")
    # Molecule expects list of (Atom, x, y, z) tuples
    mol = Molecule([
        (atom("H"), 0, 0, 0),
        (atom("H"), r, 0, 0)
    ])
    return mol


@pytest.fixture
def sample_molecule_water():
    """Water molecule for testing."""
    from cm.qm import Molecule, atom
    from cm.symbols import Var, Sin, Cos

    r = Var("r")  # O-H bond length
    theta = Var("theta")  # H-O-H angle

    # O at origin
    # H1 at (r, 0, 0)
    # H2 at (r*cos(theta), r*sin(theta), 0)
    mol = Molecule([
        (atom("O"), 0, 0, 0),
        (atom("H"), r, 0, 0),
        (atom("H"), r * Cos(theta), r * Sin(theta), 0)
    ])
    return mol


@pytest.fixture
def sample_molecule_h2_numeric():
    """Simple H2 molecule with numeric positions for Hamiltonian testing."""
    from cm.qm import Molecule, atom

    # H2 with equilibrium bond length (numeric, not symbolic)
    mol = Molecule([
        (atom("H"), 0, 0, 0),
        (atom("H"), 0.74, 0, 0)
    ])
    return mol


@pytest.fixture
def simple_hamiltonian():
    """Simple electronic Hamiltonian for testing."""
    from cm.qm import HamiltonianBuilder
    return HamiltonianBuilder.electronic().build()
