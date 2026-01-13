"""Tests for cm.qm.molecules and cm.qm.atoms modules."""

import pytest
from cm.qm import Atom, Molecule, atom
from cm.symbols import Var, Sin, Cos


class TestAtom:
    """Tests for Atom class."""

    def test_create_atom(self):
        """Test creating an atom."""
        atom = Atom("H", (0, 0, 0))
        assert atom.symbol == "H"
        assert atom.Z == 1
        assert len(atom.position) == 3

    def test_atom_elements(self):
        """Test various elements."""
        elements = [
            ("H", 1),
            ("He", 2),
            ("C", 6),
            ("N", 7),
            ("O", 8),
            ("F", 9),
        ]

        for symbol, expected_Z in elements:
            atom = Atom(symbol, (0, 0, 0))
            assert atom.Z == expected_Z

    def test_atom_position(self):
        """Test atom position."""
        pos = (1.0, 2.0, 3.0)
        atom = Atom("C", pos)
        assert atom.position == pos

    def test_atom_symbolic_position(self):
        """Test atom with symbolic coordinates."""
        x = Var("x")
        y = Var("y")
        atom = Atom("O", (x, y, 0))
        assert atom.position[0] == x
        assert atom.position[1] == y


class TestMolecule:
    """Tests for Molecule class."""

    def test_create_molecule(self):
        """Test creating a molecule."""
        # Molecule expects (Atom, x, y, z) tuples
        mol = Molecule([
            (atom("H"), 0, 0, 0),
            (atom("H"), 0.74, 0, 0)
        ])

        assert mol.n_atoms == 2
        assert mol.n_electrons == 2

    def test_h2_molecule(self):
        """Test H2 molecule creation."""
        r = Var("r")
        mol = Molecule([
            (atom("H"), 0, 0, 0),
            (atom("H"), r, 0, 0)
        ])

        assert mol.n_atoms == 2
        assert mol.n_electrons == 2
        assert mol.total_nuclear_charge == 2  # 2 protons

    def test_water_molecule(self):
        """Test H2O molecule."""
        mol = Molecule([
            (atom("O"), 0, 0, 0),
            (atom("H"), 0.96, 0, 0),
            (atom("H"), 0.96 * 0.866, 0.96 * 0.5, 0)  # 104.5 degree angle
        ])

        assert mol.n_atoms == 3
        assert mol.n_electrons == 10  # 8 + 1 + 1
        assert mol.total_nuclear_charge == 10  # 8 + 1 + 1

    def test_molecule_positions(self):
        """Test accessing atomic positions."""
        mol = Molecule([
            (atom("C"), 0, 0, 0),
            (atom("O"), 1.2, 0, 0)
        ])

        positions = mol.positions
        assert len(positions) == 2
        assert positions[0] == (0, 0, 0)
        assert positions[1] == (1.2, 0, 0)

    def test_molecule_geometry_variables(self):
        """Test extracting geometry variables."""
        r = Var("r")
        theta = Var("theta")

        mol = Molecule([
            (atom("O"), 0, 0, 0),
            (atom("H"), r, 0, 0),
            (atom("H"), r * Cos(theta), r * Sin(theta), 0)
        ])

        # Should identify r and theta as geometry variables
        geom_vars = mol.geometry_variables
        assert len(geom_vars) >= 2

    def test_slater_determinant(self):
        """Test generating Slater determinant."""
        mol = Molecule([
            (atom("H"), 0, 0, 0),
            (atom("H"), 0.74, 0, 0)
        ])

        det = mol.slater_determinant()
        assert det is not None
        assert det.n_electrons == 2


class TestMolecularGeometry:
    """Tests for molecular geometry."""

    def test_bond_length_symbolic(self):
        """Test molecule with symbolic bond length."""
        r = Var("r")
        mol = Molecule([
            (atom("H"), 0, 0, 0),
            (atom("H"), r, 0, 0)
        ])

        # Verify symbolic position is preserved
        assert mol.positions[1][0] == r

    def test_bond_angle_symbolic(self):
        """Test molecule with symbolic bond angle."""
        r = Var("r")
        theta = Var("theta")

        mol = Molecule([
            (atom("O"), 0, 0, 0),
            (atom("H"), r, 0, 0),
            (atom("H"), r * Cos(theta), r * Sin(theta), 0)
        ])

        # Second hydrogen position should involve theta
        pos_h2 = mol.positions[2]
        assert any(hasattr(coord, 'to_latex') for coord in pos_h2)


@pytest.mark.unit
class TestMolecularProperties:
    """Tests for molecular property calculations."""

    def test_electron_count_neutral(self):
        """Test electron count for neutral molecules."""
        # H2
        h2 = Molecule([(atom("H"), 0, 0, 0), (atom("H"), 0.74, 0, 0)])
        assert h2.n_electrons == 2

        # H2O
        water = Molecule([
            (atom("O"), 0, 0, 0),
            (atom("H"), 1, 0, 0),
            (atom("H"), 0, 1, 0)
        ])
        assert water.n_electrons == 10

        # CH4
        methane = Molecule([
            (atom("C"), 0, 0, 0),
            (atom("H"), 1, 0, 0),
            (atom("H"), -1, 0, 0),
            (atom("H"), 0, 1, 0),
            (atom("H"), 0, -1, 0),
        ])
        assert methane.n_electrons == 10  # 6 + 4*1

    def test_molecular_charge(self):
        """Test total charge calculation."""
        mol = Molecule([
            (atom("H"), 0, 0, 0),
            (atom("H"), 0.74, 0, 0)
        ])

        # Total nuclear charge
        assert mol.total_nuclear_charge == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
