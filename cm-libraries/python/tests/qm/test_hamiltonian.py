"""Tests for cm.qm.hamiltonian module."""

import pytest
from cm.qm import HamiltonianBuilder, HamiltonianTerm, MolecularHamiltonian
from cm.symbols import Const, Var


class TestHamiltonianTerm:
    """Tests for HamiltonianTerm class."""

    def test_create_term(self):
        """Test creating a Hamiltonian term."""
        term = HamiltonianTerm(
            name="kinetic",
            symbol="T",
            n_body=1,
            coefficient=Const(-0.5),
            expression=None,
            description="Kinetic energy"
        )
        assert term.name == "kinetic"
        assert term.symbol == "T"
        assert term.n_body == 1

    def test_term_to_latex(self):
        """Test LaTeX representation."""
        term = HamiltonianTerm(
            name="kinetic",
            symbol="T",
            n_body=1,
            coefficient=Const(1),
            expression=None
        )
        latex = term.to_latex()
        assert "T" in latex


class TestHamiltonianBuilder:
    """Tests for HamiltonianBuilder class."""

    def test_empty_builder(self):
        """Test creating an empty builder."""
        builder = HamiltonianBuilder()
        assert len(builder.terms) == 0

    def test_with_kinetic(self):
        """Test adding kinetic energy term."""
        builder = HamiltonianBuilder().with_kinetic()
        assert "kinetic" in builder.terms
        assert len(builder.terms) == 1

    def test_with_nuclear_attraction(self):
        """Test adding nuclear attraction term."""
        builder = HamiltonianBuilder().with_nuclear_attraction()
        assert "nuclear_attraction" in builder.terms

    def test_with_coulomb(self):
        """Test adding Coulomb repulsion term."""
        builder = HamiltonianBuilder().with_coulomb()
        assert "coulomb" in builder.terms
        # Verify it's a 2-body term
        H = builder.build()
        coulomb_term = [t for t in H.terms if t.name == "coulomb"][0]
        assert coulomb_term.n_body == 2

    def test_electronic_preset(self):
        """Test electronic Hamiltonian preset."""
        builder = HamiltonianBuilder.electronic()
        assert "kinetic" in builder.terms
        assert "nuclear_attraction" in builder.terms
        assert "coulomb" in builder.terms
        assert len(builder.terms) == 3

    def test_spin_orbit_preset(self):
        """Test spin-orbit Hamiltonian preset."""
        builder = HamiltonianBuilder.spin_orbit()
        assert "kinetic" in builder.terms
        assert "spin_orbit" in builder.terms

    def test_relativistic_preset(self):
        """Test relativistic Hamiltonian preset."""
        builder = HamiltonianBuilder.relativistic()
        assert "kinetic" in builder.terms
        assert "breit" in builder.terms or any("breit" in t or "gaunt" in t for t in builder.terms)

    def test_with_external_electric_field(self):
        """Test adding external electric field."""
        builder = HamiltonianBuilder().with_external_field(
            field_type='electric',
            strength=0.1,
            direction=(0, 0, 1)
        )
        assert "electric_field" in builder.terms

    def test_with_external_magnetic_field(self):
        """Test adding external magnetic field."""
        builder = HamiltonianBuilder().with_external_field(
            field_type='magnetic',
            strength=0.05,
            direction=(0, 0, 1)
        )
        assert "magnetic_field" in builder.terms

    def test_with_custom_term(self):
        """Test adding custom term."""
        x = Var("x")
        builder = HamiltonianBuilder().with_custom(
            name="test_term",
            symbol="T_{test}",
            expression=x**2,
            n_body=1,
            coefficient=2.0
        )
        assert "test_term" in builder.terms

    def test_scale_term(self):
        """Test scaling a term."""
        builder = HamiltonianBuilder().with_kinetic().scale("kinetic", 2.0)
        H = builder.build()
        kinetic_term = [t for t in H.terms if t.name == "kinetic"][0]
        # Coefficient should be scaled

    def test_remove_term(self):
        """Test removing a term."""
        builder = HamiltonianBuilder().with_kinetic().with_coulomb()
        assert "kinetic" in builder.terms
        builder.remove("kinetic")
        assert "kinetic" not in builder.terms
        assert "coulomb" in builder.terms

    def test_to_latex(self):
        """Test LaTeX representation of builder."""
        builder = HamiltonianBuilder().with_kinetic().with_coulomb()
        latex = builder.to_latex()
        assert latex is not None
        assert len(latex) > 0

    def test_build(self):
        """Test building the Hamiltonian."""
        builder = HamiltonianBuilder.electronic()
        H = builder.build()
        assert isinstance(H, MolecularHamiltonian)
        assert len(H.terms) == 3


class TestMolecularHamiltonian:
    """Tests for MolecularHamiltonian class."""

    def test_create_hamiltonian(self):
        """Test creating a molecular Hamiltonian."""
        H = HamiltonianBuilder.electronic().build()
        assert isinstance(H, MolecularHamiltonian)

    def test_hamiltonian_properties(self):
        """Test Hamiltonian properties."""
        H = HamiltonianBuilder.electronic().build()
        assert len(H.term_names) == 3
        assert H.n_body_max == 2
        assert H.has_term("kinetic")
        assert H.has_term("nuclear_attraction")
        assert H.has_term("coulomb")
        assert not H.has_term("nonexistent")

    def test_is_relativistic(self):
        """Test relativistic flag."""
        H_nonrel = HamiltonianBuilder.electronic().build()
        assert not H_nonrel.is_relativistic

        H_rel = HamiltonianBuilder.relativistic().build()
        assert H_rel.is_relativistic

    def test_to_latex(self):
        """Test LaTeX representation."""
        H = HamiltonianBuilder.electronic().build()
        latex = H.to_latex()
        assert latex is not None
        assert "T" in latex

    def test_repr(self):
        """Test string representation."""
        H = HamiltonianBuilder.electronic().build()
        repr_str = repr(H)
        assert "MolecularHamiltonian" in repr_str
        assert "kinetic" in repr_str


class TestMatrixElement:
    """Tests for matrix element computation."""

    def test_matrix_element_with_determinant(self, sample_molecule_h2_numeric):
        """Test computing matrix element with Slater determinant."""
        from cm.qm import SlaterDeterminant, SpinOrbital, spherical_coord, spherical_coord

        # Create simple Slater determinant for H2
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=1)
        det = SlaterDeterminant([orb1, orb2])

        H = HamiltonianBuilder.electronic().build()
        E = H.element(det, det, sample_molecule_h2_numeric)

        assert E is not None
        assert hasattr(E, 'analytical')
        assert hasattr(E, 'numerical')

    def test_diagonal_element(self, sample_molecule_h2_numeric):
        """Test diagonal matrix element."""
        from cm.qm import SlaterDeterminant, SpinOrbital, spherical_coord

        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=1)
        det = SlaterDeterminant([orb1, orb2])

        H = HamiltonianBuilder.electronic().build()
        E = H.diagonal(det, sample_molecule_h2_numeric)

        assert E is not None
        assert E.is_diagonal


class TestHamiltonianMatrix:
    """Tests for HamiltonianMatrix class."""

    def test_matrix_creation(self, sample_molecule_h2_numeric):
        """Test creating a Hamiltonian matrix."""
        from cm.qm import SlaterDeterminant, SpinOrbital, spherical_coord

        # Create basis of 2 determinants
        orb1_a = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb1_b = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        orb2_a = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=1)
        orb2_b = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=1)

        det1 = SlaterDeterminant([orb1_a, orb1_b])
        det2 = SlaterDeterminant([orb2_a, orb2_b])
        basis = [det1, det2]

        H = HamiltonianBuilder.electronic().build()
        H_mat = H.matrix(basis, sample_molecule_h2_numeric)

        assert H_mat.shape == (2, 2)
        assert H_mat.n_basis == 2

    def test_matrix_element_access(self, sample_molecule_h2_numeric):
        """Test accessing matrix elements."""
        from cm.qm import SlaterDeterminant, SpinOrbital, spherical_coord

        orb1_a = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb1_b = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)

        det = SlaterDeterminant([orb1_a, orb1_b])
        basis = [det]

        H = HamiltonianBuilder.electronic().build()
        H_mat = H.matrix(basis, sample_molecule_h2_numeric)

        # Access element
        elem = H_mat[0, 0]
        assert elem is not None

    def test_matrix_index_error(self, sample_molecule_h2_numeric):
        """Test that out-of-bounds access raises error."""
        from cm.qm import SlaterDeterminant, SpinOrbital, spherical_coord

        orb1_a = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        det = SlaterDeterminant([orb1_a])
        basis = [det]

        H = HamiltonianBuilder.electronic().build()
        H_mat = H.matrix(basis, sample_molecule_h2_numeric)

        with pytest.raises(IndexError):
            _ = H_mat[5, 5]


@pytest.mark.unit
class TestHamiltonianEvaluation:
    """Tests for Hamiltonian evaluation."""

    def test_analytical_evaluation(self, sample_molecule_h2_numeric):
        """Test analytical (symbolic) evaluation."""
        from cm.qm import SlaterDeterminant, SpinOrbital, spherical_coord

        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=1)
        det = SlaterDeterminant([orb1, orb2])

        H = HamiltonianBuilder.electronic().build()
        E = H.element(det, det, sample_molecule_h2_numeric)

        # Get symbolic expression
        expr = E.analytical()
        assert expr is not None

    def test_numerical_evaluation(self, sample_molecule_h2_numeric):
        """Test numerical evaluation."""
        from cm.qm import SlaterDeterminant, SpinOrbital, spherical_coord

        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=1)
        det = SlaterDeterminant([orb1, orb2])

        H = HamiltonianBuilder.electronic().build()
        E = H.element(det, det, sample_molecule_h2_numeric)

        # Evaluate numerically with r=0.74 Angstroms (equilibrium H2 bond length)
        E_val = E.numerical(r=0.74)
        assert isinstance(E_val, (int, float))
        # H2 ground state energy should be negative (bound state)
        assert E_val < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
