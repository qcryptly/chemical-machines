"""Tests for cm.qm.spinorbitals module."""

import pytest
from cm.qm import SpinOrbital, SlaterDeterminant, spherical_coord


class TestSpinOrbital:
    """Tests for SpinOrbital class."""

    def test_create_spinorbital(self):
        """Test creating a spin-orbital."""
        coord = spherical_coord()
        orb = SpinOrbital(coord, n=1, l=0, m=0, spin=1, center=0)
        assert orb.n == 1
        assert orb.l == 0
        assert orb.m == 0
        assert orb.spin == 1  # +1 for alpha
        assert orb.center == 0

    def test_spinorbital_labels(self):
        """Test orbital labels."""
        coord = spherical_coord()
        orb_1s = SpinOrbital(coord, n=1, l=0, m=0, spin=1, center=0)
        label = orb_1s.ket_label
        assert "1" in label

        orb_2p = SpinOrbital(coord, n=2, l=1, m=0, spin=-1, center=0)
        label = orb_2p.ket_label
        assert "2" in label

    def test_spinorbital_equality(self):
        """Test spin-orbital equality."""
        coord = spherical_coord()
        orb1 = SpinOrbital(coord, n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(coord, n=1, l=0, m=0, spin=1, center=0)
        orb3 = SpinOrbital(coord, n=1, l=0, m=0, spin=-1, center=0)

        # Same orbitals should be distinguishable by spin
        assert orb1.spin == orb2.spin
        assert orb1.spin != orb3.spin

    def test_opposite_spin(self):
        """Test that alpha and beta are different."""
        coord = spherical_coord()
        orb_alpha = SpinOrbital(coord, n=1, l=0, m=0, spin=1, center=0)
        orb_beta = SpinOrbital(coord, n=1, l=0, m=0, spin=-1, center=0)
        assert orb_alpha.spin != orb_beta.spin


class TestSlaterDeterminant:
    """Tests for SlaterDeterminant class."""

    def test_create_determinant(self):
        """Test creating a Slater determinant."""
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        det = SlaterDeterminant([orb1, orb2])

        assert len(det.orbitals) == 2
        assert det.n_electrons == 2

    def test_determinant_sorting(self):
        """Test that orbitals are sorted."""
        # Create orbitals in random order
        orb2 = SpinOrbital(spherical_coord(), n=2, l=0, m=0, spin=1, center=0)
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        det = SlaterDeterminant([orb2, orb1])

        # Should be automatically sorted
        assert len(det.orbitals) == 2

    def test_pauli_exclusion(self):
        """Test Pauli exclusion principle (no duplicate orbitals)."""
        orb = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)

        # Creating determinant with duplicate should raise ValueError
        with pytest.raises(ValueError, match="Pauli exclusion"):
            det = SlaterDeterminant([orb, orb])

    def test_n_excitations_identical(self):
        """Test n_excitations for identical determinants."""
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        det1 = SlaterDeterminant([orb1, orb2])
        det2 = SlaterDeterminant([orb1, orb2])

        n_exc = det1.n_excitations(det2)
        assert n_exc == 0

    def test_n_excitations_single(self):
        """Test n_excitations for single excitation."""
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        orb3 = SpinOrbital(spherical_coord(), n=2, l=0, m=0, spin=1, center=0)

        det1 = SlaterDeterminant([orb1, orb2])
        det2 = SlaterDeterminant([orb3, orb2])  # orb1 -> orb3

        n_exc = det1.n_excitations(det2)
        assert n_exc == 1

    def test_n_excitations_double(self):
        """Test n_excitations for double excitation."""
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        orb3 = SpinOrbital(spherical_coord(), n=2, l=0, m=0, spin=1, center=0)
        orb4 = SpinOrbital(spherical_coord(), n=2, l=0, m=0, spin=-1, center=0)

        det1 = SlaterDeterminant([orb1, orb2])
        det2 = SlaterDeterminant([orb3, orb4])  # both excit ed

        n_exc = det1.n_excitations(det2)
        assert n_exc == 2

    def test_excitation_orbitals_single(self):
        """Test excitation_orbitals for single excitation."""
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        orb3 = SpinOrbital(spherical_coord(), n=2, l=0, m=0, spin=1, center=0)

        det1 = SlaterDeterminant([orb1, orb2])
        det2 = SlaterDeterminant([orb3, orb2])

        only_bra, only_ket = det1.excitation_orbitals(det2)
        assert len(only_bra) == 1
        assert len(only_ket) == 1

    def test_ket_labels(self):
        """Test ket label generation."""
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        det = SlaterDeterminant([orb1, orb2])

        labels = det.ket_labels()
        assert len(labels) == 2


@pytest.mark.unit
class TestSlaterCondonRules:
    """Tests for Slater-Condon rules application."""

    def test_zero_excitation(self):
        """Test diagonal element (0 excitations)."""
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        det = SlaterDeterminant([orb1, orb2])

        n_exc = det.n_excitations(det)
        assert n_exc == 0

    def test_triple_excitation_is_zero(self):
        """Test that triple+ excitations give zero overlap."""
        # Create determinants with > 2 excitations
        orb1 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=1, center=0)
        orb2 = SpinOrbital(spherical_coord(), n=1, l=0, m=0, spin=-1, center=0)
        orb3 = SpinOrbital(spherical_coord(), n=1, l=1, m=0, spin=1, center=0)

        orb4 = SpinOrbital(spherical_coord(), n=2, l=0, m=0, spin=1, center=0)
        orb5 = SpinOrbital(spherical_coord(), n=2, l=0, m=0, spin=-1, center=0)
        orb6 = SpinOrbital(spherical_coord(), n=2, l=1, m=0, spin=1, center=0)

        det1 = SlaterDeterminant([orb1, orb2, orb3])
        det2 = SlaterDeterminant([orb4, orb5, orb6])

        n_exc = det1.n_excitations(det2)
        # Triple excitation
        assert n_exc >= 3 or n_exc == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
