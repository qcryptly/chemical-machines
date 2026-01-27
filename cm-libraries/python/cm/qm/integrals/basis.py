"""
Gaussian Basis Functions

Implements Gaussian-type orbitals (GTOs) for molecular integral evaluation.

A Gaussian primitive has the form:
    g(r) = x^i y^j z^k exp(-α|r-R|²)

where (i,j,k) are angular momentum indices and α is the exponent.

Contracted Gaussians are linear combinations:
    φ(r) = Σ_p c_p g_p(r)
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class GaussianPrimitive:
    """
    A single Gaussian primitive function.

    g(r) = N * x^i * y^j * z^k * exp(-alpha * |r - center|^2)

    where N is the normalization constant.

    Attributes:
        alpha: Gaussian exponent
        center: (x, y, z) position in Bohr
        angular: (i, j, k) angular momentum indices
    """
    alpha: float
    center: Tuple[float, float, float]
    angular: Tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        """Compute normalization constant."""
        self._norm = self._compute_normalization()

    def _compute_normalization(self) -> float:
        """
        Compute normalization constant N such that <g|g> = 1.

        For a Gaussian primitive:
        N² = (2α/π)^(3/2) * (4α)^(i+j+k) / [(2i-1)!!(2j-1)!!(2k-1)!!]
        """
        i, j, k = self.angular
        L = i + j + k

        # Double factorial: (2n-1)!! = 1*3*5*...*(2n-1)
        def double_factorial(n):
            if n <= 0:
                return 1
            result = 1
            for x in range(n, 0, -2):
                result *= x
            return result

        prefactor = (2 * self.alpha / math.pi) ** 0.75
        angular_factor = (4 * self.alpha) ** (L / 2)
        denom = math.sqrt(double_factorial(2*i - 1) *
                         double_factorial(2*j - 1) *
                         double_factorial(2*k - 1))

        return prefactor * angular_factor / denom if denom > 0 else prefactor * angular_factor

    @property
    def norm(self) -> float:
        """Normalization constant."""
        return self._norm

    @property
    def L(self) -> int:
        """Total angular momentum."""
        return sum(self.angular)

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate the primitive at position(s) r.

        Args:
            r: Position array of shape (3,) or (N, 3)

        Returns:
            Value(s) of the primitive
        """
        r = np.atleast_2d(r)
        dx = r[:, 0] - self.center[0]
        dy = r[:, 1] - self.center[1]
        dz = r[:, 2] - self.center[2]

        r2 = dx**2 + dy**2 + dz**2

        i, j, k = self.angular
        angular_part = (dx ** i) * (dy ** j) * (dz ** k)
        radial_part = np.exp(-self.alpha * r2)

        return self._norm * angular_part * radial_part


@dataclass
class ContractedGaussian:
    """
    A contracted Gaussian function (CGF).

    φ(r) = Σ_p c_p * g_p(r)

    where c_p are contraction coefficients and g_p are primitives.

    All primitives share the same center and angular momentum.

    Attributes:
        primitives: List of (coefficient, exponent) pairs
        center: (x, y, z) position in Bohr
        angular: (i, j, k) angular momentum indices
    """
    primitives: List[Tuple[float, float]]  # [(coeff, exponent), ...]
    center: Tuple[float, float, float]
    angular: Tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        """Build primitive objects and normalize."""
        self._prims = [
            GaussianPrimitive(alpha=exp, center=self.center, angular=self.angular)
            for coeff, exp in self.primitives
        ]
        self._coeffs = np.array([c for c, _ in self.primitives])

        # Renormalize the contracted function
        self._normalize()

    def _normalize(self):
        """Normalize the contracted Gaussian."""
        # Compute <φ|φ> = Σ_pq c_p c_q <g_p|g_q>
        S = 0.0
        for p, (cp, _) in enumerate(self.primitives):
            for q, (cq, _) in enumerate(self.primitives):
                # Overlap between two primitives on same center
                S += cp * cq * self._prim_overlap(p, q)

        if S > 0:
            self._norm_factor = 1.0 / math.sqrt(S)
        else:
            self._norm_factor = 1.0

    def _prim_overlap(self, p: int, q: int) -> float:
        """Overlap integral between primitives p and q (same center)."""
        from .overlap import overlap_1d

        prim_p = self._prims[p]
        prim_q = self._prims[q]

        # For same-center primitives, the overlap simplifies
        alpha = prim_p.alpha
        beta = prim_q.alpha
        gamma = alpha + beta

        # Overlap of two s-type Gaussians
        S_00 = (math.pi / gamma) ** 1.5

        # For higher angular momentum, use recurrence
        i1, j1, k1 = prim_p.angular
        i2, j2, k2 = prim_q.angular

        # Same center means P = A = B
        Sx = overlap_1d(i1, i2, 0.0, alpha, beta)
        Sy = overlap_1d(j1, j2, 0.0, alpha, beta)
        Sz = overlap_1d(k1, k2, 0.0, alpha, beta)

        return prim_p.norm * prim_q.norm * Sx * Sy * Sz

    @property
    def L(self) -> int:
        """Total angular momentum."""
        return sum(self.angular)

    @property
    def n_primitives(self) -> int:
        """Number of primitives in contraction."""
        return len(self.primitives)

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate the contracted Gaussian at position(s) r.

        Args:
            r: Position array of shape (3,) or (N, 3)

        Returns:
            Value(s) of the function
        """
        result = np.zeros(r.shape[0] if r.ndim > 1 else 1)

        for coeff, prim in zip(self._coeffs, self._prims):
            result += coeff * prim.evaluate(r)

        return self._norm_factor * result


@dataclass
class BasisFunction:
    """
    A basis function with quantum numbers and center information.

    Wraps a ContractedGaussian with additional metadata for
    molecular calculations.

    Attributes:
        cgf: The contracted Gaussian function
        atom_index: Index of the atom this function is centered on
        shell_type: 's', 'p', 'd', 'f', etc.
        m: Magnetic quantum number (for labeling)
    """
    cgf: ContractedGaussian
    atom_index: int
    shell_type: str
    m: int = 0

    @property
    def center(self) -> Tuple[float, float, float]:
        return self.cgf.center

    @property
    def angular(self) -> Tuple[int, int, int]:
        return self.cgf.angular

    @property
    def L(self) -> int:
        return self.cgf.L

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        return self.cgf.evaluate(r)


# Standard basis set data
# STO-3G: Minimal basis, 3 Gaussians fit to Slater orbital
STO_3G_DATA = {
    'H': {
        '1s': [
            (0.1543289673, 3.4252509140),
            (0.5353281423, 0.6239137298),
            (0.4446345422, 0.1688554040),
        ]
    },
    'He': {
        '1s': [
            (0.1562849787, 6.3624213940),
            (0.5353281423, 1.1589229990),
            (0.4446345422, 0.3136497915),
        ]
    },
    'C': {
        '1s': [
            (0.1543289673, 71.6168370000),
            (0.5353281423, 13.0450960000),
            (0.4446345422, 3.5305122000),
        ],
        '2s': [
            (-0.0999672292, 2.9412494000),
            (0.3995128261, 0.6834831000),
            (0.7001154689, 0.2222899000),
        ],
        '2p': [
            (0.1559162750, 2.9412494000),
            (0.6076837186, 0.6834831000),
            (0.3919573931, 0.2222899000),
        ]
    },
    'N': {
        '1s': [
            (0.1543289673, 99.1061690000),
            (0.5353281423, 18.0523120000),
            (0.4446345422, 4.8856602000),
        ],
        '2s': [
            (-0.0999672292, 3.7804559000),
            (0.3995128261, 0.8784966400),
            (0.7001154689, 0.2857143900),
        ],
        '2p': [
            (0.1559162750, 3.7804559000),
            (0.6076837186, 0.8784966400),
            (0.3919573931, 0.2857143900),
        ]
    },
    'O': {
        '1s': [
            (0.1543289673, 130.7093200000),
            (0.5353281423, 23.8088610000),
            (0.4446345422, 6.4436083000),
        ],
        '2s': [
            (-0.0999672292, 5.0331513000),
            (0.3995128261, 1.1695961000),
            (0.7001154689, 0.3803890000),
        ],
        '2p': [
            (0.1559162750, 5.0331513000),
            (0.6076837186, 1.1695961000),
            (0.3919573931, 0.3803890000),
        ]
    },
}


class BasisSet:
    """
    A molecular basis set.

    Builds basis functions for all atoms in a molecule using
    standard basis set parameters.

    Attributes:
        name: Basis set name (e.g., 'STO-3G')
        functions: List of BasisFunction objects
    """

    def __init__(self, name: str = 'STO-3G'):
        self.name = name
        self.functions: List[BasisFunction] = []
        self._data = self._load_basis_data(name)

    def _load_basis_data(self, name: str) -> Dict:
        """Load basis set parameters."""
        if name.upper() == 'STO-3G':
            return STO_3G_DATA
        else:
            raise ValueError(f"Unknown basis set: {name}")

    def build_for_molecule(self, atoms: List[Tuple[str, Tuple[float, float, float]]]):
        """
        Build basis functions for a molecule.

        Args:
            atoms: List of (element_symbol, (x, y, z)) tuples
                   Positions in Angstroms (will be converted to Bohr)
        """
        ANGSTROM_TO_BOHR = 1.8897259886

        self.functions = []

        for atom_idx, (element, pos) in enumerate(atoms):
            # Convert to Bohr
            center = (
                pos[0] * ANGSTROM_TO_BOHR,
                pos[1] * ANGSTROM_TO_BOHR,
                pos[2] * ANGSTROM_TO_BOHR
            )

            if element not in self._data:
                raise ValueError(f"Element {element} not in basis set {self.name}")

            atom_basis = self._data[element]

            for shell_name, primitives in atom_basis.items():
                # Parse shell type (1s, 2s, 2p, etc.)
                shell_type = shell_name[-1]  # 's', 'p', 'd', etc.

                if shell_type == 's':
                    # Single s function
                    cgf = ContractedGaussian(
                        primitives=primitives,
                        center=center,
                        angular=(0, 0, 0)
                    )
                    self.functions.append(BasisFunction(
                        cgf=cgf,
                        atom_index=atom_idx,
                        shell_type='s',
                        m=0
                    ))

                elif shell_type == 'p':
                    # Three p functions: px, py, pz
                    for m, angular in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
                        cgf = ContractedGaussian(
                            primitives=primitives,
                            center=center,
                            angular=angular
                        )
                        self.functions.append(BasisFunction(
                            cgf=cgf,
                            atom_index=atom_idx,
                            shell_type='p',
                            m=m
                        ))

                elif shell_type == 'd':
                    # Six d functions (Cartesian)
                    d_angular = [
                        (2, 0, 0), (0, 2, 0), (0, 0, 2),
                        (1, 1, 0), (1, 0, 1), (0, 1, 1)
                    ]
                    for m, angular in enumerate(d_angular):
                        cgf = ContractedGaussian(
                            primitives=primitives,
                            center=center,
                            angular=angular
                        )
                        self.functions.append(BasisFunction(
                            cgf=cgf,
                            atom_index=atom_idx,
                            shell_type='d',
                            m=m
                        ))

    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        return len(self.functions)

    def __len__(self) -> int:
        return self.n_basis

    def __getitem__(self, i: int) -> BasisFunction:
        return self.functions[i]
