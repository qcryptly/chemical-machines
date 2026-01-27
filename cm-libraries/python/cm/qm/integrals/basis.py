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


# cc-pVTZ: Correlation-consistent polarized valence triple-zeta
# Data from Basis Set Exchange (Dunning, J. Chem. Phys. 90, 1007 (1989))
# Format: shell_type -> list of contracted functions
# Each contracted function: [(coefficient, exponent), ...]
CC_PVTZ_DATA = {
    'H': {
        # 3 s-type contracted functions
        's1': [
            (0.0060680, 33.8700000),
            (0.0453160, 5.0950000),
            (0.2028460, 1.1590000),
        ],
        's2': [
            (1.0000000, 0.3258000),
        ],
        's3': [
            (1.0000000, 0.1027000),
        ],
        # 2 p-type functions (polarization)
        'p1': [
            (1.0000000, 1.4070000),
        ],
        'p2': [
            (1.0000000, 0.3880000),
        ],
        # 1 d-type function (polarization)
        'd1': [
            (1.0000000, 1.0570000),
        ],
    },
    'He': {
        's1': [
            (0.0052630, 98.1243000),
            (0.0402490, 14.7689000),
            (0.1800820, 3.3188300),
        ],
        's2': [
            (1.0000000, 0.8740470),
        ],
        's3': [
            (1.0000000, 0.2445640),
        ],
        'p1': [
            (1.0000000, 1.2750000),
        ],
        'p2': [
            (1.0000000, 0.3600000),
        ],
        'd1': [
            (1.0000000, 0.9500000),
        ],
    },
    'C': {
        # Core 1s (contracted)
        's1': [
            (0.0005310, 8236.0000000),
            (0.0041080, 1235.0000000),
            (0.0212870, 280.8000000),
            (0.0818680, 79.2700000),
            (0.2348170, 25.5900000),
            (0.4344010, 8.9970000),
            (0.3461290, 3.3190000),
            (0.0393780, 0.3643000),
        ],
        # Valence s functions
        's2': [
            (-0.0089830, 8.9970000),
            (-0.0548440, 3.3190000),
            (-0.1452520, 1.0970000),
            (0.5612140, 0.3643000),
            (0.5389460, 0.1285000),
        ],
        's3': [
            (1.0000000, 0.0440200),
        ],
        # Core p (contracted)
        'p1': [
            (0.0139280, 18.7100000),
            (0.0868630, 4.1330000),
            (0.2902330, 1.2000000),
            (0.5008060, 0.3827000),
            (0.3434060, 0.1209000),
        ],
        # Valence p functions
        'p2': [
            (1.0000000, 0.0356900),
        ],
        # Polarization d functions
        'd1': [
            (1.0000000, 1.0970000),
        ],
        'd2': [
            (1.0000000, 0.3180000),
        ],
        # Polarization f function
        'f1': [
            (1.0000000, 0.7610000),
        ],
    },
    'N': {
        's1': [
            (0.0004590, 11420.0000000),
            (0.0035550, 1712.0000000),
            (0.0182910, 389.3000000),
            (0.0716850, 110.0000000),
            (0.2120600, 35.5700000),
            (0.4134010, 12.5400000),
            (0.3702260, 4.6440000),
            (0.0467860, 0.5118000),
        ],
        's2': [
            (-0.0083890, 12.5400000),
            (-0.0535740, 4.6440000),
            (-0.1386480, 1.2930000),
            (0.5663570, 0.5118000),
            (0.5235460, 0.1787000),
        ],
        's3': [
            (1.0000000, 0.0576000),
        ],
        'p1': [
            (0.0128500, 26.6300000),
            (0.0838380, 5.9480000),
            (0.2907530, 1.7420000),
            (0.5005050, 0.5550000),
            (0.3433060, 0.1725000),
        ],
        'p2': [
            (1.0000000, 0.0491000),
        ],
        'd1': [
            (1.0000000, 1.6540000),
        ],
        'd2': [
            (1.0000000, 0.4690000),
        ],
        'f1': [
            (1.0000000, 1.0930000),
        ],
    },
    'O': {
        's1': [
            (0.0003800, 15330.0000000),
            (0.0029420, 2299.0000000),
            (0.0152060, 522.4000000),
            (0.0598190, 147.3000000),
            (0.1818580, 47.5500000),
            (0.3641700, 16.7600000),
            (0.3987730, 6.2070000),
            (0.0698300, 0.6882000),
        ],
        's2': [
            (-0.0077580, 16.7600000),
            (-0.0515070, 6.2070000),
            (-0.1283180, 1.7520000),
            (0.5564440, 0.6882000),
            (0.5315170, 0.2384000),
        ],
        's3': [
            (1.0000000, 0.0737600),
        ],
        'p1': [
            (0.0108800, 34.4600000),
            (0.0728390, 7.7490000),
            (0.2644840, 2.2800000),
            (0.5021650, 0.7156000),
            (0.3567890, 0.2140000),
        ],
        'p2': [
            (1.0000000, 0.0598000),
        ],
        'd1': [
            (1.0000000, 2.3140000),
        ],
        'd2': [
            (1.0000000, 0.6450000),
        ],
        'f1': [
            (1.0000000, 1.4280000),
        ],
    },
    'F': {
        's1': [
            (0.0003190, 19500.0000000),
            (0.0024690, 2923.0000000),
            (0.0127700, 664.5000000),
            (0.0505010, 187.5000000),
            (0.1559280, 60.6200000),
            (0.3242700, 21.4200000),
            (0.4196430, 7.9500000),
            (0.0941260, 0.8815000),
        ],
        's2': [
            (-0.0070880, 21.4200000),
            (-0.0496310, 7.9500000),
            (-0.1183790, 2.2570000),
            (0.5490030, 0.8815000),
            (0.5397980, 0.3041000),
        ],
        's3': [
            (1.0000000, 0.0915800),
        ],
        'p1': [
            (0.0095360, 44.3600000),
            (0.0655040, 10.0800000),
            (0.2472700, 2.9960000),
            (0.5036830, 0.9383000),
            (0.3570740, 0.2733000),
        ],
        'p2': [
            (1.0000000, 0.0736100),
        ],
        'd1': [
            (1.0000000, 3.1070000),
        ],
        'd2': [
            (1.0000000, 0.8550000),
        ],
        'f1': [
            (1.0000000, 1.9170000),
        ],
    },
    'Ne': {
        's1': [
            (0.0002640, 24350.0000000),
            (0.0020500, 3650.0000000),
            (0.0106200, 829.6000000),
            (0.0421940, 234.0000000),
            (0.1324870, 75.6100000),
            (0.2855040, 26.7300000),
            (0.4295560, 9.9270000),
            (0.1264100, 1.1020000),
        ],
        's2': [
            (-0.0065160, 26.7300000),
            (-0.0478050, 9.9270000),
            (-0.1097760, 2.8360000),
            (0.5378840, 1.1020000),
            (0.5487750, 0.3782000),
        ],
        's3': [
            (1.0000000, 0.1133000),
        ],
        'p1': [
            (0.0082970, 56.4500000),
            (0.0585370, 12.9200000),
            (0.2294940, 3.8650000),
            (0.5031130, 1.2030000),
            (0.3602310, 0.3444000),
        ],
        'p2': [
            (1.0000000, 0.0917500),
        ],
        'd1': [
            (1.0000000, 4.0140000),
        ],
        'd2': [
            (1.0000000, 1.0960000),
        ],
        'f1': [
            (1.0000000, 2.5440000),
        ],
    },
}


# cc-pVQZ: Correlation-consistent polarized valence quadruple-zeta
# Data from Basis Set Exchange (Dunning, J. Chem. Phys. 90, 1007 (1989))
CC_PVQZ_DATA = {
    'H': {
        # 4 s-type contracted functions
        's1': [
            (0.0002520, 82.6400000),
            (0.0019550, 12.4100000),
            (0.0100420, 2.8240000),
            (0.0394260, 0.7977000),
            (0.1164070, 0.2581000),
        ],
        's2': [
            (1.0000000, 0.0898900),
        ],
        's3': [
            (1.0000000, 0.0236300),
        ],
        's4': [
            (1.0000000, 0.0073200),
        ],
        # 3 p-type functions
        'p1': [
            (1.0000000, 2.2920000),
        ],
        'p2': [
            (1.0000000, 0.8380000),
        ],
        'p3': [
            (1.0000000, 0.2920000),
        ],
        # 2 d-type functions
        'd1': [
            (1.0000000, 2.0620000),
        ],
        'd2': [
            (1.0000000, 0.6620000),
        ],
        # 1 f-type function
        'f1': [
            (1.0000000, 1.3970000),
        ],
    },
    'He': {
        's1': [
            (0.0002360, 528.5000000),
            (0.0018260, 79.3100000),
            (0.0094520, 18.0500000),
            (0.0378030, 5.0850000),
            (0.1125420, 1.6090000),
        ],
        's2': [
            (1.0000000, 0.5363000),
        ],
        's3': [
            (1.0000000, 0.1833000),
        ],
        's4': [
            (1.0000000, 0.0598100),
        ],
        'p1': [
            (1.0000000, 5.9940000),
        ],
        'p2': [
            (1.0000000, 1.7450000),
        ],
        'p3': [
            (1.0000000, 0.5600000),
        ],
        'd1': [
            (1.0000000, 4.2990000),
        ],
        'd2': [
            (1.0000000, 1.2230000),
        ],
        'f1': [
            (1.0000000, 2.6800000),
        ],
    },
    'C': {
        's1': [
            (0.0000530, 33980.0000000),
            (0.0004140, 5089.0000000),
            (0.0021780, 1157.0000000),
            (0.0090870, 326.6000000),
            (0.0318050, 106.1000000),
            (0.0929220, 38.1100000),
            (0.2141830, 14.7500000),
            (0.3638870, 6.0350000),
            (0.3063210, 2.5300000),
            (0.0578590, 0.7355000),
        ],
        's2': [
            (0.0008270, 38.1100000),
            (-0.0031050, 14.7500000),
            (-0.0108740, 6.0350000),
            (-0.0722860, 2.5300000),
            (-0.0643760, 1.0730000),
            (0.2414250, 0.3290000),
            (0.5899190, 0.1210000),
            (0.2903010, 0.0447700),
        ],
        's3': [
            (1.0000000, 0.0441000),
        ],
        's4': [
            (1.0000000, 0.0154800),
        ],
        'p1': [
            (0.0006200, 34.5100000),
            (0.0050210, 7.9150000),
            (0.0233150, 2.3680000),
            (0.0754350, 0.8132000),
            (0.1678100, 0.2890000),
        ],
        'p2': [
            (1.0000000, 0.1007000),
        ],
        'p3': [
            (1.0000000, 0.0321800),
        ],
        'd1': [
            (1.0000000, 1.8480000),
        ],
        'd2': [
            (1.0000000, 0.6490000),
        ],
        'd3': [
            (1.0000000, 0.2280000),
        ],
        'f1': [
            (1.0000000, 1.4190000),
        ],
        'f2': [
            (1.0000000, 0.4850000),
        ],
        'g1': [
            (1.0000000, 1.0110000),
        ],
    },
    'N': {
        's1': [
            (0.0000450, 45840.0000000),
            (0.0003520, 6868.0000000),
            (0.0018490, 1563.0000000),
            (0.0077600, 442.4000000),
            (0.0274210, 144.3000000),
            (0.0820500, 52.1800000),
            (0.1952090, 20.0000000),
            (0.3527060, 8.2470000),
            (0.3150280, 3.4820000),
            (0.0587430, 0.9967000),
        ],
        's2': [
            (0.0008370, 52.1800000),
            (-0.0027320, 20.0000000),
            (-0.0108700, 8.2470000),
            (-0.0680250, 3.4820000),
            (-0.0670930, 1.4820000),
            (0.2339050, 0.4559000),
            (0.5910090, 0.1649000),
            (0.2928820, 0.0549000),
        ],
        's3': [
            (1.0000000, 0.0557200),
        ],
        's4': [
            (1.0000000, 0.0189000),
        ],
        'p1': [
            (0.0005420, 49.2700000),
            (0.0044640, 11.3700000),
            (0.0214040, 3.4350000),
            (0.0730010, 1.1820000),
            (0.1704640, 0.4173000),
        ],
        'p2': [
            (1.0000000, 0.1428000),
        ],
        'p3': [
            (1.0000000, 0.0440200),
        ],
        'd1': [
            (1.0000000, 2.8370000),
        ],
        'd2': [
            (1.0000000, 0.9680000),
        ],
        'd3': [
            (1.0000000, 0.3350000),
        ],
        'f1': [
            (1.0000000, 2.0270000),
        ],
        'f2': [
            (1.0000000, 0.6850000),
        ],
        'g1': [
            (1.0000000, 1.4270000),
        ],
    },
    'O': {
        's1': [
            (0.0000380, 61420.0000000),
            (0.0003000, 9199.0000000),
            (0.0015780, 2091.0000000),
            (0.0066370, 590.9000000),
            (0.0235870, 192.3000000),
            (0.0718520, 69.3200000),
            (0.1750150, 26.9700000),
            (0.3382700, 11.1000000),
            (0.3272040, 4.6820000),
            (0.0719430, 1.4280000),
        ],
        's2': [
            (0.0007110, 69.3200000),
            (-0.0024590, 26.9700000),
            (-0.0093260, 11.1000000),
            (-0.0601900, 4.6820000),
            (-0.0652340, 1.9280000),
            (0.2193780, 0.6005000),
            (0.5892470, 0.2140000),
            (0.3006660, 0.0776500),
        ],
        's3': [
            (1.0000000, 0.0709000),
        ],
        's4': [
            (1.0000000, 0.0238400),
        ],
        'p1': [
            (0.0004880, 63.4200000),
            (0.0040880, 14.6600000),
            (0.0199890, 4.4590000),
            (0.0698010, 1.5310000),
            (0.1695390, 0.5302000),
        ],
        'p2': [
            (1.0000000, 0.1750000),
        ],
        'p3': [
            (1.0000000, 0.0534000),
        ],
        'd1': [
            (1.0000000, 3.7750000),
        ],
        'd2': [
            (1.0000000, 1.3000000),
        ],
        'd3': [
            (1.0000000, 0.4440000),
        ],
        'f1': [
            (1.0000000, 2.6660000),
        ],
        'f2': [
            (1.0000000, 0.8590000),
        ],
        'g1': [
            (1.0000000, 1.8460000),
        ],
    },
    'F': {
        's1': [
            (0.0000330, 74530.0000000),
            (0.0002530, 11170.0000000),
            (0.0013290, 2543.0000000),
            (0.0056060, 721.0000000),
            (0.0200240, 235.9000000),
            (0.0621580, 85.6000000),
            (0.1563500, 33.5500000),
            (0.3197780, 13.9300000),
            (0.3380880, 5.9150000),
            (0.0837250, 1.8430000),
        ],
        's2': [
            (0.0006100, 85.6000000),
            (-0.0022420, 33.5500000),
            (-0.0081930, 13.9300000),
            (-0.0546890, 5.9150000),
            (-0.0635990, 2.4430000),
            (0.2083780, 0.7659000),
            (0.5848530, 0.2706000),
            (0.3116750, 0.1015000),
        ],
        's3': [
            (1.0000000, 0.0875000),
        ],
        's4': [
            (1.0000000, 0.0291900),
        ],
        'p1': [
            (0.0004410, 80.3900000),
            (0.0037300, 18.6300000),
            (0.0185340, 5.6940000),
            (0.0660680, 1.9530000),
            (0.1665240, 0.6702000),
        ],
        'p2': [
            (1.0000000, 0.2166000),
        ],
        'p3': [
            (1.0000000, 0.0656100),
        ],
        'd1': [
            (1.0000000, 5.0140000),
        ],
        'd2': [
            (1.0000000, 1.7250000),
        ],
        'd3': [
            (1.0000000, 0.5860000),
        ],
        'f1': [
            (1.0000000, 3.5620000),
        ],
        'f2': [
            (1.0000000, 1.1080000),
        ],
        'g1': [
            (1.0000000, 2.3760000),
        ],
    },
    'Ne': {
        's1': [
            (0.0000280, 99920.0000000),
            (0.0002140, 14960.0000000),
            (0.0011270, 3399.0000000),
            (0.0047530, 962.5000000),
            (0.0170280, 313.6000000),
            (0.0536540, 113.2000000),
            (0.1379960, 43.9800000),
            (0.2954160, 18.0300000),
            (0.3476720, 7.6180000),
            (0.1052840, 2.4130000),
        ],
        's2': [
            (0.0005340, 113.2000000),
            (-0.0020800, 43.9800000),
            (-0.0073070, 18.0300000),
            (-0.0505900, 7.6180000),
            (-0.0638440, 3.0990000),
            (0.1993830, 0.9754000),
            (0.5768580, 0.3422000),
            (0.3221820, 0.1324000),
        ],
        's3': [
            (1.0000000, 0.1055000),
        ],
        's4': [
            (1.0000000, 0.0349900),
        ],
        'p1': [
            (0.0003990, 99.6800000),
            (0.0034020, 23.1500000),
            (0.0171500, 7.1080000),
            (0.0624600, 2.4410000),
            (0.1636850, 0.8339000),
        ],
        'p2': [
            (1.0000000, 0.2662000),
        ],
        'p3': [
            (1.0000000, 0.0793700),
        ],
        'd1': [
            (1.0000000, 6.4710000),
        ],
        'd2': [
            (1.0000000, 2.2130000),
        ],
        'd3': [
            (1.0000000, 0.7470000),
        ],
        'f1': [
            (1.0000000, 4.6570000),
        ],
        'f2': [
            (1.0000000, 1.5240000),
        ],
        'g1': [
            (1.0000000, 2.9830000),
        ],
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
        name_upper = name.upper().replace('-', '').replace('_', '')
        if name_upper == 'STO3G':
            return STO_3G_DATA
        elif name_upper in ('CCPVTZ', 'PVTZ'):
            return CC_PVTZ_DATA
        elif name_upper in ('CCPVQZ', 'PVQZ'):
            return CC_PVQZ_DATA
        else:
            raise ValueError(f"Unknown basis set: {name}. Available: STO-3G, cc-pVTZ, cc-pVQZ")

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
                # Parse shell type - handle both '1s'/'2p' and 's1'/'p2' naming
                # Extract the letter (s, p, d, f, g)
                shell_type = None
                for char in shell_name:
                    if char in 'spdfg':
                        shell_type = char
                        break

                if shell_type is None:
                    raise ValueError(f"Unknown shell type in {shell_name}")

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
                    # Six d functions (Cartesian: xx, yy, zz, xy, xz, yz)
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

                elif shell_type == 'f':
                    # Ten f functions (Cartesian: xxx, yyy, zzz, xxy, xxz, xyy, yyz, xzz, yzz, xyz)
                    f_angular = [
                        (3, 0, 0), (0, 3, 0), (0, 0, 3),  # xxx, yyy, zzz
                        (2, 1, 0), (2, 0, 1), (1, 2, 0),  # xxy, xxz, xyy
                        (0, 2, 1), (1, 0, 2), (0, 1, 2),  # yyz, xzz, yzz
                        (1, 1, 1)                          # xyz
                    ]
                    for m, angular in enumerate(f_angular):
                        cgf = ContractedGaussian(
                            primitives=primitives,
                            center=center,
                            angular=angular
                        )
                        self.functions.append(BasisFunction(
                            cgf=cgf,
                            atom_index=atom_idx,
                            shell_type='f',
                            m=m
                        ))

                elif shell_type == 'g':
                    # Fifteen g functions (Cartesian: all (i,j,k) where i+j+k=4)
                    g_angular = [
                        (4, 0, 0), (0, 4, 0), (0, 0, 4),  # xxxx, yyyy, zzzz
                        (3, 1, 0), (3, 0, 1), (1, 3, 0),  # xxxy, xxxz, xyyy
                        (0, 3, 1), (1, 0, 3), (0, 1, 3),  # yyyz, xzzz, yzzz
                        (2, 2, 0), (2, 0, 2), (0, 2, 2),  # xxyy, xxzz, yyzz
                        (2, 1, 1), (1, 2, 1), (1, 1, 2),  # xxyz, xyyz, xyzz
                    ]
                    for m, angular in enumerate(g_angular):
                        cgf = ContractedGaussian(
                            primitives=primitives,
                            center=center,
                            angular=angular
                        )
                        self.functions.append(BasisFunction(
                            cgf=cgf,
                            atom_index=atom_idx,
                            shell_type='g',
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
