"""
Range-Separated Hybrid Functionals

Implements functionals with different treatment of short-range and long-range exchange:
- CAM-B3LYP (Coulomb-attenuating method)
- ωB97X-D (range-separated with dispersion)

References:
- Yanai et al., Chem. Phys. Lett. 393, 51 (2004)
- Chai, Head-Gordon, Phys. Chem. Chem. Phys. 10, 6615 (2008)
"""

import numpy as np

from .base import (
    XCFunctional, DensityData, XCOutput, FunctionalType,
    RHO_THRESHOLD, SIGMA_THRESHOLD
)
from .gga import B88Exchange, LYPCorrelation, PBEExchange, PBECorrelation
from .lda import SlaterExchange, VWN5Correlation


class CAMB3LYP(XCFunctional):
    """
    CAM-B3LYP range-separated hybrid functional.

    Uses Coulomb-attenuating method to switch between short-range
    and long-range exchange:

    1/r₁₂ = [α + β·erf(μr₁₂)]/r₁₂ + [1 - α - β·erf(μr₁₂)]/r₁₂
            |______ HF _______|   |________ DFT ________|

    Parameters:
    - α = 0.19 (fraction of HF exchange at short range)
    - β = 0.46 (additional HF exchange at long range)
    - μ = 0.33 (range-separation parameter in Bohr⁻¹)

    Total HF exchange: α at r→0, α+β at r→∞

    Reference: Yanai et al., Chem. Phys. Lett. 393, 51 (2004)
    """

    # CAM-B3LYP parameters
    ALPHA = 0.19    # Short-range HF exchange
    BETA = 0.46     # Long-range HF exchange increment
    MU = 0.33       # Range-separation parameter (Bohr⁻¹)

    # Correlation mixing (same as B3LYP)
    AC = 0.81

    def __init__(self):
        super().__init__("CAM-B3LYP", FunctionalType.RANGE_SEPARATED)

        self._slater = SlaterExchange()
        self._b88 = B88Exchange()
        self._vwn = VWN5Correlation()
        self._lyp = LYPCorrelation()

    @property
    def exact_exchange_sr(self) -> float:
        """Fraction of short-range exact exchange."""
        return self.ALPHA

    @property
    def exact_exchange_lr(self) -> float:
        """Fraction of long-range exact exchange."""
        return self.ALPHA + self.BETA

    @property
    def range_separation_parameter(self) -> float:
        """Range-separation parameter μ (Bohr⁻¹)."""
        return self.MU

    def compute(self, density: DensityData) -> XCOutput:
        """
        Compute CAM-B3LYP XC energy density and derivatives.

        Note: This computes the DFT part. Range-separated exact exchange
        requires special integral handling for erf/erfc attenuated Coulomb.
        """
        rho = np.maximum(density.rho, RHO_THRESHOLD)

        # LDA exchange
        slater_out = self._slater.compute(density)

        # B88 exchange (includes gradient correction)
        b88_out = self._b88.compute(density)
        b88_correction_exc = b88_out.exc - slater_out.exc
        b88_correction_vrho = b88_out.vrho - slater_out.vrho
        b88_correction_vsigma = b88_out.vsigma

        # VWN correlation
        vwn_out = self._vwn.compute(density)

        # LYP correlation
        lyp_out = self._lyp.compute(density)

        # DFT exchange part: (1-α-β) at long range, (1-α) at short range
        # Simplified: use average weight for DFT exchange
        dft_x_weight = 1 - self.ALPHA - 0.5 * self.BETA

        exc = (dft_x_weight * slater_out.exc +
               0.65 * b88_correction_exc +  # B88 weight (empirical)
               (1 - self.AC) * vwn_out.exc +
               self.AC * lyp_out.exc)

        vrho = (dft_x_weight * slater_out.vrho +
                0.65 * b88_correction_vrho +
                (1 - self.AC) * vwn_out.vrho +
                self.AC * lyp_out.vrho)

        vsigma = None
        if b88_correction_vsigma is not None or lyp_out.vsigma is not None:
            vsigma = np.zeros_like(density.sigma)
            if b88_correction_vsigma is not None:
                vsigma += 0.65 * b88_correction_vsigma
            if lyp_out.vsigma is not None:
                vsigma += self.AC * lyp_out.vsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


class wB97XD(XCFunctional):
    """
    ωB97X-D range-separated hybrid functional with dispersion.

    A range-separated functional with 100% long-range HF exchange
    and empirical dispersion correction (D2).

    Parameters:
    - ω = 0.2 (range-separation parameter)
    - cₓ = varies with r (short-range DFT, long-range HF)
    - D2 dispersion with s₆ = 1.0

    Reference: Chai, Head-Gordon, Phys. Chem. Chem. Phys. 10, 6615 (2008)
    """

    # ωB97X-D parameters
    OMEGA = 0.2     # Range-separation parameter (Bohr⁻¹)

    def __init__(self):
        super().__init__("ωB97X-D", FunctionalType.RANGE_SEPARATED)

        self._pbe_x = PBEExchange()
        self._pbe_c = PBECorrelation()

    @property
    def exact_exchange_sr(self) -> float:
        """Fraction of short-range exact exchange."""
        return 0.222036

    @property
    def exact_exchange_lr(self) -> float:
        """Fraction of long-range exact exchange."""
        return 1.0  # 100% HF at long range

    @property
    def range_separation_parameter(self) -> float:
        """Range-separation parameter ω (Bohr⁻¹)."""
        return self.OMEGA

    def compute(self, density: DensityData) -> XCOutput:
        """
        Compute ωB97X-D XC energy density and derivatives.

        Note: This is a simplified implementation. Full ωB97X-D
        requires the specific B97-style exchange and correlation
        with range-separated integrals and D2 dispersion.
        """
        rho = np.maximum(density.rho, RHO_THRESHOLD)

        # Use PBE as base (simplified)
        pbe_x_out = self._pbe_x.compute(density)
        pbe_c_out = self._pbe_c.compute(density)

        # Weight DFT exchange for short-range contribution
        sr_weight = 1 - self.exact_exchange_sr

        exc = sr_weight * pbe_x_out.exc + pbe_c_out.exc
        vrho = sr_weight * pbe_x_out.vrho + pbe_c_out.vrho

        vsigma = None
        if pbe_x_out.vsigma is not None or pbe_c_out.vsigma is not None:
            vsigma = np.zeros_like(density.sigma)
            if pbe_x_out.vsigma is not None:
                vsigma += sr_weight * pbe_x_out.vsigma
            if pbe_c_out.vsigma is not None:
                vsigma += pbe_c_out.vsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


class wB97MV(XCFunctional):
    """
    ωB97M-V range-separated functional with VV10 nonlocal correlation.

    A modern range-separated functional optimized for thermochemistry
    and non-covalent interactions.

    Reference: Mardirossian, Head-Gordon, J. Chem. Phys. 144, 214110 (2016)
    """

    OMEGA = 0.3  # Range-separation parameter

    def __init__(self):
        super().__init__("ωB97M-V", FunctionalType.RANGE_SEPARATED)

        self._pbe_x = PBEExchange()
        self._pbe_c = PBECorrelation()

    @property
    def exact_exchange_sr(self) -> float:
        return 0.15

    @property
    def exact_exchange_lr(self) -> float:
        return 1.0

    @property
    def range_separation_parameter(self) -> float:
        return self.OMEGA

    def compute(self, density: DensityData) -> XCOutput:
        """Compute ωB97M-V (placeholder using PBE)."""
        pbe_x_out = self._pbe_x.compute(density)
        pbe_c_out = self._pbe_c.compute(density)

        sr_weight = 1 - self.exact_exchange_sr

        exc = sr_weight * pbe_x_out.exc + pbe_c_out.exc
        vrho = sr_weight * pbe_x_out.vrho + pbe_c_out.vrho

        vsigma = None
        if pbe_x_out.vsigma is not None or pbe_c_out.vsigma is not None:
            vsigma = np.zeros_like(density.sigma)
            if pbe_x_out.vsigma is not None:
                vsigma += sr_weight * pbe_x_out.vsigma
            if pbe_c_out.vsigma is not None:
                vsigma += pbe_c_out.vsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


# Convenience functions
def cam_b3lyp() -> CAMB3LYP:
    """Create CAM-B3LYP range-separated hybrid functional."""
    return CAMB3LYP()


def wb97xd() -> wB97XD:
    """Create ωB97X-D range-separated hybrid functional."""
    return wB97XD()


def wb97mv() -> wB97MV:
    """Create ωB97M-V range-separated functional."""
    return wB97MV()
