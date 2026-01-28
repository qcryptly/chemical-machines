"""
Hybrid Exchange-Correlation Functionals

Implements hybrid functionals that mix DFT exchange with exact (HF) exchange:
- B3LYP (Becke 3-parameter with LYP correlation)
- PBE0 (25% exact exchange with PBE)

References:
- Becke, J. Chem. Phys. 98, 5648 (1993)
- Adamo, Barone, J. Chem. Phys. 110, 6158 (1999)
"""

import numpy as np

from .base import (
    XCFunctional, CombinedFunctional, DensityData, XCOutput, FunctionalType,
    RHO_THRESHOLD, SIGMA_THRESHOLD
)
from .lda import SlaterExchange, VWN5Correlation
from .gga import B88Exchange, LYPCorrelation, PBEExchange, PBECorrelation


class B3LYP(XCFunctional):
    """
    B3LYP hybrid functional.

    The most widely used hybrid functional, combining:
    - 20% Hartree-Fock exchange
    - 8% Slater (LDA) exchange
    - 72% Becke88 exchange
    - 19% VWN5 correlation
    - 81% LYP correlation

    E_xc = a₀ E_x^HF + (1-a₀) E_x^LDA + a_x ΔE_x^B88 + a_c E_c^LYP + (1-a_c) E_c^VWN

    with a₀ = 0.20, a_x = 0.72, a_c = 0.81

    Reference: Becke, J. Chem. Phys. 98, 5648 (1993)
    """

    # B3LYP parameters
    A0 = 0.20   # Exact exchange
    AX = 0.72   # B88 exchange gradient correction
    AC = 0.81   # LYP correlation

    def __init__(self):
        super().__init__("B3LYP", FunctionalType.HYBRID)

        # Component functionals
        self._slater = SlaterExchange()
        self._b88 = B88Exchange()
        self._vwn = VWN5Correlation()
        self._lyp = LYPCorrelation()

    @property
    def exact_exchange(self) -> float:
        """Fraction of exact (HF) exchange."""
        return self.A0

    def compute(self, density: DensityData) -> XCOutput:
        """
        Compute B3LYP XC energy density and derivatives.

        Note: This computes only the DFT part. The exact exchange
        contribution must be computed separately using HF exchange
        integrals and added with weight self.exact_exchange.
        """
        rho = np.maximum(density.rho, RHO_THRESHOLD)

        # LDA exchange
        slater_out = self._slater.compute(density)

        # B88 exchange (includes LDA + gradient correction)
        b88_out = self._b88.compute(density)
        # B88 gradient correction only
        b88_correction_exc = b88_out.exc - slater_out.exc
        b88_correction_vrho = b88_out.vrho - slater_out.vrho
        b88_correction_vsigma = b88_out.vsigma if b88_out.vsigma is not None else None

        # VWN correlation
        vwn_out = self._vwn.compute(density)

        # LYP correlation
        lyp_out = self._lyp.compute(density)

        # Combine according to B3LYP formula
        # E_xc = (1-a₀) E_x^LDA + a_x ΔE_x^B88 + (1-a_c) E_c^VWN + a_c E_c^LYP

        exc = ((1 - self.A0) * slater_out.exc +
               self.AX * b88_correction_exc +
               (1 - self.AC) * vwn_out.exc +
               self.AC * lyp_out.exc)

        vrho = ((1 - self.A0) * slater_out.vrho +
                self.AX * b88_correction_vrho +
                (1 - self.AC) * vwn_out.vrho +
                self.AC * lyp_out.vrho)

        # Gradient terms
        vsigma = None
        if b88_correction_vsigma is not None or lyp_out.vsigma is not None:
            vsigma = np.zeros_like(density.sigma)
            if b88_correction_vsigma is not None:
                vsigma += self.AX * b88_correction_vsigma
            if lyp_out.vsigma is not None:
                vsigma += self.AC * lyp_out.vsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


class PBE0(XCFunctional):
    """
    PBE0 hybrid functional.

    Combines 25% exact exchange with PBE:
    E_xc = a₀ E_x^HF + (1-a₀) E_x^PBE + E_c^PBE

    with a₀ = 0.25

    Also known as PBE1PBE or PBEh.

    Reference: Adamo, Barone, J. Chem. Phys. 110, 6158 (1999)
    """

    # PBE0 exact exchange fraction
    A0 = 0.25

    def __init__(self):
        super().__init__("PBE0", FunctionalType.HYBRID)

        # Component functionals
        self._pbe_x = PBEExchange()
        self._pbe_c = PBECorrelation()

    @property
    def exact_exchange(self) -> float:
        """Fraction of exact (HF) exchange."""
        return self.A0

    def compute(self, density: DensityData) -> XCOutput:
        """
        Compute PBE0 XC energy density and derivatives.

        Note: This computes only the DFT part. The exact exchange
        contribution must be computed separately using HF exchange
        integrals and added with weight self.exact_exchange.
        """
        # PBE exchange
        pbe_x_out = self._pbe_x.compute(density)

        # PBE correlation
        pbe_c_out = self._pbe_c.compute(density)

        # Combine: (1-a₀) E_x^PBE + E_c^PBE
        exc = (1 - self.A0) * pbe_x_out.exc + pbe_c_out.exc
        vrho = (1 - self.A0) * pbe_x_out.vrho + pbe_c_out.vrho

        vsigma = None
        if pbe_x_out.vsigma is not None or pbe_c_out.vsigma is not None:
            vsigma = np.zeros_like(density.sigma)
            if pbe_x_out.vsigma is not None:
                vsigma += (1 - self.A0) * pbe_x_out.vsigma
            if pbe_c_out.vsigma is not None:
                vsigma += pbe_c_out.vsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


class TPSSh(XCFunctional):
    """
    TPSSh hybrid meta-GGA functional.

    Combines 10% exact exchange with TPSS:
    E_xc = 0.10 E_x^HF + 0.90 E_x^TPSS + E_c^TPSS

    Note: Full TPSS implementation requires kinetic energy density (tau).
    This is a placeholder that currently falls back to PBE.

    Reference: Staroverov et al., J. Chem. Phys. 119, 12129 (2003)
    """

    A0 = 0.10

    def __init__(self):
        super().__init__("TPSSh", FunctionalType.HYBRID)
        # Placeholder: use PBE until TPSS is implemented
        self._pbe_x = PBEExchange()
        self._pbe_c = PBECorrelation()

    @property
    def exact_exchange(self) -> float:
        return self.A0

    def compute(self, density: DensityData) -> XCOutput:
        """Compute TPSSh (currently using PBE as placeholder)."""
        pbe_x_out = self._pbe_x.compute(density)
        pbe_c_out = self._pbe_c.compute(density)

        exc = (1 - self.A0) * pbe_x_out.exc + pbe_c_out.exc
        vrho = (1 - self.A0) * pbe_x_out.vrho + pbe_c_out.vrho

        vsigma = None
        if pbe_x_out.vsigma is not None or pbe_c_out.vsigma is not None:
            vsigma = np.zeros_like(density.sigma)
            if pbe_x_out.vsigma is not None:
                vsigma += (1 - self.A0) * pbe_x_out.vsigma
            if pbe_c_out.vsigma is not None:
                vsigma += pbe_c_out.vsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


class M06(XCFunctional):
    """
    M06 hybrid meta-GGA functional.

    Minnesota functional with 27% exact exchange. Good for
    main-group thermochemistry and transition metal chemistry.

    Note: Full implementation requires kinetic energy density.
    This is a placeholder.

    Reference: Zhao, Truhlar, Theor. Chem. Acc. 120, 215 (2008)
    """

    A0 = 0.27

    def __init__(self):
        super().__init__("M06", FunctionalType.HYBRID)
        self._pbe_x = PBEExchange()
        self._pbe_c = PBECorrelation()

    @property
    def exact_exchange(self) -> float:
        return self.A0

    def compute(self, density: DensityData) -> XCOutput:
        """Compute M06 (placeholder using PBE)."""
        pbe_x_out = self._pbe_x.compute(density)
        pbe_c_out = self._pbe_c.compute(density)

        exc = (1 - self.A0) * pbe_x_out.exc + pbe_c_out.exc
        vrho = (1 - self.A0) * pbe_x_out.vrho + pbe_c_out.vrho

        vsigma = None
        if pbe_x_out.vsigma is not None or pbe_c_out.vsigma is not None:
            vsigma = np.zeros_like(density.sigma)
            if pbe_x_out.vsigma is not None:
                vsigma += (1 - self.A0) * pbe_x_out.vsigma
            if pbe_c_out.vsigma is not None:
                vsigma += pbe_c_out.vsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


class M062X(XCFunctional):
    """
    M06-2X hybrid meta-GGA functional.

    Minnesota functional with 54% exact exchange. Recommended for
    main-group chemistry, especially non-covalent interactions.

    Note: Full implementation requires kinetic energy density.
    This is a placeholder.

    Reference: Zhao, Truhlar, Theor. Chem. Acc. 120, 215 (2008)
    """

    A0 = 0.54

    def __init__(self):
        super().__init__("M06-2X", FunctionalType.HYBRID)
        self._pbe_x = PBEExchange()
        self._pbe_c = PBECorrelation()

    @property
    def exact_exchange(self) -> float:
        return self.A0

    def compute(self, density: DensityData) -> XCOutput:
        """Compute M06-2X (placeholder using PBE)."""
        pbe_x_out = self._pbe_x.compute(density)
        pbe_c_out = self._pbe_c.compute(density)

        exc = (1 - self.A0) * pbe_x_out.exc + pbe_c_out.exc
        vrho = (1 - self.A0) * pbe_x_out.vrho + pbe_c_out.vrho

        vsigma = None
        if pbe_x_out.vsigma is not None or pbe_c_out.vsigma is not None:
            vsigma = np.zeros_like(density.sigma)
            if pbe_x_out.vsigma is not None:
                vsigma += (1 - self.A0) * pbe_x_out.vsigma
            if pbe_c_out.vsigma is not None:
                vsigma += pbe_c_out.vsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


# Convenience functions
def b3lyp() -> B3LYP:
    """Create B3LYP hybrid functional."""
    return B3LYP()


def pbe0() -> PBE0:
    """Create PBE0 hybrid functional."""
    return PBE0()


def tpssh() -> TPSSh:
    """Create TPSSh hybrid functional."""
    return TPSSh()


def m06() -> M06:
    """Create M06 hybrid functional."""
    return M06()


def m06_2x() -> M062X:
    """Create M06-2X hybrid functional."""
    return M062X()
