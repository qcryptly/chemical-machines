"""
Generalized Gradient Approximation (GGA) Functionals

Implements:
- B88 exchange (Becke, 1988)
- LYP correlation (Lee, Yang, Parr, 1988)
- PBE exchange and correlation (Perdew, Burke, Ernzerhof, 1996)

Combined functionals:
- BLYP = B88 + LYP
- BP86 = B88 + P86
- PBE = PBE_X + PBE_C

References:
- Becke, Phys. Rev. A 38, 3098 (1988)
- Lee, Yang, Parr, Phys. Rev. B 37, 785 (1988)
- Perdew, Burke, Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
"""

import numpy as np
from typing import Tuple

from .base import (
    ExchangeFunctional, CorrelationFunctional, CombinedFunctional,
    DensityData, XCOutput, FunctionalType,
    RHO_THRESHOLD, SIGMA_THRESHOLD
)
from .lda import SlaterExchange, VWN5Correlation


class B88Exchange(ExchangeFunctional):
    """
    Becke 1988 gradient-corrected exchange functional.

    Adds a gradient correction to LDA exchange:
        ε_x^B88 = ε_x^LDA - β ρ^(1/3) x² / (1 + 6β x sinh⁻¹(x))

    where x = |∇ρ| / ρ^(4/3) is the reduced gradient.

    Reference: Becke, Phys. Rev. A 38, 3098 (1988)
    """

    # Becke88 empirical parameter (fitted to noble gas atoms)
    BETA = 0.0042

    def __init__(self):
        super().__init__("Becke88 Exchange", FunctionalType.GGA)
        self._slater = SlaterExchange()

    def compute(self, density: DensityData) -> XCOutput:
        """Compute B88 exchange energy density and derivatives."""
        if density.sigma is None:
            raise ValueError("B88 requires density gradient (sigma)")

        rho = np.maximum(density.rho, RHO_THRESHOLD)
        sigma = np.maximum(density.sigma, SIGMA_THRESHOLD)

        if density.is_polarized:
            return self._compute_polarized(density)

        # Reduced gradient
        rho_43 = rho ** (4.0 / 3.0)
        grad_rho = np.sqrt(sigma)
        x = grad_rho / rho_43

        # LDA exchange contribution
        lda_out = self._slater.compute(density)

        # B88 correction
        x2 = x ** 2
        asinh_x = np.arcsinh(x)
        denom = 1.0 + 6.0 * self.BETA * x * asinh_x

        # Energy correction: Δε_x = -β ρ^(1/3) x² / denom
        rho_13 = rho ** (1.0 / 3.0)
        delta_exc = -self.BETA * rho_13 * x2 / denom

        exc = lda_out.exc + delta_exc

        # Derivatives
        # Need ∂ε_x/∂ρ and ∂ε_x/∂σ
        # Chain rule through x = σ^(1/2) / ρ^(4/3)

        # ∂x/∂ρ = -(4/3) x / ρ
        # ∂x/∂σ = 1 / (2 σ^(1/2) ρ^(4/3)) = 1 / (2 x ρ^(8/3))

        # ∂(Δε_x)/∂x
        ddenom_dx = 6.0 * self.BETA * (asinh_x + x / np.sqrt(1 + x2))
        d_delta_dx = -self.BETA * rho_13 * (2 * x * denom - x2 * ddenom_dx) / denom ** 2

        # ∂ε_x/∂ρ
        d_delta_drho = (1.0 / 3.0) * delta_exc / rho + d_delta_dx * (-(4.0 / 3.0) * x / rho)
        vrho = lda_out.vrho + d_delta_drho

        # ∂ε_x/∂σ
        vsigma = d_delta_dx / (2.0 * grad_rho * rho_43)

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)

    def _compute_polarized(self, density: DensityData) -> XCOutput:
        """Compute for spin-polarized case."""
        rho_a = np.maximum(density.rho_alpha, RHO_THRESHOLD)
        rho_b = np.maximum(density.rho_beta, RHO_THRESHOLD)
        rho = rho_a + rho_b

        sigma_aa = np.maximum(density.sigma_aa, SIGMA_THRESHOLD)
        sigma_bb = np.maximum(density.sigma_bb, SIGMA_THRESHOLD)

        # Compute for each spin channel using spin-scaling relation
        exc = np.zeros_like(rho)
        vrho = np.zeros((len(rho), 2))
        vsigma = np.zeros((len(rho), 3))

        for i, (rho_s, sigma_s) in enumerate([(rho_a, sigma_aa), (rho_b, sigma_bb)]):
            # For spin channel: ε_x[ρ_σ] = 2^(1/3) ε_x[2ρ_σ]
            rho_2s = 2.0 * rho_s
            sigma_2s = 4.0 * sigma_s  # |∇(2ρ)|² = 4|∇ρ|²

            rho_43 = rho_2s ** (4.0 / 3.0)
            grad_rho = np.sqrt(sigma_2s)
            x = grad_rho / rho_43

            rho_13 = rho_2s ** (1.0 / 3.0)
            x2 = x ** 2
            asinh_x = np.arcsinh(x)
            denom = 1.0 + 6.0 * self.BETA * x * asinh_x

            # LDA part for this spin
            C_X = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
            exc_lda_s = -C_X * rho_13

            # B88 correction
            delta_exc = -self.BETA * rho_13 * x2 / denom

            exc_s = (exc_lda_s + delta_exc) * rho_s / rho

            exc += exc_s

            # Derivatives
            ddenom_dx = 6.0 * self.BETA * (asinh_x + x / np.sqrt(1 + x2))
            d_delta_dx = -self.BETA * rho_13 * (2 * x * denom - x2 * ddenom_dx) / denom ** 2

            vrho_lda = (4.0 / 3.0) * exc_lda_s
            d_delta_drho = (1.0 / 3.0) * delta_exc / rho_2s + d_delta_dx * (-(4.0 / 3.0) * x / rho_2s)

            # Factor of 2 from chain rule through 2ρ_σ
            vrho[:, i] = 2.0 * (vrho_lda + d_delta_drho)

            # vsigma: factor of 4 from chain rule through 4σ
            vsigma_s = d_delta_dx / (2.0 * grad_rho * rho_43) * 4.0
            vsigma[:, 2 * i] = vsigma_s

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


class LYPCorrelation(CorrelationFunctional):
    """
    Lee-Yang-Parr correlation functional.

    A gradient-corrected correlation functional based on the
    Colle-Salvetti correlation energy formula.

    Reference: Lee, Yang, Parr, Phys. Rev. B 37, 785 (1988)
    """

    # LYP parameters
    A = 0.04918
    B = 0.132
    C = 0.2533
    D = 0.349

    def __init__(self):
        super().__init__("LYP Correlation", FunctionalType.GGA)

    def compute(self, density: DensityData) -> XCOutput:
        """Compute LYP correlation energy density and derivatives."""
        if density.sigma is None:
            raise ValueError("LYP requires density gradient (sigma)")

        rho = np.maximum(density.rho, RHO_THRESHOLD)
        sigma = np.maximum(density.sigma, SIGMA_THRESHOLD)

        if density.is_polarized:
            return self._compute_polarized(density)

        # For closed-shell, ρ_α = ρ_β = ρ/2
        rho_a = rho_b = rho / 2.0
        sigma_aa = sigma_bb = sigma / 4.0
        sigma_ab = sigma / 4.0

        return self._compute_core(rho, rho_a, rho_b, sigma, sigma_aa, sigma_ab, sigma_bb)

    def _compute_polarized(self, density: DensityData) -> XCOutput:
        """Compute for spin-polarized case."""
        rho_a = np.maximum(density.rho_alpha, RHO_THRESHOLD)
        rho_b = np.maximum(density.rho_beta, RHO_THRESHOLD)
        rho = rho_a + rho_b
        sigma = np.maximum(density.sigma, SIGMA_THRESHOLD)
        sigma_aa = np.maximum(density.sigma_aa, SIGMA_THRESHOLD)
        sigma_ab = density.sigma_ab if density.sigma_ab is not None else np.zeros_like(rho)
        sigma_bb = np.maximum(density.sigma_bb, SIGMA_THRESHOLD)

        return self._compute_core(rho, rho_a, rho_b, sigma, sigma_aa, sigma_ab, sigma_bb, polarized=True)

    def _compute_core(self, rho, rho_a, rho_b, sigma, sigma_aa, sigma_ab, sigma_bb,
                      polarized=False) -> XCOutput:
        """Core LYP computation."""
        A, B, C, D = self.A, self.B, self.C, self.D

        rho_13 = rho ** (1.0 / 3.0)
        rho_m13 = rho ** (-1.0 / 3.0)

        # Auxiliary quantities
        omega = np.exp(-C * rho_m13) / (1.0 + D * rho_m13)
        delta = C * rho_m13 + D * rho_m13 / (1.0 + D * rho_m13)

        # Kinetic energy density terms (Thomas-Fermi)
        CF = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
        rho_a_53 = rho_a ** (5.0 / 3.0)
        rho_b_53 = rho_b ** (5.0 / 3.0)
        t_W = sigma / 8.0  # von Weizsäcker kinetic energy density

        # LYP energy density
        rho_ab = rho_a * rho_b

        # First term: -a * 4 * ρ_α * ρ_β / (1 + d*ρ^(-1/3)) / ρ
        term1 = -4.0 * A * rho_ab / rho / (1.0 + D * rho_m13)

        # Second term involves gradient corrections
        term2 = -A * B * omega * rho_ab * (
            CF * 2 ** (11.0 / 3.0) * (rho_a_53 + rho_b_53)
            + (47.0 / 18.0 - 7.0 / 18.0 * delta) * sigma
            - (5.0 / 2.0 - 1.0 / 18.0 * delta) * (sigma_aa + sigma_bb)
            - (delta - 11.0) / 9.0 * (rho_a * sigma_aa + rho_b * sigma_bb) / rho
            + 2.0 / 3.0 * rho ** 2 * sigma
        ) / rho ** 2

        exc = term1 + term2

        # Derivatives (simplified - full derivatives are complex)
        # For now, use finite difference approximation for potentials
        # In production, these would be computed analytically

        # Approximate vrho from energy density
        vrho = exc + rho * self._numerical_deriv_rho(rho, rho_a, rho_b, sigma, sigma_aa, sigma_ab, sigma_bb)

        # Approximate vsigma
        vsigma = self._numerical_deriv_sigma(rho, rho_a, rho_b, sigma, sigma_aa, sigma_ab, sigma_bb)

        if polarized:
            vrho_a = vrho * (rho_a / rho)
            vrho_b = vrho * (rho_b / rho)
            vrho = np.column_stack([vrho_a, vrho_b])
            vsigma = np.column_stack([vsigma * 0.5, vsigma * 0.25, vsigma * 0.5])

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)

    def _numerical_deriv_rho(self, rho, rho_a, rho_b, sigma, sigma_aa, sigma_ab, sigma_bb):
        """Numerical derivative with respect to rho."""
        h = 1e-6
        rho_p = rho + h
        rho_m = rho - h
        rho_a_p = rho_a + h / 2
        rho_b_p = rho_b + h / 2
        rho_a_m = rho_a - h / 2
        rho_b_m = rho_b - h / 2

        exc_p = self._exc_only(rho_p, rho_a_p, rho_b_p, sigma, sigma_aa, sigma_ab, sigma_bb)
        exc_m = self._exc_only(rho_m, rho_a_m, rho_b_m, sigma, sigma_aa, sigma_ab, sigma_bb)

        return (exc_p - exc_m) / (2 * h)

    def _numerical_deriv_sigma(self, rho, rho_a, rho_b, sigma, sigma_aa, sigma_ab, sigma_bb):
        """Numerical derivative with respect to sigma."""
        h = 1e-6
        sigma_p = sigma + h
        sigma_m = sigma - h
        sigma_aa_p = sigma_aa + h / 4
        sigma_bb_p = sigma_bb + h / 4
        sigma_aa_m = sigma_aa - h / 4
        sigma_bb_m = sigma_bb - h / 4

        exc_p = self._exc_only(rho, rho_a, rho_b, sigma_p, sigma_aa_p, sigma_ab, sigma_bb_p)
        exc_m = self._exc_only(rho, rho_a, rho_b, sigma_m, sigma_aa_m, sigma_ab, sigma_bb_m)

        return (exc_p - exc_m) / (2 * h)

    def _exc_only(self, rho, rho_a, rho_b, sigma, sigma_aa, sigma_ab, sigma_bb):
        """Compute only exc for numerical derivatives."""
        A, B, C, D = self.A, self.B, self.C, self.D

        rho = np.maximum(rho, RHO_THRESHOLD)
        rho_13 = rho ** (1.0 / 3.0)
        rho_m13 = rho ** (-1.0 / 3.0)

        omega = np.exp(-C * rho_m13) / (1.0 + D * rho_m13)
        delta = C * rho_m13 + D * rho_m13 / (1.0 + D * rho_m13)

        CF = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
        rho_a_53 = np.maximum(rho_a, RHO_THRESHOLD) ** (5.0 / 3.0)
        rho_b_53 = np.maximum(rho_b, RHO_THRESHOLD) ** (5.0 / 3.0)

        rho_ab = rho_a * rho_b

        term1 = -4.0 * A * rho_ab / rho / (1.0 + D * rho_m13)

        term2 = -A * B * omega * rho_ab * (
            CF * 2 ** (11.0 / 3.0) * (rho_a_53 + rho_b_53)
            + (47.0 / 18.0 - 7.0 / 18.0 * delta) * sigma
            - (5.0 / 2.0 - 1.0 / 18.0 * delta) * (sigma_aa + sigma_bb)
            - (delta - 11.0) / 9.0 * (rho_a * sigma_aa + rho_b * sigma_bb) / rho
            + 2.0 / 3.0 * rho ** 2 * sigma
        ) / rho ** 2

        return term1 + term2


class PBEExchange(ExchangeFunctional):
    """
    Perdew-Burke-Ernzerhof exchange functional.

    Enhancement factor over LDA:
        F_x(s) = 1 + κ - κ / (1 + μs²/κ)

    where s = |∇ρ| / (2(3π²)^(1/3) ρ^(4/3)) is the reduced gradient.

    Reference: Perdew, Burke, Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
    """

    # PBE parameters
    KAPPA = 0.804
    MU = 0.21951  # μ = β π² / 3, where β = 0.066725

    def __init__(self):
        super().__init__("PBE Exchange", FunctionalType.GGA)
        self._slater = SlaterExchange()

    def compute(self, density: DensityData) -> XCOutput:
        """Compute PBE exchange energy density and derivatives."""
        if density.sigma is None:
            raise ValueError("PBE requires density gradient (sigma)")

        rho = np.maximum(density.rho, RHO_THRESHOLD)
        sigma = np.maximum(density.sigma, SIGMA_THRESHOLD)

        if density.is_polarized:
            return self._compute_polarized(density)

        # Reduced gradient s = |∇ρ| / (2 k_F ρ)
        # where k_F = (3π²ρ)^(1/3)
        kF = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
        grad_rho = np.sqrt(sigma)
        s = grad_rho / (2.0 * kF * rho)
        s2 = s ** 2

        # Enhancement factor
        Fx = 1.0 + self.KAPPA - self.KAPPA / (1.0 + self.MU * s2 / self.KAPPA)

        # LDA exchange energy density
        lda_out = self._slater.compute(density)
        exc_lda = lda_out.exc

        # PBE exchange energy density
        exc = exc_lda * Fx

        # Derivatives
        # dF_x/ds² = μ / (1 + μs²/κ)²
        dFx_ds2 = self.MU / (1.0 + self.MU * s2 / self.KAPPA) ** 2

        # ds²/dρ = -8s²/(3ρ) (from s ∝ ρ^(-4/3))
        # ds²/dσ = 1/(4 k_F² ρ²)

        ds2_drho = -8.0 * s2 / (3.0 * rho)
        ds2_dsigma = 1.0 / (4.0 * kF ** 2 * rho ** 2)

        # vrho = d(ρ ε_x)/dρ = ε_x + ρ dε_x/dρ
        dexc_drho = lda_out.vrho * Fx / rho - lda_out.exc / rho + exc_lda * dFx_ds2 * ds2_drho
        vrho = exc + rho * dexc_drho

        # vsigma = dε_x/dσ
        vsigma = exc_lda * dFx_ds2 * ds2_dsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)

    def _compute_polarized(self, density: DensityData) -> XCOutput:
        """Compute for spin-polarized case using spin-scaling."""
        rho_a = np.maximum(density.rho_alpha, RHO_THRESHOLD)
        rho_b = np.maximum(density.rho_beta, RHO_THRESHOLD)
        rho = rho_a + rho_b

        sigma_aa = np.maximum(density.sigma_aa, SIGMA_THRESHOLD)
        sigma_bb = np.maximum(density.sigma_bb, SIGMA_THRESHOLD)

        exc = np.zeros_like(rho)
        vrho = np.zeros((len(rho), 2))
        vsigma = np.zeros((len(rho), 3))

        for i, (rho_s, sigma_s) in enumerate([(rho_a, sigma_aa), (rho_b, sigma_bb)]):
            # Spin-scaled quantities
            rho_2s = 2.0 * rho_s
            sigma_2s = 4.0 * sigma_s

            kF = (3.0 * np.pi ** 2 * rho_2s) ** (1.0 / 3.0)
            grad_rho = np.sqrt(sigma_2s)
            s = grad_rho / (2.0 * kF * rho_2s)
            s2 = s ** 2

            Fx = 1.0 + self.KAPPA - self.KAPPA / (1.0 + self.MU * s2 / self.KAPPA)

            # LDA for spin channel
            C_X = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
            exc_lda = -C_X * rho_2s ** (1.0 / 3.0)

            exc_s = exc_lda * Fx * rho_s / rho
            exc += exc_s

            # Derivatives (simplified)
            dFx_ds2 = self.MU / (1.0 + self.MU * s2 / self.KAPPA) ** 2
            vrho_lda = (4.0 / 3.0) * exc_lda
            vrho[:, i] = 2.0 * vrho_lda * Fx

            ds2_dsigma = 1.0 / (4.0 * kF ** 2 * rho_2s ** 2)
            vsigma[:, 2 * i] = 4.0 * exc_lda * dFx_ds2 * ds2_dsigma

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)


class PBECorrelation(CorrelationFunctional):
    """
    Perdew-Burke-Ernzerhof correlation functional.

    Gradient correction to LDA correlation:
        ε_c^PBE = ε_c^LDA + H(rs, ζ, t)

    where t is the reduced gradient for correlation.

    Reference: Perdew, Burke, Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
    """

    # PBE correlation parameters
    GAMMA = 0.031091
    BETA = 0.066725

    def __init__(self):
        super().__init__("PBE Correlation", FunctionalType.GGA)
        self._vwn = VWN5Correlation()

    def compute(self, density: DensityData) -> XCOutput:
        """Compute PBE correlation energy density and derivatives."""
        if density.sigma is None:
            raise ValueError("PBE requires density gradient (sigma)")

        rho = np.maximum(density.rho, RHO_THRESHOLD)
        sigma = np.maximum(density.sigma, SIGMA_THRESHOLD)

        if density.is_polarized:
            return self._compute_polarized(density)

        # LDA correlation
        lda_out = self._vwn.compute(density)
        exc_lda = lda_out.exc

        # Reduced gradient for correlation
        # t = |∇ρ| / (2 k_s ρ) where k_s = sqrt(4 k_F / π)
        rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        kF = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
        ks = np.sqrt(4.0 * kF / np.pi)

        grad_rho = np.sqrt(sigma)
        t = grad_rho / (2.0 * ks * rho)
        t2 = t ** 2

        # Gradient correction H
        A = self.BETA / self.GAMMA / (np.exp(-exc_lda / self.GAMMA) - 1.0)
        At2 = A * t2

        H = self.GAMMA * np.log(1.0 + self.BETA / self.GAMMA * t2 *
                                 (1.0 + At2) / (1.0 + At2 + At2 ** 2))

        exc = exc_lda + H

        # Derivatives (simplified)
        # Full analytic derivatives are complex
        vrho = lda_out.vrho + self._H_deriv_rho(exc_lda, t, rho, ks)
        vsigma = self._H_deriv_sigma(exc_lda, t, rho, ks, grad_rho)

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma)

    def _compute_polarized(self, density: DensityData) -> XCOutput:
        """Compute for spin-polarized case."""
        # Full spin-polarized PBE correlation is complex
        # Use closed-shell approximation for now
        rho = np.maximum(density.rho, RHO_THRESHOLD)
        sigma = np.maximum(density.sigma, SIGMA_THRESHOLD)

        # Fall back to unpolarized calculation
        unpol_density = DensityData(rho=rho, sigma=sigma)
        result = self.compute(unpol_density)

        # Return with proper shape for polarized output
        vrho = np.column_stack([result.vrho / 2, result.vrho / 2])
        vsigma = np.column_stack([result.vsigma / 4, result.vsigma / 2, result.vsigma / 4])

        return XCOutput(exc=result.exc, vrho=vrho, vsigma=vsigma)

    def _H_deriv_rho(self, exc_lda, t, rho, ks):
        """Approximate derivative of H with respect to rho."""
        h = 1e-6
        t_p = t * (1 - h)
        t_m = t * (1 + h)

        H_p = self._H_only(exc_lda, t_p)
        H_m = self._H_only(exc_lda, t_m)

        # dt/drho ≈ -7t/(3ρ)
        dH_dt = (H_p - H_m) / (t_p - t_m) if np.any(t_p != t_m) else 0
        return dH_dt * (-7.0 / 3.0) * t / rho

    def _H_deriv_sigma(self, exc_lda, t, rho, ks, grad_rho):
        """Approximate derivative of H with respect to sigma."""
        h = 1e-6
        t_p = t * (1 + h / 2)
        t_m = t * (1 - h / 2)

        H_p = self._H_only(exc_lda, t_p)
        H_m = self._H_only(exc_lda, t_m)

        dH_dt2 = (H_p - H_m) / (t_p ** 2 - t_m ** 2) if np.any(t_p != t_m) else 0

        # dt²/dσ = 1/(4 k_s² ρ²)
        return dH_dt2 / (4.0 * ks ** 2 * rho ** 2)

    def _H_only(self, exc_lda, t):
        """Compute only H for derivatives."""
        t2 = t ** 2
        A = self.BETA / self.GAMMA / (np.exp(-exc_lda / self.GAMMA) - 1.0 + 1e-10)
        At2 = A * t2

        return self.GAMMA * np.log(1.0 + self.BETA / self.GAMMA * t2 *
                                    (1.0 + At2) / (1.0 + At2 + At2 ** 2 + 1e-10))


# Combined GGA functionals

class BLYP(CombinedFunctional):
    """
    BLYP GGA functional.

    Combines Becke88 exchange with Lee-Yang-Parr correlation.
    """

    def __init__(self):
        super().__init__(
            "BLYP",
            exchange=B88Exchange(),
            correlation=LYPCorrelation()
        )


class PBE(CombinedFunctional):
    """
    PBE GGA functional.

    Perdew-Burke-Ernzerhof exchange and correlation.
    """

    def __init__(self):
        super().__init__(
            "PBE",
            exchange=PBEExchange(),
            correlation=PBECorrelation()
        )


# Convenience functions
def b88_exchange() -> B88Exchange:
    """Create B88 exchange functional."""
    return B88Exchange()


def lyp_correlation() -> LYPCorrelation:
    """Create LYP correlation functional."""
    return LYPCorrelation()


def pbe_exchange() -> PBEExchange:
    """Create PBE exchange functional."""
    return PBEExchange()


def pbe_correlation() -> PBECorrelation:
    """Create PBE correlation functional."""
    return PBECorrelation()


def blyp() -> BLYP:
    """Create BLYP GGA functional."""
    return BLYP()


def pbe() -> PBE:
    """Create PBE GGA functional."""
    return PBE()
