"""
Local Density Approximation (LDA) Functionals

Implements:
- Slater exchange (Dirac, 1930)
- VWN5 correlation (Vosko, Wilk, Nusair, 1980)
- SVWN5 combined LDA functional

References:
- Dirac, P.A.M., Proc. Cambridge Phil. Soc. 26, 376 (1930)
- Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980)
"""

import numpy as np
from typing import Tuple

from .base import (
    ExchangeFunctional, CorrelationFunctional, CombinedFunctional,
    DensityData, XCOutput, FunctionalType,
    LDA_X_FACTOR, SPIN_SCALING, RHO_THRESHOLD
)


class SlaterExchange(ExchangeFunctional):
    """
    Slater (Dirac) local exchange functional.

    The exchange energy density is:
        ε_x = -C_x ρ^(1/3)

    where C_x = (3/4)(3/π)^(1/3) ≈ 0.7386

    For spin-polarized:
        ε_x = ε_x[2ρ_α] + ε_x[2ρ_β]
    """

    # Slater exchange constant: (3/4)(3/π)^(1/3)
    C_X = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

    def __init__(self):
        super().__init__("Slater Exchange", FunctionalType.LDA)

    def compute(self, density: DensityData) -> XCOutput:
        """Compute Slater exchange energy density and potential."""
        rho = np.maximum(density.rho, RHO_THRESHOLD)

        if density.is_polarized:
            return self._compute_polarized(density)

        # Closed-shell: factor of 2 for spin
        rho_13 = rho ** (1.0 / 3.0)

        # Energy density per electron: ε_x = -C_x * ρ^(1/3)
        exc = -self.C_X * rho_13

        # Potential: v_x = dε_x/dρ * ρ + ε_x = (4/3) ε_x
        vrho = (4.0 / 3.0) * exc

        return XCOutput(exc=exc, vrho=vrho)

    def _compute_polarized(self, density: DensityData) -> XCOutput:
        """Compute for spin-polarized case."""
        rho_a = np.maximum(density.rho_alpha, RHO_THRESHOLD)
        rho_b = np.maximum(density.rho_beta, RHO_THRESHOLD)
        rho = rho_a + rho_b

        # For spin-polarized: ε_x[ρ_α, ρ_β] = (ε_x[2ρ_α] + ε_x[2ρ_β]) / 2
        # This gives spin-scaling of 2^(1/3)
        rho_a_13 = (2 * rho_a) ** (1.0 / 3.0)
        rho_b_13 = (2 * rho_b) ** (1.0 / 3.0)

        exc_a = -self.C_X * rho_a_13 * rho_a / rho
        exc_b = -self.C_X * rho_b_13 * rho_b / rho
        exc = exc_a + exc_b

        # Potentials
        vrho_a = -(4.0 / 3.0) * self.C_X * rho_a_13
        vrho_b = -(4.0 / 3.0) * self.C_X * rho_b_13

        # Return as (n_points, 2) array
        vrho = np.column_stack([vrho_a, vrho_b])

        return XCOutput(exc=exc, vrho=vrho)


class VWN5Correlation(CorrelationFunctional):
    """
    Vosko-Wilk-Nusair correlation functional (form V).

    VWN5 is the most commonly used form of the VWN correlation.
    It parameterizes the correlation energy of the uniform electron gas.

    The correlation energy is expressed as:
        ε_c = ε_c^P + f(ζ)[ε_c^F - ε_c^P + ε_c^α/f''(0)]

    where ε_c^P and ε_c^F are paramagnetic and ferromagnetic limits,
    and ζ = (ρ_α - ρ_β)/ρ is the spin polarization.

    Reference: Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980)
    """

    # VWN5 parameters for paramagnetic, ferromagnetic, and spin stiffness
    # [A, x0, b, c] for each component
    PARAMS_P = [0.0310907, -0.10498, 3.72744, 12.9352]   # Paramagnetic
    PARAMS_F = [0.01554535, -0.32500, 7.06042, 18.0578]  # Ferromagnetic
    PARAMS_A = [-1.0 / (6.0 * np.pi ** 2), -0.00475840, 1.13107, 13.0045]  # Alpha (spin stiffness)

    def __init__(self):
        super().__init__("VWN5 Correlation", FunctionalType.LDA)

    def compute(self, density: DensityData) -> XCOutput:
        """Compute VWN5 correlation energy density and potential."""
        rho = np.maximum(density.rho, RHO_THRESHOLD)

        if density.is_polarized:
            return self._compute_polarized(density)

        # Closed-shell (ζ = 0)
        rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

        ec_p, dec_p = self._epsilon_c(rs, self.PARAMS_P)

        exc = ec_p
        vrho = ec_p - rs / 3.0 * dec_p

        return XCOutput(exc=exc, vrho=vrho)

    def _compute_polarized(self, density: DensityData) -> XCOutput:
        """Compute for spin-polarized case."""
        rho_a = np.maximum(density.rho_alpha, RHO_THRESHOLD)
        rho_b = np.maximum(density.rho_beta, RHO_THRESHOLD)
        rho = rho_a + rho_b

        rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        zeta = (rho_a - rho_b) / rho
        zeta = np.clip(zeta, -1.0 + 1e-10, 1.0 - 1e-10)

        # Get ε_c components
        ec_p, dec_p = self._epsilon_c(rs, self.PARAMS_P)
        ec_f, dec_f = self._epsilon_c(rs, self.PARAMS_F)
        ec_a, dec_a = self._epsilon_c(rs, self.PARAMS_A)

        # f(ζ) and its derivatives
        fz, dfz = self._spin_interpolation(zeta)

        # f''(0) = 4/(9(2^(1/3) - 1)) ≈ 1.7099
        fpp0 = 4.0 / (9.0 * (2 ** (1.0 / 3.0) - 1.0))

        # Combined correlation
        delta_ec = ec_f - ec_p
        alpha_c = ec_a / fpp0

        exc = ec_p + fz * (delta_ec + alpha_c * (1 - zeta ** 4))

        # Derivatives
        ddelta_ec = dec_f - dec_p
        dalpha_c = dec_a / fpp0

        # dε_c/drs
        dec_drs = dec_p + fz * (ddelta_ec + dalpha_c * (1 - zeta ** 4))

        # dε_c/dζ
        dec_dzeta = dfz * (delta_ec + alpha_c * (1 - zeta ** 4)) - 4 * fz * alpha_c * zeta ** 3

        # Chain rule for spin densities
        # dζ/dρ_α = (1 - ζ)/ρ, dζ/dρ_β = -(1 + ζ)/ρ
        vrho_a = exc - rs / 3.0 * dec_drs + dec_dzeta * (1 - zeta) / rho
        vrho_b = exc - rs / 3.0 * dec_drs - dec_dzeta * (1 + zeta) / rho

        vrho = np.column_stack([vrho_a, vrho_b])

        return XCOutput(exc=exc, vrho=vrho)

    def _epsilon_c(self, rs: np.ndarray,
                   params: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute VWN correlation parameterization.

        ε_c(rs) = A * [ln(x/X(x)) + 2b/Q * arctan(Q/(2x+b))
                      - bx0/X(x0) * (ln((x-x0)²/X(x)) + 2(b+2x0)/Q * arctan(Q/(2x+b)))]

        where x = √rs, X(x) = x² + bx + c, Q = √(4c - b²)

        Returns:
            (ε_c, dε_c/drs)
        """
        A, x0, b, c = params

        x = np.sqrt(rs)
        X_x = x ** 2 + b * x + c
        X_x0 = x0 ** 2 + b * x0 + c
        Q = np.sqrt(4 * c - b ** 2)

        # Main expression
        arctan_term = np.arctan(Q / (2 * x + b))
        log_term = np.log(x / X_x)
        log_term2 = np.log((x - x0) ** 2 / X_x)

        ec = A * (log_term + 2 * b / Q * arctan_term
                  - b * x0 / X_x0 * (log_term2 + 2 * (b + 2 * x0) / Q * arctan_term))

        # Derivative dε_c/drs = (dε_c/dx)(dx/drs) = (dε_c/dx) / (2x)
        # dX/dx = 2x + b
        dX_dx = 2 * x + b

        # d(ln(x/X))/dx = 1/x - dX/dx/X = 1/x - (2x+b)/(x² + bx + c)
        dlog_dx = 1 / x - dX_dx / X_x

        # d(arctan(Q/(2x+b)))/dx = -2Q/((2x+b)² + Q²) = -2Q/(4X)
        darctan_dx = -2 * Q / (4 * X_x)

        # d(ln((x-x0)²/X))/dx = 2/(x-x0) - dX/dx/X
        dlog2_dx = 2 / (x - x0) - dX_dx / X_x

        dec_dx = A * (dlog_dx + 2 * b / Q * darctan_dx
                      - b * x0 / X_x0 * (dlog2_dx + 2 * (b + 2 * x0) / Q * darctan_dx))

        dec_drs = dec_dx / (2 * x)

        return ec, dec_drs

    def _spin_interpolation(self, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spin interpolation function f(ζ) and its derivative.

        f(ζ) = [(1+ζ)^(4/3) + (1-ζ)^(4/3) - 2] / [2(2^(1/3) - 1)]

        Returns:
            (f(ζ), df/dζ)
        """
        denom = 2.0 * (2 ** (1.0 / 3.0) - 1.0)

        onepz = 1.0 + zeta
        onemz = 1.0 - zeta

        onepz_43 = onepz ** (4.0 / 3.0)
        onemz_43 = onemz ** (4.0 / 3.0)

        fz = (onepz_43 + onemz_43 - 2.0) / denom

        # df/dζ = (4/3) * [(1+ζ)^(1/3) - (1-ζ)^(1/3)] / denom
        onepz_13 = onepz ** (1.0 / 3.0)
        onemz_13 = onemz ** (1.0 / 3.0)
        dfz = (4.0 / 3.0) * (onepz_13 - onemz_13) / denom

        return fz, dfz


class SVWN5(CombinedFunctional):
    """
    SVWN5 LDA functional.

    Combines Slater exchange with VWN5 correlation.
    This is the standard LDA functional used in most DFT codes.
    """

    def __init__(self):
        super().__init__(
            "SVWN5",
            exchange=SlaterExchange(),
            correlation=VWN5Correlation()
        )


# Convenience functions
def slater_exchange() -> SlaterExchange:
    """Create Slater exchange functional."""
    return SlaterExchange()


def vwn5_correlation() -> VWN5Correlation:
    """Create VWN5 correlation functional."""
    return VWN5Correlation()


def svwn5() -> SVWN5:
    """Create SVWN5 LDA functional."""
    return SVWN5()
