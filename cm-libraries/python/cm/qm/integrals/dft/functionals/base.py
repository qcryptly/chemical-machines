"""
Exchange-Correlation Functional Base Classes

Provides abstract base classes and data structures for DFT functionals.

Functional types:
- LDA: Depends only on density ρ
- GGA: Depends on ρ and |∇ρ|²
- meta-GGA: Also depends on kinetic energy density τ
- Hybrid: Mix of DFT exchange with exact (HF) exchange
- Range-separated: Different treatment for short/long range
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class FunctionalType(Enum):
    """Classification of XC functional types."""
    LDA = "lda"
    GGA = "gga"
    META_GGA = "meta-gga"
    HYBRID = "hybrid"
    RANGE_SEPARATED = "range-separated"


@dataclass
class DensityData:
    """
    Electron density and its derivatives on a grid.

    For closed-shell calculations:
        rho: total density ρ(r)
        sigma: |∇ρ|² (for GGA)
        tau: kinetic energy density (for meta-GGA)

    For open-shell calculations:
        rho_alpha, rho_beta: spin densities
        sigma_aa, sigma_ab, sigma_bb: gradient products
        tau_alpha, tau_beta: spin kinetic energy densities

    Attributes:
        rho: Total electron density (n_points,)
        sigma: Contracted gradient |∇ρ|² (n_points,) for GGA
        tau: Kinetic energy density (n_points,) for meta-GGA
        laplacian: Laplacian of density ∇²ρ (n_points,) optional

        # Spin-polarized quantities (None for closed-shell)
        rho_alpha: Alpha spin density
        rho_beta: Beta spin density
        sigma_aa: |∇ρ_α|²
        sigma_ab: ∇ρ_α · ∇ρ_β
        sigma_bb: |∇ρ_β|²
        tau_alpha: Alpha kinetic energy density
        tau_beta: Beta kinetic energy density
    """
    rho: np.ndarray
    sigma: Optional[np.ndarray] = None
    tau: Optional[np.ndarray] = None
    laplacian: Optional[np.ndarray] = None

    # Spin-polarized
    rho_alpha: Optional[np.ndarray] = None
    rho_beta: Optional[np.ndarray] = None
    sigma_aa: Optional[np.ndarray] = None
    sigma_ab: Optional[np.ndarray] = None
    sigma_bb: Optional[np.ndarray] = None
    tau_alpha: Optional[np.ndarray] = None
    tau_beta: Optional[np.ndarray] = None

    @property
    def is_polarized(self) -> bool:
        """Check if this is spin-polarized density data."""
        return self.rho_alpha is not None

    @property
    def n_points(self) -> int:
        """Number of grid points."""
        return len(self.rho)

    @classmethod
    def from_spin_densities(cls, rho_alpha: np.ndarray, rho_beta: np.ndarray,
                            grad_alpha: Optional[np.ndarray] = None,
                            grad_beta: Optional[np.ndarray] = None,
                            tau_alpha: Optional[np.ndarray] = None,
                            tau_beta: Optional[np.ndarray] = None) -> 'DensityData':
        """
        Create DensityData from spin densities.

        Args:
            rho_alpha: Alpha density (n_points,)
            rho_beta: Beta density (n_points,)
            grad_alpha: Alpha density gradient (n_points, 3)
            grad_beta: Beta density gradient (n_points, 3)
            tau_alpha: Alpha kinetic energy density
            tau_beta: Beta kinetic energy density
        """
        rho = rho_alpha + rho_beta

        sigma = None
        sigma_aa = sigma_ab = sigma_bb = None

        if grad_alpha is not None and grad_beta is not None:
            sigma_aa = np.sum(grad_alpha * grad_alpha, axis=1)
            sigma_bb = np.sum(grad_beta * grad_beta, axis=1)
            sigma_ab = np.sum(grad_alpha * grad_beta, axis=1)
            sigma = sigma_aa + 2 * sigma_ab + sigma_bb

        tau = None
        if tau_alpha is not None and tau_beta is not None:
            tau = tau_alpha + tau_beta

        return cls(
            rho=rho,
            sigma=sigma,
            tau=tau,
            rho_alpha=rho_alpha,
            rho_beta=rho_beta,
            sigma_aa=sigma_aa,
            sigma_ab=sigma_ab,
            sigma_bb=sigma_bb,
            tau_alpha=tau_alpha,
            tau_beta=tau_beta
        )


@dataclass
class XCOutput:
    """
    Output from XC functional evaluation.

    Contains energy density and functional derivatives needed for
    constructing the XC potential matrix.

    Attributes:
        exc: Exchange-correlation energy density per electron (n_points,)

        # First derivatives (for potential)
        vrho: dE_xc/dρ (n_points,) or (n_points, 2) for spin
        vsigma: dE_xc/d|∇ρ|² (n_points,) or (n_points, 3) for spin
        vtau: dE_xc/dτ (n_points,) or (n_points, 2) for spin

        # Second derivatives (for kernel/Hessian) - optional
        v2rho2: d²E_xc/dρ²
        v2rhosigma: d²E_xc/dρd|∇ρ|²
        v2sigma2: d²E_xc/d|∇ρ|²²
    """
    exc: np.ndarray

    # First derivatives
    vrho: np.ndarray
    vsigma: Optional[np.ndarray] = None
    vtau: Optional[np.ndarray] = None

    # Second derivatives (for response properties)
    v2rho2: Optional[np.ndarray] = None
    v2rhosigma: Optional[np.ndarray] = None
    v2sigma2: Optional[np.ndarray] = None

    @property
    def n_points(self) -> int:
        return len(self.exc)


class XCFunctional(ABC):
    """
    Abstract base class for exchange-correlation functionals.

    All XC functionals implement this interface, providing methods
    to compute the energy density and its derivatives.
    """

    def __init__(self, name: str, functional_type: FunctionalType):
        """
        Initialize functional.

        Args:
            name: Human-readable name (e.g., "Slater Exchange")
            functional_type: LDA, GGA, meta-GGA, etc.
        """
        self.name = name
        self.functional_type = functional_type

    @abstractmethod
    def compute(self, density: DensityData) -> XCOutput:
        """
        Compute XC energy density and derivatives.

        Args:
            density: Electron density data on grid

        Returns:
            XCOutput with exc, vrho, and optionally vsigma/vtau
        """
        pass

    @property
    def needs_gradient(self) -> bool:
        """Whether this functional needs density gradient."""
        return self.functional_type in (
            FunctionalType.GGA,
            FunctionalType.META_GGA,
            FunctionalType.HYBRID,
            FunctionalType.RANGE_SEPARATED
        )

    @property
    def needs_tau(self) -> bool:
        """Whether this functional needs kinetic energy density."""
        return self.functional_type == FunctionalType.META_GGA

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}')"


class ExchangeFunctional(XCFunctional):
    """Base class for pure exchange functionals."""
    pass


class CorrelationFunctional(XCFunctional):
    """Base class for pure correlation functionals."""
    pass


class CombinedFunctional(XCFunctional):
    """
    Combined exchange-correlation functional.

    Combines separate exchange and correlation functionals,
    optionally with exact exchange mixing.
    """

    def __init__(self, name: str,
                 exchange: Optional[ExchangeFunctional] = None,
                 correlation: Optional[CorrelationFunctional] = None,
                 exchange_weight: float = 1.0,
                 exact_exchange: float = 0.0):
        """
        Initialize combined functional.

        Args:
            name: Name of combined functional
            exchange: Exchange functional
            correlation: Correlation functional
            exchange_weight: Weight for DFT exchange (1 - exact_exchange typically)
            exact_exchange: Fraction of exact (HF) exchange
        """
        # Determine type based on components
        if exchange is not None:
            func_type = exchange.functional_type
        elif correlation is not None:
            func_type = correlation.functional_type
        else:
            func_type = FunctionalType.LDA

        if exact_exchange > 0:
            func_type = FunctionalType.HYBRID

        super().__init__(name, func_type)

        self.exchange = exchange
        self.correlation = correlation
        self.exchange_weight = exchange_weight
        self.exact_exchange = exact_exchange

    def compute(self, density: DensityData) -> XCOutput:
        """Compute combined XC energy density and derivatives."""
        exc = np.zeros(density.n_points)
        vrho = np.zeros_like(exc)
        vsigma = None
        vtau = None

        if self.exchange is not None:
            x_out = self.exchange.compute(density)
            exc += self.exchange_weight * x_out.exc
            vrho += self.exchange_weight * x_out.vrho
            if x_out.vsigma is not None:
                if vsigma is None:
                    vsigma = np.zeros_like(x_out.vsigma)
                vsigma += self.exchange_weight * x_out.vsigma
            if x_out.vtau is not None:
                if vtau is None:
                    vtau = np.zeros_like(x_out.vtau)
                vtau += self.exchange_weight * x_out.vtau

        if self.correlation is not None:
            c_out = self.correlation.compute(density)
            exc += c_out.exc
            vrho += c_out.vrho
            if c_out.vsigma is not None:
                if vsigma is None:
                    vsigma = np.zeros_like(c_out.vsigma)
                vsigma += c_out.vsigma
            if c_out.vtau is not None:
                if vtau is None:
                    vtau = np.zeros_like(c_out.vtau)
                vtau += c_out.vtau

        return XCOutput(exc=exc, vrho=vrho, vsigma=vsigma, vtau=vtau)


# Physical constants used in functionals
# Note: All quantities in atomic units unless otherwise specified

# (3/4π)^(1/3) - appears in LDA exchange
LDA_X_FACTOR = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

# 2^(1/3) - spin scaling factor
SPIN_SCALING = 2.0 ** (1.0 / 3.0)

# Threshold for negligible density (avoid numerical issues)
RHO_THRESHOLD = 1e-15
SIGMA_THRESHOLD = 1e-20
