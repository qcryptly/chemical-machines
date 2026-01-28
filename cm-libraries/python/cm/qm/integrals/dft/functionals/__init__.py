"""
Exchange-Correlation Functionals

Provides a comprehensive library of DFT functionals:
- LDA: SVWN5
- GGA: BLYP, PBE
- Hybrid: B3LYP, PBE0, M06, M06-2X
- Range-separated: CAM-B3LYP, ωB97X-D, ωB97M-V

Usage:
    from cm.qm.integrals.dft.functionals import get_functional
    func = get_functional('B3LYP')
    result = func.compute(density_data)
"""

from .base import (
    FunctionalType,
    DensityData,
    XCOutput,
    XCFunctional,
    ExchangeFunctional,
    CorrelationFunctional,
    CombinedFunctional,
    RHO_THRESHOLD,
    SIGMA_THRESHOLD,
)

from .lda import (
    SlaterExchange,
    VWN5Correlation,
    SVWN5,
    slater_exchange,
    vwn5_correlation,
    svwn5,
)

from .gga import (
    B88Exchange,
    LYPCorrelation,
    PBEExchange,
    PBECorrelation,
    BLYP,
    PBE,
    b88_exchange,
    lyp_correlation,
    pbe_exchange,
    pbe_correlation,
    blyp,
    pbe,
)

from .hybrid import (
    B3LYP,
    PBE0,
    TPSSh,
    M06,
    M062X,
    b3lyp,
    pbe0,
    tpssh,
    m06,
    m06_2x,
)

from .range_separated import (
    CAMB3LYP,
    wB97XD,
    wB97MV,
    cam_b3lyp,
    wb97xd,
    wb97mv,
)

from .registry import (
    FunctionalRegistry,
    get_functional,
    list_functionals,
)

__all__ = [
    # Base classes
    'FunctionalType',
    'DensityData',
    'XCOutput',
    'XCFunctional',
    'ExchangeFunctional',
    'CorrelationFunctional',
    'CombinedFunctional',
    'RHO_THRESHOLD',
    'SIGMA_THRESHOLD',

    # LDA
    'SlaterExchange',
    'VWN5Correlation',
    'SVWN5',
    'slater_exchange',
    'vwn5_correlation',
    'svwn5',

    # GGA
    'B88Exchange',
    'LYPCorrelation',
    'PBEExchange',
    'PBECorrelation',
    'BLYP',
    'PBE',
    'b88_exchange',
    'lyp_correlation',
    'pbe_exchange',
    'pbe_correlation',
    'blyp',
    'pbe',

    # Hybrid
    'B3LYP',
    'PBE0',
    'TPSSh',
    'M06',
    'M062X',
    'b3lyp',
    'pbe0',
    'tpssh',
    'm06',
    'm06_2x',

    # Range-separated
    'CAMB3LYP',
    'wB97XD',
    'wB97MV',
    'cam_b3lyp',
    'wb97xd',
    'wb97mv',

    # Registry
    'FunctionalRegistry',
    'get_functional',
    'list_functionals',
]
