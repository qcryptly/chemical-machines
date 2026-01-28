"""
Functional Registry

Provides a central registry for all XC functionals with name aliasing.
"""

from typing import Dict, Optional, Type, Union

from .base import XCFunctional, FunctionalType
from .lda import SlaterExchange, VWN5Correlation, SVWN5
from .gga import (
    B88Exchange, LYPCorrelation, PBEExchange, PBECorrelation,
    BLYP, PBE
)
from .hybrid import B3LYP, PBE0, TPSSh, M06, M062X
from .range_separated import CAMB3LYP, wB97XD, wB97MV


class FunctionalRegistry:
    """
    Central registry for XC functionals.

    Provides lookup by name with automatic aliasing for common
    functional name variations.

    Usage:
        func = FunctionalRegistry.get('B3LYP')
        func = FunctionalRegistry.get('b3lyp')  # Case insensitive
    """

    # Primary functional classes
    _functionals: Dict[str, Type[XCFunctional]] = {
        # LDA
        'SLATER': SlaterExchange,
        'VWN': VWN5Correlation,
        'VWN5': VWN5Correlation,
        'SVWN': SVWN5,
        'SVWN5': SVWN5,
        'LDA': SVWN5,
        'LSDA': SVWN5,

        # GGA Exchange
        'B88': B88Exchange,
        'BECKE': B88Exchange,

        # GGA Correlation
        'LYP': LYPCorrelation,

        # GGA Combined
        'BLYP': BLYP,
        'PBE': PBE,

        # Hybrid
        'B3LYP': B3LYP,
        'PBE0': PBE0,
        'PBE1PBE': PBE0,
        'PBEH': PBE0,
        'TPSSH': TPSSh,
        'M06': M06,
        'M06-2X': M062X,
        'M062X': M062X,

        # Range-separated
        'CAM-B3LYP': CAMB3LYP,
        'CAMB3LYP': CAMB3LYP,
        'WB97X-D': wB97XD,
        'WB97XD': wB97XD,
        'ΩB97X-D': wB97XD,
        'WB97M-V': wB97MV,
        'WB97MV': wB97MV,
    }

    # Name aliases (lowercase -> canonical)
    _aliases: Dict[str, str] = {
        'b3-lyp': 'B3LYP',
        'blyp': 'BLYP',
        'bp86': 'BP86',  # Not yet implemented
        'cam-b3lyp': 'CAM-B3LYP',
        'lda': 'LDA',
        'lsda': 'LSDA',
        'lyp': 'LYP',
        'm06': 'M06',
        'm06-2x': 'M06-2X',
        'm062x': 'M062X',
        'pbe': 'PBE',
        'pbe0': 'PBE0',
        'pbe1pbe': 'PBE1PBE',
        'pbeh': 'PBEH',
        'slater': 'SLATER',
        'svwn': 'SVWN',
        'svwn5': 'SVWN5',
        'tpssh': 'TPSSH',
        'vwn': 'VWN',
        'vwn5': 'VWN5',
        'wb97x-d': 'WB97X-D',
        'wb97xd': 'WB97XD',
        'wb97m-v': 'WB97M-V',
        'wb97mv': 'WB97MV',
        'ωb97x-d': 'WB97X-D',
        'ωb97m-v': 'WB97M-V',
    }

    @classmethod
    def get(cls, name: str) -> XCFunctional:
        """
        Get functional instance by name.

        Args:
            name: Functional name (case-insensitive)

        Returns:
            XCFunctional instance

        Raises:
            ValueError: If functional not found
        """
        # Normalize name
        canonical = cls._normalize_name(name)

        if canonical not in cls._functionals:
            available = sorted(set(cls._functionals.keys()))
            raise ValueError(
                f"Unknown functional: '{name}'. "
                f"Available: {', '.join(available[:20])}..."
            )

        return cls._functionals[canonical]()

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        """Convert name to canonical form."""
        lower = name.lower().strip()

        # Check aliases
        if lower in cls._aliases:
            return cls._aliases[lower]

        # Try uppercase
        upper = name.upper().strip()
        if upper in cls._functionals:
            return upper

        # Try original
        if name in cls._functionals:
            return name

        # Last resort: uppercase
        return upper

    @classmethod
    def list_functionals(cls, func_type: Optional[FunctionalType] = None) -> Dict[str, str]:
        """
        List available functionals.

        Args:
            func_type: Filter by functional type (optional)

        Returns:
            Dict mapping names to descriptions
        """
        result = {}

        for name, func_class in sorted(cls._functionals.items()):
            func = func_class()
            if func_type is None or func.functional_type == func_type:
                result[name] = func.name

        return result

    @classmethod
    def register(cls, name: str, func_class: Type[XCFunctional],
                 aliases: Optional[list] = None):
        """
        Register a new functional.

        Args:
            name: Canonical name
            func_class: Functional class
            aliases: Optional list of aliases
        """
        cls._functionals[name.upper()] = func_class

        if aliases:
            for alias in aliases:
                cls._aliases[alias.lower()] = name.upper()


def get_functional(name: str) -> XCFunctional:
    """
    Get XC functional by name.

    Convenience function for FunctionalRegistry.get().

    Args:
        name: Functional name (case-insensitive)

    Returns:
        XCFunctional instance
    """
    return FunctionalRegistry.get(name)


def list_functionals(func_type: Optional[FunctionalType] = None) -> Dict[str, str]:
    """
    List available functionals.

    Args:
        func_type: Filter by type (optional)

    Returns:
        Dict mapping names to descriptions
    """
    return FunctionalRegistry.list_functionals(func_type)
