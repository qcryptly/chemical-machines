"""
Geometry Optimization Module

Provides geometry optimization algorithms:
- BFGS quasi-Newton optimizer
- Steepest descent
- Conjugate gradient
- Internal coordinate transformations

Supports:
- Minima optimization
- Transition state search (eigenvector following)
- Constrained optimization
"""

from .optimizer import (
    GeometryOptimizer,
    OptimizationResult,
    optimize_geometry,
)
from .internal_coords import (
    InternalCoordinates,
    bond_length,
    bond_angle,
    dihedral_angle,
)
from .ts_search import (
    TSResult,
    TransitionStateOptimizer,
    find_transition_state,
)

__all__ = [
    # Optimizer
    'GeometryOptimizer',
    'OptimizationResult',
    'optimize_geometry',
    # Internal coordinates
    'InternalCoordinates',
    'bond_length',
    'bond_angle',
    'dihedral_angle',
    # TS search
    'TSResult',
    'TransitionStateOptimizer',
    'find_transition_state',
]
