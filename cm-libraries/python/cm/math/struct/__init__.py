"""
Structure namespace for cm.math.

Mathematical structures: sets equipped with operations and axioms.

Available structures:
    struct.lin_alg  - Linear algebra (tensors, vectors, matrices, scalars)

Custom structures:
    struct.define(name, carriers, ops, axioms) -> Structure
"""

from . import lin_alg
from . import spec_func
from .registry import define
from . import base
from . import axioms
from .constraints import ConstraintError

__all__ = ['lin_alg', 'spec_func', 'define', 'base', 'axioms', 'ConstraintError']
