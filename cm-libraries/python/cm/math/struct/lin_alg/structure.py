from ..base import Structure, Signature
from .axioms import LIN_ALG_AXIOMS

# Build the structure
_signature = Signature()
for _op in LIN_ALG_OPS:
    _signature.add(_op)

STRUCTURE_LINEAR_ALGEBRA = Structure(
    name="linear_algebra",
    signature=_signature,
    axioms=LIN_ALG_AXIOMS,
)
