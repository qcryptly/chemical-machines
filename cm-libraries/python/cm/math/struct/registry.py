"""Custom structure definition via struct.define()."""

from .base import Structure, Signature, Operation
from .axioms import Axiom

__all__ = ['define']


def define(name, carriers=None, ops=None, axioms=None):
    """
    Define a custom mathematical structure.

    Args:
        name: Structure name
        carriers: Set of carrier set names
        ops: Dict mapping op names to {"arity": int, "signature": tuple}
        axioms: List of Axiom objects

    Returns:
        A new Structure instance
    """
    sig = Signature()
    if ops:
        for op_name, op_spec in ops.items():
            arity = op_spec.get("arity", 2)
            signature = op_spec.get("signature")
            sig.add(Operation(op_name, arity, signature))

    return Structure(name, sig, axioms or [])
