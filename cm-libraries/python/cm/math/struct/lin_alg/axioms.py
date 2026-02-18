"""Axioms for the linear algebra structure."""

from ..axioms import (
    Axiom, closure, associativity, commutativity,
    distributivity, linearity, identity,
)

LIN_ALG_AXIOMS = [
    closure("add", "tensors"),
    associativity("add"),
    commutativity("add"),
    identity("add", "zero"),

    closure("matmul", "matrices"),
    associativity("matmul"),
    distributivity("matmul", "add"),

    distributivity("mul", "add"),

    linearity("trace"),

    Axiom("det_multiplicative", "det(A @ B) == det(A) * det(B)"),
    Axiom("transpose_reverse", "(A @ B).T == B.T @ A.T"),
]
