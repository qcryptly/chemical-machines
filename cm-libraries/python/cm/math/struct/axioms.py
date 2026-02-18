"""Predefined axiom constructors for mathematical structures."""

from typing import Callable, Optional

__all__ = [
    'Axiom',
    'closure', 'associativity', 'commutativity', 'identity', 'inverse',
    'distributivity', 'linearity', 'bilinearity',
    'jacobi_identity', 'anticommutativity',
]


class Axiom:
    """A machine-readable axiom (equational law)."""

    def __init__(self, name, description, check=None):
        self.name = name
        self.description = description
        self.check = check

    def __repr__(self):
        return f"Axiom({self.name!r})"


def closure(op, carrier):
    return Axiom("closure", f"{op} on {carrier} is closed")


def associativity(op):
    return Axiom("associativity", f"{op}({op}(a,b),c) == {op}(a,{op}(b,c))")


def commutativity(op):
    return Axiom("commutativity", f"{op}(a,b) == {op}(b,a)")


def identity(op, element):
    return Axiom("identity", f"{op}(a,{element}) == a")


def inverse(op, inv_op, identity_elem):
    return Axiom("inverse", f"{op}(a, {inv_op}(a)) == {identity_elem}")


def distributivity(op1, op2):
    return Axiom("distributivity",
                 f"{op1}(a,{op2}(b,c)) == {op2}({op1}(a,b),{op1}(a,c))")


def linearity(op, scalar_field="scalars"):
    return Axiom("linearity",
                 f"{op}(a*x + b*y) == a*{op}(x) + b*{op}(y)")


def bilinearity(op, scalar_field="scalars"):
    return Axiom("bilinearity", f"{op} is linear in both arguments")


def jacobi_identity(bracket):
    return Axiom("jacobi_identity",
                 f"[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] == 0")


def anticommutativity(op):
    return Axiom("anticommutativity", f"{op}(a,b) == -{op}(b,a)")
