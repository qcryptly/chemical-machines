"""Constraint system for lazy expression validation."""

from typing import List, Dict, Any, Callable

__all__ = [
    'ConstraintError', 'Constraint', 'ConstraintSet',
    'require_rank', 'require_square', 'require_nonsingular',
]


class ConstraintError(Exception):
    """Raised when a constraint is violated."""
    pass


class Constraint:
    """A condition that must hold for an expression to be valid."""

    def __init__(self, description, check):
        self.description = description
        self.check = check

    def is_satisfied(self, metadata):
        try:
            return self.check(metadata)
        except (KeyError, TypeError, IndexError):
            return False

    def __repr__(self):
        return f"Constraint({self.description!r})"


class ConstraintSet:
    """Collection of constraints on an expression."""

    def __init__(self):
        self.constraints: List[Constraint] = []

    def add(self, constraint):
        self.constraints.append(constraint)

    def is_resolved(self, metadata):
        return all(c.is_satisfied(metadata) for c in self.constraints)

    def resolve(self, metadata):
        for c in self.constraints:
            if not c.is_satisfied(metadata):
                raise ConstraintError(
                    f"Constraint violated: {c.description}\n"
                    f"Metadata: {metadata}"
                )

    def __len__(self):
        return len(self.constraints)

    def __repr__(self):
        return f"ConstraintSet({len(self.constraints)} constraints)"


def require_rank(rank):
    return Constraint(
        f"rank == {rank}",
        lambda m: len(m.get('shape', ())) == rank
    )


def require_square():
    def check(m):
        shape = m.get('shape')
        return shape is not None and len(shape) == 2 and shape[0] == shape[1]
    return Constraint("must be square matrix", check)


def require_nonsingular():
    return Constraint("must be nonsingular",
                      lambda m: m.get('nonsingular', True))
