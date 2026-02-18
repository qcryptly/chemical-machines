"""
Scalar fields and numeric dtypes for cm.math.

Defines abstract fields (Reals, Complex, Rationals, Integers)
and concrete compute dtypes (float32, float64, complexF32, etc.).
"""

import numpy as np

__all__ = [
    'Field', 'Ring', 'Dtype',
    'Reals', 'Complex', 'Rationals', 'Integers',
    'float32', 'float64', 'complexF32', 'complexF64', 'int32', 'int64',
]


class Field:
    """An abstract mathematical field."""

    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol

    def to_latex(self):
        return self.symbol

    def __repr__(self):
        return f"Field({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, Field) and self.name == other.name

    def __hash__(self):
        return hash(('Field', self.name))


class Ring(Field):
    """An abstract mathematical ring."""

    def __repr__(self):
        return f"Ring({self.name!r})"


class Dtype:
    """A concrete compute dtype mapping to NumPy and PyTorch types."""

    def __init__(self, name, field, torch_dtype_name, np_dtype):
        self.name = name
        self.field = field
        self._torch_dtype_name = torch_dtype_name
        self.np_dtype = np_dtype

    @property
    def torch_dtype(self):
        import torch
        return getattr(torch, self._torch_dtype_name)

    def __repr__(self):
        return f"Dtype({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, Dtype) and self.name == other.name

    def __hash__(self):
        return hash(('Dtype', self.name))


# Abstract fields
Reals = Field("Reals", r"\mathbb{R}")
Complex = Field("Complex", r"\mathbb{C}")
Rationals = Field("Rationals", r"\mathbb{Q}")
Integers = Ring("Integers", r"\mathbb{Z}")

# Concrete compute dtypes
float32 = Dtype("float32", Reals, "float32", np.float32)
float64 = Dtype("float64", Reals, "float64", np.float64)
complexF32 = Dtype("complexF32", Complex, "complex64", np.complex64)
complexF64 = Dtype("complexF64", Complex, "complex128", np.complex128)
int32 = Dtype("int32", Integers, "int32", np.int32)
int64 = Dtype("int64", Integers, "int64", np.int64)
