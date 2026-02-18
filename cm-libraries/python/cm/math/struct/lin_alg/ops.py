"""
Linear algebra operations and backend registrations.

Operations: matmul, det, trace, transpose, inverse, eigenvalues, svd,
            add, sub, mul, div, pow, neg
"""

from __future__ import annotations
from typing import List
from ..base import Operation


# ---- Shape inference functions ----

def _matmul_shape(left_meta, right_meta):
    ls = left_meta.get('shape')
    rs = right_meta.get('shape')
    result = {'is_tensor': True}
    if ls is not None and rs is not None:
        if len(ls) >= 2 and len(rs) >= 2:
            result['shape'] = ls[:-1] + rs[1:]
        elif len(ls) == 1 and len(rs) == 2:
            result['shape'] = (rs[1],)
        elif len(ls) == 2 and len(rs) == 1:
            result['shape'] = (ls[0],)
    return result


def _scalar_result(operand_meta):
    return {'shape': (), 'is_scalar': True}


def _transpose_shape(operand_meta):
    s = operand_meta.get('shape')
    result = {'is_tensor': True}
    if s is not None and len(s) == 2:
        result['shape'] = (s[1], s[0])
    return result


def _same_shape(operand_meta):
    result = {}
    s = operand_meta.get('shape')
    if s is not None:
        result['shape'] = s
    result['is_tensor'] = operand_meta.get('is_tensor', False)
    return result


def _eigenvalues_shape(operand_meta):
    s = operand_meta.get('shape')
    result = {}
    if s is not None and len(s) >= 1:
        result['shape'] = (s[-1],)
    return result


def _add_sub_shape(left_meta, right_meta):
    ls = left_meta.get('shape')
    rs = right_meta.get('shape')
    result = {}
    if ls is not None:
        result['shape'] = ls
    elif rs is not None:
        result['shape'] = rs
    result['is_tensor'] = (left_meta.get('is_tensor', False)
                           or right_meta.get('is_tensor', False))
    return result


def _mul_div_shape(left_meta, right_meta):
    ls = left_meta.get('shape', ())
    rs = right_meta.get('shape', ())
    result = {}
    if ls == ():
        result['shape'] = rs
    elif rs == ():
        result['shape'] = ls
    else:
        result['shape'] = ls
    result['is_tensor'] = (left_meta.get('is_tensor', False)
                           or right_meta.get('is_tensor', False))
    return result


# ---- Operation definitions ----

LIN_ALG_OPS: List[Operation] = [
    Operation("matmul", 2, result_shape_fn=_matmul_shape),
    Operation("det", 1, result_shape_fn=_scalar_result, latex_name="det"),
    Operation("trace", 1, result_shape_fn=_scalar_result, latex_name="tr"),
    Operation("transpose", 1, result_shape_fn=_transpose_shape),
    Operation("inverse", 1, result_shape_fn=_same_shape),
    Operation("eigenvalues", 1, result_shape_fn=_eigenvalues_shape),
    Operation("norm", 1, result_shape_fn=_scalar_result),
    Operation("svd", 1),
    # Arithmetic
    Operation("add", 2, result_shape_fn=_add_sub_shape),
    Operation("sub", 2, result_shape_fn=_add_sub_shape),
    Operation("mul", 2, result_shape_fn=_mul_div_shape),
    Operation("div", 2, result_shape_fn=_mul_div_shape),
    Operation("pow", 2),
    Operation("neg", 1),
]


def register_lin_alg_ops():
    """Register all lin_alg operations with all three backends."""
    _register_eager()
    _register_torch()
    _register_latex()


def _register_eager():
    from ..backends.eager_be import eager_backend
    import numpy as np

    S = "linear_algebra"
    eager_backend.register(S, "matmul", lambda a, b: np.matmul(a, b))
    eager_backend.register(S, "det", lambda a: np.linalg.det(a))
    eager_backend.register(S, "trace", lambda a: np.trace(a))
    eager_backend.register(S, "transpose", lambda a: np.transpose(a))
    eager_backend.register(S, "inverse", lambda a: np.linalg.inv(a))
    eager_backend.register(S, "eigenvalues", lambda a: np.linalg.eigvalsh(a))
    eager_backend.register(S, "norm", lambda a: np.linalg.norm(a))
    eager_backend.register(S, "svd", lambda a: np.linalg.svd(a))


def _register_torch():
    from ..backends.torch_be import torch_backend

    S = "linear_algebra"

    def _torch_matmul(a, b):
        import torch
        return torch.matmul(a, b)

    def _torch_det(a):
        import torch
        return torch.linalg.det(a)

    def _torch_trace(a):
        import torch
        return torch.trace(a)

    def _torch_transpose(a):
        import torch
        return torch.transpose(a, 0, 1)

    def _torch_inverse(a):
        import torch
        return torch.linalg.inv(a)

    def _torch_eigenvalues(a):
        import torch
        return torch.linalg.eigvalsh(a)

    def _torch_norm(a):
        import torch
        return torch.linalg.norm(a)

    def _torch_svd(a):
        import torch
        return torch.linalg.svd(a)

    torch_backend.register(S, "matmul", _torch_matmul)
    torch_backend.register(S, "det", _torch_det)
    torch_backend.register(S, "trace", _torch_trace)
    torch_backend.register(S, "transpose", _torch_transpose)
    torch_backend.register(S, "inverse", _torch_inverse)
    torch_backend.register(S, "eigenvalues", _torch_eigenvalues)
    torch_backend.register(S, "norm", _torch_norm)
    torch_backend.register(S, "svd", _torch_svd)


def _register_latex():
    from ..backends.latex_be import latex_backend

    S = "linear_algebra"

    def _latex_matmul(left, right, **kw):
        return f"{left} {right}"

    def _latex_det(child, **kw):
        return rf"\det\left({child}\right)"

    def _latex_trace(child, **kw):
        return rf"\mathrm{{tr}}\left({child}\right)"

    def _latex_transpose(child, **kw):
        return f"{child}^{{\\top}}"

    def _latex_inverse(child, **kw):
        return f"{child}^{{-1}}"

    def _latex_eigenvalues(child, **kw):
        return rf"\mathrm{{eig}}\left({child}\right)"

    def _latex_norm(child, **kw):
        return rf"\left\|{child}\right\|"

    def _latex_svd(child, **kw):
        return rf"\mathrm{{svd}}\left({child}\right)"

    latex_backend.register(S, "matmul", _latex_matmul)
    latex_backend.register(S, "det", _latex_det)
    latex_backend.register(S, "trace", _latex_trace)
    latex_backend.register(S, "transpose", _latex_transpose)
    latex_backend.register(S, "inverse", _latex_inverse)
    latex_backend.register(S, "eigenvalues", _latex_eigenvalues)
    latex_backend.register(S, "norm", _latex_norm)
    latex_backend.register(S, "svd", _latex_svd)
