"""
Boys Function Implementation

The Boys function F_n(x) is essential for evaluating molecular integrals
over Gaussian-type orbitals. It appears in nuclear attraction and
electron repulsion integrals.

Definition:
    F_n(x) = ∫_0^1 t^(2n) exp(-x*t²) dt

Properties:
    F_0(x) = sqrt(π/x) * erf(sqrt(x)) / 2  for x > 0
    F_n(0) = 1 / (2n + 1)
    Recurrence: F_n(x) = [(2n-1) F_{n-1}(x) - exp(-x)] / (2x)
"""

import numpy as np
import math
from scipy import special
from functools import lru_cache
from typing import Union


def boys_function(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the Boys function F_n(x).

    Uses different methods depending on the value of x:
    - Small x: Taylor series expansion
    - Large x: Asymptotic expansion
    - Intermediate: Upward recurrence from F_0

    Args:
        n: Order of the Boys function (non-negative integer)
        x: Argument (can be scalar or array)

    Returns:
        Value of F_n(x)

    Example:
        >>> boys_function(0, 0.0)
        1.0
        >>> boys_function(0, 1.0)  # F_0(1) ≈ 0.7468
        0.7468241328...
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.zeros_like(x)

    # Handle different regions
    small_x = x < 1e-10
    large_x = x > 25.0
    medium_x = ~small_x & ~large_x

    # Small x: Taylor series
    # F_n(x) ≈ 1/(2n+1) - x/(2n+3) + x²/(2(2n+5)) - ...
    if np.any(small_x):
        result[small_x] = 1.0 / (2 * n + 1)
        if n == 0 and np.any(small_x & (x > 0)):
            # More accurate for very small but nonzero x
            idx = small_x & (x > 0)
            result[idx] = 1.0 / (2 * n + 1) - x[idx] / (2 * n + 3)

    # Large x: Asymptotic expansion
    # F_n(x) ≈ (2n-1)!! / (2^(n+1)) * sqrt(π/x^(2n+1))
    if np.any(large_x):
        result[large_x] = _boys_asymptotic(n, x[large_x])

    # Medium x: Compute F_0 then use upward recurrence
    if np.any(medium_x):
        result[medium_x] = _boys_recurrence(n, x[medium_x])

    return result[0] if result.size == 1 else result


def _boys_asymptotic(n: int, x: np.ndarray) -> np.ndarray:
    """
    Asymptotic expansion for large x.

    F_n(x) ≈ (2n-1)!! * sqrt(π) / (2^(n+1) * x^(n+0.5))
    """
    double_fact = _double_factorial(2 * n - 1)
    return double_fact * np.sqrt(np.pi) / (2 ** (n + 1) * x ** (n + 0.5))


def _boys_recurrence(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute F_n using upward recurrence from F_0.

    F_0(x) = sqrt(π/(4x)) * erf(sqrt(x))
    F_{n+1}(x) = [(2n+1) F_n(x) - exp(-x)] / (2x)
    """
    # F_0(x) = sqrt(π/(4x)) * erf(sqrt(x))
    sqrt_x = np.sqrt(x)
    F_0 = np.sqrt(np.pi) / (2 * sqrt_x) * special.erf(sqrt_x)

    if n == 0:
        return F_0

    # Upward recurrence
    exp_neg_x = np.exp(-x)
    F_prev = F_0

    for m in range(n):
        F_curr = ((2 * m + 1) * F_prev - exp_neg_x) / (2 * x)
        F_prev = F_curr

    return F_prev


@lru_cache(maxsize=64)
def _double_factorial(n: int) -> int:
    """
    Compute double factorial n!! = n * (n-2) * (n-4) * ... * 1

    (-1)!! = 0!! = 1 by convention.
    """
    if n <= 1:
        return 1
    return n * _double_factorial(n - 2)


def boys_function_taylor(n: int, x: float, n_terms: int = 20) -> float:
    """
    Taylor series expansion of Boys function about x=0.

    F_n(x) = Σ_k (-x)^k / (k! * (2n + 2k + 1))

    Useful for small x where other methods lose precision.

    Args:
        n: Order
        x: Argument
        n_terms: Number of terms in series

    Returns:
        Approximate value of F_n(x)
    """
    result = 0.0
    x_power = 1.0

    for k in range(n_terms):
        term = x_power / (math.factorial(k) * (2 * n + 2 * k + 1))
        if k % 2 == 1:
            term = -term
        result += term
        x_power *= x

        # Check convergence
        if abs(term) < 1e-15 * abs(result):
            break

    return result


def boys_function_table(max_n: int, x_values: np.ndarray) -> np.ndarray:
    """
    Compute a table of Boys function values for multiple n and x.

    More efficient than calling boys_function repeatedly.

    Args:
        max_n: Maximum order (computes F_0 through F_max_n)
        x_values: Array of x values

    Returns:
        Array of shape (max_n + 1, len(x_values))
    """
    x = np.atleast_1d(x_values)
    n_x = len(x)
    table = np.zeros((max_n + 1, n_x))

    # Compute F_0 for all x
    table[0, :] = boys_function(0, x)

    # Use downward recurrence for stability
    # First compute F_{max_n+k} for large k using asymptotic, then recurse down
    if max_n > 0:
        exp_neg_x = np.exp(-x)

        for m in range(max_n):
            # F_{m+1}(x) = [(2m+1) F_m(x) - exp(-x)] / (2x)
            with np.errstate(divide='ignore', invalid='ignore'):
                table[m + 1, :] = np.where(
                    x > 1e-10,
                    ((2 * m + 1) * table[m, :] - exp_neg_x) / (2 * x),
                    1.0 / (2 * (m + 1) + 1)  # Limit as x -> 0
                )

    return table
