# %% Cell 1 - Basic Variable and Expression Building
from cm.symbols import Math, inf

# Create variables
x = Math.var("x")
y = Math.var("y")

# Auto-named variable
z = Math.var()  # x_0

# Greek letter variable
t = Math.var("tau")

# Build expressions with operators
expr = x**2 + y**2
expr.render()

# %% Cell 2 - More Complex Expressions
from cm.symbols import Math

x = Math.var("x")
y = Math.var("y")

# Division renders as fraction
frac_expr = (x + 1) / (y - 1)
frac_expr.render()

# Multiplication
mult_expr = 2 * x * y
mult_expr.render()

# Power with complex base
power_expr = (x + y) ** 2
power_expr.render()

# %% Cell 3 - Mathematical Functions
from cm.symbols import Math

x = Math.var("x")

# Trig functions
sin_expr = Math.sin(x)
sin_expr.render()

cos_expr = Math.cos(2 * x)
cos_expr.render()

# Exponential and log
exp_expr = Math.exp(-x**2)
exp_expr.render()

log_expr = Math.log(x + 1)
log_expr.render()

# Square root
sqrt_expr = Math.sqrt(x**2 + 1)
sqrt_expr.render()

# %% Cell 4 - Indefinite Integration
from cm.symbols import Math

x = Math.var("x")

# Simple indefinite integral
expr = x**2
integral = expr.integrate(x)
integral.render()

# After simplification, get the antiderivative
result = integral.simplify()
result.render()

# %% Cell 5 - Definite Integration with Numeric Bounds
from cm.symbols import Math

x = Math.var("x")
y = Math.var("y")

# Single variable definite integral
expr = x**2
integral = expr.integrate(x, bounds=[0, 1])
integral.render()

# Evaluate it (should be 1/3)
from cm.views import log
value = integral.evaluate()
log(f"∫₀¹ x² dx = {value.item():.6f}", level="info")

# Double integral
expr2 = x**2 + y**2
double_int = expr2.integrate(x, bounds=[0, 1]).integrate(y, bounds=[0, 1])
double_int.render()

value2 = double_int.evaluate()
log(f"∫∫ (x² + y²) dx dy = {value2.item():.6f}", level="info")

# %% Cell 6 - Symbolic Bounds
from cm.symbols import Math

x = Math.var("x")
t = Math.var("tau")

# Integration with symbolic upper bound
expr = x**2
symbolic_int = expr.integrate(x, bounds=[0, t])
symbolic_int.render()

# Simplify to get closed form
closed_form = symbolic_int.simplify()
closed_form.render()

# %% Cell 7 - Infinite Bounds
from cm.symbols import Math, inf

x = Math.var("x")

# Gaussian integral setup
gaussian = Math.exp(-x**2)
gaussian_int = gaussian.integrate(x, bounds=[-inf, inf])
gaussian_int.render()

# Simplify (SymPy knows this is sqrt(pi))
result = gaussian_int.simplify()
result.render()

# %% Cell 8 - Derivatives
from cm.symbols import Math

x = Math.var("x")

# First derivative
expr = x**3 + 2*x**2 + x
deriv = expr.diff(x)
deriv.render()

# Simplify to compute
result = deriv.simplify()
result.render()

# Second derivative
deriv2 = expr.diff(x, 2)
deriv2.render()

result2 = deriv2.simplify()
result2.render()

# %% Cell 9 - Evaluation with Variables
from cm.symbols import Math
from cm.views import log

x = Math.var("x")
y = Math.var("y")

expr = x**2 + y**2

# Evaluate at specific point
result = expr.evaluate(x=3, y=4)
log(f"x² + y² at (3,4) = {result.item()}", level="success")

# %% Cell 10 - Error Handling for Indefinite Integrals
from cm.symbols import Math, EvaluationError
from cm.views import log

x = Math.var("x")

# Indefinite integral
expr = x**2
indef = expr.integrate(x)
indef.render()

# This should raise an error
try:
    indef.evaluate()
except EvaluationError as e:
    log(f"Expected error: {e}", level="warning")

# %% Cell 11 - Substitution
from cm.symbols import Math

x = Math.var("x")
t = Math.var("t")

expr = x**2 + 2*x + 1

# Substitute x = t + 1
result = expr.subs({x: t + 1})
result.render()

# Expand to simplify
expanded = result.expand()
expanded.render()
