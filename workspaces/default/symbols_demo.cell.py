# %% Cell 1 - Basic LaTeX Rendering
# Demonstrates the cm.symbols module for LaTeX math

from cm.symbols import latex, equation, align, Math, set_line_height
from cm.views import html, log

html("<h2>LaTeX Math Rendering Demo</h2>")

# Set custom line height for better readability
set_line_height("1.8")

# Simple equations
latex(r"E = mc^2")
latex(r"\int_0^\infty e^{-x} dx = 1")

# Numbered equations
equation(r"F = ma", number=1)
equation(r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}", number=2)

log("Basic equations rendered!", level="success")

# %% Cell 2 - Math Builder
# Using the Math builder for programmatic formula construction

from cm.symbols import Math, latex, set_line_height
from cm.views import html, log

# Adjust line height (try different values: "1", "1.5", "2", "normal")
set_line_height("1.6")

html("<h3>Math Builder Examples</h3>")

# Build a fraction
m = Math()
m.var("x").equals().frac("a + b", "c - d")
m.render()

# Quadratic formula
m = Math()
m.var("x").equals().frac("-b \\pm \\sqrt{b^2 - 4ac}", "2a")
m.render(label="Quadratic Formula")

# Summation
m = Math()
m.sum("i=1", "n").var("i").equals().frac("n(n+1)", "2")
m.render()

log("Math builder examples complete!", level="success")

# %% Cell 3 - Quantum Mechanics (Bra-ket Notation)

from cm.symbols import set_notation, Math, latex, set_line_height
from cm.views import html, log

html("<h3>Quantum Mechanics - Bra-ket Notation</h3>")

# Tighter line height for compact display
set_line_height("1.4")
set_notation("braket")

# Bra-ket examples
m = Math()
m.braket("\\psi", "\\phi")
m.render()

# Expectation value
m = Math()
m.expval("\\hat{H}")
m.render()

# Matrix element
m = Math()
m.matelem("n", "\\hat{H}", "m")
m.render()

# Commutator
m = Math()
m.comm("\\hat{x}", "\\hat{p}").equals().var("i").hbar()
m.render()

# Schrodinger equation
latex(r"i\hbar \frac{\partial}{\partial t} |\psi\rangle = \hat{H} |\psi\rangle")

log("Quantum mechanics notation rendered!", level="success")

# %% Cell 4 - Chemistry Notation

from cm.symbols import set_notation, chemical, reaction, latex, Math, set_line_height
from cm.views import html, log

html("<h3>Chemistry Notation</h3>")

# More spacing for chemistry equations
set_line_height("2")
set_notation("chemist")

# Chemical formulas
html("<p>Water formation:</p>")
chemical("2H2 + O2 -> 2H2O")

html("<p>Ammonia synthesis:</p>")
reaction("N2 + 3H2", "2NH3", reversible=True)

# Thermodynamics
html("<p>Gibbs free energy:</p>")
latex(r"\Delta G = \Delta H - T \Delta S")

# Rate equation
html("<p>Rate law:</p>")
latex(r"r = k[A]^m[B]^n")

log("Chemistry notation rendered!", level="success")

# %% Cell 5 - Aligned Equations and Matrices

from cm.symbols import align, matrix, latex, set_line_height
from cm.views import html, log

# Standard line height for matrices
set_line_height("1.5")

html("<h3>Aligned Equations</h3>")

# Multi-step derivation
align(
    r"(a + b)^2 &= (a + b)(a + b)",
    r"&= a^2 + ab + ba + b^2",
    r"&= a^2 + 2ab + b^2"
)

html("<h3>Matrices</h3>")

# Identity matrix
matrix([["1", "0", "0"],
        ["0", "1", "0"],
        ["0", "0", "1"]], style="bmatrix")

# Pauli matrices
html("<p>Pauli X matrix:</p>")
matrix([["0", "1"],
        ["1", "0"]], style="pmatrix")

log("Aligned equations and matrices rendered!", level="success")

# %% Cell 6 - Physics Notation

from cm.symbols import set_notation, Math, latex, set_line_height
from cm.views import html, log

html("<h3>Physics Notation</h3>")

# Reset to default line height
set_line_height("normal")
set_notation("physicist")

# Maxwell's equations (differential form)
html("<p>Maxwell's Equations:</p>")

latex(r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}")
latex(r"\nabla \cdot \mathbf{B} = 0")
latex(r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}")
latex(r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}")

# Vector calculus
html("<p>Vector identities:</p>")
m = Math()
m.nabla().times().paren("f\\mathbf{A}").equals().var("f").paren("\\nabla \\times \\mathbf{A}").plus().paren("\\nabla f").times().var("\\mathbf{A}")
m.render()

log("Physics notation rendered!", level="success")
