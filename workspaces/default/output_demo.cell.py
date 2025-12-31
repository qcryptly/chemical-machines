# %% Cell 1 - HTML Output Demo
# Demonstrates the cm.views module for rich HTML outputs

from cm.views import html, text, log, table

# Output formatted HTML
html("<h2>Welcome to Chemical Machines!</h2>")
html("<p>This cell demonstrates <strong>rich HTML output</strong>.</p>")

# Log messages with different levels
log("Processing started...")
log("Computation complete!", level="success")

# %% Cell 2 - Tables and Data
from cm.views import html, table, log

# Display a table
data = [
    ["Hydrogen", "H", 1.008],
    ["Carbon", "C", 12.011],
    ["Nitrogen", "N", 14.007],
    ["Oxygen", "O", 15.999],
]
table(data, headers=["Element", "Symbol", "Atomic Mass"])

log("Table displayed with 4 elements")

# %% Cell 3 - Matplotlib Integration
from cm.views import savefig, log

try:
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a simple plot
    x = np.linspace(0, 2 * np.pi, 100)
    plt.figure(figsize=(8, 4))
    plt.plot(x, np.sin(x), label='sin(x)')
    plt.plot(x, np.cos(x), label='cos(x)')
    plt.legend()
    plt.title('Trigonometric Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)

    # Save to HTML output
    savefig()
    plt.close()

    log("Plot saved successfully!", level="success")
except ImportError:
    log("matplotlib not available - skipping plot demo", level="warning")
