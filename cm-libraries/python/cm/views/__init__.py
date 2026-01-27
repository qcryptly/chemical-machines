"""
Chemical Machines Views Package

A package for rendering HTML outputs from Python cells.
Outputs are written to .out/ directory and displayed in the workspace UI.

Submodules:
- output: Core HTML output functions (html, text, log, clear, etc.)
- visualization: Scientific visualization (molecule, scatter_3d, surface, etc.)

Usage:
    from cm.views import html, text, log, clear
    from cm.views import molecule, scatter_3d, surface

    # Output HTML content
    html("<h1>Hello World</h1>")

    # Visualize a molecule
    molecule(atoms=[('C', 0, 0, 0), ('O', 1.2, 0, 0)])
"""

# Core output functions
from .output import (
    html,
    text,
    log,
    clear,
    image,
    savefig,
    dataframe,
    table,
)

# Scientific visualization
from .visualization import (
    ELEMENT_DATA,
    COLORMAPS,
    scatter_3d,
    surface,
    molecule,
    crystal,
    orbital,
)

__all__ = [
    # Core output
    'html',
    'text',
    'log',
    'clear',
    'image',
    'savefig',
    'dataframe',
    'table',
    # Visualization
    'ELEMENT_DATA',
    'COLORMAPS',
    'scatter_3d',
    'surface',
    'molecule',
    'crystal',
    'orbital',
]
