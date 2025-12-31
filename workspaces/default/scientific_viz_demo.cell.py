# %% Cell 1 - Surface Plot from Function
# Demonstrates surface() with a mathematical function

import numpy as np
from cm.views import surface, log

log("Rendering a mathematical surface...", level="info")

# Ripple function
surface(
    f=lambda x, y: np.sin(np.sqrt(x**2 + y**2)) * np.exp(-0.1 * (x**2 + y**2)),
    x_range=(-5, 5),
    y_range=(-5, 5),
    resolution=60,
    colormap='viridis',
    unit_box=True
)

log("Surface rendered! Use mouse to orbit, scroll to zoom.", level="success")

# %% Cell 2 - Scatter Plot
# Demonstrates scatter_3d() with random points

import numpy as np
from cm.views import scatter_3d, log

log("Rendering a 3D scatter plot...", level="info")

# Generate random points in a sphere
n = 200
theta = np.random.uniform(0, 2*np.pi, n)
phi = np.arccos(2*np.random.uniform(0, 1, n) - 1)
r = 2 * np.cbrt(np.random.uniform(0, 1, n))

points = np.column_stack([
    r * np.sin(phi) * np.cos(theta),
    r * np.sin(phi) * np.sin(theta),
    r * np.cos(phi)
])

# Color by distance from origin
scatter_3d(
    points,
    colors=np.linalg.norm(points, axis=1),
    colormap='plasma',
    point_size=0.08,
    unit_box=True
)

log("Scatter plot with 200 points rendered!", level="success")

# %% Cell 3 - Line Plot (Helix)
# Demonstrates line_3d() with a parametric curve

import numpy as np
from cm.views import line_3d, log

log("Rendering a helix curve...", level="info")

# Parametric helix
t = np.linspace(0, 6*np.pi, 200)
helix = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (2*np.pi)
])

line_3d(helix, color='#ff6b6b', unit_box=True)

log("Helix curve rendered!", level="success")

# %% Cell 4 - Vector Field
# Demonstrates vector_field() with a rotational field

import numpy as np
from cm.views import vector_field, log

log("Rendering a 3D vector field...", level="info")

# Rotational field around Z axis
vector_field(
    f=lambda x, y, z: (-y, x, 0.2),
    bounds=(-2, 2, -2, 2, -1, 1),
    resolution=5,
    scale=0.3,
    colormap='coolwarm',
    unit_box=True
)

log("Vector field rendered! Arrows colored by magnitude.", level="success")

# %% Cell 5 - Water Molecule
# Demonstrates molecule() with explicit atoms and bonds

from cm.views import molecule, log

log("Rendering a water molecule (H2O)...", level="info")

# Water molecule geometry (Angstroms)
water_atoms = [
    ('O', 0.000, 0.000, 0.117),
    ('H', 0.756, 0.000, -0.469),
    ('H', -0.756, 0.000, -0.469)
]

water_bonds = [(0, 1), (0, 2)]  # O-H bonds

molecule(
    water_atoms,
    bonds=water_bonds,
    style='ball-stick',
    atom_scale=1.0,
    auto_rotate=True
)

log("Water molecule rendered with ball-stick style!", level="success")

# %% Cell 6 - Caffeine from XYZ
# Demonstrates molecule_xyz() with XYZ file format

from cm.views import molecule_xyz, log

log("Rendering caffeine molecule from XYZ data...", level="info")

caffeine_xyz = """24
Caffeine molecule
N     1.282    0.710   -0.037
C     2.471    1.443    0.008
N     2.286    2.766   -0.017
C     0.965    2.839   -0.082
C     0.106    1.707   -0.097
C    -1.304    1.506   -0.154
O    -1.930    0.451   -0.168
N    -1.937    2.748   -0.196
C    -0.994    3.764   -0.166
O    -1.282    4.938   -0.197
N     0.364    3.993   -0.108
C     1.163    5.215   -0.073
C     3.778    0.862    0.074
C    -3.367    2.946   -0.258
H     3.061    3.412    0.012
H     3.654   -0.218    0.097
H     4.361    1.157    0.948
H     4.313    1.172   -0.823
H     0.509    6.081   -0.035
H     1.780    5.286    0.824
H     1.810    5.242   -0.951
H    -3.861    1.975   -0.276
H    -3.594    3.502   -1.168
H    -3.701    3.507    0.617
"""

molecule_xyz(caffeine_xyz, style='ball-stick', infer_bonds=True)

log("Caffeine molecule rendered with auto-detected bonds!", level="success")

# %% Cell 7 - Multiple Surfaces Comparison
# Shows wireframe vs solid surface

import numpy as np
from cm.views import surface, log

log("Rendering saddle surface (wireframe mode)...", level="info")

# Saddle surface (hyperbolic paraboloid)
surface(
    f=lambda x, y: x**2 - y**2,
    x_range=(-2, 2),
    y_range=(-2, 2),
    resolution=30,
    colormap='coolwarm',
    wireframe=True,
    unit_box=True
)

log("Saddle surface rendered in wireframe mode!", level="success")

# %% Cell 8 - Surface from Data Arrays
# Demonstrates surface() with numpy meshgrid data

import numpy as np
from cm.views import surface, log

log("Rendering a Gaussian peak from data arrays...", level="info")

# Create meshgrid
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)

# Gaussian peak
Z = np.exp(-(X**2 + Y**2))

surface(
    x=X, y=Y, z=Z,
    colormap='magma',
    unit_box=True,
    auto_rotate=True
)

log("Gaussian peak surface rendered from data arrays!", level="success")
