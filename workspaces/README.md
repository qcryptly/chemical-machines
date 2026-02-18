# Chemical Machines Library Reference

Complete reference documentation for the `cm` library - a Python library for creating rich outputs and scientific visualizations in Chemical Machines workspaces.

## Table of Contents

- [Getting Started](#getting-started)
- [cm.views Module](#cmviews-module)
  - [Basic Output](#basic-output)
  - [Tables and DataFrames](#tables-and-dataframes)
  - [Images and Matplotlib](#images-and-matplotlib)
  - [3D Scatter Plots](#3d-scatter-plots)
  - [3D Line Plots](#3d-line-plots)
  - [Surface Plots](#surface-plots)
  - [Vector Fields](#vector-fields)
  - [Molecule Visualization](#molecule-visualization)
  - [Crystal Structures](#crystal-structures)
  - [WebGL Custom Rendering](#webgl-custom-rendering)
- [cm.symbols Module](#cmsymbols-module)
  - [LaTeX Rendering](#latex-rendering)
  - [Equations and Alignment](#equations-and-alignment)
  - [Matrices](#matrices)
  - [Math Lists](#math-lists)
  - [Symbolic Expressions](#symbolic-expressions)
  - [Math Builder Class](#math-builder-class)
  - [Notation Styles](#notation-styles)
  - [Chemistry Helpers](#chemistry-helpers)
  - [Angular Momentum Coupling](#angular-momentum-coupling)
  - [Differential Operators](#differential-operators)
  - [Hydrogen Wavefunctions](#hydrogen-wavefunctions)
  - [Basis Functions](#basis-functions)
- [cm.math Module](#cmmath-module)
  - [Overview](#overview)
  - [Numbers](#numbers)
  - [Linear Algebra (struct.lin_alg)](#linear-algebra-structlin_alg)
  - [Expression Operations](#expression-operations)
  - [LaTeX Rendering](#cmmath-latex-rendering)
  - [Eager Evaluation (NumPy)](#eager-evaluation-numpy)
  - [PyTorch Compute Graphs](#pytorch-compute-graphs)
  - [Custom Structures](#custom-structures)
  - [Symbolic Indexing](#symbolic-indexing)
  - [Special Functions (struct.spec_func)](#special-functions-structspec_func)
  - [Constraints](#constraints)
  - [Variable Binding and Substitution](#variable-binding-and-substitution)
  - [Symbolic Tensors](#symbolic-tensors)
  - [Mathematical Functions (struct.lin_alg.fxn)](#mathematical-functions-structlin_algfxn)
  - [Operators (cm.math.operator)](#operators-cmmathoperator)
- [cm.qm Module](#cmqm-module)
  - [Atoms](#atoms)
  - [Electron Configurations](#electron-configurations)
  - [Molecules](#molecules)
  - [Spin-Orbitals](#spin-orbitals)
  - [Slater Determinants](#slater-determinants)
  - [Overlaps with @ Operator](#overlaps-with--operator)
  - [Hamiltonian Matrix Elements](#hamiltonian-matrix-elements)
  - [Slater-Condon Rules](#slater-condon-rules)
  - [Relativistic Quantum Mechanics](#relativistic-quantum-mechanics)
  - [Hamiltonian Builder](#hamiltonian-builder)
  - [Matrix Expressions](#matrix-expressions)
  - [CI Basis Generation](#ci-basis-generation)
  - [Molecular Integrals (cm.qm.integrals)](#molecular-integrals-cmqmintegrals)
    - [Basis Sets](#basis-sets)
    - [Gaussian Primitives](#gaussian-primitives-and-contracted-functions)
    - [One-Electron Integrals](#one-electron-integrals)
    - [Two-Electron Integrals (ERI)](#two-electron-integrals-eri)
    - [Hartree-Fock Solver](#hartree-fock-solver)
    - [Kohn-Sham DFT](#kohn-sham-dft)
    - [Unrestricted HF (UHF)](#unrestricted-hartree-fock-uhf)
    - [MP2 Perturbation Theory](#mp2-perturbation-theory)
    - [CCSD(T) Coupled Cluster](#ccsdt-coupled-cluster)
    - [Analytic Gradients](#analytic-gradients)
    - [Geometry Optimization](#geometry-optimization)
    - [Transition State Search](#transition-state-search)
    - [Internal Coordinates](#internal-coordinates)
    - [Frequency Analysis](#frequency-analysis)
    - [Thermochemistry](#thermochemistry)
    - [TDDFT Excited States](#tddft-excited-states)
    - [PCM Solvation](#pcm-solvation)
    - [Dipole Moments](#dipole-moments)
    - [Polarizability](#polarizability)
    - [Effective Core Potentials](#effective-core-potentials-ecps)
    - [Molecular Orbital Visualization](#molecular-orbital-visualization)
- [cm.data Module](#cmdata-module)
  - [Searching for Molecules](#searching-for-molecules)
  - [Getting Molecule Data](#getting-molecule-data)
  - [Comparing Computed vs. Benchmark Data](#comparing-computed-vs-benchmark-data)
  - [Syncing Benchmark Databases](#syncing-benchmark-databases)
  - [Checking Status](#checking-status)
  - [Complete Workflow Example](#complete-workflow-example)
- [C++ Support](#c-support)
- [Colormaps Reference](#colormaps-reference)
- [Element Data Reference](#element-data-reference)

---

## Getting Started

Chemical Machines supports both Python and C++ with cell-based execution. Files ending in `.cell.py` or `.cell.cpp` are treated as notebooks with cells separated by `# %%` or `// %%` comments.

```python
# %% Cell 1 - My First Cell
from cm.views import log

log("Hello, Chemical Machines!")
```

---

## cm.views Module

The `cm.views` module provides functions for outputting HTML, text, tables, images, and 3D visualizations.

### Basic Output

#### `html(content: str)`

Output raw HTML content.

```python
from cm.views import html

html("<h1>Title</h1>")
html("<div style='color: blue'>Blue text</div>")
html("<p>This is <strong>formatted</strong> HTML.</p>")
```

#### `text(content: str)`

Output plain text (HTML-escaped automatically).

```python
from cm.views import text

text("Hello, World!")
text("Special chars: <, >, &")  # Will be escaped
```

#### `log(*args, level: str = 'info')`

Log values with automatic formatting. Dicts and lists are formatted as JSON.

**Parameters:**
- `*args`: Values to log (strings, dicts, lists, etc.)
- `level`: Log level - `'info'` (default), `'warning'`, `'error'`, `'success'`

```python
from cm.views import log

log("Processing file:", filename)
log({"status": "ok", "count": 42})  # Auto-formatted as JSON
log("Success!", level="success")    # Green success style
log("Warning message", level="warning")  # Yellow warning style
log("Error occurred!", level="error")    # Red error style

# Multiple arguments
log("The answer is:", 42)
log("Data:", {"key": "value"})
```

#### `clear()`

Clear all outputs for the current cell.

```python
from cm.views import clear

clear()  # Clear current cell's outputs
```

#### `clear_all()`

Clear all outputs for all cells (deletes the output file).

```python
from cm.views import clear_all

clear_all()
```

---

### Tables and DataFrames

#### `table(data: list[list], headers: list[str] = None)`

Output an HTML table.

**Parameters:**
- `data`: 2D list of cell values
- `headers`: Optional list of header strings

```python
from cm.views import table

data = [
    ["Hydrogen", "H", 1.008],
    ["Carbon", "C", 12.011],
    ["Nitrogen", "N", 14.007],
    ["Oxygen", "O", 15.999],
]

table(data, headers=["Element", "Symbol", "Atomic Mass"])
```

#### `dataframe(df, max_rows: int = 50)`

Output a pandas DataFrame as an HTML table.

**Parameters:**
- `df`: pandas DataFrame
- `max_rows`: Maximum rows to display (default: 50)

```python
from cm.views import dataframe
import pandas as pd

df = pd.DataFrame({
    'Element': ['H', 'He', 'Li'],
    'Atomic Number': [1, 2, 3],
    'Mass': [1.008, 4.003, 6.941]
})

dataframe(df, max_rows=50)
```

---

### Images and Matplotlib

#### `image(source, mime_type: str = 'image/png', alt: str = '', width: int = None, height: int = None)`

Output an image from file, URL, or bytes.

**Parameters:**
- `source`: File path, URL, or base64-encoded bytes
- `mime_type`: MIME type for binary data (default: `'image/png'`)
- `alt`: Alt text for accessibility
- `width`: Optional width in pixels
- `height`: Optional height in pixels

```python
from cm.views import image

# From file
image("plot.png")
image("/path/to/image.jpg")

# From URL
image("https://example.com/image.png")

# From bytes with explicit MIME type
image(png_bytes, mime_type="image/png")

# With dimensions
image("diagram.png", width=400, height=300)
```

#### `savefig(fig=None, **kwargs)`

Save a matplotlib figure as an image output.

**Parameters:**
- `fig`: Optional matplotlib figure (uses current figure if not specified)
- `**kwargs`: Additional arguments passed to `fig.savefig()` (format, dpi, bbox_inches, etc.)

```python
from cm.views import savefig
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
plt.figure(figsize=(8, 4))
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()
plt.title('Trigonometric Functions')
plt.grid(True, alpha=0.3)

savefig()  # Automatically saves current figure to output
plt.close()

# With custom settings
savefig(dpi=150, format='png')
```

---

### 3D Scatter Plots

#### `scatter_3d(points, colors=None, sizes=None, colormap='viridis', point_size=0.1, opacity=1.0, unit_box=True, box_size=None, box_color='#444444', box_labels=True, background='#1e1e2e', auto_rotate=False)`

Render a 3D scatter plot of points.

**Parameters:**
- `points`: Nx3 array-like of (x, y, z) coordinates
- `colors`: Optional - single color string, Nx3 RGB array (0-1), or N array for colormap
- `sizes`: Optional - single size or N array of per-point sizes (multiplied by point_size)
- `colormap`: Colormap name (`'viridis'`, `'plasma'`, `'coolwarm'`, etc.)
- `point_size`: Base point size (default: 0.1)
- `opacity`: Point opacity 0-1 (default: 1.0)
- `unit_box`: Show bounding box (default: True)
- `box_size`: Box size (w, h, d), auto-calculated if None
- `box_color`: Color of box edges
- `box_labels`: Show axis labels on box
- `background`: Background color
- `auto_rotate`: Auto-rotate the scene

```python
from cm.views import scatter_3d
import numpy as np

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

# Default: color by Z coordinate
scatter_3d(points)

# Color by distance from origin (scalar values mapped to colormap)
scatter_3d(
    points,
    colors=np.linalg.norm(points, axis=1),
    colormap='plasma',
    point_size=0.08,
    unit_box=True
)

# With RGB colors (0-1 range)
scatter_3d(points, colors=np.random.rand(200, 3))

# Single color for all points
scatter_3d(points, colors='#ff6b6b')
```

---

### 3D Line Plots

#### `line_3d(points, color='#00d4ff', width=2.0, opacity=1.0, unit_box=True, box_size=None, box_color='#444444', box_labels=True, background='#1e1e2e', auto_rotate=False)`

Render a 3D line/path through points.

**Parameters:**
- `points`: Nx3 array-like of (x, y, z) coordinates
- `color`: Line color (hex string)
- `width`: Line width in pixels
- `opacity`: Line opacity 0-1
- `unit_box`: Show bounding box
- `box_size`: Box size (w, h, d), auto-calculated if None
- `box_color`: Color of box edges
- `box_labels`: Show axis labels
- `background`: Background color
- `auto_rotate`: Auto-rotate the scene

```python
from cm.views import line_3d
import numpy as np

# Helix
t = np.linspace(0, 6*np.pi, 200)
helix = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (2*np.pi)
])

line_3d(helix, color='#ff6b6b', unit_box=True)
```

#### `lines_3d(paths, colors=None, width=2.0, opacity=1.0, unit_box=True, box_size=None, box_color='#444444', box_labels=True, background='#1e1e2e', auto_rotate=False)`

Render multiple 3D lines/paths.

**Parameters:**
- `paths`: List of Nx3 array-like, each representing a path
- `colors`: Optional list of colors (one per path), or single color for all
- `width`: Line width in pixels
- `opacity`: Line opacity 0-1
- `unit_box`: Show bounding box
- `box_size`: Box size (w, h, d), auto-calculated if None
- `box_color`: Color of box edges
- `box_labels`: Show axis labels
- `background`: Background color
- `auto_rotate`: Auto-rotate the scene

```python
from cm.views import lines_3d
import numpy as np

# Multiple trajectories
paths = []
for i in range(5):
    t = np.linspace(0, 2*np.pi, 50)
    path = np.column_stack([np.cos(t) + i, np.sin(t), t])
    paths.append(path)

lines_3d(paths, colors=['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'])
```

---

### Surface Plots

#### `surface(f=None, x=None, y=None, z=None, x_range=(-5, 5), y_range=(-5, 5), resolution=50, colormap='viridis', wireframe=False, opacity=1.0, unit_box=True, box_size=None, box_color='#444444', box_labels=True, background='#1e1e2e', auto_rotate=False)`

Render a 3D surface from a function or data arrays.

**Parameters:**
- `f`: Optional function f(x, y) -> z, where x, y can be scalars or arrays
- `x`: Optional 2D array of X coordinates (from meshgrid)
- `y`: Optional 2D array of Y coordinates (from meshgrid)
- `z`: Optional 2D array of Z values
- `x_range`: Range for X axis when using function (min, max)
- `y_range`: Range for Y axis when using function (min, max)
- `resolution`: Grid resolution when using function
- `colormap`: Colormap for surface coloring
- `wireframe`: Show as wireframe instead of solid
- `opacity`: Surface opacity 0-1
- `unit_box`: Show bounding box
- `box_size`: Box size (w, h, d), auto-calculated if None
- `box_color`: Color of box edges
- `box_labels`: Show axis labels
- `background`: Background color
- `auto_rotate`: Auto-rotate the scene

```python
from cm.views import surface
import numpy as np

# From a function
surface(
    f=lambda x, y: np.sin(np.sqrt(x**2 + y**2)) * np.exp(-0.1 * (x**2 + y**2)),
    x_range=(-5, 5),
    y_range=(-5, 5),
    resolution=60,
    colormap='viridis',
    unit_box=True
)

# From data arrays (meshgrid)
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

surface(x=X, y=Y, z=Z, colormap='magma', auto_rotate=True)

# Wireframe mode
surface(
    f=lambda x, y: x**2 - y**2,  # Saddle surface
    x_range=(-2, 2),
    y_range=(-2, 2),
    wireframe=True,
    colormap='coolwarm'
)
```

---

### Vector Fields

#### `vector_field(positions=None, vectors=None, f=None, bounds=(-2, 2, -2, 2, -2, 2), resolution=5, scale=1.0, colormap='coolwarm', arrow_head_length=0.2, arrow_head_width=0.1, opacity=1.0, unit_box=True, box_size=None, box_color='#444444', box_labels=True, background='#1e1e2e', auto_rotate=False)`

Render a 3D vector field.

**Parameters:**
- `positions`: Nx3 array of base positions for vectors
- `vectors`: Nx3 array of vector directions/magnitudes
- `f`: Optional function f(x, y, z) -> (vx, vy, vz) for grid sampling
- `bounds`: Grid bounds (x_min, x_max, y_min, y_max, z_min, z_max) when using f
- `resolution`: Grid resolution per axis when using f
- `scale`: Arrow length scale factor
- `colormap`: Colormap for vector magnitude coloring
- `arrow_head_length`: Arrow head length as fraction of total
- `arrow_head_width`: Arrow head width as fraction of length
- `opacity`: Arrow opacity 0-1
- `unit_box`: Show bounding box
- `box_size`: Box size (w, h, d), auto-calculated if None
- `box_color`: Color of box edges
- `box_labels`: Show axis labels
- `background`: Background color
- `auto_rotate`: Auto-rotate the scene

```python
from cm.views import vector_field
import numpy as np

# Rotational field around Z axis (from function)
vector_field(
    f=lambda x, y, z: (-y, x, 0.2),
    bounds=(-2, 2, -2, 2, -1, 1),
    resolution=5,
    scale=0.3,
    colormap='coolwarm',
    unit_box=True
)

# From data arrays
positions = np.random.uniform(-2, 2, (50, 3))
vectors = np.column_stack([
    -positions[:, 1],
    positions[:, 0],
    np.zeros(50)
])
vector_field(positions, vectors, scale=0.5)
```

---

### Molecule Visualization

#### `molecule(atoms, bonds=None, style='ball-stick', atom_scale=1.0, bond_radius=0.1, unit_box=False, box_size=None, box_color='#444444', box_labels=True, background='#1e1e2e', auto_rotate=True)`

Render a molecular structure.

**Parameters:**
- `atoms`: List of (element, x, y, z) tuples
- `bonds`: Optional list of (atom_index_1, atom_index_2) tuples
- `style`: Rendering style - `'ball-stick'`, `'spacefill'`, or `'stick'`
- `atom_scale`: Scale factor for atom radii
- `bond_radius`: Bond cylinder radius (for ball-stick and stick styles)
- `unit_box`: Show bounding box
- `box_size`: Box size (w, h, d), auto-calculated if None
- `box_color`: Color of box edges
- `box_labels`: Show axis labels
- `background`: Background color
- `auto_rotate`: Auto-rotate the scene

```python
from cm.views import molecule

# Water molecule
water_atoms = [
    ('O', 0.000, 0.000, 0.117),
    ('H', 0.756, 0.000, -0.469),
    ('H', -0.756, 0.000, -0.469)
]
water_bonds = [(0, 1), (0, 2)]  # O-H bonds (indices)

molecule(
    water_atoms,
    bonds=water_bonds,
    style='ball-stick',
    atom_scale=1.0,
    auto_rotate=True
)

# Different rendering styles
molecule(atoms, bonds, style='spacefill')  # Space-filling model
molecule(atoms, bonds, style='stick')       # Stick model
```

#### `molecule_xyz(xyz_content: str, style='ball-stick', infer_bonds=True, bond_tolerance=0.4, atom_scale=1.0, bond_radius=0.1, unit_box=False, box_size=None, box_color='#444444', box_labels=True, background='#1e1e2e', auto_rotate=True)`

Render a molecule from XYZ file content.

**Parameters:**
- `xyz_content`: String content of XYZ file
- `style`: Rendering style - `'ball-stick'`, `'spacefill'`, or `'stick'`
- `infer_bonds`: Automatically detect bonds based on atomic distances
- `bond_tolerance`: Extra distance tolerance for bond detection (Angstroms)
- `atom_scale`: Scale factor for atom radii
- `bond_radius`: Bond cylinder radius
- `unit_box`: Show bounding box
- `box_size`: Box size (w, h, d)
- `box_color`: Color of box edges
- `box_labels`: Show axis labels
- `background`: Background color
- `auto_rotate`: Auto-rotate the scene

**XYZ File Format:**
```
N                     # Number of atoms
comment line          # Any text (ignored)
Element X Y Z         # Element symbol and coordinates
Element X Y Z
...
```

```python
from cm.views import molecule_xyz

xyz_data = """3
Water molecule
O   0.000   0.000   0.117
H   0.756   0.000  -0.469
H  -0.756   0.000  -0.469
"""

molecule_xyz(xyz_data, style='ball-stick', infer_bonds=True)

# From file
with open('caffeine.xyz') as f:
    molecule_xyz(f.read())
```

---

### Crystal Structures

#### `crystal(cif_content: str, supercell=(1, 1, 1), style='ball-stick', infer_bonds=True, bond_tolerance=0.4, atom_scale=1.0, bond_radius=0.1, unit_box=True, box_color='#666666', box_labels=True, background='#1e1e2e', auto_rotate=True)`

Render a crystal structure from CIF file content.

**Parameters:**
- `cif_content`: String content of CIF file
- `supercell`: Number of unit cells to replicate in (a, b, c) directions
- `style`: Rendering style - `'ball-stick'`, `'spacefill'`, or `'stick'`
- `infer_bonds`: Automatically detect bonds based on atomic distances
- `bond_tolerance`: Extra distance tolerance for bond detection (Angstroms)
- `atom_scale`: Scale factor for atom radii
- `bond_radius`: Bond cylinder radius
- `unit_box`: Show unit cell box
- `box_color`: Color of unit cell edges
- `box_labels`: Show axis labels
- `background`: Background color
- `auto_rotate`: Auto-rotate the scene

```python
from cm.views import crystal

cif_data = """
data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""

crystal(cif_data, supercell=(2, 2, 2), style='ball-stick')

# From file
with open('structure.cif') as f:
    crystal(f.read())
```

---

### WebGL Custom Rendering

#### `webgl(content: str)`

Output full HTML with custom WebGL/Three.js code to the main visualization panel.

This writes to a special `.out/main.webgl.html` file that is displayed in the collapsible WebGL panel at the top of the workspace.

**Parameters:**
- `content`: Full HTML content including WebGL/Three.js code

```python
from cm.views import webgl

webgl('''
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin: 0; padding: 0; }
        body { background: #1e1e2e; overflow: hidden; }
        canvas { display: block; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#1e1e2e');

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Your custom 3D content here
        const geometry = new THREE.BoxGeometry();
        const material = new THREE.MeshNormalMaterial();
        const cube = new THREE.Mesh(geometry, material);
        scene.add(cube);

        function animate() {
            requestAnimationFrame(animate);
            cube.rotation.x += 0.01;
            cube.rotation.y += 0.01;
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
''')
```

#### `webgl_threejs(scene_setup: str, animate_loop: str = "", width: str = "100%", height: str = "100%", background: str = "#1e1e2e", camera_position: tuple = (0, 0, 5), controls: bool = True)`

Output a Three.js scene with common boilerplate handled.

**Parameters:**
- `scene_setup`: JavaScript code to set up the scene (add meshes, lights, etc.)
- `animate_loop`: Optional JavaScript code to run each animation frame
- `width`: CSS width of the canvas (default: "100%")
- `height`: CSS height of the canvas (default: "100%")
- `background`: Background color (default: dark theme)
- `camera_position`: Initial camera position tuple (x, y, z)
- `controls`: Enable OrbitControls for mouse interaction

**Available Variables in scene_setup/animate_loop:**
- `scene` - THREE.Scene instance
- `camera` - THREE.PerspectiveCamera
- `renderer` - THREE.WebGLRenderer
- `controls` - THREE.OrbitControls (if enabled)

```python
from cm.views import webgl_threejs

webgl_threejs(
    scene_setup='''
        // Create a cube with normal material
        const geometry = new THREE.BoxGeometry(2, 2, 2);
        const material = new THREE.MeshNormalMaterial();
        const cube = new THREE.Mesh(geometry, material);
        scene.add(cube);

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
    ''',
    animate_loop='''
        cube.rotation.x += 0.01;
        cube.rotation.y += 0.01;
    ''',
    camera_position=(0, 0, 5),
    background="#1e1e2e"
)
```

---

## cm.symbols Module

The `cm.symbols` module provides LaTeX math rendering with support for different notation styles.

### LaTeX Rendering

#### `latex(expression: str, display: bool = True, label: str = None, justify: str = "center")`

Render a LaTeX math expression.

**Parameters:**
- `expression`: LaTeX math expression (without delimiters)
- `display`: If True, render as display math (centered, block). If False, inline.
- `label`: Optional label/caption for the expression
- `justify`: Alignment - `'left'`, `'center'`, or `'right'`

```python
from cm.symbols import latex

latex(r"E = mc^2")
latex(r"\int_0^\infty e^{-x} dx = 1")

# Inline math
latex(r"x^2", display=False)

# With label
latex(r"F = ma", label="Newton's Second Law")

# Left-aligned
latex(r"y = mx + b", justify="left")
```

#### `set_line_height(height: str)`

Set the line height for math rendering.

**Parameters:**
- `height`: CSS line-height value (e.g., `"1"`, `"1.5"`, `"2"`, `"normal"`, `"1.2em"`)

```python
from cm.symbols import set_line_height

set_line_height("1.8")   # More space between equations
set_line_height("1.2")   # Tighter spacing
set_line_height("normal")  # Reset to default
```

---

### Equations and Alignment

#### `equation(expression: str, number: int | str = None)`

Render a numbered equation.

**Parameters:**
- `expression`: LaTeX math expression
- `number`: Optional equation number or label

```python
from cm.symbols import equation

equation(r"F = ma", number=1)
equation(r"E = mc^2", number="2.1")
equation(r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}", number=2)
```

#### `align(*equations: str)`

Render aligned equations (useful for multi-step derivations).

**Parameters:**
- `*equations`: LaTeX expressions with `&` for alignment points

```python
from cm.symbols import align

align(
    r"(a + b)^2 &= (a + b)(a + b)",
    r"&= a^2 + ab + ba + b^2",
    r"&= a^2 + 2ab + b^2"
)
```

---

### Matrices

#### `matrix(data: list[list], style: str = "pmatrix")`

Render a matrix.

**Parameters:**
- `data`: 2D list of matrix elements
- `style`: Matrix style:
  - `'pmatrix'` - parentheses (default)
  - `'bmatrix'` - square brackets
  - `'vmatrix'` - vertical bars (determinant)
  - `'Vmatrix'` - double vertical bars
  - `'matrix'` - no brackets

```python
from cm.symbols import matrix

# Identity matrix with brackets
matrix([
    ["1", "0", "0"],
    ["0", "1", "0"],
    ["0", "0", "1"]
], style="bmatrix")

# Pauli matrix with parentheses
matrix([
    ["0", "1"],
    ["1", "0"]
], style="pmatrix")

# Determinant
matrix([
    ["a", "b"],
    ["c", "d"]
], style="vmatrix")
```

---

### Math Lists

#### `bullets(*expressions: str, display: bool = True)`

Render a bulleted list of LaTeX expressions.

**Parameters:**
- `*expressions`: LaTeX expressions for each bullet point
- `display`: If True, use display math. If False, inline math.

```python
from cm.symbols import bullets

bullets(
    r"x^2 + y^2 = r^2",
    r"e^{i\pi} + 1 = 0",
    r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}"
)
```

#### `numbered(*expressions: str, start: int = 1, display: bool = True)`

Render a numbered list of LaTeX expressions.

**Parameters:**
- `*expressions`: LaTeX expressions for each numbered item
- `start`: Starting number (default: 1)
- `display`: If True, use display math. If False, inline math.

```python
from cm.symbols import numbered

numbered(
    r"F = ma",
    r"E = mc^2",
    r"p = mv"
)

# Starting from a different number
numbered(
    r"v = at",
    r"s = \frac{1}{2}at^2",
    start=4
)
```

#### `items(*expressions: str, display: bool = True)`

Render a plain list of LaTeX expressions (no bullets or numbers).

**Parameters:**
- `*expressions`: LaTeX expressions for each item
- `display`: If True, use display math. If False, inline math.

```python
from cm.symbols import items

items(
    r"\text{First equation: } x = 1",
    r"\text{Second equation: } y = 2"
)
```

---

### Symbolic Expressions

The `Math` class provides a symbolic expression system that supports:
- Variable and constant creation
- Arithmetic operations with operator overloading
- Differentiation and integration
- Special functions (Bessel, Legendre, Hermite, etc.)
- LaTeX rendering
- Numerical evaluation
- PyTorch compilation for GPU acceleration

#### Creating Expressions

```python
from cm.symbols import Math

# Create variables
x = Math.var("x")
n = Math.var("n")
theta = Math.var("theta")

# Create constants
c = Math.const(3.14159)

# Build expressions with operators
expr = x**2 + 2*x + 1
quadratic = (x - 1) * (x + 1)  # x² - 1

# Render as LaTeX
expr.render()

# Evaluate numerically
result = expr.evaluate(x=3)  # = 16
```

#### Calculus

```python
from cm.symbols import Math

x = Math.var("x")
expr = x**3 + 2*x

# Differentiation
derivative = expr.diff(x)
derivative.render()  # 3x² + 2

# Integration
integral = expr.integrate(x)
integral.render()  # x⁴/4 + x²

# Definite integration
definite = expr.integrate(x, bounds=[0, 1])
definite.evaluate()  # = 1.25
```

#### Summation and Products

```python
from cm.symbols import Math

i = Math.var("i")
n = Math.var("n")

# Discrete summation: Σ_{i=1}^{n} i²
sum_expr = Math.sum(i**2, i, 1, n)
sum_expr.render()

# Discrete product: Π_{i=1}^{5} i = 5!
prod_expr = Math.prod(i, i, 1, 5)
prod_expr.evaluate()  # = 120

# Chain syntax
(i**2).sum(i, bounds=(1, n)).render()
```

#### Special Functions

Access 50+ special functions commonly used in physics:

```python
from cm.symbols import Math

x = Math.var("x")
n = Math.var("n")

# Gamma and factorials
Math.gamma(5).evaluate()           # Γ(5) = 24
Math.factorial(5).evaluate()       # 5! = 120
Math.binomial(10, 3).evaluate()    # C(10,3) = 120
Math.beta(2, 3).evaluate()         # B(2,3)

# Error functions
Math.erf(1).evaluate()             # erf(1) ≈ 0.8427
Math.erfc(1).evaluate()            # erfc(1) ≈ 0.1573

# Bessel functions
Math.besselj(0, 2.4).evaluate()    # J₀(2.4)
Math.bessely(1, x).render()        # Y₁(x)
Math.besseli(0, x).render()        # I₀(x) - modified
Math.besselk(0, x).render()        # K₀(x) - modified
Math.jn(0, x).render()             # j₀(x) - spherical
Math.yn(0, x).render()             # y₀(x) - spherical

# Orthogonal polynomials
Math.legendre(n, x).render()               # Pₙ(x) - Legendre
Math.assoc_legendre(2, 1, x).render()      # P₂¹(x) - Associated Legendre
Math.hermite(n, x).render()                # Hₙ(x) - Hermite (physicist)
Math.hermite_prob(n, x).render()           # Heₙ(x) - Hermite (probabilist)
Math.laguerre(n, x).render()               # Lₙ(x) - Laguerre
Math.assoc_laguerre(n, 2, x).render()      # Lₙ⁽²⁾(x) - Associated Laguerre
Math.chebyshevt(n, x).render()             # Tₙ(x) - Chebyshev 1st kind
Math.chebyshevu(n, x).render()             # Uₙ(x) - Chebyshev 2nd kind
Math.gegenbauer(n, 0.5, x).render()        # Cₙ⁽⁰·⁵⁾(x) - Gegenbauer
Math.jacobi(n, 1, 2, x).render()           # Pₙ⁽¹'²⁾(x) - Jacobi

# Spherical harmonics
theta, phi = Math.var("theta"), Math.var("phi")
l, m = Math.var("l"), Math.var("m")
Math.Ylm(l, m, theta, phi).render()        # Yₗᵐ(θ,φ) - complex
Math.Ylm_real(l, m, theta, phi).render()   # Yₗₘ(θ,φ) - real

# Airy functions
Math.airyai(x).render()            # Ai(x)
Math.airybi(x).render()            # Bi(x)

# Hypergeometric functions
a, b, c, z = Math.var("a"), Math.var("b"), Math.var("c"), Math.var("z")
Math.hyper2f1(a, b, c, z).render()         # ₂F₁(a,b;c;z) - Gauss
Math.hyper1f1(a, b, z).render()            # ₁F₁(a;b;z) - Confluent
Math.hyper0f1(b, z).render()               # ₀F₁(;b;z)

# Elliptic integrals
Math.elliptic_k(0.5).evaluate()    # K(0.5) - complete 1st kind
Math.elliptic_e(0.5).evaluate()    # E(0.5) - complete 2nd kind

# Other functions
Math.zeta(2).evaluate()            # ζ(2) = π²/6
Math.polylog(2, 0.5).evaluate()    # Li₂(0.5)
Math.dirac(x).render()             # δ(x)
Math.heaviside(x).render()         # θ(x)
Math.kronecker(i, j).render()      # δᵢⱼ
Math.levi_civita(i, j, k).render() # εᵢⱼₖ
```

**Available function categories:**
- **Gamma**: `gamma`, `loggamma`, `digamma`, `beta`, `factorial`, `factorial2`, `binomial`
- **Error**: `erf`, `erfc`, `erfi`
- **Bessel**: `besselj`, `bessely`, `besseli`, `besselk`, `jn`, `yn`, `hankel1`, `hankel2`
- **Airy**: `airyai`, `airybi`, `airyaiprime`, `airybiprime`
- **Polynomials**: `legendre`, `assoc_legendre`, `hermite`, `hermite_prob`, `laguerre`, `assoc_laguerre`, `chebyshevt`, `chebyshevu`, `gegenbauer`, `jacobi`
- **Spherical**: `Ylm`, `Ylm_real`
- **Hypergeometric**: `hyper2f1`, `hyper1f1`, `hyper0f1`, `hyperpfq`
- **Elliptic**: `elliptic_k`, `elliptic_e`, `elliptic_pi`
- **Other**: `zeta`, `polylog`, `dirac`, `heaviside`, `kronecker`, `levi_civita`

#### Custom Functions with Hyperparameters

Define reusable symbolic functions with typed hyperparameters:

```python
from cm.symbols import Math, Scalar

# Define a function with typed hyperparameters
a, b, x = Math.var("a"), Math.var("b"), Math.var("x")
f = Math.function(a * Math.exp(b * x), hyperparams={"a": Scalar, "b": Scalar})
f.save("MyExponential")

# Retrieve and instantiate with specific values
func = Math.get_function("MyExponential")
inst = func.init(a=10, b=0.5)
inst.render()           # Shows: 10·e^(0.5x)

# Eager evaluation
result = inst.run(x=2)  # Evaluates numerically

# Lazy evaluation (returns ComputeGraph)
cg = inst.run_with(x=2)
cg.evaluate()           # Same result, but lazily
```

#### PyTorch Compilation

Compile expressions to GPU-accelerated PyTorch functions:

```python
from cm.symbols import Math
import torch

x = Math.var("x")
expr = Math.sin(x) * Math.exp(-x**2)

# Create a bound function
a = Math.var("a")
f = Math.function(a * expr, hyperparams={"a": Scalar})
inst = f.init(a=2.0)

# Compile to PyTorch
cg = inst.run_with(x=1.0)
torch_fn = cg.compile(device='cuda')  # or 'cpu'

# Evaluate on GPU with tensors
x_tensor = torch.linspace(0, 10, 1000).cuda()
result = torch_fn(x=x_tensor)

# Automatic differentiation
grad_fn = torch_fn.grad()
gradients = grad_fn(x=1.0)  # Returns dict of gradients
```

#### Physics Examples

**Hydrogen atom radial wavefunction:**
```python
from cm.symbols import Math

n, l, r, a0 = Math.var("n"), Math.var("l"), Math.var("r"), Math.var("a_0")

# Radial part uses associated Laguerre polynomials
rho = 2*r / (n*a0)
R = Math.assoc_laguerre(n - l - 1, 2*l + 1, rho)
R.render()  # Displays L_{n-l-1}^{(2l+1)}(2r/na₀)

# Angular part uses spherical harmonics
theta, phi, m = Math.var("theta"), Math.var("phi"), Math.var("m")
Y = Math.Ylm(l, m, theta, phi)
Y.render()  # Displays Y_l^m(θ, φ)
```

**Quantum harmonic oscillator:**
```python
from cm.symbols import Math

n, x, sigma = Math.var("n"), Math.var("x"), Math.var("sigma")

# Wavefunction: ψ_n(x) ∝ H_n(x/σ) exp(-x²/2σ²)
psi = Math.hermite(n, x/sigma) * Math.exp(-x**2 / (2*sigma**2))
psi.render()
```

---

### Math Builder Class

The `Math` class also provides a builder pattern for constructing LaTeX expressions programmatically.

#### Creating and Rendering

```python
from cm.symbols import Math

m = Math()
m.var("x").equals().frac("a + b", "c - d")
m.render()

# With label
m = Math()
m.var("x").equals().frac("-b \\pm \\sqrt{b^2 - 4ac}", "2a")
m.render(label="Quadratic Formula")
```

#### Basic Operations

```python
m = Math()

# Variables and text
m.var("x")              # x
m.text("result")        # \text{result}
m.raw(r"\alpha")        # Raw LaTeX

# Operators
m.plus()                # +
m.minus()               # -
m.times()               # ×
m.cdot()                # ·
m.div()                 # ÷
m.equals()              # =
m.approx()              # ≈
m.neq()                 # ≠
m.lt()                  # <
m.gt()                  # >
m.leq()                 # ≤
m.geq()                 # ≥
```

#### Fractions and Roots

```python
m = Math()

m.frac("a", "b")        # a/b fraction
m.sqrt("x")             # √x
m.sqrt("x", n="3")      # ³√x (cube root)
```

#### Subscripts and Superscripts

```python
m = Math()

m.var("x").sub("i")             # x_i
m.var("x").sup("2")             # x^2
m.var("x").subsup("i", "2")     # x_i^2
```

#### Greek Letters

```python
m = Math()

# Lowercase
m.alpha()    # α       m.beta()     # β       m.gamma()    # γ
m.delta()    # δ       m.epsilon()  # ε       m.zeta()     # ζ
m.eta()      # η       m.theta()    # θ       m.iota()     # ι
m.kappa()    # κ       m.lambda_()  # λ       m.mu()       # μ
m.nu()       # ν       m.xi()       # ξ       m.pi()       # π
m.rho()      # ρ       m.sigma()    # σ       m.tau()      # τ
m.upsilon()  # υ       m.phi()      # φ       m.chi()      # χ
m.psi()      # ψ       m.omega()    # ω

# Uppercase
m.Gamma()    # Γ       m.Delta()    # Δ       m.Theta()    # Θ
m.Lambda()   # Λ       m.Xi()       # Ξ       m.Pi()       # Π
m.Sigma()    # Σ       m.Phi()      # Φ       m.Psi()      # Ψ
m.Omega()    # Ω
```

#### Calculus

```python
m = Math()

# Integrals
m.integral()                    # ∫
m.integral(lower="0")           # ∫₀
m.integral("0", "\\infty")      # ∫₀^∞

# Sums and Products
m.sum()                         # Σ
m.sum("i=1", "n")               # Σᵢ₌₁ⁿ
m.prod("k=1", "n")              # Πₖ₌₁ⁿ

# Limits
m.lim("x", "0")                 # lim_{x→0}
m.lim("n", "\\infty")           # lim_{n→∞}

# Derivatives
m.deriv("f", "x")               # df/dx
m.deriv("", "t")                # d/dt
m.partial("f", "x")             # ∂f/∂x

# Nabla
m.nabla()                       # ∇
```

#### Brackets and Grouping

```python
m = Math()

m.paren("x + y")        # (x + y) with auto-sizing
m.bracket("x + y")      # [x + y] with auto-sizing
m.brace("x + y")        # {x + y} with auto-sizing
m.abs("x")              # |x| absolute value
m.norm("x")             # ||x|| norm
```

#### Quantum Mechanics (Bra-ket Notation)

```python
from cm.symbols import Math, set_notation

set_notation("braket")

m = Math()

# Bra and Ket (accept strings or lists)
m.bra("\\psi")                  # ⟨ψ|
m.ket("\\phi")                  # |φ⟩
m.bra(["1", "2"])               # ⟨1, 2|  (for multi-index states)
m.ket(["n", "l", "m"])          # |n, l, m⟩

# Bracket (inner product)
m.braket("\\psi", "\\phi")      # ⟨ψ|φ⟩

# Expectation value
m.expval("\\hat{H}")            # ⟨Ĥ⟩

# Matrix element
m.matelem("n", "\\hat{H}", "m") # ⟨n|Ĥ|m⟩

# Operators
m.op("H")                       # Ĥ (with hat)
m.dagger()                      # † (dagger superscript)

# Commutator
m.comm("\\hat{x}", "\\hat{p}")  # [x̂, p̂]
```

#### Physics

```python
m = Math()

m.vec("A")              # A with arrow
m.hbar()                # ℏ
m.infty()               # ∞
```

#### Chemistry

```python
m = Math()

m.ce("H2O")             # H₂O in upright text
m.yields()              # →
m.equilibrium()         # ⇌
```

#### Special Functions

```python
m = Math()

m.sin("x")              # sin(x)
m.cos("\\theta")        # cos(θ)
m.tan("x")              # tan(x)
m.ln("x")               # ln(x)
m.log("x")              # log(x)
m.log("x", base="2")    # log₂(x)
m.exp("x")              # exp(x)
```

#### Spacing

```python
m = Math()

m.space()               # Normal space
m.quad()                # Quad space
m.qquad()               # Double quad space
```

#### Symbolic Determinants

The Math class provides methods for rendering symbolic determinant expansions in various notations. These are useful for quantum mechanics, linear algebra, and Slater determinants.

```python
from cm.symbols import Math
import numpy as np

# Sample matrix
y = np.array([
    [2, 1, 0],
    [1, 3, 1],
    [0, 1, 2]
])

# Bra notation: ⟨a,b,c| + ⟨d,e,f| - ...
m = Math()
m.determinant_bra(y)
m.render()

# Ket notation: |a,b,c⟩ + |d,e,f⟩ - ...
m = Math()
m.determinant_ket(y)
m.render()

# Braket notation: ⟨ψ|a,b,c⟩ + ⟨ψ|d,e,f⟩ - ...
m = Math()
m.determinant_braket(y, bra_label="\\psi")
m.render()

# Product notation: (a·b·c) + (d·e·f) - ...
m = Math()
m.determinant_product(y)
m.render()

# Subscript notation: (a₁₁·a₂₂·a₃₃) + (a₁₂·a₂₃·a₃₁) - ...
m = Math()
m.determinant_subscript(y, var="a")
m.render()

# Chain with other operations
m = Math()
m.determinant_bra(y).equals().psi().dagger()
m.render()
```

**Parameters for determinant methods:**
- `matrix`: 2D array-like (numpy array or nested list) - elements are treated as symbolic placeholders
- `bra_label`: (braket only) Label for the bra side (default: "ψ")
- `var`: (subscript only) Variable name for elements (default: "a")

#### Slater Determinants (Symbolic Rendering)

For symbolic Slater determinant notation in cm.symbols, see the [Quantum Mechanics section](#slater-determinants) which uses `qm.slater()` to create actual Slater determinants with spin-orbitals that can compute overlaps and matrix elements.

**Note:** The `cm.symbols.Math` class focuses on symbolic mathematical expressions and special functions. For quantum mechanical Slater determinants with proper antisymmetry and overlap calculations, use the `cm.qm` module:

```python
from cm import qm

# Create spin-orbitals for helium ground state
orbitals = qm.basis_orbitals([
    (1, 0, 0, 1),   # 1s up: (n, l, m, spin)
    (1, 0, 0, -1),  # 1s down
])

# Create Slater determinant
psi = qm.slater(orbitals)
psi.render()  # Renders: |1s↑, 1s↓⟩

# Or from an atom directly
C = qm.atom('C')
psi_carbon = C.slater_determinant()
psi_carbon.render()  # Renders determinant for 1s² 2s² 2p²
```

#### Slater Determinant Overlaps

Compute overlaps between Slater determinants using the `@` operator:

```python
from cm import qm

# Create two determinants with same orbitals
orbitals = qm.basis_orbitals([
    (1, 0, 0, 1),   # 1s up
    (1, 0, 0, -1),  # 1s down
])
psi = qm.slater(orbitals)
phi = qm.slater(orbitals)

# Same orbitals - overlap is 1
overlap = psi @ phi
print(overlap.value)  # 1

# Different orbitals - overlap is 0 (orthogonal)
orbitals2 = qm.basis_orbitals([
    (2, 0, 0, 1),   # 2s up
    (2, 0, 0, -1),  # 2s down
])
chi = qm.slater(orbitals2)
overlap2 = psi @ chi
print(overlap2.value)  # 0
```

See the [Quantum Mechanics section](#slater-determinants) for more details on Slater determinants, matrix elements, and Slater-Condon rules

#### Building and Clearing

```python
m = Math()

# Build and get LaTeX string
m.frac("a", "b")
latex_str = m.build()   # Returns "\\frac{a}{b}"

# Render to output
m.render()
m.render(display=False)  # Inline
m.render(label="My Equation")
m.render(justify="left")

# Clear and start fresh
m.clear()
```

#### Complete Example

```python
from cm.symbols import Math, set_notation

# Quadratic formula
m = Math()
m.var("x").equals().frac("-b \\pm \\sqrt{b^2 - 4ac}", "2a")
m.render(label="Quadratic Formula")

# Summation formula
m = Math()
m.sum("i=1", "n").var("i").equals().frac("n(n+1)", "2")
m.render()

# Quantum mechanics
set_notation("braket")
m = Math()
m.comm("\\hat{x}", "\\hat{p}").equals().var("i").hbar()
m.render()

# Schrödinger equation
m = Math()
m.var("i").hbar().partial("", "t").ket("\\psi").equals().op("H").ket("\\psi")
m.render()
```

---

### Notation Styles

#### `set_notation(style: str)`

Set the notation style for math rendering. Each style includes custom macros.

**Parameters:**
- `style`: One of `'standard'`, `'physicist'`, `'chemist'`, `'braket'`, `'engineering'`

**Available Styles:**

| Style | Description | Custom Macros |
|-------|-------------|---------------|
| `standard` | Default LaTeX | None |
| `physicist` | Physics notation | `\vect`, `\grad`, `\curl`, `\divg`, `\lapl`, `\ddt`, `\ddx`, `\pderiv` |
| `chemist` | Chemistry notation | `\ce`, `\yields`, `\equilibrium`, `\gas`, `\precipitate`, `\aq`, `\solid`, `\liquid`, `\gasphase` |
| `braket` | Quantum mechanics | `\bra`, `\ket`, `\braket`, `\expval`, `\matelem`, `\op`, `\comm`, `\anticomm`, `\dagger` |
| `engineering` | Engineering notation | `\j`, `\ohm`, `\simark`, `\phasor`, `\magnitude`, `\phase`, `\conj`, `\re`, `\im` |

```python
from cm.symbols import set_notation, latex, Math

# Quantum mechanics notation
set_notation("braket")
m = Math()
m.braket("\\psi", "\\phi")
m.render()

# Chemistry notation
set_notation("chemist")
latex(r"\ce{H2O}")

# Physics notation
set_notation("physicist")
latex(r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J}")

# Reset to standard
set_notation("standard")
```

---

### Chemistry Helpers

#### `chemical(formula: str)`

Render a chemical formula with automatic subscript conversion.

**Parameters:**
- `formula`: Chemical formula (numbers after letters become subscripts, arrows converted)

```python
from cm.symbols import chemical

chemical("H2O")                    # H₂O
chemical("2H2 + O2 -> 2H2O")      # 2H₂ + O₂ → 2H₂O
chemical("CH3COOH")               # CH₃COOH
```

#### `reaction(reactants: str, products: str, reversible: bool = False)`

Render a chemical reaction.

**Parameters:**
- `reactants`: Reactant formula string
- `products`: Product formula string
- `reversible`: If True, use equilibrium arrows

```python
from cm.symbols import reaction

reaction("2H2 + O2", "2H2O")                    # 2H₂ + O₂ → 2H₂O
reaction("N2 + 3H2", "2NH3", reversible=True)  # N₂ + 3H₂ ⇌ 2NH₃
```

---

### Convenience Functions

#### `fraction(num: str, denom: str)`

Render a simple fraction.

```python
from cm.symbols import fraction

fraction("a + b", "c - d")
```

#### `sqrt(content: str, n: str = None)`

Render a square root or nth root.

```python
from cm.symbols import sqrt

sqrt("x")           # √x
sqrt("x", n="3")    # ³√x
```

---

### Angular Momentum Coupling

Compute and render angular momentum coupling coefficients used in quantum mechanics.

#### Clebsch-Gordan Coefficients

```python
from cm.symbols import ClebschGordan, Math

# Create a Clebsch-Gordan coefficient ⟨j₁ m₁ j₂ m₂ | J M⟩
cg = ClebschGordan(j1=1, m1=0, j2=1, m2=0, J=0, M=0)
cg.render()  # Display LaTeX

# Evaluate numerically
value = cg.evaluate()  # Returns float

# Using Math class
j1, m1 = Math.var("j_1"), Math.var("m_1")
cg_sym = Math.clebsch_gordan(j1, m1, 1, 0, 2, 0)
cg_sym.render()
```

#### Wigner 3j, 6j, 9j Symbols

```python
from cm.symbols import Wigner3j, Wigner6j, Wigner9j, Math

# Wigner 3j symbol (related to Clebsch-Gordan)
w3j = Wigner3j(j1=1, j2=1, j3=0, m1=1, m2=-1, m3=0)
w3j.render()
value = w3j.evaluate()

# Wigner 6j symbol (recoupling)
w6j = Wigner6j(j1=1, j2=1, j3=2, j4=1, j5=1, j6=2)
w6j.render()
value = w6j.evaluate()

# Wigner 9j symbol (coupling four angular momenta)
w9j = Wigner9j(
    j1=0.5, j2=0.5, j3=1,
    j4=0.5, j5=0.5, j6=1,
    j7=1, j8=1, j9=2
)
w9j.render()
value = w9j.evaluate()

# Using Math class shortcuts
w3j = Math.wigner_3j(1, 1, 0, 1, -1, 0)
w6j = Math.wigner_6j(1, 1, 2, 1, 1, 2)
w9j = Math.wigner_9j(0.5, 0.5, 1, 0.5, 0.5, 1, 1, 1, 2)
```

---

### Differential Operators

Create and apply differential operators in Cartesian or spherical coordinates.

#### Basic Operators

```python
from cm.symbols import PartialDerivative, Gradient, Laplacian, Math

# Partial derivative
x = Math.var("x")
f = x**2
df_dx = PartialDerivative(f, x)
df_dx.render()  # ∂(x²)/∂x
result = df_dx.apply()  # Symbolic result: 2x

# Gradient (vector of partial derivatives)
f = Math.var("f")
grad_f = Gradient(f, variables=[x, Math.var("y"), Math.var("z")])
grad_f.render()  # ∇f

# Laplacian (∇²)
lapl = Laplacian(f, coord_system='cartesian')
lapl.render()  # ∇²f

# Laplacian in spherical coordinates
r, theta, phi = Math.var("r"), Math.var("theta"), Math.var("phi")
lapl_sph = Laplacian(f, coord_system='spherical', variables=[r, theta, phi])
lapl_sph.render()  # Full spherical Laplacian
```

#### Applying Operators

```python
from cm.symbols import Laplacian, Math

r, theta, phi = Math.var("r"), Math.var("theta"), Math.var("phi")

# Define a function
psi = Math.exp(-r) * Math.Ylm(1, 0, theta, phi)

# Apply Laplacian
lapl = Laplacian(psi, coord_system='spherical', variables=[r, theta, phi])
result = lapl.apply()
result.render()
```

#### Operator Composition

```python
from cm.symbols import Laplacian, ScaledOperator, SumOperator, ComposedOperator

# Scale an operator: -½∇²
kinetic = ScaledOperator(Laplacian(f), -0.5)

# Sum operators: ∇² + V
potential = Math.var("V")
hamiltonian = SumOperator(kinetic, potential)

# Compose operators: A∘B (apply B then A)
composed = ComposedOperator(op1, op2)
```

---

### Hydrogen Wavefunctions

Built-in hydrogen-like radial and full wavefunctions.

#### Radial Wavefunctions

```python
from cm.symbols import HydrogenRadial, Math

# R_{nl}(r) = radial wavefunction
n, l = 2, 1  # 2p orbital
r = Math.var("r")
Z = Math.var("Z")
a0 = Math.var("a_0")

R = HydrogenRadial(n, l, r, Z=Z, a0=a0)
R.render()  # Shows normalized radial function

# Evaluate at specific point
R_numeric = HydrogenRadial(2, 1, r, Z=1, a0=1)
value = R_numeric.evaluate(r=1.0)

# Get the symbolic expression
expr = R.to_sympy()
```

#### Full Wavefunctions

```python
from cm.symbols import HydrogenOrbital, Math

r, theta, phi = Math.var("r"), Math.var("theta"), Math.var("phi")

# ψ_{nlm}(r,θ,φ) = R_{nl}(r) × Y_l^m(θ,φ)
psi = HydrogenOrbital(n=2, l=1, m=0, r=r, theta=theta, phi=phi)
psi.render()

# Ground state (1s)
psi_1s = HydrogenOrbital(1, 0, 0, r, theta, phi)

# 2p_z (real combination)
psi_2pz = HydrogenOrbital(2, 1, 0, r, theta, phi)

# Evaluate probability density
prob = (psi * psi.conjugate()).evaluate(r=1.0, theta=0.5, phi=0.0)
```

---

### Basis Functions

Create atomic orbital basis functions for electronic structure calculations.

#### Slater-Type Orbitals (STOs)

```python
from cm.symbols import SlaterTypeOrbital, Math

r = Math.var("r")
theta, phi = Math.var("theta"), Math.var("phi")

# Create an STO: χ = N × r^(n-1) × exp(-ζr) × Y_l^m(θ,φ)
sto = SlaterTypeOrbital(
    n=1, l=0, m=0,  # Quantum numbers
    zeta=1.0,        # Orbital exponent
    r=r, theta=theta, phi=phi
)
sto.render()

# Common STOs
sto_1s = SlaterTypeOrbital(n=1, l=0, m=0, zeta=1.0, r=r, theta=theta, phi=phi)
sto_2p = SlaterTypeOrbital(n=2, l=1, m=0, zeta=0.5, r=r, theta=theta, phi=phi)

# Evaluate
value = sto.evaluate(r=1.0, theta=0.0, phi=0.0)

# Get normalization constant
N = sto.normalization
```

#### Gaussian-Type Orbitals (GTOs)

```python
from cm.symbols import GaussianTypeOrbital, Math

x, y, z = Math.var("x"), Math.var("y"), Math.var("z")

# Cartesian GTO: χ = N × x^i × y^j × z^k × exp(-α r²)
gto = GaussianTypeOrbital(
    i=0, j=0, k=0,  # Angular momentum (s-type: i+j+k=0)
    alpha=0.5,       # Orbital exponent
    x=x, y=y, z=z,
    center=(0, 0, 0)  # Position
)
gto.render()

# p-type GTOs
gto_px = GaussianTypeOrbital(i=1, j=0, k=0, alpha=0.3, x=x, y=y, z=z)
gto_py = GaussianTypeOrbital(i=0, j=1, k=0, alpha=0.3, x=x, y=y, z=z)
gto_pz = GaussianTypeOrbital(i=0, j=0, k=1, alpha=0.3, x=x, y=y, z=z)

# d-type GTO (d_xy)
gto_dxy = GaussianTypeOrbital(i=1, j=1, k=0, alpha=0.2, x=x, y=y, z=z)

# Evaluate
value = gto.evaluate(x=0.5, y=0.5, z=0.0)
```

#### Contracted GTOs

```python
from cm.symbols import ContractedGTO, GaussianTypeOrbital, Math

x, y, z = Math.var("x"), Math.var("y"), Math.var("z")

# Create contracted GTO from primitives
# χ = Σ cᵢ × gᵢ(r)
primitives = [
    (0.15432897, GaussianTypeOrbital(0, 0, 0, 3.42525091, x, y, z)),
    (0.53532814, GaussianTypeOrbital(0, 0, 0, 0.62391373, x, y, z)),
    (0.44463454, GaussianTypeOrbital(0, 0, 0, 0.16885540, x, y, z)),
]

contracted = ContractedGTO(primitives)
contracted.render()

# Alternative: from coefficients and exponents
contracted = ContractedGTO.from_coefficients(
    coefficients=[0.15432897, 0.53532814, 0.44463454],
    exponents=[3.42525091, 0.62391373, 0.16885540],
    i=0, j=0, k=0,  # s-type
    x=x, y=y, z=z
)

# Evaluate
value = contracted.evaluate(x=0.0, y=0.0, z=0.0)
```

---

## cm.math Module

The `cm.math` module is a structure-based symbolic mathematics library. Every mathematical object belongs to a **Structure** — a set equipped with operations and axioms. Expressions form a DAG (directed acyclic graph) that can be rendered to LaTeX, evaluated eagerly with NumPy, or compiled to PyTorch compute graphs.

```python
from cm.math import struct, numbers
```

### Overview

The module is organized around two key imports:

- **`struct`** — The structure namespace containing mathematical structures (e.g., `struct.lin_alg` for linear algebra)
- **`numbers`** — Scalar fields and numeric dtypes

```python
from cm.math import struct, numbers
import numpy as np

# Create tensor expressions
A = struct.lin_alg.tensor(shape=(3,3), dtype=numbers.float64, value=np.eye(3), name="A")
B = struct.lin_alg.tensor(shape=(3,3), name="B")

# Build symbolic expressions with operator overloading
expr = (A @ B).det() + A.trace()

# Four interfaces:
expr.to_latex()                          # LaTeX string
expr.evaluate(B=np.ones((3,3)))          # NumPy eager evaluation
expr.to_torch(device="cpu")              # PyTorch compute graph
expr.render()                            # HTML output via cm.views
```

### Numbers

The `numbers` module defines abstract fields and concrete compute dtypes.

#### Abstract Fields

| Field | LaTeX | Description |
|-------|-------|-------------|
| `numbers.Reals` | `\mathbb{R}` | Real numbers |
| `numbers.Complex` | `\mathbb{C}` | Complex numbers |
| `numbers.Rationals` | `\mathbb{Q}` | Rational numbers |
| `numbers.Integers` | `\mathbb{Z}` | Integers (ring) |

#### Concrete Dtypes

| Dtype | NumPy | PyTorch | Field |
|-------|-------|---------|-------|
| `numbers.float32` | `np.float32` | `torch.float32` | Reals |
| `numbers.float64` | `np.float64` | `torch.float64` | Reals |
| `numbers.complexF32` | `np.complex64` | `torch.complex64` | Complex |
| `numbers.complexF64` | `np.complex128` | `torch.complex128` | Complex |
| `numbers.int32` | `np.int32` | `torch.int32` | Integers |
| `numbers.int64` | `np.int64` | `torch.int64` | Integers |

```python
# Dtypes carry their field association
numbers.float64.field    # => Field('Reals')
numbers.float64.np_dtype # => numpy.float64
```

### Linear Algebra (struct.lin_alg)

The linear algebra structure provides factory functions for creating tensor, vector, matrix, and scalar expressions.

#### `struct.lin_alg.tensor(shape=None, dtype=None, name=None, value=None)`

Create a tensor expression. Behavior depends on arguments:

- **With `value`**: Returns a concrete `Var` expression (eager, backed by a numpy array).
- **With `shape` but no `value`**: Returns a `SymbolicTensor` whose elements are symbolic expressions. Elements default to zero and can be assigned individually.
- **No `shape`, no `value`**: Returns an abstract `Var` placeholder that must be bound at evaluation time.

```python
# Concrete tensor (has a value)
A = struct.lin_alg.tensor(shape=(3,3), dtype=numbers.float64, value=np.eye(3), name="A")

# Symbolic tensor (elements are expressions)
y = struct.lin_alg.scalar(name="y")
X = struct.lin_alg.tensor(shape=(2,2))
X[0][0] = y ** 2 + 5
X[0][1] = y ** 3
X[1][0] = y
X[1][1] = y ** 2
print(X.det().bind(y=3).evaluate())  # symbolic determinant, then evaluate

# Abstract tensor (no shape, no value — a symbolic placeholder)
B = struct.lin_alg.tensor(shape=(3,3), name="B")

# Fully lazy tensor (no shape, no value)
Z = struct.lin_alg.tensor()
```

#### `struct.lin_alg.vector(dim=None, dtype=None, name=None, value=None)`

Create a rank-1 tensor (vector).

```python
v = struct.lin_alg.vector(dim=3, name="v")
w = struct.lin_alg.vector(dim=3, name="w", value=np.array([1, 0, 0]))
```

#### `struct.lin_alg.matrix(rows=None, cols=None, dtype=None, name=None, value=None)`

Create a rank-2 tensor (matrix).

```python
M = struct.lin_alg.matrix(rows=3, cols=3, name="M")
```

#### `struct.lin_alg.scalar(value=None, dtype=None, name=None)`

Create a scalar expression.

```python
c = struct.lin_alg.scalar(name="c")         # abstract scalar variable
k = struct.lin_alg.scalar(value=2.5)        # concrete scalar constant
```

### Expression Operations

Expressions support operator overloading and method-based operations. All operations return new `Expression` nodes in the DAG.

#### Arithmetic Operators

| Operator | Description | LaTeX |
|----------|-------------|-------|
| `A + B` | Addition | `\mathbf{A} + \mathbf{B}` |
| `A - B` | Subtraction | `\mathbf{A} - \mathbf{B}` |
| `A * B` | Element-wise / scalar multiply | `\mathbf{A} \cdot \mathbf{B}` |
| `A @ B` | Matrix multiply | `\mathbf{A} \mathbf{B}` |
| `A / B` | Division (renders as fraction) | `\frac{\mathbf{A}}{\mathbf{B}}` |
| `A ** n` | Exponentiation | `\mathbf{A}^{n}` |
| `-A` | Negation | `-\mathbf{A}` |

Scalar-tensor multiplication omits the dot: `2 * A` renders as `2 \mathbf{A}`.

#### Linear Algebra Methods

| Method | Description | LaTeX |
|--------|-------------|-------|
| `A.det()` | Determinant | `\det\left(\mathbf{A}\right)` |
| `A.trace()` | Trace | `\mathrm{tr}\left(\mathbf{A}\right)` |
| `A.transpose()` | Transpose | `\mathbf{A}^{\top}` |
| `A.inverse()` | Matrix inverse | `\mathbf{A}^{-1}` |
| `A.eigenvalues()` | Eigenvalues | `\mathrm{eig}\left(\mathbf{A}\right)` |
| `A.norm()` | Norm | `\left\|\mathbf{A}\right\|` |

```python
A = struct.lin_alg.tensor(shape=(3,3), name="A")
B = struct.lin_alg.tensor(shape=(3,3), name="B")

# Compose operations into complex expressions
expr = (A @ B).det() + 3 * A.trace() - B.det()
expr.to_latex()
# => \det\left(\mathbf{A} \mathbf{B}\right) + 3 \mathrm{tr}\left(\mathbf{A}\right) - \det\left(\mathbf{B}\right)
```

### cm.math LaTeX Rendering

Expressions render to LaTeX with automatic formatting for Greek letters, subscripts, bold tensors, and proper operator precedence.

#### `expr.to_latex() -> str`

Returns the LaTeX string representation of the expression.

#### `expr.render(display=True, justify="center")`

Renders the expression as MathJax HTML output via `cm.views.html()`. Uses the same rendering pipeline as `cm.symbols`.

```python
expr = (A @ B).det() + A.trace()
expr.render()                          # display math, centered
expr.render(display=False)             # inline math
expr.render(justify="left")            # left-aligned
```

#### Variable Naming

- **Tensor variables** render in bold: `name="A"` → `\mathbf{A}`
- **Scalar variables** render plain: `name="x"` → `x`
- **Greek letters** auto-convert: `name="alpha"` → `\alpha`, `name="Psi"` → `\Psi`
- **Subscripts**: `name="x_0"` → `x_0`, `name="x_10"` → `x_{10}`
- **Greek + subscript**: `name="sigma_1"` → `\sigma_1`
- **Greek tensors** use boldsymbol: `name="alpha"` (tensor) → `\boldsymbol{\alpha}`

```python
# Greek letter names are recognized automatically
alpha = struct.lin_alg.tensor(shape=(2,2), name="alpha")
alpha.to_latex()  # => \boldsymbol{\alpha}

psi = struct.lin_alg.scalar(name="psi")
psi.to_latex()    # => \psi

sigma_1 = struct.lin_alg.tensor(shape=(2,2), name="sigma_1")
sigma_1.to_latex()  # => \sigma_1
```

### Eager Evaluation (NumPy)

#### `expr.evaluate(**kwargs) -> numpy scalar or array`

Evaluates the expression DAG using NumPy. Concrete variables (those created with `value=`) are used directly. Abstract variables must be provided as keyword arguments.

```python
import numpy as np

A = struct.lin_alg.tensor(shape=(3,3), dtype=numbers.float64, value=np.eye(3), name="A")
B = struct.lin_alg.tensor(shape=(3,3), name="B")

expr = (A @ B).det() + A.trace()

# B is abstract, so we provide it at evaluation time
result = expr.evaluate(B=np.ones((3,3)))
# => 3.0  (det(I @ ones) = 0, trace(I) = 3)

# Fully concrete expressions need no arguments
A.det().evaluate()
# => 1.0
```

### PyTorch Compute Graphs

#### `expr.to_torch(device="cpu") -> TorchGraph`

Compiles the expression DAG into a callable `TorchGraph`. The graph accepts keyword tensor arguments and returns a tensor result. Supports autograd differentiation.

```python
import torch

A = struct.lin_alg.tensor(shape=(3,3), dtype=numbers.float64, value=np.eye(3), name="A")
B = struct.lin_alg.tensor(shape=(3,3), name="B")
expr = (A @ B).det() + A.trace()

# Compile to torch
graph = expr.to_torch(device="cpu")
print(graph.input_vars)  # => ['B']

# Evaluate with tensor inputs
result = graph(B=torch.ones(3, 3, dtype=torch.float64))
# => tensor(3., dtype=torch.float64)

# GPU acceleration (if available)
graph_gpu = expr.to_torch(device="cuda")
result_gpu = graph_gpu(B=torch.ones(3, 3, dtype=torch.float64, device="cuda"))
```

### Custom Structures

Define custom mathematical structures using `struct.define()`.

#### `struct.define(name, carriers=None, ops=None, axioms=None) -> Structure`

```python
from cm.math.struct.axioms import associativity, closure, identity

# Define a monoid
Monoid = struct.define(
    name="Monoid",
    carriers={"M"},
    ops={
        "*": {"arity": 2},
        "e": {"arity": 0},
    },
    axioms=[
        closure("*", "M"),
        associativity("*"),
        identity("*", "e"),
    ]
)
```

### Available Axiom Constructors

| Constructor | Description |
|-------------|-------------|
| `closure(op, carrier)` | Result of op stays in carrier set |
| `associativity(op)` | op(op(a,b),c) == op(a,op(b,c)) |
| `commutativity(op)` | op(a,b) == op(b,a) |
| `identity(op, elem)` | op(a,e) == a |
| `inverse(op, inv_op, id)` | op(a, inv(a)) == identity |
| `distributivity(op1, op2)` | op1 distributes over op2 |
| `linearity(op)` | op(ax + by) == a*op(x) + b*op(y) |
| `bilinearity(op)` | Linear in both arguments |
| `jacobi_identity(bracket)` | Jacobi identity for Lie brackets |
| `anticommutativity(op)` | op(a,b) == -op(b,a) |

### Symbolic Indexing

The `cm.math.index` module provides symbolic index variables for tensor element access. Indices compose with all backends (LaTeX, NumPy, PyTorch).

#### `index(name=None)`

Create a symbolic index variable. Auto-named if no name is provided.

```python
from cm.math import index

i = index("i")   # named index
j = index("j")
k = index()      # auto-named index
```

#### `index.range(start, stop, step=1)`

Create an `IndexRange` for iteration (same API as `np.arange`).

```python
r = index.range(0, 3)      # 0, 1, 2
r = index.range(0, 10, 2)  # 0, 2, 4, 6, 8
len(r)                      # 5
list(r)                     # [0, 2, 4, 6, 8]
```

#### Tensor Indexing

Use `[]` on any tensor expression to create an `IndexedExpression`. Chained indexing is flattened: `A[i][j]` is equivalent to `A[i, j]`.

```python
from cm.math import struct, index

A = struct.lin_alg.matrix(rows=3, cols=3, name="A")
i = index("i")
j = index("j")

# Symbolic indexing
expr = A[i, j]           # IndexedExpression
expr.base                # Var("A")
expr.indices             # (index('i'), index('j'))
expr.free_indices        # (index('i'), index('j'))

# Mixed indexing
expr2 = A[0, j]
expr2.concrete_indices   # (ScalarExpr(0),)
expr2.free_indices       # (index('j'),)

# LaTeX rendering
expr.to_latex()          # => A_{i, j}
```

#### `IndexedExpression.substitute_indices(**kwargs)`

Replace symbolic indices with concrete values, returning a new `IndexedExpression`.

```python
A[i, j].substitute_indices(i=0)      # => A[0, j]
A[i, j].substitute_indices(i=0, j=1) # => A[0, 1]
```

#### Evaluating Indexed Expressions

Use `bind_indices()` to bind indices to values or ranges, then `evaluate()` to compute results.

```python
import numpy as np
from cm.math import struct, index

A = struct.lin_alg.tensor(
    shape=(3, 3), name="A",
    value=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
)
i = index("i")
j = index("j")

# Single element
result = A[i, j].bind_indices(i=0, j=1).evaluate()
# => 2

# Iterate over a range
results = A[i, 0].bind_indices(i=index.range(0, 3)).evaluate()
# => [IndexResult(i=0) -> 1, IndexResult(i=1) -> 4, IndexResult(i=2) -> 7]

# Cartesian product of ranges
results = A[i, j].bind_indices(
    i=index.range(0, 2),
    j=index.range(0, 2)
).evaluate()
# => [IndexResult(i=0, j=0) -> 1, IndexResult(i=0, j=1) -> 2,
#     IndexResult(i=1, j=0) -> 4, IndexResult(i=1, j=1) -> 5]
```

### Special Functions (struct.spec_func)

#### `struct.spec_func.krok_delta(a, b)`

Kronecker delta function: returns 1 if `a == b`, else 0. Works with all three backends.

```python
from cm.math import struct, index

i = index("i")
j = index("j")

# Symbolic expression
delta = struct.spec_func.krok_delta(i, j)
delta.to_latex()  # => \delta_{i j}

# Evaluate over index ranges
expr = struct.spec_func.krok_delta(i, j)
results = expr.bind_indices(
    i=index.range(0, 3),
    j=index.range(0, 3)
).evaluate()
# => 1.0 when i==j, 0.0 otherwise
```

The Kronecker delta can also be used with indexed tensor expressions:

```python
A = struct.lin_alg.matrix(rows=3, cols=3, name="A")
expr = struct.spec_func.krok_delta(A[i, 0], A[j, 0])
```

### Constraints

Constraints accumulate lazily as operations are applied to abstract expressions. They are checked when `Var.realize()` is called to concretize a variable.

#### `Var.realize(shape=None, dtype=None, value=None)`

Concretize an abstract variable with shape, dtype, or value. Resolves all accumulated constraints and raises `ConstraintError` if any are violated.

```python
from cm.math import struct
import numpy as np

X = struct.lin_alg.tensor(name="X")

# Later, concretize with actual data
X.realize(shape=(3, 3), value=np.eye(3))
X.evaluate()  # => np.eye(3)
```

#### Constraint Functions

| Function | Description |
|----------|-------------|
| `require_rank(rank)` | Shape must have the specified rank |
| `require_square()` | Matrix must be square (rank 2, equal dims) |
| `require_nonsingular()` | Matrix must be nonsingular |

```python
from cm.math.struct.constraints import require_rank, require_square, ConstraintError

X = struct.lin_alg.tensor(name="X")
X.constraints.add(require_rank(2))
X.constraints.add(require_square())

X.realize(shape=(3, 3))  # OK
# X.realize(shape=(3, 4))  # raises ConstraintError: "must be square matrix"
```

### Variable Binding and Substitution

Expressions support three methods for providing variable values, each with different semantics.

#### `expr.bind(**kwargs)`

Store values for later use by `evaluate()` and `to_torch()`. Can be called incrementally. Returns `self` for chaining.

```python
import numpy as np
from cm.math import struct

A = struct.lin_alg.tensor(shape=(2, 2), name="A")
B = struct.lin_alg.tensor(shape=(2, 2), name="B")

expr = (A @ B).det()
expr.bind(A=np.eye(2))
expr.bind(B=np.ones((2, 2)))
expr.evaluate()  # uses bound values

# Chaining
expr.bind(A=np.eye(2)).bind(B=np.ones((2, 2))).evaluate()
```

#### `expr.substitute(**kwargs)`

Rewrite the expression tree, replacing abstract variables with concrete literals. Unlike `bind()`, this produces a new expression where substituted variables are baked in.

```python
import numpy as np
from cm.math import struct, index

A = struct.lin_alg.matrix(rows=2, cols=2, name="A")
B = struct.lin_alg.matrix(rows=2, cols=2, name="B")
i = index("i")
j = index("j")

expr = A[i, j] + B[i, j]
partial = expr.substitute(A=np.array([[1, 2], [3, 4]]))

# A is now concrete, B is still symbolic
partial.to_latex()                           # shows concrete A, symbolic B
partial.evaluate(B=np.array([[5, 6], [7, 8]]), i=0, j=1)  # only B needed
```

#### `expr.bind_indices(index_map=None, **kwargs)`

Bind symbolic index variables to concrete values or `IndexRange` objects. Returns a `BoundIndexExpression` whose `.evaluate()` iterates over any ranges (cartesian product).

```python
from cm.math import struct, index

i = index("i")
j = index("j")
A = struct.lin_alg.matrix(rows=3, cols=3, name="A")

# kwargs form (names must match index var_names)
bound = A[i, j].bind_indices(i=0, j=index.range(0, 3))
results = bound.evaluate(A=np.eye(3))

# dict form (keys are index objects)
bound = A[i, j].bind_indices({i: 0, j: index.range(0, 3)})
```

### Symbolic Tensors

When `struct.lin_alg.tensor(shape=...)` is called without a `value`, it returns a `SymbolicTensor` — a tensor whose elements are symbolic `Expression` nodes. Elements are stored sparsely (unset elements default to zero).

#### Element Assignment

Use chained indexing to assign expression elements:

```python
from cm.math import struct

y = struct.lin_alg.scalar(name="y")

x = struct.lin_alg.tensor(shape=(2,2))
x[0][0] = y ** 2 + 5
x[0][1] = y ** 3
x[1][0] = y
x[1][1] = y ** 2
```

#### Symbolic Determinant and Trace

```python
d = x.det()       # cofactor expansion → single Expression DAG
t = x.trace()     # sum of diagonal elements

d.to_latex()       # LaTeX rendering of the symbolic determinant
d.bind(y=3).evaluate()   # bind variables, then evaluate to a number
```

#### Symbolic Matrix Multiplication

```python
w = struct.lin_alg.tensor(shape=(2,2))
w[0][0] = y ** 2
w[0][1] = y
w[1][0] = y + 1
w[1][1] = y ** 3

z = x @ w          # symbolic matmul — each element is an Expression
z.bind(y=2).evaluate()   # evaluate to numpy array
```

#### Evaluation and Rendering

```python
# Bind variables and evaluate to numpy array
result = x.bind(y=3).evaluate()

# Or pass bindings directly to evaluate
result = x.evaluate(y=3)

# Render as LaTeX pmatrix
x.to_latex()       # \begin{pmatrix} ... \end{pmatrix}
x.render()         # HTML output via cm.views
```

### Mathematical Functions (struct.lin_alg.fxn)

The `fxn` sub-module provides elementary mathematical functions that operate on scalar expressions. Each function returns a new `Expression` node in the DAG with full support for all three backends (eager/NumPy, PyTorch, LaTeX).

```python
from cm.math import struct
from cm.math.struct.lin_alg import fxn

y = struct.lin_alg.scalar(name="y")
```

#### Available Functions

| Function | Description | LaTeX |
|----------|-------------|-------|
| `fxn.sin(expr)` | Sine | `\sin\left(\cdot\right)` |
| `fxn.cos(expr)` | Cosine | `\cos\left(\cdot\right)` |
| `fxn.tan(expr)` | Tangent | `\tan\left(\cdot\right)` |
| `fxn.exp(expr)` | Exponential | `\exp\left(\cdot\right)` |
| `fxn.log(expr)` | Natural logarithm | `\ln\left(\cdot\right)` |
| `fxn.sqrt(expr)` | Square root | `\sqrt{\cdot}` |
| `fxn.abs(expr)` | Absolute value | `\left\|\cdot\right\|` |
| `fxn.asin(expr)` | Inverse sine | `\arcsin\left(\cdot\right)` |
| `fxn.acos(expr)` | Inverse cosine | `\arccos\left(\cdot\right)` |
| `fxn.atan(expr)` | Inverse tangent | `\arctan\left(\cdot\right)` |
| `fxn.sinh(expr)` | Hyperbolic sine | `\sinh\left(\cdot\right)` |
| `fxn.cosh(expr)` | Hyperbolic cosine | `\cosh\left(\cdot\right)` |
| `fxn.tanh(expr)` | Hyperbolic tangent | `\tanh\left(\cdot\right)` |

#### Usage with Expressions

Functions compose naturally with arithmetic and can be used as tensor elements:

```python
expr = fxn.sin(y) ** 2 + fxn.cos(y) ** 2   # Pythagorean identity
expr.bind(y=1.0).evaluate()                  # => 1.0

# As tensor elements
x = struct.lin_alg.tensor(shape=(2,2))
x[0][0] = fxn.exp(y)
x[1][1] = fxn.cos(y)
x[0][1] = y ** 3
x[1][0] = fxn.sin(y)

x.det().bind(y=0.5).evaluate()
```

### Operators (cm.math.operator)

The operator module provides operator objects that transform expressions when applied. Operators integrate with `SymbolicTensor` matrix multiplication — when an operator element is matmul'd with an expression element, it calls `.apply()` instead of standard multiplication.

```python
from cm.math import struct, operator as op
```

#### `op.derivative(var, order=1)`

Create a derivative operator with respect to a variable.

```python
y = struct.lin_alg.scalar(name="y")

d_dy = op.derivative(y)        # first derivative d/dy
d2_dy2 = op.derivative(y, 2)   # second derivative d²/dy²
```

#### Applying Operators

Operators can be applied directly to expressions:

```python
from cm.math.struct.lin_alg import fxn

expr = y ** 3 + fxn.sin(y)
result = d_dy.apply(expr)       # 3y² + cos(y)
result.bind(y=1.0).evaluate()
```

#### Operator Algebra with Tensors

Operators can be placed as elements of a `SymbolicTensor`. When the tensor is matmul'd with another tensor, operator elements differentiate the corresponding elements instead of multiplying:

```python
# Operator matrix
J = struct.lin_alg.tensor(shape=(2,2))
J[0][0] = op.derivative(y) + y ** 2    # (d/dy + y²)
J[0][1] = op.derivative(y, 2)          # d²/dy²
J[1][0] = op.derivative(y)             # d/dy
J[1][1] = op.derivative(y)             # d/dy

# Expression matrix
x = struct.lin_alg.tensor(shape=(2,1))
x[0][0] = y ** 3
x[1][0] = fxn.sin(y)

# Matmul applies operators to expressions
result = J @ x
result.bind(y=2.0).evaluate()
```

When an operator has a scalar part (e.g., `op.derivative(y) + y**2`), applying it to `f` produces `f' + y²·f`.

#### Supported Differentiation Rules

The symbolic differentiation engine supports:

| Rule | Example |
|------|---------|
| Power rule | `d/dy(y³) = 3y²` |
| Product rule | `d/dy(f·g) = f'·g + f·g'` |
| Quotient rule | `d/dy(f/g) = (f'·g - f·g') / g²` |
| Chain rule | `d/dy(sin(y²)) = cos(y²)·2y` |
| Linearity | `d/dy(f + g) = f' + g'` |
| All `fxn` functions | `sin, cos, tan, exp, log, sqrt, asin, acos, atan, sinh, cosh, tanh` |

---

## cm.qm Module

The quantum mechanics module provides tools for working with atoms, molecules, Slater determinants, spin-orbitals, and matrix elements. It includes a powerful Hamiltonian builder for configuring terms and corrections, with support for both non-relativistic and relativistic calculations.

```python
from cm import qm
```

### Atoms

Create atoms with automatic electron configuration based on the aufbau principle.

#### `qm.atom(element, position=(0,0,0), configuration=None, relativistic=False, charge=0)`

Create an atom with nuclear charge and electron configuration.

**Parameters:**
- `element`: Element symbol (str) or atomic number (int)
- `position`: (x, y, z) coordinates or Coordinate3D for symbolic geometry
- `configuration`: Optional ElectronConfiguration (defaults to aufbau ground state)
- `relativistic`: If True, use Dirac spinors instead of spin-orbitals
- `charge`: Ion charge (positive for cations, negative for anions)

```python
from cm import qm

# Create atoms by symbol
C = qm.atom('C')
print(C.Z)  # 6
print(C.symbol)  # 'C'
print(C.n_electrons)  # 6

# Create with position
H1 = qm.atom('H', position=(0, 0, 0))
H2 = qm.atom('H', position=(0.74, 0, 0))

# Create ions
Fe2_plus = qm.atom('Fe', charge=2)  # Fe²⁺
O2_minus = qm.atom('O', charge=-2)  # O²⁻

# Create relativistic atoms (for heavy elements)
Au = qm.atom('Au', relativistic=True)

# Get electron configuration
print(C.configuration.label)  # "1s² 2s² 2p²"
```

#### `qm.atoms(specs)`

Create multiple atoms from a list of specifications.

```python
from cm import qm

# Create water molecule atoms
atoms = qm.atoms([
    ('O', 0.000, 0.000, 0.117),
    ('H', 0.756, 0.000, -0.469),
    ('H', -0.756, 0.000, -0.469),
])

# With charges
ions = qm.atoms([
    ('Na', 0, 0, 0, 1),   # Na⁺
    ('Cl', 2.8, 0, 0, -1), # Cl⁻
])
```

#### Atom Methods

```python
from cm import qm

C = qm.atom('C')

# Get orbitals (non-relativistic)
orbitals = C.orbitals  # List of SpinOrbital

# Get spinors (relativistic)
Au = qm.atom('Au', relativistic=True)
spinors = Au.spinors  # List of DiracSpinor

# Generate Slater determinant
psi = C.slater_determinant()
psi.render()

# Generate Dirac determinant (relativistic)
psi_rel = Au.dirac_determinant()

# Convenient determinant() method chooses based on relativistic flag
psi = C.determinant()  # SlaterDeterminant
psi = Au.determinant()  # DiracDeterminant

# Modify atom (returns new atom)
C_excited = C.excite(from_orbital=(2, 0, 0, 1), to_orbital=(2, 1, 0, 1))
C_ion = C.ionize()  # C⁺
C_moved = C.with_position((1, 0, 0))
C_rel = C.to_relativistic()

# Visualization tuple for cm.views.molecule()
print(C.to_molecule_tuple())  # ('C', 0, 0, 0)
```

### Electron Configurations

Control electron configurations manually or use aufbau defaults.

#### `qm.ground_state(n_electrons)`

Get the ground state configuration for a given electron count.

```python
from cm import qm

config = qm.ground_state(6)  # Carbon
print(config.label)  # "1s² 2s² 2p²"
print(config.n_electrons)  # 6
```

#### `qm.config_from_string(notation)`

Parse electron configuration from standard notation.

```python
from cm import qm

# Standard notation
config = qm.config_from_string("1s2 2s2 2p2")

# Noble gas core notation
config = qm.config_from_string("[He] 2s2 2p2")
config = qm.config_from_string("[Ne] 3s2 3p4")
config = qm.config_from_string("[Ar] 3d10 4s2")

# Use with atom
C_excited = qm.atom('C', configuration=qm.config_from_string("1s2 2s1 2p3"))
```

#### `ElectronConfiguration` Class

```python
from cm import qm

# Create manually
config = qm.ElectronConfiguration.aufbau(6)  # 6 electrons
config = qm.ElectronConfiguration.from_string("1s2 2s2 2p2")

# Properties
print(config.n_electrons)  # 6
print(config.label)  # "1s² 2s² 2p²"
print(config.latex_label)  # "1s^2\\,2s^2\\,2p^2"
print(config.orbitals)  # List of (n, l, m, spin) tuples

# Get relativistic spinor quantum numbers
print(config.spinors)  # List of (n, kappa, mj) tuples

# Modify (returns new configuration)
excited = config.excite(from_orbital=(2, 0, 0, 1), to_orbital=(2, 1, 0, 1))
ionized = config.ionize(n=1)  # Remove 1 electron
added = config.add_electron()  # Add electron to next available
```

#### Element Data

```python
from cm import qm

# Atomic numbers dict
print(qm.ATOMIC_NUMBERS['C'])  # 6
print(qm.ATOMIC_NUMBERS['Au'])  # 79

# Element symbols dict
print(qm.ELEMENT_SYMBOLS[6])   # 'C'
print(qm.ELEMENT_SYMBOLS[79])  # 'Au'

# Aufbau filling order
print(qm.AUFBAU_ORDER)  # [(1,0), (2,0), (2,1), (3,0), ...]
```

### Molecules

Create molecules from multiple atoms for multi-center calculations.

#### `qm.molecule(atoms_with_positions)`

Create a molecule from atom specifications.

**Parameters:**
- `atoms_with_positions`: List of tuples (element, x, y, z) or (element, x, y, z, charge)

```python
from cm import qm

# Water molecule
H2O = qm.molecule([
    ('O', 0.000, 0.000, 0.117),
    ('H', 0.756, 0.000, -0.469),
    ('H', -0.756, 0.000, -0.469),
])

# Access atoms
print(H2O.atoms)  # List of Atom objects
print(H2O.n_electrons)  # Total electrons
print(H2O.n_atoms)  # 3

# Generate molecular Slater determinant
psi = H2O.slater_determinant()

# Symbolic geometry for optimization
from cm import spherical_coord
r = qm.Var('r')
theta = qm.Var('theta')
H2_sym = qm.molecule([
    ('H', 0, 0, 0),
    ('H', r, 0, 0),  # Symbolic bond length
])
```

#### `Molecule` Class

```python
from cm import qm

mol = qm.molecule([('H', 0, 0, 0), ('H', 0.74, 0, 0)])

# Properties
print(mol.atoms)  # List of Atom objects
print(mol.n_atoms)  # 2
print(mol.n_electrons)  # 2
print(mol.positions)  # List of (x, y, z) tuples
print(mol.geometry)  # Geometry summary

# Determinants
psi = mol.slater_determinant()
psi_rel = mol.dirac_determinant()  # If all atoms are relativistic

# Visualization
print(mol.to_molecule_tuples())  # [('H', 0, 0, 0), ('H', 0.74, 0, 0)]
mol.render()  # Uses cm.views.molecule()

# Modify (returns new molecule)
stretched = mol.with_geometry([('H', 0, 0, 0), ('H', 1.0, 0, 0)])
```

### Spin-Orbitals

Create spin-orbital basis elements using quantum numbers.

#### `qm.basis_orbital(spec, vec3=None, t=None)`

Create a single spin-orbital from a tuple specification.

**Parameters:**
- `spec`: Tuple of `(n, l, m, spin)` or `(l, m, spin)` where:
  - `n`: Principal quantum number (optional)
  - `l`: Angular momentum quantum number (0=s, 1=p, 2=d, 3=f, ...)
  - `m`: Magnetic quantum number (-l ≤ m ≤ l)
  - `spin`: +1 for spin-up (α), -1 for spin-down (β)
- `vec3`: Optional coordinate (defaults to spherical coordinates)
- `t`: Optional time parameter

```python
from cm import qm

# 1s spin-up orbital: (n, l, m, spin)
orbital_1s_up = qm.basis_orbital((1, 0, 0, 1))

# 2p spin-down with m=-1
orbital_2p_down = qm.basis_orbital((2, 1, -1, -1))
```

#### `qm.basis_orbitals(specs, vec3=None, t=None)`

Create multiple spin-orbitals from a list of tuple specifications.

**Parameters:**
- `specs`: List of tuples `(n, l, m, spin)` or `(l, m, spin)`

```python
from cm import qm

# Helium ground state: 1s↑ 1s↓
orbitals = qm.basis_orbitals([(1, 0, 0, 1), (1, 0, 0, -1)])

# Lithium: 1s↑ 1s↓ 2s↑
orbitals = qm.basis_orbitals([
    (1, 0, 0, 1),   # 1s↑
    (1, 0, 0, -1),  # 1s↓
    (2, 0, 0, 1)    # 2s↑
])
```

### Slater Determinants

#### `qm.slater(orbitals)`

Create a Slater determinant from a list of spin-orbitals.

```python
from cm import qm

# Create orbitals using basis_orbitals
orbitals = qm.basis_orbitals([
    (1, 0, 0, 1),   # 1s↑
    (1, 0, 0, -1),  # 1s↓
])

# Create Slater determinant
psi = qm.slater(orbitals)
psi.render()  # Renders: |1s↑, 1s↓⟩
```

The Slater determinant automatically handles:
- Pauli exclusion principle (raises error for duplicate orbitals)
- Antisymmetry under particle exchange
- Orthogonality between spin-orbitals

### Overlaps with @ Operator

Use the `@` operator to compute overlaps ⟨ψ|φ⟩ between Slater determinants.

```python
from cm import qm

# Same orbitals - overlap is 1
orbitals1 = qm.basis_orbitals([(1, 0, 0, 1), (1, 0, 0, -1)])
psi = qm.slater(orbitals1)
phi = qm.slater(orbitals1)
overlap = psi @ phi
print(overlap.value)  # 1

# Different orbitals - overlap is 0 (orthogonal)
orbitals2 = qm.basis_orbitals([(2, 0, 0, 1), (2, 0, 0, -1)])
chi = qm.slater(orbitals2)
overlap = psi @ chi
print(overlap.value)  # 0

# Render the overlap
(psi @ phi).render()
```

### Hamiltonian Matrix Elements

Use the `@` operator with a Hamiltonian to compute matrix elements ⟨ψ|Ĥ|φ⟩.

#### `qm.hamiltonian(symbol="H")`

Create a Hamiltonian operator.

```python
from cm import qm

H = qm.hamiltonian()
# Or use the pre-defined alias:
# H = qm.H
```

#### Computing Matrix Elements

```python
from cm import qm

# Create determinants using basis_orbitals: (n, l, m, spin)
orbitals_A = qm.basis_orbitals([(1, 0, 0, 1), (1, 0, 0, -1)])  # 1s↑, 1s↓
orbitals_B = qm.basis_orbitals([(2, 0, 0, 1), (2, 0, 0, -1)])  # 2s↑, 2s↓
psi = qm.slater(orbitals_A)
phi = qm.slater(orbitals_B)

H = qm.hamiltonian()

# Compute matrix element ⟨ψ|Ĥ|φ⟩
matrix_elem = psi @ H @ phi
matrix_elem.render()
```

### Slater-Condon Rules

Matrix elements are automatically simplified using Slater-Condon rules based on the number of orbital differences:

| Excitations | Result |
|-------------|--------|
| 0 (diagonal) | ∑ᵢ ⟨i|ĥ|i⟩ + ½∑ᵢ≠ⱼ [⟨ij|ĝ|ij⟩ - ⟨ij|ĝ|ji⟩] |
| 1 (single) | ⟨p|ĥ|q⟩ + ∑ⱼ [⟨pj|ĝ|qj⟩ - ⟨pj|ĝ|jq⟩] |
| 2 (double) | ⟨pq|ĝ|rs⟩ - ⟨pq|ĝ|sr⟩ |
| 3+ | 0 |

```python
from cm import qm

# Define orbitals for H₂ molecule using basis_orbital: (n, l, m, spin)
# n=1 for atom A, n=2 for atom B (using n as atom index)
orbital_1sA_up = qm.basis_orbital((1, 0, 0, 1))    # atom A, 1s↑
orbital_1sA_down = qm.basis_orbital((1, 0, 0, -1)) # atom A, 1s↓
orbital_1sB_up = qm.basis_orbital((2, 0, 0, 1))    # atom B, 1s↑
orbital_1sB_down = qm.basis_orbital((2, 0, 0, -1)) # atom B, 1s↓

# Ionic configurations
ionic_A = qm.slater([orbital_1sA_up, orbital_1sA_down])
ionic_B = qm.slater([orbital_1sB_up, orbital_1sB_down])

# Covalent configuration
covalent = qm.slater([orbital_1sA_up, orbital_1sB_down])

H = qm.hamiltonian()

# Diagonal element (0 excitations)
(ionic_A @ H @ ionic_A).render()

# Single excitation
(ionic_A @ H @ covalent).render()

# Double excitation
(ionic_A @ H @ ionic_B).render()
```

### One-Electron and Two-Electron Operators

For more specific operators:

```python
from cm import qm

h = qm.one_electron_operator("h")   # Kinetic + nuclear attraction
g = qm.two_electron_operator("g")   # Electron-electron repulsion
```

### Complete H₂ Example

```python
# %%
from cm import qm
from cm.views import html

html("<h2>H₂ Molecule - Slater Determinant Basis</h2>")

# Define atomic orbitals: 1s on atom A (n=1) and atom B (n=2)
# Using basis_orbital with (n, l, m, spin) - n serves as atom index
orbital_1sA_up = qm.basis_orbital((1, 0, 0, 1))    # atom A, 1s↑
orbital_1sA_down = qm.basis_orbital((1, 0, 0, -1)) # atom A, 1s↓
orbital_1sB_up = qm.basis_orbital((2, 0, 0, 1))    # atom B, 1s↑
orbital_1sB_down = qm.basis_orbital((2, 0, 0, -1)) # atom B, 1s↓

# Ionic configurations (both electrons on one atom)
html("<h3>Ionic Configurations</h3>")
config_ionic_A = qm.slater([orbital_1sA_up, orbital_1sA_down])
config_ionic_B = qm.slater([orbital_1sB_up, orbital_1sB_down])
html("<p>H⁻ₐ H⁺ᵦ:</p>")
config_ionic_A.render()
html("<p>H⁺ₐ H⁻ᵦ:</p>")
config_ionic_B.render()

# Covalent configurations (one electron on each atom)
html("<h3>Covalent Configurations</h3>")
config_cov_1 = qm.slater([orbital_1sA_up, orbital_1sB_down])
config_cov_2 = qm.slater([orbital_1sA_down, orbital_1sB_up])
config_cov_1.render()
config_cov_2.render()

# Overlaps
html("<h3>Overlap Matrix Elements</h3>")
html(f"<p>⟨ionic_A|ionic_A⟩ = {(config_ionic_A @ config_ionic_A).value}</p>")
html(f"<p>⟨ionic_A|ionic_B⟩ = {(config_ionic_A @ config_ionic_B).value}</p>")
html(f"<p>⟨ionic_A|covalent⟩ = {(config_ionic_A @ config_cov_1).value}</p>")

# Hamiltonian matrix elements
html("<h3>Hamiltonian Matrix Elements</h3>")
H = qm.hamiltonian()

html("<p>Diagonal ⟨Ψ|H|Ψ⟩:</p>")
(config_ionic_A @ H @ config_ionic_A).render()

html("<p>Single excitation ⟨ionic|H|covalent⟩:</p>")
(config_ionic_A @ H @ config_cov_1).render()

html("<p>Double excitation ⟨ionic_A|H|ionic_B⟩:</p>")
(config_ionic_A @ H @ config_ionic_B).render()
```

---

### Relativistic Quantum Mechanics

For heavy atoms where relativistic effects are significant, use four-component Dirac spinors instead of non-relativistic spin-orbitals.

#### The κ Quantum Number

In relativistic QM, spin-orbit coupling is built-in. The κ quantum number encodes both orbital (l) and total (j) angular momentum:

| Orbital | l | j | κ | States |
|---------|---|---|---|--------|
| s₁/₂ | 0 | 1/2 | -1 | 2 |
| p₁/₂ | 1 | 1/2 | +1 | 2 |
| p₃/₂ | 1 | 3/2 | -2 | 4 |
| d₃/₂ | 2 | 3/2 | +2 | 4 |
| d₅/₂ | 2 | 5/2 | -3 | 6 |

**Formula:** κ = -(l+1) for j = l+1/2, κ = +l for j = l-1/2

#### `qm.dirac_spinor(n, kappa, mj)`

Create a Dirac spinor using the κ quantum number.

```python
from cm import qm

# 1s₁/₂ with mⱼ = +1/2
spinor_1s = qm.dirac_spinor(n=1, kappa=-1, mj=0.5)

# 2p₃/₂ with mⱼ = -3/2
spinor_2p = qm.dirac_spinor(n=2, kappa=-2, mj=-1.5)

print(spinor_1s.j)  # 0.5
print(spinor_1s.l)  # 0
print(spinor_2p.j)  # 1.5
print(spinor_2p.l)  # 1
```

#### `qm.dirac_spinor_lj(n, l, j, mj)`

Create a Dirac spinor using the more intuitive (n, l, j, mⱼ) notation.

```python
from cm import qm

# 2p₃/₂ with mⱼ = +1/2
spinor = qm.dirac_spinor_lj(n=2, l=1, j=1.5, mj=0.5)
print(spinor.kappa)  # -2
```

#### `qm.basis_dirac(quantum_numbers)`

Create multiple Dirac spinors from tuples.

```python
from cm import qm

# Using (n, κ, mⱼ) format
spinors = qm.basis_dirac([
    (1, -1, 0.5),   # 1s₁/₂ mⱼ=+1/2
    (1, -1, -0.5),  # 1s₁/₂ mⱼ=-1/2
])

# Using (n, l, j, mⱼ) format
spinors = qm.basis_dirac([
    (2, 1, 1.5, 0.5),   # 2p₃/₂ mⱼ=+1/2
    (2, 1, 1.5, -0.5),  # 2p₃/₂ mⱼ=-1/2
])
```

#### `qm.dirac_slater(spinors)`

Create a Slater determinant of Dirac spinors.

```python
from cm import qm

spinors = qm.basis_dirac([(1, -1, 0.5), (1, -1, -0.5)])
psi = qm.dirac_slater(spinors)
psi.render()  # |n, κ, mⱼ⟩ notation
psi.render(notation="spectroscopic")  # |1s_{1/2}(+1/2), ...⟩
```

#### Dirac Hamiltonians

```python
from cm import qm

# Dirac-Coulomb (standard relativistic)
H_DC = qm.dirac_hamiltonian("coulomb")

# Dirac-Coulomb-Gaunt (adds magnetic interaction)
H_DCG = qm.dirac_hamiltonian("coulomb_gaunt")

# Dirac-Coulomb-Breit (most accurate - includes retardation)
H_DCB = qm.dirac_hamiltonian("coulomb_breit")

# Pre-defined aliases
H_DC = qm.H_DC    # Dirac-Coulomb
H_DCB = qm.H_DCB  # Dirac-Coulomb-Breit
```

#### Relativistic Matrix Elements

```python
from cm import qm

# Create configurations
spinor_1s_up = qm.dirac_spinor(n=1, kappa=-1, mj=0.5)
spinor_1s_down = qm.dirac_spinor(n=1, kappa=-1, mj=-0.5)
config_1s2 = qm.dirac_slater([spinor_1s_up, spinor_1s_down])

spinor_2p_up = qm.dirac_spinor(n=2, kappa=1, mj=0.5)
spinor_2p_down = qm.dirac_spinor(n=2, kappa=1, mj=-0.5)
config_2p2 = qm.dirac_slater([spinor_2p_up, spinor_2p_down])

# Dirac-Coulomb-Breit Hamiltonian
H_DCB = qm.dirac_hamiltonian("coulomb_breit")

# Matrix elements (same Slater-Condon rules apply)
(config_1s2 @ H_DCB @ config_1s2).render()  # Diagonal
(config_1s2 @ H_DCB @ config_2p2).render()  # Double excitation

# Overlaps
overlap = config_1s2 @ config_1s2
print(overlap.value)  # 1
```

---

### Hamiltonian Builder

Build configurable Hamiltonians with a fluent API. Add physical terms, corrections, and approximations.

#### `qm.HamiltonianBuilder()`

Create a Hamiltonian builder with fluent configuration methods.

```python
from cm import qm

# Build a standard electronic Hamiltonian
H = (qm.HamiltonianBuilder()
     .with_kinetic()
     .with_nuclear_attraction()
     .with_coulomb()  # Includes exchange automatically
     .build())

# Use presets for common configurations
H_elec = qm.HamiltonianBuilder.electronic()  # Kinetic + nuclear + Coulomb
H_so = qm.HamiltonianBuilder.spin_orbit()     # Includes spin-orbit coupling
H_rel = qm.HamiltonianBuilder.relativistic()  # Full relativistic corrections
```

#### Builder Methods

```python
from cm import qm

builder = qm.HamiltonianBuilder()

# Core terms
builder.with_kinetic(mass=1.0)  # -1/(2m) ∇²
builder.with_nuclear_attraction()  # -∑ Zₐ/rᵢₐ
builder.with_coulomb()  # ∑ 1/rᵢⱼ + exchange

# Spin-orbit and relativistic
builder.with_spin_orbit(model='zeff')  # ξ(r) L·S
builder.with_spin_orbit(model='sommerfeld')  # Full Sommerfeld
builder.with_relativistic(correction='breit')  # Breit interaction
builder.with_relativistic(correction='darwin')  # Darwin term
builder.with_relativistic(correction='mass-velocity')  # Mass-velocity

# External fields
builder.with_external_field(field_type='electric', strength=0.01, direction=(0, 0, 1))
builder.with_external_field(field_type='magnetic', strength=0.001, direction=(0, 0, 1))

# Custom terms
builder.with_custom(term=qm.HamiltonianTerm(
    name="custom_potential",
    symbol="V",
    expression=some_expr
))

# Modify terms
builder.scale('coulomb', factor=0.5)  # Scale a term
builder.remove('spin_orbit')  # Remove a term

# Build the Hamiltonian
H = builder.build()
```

#### `HamiltonianTerm` Dataclass

```python
from cm import qm

# Create custom terms
term = qm.HamiltonianTerm(
    name="spin_orbit",
    symbol="H_{SO}",
    expression=spin_orbit_expr,  # Symbolic expression
    type="one_electron",  # or "two_electron"
)
```

### Matrix Expressions

Matrix elements return `MatrixExpression` objects that support analytical, numerical, and graph evaluation.

#### `MatrixExpression` Class

```python
from cm import qm

# Create determinants and Hamiltonian
psi = qm.atom('C').slater_determinant()
H = qm.HamiltonianBuilder.electronic().build()

# Compute matrix element
elem = psi @ H @ psi  # Returns MatrixExpression

# Analytical form (symbolic)
expr = elem.analytical()
expr.render()  # Display LaTeX

# Numerical evaluation
energy = elem.numerical(Z=6, a0=1.0)  # Pass variable values

# Lazy evaluation with ComputeGraph
cg = elem.graph(Z=6)
cg.render()  # Visualize graph
result = cg.evaluate()

# PyTorch compilation for GPU
torch_fn = elem.compile(device='cuda')
energies = torch_fn(Z=torch.tensor([6, 7, 8]))
```

#### `MolecularHamiltonian` Class

Apply Hamiltonians to molecules.

```python
from cm import qm

mol = qm.molecule([('H', 0, 0, 0), ('H', 0.74, 0, 0)])
H = qm.HamiltonianBuilder.electronic().build()

# Create molecular Hamiltonian
mol_H = qm.MolecularHamiltonian(H, mol)

# Generate Hamiltonian matrix over CI basis
basis = mol.ci_basis(excitations=2)
matrix = mol_H.matrix(basis)  # Returns HamiltonianMatrix

# Diagonalize
eigenvalues, eigenvectors = matrix.diagonalize()
ground_state_energy = eigenvalues[0]
```

#### `HamiltonianMatrix` Class

```python
from cm import qm

mol = qm.molecule([('H', 0, 0, 0), ('H', 0.74, 0, 0)])
basis = mol.ci_basis(excitations=2)
H = qm.HamiltonianBuilder.electronic().build()
mol_H = qm.MolecularHamiltonian(H, mol)

matrix = mol_H.matrix(basis)

# Properties
print(matrix.dimension)  # Size of matrix
print(matrix.basis)  # List of determinants

# Get individual elements
elem = matrix[0, 1]  # MatrixExpression

# Evaluate all elements numerically
numpy_matrix = matrix.to_numpy(R=0.74)

# Diagonalize
eigenvalues, eigenvectors = matrix.diagonalize(R=0.74)

# Render as LaTeX
matrix.render()
```

### CI Basis Generation

Generate configuration interaction basis sets from molecules.

#### `Molecule.ci_basis()`

Generate excited determinants for CI calculations.

```python
from cm import qm

mol = qm.molecule([('H', 0, 0, 0), ('H', 0.74, 0, 0)])

# Singles and doubles (CIS-D)
basis = mol.ci_basis(excitations=2)
print(len(basis))  # Number of determinants

# Singles only (CIS)
basis_s = mol.ci_basis(excitations=1)

# Full CI (all excitations)
basis_full = mol.ci_basis(excitations='full')

# With frozen core
basis = mol.ci_basis(excitations=2, frozen_core=2)  # Freeze 2 electrons

# Active space
basis = mol.ci_basis(
    excitations=2,
    active_space=(4, 6)  # (n_electrons, n_orbitals)
)

# Iterate over basis
for det in basis:
    det.render()
```

### Molecular Integrals (cm.qm.integrals)

The `cm.qm.integrals` submodule provides proper evaluation of molecular integrals using Gaussian-type orbitals (GTOs) with the McMurchie-Davidson scheme. This enables ab initio quantum chemistry calculations including Hartree-Fock and coupled cluster methods.

#### Basis Sets

Three basis sets are available with increasing accuracy:

| Basis Set | Type | Angular Functions | Accuracy | Use Case |
|-----------|------|-------------------|----------|----------|
| STO-3G | Minimal | s, p | ~85% correlation | Teaching, qualitative |
| cc-pVTZ | Triple-zeta | s, p, d, f | ~95% correlation | Publication quality |
| cc-pVQZ | Quadruple-zeta | s, p, d, f, g | ~98% correlation | Benchmark accuracy |

```python
from cm.qm.integrals import BasisSet

# Load basis set for a molecule
basis = BasisSet("cc-pVTZ")
basis.add_atom('H', (0.0, 0.0, 0.0))
basis.add_atom('H', (0.74, 0.0, 0.0))

print(f"Number of basis functions: {basis.n_basis}")  # 28 for H2/cc-pVTZ

# Available basis sets
basis_sto3g = BasisSet("STO-3G")      # Minimal basis (3 Gaussians per STO)
basis_pvtz = BasisSet("cc-pVTZ")      # Triple-zeta correlation consistent
basis_pvqz = BasisSet("cc-pVQZ")      # Quadruple-zeta (includes g functions)
```

#### Gaussian Primitives and Contracted Functions

```python
from cm.qm.integrals import GaussianPrimitive, ContractedGaussian, BasisFunction

# Single Gaussian primitive: g(r) = N * x^i * y^j * z^k * exp(-α|r-R|²)
primitive = GaussianPrimitive(
    exponent=0.5,           # α (Gaussian exponent)
    center=(0.0, 0.0, 0.0), # R (center position)
    angular=(0, 0, 0)       # (i, j, k) for s-type
)

# p-type Gaussian (px)
px = GaussianPrimitive(exponent=0.3, center=(0, 0, 0), angular=(1, 0, 0))

# d-type Gaussian (dxy)
dxy = GaussianPrimitive(exponent=0.2, center=(0, 0, 0), angular=(1, 1, 0))

# Contracted Gaussian: χ = Σ cᵢ gᵢ(r)
contracted = ContractedGaussian([
    (0.15432897, GaussianPrimitive(3.42525091, (0, 0, 0), (0, 0, 0))),
    (0.53532814, GaussianPrimitive(0.62391373, (0, 0, 0), (0, 0, 0))),
    (0.44463454, GaussianPrimitive(0.16885540, (0, 0, 0), (0, 0, 0))),
])
```

#### One-Electron Integrals

```python
from cm.qm.integrals import (
    overlap_integral, overlap_matrix,
    kinetic_integral, kinetic_matrix,
    nuclear_attraction_integral, nuclear_attraction_matrix,
)

# Build basis
basis = BasisSet("cc-pVTZ")
basis.add_atom('H', (0.0, 0.0, 0.0))
basis.add_atom('H', (0.74, 0.0, 0.0))

# Overlap matrix: S_μν = ⟨μ|ν⟩
S = overlap_matrix(basis)

# Kinetic energy matrix: T_μν = ⟨μ|-½∇²|ν⟩
T = kinetic_matrix(basis)

# Nuclear attraction matrix: V_μν = ⟨μ|Σ -Z_A/|r-R_A||ν⟩
nuclei = [('H', (0.0, 0.0, 0.0)), ('H', (0.74, 0.0, 0.0))]
V = nuclear_attraction_matrix(basis, nuclei)

# Core Hamiltonian
H_core = T + V
```

#### Two-Electron Integrals (ERI)

```python
from cm.qm.integrals import (
    electron_repulsion_integral,
    eri_tensor,
    eri_tensor_screened,
    eri_tensor_optimized,
    compute_schwarz_bounds,
)

# Full ERI tensor: (μν|λσ) = ∫∫ μ(r₁)ν(r₁) 1/r₁₂ λ(r₂)σ(r₂) dr₁dr₂
ERI = eri_tensor(basis)  # O(N⁴) storage

# Optimized with Schwarz screening and 8-fold symmetry
ERI_opt, stats = eri_tensor_screened(basis, threshold=1e-10)
print(f"Integrals computed: {stats['computed']} / {stats['total']}")
print(f"Screening efficiency: {stats['screened_percent']:.1f}%")

# Auto-select best method (GPU if available)
ERI = eri_tensor_optimized(basis, threshold=1e-10, use_gpu=True)

# Schwarz bounds for screening: Q_μν = √(μν|μν)
Q = compute_schwarz_bounds(basis)
```

#### Direct SCF (Avoiding Full ERI Storage)

```python
from cm.qm.integrals import eri_direct

# For large molecules, compute J and K directly without storing ERI tensor
D = density_matrix  # From previous SCF iteration
J, K = eri_direct(basis, D, threshold=1e-10)

# Fock matrix: F = H_core + J - 0.5*K (for RHF)
F = H_core + J - 0.5 * K
```

#### Hartree-Fock Solver

```python
from cm.qm.integrals import HartreeFockSolver, hartree_fock, HFResult

# Quick interface
nuclei = [('H', (0.0, 0.0, 0.0)), ('H', (0.74, 0.0, 0.0))]
result = hartree_fock(
    nuclei,
    basis="cc-pVTZ",
    n_electrons=2,
    max_iterations=100,
    convergence=1e-8,
    verbose=True
)

print(f"HF Energy: {result.energy:.6f} Hartree")
print(f"Converged: {result.converged} in {result.n_iterations} iterations")
print(f"Orbital energies: {result.orbital_energies}")

# Access MO coefficients and density matrix
C = result.C          # MO coefficient matrix
D = result.D          # Density matrix
F = result.F          # Fock matrix
S = result.S          # Overlap matrix
ERI = result.eri      # Two-electron integrals

# Full solver class for more control
solver = HartreeFockSolver(nuclei, basis="cc-pVTZ", n_electrons=2)
solver.compute_integrals()
result = solver.solve(max_iterations=100, convergence=1e-8)
```

#### CCSD(T) Coupled Cluster

The "gold standard" of quantum chemistry for recovering electron correlation energy.

```python
from cm.qm.integrals import ccsd, CCSDResult, transform_integrals_to_mo

# First run Hartree-Fock
nuclei = [('H', (0.0, 0.0, 0.0)), ('H', (0.74, 0.0, 0.0))]
hf_result = hartree_fock(nuclei, basis="cc-pVTZ", n_electrons=2)

# CCSD(T) calculation
ccsd_result = ccsd(
    hf_result,
    max_iterations=50,
    convergence=1e-8,
    diis=True,      # Use DIIS acceleration
    verbose=True
)

print(f"HF Energy:      {ccsd_result.energy_hf:.6f} Hartree")
print(f"CCSD Energy:    {ccsd_result.energy_ccsd:.6f} Hartree")
print(f"(T) Correction: {ccsd_result.energy_triples:.6f} Hartree")
print(f"Total Energy:   {ccsd_result.energy_total:.6f} Hartree")
print(f"Correlation:    {ccsd_result.energy_total - ccsd_result.energy_hf:.6f} Hartree")

# Access amplitudes
t1 = ccsd_result.t1  # Singles amplitudes (t_i^a)
t2 = ccsd_result.t2  # Doubles amplitudes (t_ij^ab)
```

#### Integral Transformation

```python
from cm.qm.integrals import transform_integrals_to_mo

# Transform AO integrals to MO basis for post-HF methods
# (μν|λσ) → (pq|rs) where p,q,r,s are molecular orbitals
eri_mo = transform_integrals_to_mo(eri_ao, C)  # O(N⁵) transformation
```

#### Molecular Orbital Visualization

```python
from cm.qm.integrals import (
    OrbitalGrid,
    create_orbital_grid,
    evaluate_orbital_on_grid,
    extract_orbital_isosurface,
    marching_cubes,
)

# Create a 3D grid around the molecule
grid = create_orbital_grid(
    basis,
    padding=3.0,      # Angstroms beyond atoms
    resolution=0.1    # Grid spacing in Angstroms
)

# Evaluate molecular orbital on grid
orbital_values = evaluate_orbital_on_grid(grid, basis, C[:, orbital_index])

# Extract isosurface using marching cubes
vertices, faces = extract_orbital_isosurface(
    grid, orbital_values,
    isovalue=0.02,    # Typical orbital isosurface value
    color='blue'
)

# Or use marching cubes directly
vertices, faces = marching_cubes(orbital_values, level=0.02)
```

#### Boys Function

The Boys function F_n(x) is used for nuclear attraction and electron repulsion integrals.

```python
from cm.qm.integrals import boys_function

# F_n(x) = ∫₀¹ t^(2n) exp(-x t²) dt
F0 = boys_function(0, 1.0)   # F₀(1.0)
F1 = boys_function(1, 2.5)   # F₁(2.5)

# Vectorized evaluation
import numpy as np
x_values = np.linspace(0, 10, 100)
F0_values = boys_function(0, x_values)
```

#### Kohn-Sham DFT

Density Functional Theory with various exchange-correlation functionals.

```python
from cm.qm.integrals import kohn_sham, DFTResult, get_functional, list_functionals

# List available functionals
print(list_functionals())
# ['SVWN5', 'BLYP', 'PBE', 'B3LYP', 'PBE0', 'M06', 'M06-2X', 'CAM-B3LYP', 'wB97X-D', ...]

# Run DFT calculation
atoms = [('O', (0, 0, 0)), ('H', (0.96, 0, 0)), ('H', (-0.24, 0.93, 0))]
result = kohn_sham(
    atoms,
    functional='B3LYP',   # Hybrid functional
    basis='6-31G*',
    verbose=True
)

print(f"DFT Energy: {result.energy:.6f} Hartree")
print(f"XC Energy: {result.E_xc:.6f} Hartree")
print(f"Exact exchange fraction: {result.exact_exchange_fraction}")

# Access orbitals and density
C = result.mo_coefficients
eps = result.orbital_energies
P = result.density
```

**Available Functionals:**

| Category | Functionals |
|----------|-------------|
| LDA | SVWN5 |
| GGA | BLYP, PBE |
| Hybrid | B3LYP, PBE0, M06, M06-2X |
| Range-separated | CAM-B3LYP, ωB97X-D, ωB97M-V |

#### Unrestricted Hartree-Fock (UHF)

For open-shell systems with unpaired electrons.

```python
from cm.qm.integrals import uhf, UHFResult

# Oxygen atom (triplet ground state)
atoms = [('O', (0, 0, 0))]
result = uhf(
    atoms,
    basis='6-31G*',
    n_alpha=5,    # 5 alpha electrons
    n_beta=3,     # 3 beta electrons
    verbose=True
)

print(f"UHF Energy: {result.energy:.6f} Hartree")
print(f"<S²>: {result.S2:.4f}")  # Spin contamination
print(f"Multiplicity: {result.multiplicity}")
```

#### MP2 Perturbation Theory

Second-order Møller-Plesset perturbation theory for electron correlation.

```python
from cm.qm.integrals import hartree_fock, mp2, MP2Result

# First run HF
hf_result = hartree_fock(atoms, basis='cc-pVTZ', n_electrons=10)

# MP2 correlation energy
mp2_result = mp2(hf_result)
print(f"HF Energy: {mp2_result.energy_hf:.6f} Hartree")
print(f"MP2 Correlation: {mp2_result.energy_mp2:.6f} Hartree")
print(f"Total Energy: {mp2_result.energy_total:.6f} Hartree")

# SCS-MP2 (spin-component scaled)
from cm.qm.integrals import compute_scs_mp2_energy
E_scs = compute_scs_mp2_energy(mp2_result)
```

#### Analytic Gradients

Compute energy gradients for geometry optimization.

```python
from cm.qm.integrals import hartree_fock, hf_gradient, dft_gradient, kohn_sham

# HF gradient
hf = hartree_fock(atoms, basis='6-31G*', n_electrons=10)
grad = hf_gradient(hf)
print(f"Gradient shape: {grad.gradient.shape}")  # (n_atoms, 3)
print(f"Max gradient: {grad.max_gradient:.6f} Hartree/Bohr")

# DFT gradient
dft = kohn_sham(atoms, functional='B3LYP', basis='6-31G*')
grad = dft_gradient(dft)
```

#### Geometry Optimization

Optimize molecular geometries using quasi-Newton methods.

```python
from cm.qm.integrals import optimize_geometry, GeometryOptimizer, OptimizationResult

# Simple interface
atoms = [('O', (0, 0, 0)), ('H', (1.0, 0, 0)), ('H', (-0.3, 0.9, 0))]
result = optimize_geometry(
    atoms,
    method='B3LYP',
    basis='6-31G*',
    algorithm='BFGS',
    convergence={'gradient': 1e-4, 'energy': 1e-6},
    verbose=True
)

print(f"Optimized energy: {result.energy:.6f} Hartree")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.n_iterations}")

# Access optimized geometry
for elem, pos in result.atoms:
    print(f"{elem}: {pos}")

# Full optimizer class
optimizer = GeometryOptimizer(
    method='B3LYP',
    basis='6-31G*',
    algorithm='BFGS',
    max_iterations=100
)
result = optimizer.optimize(atoms)
```

#### Transition State Search

Find first-order saddle points using eigenvector-following.

```python
from cm.qm.integrals import find_transition_state, TransitionStateOptimizer, TSResult

# H2 + H -> H + H2 transition state guess
atoms = [
    ('H', (-1.0, 0, 0)),
    ('H', (0.0, 0, 0)),
    ('H', (1.2, 0, 0))
]

result = find_transition_state(
    atoms,
    method='HF',
    basis='6-31G*',
    mode_follow=0,  # Follow lowest eigenmode
    verbose=True
)

print(f"TS Energy: {result.energy:.6f} Hartree")
print(f"Imaginary frequency: {result.imaginary_freq:.1f}i cm⁻¹")
print(f"Converged: {result.converged}")
```

#### Internal Coordinates

Work with internal coordinates (bonds, angles, dihedrals).

```python
from cm.qm.integrals import InternalCoordinates

# Define internal coordinates for water
atoms = [('O', (0, 0, 0)), ('H', (0.96, 0, 0)), ('H', (-0.24, 0.93, 0))]
ic = InternalCoordinates(atoms)

# Get bond lengths and angles
print(f"O-H bond 1: {ic.bond_length(0, 1):.4f} Å")
print(f"O-H bond 2: {ic.bond_length(0, 2):.4f} Å")
print(f"H-O-H angle: {ic.bond_angle(1, 0, 2):.2f}°")

# Wilson B-matrix (dq/dx)
B = ic.wilson_b_matrix()
```

#### Frequency Analysis

Compute harmonic vibrational frequencies from Hessian.

```python
from cm.qm.integrals import (
    kohn_sham, compute_hessian, harmonic_frequencies,
    FrequencyResult, HessianResult
)

# First optimize geometry, then compute Hessian
dft = kohn_sham(optimized_atoms, functional='B3LYP', basis='6-31G*')
hessian = compute_hessian(dft, optimized_atoms)

# Compute frequencies
freq = harmonic_frequencies(hessian, optimized_atoms)

print(f"Frequencies (cm⁻¹):")
for i, f in enumerate(freq.frequencies):
    if f > 0:
        print(f"  Mode {i+1}: {f:.1f} cm⁻¹")
    else:
        print(f"  Mode {i+1}: {abs(f):.1f}i cm⁻¹")  # Imaginary

print(f"Zero-point energy: {freq.zpe:.6f} Hartree")
print(f"                   {freq.zpe * 627.5:.2f} kcal/mol")

# Access normal modes and IR intensities
modes = freq.normal_modes      # (3N, 3N-6) array
ir = freq.ir_intensities       # km/mol
```

#### Thermochemistry

Calculate thermodynamic properties from frequencies.

```python
from cm.qm.integrals import thermochemistry, ThermochemistryResult

# Compute thermochemistry at 298.15 K, 1 atm
thermo = thermochemistry(
    freq_result,
    temperature=298.15,  # K
    pressure=1.0         # atm
)

thermo.print_summary()

# Access individual contributions
print(f"Electronic energy: {thermo.E_elec:.6f} Hartree")
print(f"Zero-point energy: {thermo.ZPE:.6f} Hartree")
print(f"Thermal correction to enthalpy: {thermo.H_corr:.6f} Hartree")
print(f"Thermal correction to Gibbs energy: {thermo.G_corr:.6f} Hartree")
print(f"")
print(f"Total enthalpy (H): {thermo.H_total:.6f} Hartree")
print(f"Total Gibbs energy (G): {thermo.G_total:.6f} Hartree")
print(f"Entropy (S): {thermo.S:.6f} cal/(mol·K)")
```

#### TDDFT Excited States

Compute excited states using linear-response TDDFT.

```python
from cm.qm.integrals import kohn_sham, tddft, tda, TDDFTResult

# Ground state DFT
dft = kohn_sham(atoms, functional='B3LYP', basis='6-31G*')

# Full TDDFT (includes B matrix)
excited = tddft(dft, n_states=10)
excited.print_summary()

# Or use TDA (Tamm-Dancoff Approximation, faster)
excited_tda = tda(dft, n_states=10)

# Access results
for i in range(excited.n_states):
    E_eV = excited.excitation_energies[i]
    f = excited.oscillator_strengths[i]
    wavelength = 1240.0 / E_eV  # nm

    print(f"S{i+1}: {E_eV:.2f} eV ({wavelength:.0f} nm), f = {f:.4f}")

# Access transition dipoles and amplitudes
mu = excited.transition_dipoles  # (n_states, 3)
X = excited.X_amplitudes         # (n_states, n_occ, n_virt)
Y = excited.Y_amplitudes         # (n_states, n_occ, n_virt)
```

#### PCM Solvation

Compute solvation energies using the Polarizable Continuum Model.

```python
from cm.qm.integrals import (
    kohn_sham, PCMSolver, compute_solvation_energy, build_cavity, PCMResult
)

# Ground state DFT calculation
dft = kohn_sham(atoms, functional='B3LYP', basis='6-31G*')

# Simple interface
solv = compute_solvation_energy(dft, solvent='water')
solv.print_summary()

print(f"Solvation energy: {solv.solvation_energy:.6f} Hartree")
print(f"                  {solv.solvation_energy_kcal:.2f} kcal/mol")

# Full solver with custom settings
pcm = PCMSolver(
    solvent='acetonitrile',  # or epsilon=35.94
    scaling_factor=1.2,       # Cavity radius scaling
    n_points_per_atom=60      # Surface discretization
)
result = pcm.compute(dft)

# Available solvents
solvents = ['water', 'methanol', 'ethanol', 'acetonitrile', 'dmso',
            'chloroform', 'dichloromethane', 'thf', 'toluene', 'hexane',
            'benzene', 'acetone', 'dmf', 'diethylether']
```

#### Dipole Moments

Calculate molecular dipole moments.

```python
from cm.qm.integrals import kohn_sham, dipole_moment, DipoleResult

dft = kohn_sham(atoms, functional='B3LYP', basis='6-31G*')
dipole = dipole_moment(dft, verbose=True)

print(f"Dipole moment: {dipole.magnitude:.4f} Debye")
print(f"Components (Debye): X={dipole.dipole[0]:.4f}, "
      f"Y={dipole.dipole[1]:.4f}, Z={dipole.dipole[2]:.4f}")

# Access electronic and nuclear contributions
print(f"Electronic: {dipole.dipole_electronic}")
print(f"Nuclear: {dipole.dipole_nuclear}")
```

#### Polarizability

Calculate static polarizability tensor.

```python
from cm.qm.integrals import (
    kohn_sham, static_polarizability, PolarizabilityResult, PolarizabilityCalculator
)

# From converged result (sum-over-states)
dft = kohn_sham(atoms, functional='B3LYP', basis='6-31G*')
pol = static_polarizability(dft)
pol.print_summary()

print(f"Isotropic polarizability: {pol.isotropic:.4f} a.u.")
print(f"                          {pol.isotropic_angstrom:.4f} ų")
print(f"Anisotropy: {pol.anisotropy:.4f} a.u.")

# Finite field method (more accurate)
calc = PolarizabilityCalculator(
    method='B3LYP',
    basis='6-31G*',
    field_strength=0.001
)
pol = calc.compute(atoms)
```

#### Effective Core Potentials (ECPs)

Use ECPs for transition metals.

```python
from cm.qm.integrals.basis.ecp import get_ecp, ECPPotential, LANL2DZ_ECP

# Get LANL2DZ ECP for iron
fe_ecp = get_ecp('Fe', 'LANL2DZ')
print(f"Element: {fe_ecp.element}")
print(f"Core electrons replaced: {fe_ecp.n_core}")
print(f"Max angular momentum: {fe_ecp.lmax}")

# Available ECPs: Fe, Cu, Zn, Ni, Co, Mn
available_metals = list(LANL2DZ_ECP.keys())
```

#### Complete Example: H₂ Energy Curve

```python
from cm.qm.integrals import hartree_fock, ccsd
import numpy as np

distances = np.linspace(0.5, 3.0, 20)  # Bond lengths in Angstroms
hf_energies = []
ccsd_energies = []

for r in distances:
    nuclei = [('H', (0, 0, 0)), ('H', (r, 0, 0))]

    # Hartree-Fock
    hf = hartree_fock(nuclei, basis="cc-pVTZ", n_electrons=2)
    hf_energies.append(hf.energy)

    # CCSD(T)
    cc = ccsd(hf, max_iterations=30, convergence=1e-7)
    ccsd_energies.append(cc.energy_total)

# Plot results
import matplotlib.pyplot as plt
plt.plot(distances, hf_energies, label='HF/cc-pVTZ')
plt.plot(distances, ccsd_energies, label='CCSD(T)/cc-pVTZ')
plt.xlabel('R (Å)')
plt.ylabel('Energy (Hartree)')
plt.legend()
```

---

## cm.data Module

Access benchmark molecular databases (NIST CCCBDB, PubChem, QM9) to download empirical/reference data and compare it against your ab initio calculations.

```python
from cm.data import search, get, compare, sync, stats, status
```

**Supported databases:**

| Source | Description | Size |
|--------|-------------|------|
| `pubchem` | Molecular properties, 3D structures, computed properties | On-demand |
| `nist` | Experimental geometries, energies, vibrational frequencies | On-demand |
| `qm9` | Pre-computed DFT results (B3LYP/6-31G(2df,p)) | ~134k molecules |

### Searching for Molecules

#### `search(query, *, name=None, formula=None, cas=None, smiles=None, sources=None, limit=20)`

Search benchmark databases for molecules by name, formula, CAS number, or SMILES.

- `query`: General search query (matches name, formula, CAS, or SMILES)
- `name`: Search by molecule name
- `formula`: Search by molecular formula
- `cas`: Search by CAS number
- `smiles`: Search by SMILES string
- `sources`: Filter to specific sources: `['pubchem', 'nist', 'qm9']`
- `limit`: Maximum results (default: 20)

Returns a list of `BenchmarkMolecule` objects.

```python
from cm.data import search

# Search by name
results = search("water")

# Search by formula
results = search(formula="H2O")

# Search by CAS number
results = search(cas="7732-18-5")

# Filter to specific sources
results = search("ethanol", sources=["nist", "pubchem"])

# Access result data
for mol in results:
    print(f"{mol.name} ({mol.formula}) - CAS: {mol.cas}")
```

### Getting Molecule Data

#### `get(identifier, sources=None, workspace_id='1', fetch_if_missing=True)`

Get detailed benchmark data for a specific molecule, including geometry and properties.

- `identifier`: CAS number, PubChem CID, SMILES, or InChIKey
- `sources`: Specific sources to query
- `workspace_id`: Workspace for storing downloaded files
- `fetch_if_missing`: If `True`, fetch from external sources when not cached

Returns a `BenchmarkMolecule` with full property data and geometry.

```python
from cm.data import get

# Get by CAS number
mol = get("7732-18-5")

# Get by PubChem CID
mol = get("962")

# Get by name
mol = get("water")

# Access molecule data
print(mol.name)              # "Water"
print(mol.formula)           # "H2O"
print(mol.molecular_weight)  # 18.015
print(mol.smiles)            # "O"
print(mol.sources)           # ["pubchem", "nist"]

# Access properties (energies, dipole moments, etc.)
for prop in mol.properties:
    print(f"{prop.name}: {prop.value} {prop.unit} ({prop.source})")

# Get a specific property
dipole = mol.get_property("dipole_moment")
homo = mol.get_property("homo_energy", source="qm9")

# Access 3D geometry as XYZ string
xyz_string = mol.get_xyz()

# Display molecule data as an HTML table in notebook
mol.render()
```

#### `BenchmarkMolecule` Fields

| Field | Type | Description |
|-------|------|-------------|
| `identifier` | `str` | Unique molecule identifier |
| `name` | `str` | Molecule name |
| `formula` | `str` | Molecular formula |
| `cas` | `str` | CAS number |
| `smiles` | `str` | SMILES representation |
| `inchi` | `str` | InChI format |
| `inchi_key` | `str` | InChI key |
| `cid` | `int` | PubChem Compound ID |
| `molecular_weight` | `float` | Molecular mass |
| `charge` | `int` | Formal charge |
| `multiplicity` | `int` | Spin multiplicity |
| `sources` | `list[str]` | Data sources |
| `properties` | `list[BenchmarkProperty]` | Associated properties |
| `geometry` | `dict` | Molecular coordinates |

### Comparing Computed vs. Benchmark Data

#### `compare(computed, identifier)`

Compare your computed molecular properties against benchmark/experimental data. Displays a color-coded comparison table.

- `computed`: Dictionary of property name-value pairs, or an `HFResult`/`DFTResult` object from `cm.qm.integrals`
- `identifier`: Benchmark molecule identifier (CAS, CID, SMILES, InChIKey)

Returns a `ComparisonResult` with property-by-property comparison.

```python
from cm.qm.integrals import hartree_fock, dipole_moment
from cm.data import compare

# Run a calculation
atoms = [('O', (0, 0, 0)), ('H', (0.96, 0, 0)), ('H', (-0.24, 0.93, 0))]
hf = hartree_fock(atoms, basis='cc-pVTZ')
dipole = dipole_moment(hf)

# Compare a dictionary of computed values
comparison = compare(
    {"total_energy": hf.energy, "dipole_moment": dipole.magnitude},
    "7732-18-5"  # Water
)

# Display color-coded comparison table
# Green (<1%), light green (<5%), orange (<10%), red (>10%)
comparison.render()

# Access comparison details
print(f"Properties compared: {comparison.properties_compared}")
print(f"Average % difference: {comparison.avg_percent_diff:.2f}%")

for comp in comparison.comparisons:
    print(f"{comp.property}: computed={comp.computed:.4f}, "
          f"benchmark={comp.benchmark:.4f}, diff={comp.percent_difference:.2f}%")
```

```python
# Or pass an HFResult/DFTResult directly (energy and dipole extracted automatically)
from cm.qm.integrals import hartree_fock
from cm.data import compare

atoms = [('O', (0, 0, 0)), ('H', (0.96, 0, 0)), ('H', (-0.24, 0.93, 0))]
hf = hartree_fock(atoms, basis='STO-3G')
comparison = compare(hf, "7732-18-5")
comparison.render()
```

### Syncing Benchmark Databases

#### `sync(sources=None, workspace_id='1')`

Download and index benchmark databases. Required before searching the QM9 dataset.

```python
from cm.data import sync, wait_for_sync

# Download and index QM9 (~134k molecules)
sync(['qm9'])

# Wait for completion with progress updates
wait_for_sync(show_progress=True)
# Output:
#   Syncing qm9: 45% (phase: indexing, indexed: 60000)
#   Syncing qm9: 78% (phase: indexing, indexed: 105000)
#   Syncing qm9: 100% (phase: complete, indexed: 133885)
#   Sync complete!
```

#### `stats()`

Get statistics about indexed benchmark data.

```python
from cm.data import stats

result = stats()
print(result)
# {'sources': {'pubchem': 150, 'nist': 42, 'qm9': 133885}, 'total_molecules': 134077}
```

#### `wait_for_sync(poll_interval=2.0, timeout=3600, show_progress=True)`

Block until a running sync completes.

- `poll_interval`: Seconds between status checks
- `timeout`: Maximum seconds to wait (default: 1 hour)
- `show_progress`: Print progress updates

### Checking Status

#### `status(identifier)`

Check whether a molecule is available in the benchmark database.

- `identifier`: CAS number, PubChem CID, SMILES, or InChIKey

Returns a `MoleculeStatus` object.

```python
from cm.data import status, get

st = status("7732-18-5")

if st.status == 'indexed':
    mol = get("7732-18-5")
elif st.status == 'indexing':
    print(f"Database indexing in progress: {st.indexing_progress:.0f}%")
elif st.status == 'not_found':
    # Fetch from external sources
    mol = get("7732-18-5", fetch_if_missing=True)
elif st.status == 'no_index':
    print("Run sync() to initialize the database")
```

| Status | Meaning |
|--------|---------|
| `'indexed'` | Molecule data is available |
| `'indexing'` | Database is currently being indexed |
| `'not_found'` | Not in database (can fetch from external sources) |
| `'no_index'` | Index doesn't exist yet (run `sync()` first) |

#### `sync_status()`

Get detailed sync progress information.

```python
from cm.data import sync_status

ss = sync_status()
print(ss['is_syncing'])       # True/False
print(ss['total_molecules'])  # Number of indexed molecules
for job in ss['jobs']:
    print(f"{job['source']}: {job['progress']}%")
```

### Complete Workflow Example

```python
# %% Cell 1 - Sync benchmark data (only needed once)
from cm.data import sync, wait_for_sync
sync(['qm9'])
wait_for_sync()

# %% Cell 2 - Search and retrieve benchmark data
from cm.data import search, get

results = search(formula="H2O")
mol = get("7732-18-5")  # Water by CAS number
mol.render()

# Get experimental geometry
xyz = mol.get_xyz()
print(xyz)

# %% Cell 3 - Run computation and compare
from cm.qm.integrals import hartree_fock, dipole_moment
from cm.data import compare

# Build atoms list for hartree_fock: (element, (x, y, z))
water = [
    ('O', (0, 0, 0)),
    ('H', (0.96, 0, 0)),
    ('H', (-0.24, 0.93, 0)),
]
hf = hartree_fock(water, basis='cc-pVTZ')
dipole = dipole_moment(hf)

# Compare HF results with experimental data
comparison = compare(
    {"total_energy": hf.energy, "dipole_moment": dipole.magnitude},
    "7732-18-5"
)
comparison.render()
```

### Error Handling

The module raises specific exceptions with helpful recovery messages:

| Exception | When | Recovery |
|-----------|------|----------|
| `NoIndexError` | Benchmark index not created | Run `sync(['qm9'])` |
| `IndexingInProgressError` | Database is being indexed | Wait with `wait_for_sync()` |
| `MoleculeNotFoundError` | Molecule not in database | Use `get(id, fetch_if_missing=True)` |
| `ServiceUnavailableError` | Services not running | Run `docker compose up -d` |

```python
from cm.data import get, NoIndexError, MoleculeNotFoundError

try:
    mol = get("7732-18-5")
except NoIndexError:
    print("Run sync() first to initialize the database")
except MoleculeNotFoundError as e:
    print(f"Molecule not found: {e}")
```

---

## C++ Support

The C++ API mirrors the Python API. Include `<cm/views.hpp>` and `<cm/symbols.hpp>`.

### Basic Output

```cpp
// %% Cell 1
#include <cm/views.hpp>

int main() {
    cm::views::html("<h2>C++ Output Demo</h2>");
    cm::views::html("<p>Rich HTML from C++</p>");

    cm::views::log("Info message");
    cm::views::log_success("Success!");
    cm::views::log_warning("Warning!");
    cm::views::log_error("Error!");

    // Multiple arguments
    int value = 42;
    cm::views::log("The answer is:", value);

    return 0;
}
```

### Tables

```cpp
#include <cm/views.hpp>
#include <vector>
#include <string>

int main() {
    std::vector<std::vector<std::string>> data = {
        {"Hydrogen", "H", "1.008"},
        {"Helium", "He", "4.003"},
        {"Lithium", "Li", "6.941"},
    };

    std::vector<std::string> headers = {"Element", "Symbol", "Mass"};

    cm::views::table(data, headers);

    return 0;
}
```

### LaTeX Math

```cpp
#include <cm/symbols.hpp>

int main() {
    cm::symbols::latex("E = mc^2");
    cm::symbols::equation("F = ma", "1");

    cm::symbols::set_line_height("1.8");
    cm::symbols::latex("\\int_0^\\infty e^{-x} dx = 1");

    return 0;
}
```

### 3D Visualization

```cpp
#include <cm/views.hpp>
#include <vector>
#include <array>
#include <cmath>

int main() {
    // Scatter plot
    std::vector<std::array<double, 3>> points;
    for (int i = 0; i < 100; ++i) {
        double t = i * 0.1;
        points.push_back({std::cos(t), std::sin(t), t * 0.1});
    }

    cm::views::scatter_3d(points, 0.1, true);

    // Line plot
    cm::views::line_3d(points, "#00d4ff", true);

    // Molecule
    std::vector<cm::views::Atom> atoms = {
        {"O", 0.0, 0.0, 0.117},
        {"H", 0.756, 0.0, -0.469},
        {"H", -0.756, 0.0, -0.469}
    };
    std::vector<cm::views::Bond> bonds = {{0, 1}, {0, 2}};

    cm::views::molecule(atoms, bonds, 0.4, 0.1, true);

    return 0;
}
```

---

## Colormaps Reference

Available colormaps for scientific visualizations:

| Colormap | Description | Best For |
|----------|-------------|----------|
| `viridis` | Perceptually uniform blue-green-yellow | General purpose (default) |
| `plasma` | Purple-red-yellow | Highlighting gradients |
| `magma` | Black-red-yellow-white | Dark-to-light data |
| `inferno` | Black-red-orange-yellow | High contrast |
| `coolwarm` | Blue-white-red (diverging) | Data with positive/negative values |
| `cividis` | Blue-yellow (colorblind friendly) | Accessibility |
| `rainbow` | Full spectrum | Discrete categories |

```python
from cm.views import scatter_3d, surface

# Use with scatter plots
scatter_3d(points, colors=values, colormap='plasma')

# Use with surfaces
surface(f=my_func, colormap='coolwarm')
```

---

## Element Data Reference

The molecular visualization functions use CPK coloring with covalent radii for all elements up to Americium (Am). Common elements:

| Element | Color | Radius (Å) |
|---------|-------|------------|
| H | White (#FFFFFF) | 0.31 |
| C | Gray (#909090) | 0.77 |
| N | Blue (#3050F8) | 0.71 |
| O | Red (#FF0D0D) | 0.66 |
| S | Yellow (#FFFF30) | 1.05 |
| Fe | Orange (#E06633) | 1.32 |
| Au | Gold (#FFD123) | 1.36 |

---

## Navigation Controls

All 3D visualizations include navigation controls in the bottom-right corner:

- **Top/Front/Side/Iso buttons** - Preset camera views (Z-up orientation)
- **Rotate Left/Right** - Rotate the view around the Z axis
- **Reset** - Return to initial camera position
- **Mouse drag** - Orbit camera
- **Scroll** - Zoom in/out
- **Right-click drag** - Pan

---

## Tips

1. **Cell files** (`.cell.py`, `.cell.cpp`) support multiple cells separated by `# %%` or `// %%`
2. **WebGL output** appears in the collapsible panel at the top of the workspace
3. **Regular output** (HTML, tables, logs) appears in the output panel
4. **Auto-rotate** can be enabled on most 3D visualizations with `auto_rotate=True`
5. **Unit boxes** are cubic by default and centered around your data with Z-axis pointing up
6. **Colormaps** automatically normalize scalar values to [0, 1] range
7. **Bond inference** uses covalent radii with configurable tolerance
