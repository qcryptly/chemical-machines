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
  - [Math Builder Class](#math-builder-class)
  - [Notation Styles](#notation-styles)
  - [Chemistry Helpers](#chemistry-helpers)
- [cm.qm Module](#cmqm-module)
  - [Spin-Orbitals](#spin-orbitals)
  - [Slater Determinants](#slater-determinants)
  - [Overlaps with @ Operator](#overlaps-with--operator)
  - [Hamiltonian Matrix Elements](#hamiltonian-matrix-elements)
  - [Slater-Condon Rules](#slater-condon-rules)
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

### Math Builder Class

The `Math` class provides a builder pattern for constructing LaTeX expressions programmatically.

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

#### Slater Determinants

For quantum chemistry, render Slater determinants in standard notation:

```python
from cm.symbols import Math

# Standard Slater determinant with orbital wavefunctions
m = Math()
m.slater_determinant(['\\phi_1', '\\phi_2', '\\phi_3'])
m.render()
# Renders: (1/√3!) |φ₁(r₁) φ₂(r₁) φ₃(r₁)|
#                  |φ₁(r₂) φ₂(r₂) φ₃(r₂)|
#                  |φ₁(r₃) φ₂(r₃) φ₃(r₃)|

# Without normalization factor
m = Math()
m.slater_determinant(['1s', '2s'], normalize=False)
m.render()

# Occupation number (ket) notation
m = Math()
m.slater_ket(['1s↑', '1s↓', '2s↑'])
m.render()
# Renders: (1/√3!) |1s↑, 1s↓, 2s↑⟩
```

#### Determinant Inner Products

Compute inner products of symbolic determinants with support for orthogonality conditions. When orthogonality is specified, non-matching state overlaps evaluate to zero (Kronecker delta: ⟨φᵢ|φⱼ⟩ = δᵢⱼ).

```python
from cm.symbols import Math
import numpy as np

# Two 2x2 determinants with same elements
bra = np.array([['a', 'b'], ['c', 'd']])
ket = np.array([['a', 'b'], ['c', 'd']])

# Inner product without orthogonality (shows all terms)
m = Math()
m.determinant_inner_product(bra, ket, orthogonal=False)
m.render()
# Shows: (⟨a|a⟩ ⟨d|d⟩) - (⟨a|b⟩ ⟨d|c⟩) - (⟨b|a⟩ ⟨c|d⟩) + (⟨b|b⟩ ⟨c|c⟩)

# Inner product WITH orthogonality (zeros out non-matching terms)
m = Math()
m.determinant_inner_product(bra, ket, orthogonal=True)
m.render()
# Only terms where all bra elements match ket elements survive

# Simplified inner product (shows final result after orthogonality)
m = Math()
m.determinant_inner_product_simplified(bra, ket, orthogonal=True)
m.render()
# Shows: 2 (the count of surviving terms)

# Different determinants with orthogonality - evaluates to 0
bra2 = np.array([['a', 'b'], ['c', 'd']])
ket2 = np.array([['e', 'f'], ['g', 'h']])
m = Math()
m.determinant_inner_product(bra2, ket2, orthogonal=True)
m.render()
# Shows: 0 (no matching terms when states are orthogonal)

# Partial orthogonality - specify which states are orthogonal
m = Math()
m.determinant_inner_product(bra, ket, orthogonal=True,
                            orthogonal_states=['a', 'b', 'c', 'd'])
m.render()
```

**Slater Determinant Inner Products:**

For Slater determinants with orthonormal orbitals:

```python
from cm.symbols import Math

# Same orbitals - inner product is 1
m = Math()
m.slater_inner_product(['a', 'b', 'c'], ['a', 'b', 'c'], orthogonal=True)
m.render()
# Shows: (1/n!) 1

# Different orbitals - inner product is 0
m = Math()
m.slater_inner_product(['a', 'b', 'c'], ['a', 'b', 'd'], orthogonal=True)
m.render()
# Shows: (1/n!) 0

# Show full expansion without orthogonality
m = Math()
m.slater_inner_product(['a', 'b'], ['c', 'd'], orthogonal=False)
m.render()
# Shows: (1/n!) Σ_P (-1)^P ⟨a|c⟩ ⟨b|d⟩

# Show full overlap expansion of two determinants
m = Math()
m.determinant_overlap_expansion(bra, ket)
m.render()
# Shows: (⟨a,d| - ⟨b,c|)(|a,d⟩ - |b,c⟩)
```

**Parameters for inner product methods:**
- `bra_matrix`, `ket_matrix`: 2D array-like matrices for bra and ket determinants
- `orthogonal`: If True, apply orthonormality (⟨i|j⟩ = δᵢⱼ)
- `orthogonal_states`: Optional list of specific states that are mutually orthogonal
- `normalize`: (Slater only) Include normalization factor 1/n!

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

## cm.qm Module

The quantum mechanics module provides tools for working with Slater determinants, spin-orbitals, and matrix elements using spherical harmonic basis functions. It automatically applies Slater-Condon rules to simplify Hamiltonian matrix elements.

```python
from cm import qm
```

### Spin-Orbitals

Create spin-orbital basis elements using spherical harmonic quantum numbers.

#### `qm.basis_sh_element(spin, L, m, n=None)`

Create a single spin-orbital.

**Parameters:**
- `spin`: +1 for spin-up (α), -1 for spin-down (β)
- `L`: Angular momentum quantum number (0=s, 1=p, 2=d, 3=f, ...)
- `m`: Magnetic quantum number (-L ≤ m ≤ L)
- `n`: Optional principal quantum number

```python
from cm import qm

# 1s spin-up orbital
orbital_1s_up = qm.basis_sh_element(spin=1, L=0, m=0, n=1)

# 2p spin-down with m=-1
orbital_2p_down = qm.basis_sh_element(spin=-1, L=1, m=-1, n=2)
```

#### `qm.basis_sh(quantum_numbers)`

Create multiple spin-orbitals from a list of tuples.

**Parameters:**
- `quantum_numbers`: List of tuples `(spin, L, m)` or `(spin, L, m, n)`

```python
from cm import qm

# Helium ground state: 1s↑ 1s↓
orbitals = qm.basis_sh([(1, 0, 0, 1), (-1, 0, 0, 1)])

# Lithium: 1s↑ 1s↓ 2s↑
orbitals = qm.basis_sh([
    (1, 0, 0, 1),   # 1s↑
    (-1, 0, 0, 1),  # 1s↓
    (1, 0, 0, 2)    # 2s↑
])
```

### Slater Determinants

#### `qm.slater(orbitals)`

Create a Slater determinant from a list of spin-orbitals.

```python
from cm import qm

# Create orbitals
orbital_1sA_up = qm.basis_sh_element(spin=1, L=0, m=0, n=1)
orbital_1sA_down = qm.basis_sh_element(spin=-1, L=0, m=0, n=1)

# Create Slater determinant
psi = qm.slater([orbital_1sA_up, orbital_1sA_down])
psi.render()  # Renders: |1, 0, 0, ↑, 1, 0, 0, ↓⟩
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
psi = qm.slater(qm.basis_sh([(1, 0, 0), (-1, 0, 0)]))
phi = qm.slater(qm.basis_sh([(1, 0, 0), (-1, 0, 0)]))
overlap = psi @ phi
print(overlap.value)  # 1

# Different orbitals - overlap is 0 (orthogonal)
chi = qm.slater(qm.basis_sh([(1, 0, 0), (1, 1, 0)]))
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

# Create determinants
orbitals_A = qm.basis_sh([(1, 0, 0, 1), (-1, 0, 0, 1)])
orbitals_B = qm.basis_sh([(1, 0, 0, 2), (-1, 0, 0, 2)])
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

# Define orbitals for H₂ molecule
orbital_1sA_up = qm.basis_sh_element(spin=1, L=0, m=0, n=1)
orbital_1sA_down = qm.basis_sh_element(spin=-1, L=0, m=0, n=1)
orbital_1sB_up = qm.basis_sh_element(spin=1, L=0, m=0, n=2)
orbital_1sB_down = qm.basis_sh_element(spin=-1, L=0, m=0, n=2)

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
orbital_1sA_up = qm.basis_sh_element(spin=1, L=0, m=0, n=1)
orbital_1sA_down = qm.basis_sh_element(spin=-1, L=0, m=0, n=1)
orbital_1sB_up = qm.basis_sh_element(spin=1, L=0, m=0, n=2)
orbital_1sB_down = qm.basis_sh_element(spin=-1, L=0, m=0, n=2)

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
