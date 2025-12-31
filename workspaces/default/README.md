# Chemical Machines Library Guide

This guide covers the `cm` library for creating rich outputs in Chemical Machines workspaces.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Output (cm.views)](#basic-output-cmviews)
- [3D Scientific Visualization](#3d-scientific-visualization)
- [LaTeX Math Rendering (cm.symbols)](#latex-math-rendering-cmsymbols)
- [WebGL Custom Rendering](#webgl-custom-rendering)
- [C++ Support](#c-support)

---

## Getting Started

Chemical Machines supports both Python and C++ with cell-based execution. Files ending in `.cell.py` or `.cell.cpp` are treated as notebooks with cells separated by `# %%` or `// %%` comments.

```python
# %% Cell 1 - My First Cell
from cm.views import log

log("Hello, Chemical Machines!")
```

---

## Basic Output (cm.views)

The `cm.views` module provides functions for outputting HTML, text, tables, images, and more.

### HTML and Text

```python
from cm.views import html, text, log

# Raw HTML output
html("<h2>Welcome!</h2>")
html("<p>This is <strong>formatted</strong> HTML.</p>")

# Plain text (auto-escaped)
text("Plain text content that will be HTML-escaped")

# Log messages with different levels
log("Info message")                      # Default info style
log("Success!", level="success")         # Green success style
log("Warning message", level="warning")  # Yellow warning style
log("Error occurred", level="error")     # Red error style

# Log multiple values
log("The answer is:", 42)
log("Data:", {"key": "value"})           # Dicts/lists auto-formatted as JSON
```

### Tables

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

### Images and Matplotlib

```python
from cm.views import image, savefig
import matplotlib.pyplot as plt
import numpy as np

# Display an image from file
image("plot.png")

# Save matplotlib figure to output
x = np.linspace(0, 2 * np.pi, 100)
plt.figure(figsize=(8, 4))
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()
plt.title('Trigonometric Functions')
plt.grid(True, alpha=0.3)

savefig()  # Automatically saves current figure to output
plt.close()
```

### DataFrames (Pandas)

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

### Clearing Output

```python
from cm.views import clear, clear_all

clear()      # Clear outputs for the current cell
clear_all()  # Clear all outputs (deletes the output file)
```

---

## 3D Scientific Visualization

The `cm.views` module includes high-level 3D visualization functions powered by Three.js.

### Scatter Plots

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

# Color by distance from origin
scatter_3d(
    points,
    colors=np.linalg.norm(points, axis=1),  # Scalar values for colormap
    colormap='plasma',                       # 'viridis', 'plasma', 'coolwarm', etc.
    point_size=0.08,
    unit_box=True
)
```

### Line Plots (Parametric Curves)

```python
from cm.views import line_3d, lines_3d
import numpy as np

# Single helix
t = np.linspace(0, 6*np.pi, 200)
helix = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (2*np.pi)
])

line_3d(helix, color='#ff6b6b', unit_box=True)

# Multiple paths
paths = []
for i in range(5):
    t = np.linspace(0, 2*np.pi, 50)
    path = np.column_stack([np.cos(t) + i, np.sin(t), t])
    paths.append(path)

lines_3d(paths, colors=['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'])
```

### Surface Plots

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

### Vector Fields

```python
from cm.views import vector_field

# Rotational field around Z axis
vector_field(
    f=lambda x, y, z: (-y, x, 0.2),
    bounds=(-2, 2, -2, 2, -1, 1),  # (x_min, x_max, y_min, y_max, z_min, z_max)
    resolution=5,
    scale=0.3,
    colormap='coolwarm',
    unit_box=True
)
```

### Molecule Visualization

```python
from cm.views import molecule, molecule_xyz

# Explicit atoms and bonds
water_atoms = [
    ('O', 0.000, 0.000, 0.117),
    ('H', 0.756, 0.000, -0.469),
    ('H', -0.756, 0.000, -0.469)
]
water_bonds = [(0, 1), (0, 2)]  # O-H bonds (indices)

molecule(
    water_atoms,
    bonds=water_bonds,
    style='ball-stick',  # 'ball-stick', 'space-fill', 'stick'
    atom_scale=1.0,
    auto_rotate=True
)

# From XYZ file format
xyz_data = """3
Water molecule
O   0.000   0.000   0.117
H   0.756   0.000  -0.469
H  -0.756   0.000  -0.469
"""

molecule_xyz(xyz_data, style='ball-stick', infer_bonds=True)
```

### Crystal Structures

```python
from cm.views import crystal

# From CIF file format
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
```

### Available Colormaps

- `viridis` (default) - Perceptually uniform blue-green-yellow
- `plasma` - Purple-red-yellow
- `magma` - Black-red-yellow-white
- `inferno` - Black-red-orange-yellow
- `coolwarm` - Blue-white-red (diverging)
- `spectral` - Rainbow spectrum

---

## LaTeX Math Rendering (cm.symbols)

The `cm.symbols` module provides LaTeX math rendering with support for different notation styles.

### Basic LaTeX

```python
from cm.symbols import latex, equation, align

# Simple equations
latex(r"E = mc^2")
latex(r"\int_0^\infty e^{-x} dx = 1")

# Numbered equations
equation(r"F = ma", number=1)
equation(r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}", number=2)

# Aligned multi-line equations
align(
    r"(a + b)^2 &= (a + b)(a + b)",
    r"&= a^2 + ab + ba + b^2",
    r"&= a^2 + 2ab + b^2"
)
```

### Matrices

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

# Styles: 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix'
```

### Math Builder

```python
from cm.symbols import Math

# Build equations programmatically
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
```

### Line Height Control

```python
from cm.symbols import set_line_height

# Adjust spacing between equations
set_line_height("1.8")  # More space
set_line_height("1.2")  # Tighter
set_line_height("normal")  # Reset to default
```

### Notation Styles

```python
from cm.symbols import set_notation, Math, latex

# Quantum Mechanics (Bra-ket notation)
set_notation("braket")

m = Math()
m.braket("\\psi", "\\phi")  # <psi|phi>
m.render()

m = Math()
m.expval("\\hat{H}")  # <H>
m.render()

m = Math()
m.comm("\\hat{x}", "\\hat{p}").equals().var("i").hbar()  # [x, p] = i*hbar
m.render()

# Chemistry notation
set_notation("chemist")

from cm.symbols import chemical, reaction
chemical("2H2 + O2 -> 2H2O")
reaction("N2 + 3H2", "2NH3", reversible=True)

# Physics notation
set_notation("physicist")
latex(r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J}")

# Reset to standard
set_notation("standard")
```

---

## WebGL Custom Rendering

For full control over 3D rendering, use the low-level WebGL functions.

### Using webgl_threejs Helper

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

### Raw WebGL HTML

```python
from cm.views import webgl

# Full HTML with Three.js
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

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
''')
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

## Navigation Controls

All 3D visualizations include navigation controls in the bottom-right corner:

- **Top/Front/Side/Iso buttons** - Preset camera views
- **Rotate Left/Right** - Rotate the view
- **Reset** - Return to initial camera position
- **Mouse drag** - Orbit camera
- **Scroll** - Zoom in/out

---

## Tips

1. **Cell files** (`.cell.py`, `.cell.cpp`) support multiple cells separated by `# %%` or `// %%`
2. **WebGL output** appears in the collapsible panel at the top of the workspace
3. **Regular output** (HTML, tables, logs) appears in the output panel
4. **Auto-rotate** can be enabled on most 3D visualizations with `auto_rotate=True`
5. **Unit boxes** are cubic by default and centered around your data with Z-axis pointing up
