"""
Chemical Machines Views Visualization Module

Scientific visualization including molecular rendering, 3D plots, and colormaps.
"""

from typing import Optional, List, Union, Dict, Any, Tuple
import json
import math

from .output import html, webgl

__all__ = [
    'ELEMENT_DATA',
    'COLORMAPS',
    'scatter_3d',
    'surface',
    'molecule',
    'crystal',
    'orbital',
]


# =============================================================================
# Scientific Visualization Library
# =============================================================================

# Element data for molecular visualization (CPK coloring, covalent radii in Angstroms)
ELEMENT_DATA = {
    'H':  {'color': '#FFFFFF', 'radius': 0.31, 'name': 'Hydrogen'},
    'He': {'color': '#D9FFFF', 'radius': 0.28, 'name': 'Helium'},
    'Li': {'color': '#CC80FF', 'radius': 1.28, 'name': 'Lithium'},
    'Be': {'color': '#C2FF00', 'radius': 0.96, 'name': 'Beryllium'},
    'B':  {'color': '#FFB5B5', 'radius': 0.84, 'name': 'Boron'},
    'C':  {'color': '#909090', 'radius': 0.77, 'name': 'Carbon'},
    'N':  {'color': '#3050F8', 'radius': 0.71, 'name': 'Nitrogen'},
    'O':  {'color': '#FF0D0D', 'radius': 0.66, 'name': 'Oxygen'},
    'F':  {'color': '#90E050', 'radius': 0.57, 'name': 'Fluorine'},
    'Ne': {'color': '#B3E3F5', 'radius': 0.58, 'name': 'Neon'},
    'Na': {'color': '#AB5CF2', 'radius': 1.66, 'name': 'Sodium'},
    'Mg': {'color': '#8AFF00', 'radius': 1.41, 'name': 'Magnesium'},
    'Al': {'color': '#BFA6A6', 'radius': 1.21, 'name': 'Aluminum'},
    'Si': {'color': '#F0C8A0', 'radius': 1.11, 'name': 'Silicon'},
    'P':  {'color': '#FF8000', 'radius': 1.07, 'name': 'Phosphorus'},
    'S':  {'color': '#FFFF30', 'radius': 1.05, 'name': 'Sulfur'},
    'Cl': {'color': '#1FF01F', 'radius': 1.02, 'name': 'Chlorine'},
    'Ar': {'color': '#80D1E3', 'radius': 1.06, 'name': 'Argon'},
    'K':  {'color': '#8F40D4', 'radius': 2.03, 'name': 'Potassium'},
    'Ca': {'color': '#3DFF00', 'radius': 1.76, 'name': 'Calcium'},
    'Sc': {'color': '#E6E6E6', 'radius': 1.70, 'name': 'Scandium'},
    'Ti': {'color': '#BFC2C7', 'radius': 1.60, 'name': 'Titanium'},
    'V':  {'color': '#A6A6AB', 'radius': 1.53, 'name': 'Vanadium'},
    'Cr': {'color': '#8A99C7', 'radius': 1.39, 'name': 'Chromium'},
    'Mn': {'color': '#9C7AC7', 'radius': 1.39, 'name': 'Manganese'},
    'Fe': {'color': '#E06633', 'radius': 1.32, 'name': 'Iron'},
    'Co': {'color': '#F090A0', 'radius': 1.26, 'name': 'Cobalt'},
    'Ni': {'color': '#50D050', 'radius': 1.24, 'name': 'Nickel'},
    'Cu': {'color': '#C88033', 'radius': 1.32, 'name': 'Copper'},
    'Zn': {'color': '#7D80B0', 'radius': 1.22, 'name': 'Zinc'},
    'Ga': {'color': '#C28F8F', 'radius': 1.22, 'name': 'Gallium'},
    'Ge': {'color': '#668F8F', 'radius': 1.20, 'name': 'Germanium'},
    'As': {'color': '#BD80E3', 'radius': 1.19, 'name': 'Arsenic'},
    'Se': {'color': '#FFA100', 'radius': 1.20, 'name': 'Selenium'},
    'Br': {'color': '#A62929', 'radius': 1.20, 'name': 'Bromine'},
    'Kr': {'color': '#5CB8D1', 'radius': 1.16, 'name': 'Krypton'},
    'Rb': {'color': '#702EB0', 'radius': 2.20, 'name': 'Rubidium'},
    'Sr': {'color': '#00FF00', 'radius': 1.95, 'name': 'Strontium'},
    'Y':  {'color': '#94FFFF', 'radius': 1.90, 'name': 'Yttrium'},
    'Zr': {'color': '#94E0E0', 'radius': 1.75, 'name': 'Zirconium'},
    'Nb': {'color': '#73C2C9', 'radius': 1.64, 'name': 'Niobium'},
    'Mo': {'color': '#54B5B5', 'radius': 1.54, 'name': 'Molybdenum'},
    'Tc': {'color': '#3B9E9E', 'radius': 1.47, 'name': 'Technetium'},
    'Ru': {'color': '#248F8F', 'radius': 1.46, 'name': 'Ruthenium'},
    'Rh': {'color': '#0A7D8C', 'radius': 1.42, 'name': 'Rhodium'},
    'Pd': {'color': '#006985', 'radius': 1.39, 'name': 'Palladium'},
    'Ag': {'color': '#C0C0C0', 'radius': 1.45, 'name': 'Silver'},
    'Cd': {'color': '#FFD98F', 'radius': 1.44, 'name': 'Cadmium'},
    'In': {'color': '#A67573', 'radius': 1.42, 'name': 'Indium'},
    'Sn': {'color': '#668080', 'radius': 1.39, 'name': 'Tin'},
    'Sb': {'color': '#9E63B5', 'radius': 1.39, 'name': 'Antimony'},
    'Te': {'color': '#D47A00', 'radius': 1.38, 'name': 'Tellurium'},
    'I':  {'color': '#940094', 'radius': 1.39, 'name': 'Iodine'},
    'Xe': {'color': '#429EB0', 'radius': 1.40, 'name': 'Xenon'},
    'Cs': {'color': '#57178F', 'radius': 2.44, 'name': 'Cesium'},
    'Ba': {'color': '#00C900', 'radius': 2.15, 'name': 'Barium'},
    'La': {'color': '#70D4FF', 'radius': 2.07, 'name': 'Lanthanum'},
    'Ce': {'color': '#FFFFC7', 'radius': 2.04, 'name': 'Cerium'},
    'Pr': {'color': '#D9FFC7', 'radius': 2.03, 'name': 'Praseodymium'},
    'Nd': {'color': '#C7FFC7', 'radius': 2.01, 'name': 'Neodymium'},
    'Pm': {'color': '#A3FFC7', 'radius': 1.99, 'name': 'Promethium'},
    'Sm': {'color': '#8FFFC7', 'radius': 1.98, 'name': 'Samarium'},
    'Eu': {'color': '#61FFC7', 'radius': 1.98, 'name': 'Europium'},
    'Gd': {'color': '#45FFC7', 'radius': 1.96, 'name': 'Gadolinium'},
    'Tb': {'color': '#30FFC7', 'radius': 1.94, 'name': 'Terbium'},
    'Dy': {'color': '#1FFFC7', 'radius': 1.92, 'name': 'Dysprosium'},
    'Ho': {'color': '#00FF9C', 'radius': 1.92, 'name': 'Holmium'},
    'Er': {'color': '#00E675', 'radius': 1.89, 'name': 'Erbium'},
    'Tm': {'color': '#00D452', 'radius': 1.90, 'name': 'Thulium'},
    'Yb': {'color': '#00BF38', 'radius': 1.87, 'name': 'Ytterbium'},
    'Lu': {'color': '#00AB24', 'radius': 1.87, 'name': 'Lutetium'},
    'Hf': {'color': '#4DC2FF', 'radius': 1.75, 'name': 'Hafnium'},
    'Ta': {'color': '#4DA6FF', 'radius': 1.70, 'name': 'Tantalum'},
    'W':  {'color': '#2194D6', 'radius': 1.62, 'name': 'Tungsten'},
    'Re': {'color': '#267DAB', 'radius': 1.51, 'name': 'Rhenium'},
    'Os': {'color': '#266696', 'radius': 1.44, 'name': 'Osmium'},
    'Ir': {'color': '#175487', 'radius': 1.41, 'name': 'Iridium'},
    'Pt': {'color': '#D0D0E0', 'radius': 1.36, 'name': 'Platinum'},
    'Au': {'color': '#FFD123', 'radius': 1.36, 'name': 'Gold'},
    'Hg': {'color': '#B8B8D0', 'radius': 1.32, 'name': 'Mercury'},
    'Tl': {'color': '#A6544D', 'radius': 1.45, 'name': 'Thallium'},
    'Pb': {'color': '#575961', 'radius': 1.46, 'name': 'Lead'},
    'Bi': {'color': '#9E4FB5', 'radius': 1.48, 'name': 'Bismuth'},
    'Po': {'color': '#AB5C00', 'radius': 1.40, 'name': 'Polonium'},
    'At': {'color': '#754F45', 'radius': 1.50, 'name': 'Astatine'},
    'Rn': {'color': '#428296', 'radius': 1.50, 'name': 'Radon'},
    'Fr': {'color': '#420066', 'radius': 2.60, 'name': 'Francium'},
    'Ra': {'color': '#007D00', 'radius': 2.21, 'name': 'Radium'},
    'Ac': {'color': '#70ABFA', 'radius': 2.15, 'name': 'Actinium'},
    'Th': {'color': '#00BAFF', 'radius': 2.06, 'name': 'Thorium'},
    'Pa': {'color': '#00A1FF', 'radius': 2.00, 'name': 'Protactinium'},
    'U':  {'color': '#008FFF', 'radius': 1.96, 'name': 'Uranium'},
    'Np': {'color': '#0080FF', 'radius': 1.90, 'name': 'Neptunium'},
    'Pu': {'color': '#006BFF', 'radius': 1.87, 'name': 'Plutonium'},
    'Am': {'color': '#545CF2', 'radius': 1.80, 'name': 'Americium'},
    'Cm': {'color': '#785CE3', 'radius': 1.69, 'name': 'Curium'},
}

# Colormaps for scientific visualization
COLORMAPS = {
    'viridis': [
        '#440154', '#482878', '#3E4A89', '#31688E', '#26838E',
        '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'
    ],
    'plasma': [
        '#0D0887', '#47039F', '#7301A8', '#9C179E', '#BD3786',
        '#D8576B', '#ED7953', '#FA9E3B', '#FDC926', '#F0F921'
    ],
    'coolwarm': [
        '#3B4CC0', '#5977E3', '#7B9FF9', '#9EBEFF', '#C0D4F5',
        '#F2CAC1', '#F7A889', '#E87B64', '#CA4E4A', '#B40426'
    ],
    'rainbow': [
        '#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#00FF00',
        '#00FF80', '#00FFFF', '#0080FF', '#0000FF', '#8000FF'
    ],
    'magma': [
        '#000004', '#180F3D', '#440F76', '#721F81', '#9E2F7F',
        '#CD4071', '#F1605D', '#FD9668', '#FECA8D', '#FCFDBF'
    ],
    'inferno': [
        '#000004', '#1B0C41', '#4A0C6B', '#781C6D', '#A52C60',
        '#CF4446', '#ED6925', '#FB9A06', '#F7D13D', '#FCFFA4'
    ],
    'cividis': [
        '#00224E', '#123570', '#3B496C', '#575D6D', '#707173',
        '#8A8678', '#A59C74', '#C3B369', '#E1CC55', '#FDEA45'
    ],
}


def _interpolate_colormap(value: float, colormap: str = 'viridis') -> str:
    """Interpolate a color from a colormap for a value between 0 and 1."""
    colors = COLORMAPS.get(colormap, COLORMAPS['viridis'])
    n = len(colors)

    # Clamp value to [0, 1]
    value = max(0.0, min(1.0, value))

    # Find the two colors to interpolate between
    idx = value * (n - 1)
    idx_low = int(idx)
    idx_high = min(idx_low + 1, n - 1)
    t = idx - idx_low

    # Parse hex colors
    c1 = colors[idx_low]
    c2 = colors[idx_high]
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)

    # Linear interpolation
    r = int(r1 + t * (r2 - r1))
    g = int(g1 + t * (g2 - g1))
    b = int(b1 + t * (b2 - b1))

    return f'#{r:02x}{g:02x}{b:02x}'


def _generate_unit_box_js(size: tuple, color: str = '#444444', labels: bool = True,
                          center: tuple = (0, 0, 0)) -> str:
    """Generate Three.js code for a cubic unit box with Z axis up.

    The box is always cubic, using the maximum dimension from the input size.
    The box is centered around the given center point.
    Coordinate system: X = right, Y = forward, Z = up.

    Args:
        size: (width, height, depth) of the data extent
        color: Box edge color
        labels: Whether to show axis labels
        center: (cx, cy, cz) center point for the box
    """
    # Use maximum dimension to create a cube (with some padding)
    max_dim = max(size[0], size[1], size[2])
    s = max_dim / 2  # half-size for cube

    cx, cy, cz = center

    # Box vertices centered around (cx, cy, cz)
    x_min, x_max = cx - s, cx + s
    y_min, y_max = cy - s, cy + s
    z_min, z_max = cz - s, cz + s

    js = f'''
        // Unit box (cubic, Z-up, centered at ({cx}, {cy}, {cz}))
        const boxGeometry = new THREE.BufferGeometry();
        const boxVertices = new Float32Array([
            // Bottom face (Z = z_min)
            {x_min}, {y_min}, {z_min}, {x_max}, {y_min}, {z_min},
            {x_max}, {y_min}, {z_min}, {x_max}, {y_max}, {z_min},
            {x_max}, {y_max}, {z_min}, {x_min}, {y_max}, {z_min},
            {x_min}, {y_max}, {z_min}, {x_min}, {y_min}, {z_min},
            // Top face (Z = z_max)
            {x_min}, {y_min}, {z_max}, {x_max}, {y_min}, {z_max},
            {x_max}, {y_min}, {z_max}, {x_max}, {y_max}, {z_max},
            {x_max}, {y_max}, {z_max}, {x_min}, {y_max}, {z_max},
            {x_min}, {y_max}, {z_max}, {x_min}, {y_min}, {z_max},
            // Vertical edges (connecting bottom to top)
            {x_min}, {y_min}, {z_min}, {x_min}, {y_min}, {z_max},
            {x_max}, {y_min}, {z_min}, {x_max}, {y_min}, {z_max},
            {x_max}, {y_max}, {z_min}, {x_max}, {y_max}, {z_max},
            {x_min}, {y_max}, {z_min}, {x_min}, {y_max}, {z_max}
        ]);
        boxGeometry.setAttribute('position', new THREE.BufferAttribute(boxVertices, 3));
        const boxMaterial = new THREE.LineBasicMaterial({{ color: '{color}', transparent: true, opacity: 0.5 }});
        const unitBox = new THREE.LineSegments(boxGeometry, boxMaterial);
        scene.add(unitBox);
    '''

    if labels:
        # Labels positioned at the positive ends of each axis from the center
        js += f'''
        // Axis labels using sprites (Z-up coordinate system)
        function createTextSprite(text, position) {{
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 64;
            canvas.height = 32;
            ctx.fillStyle = '#888888';
            ctx.font = '20px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(text, 32, 24);
            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({{ map: texture }});
            const sprite = new THREE.Sprite(material);
            sprite.position.copy(position);
            sprite.scale.set(1, 0.5, 1);
            return sprite;
        }}
        scene.add(createTextSprite('X', new THREE.Vector3({x_max + 0.3}, {cy}, {cz})));
        scene.add(createTextSprite('Y', new THREE.Vector3({cx}, {y_max + 0.3}, {cz})));
        scene.add(createTextSprite('Z', new THREE.Vector3({cx}, {cy}, {z_max + 0.3})));
        '''

    return js


def _webgl_scene(
    scene_objects: str,
    background: str = '#1e1e2e',
    camera_position: tuple = (5, 5, 5),
    camera_target: tuple = (0, 0, 0),
    animate_code: str = '',
    unit_box: bool = False,
    box_size: tuple = None,
    box_center: tuple = None,
    box_color: str = '#444444',
    box_labels: bool = True,
    auto_rotate: bool = False
):
    """
    Internal helper to generate a complete WebGL scene with Z-up orientation.

    Args:
        scene_objects: JavaScript code to add objects to the scene
        background: Background color
        camera_position: Initial camera position (x, y, z) - default views Z as up
        camera_target: Point the camera looks at (x, y, z)
        animate_code: Code to run each animation frame
        unit_box: Whether to show a unit box
        box_size: Size of the unit box (w, h, d), auto-calculated if None
        box_center: Center of the unit box (x, y, z), defaults to camera_target
        box_color: Color of the unit box edges
        box_labels: Whether to show axis labels on the unit box
        auto_rotate: Whether to auto-rotate the scene
    """
    unit_box_js = ''
    if unit_box and box_size:
        # Default box center to camera target (which is usually the data center)
        center = box_center if box_center else camera_target
        unit_box_js = _generate_unit_box_js(box_size, box_color, box_labels, center)

    auto_rotate_js = 'scene.rotation.z += 0.002;' if auto_rotate else ''

    content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: {background};
        }}
        canvas {{ display: block; width: 100% !important; height: 100% !important; }}
        .nav-controls {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 4px;
            z-index: 100;
        }}
        .nav-btn {{
            width: 32px;
            height: 32px;
            background: rgba(40, 40, 55, 0.85);
            border: 1px solid #444;
            border-radius: 4px;
            color: #aaa;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            transition: all 0.15s;
        }}
        .nav-btn:hover {{
            background: rgba(60, 60, 80, 0.9);
            color: #fff;
            border-color: #666;
        }}
        .nav-btn:active {{
            background: rgba(80, 80, 100, 0.9);
        }}
        .nav-btn svg {{
            width: 16px;
            height: 16px;
            fill: currentColor;
        }}
        .nav-separator {{
            width: 1px;
            background: #444;
            margin: 4px 2px;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <!-- Navigation controls -->
    <div class="nav-controls">
        <button class="nav-btn" id="viewTop" title="Top view (Z+)">
            <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4"/></svg>
        </button>
        <button class="nav-btn" id="viewFront" title="Front view (Y-)">
            <svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"/></svg>
        </button>
        <button class="nav-btn" id="viewSide" title="Side view (X+)">
            <svg viewBox="0 0 24 24"><path d="M4 6h16v12H4z" fill="none" stroke="currentColor" stroke-width="2"/><path d="M4 6l4-4h12l-4 4" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>
        </button>
        <button class="nav-btn" id="viewIso" title="Isometric view">
            <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" fill="none" stroke="currentColor" stroke-width="2"/></svg>
        </button>
        <div class="nav-separator"></div>
        <button class="nav-btn" id="rotateLeft" title="Rotate left">
            <svg viewBox="0 0 24 24"><path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/></svg>
        </button>
        <button class="nav-btn" id="rotateRight" title="Rotate right">
            <svg viewBox="0 0 24 24"><path d="M12 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z"/></svg>
        </button>
        <div class="nav-separator"></div>
        <button class="nav-btn" id="resetView" title="Reset view">
            <svg viewBox="0 0 24 24"><path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/><circle cx="12" cy="13" r="2"/></svg>
        </button>
    </div>

    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color('{background}');

        // Camera (Z-up orientation)
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.up.set(0, 0, 1);  // Z is up
        const initialCameraPos = new THREE.Vector3({camera_position[0]}, {camera_position[1]}, {camera_position[2]});
        camera.position.copy(initialCameraPos);
        camera.lookAt({camera_target[0]}, {camera_target[1]}, {camera_target[2]});

        // Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set({camera_target[0]}, {camera_target[1]}, {camera_target[2]});

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 10);  // Light from above (Z+)
        scene.add(directionalLight);
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight2.position.set(-5, -5, -5);
        scene.add(directionalLight2);

        // Unit box
        {unit_box_js}

        // Scene objects
        {scene_objects}

        // Navigation control functions
        const cameraDistance = initialCameraPos.length();

        function setCameraView(x, y, z) {{
            const dir = new THREE.Vector3(x, y, z).normalize().multiplyScalar(cameraDistance);
            camera.position.copy(dir);
            controls.update();
        }}

        document.getElementById('viewTop').onclick = () => setCameraView(0, 0, 1);
        document.getElementById('viewFront').onclick = () => setCameraView(0, -1, 0.3);
        document.getElementById('viewSide').onclick = () => setCameraView(1, 0, 0.3);
        document.getElementById('viewIso').onclick = () => setCameraView(1, 1, 1);

        document.getElementById('rotateLeft').onclick = () => {{
            const spherical = new THREE.Spherical().setFromVector3(camera.position);
            spherical.theta += Math.PI / 8;
            camera.position.setFromSpherical(spherical);
            controls.update();
        }};

        document.getElementById('rotateRight').onclick = () => {{
            const spherical = new THREE.Spherical().setFromVector3(camera.position);
            spherical.theta -= Math.PI / 8;
            camera.position.setFromSpherical(spherical);
            controls.update();
        }};

        document.getElementById('resetView').onclick = () => {{
            camera.position.copy(initialCameraPos);
            controls.target.set({camera_target[0]}, {camera_target[1]}, {camera_target[2]});
            controls.update();
        }};

        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            {auto_rotate_js}
            {animate_code}
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>'''

    webgl(content)


def scatter_3d(
    points,
    colors=None,
    sizes=None,
    colormap: str = 'viridis',
    point_size: float = 0.1,
    opacity: float = 1.0,
    unit_box: bool = True,
    box_size: tuple = None,
    box_color: str = '#444444',
    box_labels: bool = True,
    background: str = '#1e1e2e',
    auto_rotate: bool = False
):
    """
    Render a 3D scatter plot of points.

    Args:
        points: Nx3 array-like of (x, y, z) coordinates
        colors: Optional - single color string, Nx3 RGB array (0-1), or N array for colormap
        sizes: Optional - single size or N array of per-point sizes (multiplied by point_size)
        colormap: Colormap name for scalar colors ('viridis', 'plasma', 'coolwarm', etc.)
        point_size: Base point size (default: 0.1)
        opacity: Point opacity 0-1 (default: 1.0)
        unit_box: Show bounding box (default: True)
        box_size: Box size (w, h, d), auto-calculated if None
        box_color: Color of box edges
        box_labels: Show axis labels on box
        background: Background color
        auto_rotate: Auto-rotate the scene

    Example:
        import numpy as np
        from cm.views import scatter_3d

        # Random points
        points = np.random.randn(100, 3)
        scatter_3d(points)

        # With scalar colors (mapped to colormap)
        scatter_3d(points, colors=points[:, 2])  # Color by Z

        # With RGB colors
        scatter_3d(points, colors=np.random.rand(100, 3))
    """
    import numpy as np

    # Convert to numpy array
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    n_points = len(points)

    # Calculate bounding box if not provided
    if unit_box and box_size is None:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2
        extent = maxs - mins
        # Add 10% padding
        box_size = tuple(extent * 1.1)
    else:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2

    # Process colors
    if colors is None:
        # Default: use Z coordinate for coloring
        z_vals = points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        if z_max > z_min:
            z_norm = (z_vals - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros(n_points)
        color_hexes = [_interpolate_colormap(v, colormap) for v in z_norm]
    elif isinstance(colors, str):
        # Single color
        color_hexes = [colors] * n_points
    elif hasattr(colors, '__len__'):
        colors = np.asarray(colors)
        if colors.ndim == 1:
            # Scalar values - map to colormap
            c_min, c_max = colors.min(), colors.max()
            if c_max > c_min:
                c_norm = (colors - c_min) / (c_max - c_min)
            else:
                c_norm = np.zeros(n_points)
            color_hexes = [_interpolate_colormap(v, colormap) for v in c_norm]
        else:
            # RGB values (0-1)
            color_hexes = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                          for r, g, b in colors]
    else:
        color_hexes = ['#00d4ff'] * n_points

    # Process sizes
    if sizes is None:
        size_array = [point_size] * n_points
    elif hasattr(sizes, '__len__'):
        sizes = np.asarray(sizes)
        size_array = [point_size * s for s in sizes]
    else:
        size_array = [point_size * sizes] * n_points

    # For small point clouds, use individual meshes (cleaner approach)
    scene_js = '''
        const pointsGroup = new THREE.Group();
    '''
    for pt, color, size in zip(points, color_hexes, size_array):
        scene_js += f'''
        {{
            const geo = new THREE.SphereGeometry({size}, 16, 12);
            const mat = new THREE.MeshPhongMaterial({{
                color: '{color}',
                transparent: {str(opacity < 1).lower()},
                opacity: {opacity}
            }});
            const sphere = new THREE.Mesh(geo, mat);
            sphere.position.set({pt[0]}, {pt[1]}, {pt[2]});
            pointsGroup.add(sphere);
        }}
    '''
    scene_js += '''
        scene.add(pointsGroup);
    '''

    # Calculate camera position
    extent = np.array(box_size) if box_size else (points.max(axis=0) - points.min(axis=0))
    max_extent = max(extent) * 1.5
    cam_pos = (center[0] + max_extent, center[1] + max_extent * 0.5, center[2] + max_extent)

    _webgl_scene(
        scene_objects=scene_js,
        background=background,
        camera_position=cam_pos,
        camera_target=tuple(center),
        unit_box=unit_box,
        box_size=box_size,
        box_color=box_color,
        box_labels=box_labels,
        auto_rotate=auto_rotate
    )


def line_3d(
    points,
    color: str = '#00d4ff',
    width: float = 2.0,
    opacity: float = 1.0,
    unit_box: bool = True,
    box_size: tuple = None,
    box_color: str = '#444444',
    box_labels: bool = True,
    background: str = '#1e1e2e',
    auto_rotate: bool = False
):
    """
    Render a 3D line/path through points.

    Args:
        points: Nx3 array-like of (x, y, z) coordinates
        color: Line color (hex string)
        width: Line width in pixels
        opacity: Line opacity 0-1
        unit_box: Show bounding box
        box_size: Box size (w, h, d), auto-calculated if None
        box_color: Color of box edges
        box_labels: Show axis labels
        background: Background color
        auto_rotate: Auto-rotate the scene

    Example:
        import numpy as np
        from cm.views import line_3d

        # Helix
        t = np.linspace(0, 4*np.pi, 100)
        points = np.column_stack([np.cos(t), np.sin(t), t/5])
        line_3d(points, color='#ff6b6b')
    """
    lines_3d(
        [points],
        colors=[color],
        width=width,
        opacity=opacity,
        unit_box=unit_box,
        box_size=box_size,
        box_color=box_color,
        box_labels=box_labels,
        background=background,
        auto_rotate=auto_rotate
    )


def lines_3d(
    paths,
    colors=None,
    width: float = 2.0,
    opacity: float = 1.0,
    unit_box: bool = True,
    box_size: tuple = None,
    box_color: str = '#444444',
    box_labels: bool = True,
    background: str = '#1e1e2e',
    auto_rotate: bool = False
):
    """
    Render multiple 3D lines/paths.

    Args:
        paths: List of Nx3 array-like, each representing a path
        colors: Optional list of colors (one per path), or single color for all
        width: Line width in pixels
        opacity: Line opacity 0-1
        unit_box: Show bounding box
        box_size: Box size (w, h, d), auto-calculated if None
        box_color: Color of box edges
        box_labels: Show axis labels
        background: Background color
        auto_rotate: Auto-rotate the scene

    Example:
        import numpy as np
        from cm.views import lines_3d

        # Multiple trajectories
        paths = []
        for i in range(5):
            t = np.linspace(0, 2*np.pi, 50)
            path = np.column_stack([np.cos(t) + i, np.sin(t), t])
            paths.append(path)
        lines_3d(paths, colors=['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'])
    """
    import numpy as np

    # Convert paths to numpy arrays
    paths = [np.asarray(p) for p in paths]
    n_paths = len(paths)

    # Process colors
    if colors is None:
        colors = ['#00d4ff'] * n_paths
    elif isinstance(colors, str):
        colors = [colors] * n_paths

    # Calculate bounding box
    all_points = np.vstack(paths)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2
    extent = maxs - mins

    if unit_box and box_size is None:
        box_size = tuple(extent * 1.1)

    # Generate line geometry
    scene_js = ''
    for path, color in zip(paths, colors):
        vertices = ', '.join([f'{p[0]}, {p[1]}, {p[2]}' for p in path])
        scene_js += f'''
        {{
            const lineGeometry = new THREE.BufferGeometry();
            const lineVertices = new Float32Array([{vertices}]);
            lineGeometry.setAttribute('position', new THREE.BufferAttribute(lineVertices, 3));
            const lineMaterial = new THREE.LineBasicMaterial({{
                color: '{color}',
                linewidth: {width},
                transparent: {str(opacity < 1).lower()},
                opacity: {opacity}
            }});
            const line = new THREE.Line(lineGeometry, lineMaterial);
            scene.add(line);
        }}
    '''

    # Calculate camera position
    max_extent = max(extent) * 1.5
    cam_pos = (center[0] + max_extent, center[1] + max_extent * 0.5, center[2] + max_extent)

    _webgl_scene(
        scene_objects=scene_js,
        background=background,
        camera_position=cam_pos,
        camera_target=tuple(center),
        unit_box=unit_box,
        box_size=box_size,
        box_color=box_color,
        box_labels=box_labels,
        auto_rotate=auto_rotate
    )


def surface(
    f=None,
    x=None,
    y=None,
    z=None,
    x_range: tuple = (-5, 5),
    y_range: tuple = (-5, 5),
    resolution: int = 50,
    colormap: str = 'viridis',
    wireframe: bool = False,
    opacity: float = 1.0,
    unit_box: bool = True,
    box_size: tuple = None,
    box_color: str = '#444444',
    box_labels: bool = True,
    background: str = '#1e1e2e',
    auto_rotate: bool = False
):
    """
    Render a 3D surface from a function or data arrays.

    Args:
        f: Optional function f(x, y) -> z, where x, y can be scalars or arrays
        x: Optional 2D array of X coordinates (from meshgrid)
        y: Optional 2D array of Y coordinates (from meshgrid)
        z: Optional 2D array of Z values
        x_range: Range for X axis when using function (min, max)
        y_range: Range for Y axis when using function (min, max)
        resolution: Grid resolution when using function
        colormap: Colormap for surface coloring
        wireframe: Show as wireframe instead of solid
        opacity: Surface opacity 0-1
        unit_box: Show bounding box
        box_size: Box size (w, h, d), auto-calculated if None
        box_color: Color of box edges
        box_labels: Show axis labels
        background: Background color
        auto_rotate: Auto-rotate the scene

    Example:
        import numpy as np
        from cm.views import surface

        # From function
        surface(f=lambda x, y: np.sin(np.sqrt(x**2 + y**2)),
                x_range=(-5, 5), y_range=(-5, 5))

        # From data arrays
        X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        Z = np.sin(X) * np.cos(Y)
        surface(x=X, y=Y, z=Z)
    """
    import numpy as np

    # Generate grid from function or use provided arrays
    if f is not None:
        xs = np.linspace(x_range[0], x_range[1], resolution)
        ys = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(xs, ys)
        Z = f(X, Y)
    elif x is not None and y is not None and z is not None:
        X, Y, Z = np.asarray(x), np.asarray(y), np.asarray(z)
    else:
        raise ValueError("Must provide either f (function) or x, y, z arrays")

    rows, cols = Z.shape

    # Calculate bounds and normalization
    z_min, z_max = Z.min(), Z.max()
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # Normalize Z for coloring
    if z_max > z_min:
        Z_norm = (Z - z_min) / (z_max - z_min)
    else:
        Z_norm = np.zeros_like(Z)

    # Build vertex and face data
    # Vertices: flatten grid to list of (x, y, z)
    vertices = []
    colors = []
    for i in range(rows):
        for j in range(cols):
            vertices.append((X[i, j], Y[i, j], Z[i, j]))
            colors.append(_interpolate_colormap(Z_norm[i, j], colormap))

    # Faces: two triangles per grid cell
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Vertex indices
            v00 = i * cols + j
            v01 = i * cols + (j + 1)
            v10 = (i + 1) * cols + j
            v11 = (i + 1) * cols + (j + 1)
            # Two triangles
            faces.append((v00, v10, v01))
            faces.append((v01, v10, v11))

    # Generate JavaScript for the mesh
    vertex_data = ', '.join([f'{v[0]}, {v[1]}, {v[2]}' for v in vertices])
    color_data = ', '.join([f'{int(c[1:3], 16)/255}, {int(c[3:5], 16)/255}, {int(c[5:7], 16)/255}' for c in colors])
    face_data = ', '.join([f'{f[0]}, {f[1]}, {f[2]}' for f in faces])

    scene_js = f'''
        // Surface mesh
        const surfaceGeometry = new THREE.BufferGeometry();
        const vertices = new Float32Array([{vertex_data}]);
        const colors = new Float32Array([{color_data}]);
        const indices = new Uint32Array([{face_data}]);

        surfaceGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        surfaceGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        surfaceGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
        surfaceGeometry.computeVertexNormals();

        const surfaceMaterial = new THREE.MeshPhongMaterial({{
            vertexColors: true,
            side: THREE.DoubleSide,
            wireframe: {str(wireframe).lower()},
            transparent: {str(opacity < 1).lower()},
            opacity: {opacity},
            shininess: 30
        }});

        const surfaceMesh = new THREE.Mesh(surfaceGeometry, surfaceMaterial);
        scene.add(surfaceMesh);
    '''

    # Calculate camera and box
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2)
    extent = (x_max - x_min, y_max - y_min, z_max - z_min)

    if unit_box and box_size is None:
        box_size = tuple(e * 1.1 for e in extent)

    max_extent = max(extent) * 1.5
    cam_pos = (center[0] + max_extent * 0.8, center[1] - max_extent * 0.3, center[2] + max_extent * 0.6)

    _webgl_scene(
        scene_objects=scene_js,
        background=background,
        camera_position=cam_pos,
        camera_target=center,
        unit_box=unit_box,
        box_size=box_size,
        box_color=box_color,
        box_labels=box_labels,
        auto_rotate=auto_rotate
    )


def vector_field(
    positions=None,
    vectors=None,
    f=None,
    bounds: tuple = (-2, 2, -2, 2, -2, 2),
    resolution: int = 5,
    scale: float = 1.0,
    colormap: str = 'coolwarm',
    arrow_head_length: float = 0.2,
    arrow_head_width: float = 0.1,
    opacity: float = 1.0,
    unit_box: bool = True,
    box_size: tuple = None,
    box_color: str = '#444444',
    box_labels: bool = True,
    background: str = '#1e1e2e',
    auto_rotate: bool = False
):
    """
    Render a 3D vector field.

    Args:
        positions: Nx3 array of base positions for vectors
        vectors: Nx3 array of vector directions/magnitudes
        f: Optional function f(x, y, z) -> (vx, vy, vz) for grid sampling
        bounds: Grid bounds (x_min, x_max, y_min, y_max, z_min, z_max) when using f
        resolution: Grid resolution per axis when using f
        scale: Arrow length scale factor
        colormap: Colormap for vector magnitude coloring
        arrow_head_length: Arrow head length as fraction of total
        arrow_head_width: Arrow head width as fraction of length
        opacity: Arrow opacity 0-1
        unit_box: Show bounding box
        box_size: Box size (w, h, d), auto-calculated if None
        box_color: Color of box edges
        box_labels: Show axis labels
        background: Background color
        auto_rotate: Auto-rotate the scene

    Example:
        import numpy as np
        from cm.views import vector_field

        # From function (rotational field)
        vector_field(f=lambda x, y, z: (-y, x, 0),
                     bounds=(-2, 2, -2, 2, -0.5, 0.5),
                     resolution=6)

        # From data
        positions = np.random.uniform(-2, 2, (50, 3))
        vectors = np.column_stack([
            -positions[:, 1],
            positions[:, 0],
            np.zeros(50)
        ])
        vector_field(positions, vectors)
    """
    import numpy as np

    # Generate from function or use provided arrays
    if f is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        zs = np.linspace(z_min, z_max, resolution)

        positions_list = []
        vectors_list = []

        for x in xs:
            for y in ys:
                for z in zs:
                    result = f(x, y, z)
                    positions_list.append((x, y, z))
                    vectors_list.append(result)

        positions = np.array(positions_list)
        vectors = np.array(vectors_list)
    elif positions is not None and vectors is not None:
        positions = np.asarray(positions)
        vectors = np.asarray(vectors)
    else:
        raise ValueError("Must provide either f (function) or positions and vectors arrays")

    n_vectors = len(positions)

    # Calculate magnitudes for coloring
    magnitudes = np.linalg.norm(vectors, axis=1)
    mag_min, mag_max = magnitudes.min(), magnitudes.max()
    if mag_max > mag_min:
        mag_norm = (magnitudes - mag_min) / (mag_max - mag_min)
    else:
        mag_norm = np.zeros(n_vectors)

    # Scale vectors
    scaled_vectors = vectors * scale

    # Generate arrow geometry
    scene_js = '''
        const arrowsGroup = new THREE.Group();
    '''

    for i, (pos, vec, mag) in enumerate(zip(positions, scaled_vectors, mag_norm)):
        length = np.linalg.norm(vec)
        if length < 0.001:
            continue

        color = _interpolate_colormap(mag, colormap)

        # Arrow direction
        direction = vec / length

        scene_js += f'''
        {{
            const dir = new THREE.Vector3({direction[0]}, {direction[1]}, {direction[2]});
            const origin = new THREE.Vector3({pos[0]}, {pos[1]}, {pos[2]});
            const length = {length};
            const headLength = length * {arrow_head_length};
            const headWidth = headLength * {arrow_head_width / arrow_head_length};
            const arrow = new THREE.ArrowHelper(dir, origin, length, '{color}', headLength, headWidth);
            arrowsGroup.add(arrow);
        }}
    '''

    scene_js += '''
        scene.add(arrowsGroup);
    '''

    # Calculate camera and box
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2
    extent = maxs - mins

    if unit_box and box_size is None:
        box_size = tuple(extent * 1.2)

    max_extent = max(extent) * 2.0
    cam_pos = (center[0] + max_extent, center[1] + max_extent * 0.5, center[2] + max_extent)

    _webgl_scene(
        scene_objects=scene_js,
        background=background,
        camera_position=cam_pos,
        camera_target=tuple(center),
        unit_box=unit_box,
        box_size=box_size,
        box_color=box_color,
        box_labels=box_labels,
        auto_rotate=auto_rotate
    )


def molecule(
    atoms,
    bonds=None,
    style: str = 'ball-stick',
    atom_scale: float = 1.0,
    bond_radius: float = 0.1,
    unit_box: bool = False,
    box_size: tuple = None,
    box_color: str = '#444444',
    box_labels: bool = True,
    background: str = '#1e1e2e',
    auto_rotate: bool = True
):
    """
    Render a molecular structure.

    Args:
        atoms: List of (element, x, y, z) tuples
        bonds: Optional list of (atom_index_1, atom_index_2) tuples
        style: Rendering style - 'ball-stick', 'spacefill', or 'stick'
        atom_scale: Scale factor for atom radii
        bond_radius: Bond cylinder radius (for ball-stick and stick styles)
        unit_box: Show bounding box
        box_size: Box size (w, h, d), auto-calculated if None
        box_color: Color of box edges
        box_labels: Show axis labels
        background: Background color
        auto_rotate: Auto-rotate the scene

    Example:
        from cm.views import molecule

        # Water molecule
        molecule([
            ('O', 0.0, 0.0, 0.0),
            ('H', 0.96, 0.0, 0.0),
            ('H', -0.24, 0.93, 0.0)
        ], bonds=[(0, 1), (0, 2)])

        # With different styles
        molecule(atoms, bonds, style='spacefill')
        molecule(atoms, bonds, style='stick')
    """
    import numpy as np

    # Normalize to flat (elem, x, y, z) format
    if atoms and isinstance(atoms[0][1], (tuple, list, np.ndarray)):
        atoms = [(a[0], float(a[1][0]), float(a[1][1]), float(a[1][2])) for a in atoms]

    n_atoms = len(atoms)
    positions = np.array([[a[1], a[2], a[3]] for a in atoms])
    elements = [a[0] for a in atoms]

    # Calculate bounding box
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2
    extent = maxs - mins

    if unit_box and box_size is None:
        box_size = tuple(extent * 1.2 + 2.0)

    # Generate atom geometry
    scene_js = '''
        const moleculeGroup = new THREE.Group();
    '''

    # Determine radius scale based on style
    if style == 'spacefill':
        radius_scale = atom_scale * 1.0
        show_bonds = False
    elif style == 'stick':
        radius_scale = atom_scale * 0.2
        show_bonds = True
    else:  # ball-stick
        radius_scale = atom_scale * 0.4
        show_bonds = True

    # Add atoms
    for i, (element, x, y, z) in enumerate(atoms):
        elem_data = ELEMENT_DATA.get(element, {'color': '#FF00FF', 'radius': 1.0})
        color = elem_data['color']
        radius = elem_data['radius'] * radius_scale

        scene_js += f'''
        {{
            const atomGeo = new THREE.SphereGeometry({radius}, 32, 24);
            const atomMat = new THREE.MeshPhongMaterial({{
                color: '{color}',
                shininess: 80,
                specular: 0x444444
            }});
            const atom = new THREE.Mesh(atomGeo, atomMat);
            atom.position.set({x}, {y}, {z});
            moleculeGroup.add(atom);
        }}
    '''

    # Add bonds
    if show_bonds and bonds:
        for i1, i2 in bonds:
            p1 = positions[i1]
            p2 = positions[i2]
            mid = (p1 + p2) / 2
            diff = p2 - p1
            length = np.linalg.norm(diff)

            if length < 0.01:
                continue

            # Calculate rotation to align cylinder with bond
            direction = diff / length

            scene_js += f'''
        {{
            const bondGeo = new THREE.CylinderGeometry({bond_radius}, {bond_radius}, {length}, 8);
            const bondMat = new THREE.MeshPhongMaterial({{
                color: '#888888',
                shininess: 30
            }});
            const bond = new THREE.Mesh(bondGeo, bondMat);

            // Position at midpoint
            bond.position.set({mid[0]}, {mid[1]}, {mid[2]});

            // Align cylinder to bond direction
            const direction = new THREE.Vector3({direction[0]}, {direction[1]}, {direction[2]});
            const axis = new THREE.Vector3(0, 1, 0);
            bond.quaternion.setFromUnitVectors(axis, direction);

            moleculeGroup.add(bond);
        }}
    '''

    scene_js += '''
        scene.add(moleculeGroup);
    '''

    # Calculate camera position
    max_extent = max(extent) + 5
    cam_pos = (center[0] + max_extent, center[1] + max_extent * 0.3, center[2] + max_extent)

    _webgl_scene(
        scene_objects=scene_js,
        background=background,
        camera_position=cam_pos,
        camera_target=tuple(center),
        unit_box=unit_box,
        box_size=box_size,
        box_color=box_color,
        box_labels=box_labels,
        auto_rotate=auto_rotate
    )


def _parse_xyz(content: str):
    """
    Parse XYZ file format into list of (element, x, y, z) tuples.

    XYZ format:
        N                    (number of atoms)
        comment line
        Element X Y Z
        Element X Y Z
        ...
    """
    lines = content.strip().split('\n')
    if len(lines) < 3:
        raise ValueError("XYZ file must have at least 3 lines")

    n_atoms = int(lines[0].strip())
    # Skip comment line (lines[1])

    atoms = []
    for i in range(2, 2 + n_atoms):
        if i >= len(lines):
            break
        parts = lines[i].split()
        if len(parts) >= 4:
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((element, x, y, z))

    return atoms


def _infer_bonds(atoms, bond_tolerance: float = 0.4):
    """
    Infer bonds between atoms based on distance and covalent radii.

    A bond is created if the distance between two atoms is less than
    the sum of their covalent radii plus a tolerance.
    """
    import numpy as np

    positions = np.array([[a[1], a[2], a[3]] for a in atoms])
    elements = [a[0] for a in atoms]
    n = len(atoms)

    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            r1 = ELEMENT_DATA.get(elements[i], {'radius': 1.0})['radius']
            r2 = ELEMENT_DATA.get(elements[j], {'radius': 1.0})['radius']
            max_bond_length = r1 + r2 + bond_tolerance
            if dist < max_bond_length:
                bonds.append((i, j))

    return bonds


def molecule_xyz(
    xyz_content: str,
    style: str = 'ball-stick',
    infer_bonds: bool = True,
    bond_tolerance: float = 0.4,
    atom_scale: float = 1.0,
    bond_radius: float = 0.1,
    unit_box: bool = False,
    box_size: tuple = None,
    box_color: str = '#444444',
    box_labels: bool = True,
    background: str = '#1e1e2e',
    auto_rotate: bool = True
):
    """
    Render a molecule from XYZ file content.

    Args:
        xyz_content: String content of XYZ file
        style: Rendering style - 'ball-stick', 'spacefill', or 'stick'
        infer_bonds: Automatically detect bonds based on atomic distances
        bond_tolerance: Extra distance tolerance for bond detection (Angstroms)
        atom_scale: Scale factor for atom radii
        bond_radius: Bond cylinder radius
        unit_box: Show bounding box
        box_size: Box size (w, h, d)
        box_color: Color of box edges
        box_labels: Show axis labels
        background: Background color
        auto_rotate: Auto-rotate the scene

    Example:
        from cm.views import molecule_xyz

        xyz = '''3
        Water molecule
        O  0.000  0.000  0.117
        H  0.756  0.000 -0.469
        H -0.756  0.000 -0.469
        '''
        molecule_xyz(xyz)

        # From file
        with open('caffeine.xyz') as f:
            molecule_xyz(f.read())
    """
    atoms = _parse_xyz(xyz_content)

    bonds = None
    if infer_bonds:
        bonds = _infer_bonds(atoms, bond_tolerance)

    molecule(
        atoms=atoms,
        bonds=bonds,
        style=style,
        atom_scale=atom_scale,
        bond_radius=bond_radius,
        unit_box=unit_box,
        box_size=box_size,
        box_color=box_color,
        box_labels=box_labels,
        background=background,
        auto_rotate=auto_rotate
    )


def _parse_cif(content: str):
    """
    Parse CIF (Crystallographic Information File) format.

    Returns a dict with:
        - cell_a, cell_b, cell_c: Unit cell lengths
        - cell_alpha, cell_beta, cell_gamma: Unit cell angles (degrees)
        - atoms: List of (element, frac_x, frac_y, frac_z) in fractional coords
    """
    import re
    import math

    result = {
        'cell_a': 1.0, 'cell_b': 1.0, 'cell_c': 1.0,
        'cell_alpha': 90.0, 'cell_beta': 90.0, 'cell_gamma': 90.0,
        'atoms': []
    }

    lines = content.split('\n')

    # Parse cell parameters
    for line in lines:
        line = line.strip()
        if line.startswith('_cell_length_a'):
            match = re.search(r'[\d.]+', line.split()[-1])
            if match:
                result['cell_a'] = float(match.group())
        elif line.startswith('_cell_length_b'):
            match = re.search(r'[\d.]+', line.split()[-1])
            if match:
                result['cell_b'] = float(match.group())
        elif line.startswith('_cell_length_c'):
            match = re.search(r'[\d.]+', line.split()[-1])
            if match:
                result['cell_c'] = float(match.group())
        elif line.startswith('_cell_angle_alpha'):
            match = re.search(r'[\d.]+', line.split()[-1])
            if match:
                result['cell_alpha'] = float(match.group())
        elif line.startswith('_cell_angle_beta'):
            match = re.search(r'[\d.]+', line.split()[-1])
            if match:
                result['cell_beta'] = float(match.group())
        elif line.startswith('_cell_angle_gamma'):
            match = re.search(r'[\d.]+', line.split()[-1])
            if match:
                result['cell_gamma'] = float(match.group())

    # Find atom_site loop
    in_atom_loop = False
    atom_columns = []
    column_indices = {}

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith('loop_'):
            in_atom_loop = False
            atom_columns = []

        if line.startswith('_atom_site'):
            in_atom_loop = True
            col_name = line.split()[0]
            atom_columns.append(col_name)
            column_indices[col_name] = len(atom_columns) - 1

        elif in_atom_loop and not line.startswith('_') and line and not line.startswith('loop_'):
            # This is an atom data line
            parts = line.split()
            if len(parts) >= len(atom_columns):
                # Get element/type
                element = None
                if '_atom_site_type_symbol' in column_indices:
                    idx = column_indices['_atom_site_type_symbol']
                    element = re.sub(r'[^A-Za-z]', '', parts[idx])
                elif '_atom_site_label' in column_indices:
                    idx = column_indices['_atom_site_label']
                    element = re.sub(r'[^A-Za-z]', '', parts[idx])

                # Get fractional coordinates
                frac_x = frac_y = frac_z = 0.0
                if '_atom_site_fract_x' in column_indices:
                    val = parts[column_indices['_atom_site_fract_x']]
                    frac_x = float(re.search(r'[-\d.]+', val).group())
                if '_atom_site_fract_y' in column_indices:
                    val = parts[column_indices['_atom_site_fract_y']]
                    frac_y = float(re.search(r'[-\d.]+', val).group())
                if '_atom_site_fract_z' in column_indices:
                    val = parts[column_indices['_atom_site_fract_z']]
                    frac_z = float(re.search(r'[-\d.]+', val).group())

                if element:
                    result['atoms'].append((element, frac_x, frac_y, frac_z))

    return result


def _frac_to_cart(frac_coords, a, b, c, alpha, beta, gamma):
    """Convert fractional coordinates to Cartesian using cell parameters."""
    import numpy as np

    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Calculate transformation matrix
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)

    # Volume factor
    v = np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
                + 2*cos_alpha*cos_beta*cos_gamma)

    # Transformation matrix (fractional to Cartesian)
    M = np.array([
        [a, b*cos_gamma, c*cos_beta],
        [0, b*sin_gamma, c*(cos_alpha - cos_beta*cos_gamma)/sin_gamma],
        [0, 0, c*v/sin_gamma]
    ])

    frac = np.array(frac_coords)
    return M @ frac


def crystal(
    cif_content: str,
    supercell: tuple = (1, 1, 1),
    style: str = 'ball-stick',
    infer_bonds: bool = True,
    bond_tolerance: float = 0.4,
    atom_scale: float = 1.0,
    bond_radius: float = 0.1,
    unit_box: bool = True,
    box_color: str = '#666666',
    box_labels: bool = True,
    background: str = '#1e1e2e',
    auto_rotate: bool = True
):
    """
    Render a crystal structure from CIF file content.

    Args:
        cif_content: String content of CIF file
        supercell: Number of unit cells to replicate in (a, b, c) directions
        style: Rendering style - 'ball-stick', 'spacefill', or 'stick'
        infer_bonds: Automatically detect bonds based on atomic distances
        bond_tolerance: Extra distance tolerance for bond detection (Angstroms)
        atom_scale: Scale factor for atom radii
        bond_radius: Bond cylinder radius
        unit_box: Show unit cell box
        box_color: Color of unit cell edges
        box_labels: Show axis labels
        background: Background color
        auto_rotate: Auto-rotate the scene

    Example:
        from cm.views import crystal

        # Load and display a crystal structure
        with open('structure.cif') as f:
            crystal(f.read())

        # With supercell expansion
        crystal(cif_content, supercell=(2, 2, 2))
    """
    import numpy as np

    cif_data = _parse_cif(cif_content)

    a = cif_data['cell_a']
    b = cif_data['cell_b']
    c = cif_data['cell_c']
    alpha = cif_data['cell_alpha']
    beta = cif_data['cell_beta']
    gamma = cif_data['cell_gamma']

    # Generate atoms for supercell
    atoms = []
    na, nb, nc = supercell

    for ia in range(na):
        for ib in range(nb):
            for ic in range(nc):
                for element, fx, fy, fz in cif_data['atoms']:
                    # Shift fractional coordinates for supercell
                    new_fx = (fx + ia) / na
                    new_fy = (fy + ib) / nb
                    new_fz = (fz + ic) / nc

                    # Scale to full supercell
                    frac = (new_fx * na, new_fy * nb, new_fz * nc)

                    # Convert to Cartesian
                    cart = _frac_to_cart(frac, a, b, c, alpha, beta, gamma)
                    atoms.append((element, cart[0], cart[1], cart[2]))

    # Calculate unit cell box size for supercell
    # Get the corners of the supercell in Cartesian coordinates
    corners = [
        _frac_to_cart((0, 0, 0), a*na, b*nb, c*nc, alpha, beta, gamma),
        _frac_to_cart((1, 0, 0), a*na, b*nb, c*nc, alpha, beta, gamma),
        _frac_to_cart((0, 1, 0), a*na, b*nb, c*nc, alpha, beta, gamma),
        _frac_to_cart((0, 0, 1), a*na, b*nb, c*nc, alpha, beta, gamma),
        _frac_to_cart((1, 1, 0), a*na, b*nb, c*nc, alpha, beta, gamma),
        _frac_to_cart((1, 0, 1), a*na, b*nb, c*nc, alpha, beta, gamma),
        _frac_to_cart((0, 1, 1), a*na, b*nb, c*nc, alpha, beta, gamma),
        _frac_to_cart((1, 1, 1), a*na, b*nb, c*nc, alpha, beta, gamma),
    ]
    corners = np.array(corners)
    box_size = tuple(corners.max(axis=0) - corners.min(axis=0))

    # Infer bonds
    bonds = None
    if infer_bonds:
        bonds = _infer_bonds(atoms, bond_tolerance)

    molecule(
        atoms=atoms,
        bonds=bonds,
        style=style,
        atom_scale=atom_scale,
        bond_radius=bond_radius,
        unit_box=unit_box,
        box_size=box_size,
        box_color=box_color,
        box_labels=box_labels,
        background=background,
        auto_rotate=auto_rotate
    )


def orbital(
    hf_result,
    mo_index: Union[int, str] = -1,
    isovalue: float = 0.02,
    resolution: int = 40,
    positive_color: str = '#4444ff',
    negative_color: str = '#ff4444',
    opacity: float = 0.7,
    show_molecule: bool = True,
    atom_scale: float = 0.3,
    bond_radius: float = 0.08,
    unit_box: bool = False,
    background: str = '#1e1e2e',
    auto_rotate: bool = False
):
    """
    Visualize a molecular orbital from Hartree-Fock results.

    Renders isosurfaces showing the positive (blue) and negative (red) lobes
    of a molecular orbital, optionally with the molecular structure.

    Args:
        hf_result: HFResult from hartree_fock() calculation. Must include
                   atoms attribute or be passed with atoms separately.
        mo_index: Which MO to visualize:
                  - Integer: 0-indexed orbital number
                  - Negative int: HOMO-n (e.g., -1 = HOMO, -2 = HOMO-1)
                  - 'HOMO': Highest occupied MO
                  - 'LUMO': Lowest unoccupied MO
        isovalue: Isosurface value (default 0.02, typical range 0.01-0.05)
        resolution: Grid resolution for orbital evaluation (default 40)
        positive_color: Color for positive lobe (default blue)
        negative_color: Color for negative lobe (default red)
        opacity: Surface opacity 0-1 (default 0.7)
        show_molecule: Show molecular structure (default True)
        atom_scale: Atom radius scale (default 0.3)
        bond_radius: Bond cylinder radius (default 0.08)
        unit_box: Show bounding box (default False)
        background: Background color
        auto_rotate: Auto-rotate the scene (default True)

    Example:
        from cm.qm import hartree_fock
        from cm.views import orbital

        # H2 molecule
        result = hartree_fock([
            ('H', (0, 0, 0)),
            ('H', (0.74, 0, 0))
        ], n_electrons=2)

        # Visualize HOMO (bonding orbital)
        orbital(result, mo_index='HOMO')

        # Visualize LUMO (antibonding orbital)
        orbital(result, mo_index='LUMO')

        # Visualize specific orbital with custom settings
        orbital(result, mo_index=0, isovalue=0.03, opacity=0.8)
    """
    import numpy as np

    # Import orbital extraction functions
    from cm.qm.integrals.visualization.orbital import (
        extract_orbital_isosurface,
        create_orbital_grid
    )
    from cm.qm.integrals.basis import BasisSet

    # Get atoms from hf_result or require them
    if hasattr(hf_result, 'atoms'):
        atoms = hf_result.atoms
    else:
        raise ValueError(
            "hf_result must have 'atoms' attribute. "
            "Use the newer hartree_fock() that stores atom information."
        )

    # Build basis set
    basis = BasisSet('STO-3G')
    basis.build_for_molecule(atoms)

    # Handle string mo_index
    n_electrons = sum(
        {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}.get(el, 1)
        for el, _ in atoms
    )
    n_occ = n_electrons // 2  # Number of occupied orbitals

    if isinstance(mo_index, str):
        if mo_index.upper() == 'HOMO':
            mo_index = n_occ - 1
        elif mo_index.upper() == 'LUMO':
            mo_index = n_occ
        else:
            raise ValueError(f"Unknown orbital name: {mo_index}. Use 'HOMO', 'LUMO', or integer.")

    # Extract isosurfaces
    pos_verts, pos_faces, neg_verts, neg_faces = extract_orbital_isosurface(
        hf_result.mo_coefficients,
        mo_index,
        basis,
        atoms,
        isovalue=isovalue,
        resolution=resolution
    )

    # Convert from Bohr to Angstrom for visualization
    BOHR_TO_ANGSTROM = 0.529177
    pos_verts = pos_verts * BOHR_TO_ANGSTROM
    neg_verts = neg_verts * BOHR_TO_ANGSTROM

    # Calculate molecule extent
    positions = np.array([[a[1][0], a[1][1], a[1][2]] for a in atoms])
    center = positions.mean(axis=0)
    extent = positions.max(axis=0) - positions.min(axis=0)
    max_extent = max(extent) + 5

    # Generate JavaScript for orbital surfaces
    scene_js = '''
        const orbitalGroup = new THREE.Group();
    '''

    # Add positive lobe
    if len(pos_verts) > 0:
        pos_vertex_data = ', '.join([f'{v[0]}, {v[1]}, {v[2]}' for v in pos_verts])
        pos_face_data = ', '.join([f'{f[0]}, {f[1]}, {f[2]}' for f in pos_faces])

        scene_js += f'''
        // Positive lobe (blue)
        {{
            const posGeometry = new THREE.BufferGeometry();
            const posVertices = new Float32Array([{pos_vertex_data}]);
            const posIndices = new Uint32Array([{pos_face_data}]);
            posGeometry.setAttribute('position', new THREE.BufferAttribute(posVertices, 3));
            posGeometry.setIndex(new THREE.BufferAttribute(posIndices, 1));
            posGeometry.computeVertexNormals();

            const posMaterial = new THREE.MeshPhongMaterial({{
                color: '{positive_color}',
                transparent: true,
                opacity: {opacity},
                side: THREE.DoubleSide,
                shininess: 50
            }});

            const posMesh = new THREE.Mesh(posGeometry, posMaterial);
            orbitalGroup.add(posMesh);
        }}
    '''

    # Add negative lobe
    if len(neg_verts) > 0:
        neg_vertex_data = ', '.join([f'{v[0]}, {v[1]}, {v[2]}' for v in neg_verts])
        neg_face_data = ', '.join([f'{f[0]}, {f[1]}, {f[2]}' for f in neg_faces])

        scene_js += f'''
        // Negative lobe (red)
        {{
            const negGeometry = new THREE.BufferGeometry();
            const negVertices = new Float32Array([{neg_vertex_data}]);
            const negIndices = new Uint32Array([{neg_face_data}]);
            negGeometry.setAttribute('position', new THREE.BufferAttribute(negVertices, 3));
            negGeometry.setIndex(new THREE.BufferAttribute(negIndices, 1));
            negGeometry.computeVertexNormals();

            const negMaterial = new THREE.MeshPhongMaterial({{
                color: '{negative_color}',
                transparent: true,
                opacity: {opacity},
                side: THREE.DoubleSide,
                shininess: 50
            }});

            const negMesh = new THREE.Mesh(negGeometry, negMaterial);
            orbitalGroup.add(negMesh);
        }}
    '''

    # Add molecule if requested
    if show_molecule:
        elements = [a[0] for a in atoms]

        # Add atoms
        for i, (element, pos) in enumerate(atoms):
            x, y, z = pos
            elem_data = ELEMENT_DATA.get(element, {'color': '#FF00FF', 'radius': 1.0})
            color = elem_data['color']
            radius = elem_data['radius'] * atom_scale

            scene_js += f'''
        {{
            const atomGeo = new THREE.SphereGeometry({radius}, 24, 16);
            const atomMat = new THREE.MeshPhongMaterial({{
                color: '{color}',
                shininess: 80
            }});
            const atom = new THREE.Mesh(atomGeo, atomMat);
            atom.position.set({x}, {y}, {z});
            orbitalGroup.add(atom);
        }}
    '''

        # Infer and add bonds
        bonds = _infer_bonds([(el, x, y, z) for (el, (x, y, z)) in atoms], bond_tolerance=0.4)
        for i1, i2 in bonds:
            p1 = np.array(atoms[i1][1])
            p2 = np.array(atoms[i2][1])
            mid = (p1 + p2) / 2
            diff = p2 - p1
            length = np.linalg.norm(diff)

            if length < 0.01:
                continue

            direction = diff / length

            scene_js += f'''
        {{
            const bondGeo = new THREE.CylinderGeometry({bond_radius}, {bond_radius}, {length}, 8);
            const bondMat = new THREE.MeshPhongMaterial({{
                color: '#888888',
                shininess: 30
            }});
            const bond = new THREE.Mesh(bondGeo, bondMat);
            bond.position.set({mid[0]}, {mid[1]}, {mid[2]});

            const direction = new THREE.Vector3({direction[0]}, {direction[1]}, {direction[2]});
            const axis = new THREE.Vector3(0, 1, 0);
            bond.quaternion.setFromUnitVectors(axis, direction);

            orbitalGroup.add(bond);
        }}
    '''

    scene_js += '''
        scene.add(orbitalGroup);
    '''

    # Camera position
    cam_pos = (center[0] + max_extent, center[1] + max_extent * 0.3, center[2] + max_extent)

    _webgl_scene(
        scene_objects=scene_js,
        background=background,
        camera_position=cam_pos,
        camera_target=tuple(center),
        unit_box=unit_box,
        auto_rotate=auto_rotate
    )
