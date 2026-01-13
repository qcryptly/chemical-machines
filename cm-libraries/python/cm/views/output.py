"""
Chemical Machines Views Module

A module for rendering HTML outputs from Python cells.
Outputs are written to .out/ directory and displayed in the workspace UI.

Environment Variables:
    CM_OUTPUT_FILE: Path to the output HTML file (e.g., .out/myfile.cell.py.html)
    CM_CELL_INDEX: Current cell index (0-based), or -1 for non-cell files
    CM_IS_CELL_FILE: "true" if this is a cell-based file, "false" otherwise
    CM_WORKSPACE_DIR: Path to the workspace directory

Usage:
    from cm.views import html, text, image, clear, log

    # Output HTML content
    html("<h1>Hello World</h1>")
    html("<p>Some <b>formatted</b> text</p>")

    # Output plain text (auto-escaped)
    text("Some plain text that will be escaped")

    # Output an image from file or base64
    image("plot.png")
    image(base64_data, mime_type="image/png")

    # Clear previous outputs for this cell
    clear()

    # Log with automatic formatting
    log("Status:", "Running...")
    log({"key": "value"})  # Auto-formats dicts/lists as JSON
"""

__all__ = [
    'html',
    'text',
    'log',
    'clear',
    'image',
    'savefig',
    'dataframe',
    'table',
]

import os
import json
import base64
import re
import html as html_module
from pathlib import Path
from typing import Any, Optional, Union

# Read environment variables
_OUTPUT_FILE = os.environ.get('CM_OUTPUT_FILE', '')
_CELL_INDEX = int(os.environ.get('CM_CELL_INDEX', '-1'))
_IS_CELL_FILE = os.environ.get('CM_IS_CELL_FILE', 'false').lower() == 'true'
_WORKSPACE_DIR = os.environ.get('CM_WORKSPACE_DIR', '')

# Cell delimiter for separating outputs between cells
CELL_DELIMITER = '<!-- CELL_DELIMITER -->'

# Track outputs for the CURRENT cell only (each cell runs in separate process)
_current_cell_outputs: list[str] = []

# HTML template
_HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 1rem;
            background: white;
            color: #333;
        }}
        pre {{
            background: #f5f5f5;
            padding: 0.5rem;
            border-radius: 4px;
            overflow-x: auto;
        }}
        code {{
            background: #f5f5f5;
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        table {{
            border-collapse: collapse;
            margin: 0.5rem 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 0.5rem;
            text-align: left;
        }}
        th {{
            background: #f5f5f5;
        }}
        .cm-log {{
            margin: 0.25rem 0;
            padding: 0.25rem 0.5rem;
            border-left: 3px solid #007acc;
            background: #f8f9fa;
        }}
        .cm-error {{
            border-left-color: #d32f2f;
            background: #ffebee;
        }}
        .cm-warning {{
            border-left-color: #f9a825;
            background: #fff8e1;
        }}
        .cm-success {{
            border-left-color: #388e3c;
            background: #e8f5e9;
        }}
        .cm-math {{
            margin: 1rem 0;
        }}
        .cm-math-center {{
            text-align: center;
        }}
        .cm-math-left {{
            text-align: left;
        }}
        .cm-math-right {{
            text-align: right;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>'''


def _get_output_path() -> Optional[Path]:
    """Get the output file path, creating parent directories if needed."""
    if not _OUTPUT_FILE:
        return None

    output_path = Path(_OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _read_existing_cells() -> dict[int, str]:
    """Read existing cell outputs from the HTML file."""
    output_path = _get_output_path()
    if not output_path or not output_path.exists():
        return {}

    try:
        content = output_path.read_text(encoding='utf-8')

        # Extract body content
        body_match = re.search(r'<body>(.*?)</body>', content, re.DOTALL)
        if not body_match:
            return {}

        body_content = body_match.group(1).strip()

        # Split by cell delimiter
        parts = body_content.split(CELL_DELIMITER)

        # Build dict of cell index -> content
        cells = {}
        for idx, part in enumerate(parts):
            stripped = part.strip()
            if stripped:
                cells[idx] = stripped

        return cells
    except Exception:
        return {}


def _write_outputs():
    """Write outputs to the HTML file, preserving other cells."""
    output_path = _get_output_path()
    if not output_path:
        return

    cell_idx = _CELL_INDEX if _CELL_INDEX >= 0 else 0

    if _IS_CELL_FILE:
        # Read existing cells from file
        existing_cells = _read_existing_cells()

        # Update current cell's content
        current_content = '\n'.join(_current_cell_outputs)
        existing_cells[cell_idx] = current_content

        # Build output with all cells in order
        max_idx = max(existing_cells.keys()) if existing_cells else 0
        parts = []
        for i in range(max_idx + 1):
            cell_content = existing_cells.get(i, '')
            parts.append(cell_content)

        content = f'\n{CELL_DELIMITER}\n'.join(parts)
    else:
        # Non-cell file: just output current content
        content = '\n'.join(_current_cell_outputs)

    # Write full HTML
    full_html = _HTML_TEMPLATE.format(content=content)
    output_path.write_text(full_html, encoding='utf-8')


def _add_output(content: str):
    """Add output content for the current cell."""
    _current_cell_outputs.append(content)
    _write_outputs()


def html(content):
    """
    Output raw HTML content.

    Args:
        content: HTML string or any value to output (will be converted to string)

    Example:
        html("<h1>Title</h1>")
        html("<div style='color: blue'>Blue text</div>")
        html(42)  # Numeric values are converted to strings
    """
    _add_output(str(content))


def text(content: str):
    """
    Output plain text (HTML-escaped).

    Args:
        content: Plain text string to output

    Example:
        text("Hello, World!")
        text("Special chars: <, >, &")
    """
    escaped = html_module.escape(str(content))
    _add_output(f'<pre>{escaped}</pre>')


def log(*args, level: str = 'info'):
    """
    Log values with automatic formatting.

    Args:
        *args: Values to log (strings, dicts, lists, etc.)
        level: Log level ('info', 'warning', 'error', 'success')

    Example:
        log("Processing file:", filename)
        log({"status": "ok", "count": 42})
        log("Error occurred!", level="error")
    """
    parts = []
    for arg in args:
        if isinstance(arg, (dict, list)):
            parts.append(json.dumps(arg, indent=2, default=str))
        else:
            parts.append(str(arg))

    content = ' '.join(parts)
    escaped = html_module.escape(content)

    css_class = 'cm-log'
    if level == 'error':
        css_class += ' cm-error'
    elif level == 'warning':
        css_class += ' cm-warning'
    elif level == 'success':
        css_class += ' cm-success'

    _add_output(f'<div class="{css_class}"><pre>{escaped}</pre></div>')


def image(
    source: Union[str, bytes],
    mime_type: str = 'image/png',
    alt: str = '',
    width: Optional[int] = None,
    height: Optional[int] = None
):
    """
    Output an image.

    Args:
        source: File path, URL, or base64-encoded bytes
        mime_type: MIME type for base64 data (default: image/png)
        alt: Alt text for accessibility
        width: Optional width in pixels
        height: Optional height in pixels

    Example:
        image("plot.png")
        image("/path/to/image.jpg")
        image(png_bytes, mime_type="image/png")
    """
    if isinstance(source, bytes):
        # Base64 encode binary data
        b64 = base64.b64encode(source).decode('ascii')
        src = f'data:{mime_type};base64,{b64}'
    elif isinstance(source, str):
        if source.startswith(('http://', 'https://', 'data:')):
            # URL or data URI
            src = source
        else:
            # File path - try to read and embed
            try:
                file_path = Path(source)
                if not file_path.is_absolute() and _WORKSPACE_DIR:
                    file_path = Path(_WORKSPACE_DIR) / source

                if file_path.exists():
                    data = file_path.read_bytes()
                    b64 = base64.b64encode(data).decode('ascii')
                    # Detect mime type from extension
                    ext = file_path.suffix.lower()
                    mime_types = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.svg': 'image/svg+xml',
                        '.webp': 'image/webp'
                    }
                    mime = mime_types.get(ext, mime_type)
                    src = f'data:{mime};base64,{b64}'
                else:
                    # File not found, use as-is (maybe relative URL)
                    src = source
            except Exception as e:
                log(f"Error loading image: {e}", level="error")
                return
    else:
        log(f"Invalid image source type: {type(source)}", level="error")
        return

    # Build style attribute
    styles = []
    if width:
        styles.append(f'width: {width}px')
    if height:
        styles.append(f'height: {height}px')
    style_attr = f' style="{"; ".join(styles)}"' if styles else ''

    alt_escaped = html_module.escape(alt)
    _add_output(f'<img src="{src}" alt="{alt_escaped}"{style_attr}>')


def table(
    data: list[list[Any]],
    headers: Optional[list[str]] = None
):
    """
    Output a table.

    Args:
        data: 2D list of cell values
        headers: Optional list of header strings

    Example:
        table([["Alice", 25], ["Bob", 30]], headers=["Name", "Age"])
    """
    rows = []

    if headers:
        header_cells = ''.join(f'<th>{html_module.escape(str(h))}</th>' for h in headers)
        rows.append(f'<tr>{header_cells}</tr>')

    for row in data:
        cells = ''.join(f'<td>{html_module.escape(str(cell))}</td>' for cell in row)
        rows.append(f'<tr>{cells}</tr>')

    table_html = f'<table>{"".join(rows)}</table>'
    _add_output(table_html)


def clear():
    """
    Clear all outputs for the current cell.

    Example:
        clear()  # Clear current cell's outputs
    """
    global _current_cell_outputs
    _current_cell_outputs = []
    _write_outputs()


def clear_all():
    """Clear all outputs for all cells (deletes the output file)."""
    output_path = _get_output_path()
    if output_path and output_path.exists():
        output_path.unlink()


# Matplotlib integration
def savefig(fig=None, **kwargs):
    """
    Save a matplotlib figure as an image output.

    Args:
        fig: Optional matplotlib figure (uses current figure if not specified)
        **kwargs: Additional arguments passed to fig.savefig()

    Example:
        import matplotlib.pyplot as plt
        plt.plot([1, 2, 3], [1, 4, 9])
        savefig()
    """
    try:
        import matplotlib.pyplot as plt
        import io

        if fig is None:
            fig = plt.gcf()

        buf = io.BytesIO()
        kwargs.setdefault('format', 'png')
        kwargs.setdefault('dpi', 100)
        kwargs.setdefault('bbox_inches', 'tight')
        fig.savefig(buf, **kwargs)
        buf.seek(0)

        image(buf.read(), mime_type='image/png')

    except ImportError:
        log("matplotlib is required for savefig()", level="error")


# Pandas integration
def dataframe(df, max_rows: int = 50):
    """
    Output a pandas DataFrame as an HTML table.

    Args:
        df: pandas DataFrame
        max_rows: Maximum rows to display (default: 50)

    Example:
        import pandas as pd
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        dataframe(df)
    """
    try:
        html_str = df.head(max_rows).to_html(index=True, classes='dataframe')
        if len(df) > max_rows:
            html_str += f'<p><em>Showing {max_rows} of {len(df)} rows</em></p>'
        _add_output(html_str)
    except Exception as e:
        log(f"Error rendering DataFrame: {e}", level="error")


# WebGL Main View
def webgl(content: str):
    """
    Output WebGL/3D content to the main visualization panel.

    This writes to a special .out/main.webgl.html file that is displayed
    in the collapsible WebGL panel at the top of the workspace.

    Args:
        content: Full HTML content including WebGL/Three.js code

    Example:
        webgl('''
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        </head>
        <body>
            <script>
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer();
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                const geometry = new THREE.BoxGeometry();
                const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
                const cube = new THREE.Mesh(geometry, material);
                scene.add(cube);

                camera.position.z = 5;

                function animate() {
                    requestAnimationFrame(animate);
                    cube.rotation.x += 0.01;
                    cube.rotation.y += 0.01;
                    renderer.render(scene, camera);
                }
                animate();
            </script>
        </body>
        </html>
        ''')
    """
    if not _WORKSPACE_DIR:
        log("CM_WORKSPACE_DIR not set, cannot write WebGL output", level="error")
        return

    webgl_path = Path(_WORKSPACE_DIR) / '.out' / 'main.webgl.html'
    webgl_path.parent.mkdir(parents=True, exist_ok=True)
    webgl_path.write_text(content, encoding='utf-8')


def webgl_threejs(
    scene_setup: str,
    animate_loop: str = "",
    width: str = "100%",
    height: str = "100%",
    background: str = "#1e1e2e",
    camera_position: tuple = (0, 0, 5),
    controls: bool = True
):
    """
    Output a Three.js scene with common boilerplate handled.

    Args:
        scene_setup: JavaScript code to set up the scene (add meshes, lights, etc.)
        animate_loop: Optional JavaScript code to run each animation frame
        width: CSS width of the canvas (default: 100%)
        height: CSS height of the canvas (default: 100%)
        background: Background color (default: dark theme)
        camera_position: Initial camera position tuple (x, y, z)
        controls: Enable OrbitControls for mouse interaction

    Example:
        webgl_threejs(
            scene_setup='''
                const geometry = new THREE.BoxGeometry();
                const material = new THREE.MeshNormalMaterial();
                const cube = new THREE.Mesh(geometry, material);
                scene.add(cube);

                const light = new THREE.DirectionalLight(0xffffff, 1);
                light.position.set(1, 1, 1);
                scene.add(light);
            ''',
            animate_loop='''
                cube.rotation.x += 0.01;
                cube.rotation.y += 0.01;
            ''',
            camera_position=(0, 0, 5)
        )
    """
    controls_import = '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>' if controls else ''
    controls_setup = '''
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
    ''' if controls else ''
    controls_update = 'controls.update();' if controls else ''

    content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            width: {width};
            height: {height};
            overflow: hidden;
            background: {background};
        }}
        canvas {{ display: block; width: 100% !important; height: 100% !important; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    {controls_import}
</head>
<body>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color('{background}');

        // Camera
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set({camera_position[0]}, {camera_position[1]}, {camera_position[2]});

        // Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        {controls_setup}

        // User scene setup
        {scene_setup}

        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            {controls_update}
            {animate_loop}
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>'''

    webgl(content)


# =============================================================================
