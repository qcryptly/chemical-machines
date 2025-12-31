# %% Cell 1 - Basic WebGL with Three.js Helper
# Demonstrates the cm.views.webgl_threejs() function for easy 3D rendering

from cm.views import webgl_threejs, log

log("Rendering a spinning cube with Three.js...", level="info")

# Use the helper function for common Three.js setup
webgl_threejs(
    scene_setup='''
        // Create a cube with normal material (shows rainbow colors based on normals)
        const geometry = new THREE.BoxGeometry(2, 2, 2);
        const material = new THREE.MeshNormalMaterial();
        const cube = new THREE.Mesh(geometry, material);
        scene.add(cube);

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);

        // Add directional light
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

log("WebGL cube rendered! Check the WebGL View panel above.", level="success")

# %% Cell 2 - Molecule-like Spheres
# Create a simple molecule visualization

from cm.views import webgl_threejs, log

log("Rendering a molecule-like structure...", level="info")

webgl_threejs(
    scene_setup='''
        // Water molecule-like structure (H2O)
        const sphereGeo = new THREE.SphereGeometry(0.5, 32, 32);

        // Oxygen (red, center)
        const oxygenMat = new THREE.MeshPhongMaterial({ color: 0xff0000 });
        const oxygen = new THREE.Mesh(sphereGeo, oxygenMat);
        scene.add(oxygen);

        // Hydrogen atoms (white)
        const hydrogenMat = new THREE.MeshPhongMaterial({ color: 0xffffff });
        const hydrogenGeo = new THREE.SphereGeometry(0.3, 32, 32);

        const h1 = new THREE.Mesh(hydrogenGeo, hydrogenMat);
        h1.position.set(-0.8, 0.6, 0);
        scene.add(h1);

        const h2 = new THREE.Mesh(hydrogenGeo, hydrogenMat);
        h2.position.set(0.8, 0.6, 0);
        scene.add(h2);

        // Bonds (cylinders)
        const bondGeo = new THREE.CylinderGeometry(0.1, 0.1, 1, 8);
        const bondMat = new THREE.MeshPhongMaterial({ color: 0x888888 });

        const bond1 = new THREE.Mesh(bondGeo, bondMat);
        bond1.position.set(-0.4, 0.3, 0);
        bond1.rotation.z = Math.PI / 4;
        scene.add(bond1);

        const bond2 = new THREE.Mesh(bondGeo, bondMat);
        bond2.position.set(0.4, 0.3, 0);
        bond2.rotation.z = -Math.PI / 4;
        scene.add(bond2);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);

        const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
        light1.position.set(5, 5, 5);
        scene.add(light1);

        const light2 = new THREE.DirectionalLight(0x4488ff, 0.4);
        light2.position.set(-5, -5, -5);
        scene.add(light2);
    ''',
    animate_loop='''
        // Slow rotation
        scene.rotation.y += 0.005;
    ''',
    camera_position=(0, 0, 4),
    background="#1e1e2e"
)

log("H2O molecule rendered! Use mouse to orbit.", level="success")

# %% Cell 3 - Custom Raw WebGL
# For full control, use webgl() directly with raw HTML

from cm.views import webgl, log

log("Rendering custom WebGL content...", level="info")

webgl('''
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin: 0; padding: 0; }
        body { background: #1e1e2e; overflow: hidden; }
        canvas { display: block; }
        .info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: #cdd6f4;
            font-family: sans-serif;
            font-size: 14px;
            background: rgba(0,0,0,0.5);
            padding: 8px 12px;
            border-radius: 4px;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div class="info">Particle System - 1000 particles</div>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#1e1e2e');

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 50;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Create particle system
        const particleCount = 1000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            // Random positions in a sphere
            const r = 30 * Math.cbrt(Math.random());
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);

            positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = r * Math.cos(phi);

            // Random colors (blues and purples)
            colors[i * 3] = 0.3 + Math.random() * 0.4;
            colors[i * 3 + 1] = 0.3 + Math.random() * 0.5;
            colors[i * 3 + 2] = 0.8 + Math.random() * 0.2;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.5,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });

        const particles = new THREE.Points(geometry, material);
        scene.add(particles);

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            particles.rotation.y += 0.002;
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
''')

log("Particle system rendered!", level="success")
