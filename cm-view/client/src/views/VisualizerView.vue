<template>
  <div class="visualizer">
    <div class="controls">
      <h2>Molecular Visualizer</h2>
      <div class="control-group">
        <label>Molecule:</label>
        <input
          v-model="searchQuery"
          @input="searchMolecules"
          placeholder="Search molecules..."
        />
      </div>
      <div class="control-group">
        <button @click="loadMolecule">Load</button>
        <button @click="resetCamera">Reset Camera</button>
      </div>
    </div>

    <div class="viewport" ref="viewport"></div>

    <div class="info" v-if="currentMolecule">
      <h3>{{ currentMolecule.name }}</h3>
      <p>Formula: {{ currentMolecule.formula }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

const viewport = ref(null)
const searchQuery = ref('')
const currentMolecule = ref(null)

let scene, camera, renderer, controls
let animationId = null

function initThree() {
  // Scene
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x1a1a1a)

  // Camera
  camera = new THREE.PerspectiveCamera(
    75,
    viewport.value.clientWidth / viewport.value.clientHeight,
    0.1,
    1000
  )
  camera.position.z = 5

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(viewport.value.clientWidth, viewport.value.clientHeight)
  viewport.value.appendChild(renderer.domElement)

  // Controls
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
  scene.add(ambientLight)

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight.position.set(5, 5, 5)
  scene.add(directionalLight)

  // Add sample molecule (water)
  addSampleMolecule()

  // Animation loop
  animate()

  // Handle resize
  window.addEventListener('resize', onResize)
}

function addSampleMolecule() {
  // Simple water molecule representation
  const oxygenGeometry = new THREE.SphereGeometry(0.3, 32, 32)
  const oxygenMaterial = new THREE.MeshPhongMaterial({ color: 0xff0000 })
  const oxygen = new THREE.Mesh(oxygenGeometry, oxygenMaterial)
  scene.add(oxygen)

  const hydrogenGeometry = new THREE.SphereGeometry(0.2, 32, 32)
  const hydrogenMaterial = new THREE.MeshPhongMaterial({ color: 0xffffff })

  const hydrogen1 = new THREE.Mesh(hydrogenGeometry, hydrogenMaterial)
  hydrogen1.position.set(0.8, 0.6, 0)
  scene.add(hydrogen1)

  const hydrogen2 = new THREE.Mesh(hydrogenGeometry, hydrogenMaterial)
  hydrogen2.position.set(-0.8, 0.6, 0)
  scene.add(hydrogen2)

  // Bonds
  const bondMaterial = new THREE.MeshPhongMaterial({ color: 0x888888 })

  const bond1Geometry = new THREE.CylinderGeometry(0.05, 0.05, 1, 8)
  const bond1 = new THREE.Mesh(bond1Geometry, bondMaterial)
  bond1.position.set(0.4, 0.3, 0)
  bond1.rotation.z = -Math.PI / 4
  scene.add(bond1)

  const bond2 = new THREE.Mesh(bond1Geometry, bondMaterial)
  bond2.position.set(-0.4, 0.3, 0)
  bond2.rotation.z = Math.PI / 4
  scene.add(bond2)
}

function animate() {
  animationId = requestAnimationFrame(animate)
  controls.update()
  renderer.render(scene, camera)
}

function onResize() {
  if (!viewport.value) return

  camera.aspect = viewport.value.clientWidth / viewport.value.clientHeight
  camera.updateProjectionMatrix()
  renderer.setSize(viewport.value.clientWidth, viewport.value.clientHeight)
}

function resetCamera() {
  camera.position.set(0, 0, 5)
  controls.reset()
}

function searchMolecules() {
  // TODO: Implement search with Elasticsearch
  console.log('Searching for:', searchQuery.value)
}

function loadMolecule() {
  // TODO: Load molecule from database and render
  console.log('Loading molecule...')
}

onMounted(() => {
  initThree()
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  window.removeEventListener('resize', onResize)
  if (renderer) {
    renderer.dispose()
  }
})
</script>

<style scoped>
.visualizer {
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-columns: 300px 1fr;
  grid-template-rows: 1fr auto;
  gap: 0;
}

.controls {
  grid-column: 1;
  grid-row: 1 / 3;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  padding: 2rem;
  overflow-y: auto;
}

.controls h2 {
  margin-bottom: 1.5rem;
  color: var(--accent);
}

.control-group {
  margin-bottom: 1.5rem;
}

.control-group label {
  display: block;
  margin-bottom: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.control-group input {
  width: 100%;
}

.control-group button {
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
}

.viewport {
  grid-column: 2;
  grid-row: 1;
  position: relative;
  width: 100%;
  height: 100%;
}

.info {
  grid-column: 2;
  grid-row: 2;
  background: var(--bg-secondary);
  border-top: 1px solid var(--border);
  padding: 1rem 2rem;
}

.info h3 {
  margin-bottom: 0.5rem;
  color: var(--accent);
}

.info p {
  color: var(--text-secondary);
  font-size: 0.9rem;
}
</style>
