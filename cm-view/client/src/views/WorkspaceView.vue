<template>
  <div class="workspace">
    <!-- Left Sidebar: Environments -->
    <div class="env-sidebar" :class="{ collapsed: !sidebarOpen }">
      <div class="sidebar-header">
        <span v-if="sidebarOpen">Environments</span>
        <button @click="sidebarOpen = !sidebarOpen" class="toggle-btn">
          {{ sidebarOpen ? '◀' : '▶' }}
        </button>
      </div>

      <div class="sidebar-content" v-if="sidebarOpen">
        <!-- Current Environment -->
        <div class="current-env">
          <label>Active:</label>
          <select v-model="selectedEnvironment" @change="saveNotebook">
            <option v-for="env in environments" :key="env.name" :value="env.name">
              {{ env.name }} ({{ env.pythonVersion }})
            </option>
          </select>
        </div>

        <!-- Environment List -->
        <div class="env-list">
          <div
            v-for="env in environments"
            :key="env.name"
            class="env-item"
            :class="{ active: env.name === selectedEnvironment }"
            @click="selectedEnvironment = env.name"
          >
            <div class="env-info">
              <span class="env-name">{{ env.name }}</span>
              <span class="env-python">Python {{ env.pythonVersion }}</span>
            </div>
            <div class="env-meta">
              <span class="pkg-count">{{ env.packageCount }} packages</span>
              <button
                v-if="!env.isBase && env.name !== 'chemcomp'"
                @click.stop="deleteEnvironment(env.name)"
                class="delete-env-btn"
                title="Delete environment"
              >&times;</button>
            </div>
          </div>
        </div>

        <!-- Create New Environment -->
        <div class="new-env-section">
          <button @click="showCreateDialog = true" class="create-env-btn">
            + New Environment
          </button>
        </div>

        <!-- Refresh Button -->
        <button @click="loadEnvironments" class="refresh-btn">
          Refresh
        </button>
      </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
      <!-- Top Panel: Visualization -->
      <div class="visualizer-panel" :style="{ height: visualizerHeight + 'px' }">
        <div class="viewport" ref="viewport"></div>

        <div class="viz-overlay">
          <div class="viz-controls">
            <button @click="resetCamera" title="Reset camera">Reset</button>
            <button @click="clearScene" title="Clear scene">Clear</button>
            <button @click="toggleBoxVisible" title="Toggle unit box">
              {{ boxVisible ? 'Hide Box' : 'Show Box' }}
            </button>
          </div>
          <div class="molecule-info" v-if="currentMolecule">
            <span class="molecule-name">{{ currentMolecule.name }}</span>
            <span class="molecule-formula">{{ currentMolecule.formula }}</span>
          </div>
          <div class="box-info">
            <span>Unit Box: {{ boxSize.toFixed(1) }} nm</span>
          </div>
        </div>
      </div>

      <!-- Resize Handle (Horizontal) -->
      <div class="resize-handle-h" @mousedown="startResize"></div>

      <!-- Bottom Panel: Code Cells -->
      <div class="notebook-panel">
        <div class="panel-header">
          <input
            v-model="notebookName"
            @blur="saveNotebook"
            class="notebook-title"
            placeholder="Untitled Notebook"
          />
          <div class="header-right">
            <span class="env-badge" :title="'Using ' + selectedEnvironment">
              {{ selectedEnvironment }}
            </span>
            <button @click="addCell">+ Cell</button>
          </div>
        </div>

        <div class="cells">
          <div
            v-for="(cell, index) in cells"
            :key="cell.id"
            class="cell"
          >
            <div class="cell-toolbar">
              <span class="cell-number">[{{ index + 1 }}]</span>
              <button @click="executeCell(index)" class="run-btn" title="Run cell (Ctrl+Enter)">
                <span v-if="cell.status === 'running'" class="spinner"></span>
                <span v-else>&#9654;</span>
              </button>
              <button @click="deleteCell(index)" class="delete-btn" title="Delete cell">&times;</button>
            </div>

            <div class="cell-input">
              <textarea
                v-model="cell.content"
                @blur="saveNotebook"
                @keydown.ctrl.enter="executeCell(index)"
                placeholder="# Enter Python code... (Ctrl+Enter to run)"
                rows="3"
              />
            </div>

            <div class="cell-output" v-if="cell.output || cell.status === 'running'">
              <div v-if="cell.status === 'running'" class="running-indicator">
                Running...
              </div>
              <pre v-else :class="{ error: cell.status === 'error' }">{{ cell.output }}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Create Environment Dialog -->
    <div class="dialog-overlay" v-if="showCreateDialog" @click.self="showCreateDialog = false">
      <div class="dialog">
        <h3>Create New Environment</h3>

        <div class="form-group">
          <label>Environment Name:</label>
          <input
            v-model="newEnvName"
            placeholder="my-env"
            @keydown.enter="createEnvironment"
          />
        </div>

        <div class="form-group">
          <label>Python Version:</label>
          <select v-model="newEnvPython">
            <option v-for="ver in pythonVersions" :key="ver" :value="ver">
              Python {{ ver }}
            </option>
          </select>
        </div>

        <div class="form-group">
          <label>Initial Packages (optional):</label>
          <input
            v-model="newEnvPackages"
            placeholder="numpy pandas scipy"
          />
          <span class="hint">Space-separated package names</span>
        </div>

        <div class="dialog-actions">
          <button @click="showCreateDialog = false" class="cancel-btn">Cancel</button>
          <button @click="createEnvironment" class="create-btn" :disabled="!newEnvName">
            Create
          </button>
        </div>

        <div class="dialog-status" v-if="createStatus">
          {{ createStatus }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

const route = useRoute()
const viewport = ref(null)

// Sidebar state
const sidebarOpen = ref(true)

// Environment state
const environments = ref([])
const selectedEnvironment = ref('chemcomp')
const pythonVersions = ref(['3.12', '3.11', '3.10', '3.9', '3.8'])

// Create environment dialog
const showCreateDialog = ref(false)
const newEnvName = ref('')
const newEnvPython = ref('3.12')
const newEnvPackages = ref('')
const createStatus = ref('')

// Notebook state
const notebookId = ref(null)
const notebookName = ref('Untitled Notebook')
const cells = ref([])

// Visualizer state
const currentMolecule = ref(null)
const visualizerHeight = ref(400)
const boxVisible = ref(true)
const boxSize = ref(10.0) // nanometers

// Three.js objects
let scene, camera, renderer, controls
let animationId = null
let moleculeGroup = null
let unitBox = null
let unitBoxEdges = null

// Resize handling
let isResizing = false

// ================== Environment Functions ==================

async function loadEnvironments() {
  try {
    const response = await axios.get('/api/environments')
    environments.value = response.data.environments || []
  } catch (error) {
    console.error('Error loading environments:', error)
    // Fallback
    environments.value = [
      { name: 'base', pythonVersion: '3.12', packageCount: 0, isBase: true },
      { name: 'chemcomp', pythonVersion: '3.12', packageCount: 0, isBase: false }
    ]
  }
}

async function loadPythonVersions() {
  try {
    const response = await axios.get('/api/environments/python-versions')
    pythonVersions.value = response.data.versions || ['3.12', '3.11', '3.10', '3.9', '3.8']
  } catch (error) {
    console.error('Error loading Python versions:', error)
  }
}

async function createEnvironment() {
  if (!newEnvName.value) return

  createStatus.value = 'Creating environment...'

  try {
    const packages = newEnvPackages.value.trim().split(/\s+/).filter(p => p)
    await axios.post('/api/environments', {
      name: newEnvName.value,
      pythonVersion: newEnvPython.value,
      packages
    })

    createStatus.value = `Creating '${newEnvName.value}'... This may take a few minutes.`

    // Poll for environment to appear
    setTimeout(async () => {
      await loadEnvironments()
      if (environments.value.some(e => e.name === newEnvName.value)) {
        selectedEnvironment.value = newEnvName.value
        showCreateDialog.value = false
        createStatus.value = ''
        newEnvName.value = ''
        newEnvPackages.value = ''
      }
    }, 5000)

  } catch (error) {
    createStatus.value = `Error: ${error.response?.data?.error || error.message}`
  }
}

async function deleteEnvironment(name) {
  if (!confirm(`Delete environment '${name}'?`)) return

  try {
    await axios.delete(`/api/environments/${name}`)
    await loadEnvironments()
    if (selectedEnvironment.value === name) {
      selectedEnvironment.value = 'chemcomp'
    }
  } catch (error) {
    console.error('Error deleting environment:', error)
    alert(`Failed to delete: ${error.response?.data?.error || error.message}`)
  }
}

// ================== Notebook Functions ==================

async function loadNotebook() {
  if (!route.params.id) {
    addCell()
    return
  }

  notebookId.value = route.params.id

  try {
    const response = await axios.get(`/api/notebooks/${route.params.id}`)
    notebookName.value = response.data.name
    cells.value = JSON.parse(response.data.cells || '[]')

    if (cells.value.length === 0) {
      addCell()
    }
  } catch (error) {
    console.error('Error loading notebook:', error)
    addCell()
  }
}

async function saveNotebook() {
  if (!notebookId.value) return

  try {
    await axios.put(`/api/notebooks/${notebookId.value}`, {
      name: notebookName.value,
      cells: cells.value
    })
  } catch (error) {
    console.error('Error saving notebook:', error)
  }
}

function addCell() {
  cells.value.push({
    id: Date.now(),
    type: 'code',
    content: '',
    output: null,
    status: null
  })
  saveNotebook()
}

function deleteCell(index) {
  cells.value.splice(index, 1)
  if (cells.value.length === 0) {
    addCell()
  }
  saveNotebook()
}

async function executeCell(index) {
  const cell = cells.value[index]
  cell.status = 'running'
  cell.output = null

  try {
    const response = await axios.post('/api/compute', {
      type: 'execute',
      params: {
        code: cell.content,
        environment: selectedEnvironment.value
      }
    })

    cell.jobId = response.data.jobId
    pollJobStatus(cell)
  } catch (error) {
    cell.output = `Error: ${error.message}`
    cell.status = 'error'
  }
}

async function pollJobStatus(cell) {
  const interval = setInterval(async () => {
    try {
      const response = await axios.get(`/api/compute/${cell.jobId}`)
      const job = response.data

      if (job.status === 'completed') {
        clearInterval(interval)
        handleJobResult(cell, job.result)
      } else if (job.status === 'failed') {
        cell.output = `Error: ${job.result?.error || 'Unknown error'}`
        cell.status = 'error'
        clearInterval(interval)
      }
    } catch (error) {
      cell.output = `Error: ${error.message}`
      cell.status = 'error'
      clearInterval(interval)
    }
  }, 1000)
}

function handleJobResult(cell, result) {
  cell.status = 'completed'

  try {
    const parsed = typeof result === 'string' ? JSON.parse(result) : result

    // Check if result contains error
    if (parsed.error) {
      cell.status = 'error'
      cell.output = parsed.stderr || parsed.error
      return
    }

    // Check if result contains visualization data
    if (parsed.visualization) {
      renderMolecule(parsed.visualization)
      cell.output = parsed.output || 'Visualization updated'
    } else if (parsed.atoms || parsed.positions) {
      renderMolecule(parsed)
      cell.output = 'Molecule rendered'
    } else if (parsed.output !== undefined) {
      cell.output = parsed.output + (parsed.stderr ? '\n' + parsed.stderr : '')
    } else {
      cell.output = JSON.stringify(parsed, null, 2)
    }
  } catch {
    cell.output = String(result)
  }
}

// ================== Visualizer Functions ==================

function initThree() {
  if (!viewport.value) return

  // Scene
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0a0a12)

  // Camera
  camera = new THREE.PerspectiveCamera(
    60,
    viewport.value.clientWidth / viewport.value.clientHeight,
    0.1,
    1000
  )
  camera.position.set(12, 8, 12)

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setSize(viewport.value.clientWidth, viewport.value.clientHeight)
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setClearColor(0x0a0a12, 1)
  viewport.value.appendChild(renderer.domElement)

  // Controls
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05
  controls.target.set(0, 0, 0)

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
  scene.add(ambientLight)

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight.position.set(10, 15, 10)
  scene.add(directionalLight)

  const backLight = new THREE.DirectionalLight(0x4488ff, 0.3)
  backLight.position.set(-10, -5, -10)
  scene.add(backLight)

  // Create unit box
  createUnitBox()

  // Molecule group (inside the box)
  moleculeGroup = new THREE.Group()
  scene.add(moleculeGroup)

  // Add sample molecule
  addSampleMolecule()

  // Animation loop
  animate()

  // Handle resize
  window.addEventListener('resize', onWindowResize)
}

function createUnitBox() {
  const size = boxSize.value

  // Translucent box faces
  const boxGeometry = new THREE.BoxGeometry(size, size, size)
  const boxMaterial = new THREE.MeshPhongMaterial({
    color: 0x00d4ff,
    transparent: true,
    opacity: 0.08,
    side: THREE.DoubleSide,
    depthWrite: false
  })
  unitBox = new THREE.Mesh(boxGeometry, boxMaterial)
  scene.add(unitBox)

  // Box edges (wireframe)
  const edgesGeometry = new THREE.EdgesGeometry(boxGeometry)
  const edgesMaterial = new THREE.LineBasicMaterial({
    color: 0x00d4ff,
    transparent: true,
    opacity: 0.6
  })
  unitBoxEdges = new THREE.LineSegments(edgesGeometry, edgesMaterial)
  scene.add(unitBoxEdges)

  // Corner markers
  const cornerSize = 0.15
  const cornerMaterial = new THREE.MeshBasicMaterial({
    color: 0x00d4ff,
    transparent: true,
    opacity: 0.8
  })
  const cornerGeometry = new THREE.SphereGeometry(cornerSize, 8, 8)

  const halfSize = size / 2
  const corners = [
    [-halfSize, -halfSize, -halfSize],
    [-halfSize, -halfSize, halfSize],
    [-halfSize, halfSize, -halfSize],
    [-halfSize, halfSize, halfSize],
    [halfSize, -halfSize, -halfSize],
    [halfSize, -halfSize, halfSize],
    [halfSize, halfSize, -halfSize],
    [halfSize, halfSize, halfSize]
  ]

  corners.forEach(pos => {
    const corner = new THREE.Mesh(cornerGeometry, cornerMaterial)
    corner.position.set(...pos)
    unitBox.add(corner)
  })

  // Axis indicators at origin
  const axisLength = size * 0.15
  const axisWidth = 0.03

  // X axis (red)
  const xAxis = new THREE.Mesh(
    new THREE.CylinderGeometry(axisWidth, axisWidth, axisLength, 8),
    new THREE.MeshBasicMaterial({ color: 0xff4444 })
  )
  xAxis.rotation.z = -Math.PI / 2
  xAxis.position.set(axisLength / 2 - halfSize, -halfSize, -halfSize)
  unitBox.add(xAxis)

  // Y axis (green)
  const yAxis = new THREE.Mesh(
    new THREE.CylinderGeometry(axisWidth, axisWidth, axisLength, 8),
    new THREE.MeshBasicMaterial({ color: 0x44ff44 })
  )
  yAxis.position.set(-halfSize, axisLength / 2 - halfSize, -halfSize)
  unitBox.add(yAxis)

  // Z axis (blue)
  const zAxis = new THREE.Mesh(
    new THREE.CylinderGeometry(axisWidth, axisWidth, axisLength, 8),
    new THREE.MeshBasicMaterial({ color: 0x4444ff })
  )
  zAxis.rotation.x = Math.PI / 2
  zAxis.position.set(-halfSize, -halfSize, axisLength / 2 - halfSize)
  unitBox.add(zAxis)
}

function toggleBoxVisible() {
  boxVisible.value = !boxVisible.value
  if (unitBox) unitBox.visible = boxVisible.value
  if (unitBoxEdges) unitBoxEdges.visible = boxVisible.value
}

const atomColors = {
  H: 0xffffff,
  C: 0x404040,
  N: 0x3050f8,
  O: 0xff2020,
  S: 0xffff30,
  P: 0xff8000,
  F: 0x90e050,
  Cl: 0x1ff01f,
  Br: 0xa62929,
  I: 0x940094,
  Fe: 0xe06633,
  Ca: 0x3dff00,
  Mg: 0x8aff00,
  Zn: 0x7d80b0,
  default: 0xff69b4
}

const atomRadii = {
  H: 0.12,
  C: 0.17,
  N: 0.155,
  O: 0.152,
  S: 0.18,
  P: 0.18,
  Fe: 0.14,
  Ca: 0.18,
  Mg: 0.15,
  Zn: 0.14,
  default: 0.15
}

function addSampleMolecule() {
  // Sample ATP molecule (simplified)
  const atoms = [
    { element: 'N', position: [0, 0, 0] },
    { element: 'C', position: [0.5, 0.8, 0] },
    { element: 'N', position: [1.5, 0.8, 0] },
    { element: 'C', position: [2.0, 0, 0] },
    { element: 'C', position: [1.5, -0.8, 0] },
    { element: 'N', position: [0.5, -0.8, 0] },
    { element: 'O', position: [-1.0, 0, 0] },
    { element: 'C', position: [-1.5, 0.8, 0] },
    { element: 'C', position: [-2.5, 0.5, 0] },
    { element: 'C', position: [-2.5, -0.5, 0] },
    { element: 'C', position: [-1.5, -0.8, 0] },
    { element: 'P', position: [-3.5, 0, 0] },
    { element: 'O', position: [-3.5, 0.8, 0.5] },
    { element: 'O', position: [-3.5, -0.8, 0.5] },
    { element: 'P', position: [-4.5, 0, 0] },
    { element: 'O', position: [-4.5, 0.8, -0.5] },
    { element: 'P', position: [-5.5, 0, 0] },
    { element: 'O', position: [-5.5, 0.8, 0.5] },
    { element: 'O', position: [-6.2, 0, 0] }
  ]

  const bonds = [
    { start: 0, end: 1 }, { start: 1, end: 2 }, { start: 2, end: 3 },
    { start: 3, end: 4 }, { start: 4, end: 5 }, { start: 5, end: 0 },
    { start: 0, end: 6 }, { start: 6, end: 7 }, { start: 7, end: 8 },
    { start: 8, end: 9 }, { start: 9, end: 10 }, { start: 10, end: 6 },
    { start: 8, end: 11 }, { start: 11, end: 12 }, { start: 11, end: 13 },
    { start: 11, end: 14 }, { start: 14, end: 15 }, { start: 14, end: 16 },
    { start: 16, end: 17 }, { start: 16, end: 18 }
  ]

  renderMolecule({ atoms, bonds, name: 'ATP', formula: 'C10H16N5O13P3' })
}

function renderMolecule(data) {
  while (moleculeGroup.children.length > 0) {
    const child = moleculeGroup.children[0]
    moleculeGroup.remove(child)
    if (child.geometry) child.geometry.dispose()
    if (child.material) child.material.dispose()
  }

  if (!data.atoms) return

  const scaleFactor = 0.8

  data.atoms.forEach((atom, index) => {
    const element = atom.element || atom.type || 'C'
    const pos = atom.position || atom.pos || [0, 0, 0]

    const radius = (atomRadii[element] || atomRadii.default) * scaleFactor
    const color = atomColors[element] || atomColors.default

    const geometry = new THREE.SphereGeometry(radius, 24, 24)
    const material = new THREE.MeshPhongMaterial({
      color: color,
      shininess: 80,
      specular: 0x444444
    })

    const sphere = new THREE.Mesh(geometry, material)
    sphere.position.set(pos[0] * scaleFactor, pos[1] * scaleFactor, pos[2] * scaleFactor)
    sphere.userData = { index, element, ...atom }
    moleculeGroup.add(sphere)
  })

  if (data.bonds) {
    const bondMaterial = new THREE.MeshPhongMaterial({
      color: 0x888888,
      shininess: 30
    })

    data.bonds.forEach(bond => {
      const startAtom = data.atoms[bond.start]
      const endAtom = data.atoms[bond.end]

      if (!startAtom || !endAtom) return

      const startPos = new THREE.Vector3(...(startAtom.position || startAtom.pos || [0, 0, 0])).multiplyScalar(scaleFactor)
      const endPos = new THREE.Vector3(...(endAtom.position || endAtom.pos || [0, 0, 0])).multiplyScalar(scaleFactor)

      const direction = new THREE.Vector3().subVectors(endPos, startPos)
      const length = direction.length()

      const geometry = new THREE.CylinderGeometry(0.04, 0.04, length, 8)
      const cylinder = new THREE.Mesh(geometry, bondMaterial)

      cylinder.position.copy(startPos).add(endPos).multiplyScalar(0.5)
      cylinder.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        direction.normalize()
      )

      moleculeGroup.add(cylinder)
    })
  }

  if (data.name || data.formula) {
    currentMolecule.value = {
      name: data.name || 'Unknown',
      formula: data.formula || ''
    }
  }
}

function animate() {
  animationId = requestAnimationFrame(animate)
  controls.update()
  renderer.render(scene, camera)
}

function onWindowResize() {
  if (!viewport.value || !camera || !renderer) return

  camera.aspect = viewport.value.clientWidth / viewport.value.clientHeight
  camera.updateProjectionMatrix()
  renderer.setSize(viewport.value.clientWidth, viewport.value.clientHeight)
}

function resetCamera() {
  camera.position.set(12, 8, 12)
  controls.target.set(0, 0, 0)
  controls.update()
}

function clearScene() {
  while (moleculeGroup.children.length > 0) {
    const child = moleculeGroup.children[0]
    moleculeGroup.remove(child)
    if (child.geometry) child.geometry.dispose()
    if (child.material) child.material.dispose()
  }
  currentMolecule.value = null
}

// ================== Resize Panel Functions ==================

function startResize(e) {
  isResizing = true
  document.addEventListener('mousemove', doResize)
  document.addEventListener('mouseup', stopResize)
}

function doResize(e) {
  if (!isResizing) return

  const container = document.querySelector('.main-content')
  const containerRect = container.getBoundingClientRect()
  const newHeight = e.clientY - containerRect.top

  if (newHeight >= 200 && newHeight <= containerRect.height - 150) {
    visualizerHeight.value = newHeight
    nextTick(() => {
      onWindowResize()
    })
  }
}

function stopResize() {
  isResizing = false
  document.removeEventListener('mousemove', doResize)
  document.removeEventListener('mouseup', stopResize)
}

// ================== Lifecycle ==================

onMounted(async () => {
  await Promise.all([
    loadNotebook(),
    loadEnvironments(),
    loadPythonVersions()
  ])
  await nextTick()
  initThree()
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  window.removeEventListener('resize', onWindowResize)
  if (renderer) {
    renderer.dispose()
  }
})
</script>

<style scoped>
.workspace {
  width: 100%;
  height: 100%;
  display: flex;
  overflow: hidden;
  background: var(--bg-primary);
}

/* Environment Sidebar */
.env-sidebar {
  width: 220px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  transition: width 0.2s;
}

.env-sidebar.collapsed {
  width: 40px;
}

.sidebar-header {
  padding: 0.75rem;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--accent);
}

.toggle-btn {
  width: 24px;
  height: 24px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.7rem;
}

.toggle-btn:hover {
  color: var(--accent);
}

.sidebar-content {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.current-env {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.current-env label {
  font-size: 0.7rem;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.current-env select {
  padding: 0.4rem;
  font-size: 0.8rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
}

.env-list {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.env-item {
  padding: 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.env-item:hover {
  border-color: var(--accent);
}

.env-item.active {
  border-color: var(--accent);
  background: rgba(0, 212, 255, 0.1);
}

.env-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.25rem;
}

.env-name {
  font-weight: 600;
  font-size: 0.8rem;
  color: var(--text-primary);
}

.env-python {
  font-size: 0.7rem;
  color: var(--accent);
}

.env-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pkg-count {
  font-size: 0.7rem;
  color: var(--text-secondary);
}

.delete-env-btn {
  width: 18px;
  height: 18px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.9rem;
  line-height: 1;
  border-radius: 2px;
}

.delete-env-btn:hover {
  background: var(--error);
  color: white;
}

.new-env-section {
  margin-top: auto;
}

.create-env-btn {
  width: 100%;
  padding: 0.5rem;
  font-size: 0.8rem;
  background: var(--accent);
  color: var(--bg-primary);
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.create-env-btn:hover {
  opacity: 0.9;
}

.refresh-btn {
  width: 100%;
  padding: 0.4rem;
  font-size: 0.75rem;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  border-radius: 4px;
  cursor: pointer;
}

.refresh-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}

/* Main Content */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Visualizer Panel (Top) */
.visualizer-panel {
  position: relative;
  min-height: 200px;
  background: #0a0a12;
}

.viewport {
  width: 100%;
  height: 100%;
}

.viz-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  padding: 0.75rem 1rem;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  pointer-events: none;
  background: linear-gradient(to bottom, rgba(0,0,0,0.5) 0%, transparent 100%);
}

.viz-controls {
  display: flex;
  gap: 0.5rem;
  pointer-events: auto;
}

.viz-controls button {
  font-size: 0.75rem;
  padding: 0.35rem 0.6rem;
  background: rgba(0, 212, 255, 0.2);
  border: 1px solid rgba(0, 212, 255, 0.4);
  color: #00d4ff;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.viz-controls button:hover {
  background: rgba(0, 212, 255, 0.3);
  border-color: #00d4ff;
}

.molecule-info {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.25rem;
}

.molecule-name {
  color: #00d4ff;
  font-weight: 600;
  font-size: 0.9rem;
}

.molecule-formula {
  color: rgba(255, 255, 255, 0.7);
  font-family: monospace;
  font-size: 0.8rem;
}

.box-info {
  position: absolute;
  bottom: 0.75rem;
  left: 1rem;
  color: rgba(0, 212, 255, 0.6);
  font-size: 0.75rem;
  font-family: monospace;
}

/* Resize Handle (Horizontal) */
.resize-handle-h {
  height: 6px;
  background: var(--border);
  cursor: row-resize;
  transition: background 0.2s;
}

.resize-handle-h:hover {
  background: var(--accent);
}

/* Notebook Panel (Bottom) */
.notebook-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 150px;
  overflow: hidden;
}

.panel-header {
  background: var(--bg-secondary);
  padding: 0.5rem 1rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.notebook-title {
  font-size: 1rem;
  font-weight: 600;
  background: transparent;
  border: none;
  color: var(--text-primary);
  flex: 1;
  margin-right: 1rem;
}

.notebook-title:focus {
  outline: none;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.env-badge {
  font-size: 0.7rem;
  padding: 0.2rem 0.5rem;
  background: rgba(0, 212, 255, 0.2);
  border: 1px solid rgba(0, 212, 255, 0.4);
  border-radius: 4px;
  color: var(--accent);
}

.panel-header button {
  font-size: 0.8rem;
  padding: 0.3rem 0.6rem;
}

.cells {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem;
}

.cell {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 0.5rem;
  overflow: hidden;
}

.cell:focus-within {
  border-color: var(--accent);
}

.cell-toolbar {
  background: var(--bg-tertiary);
  padding: 0.2rem 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.cell-number {
  color: var(--text-secondary);
  font-size: 0.7rem;
  font-family: monospace;
  min-width: 25px;
}

.run-btn, .delete-btn {
  width: 22px;
  height: 22px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  border-radius: 4px;
}

.run-btn {
  background: var(--success);
  margin-left: auto;
}

.delete-btn {
  background: transparent;
  color: var(--text-secondary);
}

.delete-btn:hover {
  background: var(--error);
  color: white;
}

.spinner {
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.cell-input textarea {
  width: 100%;
  min-height: 60px;
  background: var(--bg-primary);
  border: none;
  color: var(--text-primary);
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.8rem;
  padding: 0.5rem;
  resize: vertical;
  line-height: 1.4;
}

.cell-input textarea:focus {
  outline: none;
}

.cell-output {
  border-top: 1px solid var(--border);
  padding: 0.5rem;
  background: var(--bg-tertiary);
  max-height: 200px;
  overflow-y: auto;
}

.cell-output pre {
  margin: 0;
  color: var(--text-secondary);
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.75rem;
  white-space: pre-wrap;
  word-break: break-word;
}

.cell-output pre.error {
  color: var(--error);
}

.running-indicator {
  color: var(--accent);
  font-size: 0.8rem;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Create Environment Dialog */
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.dialog {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  width: 400px;
  max-width: 90%;
}

.dialog h3 {
  margin: 0 0 1.25rem 0;
  color: var(--accent);
  font-size: 1.1rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.35rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.form-group input:focus,
.form-group select:focus {
  outline: none;
  border-color: var(--accent);
}

.hint {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.7rem;
  color: var(--text-secondary);
}

.dialog-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  margin-top: 1.5rem;
}

.cancel-btn {
  padding: 0.5rem 1rem;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  border-radius: 4px;
  cursor: pointer;
}

.cancel-btn:hover {
  border-color: var(--text-secondary);
}

.create-btn {
  padding: 0.5rem 1rem;
  background: var(--accent);
  border: none;
  color: var(--bg-primary);
  border-radius: 4px;
  cursor: pointer;
  font-weight: 600;
}

.create-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.create-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.dialog-status {
  margin-top: 1rem;
  padding: 0.5rem;
  background: var(--bg-primary);
  border-radius: 4px;
  font-size: 0.8rem;
  color: var(--accent);
}
</style>
