<template>
  <div class="workspace">
    <!-- Left Sidebar with Tabs -->
    <div class="left-sidebar" :class="{ collapsed: !sidebarOpen }">
      <div class="sidebar-header">
        <div class="sidebar-tabs" v-if="sidebarOpen">
          <button
            :class="{ active: sidebarTab === 'files' }"
            @click="sidebarTab = 'files'"
            title="Files"
          >Files</button>
          <button
            :class="{ active: sidebarTab === 'envs' }"
            @click="sidebarTab = 'envs'"
            title="Environments"
          >Envs</button>
        </div>
        <button @click="sidebarOpen = !sidebarOpen" class="toggle-btn">
          {{ sidebarOpen ? '◀' : '▶' }}
        </button>
      </div>

      <!-- File Browser Tab -->
      <div class="sidebar-content" v-if="sidebarOpen && sidebarTab === 'files'">
        <FileBrowser
          :workspace-id="workspaceId"
          @file-open="handleFileOpen"
          @file-select="handleFileSelect"
        />
      </div>

      <!-- Environments Tab -->
      <div class="sidebar-content env-panel" v-if="sidebarOpen && sidebarTab === 'envs'">
        <!-- Sub-tabs for Python/C++ -->
        <div class="env-sub-tabs">
          <button
            :class="{ active: envSubTab === 'python' }"
            @click="envSubTab = 'python'"
          >Python</button>
          <button
            :class="{ active: envSubTab === 'cpp' }"
            @click="envSubTab = 'cpp'"
          >C++</button>
        </div>

        <!-- Python Environments -->
        <template v-if="envSubTab === 'python'">
          <div class="current-env">
            <label>Active:</label>
            <select v-model="selectedEnvironment">
              <option v-for="env in environments" :key="env.name" :value="env.name">
                {{ env.name }} ({{ env.pythonVersion }})
              </option>
            </select>
          </div>

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

          <div class="new-env-section">
            <button @click="showCreateDialog = true" class="create-env-btn">
              + New Python Env
            </button>
          </div>

          <button @click="loadEnvironments" class="refresh-btn">Refresh</button>
        </template>

        <!-- C++ Environments -->
        <template v-if="envSubTab === 'cpp'">
          <!-- C++ System Libraries -->
          <div class="env-section-header">System Libraries</div>
          <div class="current-env" v-if="cppEnvironments.length > 0">
            <label>Active:</label>
            <select v-model="selectedCppEnvironment">
              <option value="">None</option>
              <option v-for="env in cppEnvironments" :key="env.name" :value="env.name">
                {{ env.name }}
              </option>
            </select>
          </div>

          <div class="env-list">
            <div
              v-for="env in cppEnvironments"
              :key="env.name"
              class="env-item"
              :class="{ active: env.name === selectedCppEnvironment }"
              @click="selectedCppEnvironment = env.name"
            >
              <div class="env-info">
                <span class="env-name">{{ env.name }}</span>
                <span class="env-lang">C++</span>
              </div>
              <div class="env-meta">
                <span class="pkg-count">{{ env.packages?.length || 0 }} packages</span>
                <button
                  @click.stop="deleteCppEnvironment(env.name)"
                  class="delete-env-btn"
                  title="Delete environment"
                >&times;</button>
              </div>
            </div>
          </div>

          <div class="new-env-section">
            <button @click="showCppDialog = true" class="create-env-btn">
              + New C++ Env
            </button>
          </div>

          <!-- Vendor Libraries -->
          <div class="env-section-header">Vendor Libraries</div>
          <div class="current-env" v-if="vendorEnvironments.length > 0">
            <label>Active:</label>
            <select v-model="selectedVendorEnvironment">
              <option value="">None</option>
              <option v-for="env in vendorEnvironments" :key="env.name" :value="env.name">
                {{ env.name }}
              </option>
            </select>
          </div>

          <div class="env-list">
            <div
              v-for="env in vendorEnvironments"
              :key="env.name"
              class="env-item vendor"
              :class="{ active: env.name === selectedVendorEnvironment }"
              @click="selectedVendorEnvironment = env.name"
            >
              <div class="env-info">
                <span class="env-name">{{ env.name }}</span>
                <span class="env-lang">Vendor</span>
              </div>
              <div class="env-meta">
                <span class="pkg-count">{{ env.installations?.length || 0 }} installs</span>
                <button
                  @click.stop="deleteVendorEnvironment(env.name)"
                  class="delete-env-btn"
                  title="Delete environment"
                >&times;</button>
              </div>
            </div>
          </div>

          <div class="new-env-section">
            <button @click="showVendorDialog = true" class="create-env-btn vendor-btn">
              + Build from Source
            </button>
          </div>

          <button @click="loadCppEnvironments(); loadVendorEnvironments()" class="refresh-btn">Refresh</button>
        </template>
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
        <!-- Tab Bar -->
        <div class="tab-bar" v-if="openTabs.length > 0">
          <div
            v-for="(tab, index) in openTabs"
            :key="tab.path"
            class="tab"
            :class="{ active: index === activeTabIndex, dirty: tab.isDirty }"
            @click="switchToTab(index)"
            :title="tab.path"
          >
            <span class="tab-icon">{{ getFileIcon(tab.name) }}</span>
            <span class="tab-name">{{ tab.name }}</span>
            <span v-if="tab.isDirty" class="tab-dirty-indicator">●</span>
            <button class="tab-close" @click.stop="closeTab(index)" title="Close">×</button>
          </div>
        </div>

        <div class="panel-header">
          <div class="file-info" v-if="currentFile">
            <span class="file-name" :class="{ modified: hasUnsavedChanges }">
              {{ currentFile.name }}{{ hasUnsavedChanges ? ' •' : '' }}
            </span>
            <span class="file-path">{{ currentFile.path }}</span>
          </div>
          <div class="file-info" v-else>
            <span class="file-name empty">No file open</span>
            <span class="file-path">Open a file from the sidebar</span>
          </div>
          <div class="header-right">
            <!-- Python environment selector -->
            <select
              v-if="currentFile && currentFile.language === 'python'"
              v-model="selectedEnvironment"
              class="file-env-select"
              title="Python environment"
            >
              <option v-for="env in environmentNames" :key="env" :value="env">{{ env }}</option>
            </select>
            <!-- C++ environment selectors -->
            <select
              v-if="currentFile && currentFile.language === 'cpp'"
              v-model="selectedCppEnvironment"
              class="file-env-select cpp"
              title="C++ environment"
            >
              <option value="">System</option>
              <option v-for="env in cppEnvironmentNames" :key="env" :value="env">{{ env }}</option>
            </select>
            <select
              v-if="currentFile && currentFile.language === 'cpp'"
              v-model="selectedVendorEnvironment"
              class="file-env-select vendor"
              title="Vendor libraries"
            >
              <option value="">No vendor</option>
              <option v-for="env in vendorEnvironmentNames" :key="env" :value="env">{{ env }}</option>
            </select>
            <!-- C++ compiler selector -->
            <select
              v-if="currentFile && currentFile.language === 'cpp'"
              v-model="selectedCompiler"
              class="file-env-select compiler"
              title="C++ compiler"
            >
              <option v-for="c in compilers" :key="c.name" :value="c.name" :disabled="!c.available">
                {{ c.name }}{{ c.version ? ` (${c.version})` : '' }}
              </option>
            </select>
            <!-- C++ standard selector -->
            <select
              v-if="currentFile && currentFile.language === 'cpp'"
              v-model="selectedCppStandard"
              class="file-env-select std"
              title="C++ standard"
            >
              <option value="c++11">C++11</option>
              <option value="c++14">C++14</option>
              <option value="c++17">C++17</option>
              <option value="c++20">C++20</option>
              <option value="c++23">C++23</option>
            </select>
            <button v-if="currentFile" @click="saveFile" class="save-btn" :disabled="!hasUnsavedChanges" title="Save file (Ctrl+S)">
              Save
            </button>
            <button v-if="currentFile && currentUseCells" @click="addCell" title="Add new cell">+ Cell</button>
            <button v-if="currentFile" @click="closeFile" class="close-btn" title="Close file">×</button>
          </div>
        </div>

        <div class="cells" v-if="currentFile && cells.length > 0">
          <CodeCell
            v-for="(cell, index) in cells"
            :key="cell.id"
            :cell="cell"
            :index="index"
            :language="currentFile.language"
            @update="updateCell(index, $event)"
            @run="executeCell(index)"
            @delete="deleteCell(index)"
          />
        </div>
        <div class="no-file-message" v-else-if="!currentFile">
          <p>Open a file from the Files sidebar to start editing</p>
          <p class="hint">Use <code># %%</code> in <code>.cell.py</code> / <code>.sh</code> or <code>// %%</code> in <code>.cell.cpp</code> for cell boundaries</p>
          <p class="hint">Regular <code>.py</code> and <code>.cpp/.c/.h/.hpp</code> files are single execution units</p>
          <p class="hint">Import workspace modules with <code>from workspace import module</code></p>
        </div>
      </div>
    </div>

    <!-- Create Environment Dialog -->
    <div class="dialog-overlay" v-if="showCreateDialog" @click.self="showCreateDialog = false">
      <div class="dialog" :class="{ expanded: dialogExpanded }">
        <div class="dialog-header">
          <h3>Create New Environment</h3>
          <button
            v-if="createLogs.length > 0"
            @click="dialogExpanded = !dialogExpanded"
            class="expand-btn"
            :title="dialogExpanded ? 'Collapse' : 'Expand to see full log'"
          >
            {{ dialogExpanded ? '⊟' : '⊞' }}
          </button>
        </div>

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
          <button @click="showCreateDialog = false" class="cancel-btn" :disabled="isCreating">Cancel</button>
          <button @click="createEnvironment" class="create-btn" :disabled="!newEnvName || isCreating">
            {{ isCreating ? 'Creating...' : 'Create' }}
          </button>
        </div>

        <div class="create-log" v-if="createLogs.length > 0">
          <div class="log-header">{{ createStatus }}</div>
          <div class="log-output" ref="logOutput">
            <div v-for="(log, i) in createLogs" :key="i" :class="['log-line', log.type]">
              {{ log.message }}
            </div>
          </div>
        </div>
        <div class="dialog-status" v-else-if="createStatus">
          {{ createStatus }}
        </div>
      </div>
    </div>

    <!-- C++ Environment Dialog -->
    <CppEnvironmentDialog
      v-if="showCppDialog"
      @close="showCppDialog = false"
      @created="handleCppEnvCreated"
    />

    <!-- Vendor Environment Dialog -->
    <VendorEnvironmentDialog
      v-if="showVendorDialog"
      @close="showVendorDialog = false"
      @created="handleVendorEnvCreated"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import CodeCell from '../components/CodeCell.vue'
import FileBrowser from '../components/FileBrowser.vue'
import CppEnvironmentDialog from '../components/CppEnvironmentDialog.vue'
import VendorEnvironmentDialog from '../components/VendorEnvironmentDialog.vue'

const route = useRoute()
const viewport = ref(null)

// Workspace ID from route
const workspaceId = computed(() => route.params.id)

// Helper to build workspace file API URL
function fileApiUrl(filePath = '') {
  const base = `/api/workspaces/${workspaceId.value}/files`
  return filePath ? `${base}/${encodeURIComponent(filePath)}` : base
}

// Sidebar state
const sidebarOpen = ref(true)
const sidebarTab = ref('files')

// Environment state (Python/Conda)
const environments = ref([])
const selectedEnvironment = ref('chemcomp')
const pythonVersions = ref(['3.12', '3.11', '3.10', '3.9', '3.8'])

// C++ Environment state
const cppEnvironments = ref([])
const vendorEnvironments = ref([])
const selectedCppEnvironment = ref('')
const selectedVendorEnvironment = ref('')
const envSubTab = ref('python')  // 'python' or 'cpp'
const compilers = ref([
  { name: 'g++', version: null, available: true },
  { name: 'clang++', version: null, available: true }
])
const selectedCompiler = ref('clang++')
const selectedCppStandard = ref('c++23')

// Computed environment names for cell selector
const environmentNames = computed(() => environments.value.map(e => e.name))
const cppEnvironmentNames = computed(() => cppEnvironments.value.map(e => e.name))
const vendorEnvironmentNames = computed(() => vendorEnvironments.value.map(e => e.name))

// Create environment dialogs
const showCreateDialog = ref(false)
const showCppDialog = ref(false)
const showVendorDialog = ref(false)
const newEnvName = ref('')
const newEnvPython = ref('3.12')
const newEnvPackages = ref('')
const createStatus = ref('')
const createLogs = ref([])
const isCreating = ref(false)
const logOutput = ref(null)
const dialogExpanded = ref(false)

// File editor state - now with tabs
const openTabs = ref([])  // Array of { path, name, language, cells, isDirty }
const activeTabIndex = ref(-1)  // Currently active tab index

// Computed properties for current file
const currentFile = computed(() => {
  if (activeTabIndex.value >= 0 && activeTabIndex.value < openTabs.value.length) {
    const tab = openTabs.value[activeTabIndex.value]
    return { path: tab.path, name: tab.name, language: tab.language }
  }
  return null
})

const cells = computed({
  get: () => {
    if (activeTabIndex.value >= 0 && activeTabIndex.value < openTabs.value.length) {
      return openTabs.value[activeTabIndex.value].cells
    }
    return []
  },
  set: (value) => {
    if (activeTabIndex.value >= 0 && activeTabIndex.value < openTabs.value.length) {
      openTabs.value[activeTabIndex.value].cells = value
    }
  }
})

const hasUnsavedChanges = computed({
  get: () => {
    if (activeTabIndex.value >= 0 && activeTabIndex.value < openTabs.value.length) {
      return openTabs.value[activeTabIndex.value].isDirty
    }
    return false
  },
  set: (value) => {
    if (activeTabIndex.value >= 0 && activeTabIndex.value < openTabs.value.length) {
      openTabs.value[activeTabIndex.value].isDirty = value
    }
  }
})

// Whether the current file uses cell mode
const currentUseCells = computed(() => {
  if (activeTabIndex.value >= 0 && activeTabIndex.value < openTabs.value.length) {
    return openTabs.value[activeTabIndex.value].useCells !== false
  }
  return true
})

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

async function loadCompilers() {
  try {
    const response = await axios.get('/api/compilers')
    if (response.data.compilers && response.data.compilers.length > 0) {
      compilers.value = response.data.compilers
    }
  } catch (error) {
    console.error('Error loading compilers:', error)
  }
}

/**
 * Process terminal output for cell, handling carriage returns properly.
 * When \r is used (common for progress bars), replace the current line.
 */
function appendCellOutput(cell, text) {
  if (!text) return

  const currentOutput = cell.output || ''

  // Check if this chunk starts with \r (carriage return) - means "go back to line start"
  if (text.includes('\r') && !text.includes('\n')) {
    // Pure \r update - replace the last line
    const lines = currentOutput.split('\n')
    // Get the text after the last \r
    const newText = text.split('\r').filter(p => p.length > 0).pop() || ''
    if (lines.length > 0) {
      lines[lines.length - 1] = newText
      cell.output = lines.join('\n')
    } else {
      cell.output = newText
    }
  } else if (text.includes('\r')) {
    // Mixed \r and \n - process line by line
    const chunks = text.split('\n')
    let result = currentOutput

    for (const chunk of chunks) {
      if (chunk.includes('\r')) {
        // Take the last part after \r
        const parts = chunk.split('\r')
        const finalPart = parts.filter(p => p.length > 0).pop() || ''
        const lines = result.split('\n')
        if (lines.length > 0) {
          lines[lines.length - 1] = finalPart
          result = lines.join('\n')
        } else {
          result = finalPart
        }
      } else if (chunk) {
        result = result ? result + '\n' + chunk : chunk
      } else if (text.includes('\n')) {
        // Empty chunk from split means there was a newline
        result = result + '\n'
      }
    }
    cell.output = result
  } else {
    // No \r, just append
    cell.output = currentOutput + text
  }

  // Clean up ANSI escape codes
  cell.output = cell.output.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
}

/**
 * Process terminal output handling carriage returns and newlines properly
 * Conda uses \r to update spinners/progress in place
 */
function appendTerminalOutput(text, logType = 'progress') {
  if (!text) return

  // Split by newlines, but also handle carriage returns
  const lines = text.split(/\n/)

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i]

    // Handle carriage return - update the last line instead of adding new one
    if (line.includes('\r')) {
      const parts = line.split('\r')
      // Take the last non-empty part (what should be displayed after \r overwrites)
      line = parts.filter(p => p.length > 0).pop() || ''
    }

    // Skip empty lines unless it's between content
    if (!line && i === lines.length - 1) continue

    // Clean up ANSI escape codes (color codes, cursor movement, etc.)
    line = line.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')

    // If this is a continuation (same line update), update the last log entry
    if (createLogs.value.length > 0 && text.startsWith('\r')) {
      const lastLog = createLogs.value[createLogs.value.length - 1]
      if (lastLog.type === logType && line) {
        lastLog.message = line
        continue
      }
    }

    if (line) {
      createLogs.value.push({ type: logType, message: line })
    }
  }
}

async function createEnvironment() {
  if (!newEnvName.value || isCreating.value) return

  isCreating.value = true
  createStatus.value = 'Starting...'
  createLogs.value = []

  const envName = newEnvName.value
  const packages = newEnvPackages.value.trim().split(/\s+/).filter(p => p)

  try {
    // POST to create environment job
    const response = await axios.post('/api/environments', {
      name: envName,
      pythonVersion: newEnvPython.value,
      packages
    })

    if (response.data.error) {
      createStatus.value = `Error: ${response.data.error}`
      isCreating.value = false
      return
    }

    const { jobId } = response.data
    createStatus.value = response.data.message || `Creating environment '${envName}'...`

    // Subscribe to job output via WebSocket
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      // Subscribe to this job's output
      ws.send(JSON.stringify({
        type: 'subscribe',
        jobId
      }))
    }

    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === 'subscribed') {
          createLogs.value.push({ type: 'progress', message: 'Connected to job stream...' })
        } else if (data.type === 'job_output') {
          // Handle streamed output from job
          const { stream, data: outputData } = data
          const streamLower = stream?.toLowerCase()

          if (streamLower === 'stdout' || streamLower === 'stderr') {
            const logType = streamLower === 'stderr' ? 'error' : 'progress'
            // Process terminal output - handle carriage returns and newlines
            appendTerminalOutput(outputData, logType)
          } else if (streamLower === 'result') {
            createStatus.value = `Environment '${envName}' created successfully!`
            createLogs.value.push({ type: 'success', message: `Environment '${envName}' created successfully!` })

            // Refresh environments and select the new one
            await loadEnvironments()
            selectedEnvironment.value = envName

            // Close dialog after short delay
            setTimeout(() => {
              showCreateDialog.value = false
              createStatus.value = ''
              createLogs.value = []
              newEnvName.value = ''
              newEnvPackages.value = ''
              isCreating.value = false
              dialogExpanded.value = false
            }, 1500)

            ws.close()
          } else if (streamLower === 'error') {
            createStatus.value = `Error: ${outputData}`
            createLogs.value.push({ type: 'error', message: outputData })
            isCreating.value = false
            ws.close()
          } else if (streamLower === 'complete') {
            // Job completed (with exit code)
            if (!createStatus.value.includes('successfully')) {
              createStatus.value = 'Job completed'
            }
            isCreating.value = false
            ws.close()
          }

          // Auto-scroll log output
          await nextTick()
          if (logOutput.value) {
            logOutput.value.scrollTop = logOutput.value.scrollHeight
          }
        } else if (data.type === 'error') {
          createStatus.value = `Error: ${data.error}`
          createLogs.value.push({ type: 'error', message: data.error })
          isCreating.value = false
          ws.close()
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', e)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      createStatus.value = 'WebSocket connection error'
      createLogs.value.push({ type: 'error', message: 'WebSocket connection error' })
      isCreating.value = false
    }

    ws.onclose = () => {
      // Only set isCreating false if it wasn't already handled
      if (isCreating.value && !createStatus.value.includes('successfully')) {
        isCreating.value = false
      }
    }

  } catch (error) {
    createStatus.value = `Error: ${error.response?.data?.error || error.message}`
    createLogs.value.push({ type: 'error', message: error.response?.data?.error || error.message })
    isCreating.value = false
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

// ================== C++ Environment Functions ==================

async function loadCppEnvironments() {
  try {
    const response = await axios.get('/api/cpp-environments')
    cppEnvironments.value = response.data.environments || []
  } catch (error) {
    console.error('Error loading C++ environments:', error)
    cppEnvironments.value = []
  }
}

async function loadVendorEnvironments() {
  try {
    const response = await axios.get('/api/vendor-environments')
    vendorEnvironments.value = response.data.environments || []
  } catch (error) {
    console.error('Error loading vendor environments:', error)
    vendorEnvironments.value = []
  }
}

async function deleteCppEnvironment(name) {
  if (!confirm(`Delete C++ environment '${name}'?`)) return

  try {
    await axios.delete(`/api/cpp-environments/${name}`)
    await loadCppEnvironments()
    if (selectedCppEnvironment.value === name) {
      selectedCppEnvironment.value = ''
    }
  } catch (error) {
    console.error('Error deleting C++ environment:', error)
    alert(`Failed to delete: ${error.response?.data?.error || error.message}`)
  }
}

async function deleteVendorEnvironment(name) {
  if (!confirm(`Delete vendor environment '${name}'? This will remove all installed libraries.`)) return

  try {
    await axios.delete(`/api/vendor-environments/${name}`)
    await loadVendorEnvironments()
    if (selectedVendorEnvironment.value === name) {
      selectedVendorEnvironment.value = ''
    }
  } catch (error) {
    console.error('Error deleting vendor environment:', error)
    alert(`Failed to delete: ${error.response?.data?.error || error.message}`)
  }
}

function handleCppEnvCreated(name) {
  loadCppEnvironments()
  selectedCppEnvironment.value = name
}

function handleVendorEnvCreated(name) {
  loadVendorEnvironments()
  selectedVendorEnvironment.value = name
}

// ================== File Editor Functions ==================

// Cell delimiters by language
const CELL_DELIMITERS = {
  python: /^# %%(.*)$/,
  cpp: /^\/\/ %%(.*)$/,
  bash: /^# %%(.*)$/
}

/**
 * Get the language from file extension
 */
function getLanguageFromExt(filename) {
  const ext = filename.split('.').pop()?.toLowerCase()
  if (ext === 'cpp' || ext === 'c' || ext === 'h' || ext === 'hpp') return 'cpp'
  if (ext === 'sh' || ext === 'bash') return 'bash'
  return 'python'
}

/**
 * Check if a file should use cell-based processing
 * - .cell.cpp files use cells
 * - .cell.py files use cells
 * - .hpp files are single-file (header files)
 * - Regular .cpp/.c/.h files are single-file
 * - Regular .py files are single-file
 * - Bash .sh files use cells
 */
function shouldUseCells(filename) {
  const lower = filename.toLowerCase()
  // .cell.cpp and .cell.py files use cell processing
  if (lower.endsWith('.cell.cpp') || lower.endsWith('.cell.py')) return true
  // .hpp and regular .cpp/.c/.h are single file
  if (lower.endsWith('.hpp') || lower.endsWith('.cpp') || lower.endsWith('.c') || lower.endsWith('.h')) return false
  // Regular .py files are single file (can be imported as modules)
  if (lower.endsWith('.py')) return false
  // Bash .sh files use cells
  if (lower.endsWith('.sh')) return true
  // Default to single file
  return false
}

/**
 * Get the cell delimiter prefix for a language
 */
function getCellDelimiterPrefix(language) {
  return language === 'cpp' ? '// %%' : '# %%'
}

/**
 * Parse file content into cells based on cell delimiters
 */
function parseFileIntoCells(content, language) {
  const delimiter = CELL_DELIMITERS[language] || CELL_DELIMITERS.python
  const lines = content.split('\n')
  const parsedCells = []
  let currentCell = { content: '', title: '' }
  let inCell = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const match = line.match(delimiter)

    if (match) {
      // Save previous cell if it has content
      if (inCell && currentCell.content.trim()) {
        parsedCells.push({
          id: Date.now() + parsedCells.length,
          type: 'code',
          language,
          environment: selectedEnvironment.value,
          content: currentCell.content.trim(),
          title: currentCell.title.trim(),
          output: null,
          status: null
        })
      }
      // Start new cell
      currentCell = { content: '', title: match[1] || '' }
      inCell = true
    } else if (inCell) {
      currentCell.content += (currentCell.content ? '\n' : '') + line
    } else {
      // Content before first delimiter - treat as first cell
      currentCell.content += (currentCell.content ? '\n' : '') + line
    }
  }

  // Don't forget the last cell
  if (currentCell.content.trim()) {
    parsedCells.push({
      id: Date.now() + parsedCells.length,
      type: 'code',
      language,
      environment: selectedEnvironment.value,
      content: currentCell.content.trim(),
      title: currentCell.title.trim(),
      output: null,
      status: null
    })
  }

  // If no cells found, create one with all content
  if (parsedCells.length === 0) {
    parsedCells.push({
      id: Date.now(),
      type: 'code',
      language,
      environment: selectedEnvironment.value,
      content: content,
      title: '',
      output: null,
      status: null
    })
  }

  return parsedCells
}

/**
 * Combine cells back into file content with delimiters
 * For single-file mode (useCells=false), just return the content without delimiters
 */
function cellsToFileContent(cellsArray, language, useCells = true) {
  // Single-file mode: just return content without delimiters
  if (!useCells && cellsArray.length === 1) {
    return cellsArray[0].content
  }

  const delimiterPrefix = getCellDelimiterPrefix(language)

  if (cellsArray.length === 1 && !cellsArray[0].title) {
    // Single cell with no title - just return content without delimiter
    return cellsArray[0].content
  }

  return cellsArray.map((cell, index) => {
    const title = cell.title || (index === 0 ? '' : `Cell ${index + 1}`)
    const delimiter = `${delimiterPrefix}${title ? ' ' + title : ''}`
    return `${delimiter}\n${cell.content}`
  }).join('\n\n')
}

/**
 * Save current file
 */
async function saveFile() {
  if (!currentFile.value) return

  try {
    const activeTab = openTabs.value[activeTabIndex.value]
    const useCells = activeTab?.useCells !== false
    const content = cellsToFileContent(cells.value, currentFile.value.language, useCells)
    await axios.put(fileApiUrl(currentFile.value.path), {
      content
    })
    hasUnsavedChanges.value = false
  } catch (error) {
    console.error('Error saving file:', error)
    alert(`Failed to save: ${error.response?.data?.error || error.message}`)
  }
}

/**
 * Close current file (closes active tab)
 */
function closeFile() {
  if (activeTabIndex.value >= 0) {
    closeTab(activeTabIndex.value)
  }
}

/**
 * Switch to a specific tab
 */
function switchToTab(index) {
  if (index >= 0 && index < openTabs.value.length) {
    activeTabIndex.value = index
  }
}

/**
 * Close a specific tab
 */
function closeTab(index) {
  if (index < 0 || index >= openTabs.value.length) return

  const tab = openTabs.value[index]
  if (tab.isDirty) {
    if (!confirm(`"${tab.name}" has unsaved changes. Close anyway?`)) return
  }

  openTabs.value.splice(index, 1)

  // Adjust active tab index
  if (openTabs.value.length === 0) {
    activeTabIndex.value = -1
  } else if (index <= activeTabIndex.value) {
    activeTabIndex.value = Math.max(0, activeTabIndex.value - 1)
  }
}

/**
 * Get file icon based on extension
 */
function getFileIcon(filename) {
  const ext = filename.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'py': return '🐍'
    case 'cpp':
    case 'c':
    case 'h':
    case 'hpp': return '⚙'
    case 'sh':
    case 'bash': return '💻'
    case 'json': return '{}'
    case 'md': return '📄'
    default: return '📄'
  }
}

function addCell() {
  // Inherit language from current file or last cell
  const language = currentFile.value?.language || cells.value[cells.value.length - 1]?.language || 'python'
  const environment = cells.value[cells.value.length - 1]?.environment || selectedEnvironment.value

  cells.value.push({
    id: Date.now(),
    type: 'code',
    language,
    environment,
    content: '',
    title: '',
    output: null,
    status: null
  })
  hasUnsavedChanges.value = true
}

function updateCell(index, { content, language, environment, cppEnvironment, vendorEnvironment, compiler, cppStandard }) {
  const cell = cells.value[index]
  if (content !== undefined && content !== cell.content) {
    cell.content = content
    hasUnsavedChanges.value = true
  }
  if (language !== undefined) cell.language = language
  if (environment !== undefined) cell.environment = environment
  if (cppEnvironment !== undefined) cell.cppEnvironment = cppEnvironment
  if (vendorEnvironment !== undefined) cell.vendorEnvironment = vendorEnvironment
  if (compiler !== undefined) cell.compiler = compiler
  if (cppStandard !== undefined) cell.cppStandard = cppStandard
}

function deleteCell(index) {
  cells.value.splice(index, 1)
  if (cells.value.length === 0 && currentFile.value) {
    addCell()
  }
  hasUnsavedChanges.value = true
}

async function executeCell(index) {
  const cell = cells.value[index]
  cell.status = 'running'
  cell.output = ''

  // Auto-save before execution
  if (hasUnsavedChanges.value) {
    try {
      await saveFile()
    } catch (err) {
      cell.output = `Failed to save file before execution: ${err.message}`
      cell.status = 'error'
      return
    }
  }

  try {
    // Determine job type and params based on language
    const language = cell.language || 'python'
    let jobType, jobParams

    // Build sourceDir: workspace ID + file's directory path
    const fileDir = currentFile.value?.path
      ? currentFile.value.path.includes('/')
        ? currentFile.value.path.substring(0, currentFile.value.path.lastIndexOf('/'))
        : ''
      : ''
    const sourceDir = workspaceId.value
      ? (fileDir ? `${workspaceId.value}/${fileDir}` : String(workspaceId.value))
      : fileDir

    if (language === 'cpp') {
      // C++ execution - use file-level settings
      jobType = 'execute_cpp'
      jobParams = {
        code: cell.content,
        sourceDir,  // Workspace ID + directory containing the source file
        cppEnvironment: selectedCppEnvironment.value || '',
        vendorEnvironment: selectedVendorEnvironment.value || '',
        compiler: selectedCompiler.value,
        cppStandard: selectedCppStandard.value
      }
    } else {
      // Python/Bash execution - use file-level settings
      jobType = 'execute'
      jobParams = {
        code: cell.content,
        sourceDir,  // Workspace ID + directory containing the source file
        environment: selectedEnvironment.value,
        language
      }
    }

    // POST to create the job
    const response = await axios.post('/api/compute', {
      type: jobType,
      params: jobParams
    })

    if (response.data.error) {
      cell.output = `Error: ${response.data.error}`
      cell.status = 'error'
      return
    }

    const { jobId } = response.data
    cell.jobId = jobId

    // Subscribe to job output via WebSocket
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'subscribe',
        jobId
      }))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === 'subscribed') {
          // Successfully subscribed, job output will start streaming
        } else if (data.type === 'job_output') {
          const { stream, data: outputData } = data
          const streamLower = stream?.toLowerCase()

          if (streamLower === 'stdout') {
            // Append stdout to cell output, handling \r for progress updates
            appendCellOutput(cell, outputData)
          } else if (streamLower === 'stderr') {
            // Append stderr, handling \r for progress updates
            appendCellOutput(cell, outputData)
          } else if (streamLower === 'result') {
            // Job completed with result
            handleJobResult(cell, outputData)
            ws.close()
          } else if (streamLower === 'error') {
            cell.output = (cell.output || '') + `\nError: ${outputData}`
            cell.status = 'error'
            ws.close()
          } else if (streamLower === 'complete') {
            // Job finished (exit code received)
            if (cell.status === 'running') {
              cell.status = 'completed'
            }
            ws.close()
          }
        } else if (data.type === 'error') {
          cell.output = `Error: ${data.error}`
          cell.status = 'error'
          ws.close()
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', e)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      cell.output = (cell.output || '') + '\nWebSocket connection error'
      cell.status = 'error'
    }

    ws.onclose = () => {
      // Ensure status is set if not already
      if (cell.status === 'running') {
        cell.status = 'completed'
      }
    }

  } catch (error) {
    cell.output = `Error: ${error.message}`
    cell.status = 'error'
  }
}

function handleJobResult(cell, result) {
  try {
    const parsed = typeof result === 'string' ? JSON.parse(result) : result

    // Check if result contains error
    if (parsed.error || parsed.status === 'error') {
      cell.status = 'error'
      // If there's already output from streaming, append error info
      if (cell.output && cell.output.trim()) {
        cell.output = cell.output + '\n\n--- Error ---\n' + (parsed.error || parsed.stderr || `Exit code: ${parsed.exitCode}`)
      } else {
        cell.output = parsed.stderr || parsed.error || `Exit code: ${parsed.exitCode}`
      }
      return
    }

    // Success - keep streamed output and add status line
    cell.status = 'completed'

    // Check if result contains visualization data
    if (parsed.visualization) {
      renderMolecule(parsed.visualization)
      if (!cell.output) {
        cell.output = 'Visualization updated'
      }
    } else if (parsed.atoms || parsed.positions) {
      renderMolecule(parsed)
      if (!cell.output) {
        cell.output = 'Molecule rendered'
      }
    } else if (parsed.output !== undefined && !cell.output) {
      // Only set output if we don't already have streamed content
      cell.output = parsed.output + (parsed.stderr ? '\n' + parsed.stderr : '')
    }

    // For jobs that stream output (like C++ execution), just add a simple status
    // Don't replace the existing output with the result object
    if (cell.output && cell.output.trim()) {
      // Add a completion status line
      const statusLine = parsed.exitCode !== undefined
        ? `\n✓ Completed (exit code: ${parsed.exitCode})`
        : '\n✓ Completed'
      cell.output = cell.output + statusLine
    }
  } catch {
    cell.status = 'completed'
    if (!cell.output) {
      cell.output = String(result)
    }
  }
}

// ================== File Browser Functions ==================

async function handleFileOpen(file) {
  // With tabs, we can open files without losing unsaved changes
  await openFile(file)
}

function handleFileSelect(file) {
  // Just selection, no action needed
  console.log('Selected file:', file.path)
}

async function openFile(file) {
  try {
    // Check if file is already open in a tab
    const existingIndex = openTabs.value.findIndex(tab => tab.path === file.path)
    if (existingIndex >= 0) {
      // Switch to existing tab
      activeTabIndex.value = existingIndex
      return
    }

    const response = await axios.get(fileApiUrl(file.path))
    if (response.data.type === 'file') {
      const language = getLanguageFromExt(file.name)
      const content = response.data.content || ''
      const useCells = shouldUseCells(file.name)

      let fileCells
      if (useCells) {
        // Parse file content into cells (for .cell.cpp, .py, .sh files)
        fileCells = parseFileIntoCells(content, language)
      } else {
        // Single-file mode for regular .cpp, .c, .h, .hpp files
        fileCells = [{
          id: Date.now(),
          type: 'code',
          language,
          environment: selectedEnvironment.value,
          content: content,
          title: '',
          output: null,
          status: null
        }]
      }

      // Add new tab
      openTabs.value.push({
        path: file.path,
        name: file.name,
        language,
        cells: fileCells,
        isDirty: false,
        useCells  // Track whether this file uses cell mode
      })

      // Switch to the new tab
      activeTabIndex.value = openTabs.value.length - 1
    }
  } catch (error) {
    console.error('Error opening file:', error)
    alert(`Failed to open file: ${error.response?.data?.error || error.message}`)
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

// ================== Keyboard Shortcuts ==================

function handleKeydown(e) {
  // Ctrl+S or Cmd+S to save
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault()
    if (currentFile.value && hasUnsavedChanges.value) {
      saveFile()
    }
  }
}

// ================== Lifecycle ==================

onMounted(async () => {
  await Promise.all([
    loadEnvironments(),
    loadPythonVersions(),
    loadCppEnvironments(),
    loadVendorEnvironments(),
    loadCompilers()
  ])
  await nextTick()
  initThree()

  // Add keyboard shortcut listener
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  window.removeEventListener('resize', onWindowResize)
  document.removeEventListener('keydown', handleKeydown)
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

/* Left Sidebar */
.left-sidebar {
  width: 240px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  transition: width 0.2s;
}

.left-sidebar.collapsed {
  width: 40px;
}

.sidebar-header {
  padding: 0.5rem;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
}

.sidebar-tabs {
  display: flex;
  gap: 0.25rem;
}

.sidebar-tabs button {
  padding: 0.35rem 0.6rem;
  font-size: 0.75rem;
  background: transparent;
  border: 1px solid transparent;
  color: var(--text-secondary);
  border-radius: 4px;
  cursor: pointer;
}

.sidebar-tabs button:hover {
  background: var(--bg-primary);
}

.sidebar-tabs button.active {
  background: var(--bg-primary);
  color: var(--accent);
  border-color: var(--border);
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
  margin-left: auto;
}

.toggle-btn:hover {
  color: var(--accent);
}

.sidebar-content {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.sidebar-content > :deep(.file-browser) {
  height: 100%;
}

/* Environment panel inside sidebar */
.left-sidebar .sidebar-content:not(:has(.file-browser)) {
  padding: 0.75rem;
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

.env-lang {
  font-size: 0.7rem;
  color: #f59e0b;
}

.env-item.vendor .env-lang {
  color: #a855f7;
}

.env-sub-tabs {
  display: flex;
  gap: 0.25rem;
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.env-sub-tabs button {
  flex: 1;
  padding: 0.35rem 0.5rem;
  font-size: 0.75rem;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  border-radius: 4px;
  cursor: pointer;
}

.env-sub-tabs button:hover {
  background: var(--bg-primary);
}

.env-sub-tabs button.active {
  background: var(--accent);
  color: var(--bg-primary);
  border-color: var(--accent);
}

.env-section-header {
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  margin-top: 0.75rem;
  margin-bottom: 0.5rem;
  padding-bottom: 0.25rem;
  border-bottom: 1px solid var(--border);
}

.env-panel {
  padding: 0.75rem;
  gap: 0.5rem;
}

.vendor-btn {
  background: #a855f7 !important;
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

.tab-bar {
  display: flex;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border);
  overflow-x: auto;
  flex-shrink: 0;
}

.tab-bar::-webkit-scrollbar {
  height: 4px;
}

.tab-bar::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 2px;
}

.tab {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.5rem 0.75rem;
  background: var(--bg-tertiary);
  border-right: 1px solid var(--border);
  cursor: pointer;
  font-size: 0.8rem;
  color: var(--text-secondary);
  white-space: nowrap;
  transition: background 0.15s, color 0.15s;
  position: relative;
}

.tab:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.tab.active {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border-bottom: 2px solid var(--accent);
  margin-bottom: -1px;
}

.tab.dirty .tab-name {
  font-style: italic;
}

.tab-icon {
  font-size: 0.9rem;
}

.tab-name {
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.tab-dirty-indicator {
  color: var(--accent);
  font-size: 0.6rem;
  margin-left: -0.2rem;
}

.tab-close {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  padding: 0;
  margin-left: 0.25rem;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  border-radius: 3px;
  cursor: pointer;
  font-size: 0.9rem;
  line-height: 1;
  opacity: 0;
  transition: opacity 0.15s, background 0.15s;
}

.tab:hover .tab-close,
.tab.active .tab-close {
  opacity: 1;
}

.tab-close:hover {
  background: var(--error);
  color: white;
}

.panel-header {
  background: var(--bg-secondary);
  padding: 0.5rem 1rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.file-info {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
  flex: 1;
  margin-right: 1rem;
  min-width: 0;
}

.file-name {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-name.modified {
  color: var(--accent);
}

.file-name.empty {
  color: var(--text-secondary);
  font-style: italic;
}

.file-path {
  font-size: 0.7rem;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
}

.file-env-select {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--accent);
  cursor: pointer;
  max-width: 120px;
}

.file-env-select:hover {
  border-color: var(--accent);
}

.file-env-select.cpp {
  color: #f59e0b;
}

.file-env-select.vendor {
  color: #a855f7;
}

.file-env-select.compiler {
  color: #10b981;
}

.file-env-select.std {
  color: #6366f1;
}

.save-btn {
  background: var(--success) !important;
  color: white !important;
}

.save-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.close-btn {
  background: transparent !important;
  color: var(--text-secondary) !important;
  font-size: 1.1rem !important;
  padding: 0.2rem 0.5rem !important;
}

.close-btn:hover {
  color: var(--error) !important;
}

.no-file-message {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
  text-align: center;
  padding: 2rem;
}

.no-file-message p {
  margin: 0.5rem 0;
}

.no-file-message .hint {
  font-size: 0.8rem;
  opacity: 0.7;
}

.no-file-message code {
  background: var(--bg-tertiary);
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-family: monospace;
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
  transition: width 0.2s ease;
}

.dialog.expanded {
  width: 66.67vw;
  max-width: 66.67vw;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.25rem;
}

.dialog-header h3 {
  margin: 0;
  color: var(--accent);
  font-size: 1.1rem;
}

.expand-btn {
  width: 28px;
  height: 28px;
  padding: 0;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.expand-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
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

.create-log {
  margin-top: 1rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}

.log-header {
  padding: 0.5rem 0.75rem;
  background: var(--bg-primary);
  font-size: 0.8rem;
  color: var(--accent);
  border-bottom: 1px solid var(--border);
}

.log-output {
  max-height: 200px;
  overflow-y: auto;
  overflow-x: auto;
  background: #0a0a12;
  font-family: 'Monaco', 'Menlo', 'Consolas', 'Liberation Mono', 'Courier New', monospace;
  font-size: 0.75rem;
  padding: 0.5rem;
  line-height: 1.4;
}

.dialog.expanded .log-output {
  max-height: 300px;
}

.log-line {
  padding: 0.1rem 0;
  color: var(--text-secondary);
  white-space: pre;
  font-variant-ligatures: none;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.log-line.progress {
  color: #888;
}

.log-line.success {
  color: #4ade80;
}

.log-line.error {
  color: #f87171;
}
</style>
