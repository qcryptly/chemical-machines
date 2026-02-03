<template>
  <div class="workspace">
    <!-- Left Sidebar with Tabs -->
    <div class="left-sidebar" :class="{ collapsed: !sidebarOpen }" :style="{ width: sidebarOpen ? sidebarWidth + 'px' : '40px' }">
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
        <div class="header-actions">
          <button @click="showProfileDialog = true" class="profile-btn" :title="activeProfile ? activeProfile.name : 'Profile'">
            <User :size="16" />
            <span class="profile-indicator" v-if="activeProfile"></span>
          </button>
          <button @click="sidebarOpen = !sidebarOpen" class="toggle-btn">
            <PanelLeftClose v-if="sidebarOpen" :size="14" />
            <PanelLeftOpen v-else :size="14" />
          </button>
        </div>
      </div>

      <!-- File Browser Tab -->
      <div class="sidebar-content" v-if="sidebarOpen && sidebarTab === 'files'">
        <FileBrowser
          ref="fileBrowserRef"
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
                <span class="pkg-count clickable" @click.stop="openEnvDetail(env, 'python')">
                  <Package class="pkg-icon" :size="12" />
                  {{ env.packageCount }} packages
                </span>
                <button
                  v-if="!env.isBase && env.name !== 'base'"
                  @click.stop="deleteEnvironment(env.name)"
                  class="btn-icon delete-env-btn"
                  title="Delete environment"
                ><X :size="12" /></button>
              </div>
            </div>
          </div>

          <div class="new-env-section">
            <button @click="showCreateDialog = true" class="btn-secondary create-env-btn">
              <Plus :size="12" /> New Python Env
            </button>
          </div>

          <button @click="loadEnvironments" class="btn-secondary refresh-btn">Refresh</button>
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
                <span class="pkg-count clickable" @click.stop="openEnvDetail(env, 'cpp')">
                  <Package class="pkg-icon" :size="12" />
                  {{ env.packages?.length || 0 }} packages
                </span>
                <button
                  @click.stop="deleteCppEnvironment(env.name)"
                  class="btn-icon delete-env-btn"
                  title="Delete environment"
                ><X :size="12" /></button>
              </div>
            </div>
          </div>

          <div class="new-env-section">
            <button @click="showCppDialog = true" class="btn-secondary create-env-btn">
              <Plus :size="12" /> New C++ Env
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
                  class="btn-icon delete-env-btn"
                  title="Delete environment"
                ><X :size="12" /></button>
              </div>
            </div>
          </div>

          <div class="new-env-section">
            <button @click="showVendorDialog = true" class="btn-secondary create-env-btn vendor-btn">
              <Plus :size="12" /> Build from Source
            </button>
          </div>

          <button @click="loadCppEnvironments(); loadVendorEnvironments()" class="btn-secondary refresh-btn">Refresh</button>
        </template>
      </div>
    </div>

    <!-- Resize Handle (Vertical) for Sidebar -->
    <div v-if="sidebarOpen" class="resize-handle-v" @pointerdown="startSidebarResize"></div>

    <!-- Main Content -->
    <div class="main-content">
      <!-- Code Cells / Terminal -->
      <div class="notebook-panel">
        <!-- Mode Toggle Bar -->
        <div class="mode-toggle-bar">
          <button class="home-btn" @click="router.push('/')" title="Back to workspaces">
            <Home :size="14" />
            <span class="home-label">Home</span>
          </button>
          <div class="mode-tabs">
            <button
              :class="{ active: bottomPanelMode === 'editor' }"
              @click="bottomPanelMode = 'editor'"
              title="Code Editor"
            >
              <Code :size="14" />
              Editor
            </button>
            <button
              :class="{ active: bottomPanelMode === 'terminal' }"
              @click="bottomPanelMode = 'terminal'; $nextTick(() => terminalRef?.focus())"
              title="Terminal"
            >
              <TerminalSquare :size="14" />
              Terminal
            </button>
          </div>
          <div class="toolbar-spacer"></div>
          <!-- Editor controls (only shown in editor mode) -->
          <div class="toolbar-actions" v-if="bottomPanelMode === 'editor'">
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
            <!-- Markdown preview toggle -->
            <button
              v-if="isMarkdownFile"
              @click="markdownPreviewMode = !markdownPreviewMode"
              class="preview-btn"
              :class="{ active: markdownPreviewMode }"
              :title="markdownPreviewMode ? 'Edit markdown' : 'Preview markdown'"
            >
              <Pencil v-if="markdownPreviewMode" :size="12" />
              <Eye v-else :size="12" />
            </button>
            <button v-if="currentFile" @click="saveFile" class="save-btn" :disabled="!hasUnsavedChanges" title="Save file (Ctrl+S)">
              <Save :size="12" />
            </button>
            <button v-if="currentFile && currentUseCells && !isMarkdownFile" @click="addCell" class="toolbar-btn" title="Add new cell">
              <Plus :size="12" />
            </button>
            <button v-if="currentFile" @click="closeFile" class="close-btn" title="Close file"><X :size="12" /></button>
          </div>
        </div>

        <!-- Editor View -->
        <template v-if="bottomPanelMode === 'editor'">
        <div class="editor-groups-container">
          <SplitPane
            :node="layoutRoot"
            :focused-leaf-id="focusedLeafId"
            @focus-leaf="focusedLeafId = $event"
            @switch-tab="switchTabInLeaf($event.leafId, $event.tabIndex)"
            @close-tab="closeTabInLeaf($event.leafId, $event.tabIndex)"
            @split="handleSplit($event.leafId, $event.direction)"
            @update-cell="handleLeafCellUpdate($event)"
            @run-cell="handleLeafCellRun($event)"
            @delete-cell="handleLeafCellDelete($event)"
            @create-cell-below="handleLeafCellCreateBelow($event)"
            @reorder-cells="handleLeafCellReorder($event)"
            @interrupt-cell="handleLeafCellInterrupt($event)"
            @tab-drop="handleTabDrop($event)"
            @resize-start="handleResizeStart($event)"
          />
        </div>
        </template>

        <!-- Terminal View -->
        <div class="terminal-view" v-if="bottomPanelMode === 'terminal'">
          <Terminal
            ref="terminalRef"
            :workspace-id="workspaceId"
            :active="bottomPanelMode === 'terminal'"
            @files-changed="refreshFiles"
          />
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
          <button @click="showCreateDialog = false" class="btn-secondary cancel-btn" :disabled="isCreating">Cancel</button>
          <button @click="createEnvironment" class="btn-primary create-btn" :disabled="!newEnvName || isCreating">
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

    <!-- Environment Detail Dialog -->
    <EnvironmentDetailDialog
      v-if="showEnvDetailDialog && selectedEnvForDetail"
      :environment="selectedEnvForDetail"
      :env-type="selectedEnvTypeForDetail"
      @close="showEnvDetailDialog = false"
      @updated="handleEnvDetailUpdated"
    />

    <!-- Profile Dialog -->
    <ProfileDialog
      v-if="showProfileDialog"
      @close="showProfileDialog = false"
      @updated="handleProfileUpdated"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'
import EditorGroup from '../components/EditorGroup.vue'
import SplitPane from '../components/SplitPane.vue'
import FileBrowser from '../components/FileBrowser.vue'
import CppEnvironmentDialog from '../components/CppEnvironmentDialog.vue'
import VendorEnvironmentDialog from '../components/VendorEnvironmentDialog.vue'
import EnvironmentDetailDialog from '../components/EnvironmentDetailDialog.vue'
import ProfileDialog from '../components/ProfileDialog.vue'
import Terminal from '../components/Terminal.vue'
import {
  User, PanelLeftClose, PanelLeftOpen, Package, X, Plus, Home, Save,
  Code, TerminalSquare, Eye, Pencil, Columns2
} from 'lucide-vue-next'

import { marked } from 'marked'

// Helper to generate slug from text (for anchor IDs)
function slugify(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')  // Remove non-word chars
    .replace(/\s+/g, '-')       // Replace spaces with dashes
    .replace(/--+/g, '-')       // Replace multiple dashes with single
}

// Custom renderer to add IDs to headings for anchor links
// (marked v15 removed built-in headerIds, need custom renderer)
const renderer = {
  heading({ tokens, depth }) {
    const text = this.parser.parseInline(tokens)
    const id = slugify(text.replace(/<[^>]*>/g, ''))  // Strip HTML tags for ID
    return `<h${depth} id="${id}">${text}</h${depth}>\n`
  }
}

// Configure marked with custom renderer
marked.use({ renderer, gfm: true })

const route = useRoute()
const router = useRouter()

// Workspace ID from route
const workspaceId = computed(() => route.params.id)

// Helper to build workspace file API URL
function fileApiUrl(filePath = '') {
  const base = `/api/workspaces/${workspaceId.value}/files`
  return filePath ? `${base}/${encodeURIComponent(filePath)}` : base
}

// Helper to build output API URL
function outputApiUrl(filePath) {
  return `/api/workspaces/${workspaceId.value}/output/${encodeURIComponent(filePath)}`
}

// Fetch HTML outputs for a file from .out/ directory
async function fetchHtmlOutputs(filePath) {
  try {
    const response = await axios.get(outputApiUrl(filePath))
    if (response.data.exists) {
      return response.data.outputs || []
    }
    return []
  } catch (error) {
    console.error('Error fetching HTML outputs:', error)
    return []
  }
}

// Refresh HTML outputs for the current tab
async function refreshHtmlOutputs() {
  if (activeTabIndex.value >= 0 && activeTabIndex.value < openTabs.value.length) {
    const tab = openTabs.value[activeTabIndex.value]
    const outputs = await fetchHtmlOutputs(tab.path)
    tab.htmlOutputs = outputs
  }
}

// Sidebar state
const sidebarOpen = ref(true)
const sidebarTab = ref('files')

// Bottom panel mode: 'editor' or 'terminal'
const bottomPanelMode = ref('editor')
const terminalRef = ref(null)
const fileBrowserRef = ref(null)


// Environment state (Python/Conda)
const environments = ref([])
const selectedEnvironment = ref('base')
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

// Environment detail dialog
const showEnvDetailDialog = ref(false)
const selectedEnvForDetail = ref(null)
const selectedEnvTypeForDetail = ref('python')

// Profile
const showProfileDialog = ref(false)
const activeProfile = ref(null)
const newEnvName = ref('')
const newEnvPython = ref('3.12')
const newEnvPackages = ref('')
const createStatus = ref('')
const createLogs = ref([])
const isCreating = ref(false)
const logOutput = ref(null)
const dialogExpanded = ref(false)

// ================== Editor Layout Tree (Split View) ==================
import { useLayoutTree } from '../composables/useLayoutTree.js'

const {
  root: layoutRoot,
  focusedLeafId,
  focusedLeaf,
  findNode,
  getAllLeaves,
  findLeafById,
  splitLeaf,
  removeLeaf,
  updateSizes,
  moveTab
} = useLayoutTree()

// Helper to get the active tab from the focused leaf
function _focusedTab() {
  const g = focusedLeaf.value
  if (!g || g.activeTabIndex < 0 || g.activeTabIndex >= g.tabs.length) return null
  return g.tabs[g.activeTabIndex]
}

// Legacy aliases so existing code keeps working
const openTabs = computed(() => focusedLeaf.value?.tabs || [])
const activeTabIndex = computed({
  get: () => focusedLeaf.value?.activeTabIndex ?? -1,
  set: (v) => { if (focusedLeaf.value) focusedLeaf.value.activeTabIndex = v }
})

// Computed properties for current file (reads from focused group)
const currentFile = computed(() => {
  const tab = _focusedTab()
  if (!tab) return null
  return { path: tab.path, name: tab.name, language: tab.language }
})

const cells = computed({
  get: () => _focusedTab()?.cells || [],
  set: (value) => {
    const tab = _focusedTab()
    if (tab) tab.cells = value
  }
})

const hasUnsavedChanges = computed({
  get: () => _focusedTab()?.isDirty || false,
  set: (value) => {
    const tab = _focusedTab()
    if (tab) tab.isDirty = value
  }
})

const currentUseCells = computed(() => _focusedTab()?.useCells !== false)

const isMarkdownFile = computed(() => _focusedTab()?.isMarkdown === true)

const markdownPreviewMode = computed({
  get: () => _focusedTab()?.previewMode === true,
  set: (value) => {
    const tab = _focusedTab()
    if (tab) tab.previewMode = value
  }
})


// Sidebar state
const sidebarWidth = ref(240)


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
      { name: 'torch', pythonVersion: '3.12', packageCount: 0, isBase: false }
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
      selectedEnvironment.value = 'base'
    }
  } catch (error) {
    console.error('Error deleting environment:', error)
    alert(`Failed to delete: ${error.response?.data?.error || error.message}`)
  }
}

function openEnvDetail(env, envType) {
  selectedEnvForDetail.value = env
  selectedEnvTypeForDetail.value = envType
  showEnvDetailDialog.value = true
}

function handleEnvDetailUpdated() {
  if (selectedEnvTypeForDetail.value === 'python') {
    loadEnvironments()
  } else {
    loadCppEnvironments()
  }
}

// ================== Profile Functions ==================

async function loadProfile() {
  try {
    const response = await axios.get('/api/profile')
    activeProfile.value = response.data
  } catch (error) {
    console.error('Error loading profile:', error)
  }
}

function handleProfileUpdated(profile) {
  activeProfile.value = profile
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
  if (ext === 'md' || ext === 'markdown') return 'markdown'
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
function shouldUseCells(filename, content = '') {
  const lower = filename.toLowerCase()
  // .cell.cpp and .cell.py files always use cell processing
  if (lower.endsWith('.cell.cpp') || lower.endsWith('.cell.py')) return true
  // .hpp and regular .cpp/.c/.h are single file
  if (lower.endsWith('.hpp') || lower.endsWith('.cpp') || lower.endsWith('.c') || lower.endsWith('.h')) return false
  // Regular .py files: auto-detect cell delimiters in content
  if (lower.endsWith('.py')) {
    return content.split('\n').some(line => /^# %%/.test(line))
  }
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
 * Close current file (closes active tab in focused group)
 */
function closeFile() {
  if (activeTabIndex.value >= 0) {
    closeTabInLeaf(focusedLeafId.value, activeTabIndex.value)
  }
}

/**
 * Switch to a tab within a specific leaf
 */
function switchTabInLeaf(leafId, index) {
  const leaf = findLeafById(leafId)
  if (!leaf || index < 0 || index >= leaf.tabs.length) return
  leaf.activeTabIndex = index
  focusedLeafId.value = leafId
}

/**
 * Close a tab within a specific leaf
 */
function closeTabInLeaf(leafId, index) {
  const leaf = findLeafById(leafId)
  if (!leaf || index < 0 || index >= leaf.tabs.length) return

  const tab = leaf.tabs[index]
  if (tab.isDirty) {
    if (!confirm(`"${tab.name}" has unsaved changes. Close anyway?`)) return
  }

  leaf.tabs.splice(index, 1)

  if (leaf.tabs.length === 0) {
    leaf.activeTabIndex = -1
    removeLeaf(leafId)
  } else if (index <= leaf.activeTabIndex) {
    leaf.activeTabIndex = Math.max(0, leaf.activeTabIndex - 1)
  }
}

// ================== Split Editor Functions ==================

function handleSplit(leafId, direction) {
  const newLeafId = splitLeaf(leafId, direction)
  if (newLeafId != null) {
    focusedLeafId.value = newLeafId
  }
}

function handleTabDrop({ fromGroupId, fromTab, toGroupId, toTab }) {
  moveTab(fromGroupId, fromTab, toGroupId, toTab)
}

// ================== Split Resize ==================

let resizingState = null

function handleResizeStart({ event, splitId, childIndex, direction }) {
  event.preventDefault()
  const container = event.target.parentElement
  resizingState = { splitId, childIndex, direction, container }

  const overlay = document.createElement('div')
  overlay.id = 'split-resize-overlay'
  overlay.style.cssText = `
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    z-index: 9999; cursor: ${direction === 'horizontal' ? 'col-resize' : 'row-resize'};
  `
  document.body.appendChild(overlay)

  document.addEventListener('pointermove', doSplitResize)
  document.addEventListener('pointerup', stopSplitResize)
  document.addEventListener('pointercancel', stopSplitResize)
  document.body.style.userSelect = 'none'
  document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize'
}

function doSplitResize(e) {
  if (!resizingState) return
  e.preventDefault()
  const { splitId, childIndex, direction, container } = resizingState
  const rect = container.getBoundingClientRect()

  const splitNode = findNode(layoutRoot.value, splitId)
  if (!splitNode || splitNode.type !== 'split') return

  // Calculate position as percentage within container
  let position
  if (direction === 'horizontal') {
    position = (e.clientX - rect.left) / rect.width * 100
  } else {
    position = (e.clientY - rect.top) / rect.height * 100
  }

  // The handle at childIndex separates children[childIndex-1] and children[childIndex]
  const prevIdx = childIndex - 1
  const combinedSize = splitNode.sizes[prevIdx] + splitNode.sizes[childIndex]

  // Cumulative sizes before the pair being resized
  let cumulativeBefore = 0
  for (let i = 0; i < prevIdx; i++) cumulativeBefore += splitNode.sizes[i]

  let newPrevSize = position - cumulativeBefore
  let newCurrSize = combinedSize - newPrevSize

  // Enforce minimum sizes (15% of combined or 5% absolute)
  const minSize = Math.max(5, combinedSize * 0.15)
  newPrevSize = Math.max(minSize, Math.min(combinedSize - minSize, newPrevSize))
  newCurrSize = combinedSize - newPrevSize

  const newSizes = [...splitNode.sizes]
  newSizes[prevIdx] = newPrevSize
  newSizes[childIndex] = newCurrSize
  updateSizes(splitId, newSizes)
}

function stopSplitResize() {
  if (!resizingState) return
  resizingState = null

  const overlay = document.getElementById('split-resize-overlay')
  if (overlay && overlay.parentNode) {
    overlay.parentNode.removeChild(overlay)
  }

  document.removeEventListener('pointermove', doSplitResize)
  document.removeEventListener('pointerup', stopSplitResize)
  document.removeEventListener('pointercancel', stopSplitResize)
  document.body.style.userSelect = ''
  document.body.style.cursor = ''
}

// ================== Leaf-aware Cell Handlers ==================
// These route events from EditorGroup components to existing cell functions
// by temporarily ensuring the correct leaf is focused.

function _withLeaf(leafId, fn) {
  focusedLeafId.value = leafId
  fn()
}

function handleLeafCellUpdate({ leafId, cellIndex, data }) {
  _withLeaf(leafId, () => updateCell(cellIndex, data))
}

function handleLeafCellRun({ leafId, cellIndex }) {
  _withLeaf(leafId, () => executeCell(cellIndex))
}

function handleLeafCellDelete({ leafId, cellIndex }) {
  _withLeaf(leafId, () => deleteCell(cellIndex))
}

function handleLeafCellCreateBelow({ leafId, cellIndex }) {
  _withLeaf(leafId, () => createCellBelow(cellIndex))
}

function handleLeafCellReorder({ leafId, payload }) {
  _withLeaf(leafId, () => reorderCells(payload))
}

function handleLeafCellInterrupt({ leafId, cellIndex }) {
  _withLeaf(leafId, () => interruptCell(cellIndex))
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

function createCellBelow(index) {
  // Inherit language and environment from the cell above
  const cellAbove = cells.value[index]
  const language = cellAbove?.language || currentFile.value?.language || 'python'
  const environment = cellAbove?.environment || selectedEnvironment.value

  const newCell = {
    id: Date.now(),
    type: 'code',
    language,
    environment,
    content: '',
    title: '',
    output: null,
    status: null
  }

  // Insert the new cell right after the current one
  cells.value.splice(index + 1, 0, newCell)
  hasUnsavedChanges.value = true
}

function reorderCells({ fromIndex, toIndex }) {
  if (fromIndex === toIndex) return

  // Remove the cell from its current position
  const [movedCell] = cells.value.splice(fromIndex, 1)

  // Insert it at the new position
  cells.value.splice(toIndex, 0, movedCell)

  hasUnsavedChanges.value = true
}

async function executeCell(index) {
  const cell = cells.value[index]
  cell.status = 'running'
  cell.output = ''

  // Check if services are healthy before executing
  // Retry up to 12 times (60 seconds) while services are starting
  let servicesReady = false
  let retryCount = 0
  const maxRetries = 12
  const retryDelay = 5000

  while (!servicesReady && retryCount < maxRetries) {
    try {
      const healthResponse = await axios.get('/api/health', { timeout: 5000 })
      if (healthResponse.data.status === 'healthy') {
        servicesReady = true
      } else {
        // Services not ready yet - show waiting message
        const unhealthyServices = Object.entries(healthResponse.data.services || {})
          .filter(([, s]) => s.status !== 'healthy')
          .map(([name]) => name)
        cell.output = `Waiting for services to start...\nServices starting: ${unhealthyServices.join(', ')}`
        retryCount++
        if (retryCount < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, retryDelay))
        }
      }
    } catch (error) {
      // Health check failed - services likely not running
      cell.output = `Waiting for services to start...\nConnecting to compute service...`
      retryCount++
      if (retryCount < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, retryDelay))
      }
    }
  }

  if (!servicesReady) {
    cell.output = 'Services unavailable. Please wait for services to start and try again.'
    cell.status = 'error'
    return
  }

  // Clear the waiting message before execution
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

    // Cell output information for cm_output library
    const cellInfo = {
      filePath: currentFile.value?.path || '',
      cellIndex: index,
      isCellFile: currentUseCells.value
    }

    if (language === 'cpp') {
      // C++ execution - use file-level settings
      jobType = 'execute_cpp'
      jobParams = {
        code: cell.content,
        sourceDir,  // Workspace ID + directory containing the source file
        cppEnvironment: selectedCppEnvironment.value || '',
        vendorEnvironment: selectedVendorEnvironment.value || '',
        compiler: selectedCompiler.value,
        cppStandard: selectedCppStandard.value,
        cellInfo  // For HTML output support
      }
    } else {
      // Python/Bash execution - use file-level settings
      jobType = 'execute'
      jobParams = {
        code: cell.content,
        sourceDir,  // Workspace ID + directory containing the source file
        environment: selectedEnvironment.value,
        language,
        cellInfo  // For HTML output support
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
      // Refresh HTML outputs after execution completes
      // Use setTimeout to allow the compute service to write the output file
      setTimeout(() => {
        refreshHtmlOutputs()
      }, 500)
    }

  } catch (error) {
    // Check if this is a connection error to the compute service
    if (error.code === 'ECONNREFUSED' || error.message?.includes('ECONNREFUSED') ||
        error.code === 'ECONNRESET' || error.message?.includes('ECONNRESET') ||
        error.response?.status === 503 || error.response?.status === 500) {
      cell.output = 'Compute service is not available. Services may still be starting - please try again in a moment.'
    } else {
      cell.output = `Error: ${error.message}`
    }
    cell.status = 'error'
  }
}

async function interruptCell(index) {
  const cell = cells.value[index]

  if (cell.status !== 'running') {
    return
  }

  try {
    // Build sourceDir same as in executeCell
    const fileDir = currentFile.value?.path
      ? currentFile.value.path.includes('/')
        ? currentFile.value.path.substring(0, currentFile.value.path.lastIndexOf('/'))
        : ''
      : ''
    const sourceDir = workspaceId.value
      ? (fileDir ? `${workspaceId.value}/${fileDir}` : String(workspaceId.value))
      : fileDir

    // Cell info to identify the kernel
    const cellInfo = {
      filePath: currentFile.value?.path || '',
      cellIndex: index,
      isCellFile: currentUseCells.value
    }

    // Send interrupt action to the backend
    const response = await axios.post('/api/compute', {
      type: 'execute',
      params: {
        kernelAction: 'interrupt',
        sourceDir,
        cellInfo
      }
    })

    if (response.data.interrupted) {
      cell.output = (cell.output || '') + '\n\n[Execution interrupted by user]'
      cell.status = 'error'
    }
  } catch (error) {
    console.error('Failed to interrupt cell:', error)
    cell.output = (cell.output || '') + `\n\nFailed to interrupt: ${error.message}`
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

    // Check if result contains output data
    if (parsed.output !== undefined && !cell.output) {
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

function refreshFiles() {
  // Refresh the file browser when terminal creates/modifies files
  if (fileBrowserRef.value?.refresh) {
    fileBrowserRef.value.refresh()
  }
}

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
    // Switch to editor mode when opening a file
    bottomPanelMode.value = 'editor'

    // Check if file is already open in any leaf
    const allLeaves = getAllLeaves()
    for (const leaf of allLeaves) {
      const existingIndex = leaf.tabs.findIndex(tab => tab.path === file.path)
      if (existingIndex >= 0) {
        leaf.activeTabIndex = existingIndex
        focusedLeafId.value = leaf.id
        return
      }
    }

    const response = await axios.get(fileApiUrl(file.path))
    if (response.data.type === 'file') {
      const language = getLanguageFromExt(file.name)
      const content = response.data.content || ''
      const useCells = shouldUseCells(file.name, content)

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

      // Fetch HTML outputs from .out/ directory
      const fileHtmlOutputs = await fetchHtmlOutputs(file.path)

      // Check if this is a markdown file
      const isMarkdown = file.name.toLowerCase().endsWith('.md')

      // Add new tab to focused leaf
      const group = focusedLeaf.value
      group.tabs.push({
        path: file.path,
        name: file.name,
        language,
        cells: fileCells,
        isDirty: false,
        useCells,
        htmlOutputs: fileHtmlOutputs,
        isMarkdown,
        previewMode: isMarkdown
      })

      // Switch to the new tab
      group.activeTabIndex = group.tabs.length - 1
    }
  } catch (error) {
    console.error('Error opening file:', error)
    alert(`Failed to open file: ${error.response?.data?.error || error.message}`)
  }
}

// ================== Sidebar Resize ==================

let isSidebarResizing = false

function startSidebarResize(e) {
  e.preventDefault()
  isSidebarResizing = true

  // Add overlay to capture pointer events
  const overlay = document.createElement('div')
  overlay.id = 'sidebar-resize-overlay'
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
    cursor: col-resize;
  `
  document.body.appendChild(overlay)

  document.addEventListener('pointermove', doSidebarResize)
  document.addEventListener('pointerup', stopSidebarResize)
  document.addEventListener('pointercancel', stopSidebarResize)
  window.addEventListener('blur', stopSidebarResize)
  document.body.style.userSelect = 'none'
  document.body.style.cursor = 'col-resize'
}

function doSidebarResize(e) {
  if (!isSidebarResizing) return
  e.preventDefault()

  const newWidth = e.clientX

  // Set min/max widths
  if (newWidth >= 150 && newWidth <= 600) {
    sidebarWidth.value = newWidth
  }
}

function stopSidebarResize() {
  if (!isSidebarResizing) return
  isSidebarResizing = false

  // Remove overlay
  const overlay = document.getElementById('sidebar-resize-overlay')
  if (overlay && overlay.parentNode) {
    overlay.parentNode.removeChild(overlay)
  }

  document.removeEventListener('pointermove', doSidebarResize)
  document.removeEventListener('pointerup', stopSidebarResize)
  document.removeEventListener('pointercancel', stopSidebarResize)
  window.removeEventListener('blur', stopSidebarResize)
  document.body.style.userSelect = ''
  document.body.style.cursor = ''
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
  // Ctrl+\ to split focused pane horizontally, Ctrl+Shift+\ for vertical
  if ((e.ctrlKey || e.metaKey) && e.key === '\\') {
    e.preventDefault()
    if (e.shiftKey) {
      handleSplit(focusedLeafId.value, 'vertical')
    } else {
      handleSplit(focusedLeafId.value, 'horizontal')
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
    loadCompilers(),
    loadProfile(),
  ])

  // Add keyboard shortcut listener
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
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
  min-width: 150px;
  max-width: 600px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.left-sidebar.collapsed {
  min-width: 40px;
  max-width: 40px;
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

.header-actions {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  margin-left: auto;
}

.profile-btn {
  width: 28px;
  height: 28px;
  padding: 0;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 50%;
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.profile-btn:hover {
  color: var(--accent);
  border-color: var(--border);
  background: var(--bg-primary);
}

.profile-indicator {
  position: absolute;
  bottom: 2px;
  right: 2px;
  width: 6px;
  height: 6px;
  background: #4ade80;
  border-radius: 50%;
  border: 1px solid var(--bg-tertiary);
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
  background: rgba(0, 212, 255, 0.1);
  color: var(--accent);
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

.pkg-count.clickable {
  cursor: pointer;
  padding: 0.1rem 0.3rem;
  border-radius: 3px;
  transition: background 0.15s, color 0.15s;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.pkg-count.clickable:hover {
  background: rgba(0, 212, 255, 0.2);
  color: var(--accent);
}

.pkg-icon {
  display: inline-flex;
  opacity: 0.7;
  vertical-align: middle;
}

.pkg-count.clickable:hover .pkg-icon {
  opacity: 1;
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
  background: rgba(255, 68, 68, 0.15);
  color: var(--error);
}

.new-env-section {
  margin-top: auto;
}

.create-env-btn {
  width: 100%;
  padding: 0.5rem;
  font-size: 0.8rem;
  color: var(--accent);
}

.create-env-btn:hover {
  background: var(--bg-tertiary);
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


/* Vertical resize handle for sidebar */
.resize-handle-v {
  width: 6px;
  background: var(--border);
  cursor: col-resize;
  transition: background 0.2s;
  flex-shrink: 0;
}

.resize-handle-v:hover {
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

/* Mode Toggle Bar */
.mode-toggle-bar {
  display: flex;
  align-items: center;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border);
  padding: 0 0.5rem;
  gap: 0.5rem;
  flex-shrink: 0;
}

.mode-tabs {
  display: flex;
  gap: 2px;
  padding: 0.35rem 0;
}

.mode-tabs button {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.6rem;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.75rem;
  transition: all 0.15s;
}

.mode-tabs button:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.mode-tabs button.active {
  background: var(--bg-primary);
  border-color: var(--border);
  color: var(--accent);
}

.mode-tabs button svg {
  opacity: 0.7;
}

.mode-tabs button.active svg {
  opacity: 1;
}

/* Terminal View */
.terminal-view {
  flex: 1;
  overflow: hidden;
  background: #0a0a12;
}

.tab-bar {
  display: flex;
  flex: 1;
  background: transparent;
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
  display: inline-flex;
  align-items: center;
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
  background: rgba(255, 68, 68, 0.15);
  color: var(--error);
}

/* Tab drag reorder indicators */
.tab.dragging {
  opacity: 0.4;
}

.tab.drag-over-left {
  box-shadow: -2px 0 0 0 var(--accent);
}

.tab.drag-over-right {
  box-shadow: 2px 0 0 0 var(--accent);
}

.home-btn {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  height: 28px;
  padding: 0 0.5rem;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  color: var(--text-secondary);
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.15s;
  font-size: 0.8rem;
}

.home-label {
  font-size: 0.8rem;
}

.home-btn:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.toolbar-spacer {
  flex: 1;
}

.toolbar-actions {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  flex-shrink: 0;
}

.toolbar-actions button {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.25rem 0.4rem;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.75rem;
  transition: all 0.15s;
}

.toolbar-actions button:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
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
  background: transparent !important;
  color: var(--success) !important;
}

.save-btn:hover {
  background: var(--bg-secondary) !important;
}

.save-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.close-btn {
  background: transparent !important;
  color: var(--text-secondary) !important;
  padding: 0.25rem 0.4rem !important;
}

.close-btn:hover {
  color: var(--error) !important;
  background: var(--bg-secondary) !important;
}

.preview-btn.active {
  color: var(--accent);
  border-color: var(--accent);
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

.cells {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem;
}

/* Split Editor Layout */
.editor-groups-container {
  flex: 1;
  display: flex;
  overflow: hidden;
  position: relative;
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

/* Markdown Preview Button */
.preview-btn {
  padding: 0.3rem 0.6rem;
  font-size: 0.8rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.15s;
}

.preview-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.preview-btn.active {
  background: rgba(0, 212, 255, 0.1);
  border-color: var(--accent);
  color: var(--accent);
}

/* Markdown preview styles are in EditorGroup.vue where the element lives */
</style>
