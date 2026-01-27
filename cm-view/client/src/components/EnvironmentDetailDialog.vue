<template>
  <div class="dialog-overlay" @click.self="$emit('close')">
    <div class="dialog" :class="{ expanded: dialogExpanded }">
      <div class="dialog-header">
        <h3>{{ environment.name }}</h3>
        <div class="header-actions">
          <span class="env-type-badge" :class="envType">{{ envTypeLabel }}</span>
          <button
            v-if="installLogs.length > 0"
            @click="dialogExpanded = !dialogExpanded"
            class="expand-btn"
            :title="dialogExpanded ? 'Collapse' : 'Expand to see full log'"
          >
            {{ dialogExpanded ? '⊟' : '⊞' }}
          </button>
          <button @click="$emit('close')" class="close-btn">&times;</button>
        </div>
      </div>

      <div class="env-details" v-if="envType === 'python'">
        <span class="detail">Python {{ environment.pythonVersion }}</span>
        <span class="detail">{{ packages.length }} packages</span>
      </div>
      <div class="env-details" v-else-if="envType === 'cpp'">
        <span class="detail">{{ packages.length }} packages</span>
        <span class="detail" v-if="environment.description">{{ environment.description }}</span>
      </div>

      <!-- Add Package Section -->
      <div class="add-package-section">
        <div class="package-search">
          <input
            v-model="packageSearch"
            :placeholder="searchPlaceholder"
            @input="searchPackages"
            @keydown.enter="addFirstResult"
          />
          <button
            v-if="packageSearch && !isSearching"
            @click="addPackageDirectly"
            class="add-direct-btn"
            title="Add package directly"
          >+</button>
        </div>
        <div class="search-results" v-if="searchResults.length > 0 && packageSearch.length >= 2">
          <div
            v-for="pkg in searchResults"
            :key="pkg.name"
            class="search-result"
            @click="addPackageFromSearch(pkg)"
          >
            <span class="pkg-name">{{ pkg.name }}</span>
            <span class="pkg-desc" v-if="pkg.description">{{ pkg.description }}</span>
            <span class="pkg-version" v-if="pkg.version">{{ pkg.version }}</span>
          </div>
        </div>
      </div>

      <!-- Packages to Install -->
      <div class="pending-packages" v-if="packagesToInstall.length > 0">
        <div class="pending-header">
          <span>Packages to install:</span>
          <button @click="installPackages" class="install-btn" :disabled="isInstalling">
            {{ isInstalling ? 'Installing...' : 'Install' }}
          </button>
        </div>
        <div class="pending-list">
          <div
            v-for="(pkg, index) in packagesToInstall"
            :key="pkg"
            class="pending-package"
          >
            <span>{{ pkg }}</span>
            <button @click="removePendingPackage(index)" class="remove-pkg-btn">&times;</button>
          </div>
        </div>
      </div>

      <!-- Install Log -->
      <div class="install-log" v-if="installLogs.length > 0">
        <div class="log-header">{{ installStatus }}</div>
        <div class="log-output" ref="logOutput">
          <div v-for="(log, i) in installLogs" :key="i" :class="['log-line', log.type]">
            {{ log.message }}
          </div>
        </div>
      </div>

      <!-- Installed Packages List -->
      <div class="packages-section">
        <div class="packages-header">
          <span>Installed Packages ({{ filteredPackages.length }})</span>
          <input
            v-model="filterText"
            placeholder="Filter..."
            class="filter-input"
          />
        </div>
        <div class="packages-list" v-if="!loading">
          <div
            v-for="pkg in filteredPackages"
            :key="pkg.name"
            class="package-item"
          >
            <div class="package-info">
              <span class="package-name">{{ pkg.name }}</span>
              <span class="package-version">{{ pkg.version }}</span>
            </div>
            <div class="package-actions">
              <span class="package-channel" v-if="pkg.channel">{{ pkg.channel }}</span>
              <button
                @click="removePackage(pkg.name)"
                class="remove-btn"
                title="Remove package"
                :disabled="isInstalling"
              >&times;</button>
            </div>
          </div>
        </div>
        <div class="loading" v-else>Loading packages...</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import axios from 'axios'
import {
  getActiveJob,
  setActiveJob,
  updateJobStatus,
  appendJobLog,
  clearActiveJob
} from '../stores/envJobs'

const props = defineProps({
  environment: { type: Object, required: true },
  envType: { type: String, required: true } // 'python' or 'cpp'
})

const emit = defineEmits(['close', 'updated'])

// Track active WebSocket connection
let activeWs = null

const packages = ref([])
const loading = ref(true)
const filterText = ref('')
const packageSearch = ref('')
const searchResults = ref([])
const packagesToInstall = ref([])
const isInstalling = ref(false)
const isSearching = ref(false)
const installStatus = ref('')
const installLogs = ref([])
const logOutput = ref(null)
const dialogExpanded = ref(false)

let searchTimeout = null

const envTypeLabel = computed(() => {
  return props.envType === 'python' ? 'Python' : 'C++'
})

const searchPlaceholder = computed(() => {
  return props.envType === 'python'
    ? 'Search conda/pip packages...'
    : 'Search debian dev packages...'
})

const filteredPackages = computed(() => {
  if (!filterText.value) return packages.value
  const filter = filterText.value.toLowerCase()
  return packages.value.filter(pkg =>
    pkg.name.toLowerCase().includes(filter) ||
    (pkg.version && pkg.version.toLowerCase().includes(filter))
  )
})

async function loadPackages() {
  loading.value = true
  try {
    if (props.envType === 'python') {
      const response = await axios.get(`/api/environments/${props.environment.name}/packages`)
      packages.value = response.data.packages || []
    } else {
      // C++ environment - packages are stored in the environment object
      packages.value = (props.environment.packages || []).map(name => ({
        name,
        version: 'installed',
        channel: 'apt'
      }))
    }
  } catch (error) {
    console.error('Error loading packages:', error)
    packages.value = []
  } finally {
    loading.value = false
  }
}

async function searchPackages() {
  if (packageSearch.value.length < 2) {
    searchResults.value = []
    return
  }

  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(async () => {
    isSearching.value = true
    try {
      if (props.envType === 'python') {
        // For Python, we don't have a search API, just allow direct input
        searchResults.value = []
      } else {
        // C++ - search debian packages
        const response = await axios.get('/api/debian-packages/search', {
          params: { q: packageSearch.value, limit: 20 }
        })
        searchResults.value = response.data.packages || []
      }
    } catch (error) {
      console.error('Error searching packages:', error)
      searchResults.value = []
    } finally {
      isSearching.value = false
    }
  }, 300)
}

function addPackageFromSearch(pkg) {
  if (!packagesToInstall.value.includes(pkg.name)) {
    packagesToInstall.value.push(pkg.name)
  }
  packageSearch.value = ''
  searchResults.value = []
}

function addPackageDirectly() {
  const pkgName = packageSearch.value.trim()
  if (pkgName && !packagesToInstall.value.includes(pkgName)) {
    packagesToInstall.value.push(pkgName)
  }
  packageSearch.value = ''
  searchResults.value = []
}

function addFirstResult() {
  if (searchResults.value.length > 0) {
    addPackageFromSearch(searchResults.value[0])
  } else if (packageSearch.value.trim()) {
    addPackageDirectly()
  }
}

function removePendingPackage(index) {
  packagesToInstall.value.splice(index, 1)
}

function appendTerminalOutput(text, logType = 'progress') {
  if (!text) return

  const lines = text.split(/\n/)

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i]

    if (line.includes('\r')) {
      const parts = line.split('\r')
      line = parts.filter(p => p.length > 0).pop() || ''
    }

    if (!line && i === lines.length - 1) continue

    line = line.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')

    if (installLogs.value.length > 0 && text.startsWith('\r')) {
      const lastLog = installLogs.value[installLogs.value.length - 1]
      if (lastLog.type === logType && line) {
        lastLog.message = line
        continue
      }
    }

    if (line) {
      installLogs.value.push({ type: logType, message: line })
    }
  }
}

async function installPackages() {
  if (packagesToInstall.value.length === 0 || isInstalling.value) return

  isInstalling.value = true
  installStatus.value = 'Installing packages...'
  installLogs.value = []

  try {
    let response
    if (props.envType === 'python') {
      response = await axios.post(`/api/environments/${props.environment.name}/packages`, {
        packages: packagesToInstall.value
      })
    } else {
      response = await axios.post(`/api/cpp-environments/${props.environment.name}/packages`, {
        packages: packagesToInstall.value
      })
    }

    if (response.data.error) {
      installStatus.value = `Error: ${response.data.error}`
      installLogs.value.push({ type: 'error', message: response.data.error })
      isInstalling.value = false
      return
    }

    const { jobId } = response.data
    installStatus.value = response.data.message || 'Installing packages...'

    // Store the active job so we can reconnect if dialog is closed
    setActiveJob(props.environment.name, {
      jobId,
      status: installStatus.value,
      logs: [],
      packages: [...packagesToInstall.value]
    })

    // Subscribe to job output via WebSocket
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`
    const ws = new WebSocket(wsUrl)
    activeWs = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'subscribe', jobId }))
    }

    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === 'subscribed') {
          installLogs.value.push({ type: 'progress', message: 'Connected to job stream...' })
          appendJobLog(props.environment.name, { type: 'progress', message: 'Connected to job stream...' })
        } else if (data.type === 'job_output') {
          const { stream, data: outputData } = data
          const streamLower = stream?.toLowerCase()

          if (streamLower === 'stdout' || streamLower === 'stderr') {
            const logType = streamLower === 'stderr' ? 'error' : 'progress'
            appendTerminalOutput(outputData, logType)
            // Also save to store for reconnection
            const lines = outputData.split('\n').filter(l => l.trim())
            lines.forEach(line => appendJobLog(props.environment.name, { type: logType, message: line }))
          } else if (streamLower === 'result') {
            installStatus.value = 'Packages installed successfully!'
            installLogs.value.push({ type: 'success', message: 'Packages installed successfully!' })

            packagesToInstall.value = []
            await loadPackages()
            emit('updated')

            isInstalling.value = false
            clearActiveJob(props.environment.name)
            ws.close()
          } else if (streamLower === 'error') {
            installStatus.value = `Error: ${outputData}`
            installLogs.value.push({ type: 'error', message: outputData })
            isInstalling.value = false
            clearActiveJob(props.environment.name)
            ws.close()
          } else if (streamLower === 'complete') {
            if (!installStatus.value.includes('successfully')) {
              installStatus.value = 'Job completed'
            }
            isInstalling.value = false
            clearActiveJob(props.environment.name)
            ws.close()
          }

          // Update status in store
          updateJobStatus(props.environment.name, installStatus.value)

          await nextTick()
          if (logOutput.value) {
            logOutput.value.scrollTop = logOutput.value.scrollHeight
          }
        } else if (data.type === 'error') {
          installStatus.value = `Error: ${data.error}`
          installLogs.value.push({ type: 'error', message: data.error })
          isInstalling.value = false
          clearActiveJob(props.environment.name)
          ws.close()
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', e)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      installStatus.value = 'WebSocket connection error'
      installLogs.value.push({ type: 'error', message: 'WebSocket connection error' })
      isInstalling.value = false
    }

    ws.onclose = () => {
      activeWs = null
      if (isInstalling.value && !installStatus.value.includes('successfully')) {
        // Don't clear the job - it might still be running, just disconnected
        isInstalling.value = false
      }
    }

  } catch (error) {
    installStatus.value = `Error: ${error.response?.data?.error || error.message}`
    installLogs.value.push({ type: 'error', message: error.response?.data?.error || error.message })
    isInstalling.value = false
  }
}

async function removePackage(packageName) {
  if (!confirm(`Remove package '${packageName}'?`)) return

  isInstalling.value = true
  installStatus.value = `Removing ${packageName}...`
  installLogs.value = []

  try {
    let response
    if (props.envType === 'python') {
      response = await axios.delete(`/api/environments/${props.environment.name}/packages/${packageName}`)
    } else {
      response = await axios.delete(`/api/cpp-environments/${props.environment.name}/packages/${packageName}`)
    }

    if (response.data.error) {
      installStatus.value = `Error: ${response.data.error}`
      installLogs.value.push({ type: 'error', message: response.data.error })
      isInstalling.value = false
      return
    }

    const { jobId } = response.data

    // Subscribe to job output via WebSocket
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'subscribe', jobId }))
    }

    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === 'job_output') {
          const { stream, data: outputData } = data
          const streamLower = stream?.toLowerCase()

          if (streamLower === 'stdout' || streamLower === 'stderr') {
            const logType = streamLower === 'stderr' ? 'error' : 'progress'
            appendTerminalOutput(outputData, logType)
          } else if (streamLower === 'result') {
            installStatus.value = 'Package removed successfully!'
            installLogs.value.push({ type: 'success', message: 'Package removed successfully!' })

            await loadPackages()
            emit('updated')

            isInstalling.value = false
            ws.close()
          } else if (streamLower === 'error') {
            installStatus.value = `Error: ${outputData}`
            installLogs.value.push({ type: 'error', message: outputData })
            isInstalling.value = false
            ws.close()
          } else if (streamLower === 'complete') {
            if (!installStatus.value.includes('successfully')) {
              installStatus.value = 'Job completed'
            }
            isInstalling.value = false
            ws.close()
          }

          await nextTick()
          if (logOutput.value) {
            logOutput.value.scrollTop = logOutput.value.scrollHeight
          }
        } else if (data.type === 'error') {
          installStatus.value = `Error: ${data.error}`
          isInstalling.value = false
          ws.close()
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', e)
      }
    }

    ws.onerror = () => {
      installStatus.value = 'WebSocket connection error'
      isInstalling.value = false
    }

    ws.onclose = () => {
      if (isInstalling.value) {
        isInstalling.value = false
      }
    }

  } catch (error) {
    installStatus.value = `Error: ${error.response?.data?.error || error.message}`
    isInstalling.value = false
  }
}

// Reconnect to an active job's WebSocket
function reconnectToJob(jobId) {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${wsProtocol}//${window.location.host}/ws`
  const ws = new WebSocket(wsUrl)
  activeWs = ws

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'subscribe', jobId }))
  }

  ws.onmessage = async (event) => {
    try {
      const data = JSON.parse(event.data)

      if (data.type === 'subscribed') {
        installLogs.value.push({ type: 'progress', message: 'Reconnected to job stream...' })
        appendJobLog(props.environment.name, { type: 'progress', message: 'Reconnected to job stream...' })
      } else if (data.type === 'job_output') {
        const { stream, data: outputData } = data
        const streamLower = stream?.toLowerCase()

        if (streamLower === 'stdout' || streamLower === 'stderr') {
          const logType = streamLower === 'stderr' ? 'error' : 'progress'
          appendTerminalOutput(outputData, logType)
          // Also save to store
          const lines = outputData.split('\n').filter(l => l.trim())
          lines.forEach(line => appendJobLog(props.environment.name, { type: logType, message: line }))
        } else if (streamLower === 'result') {
          installStatus.value = 'Packages installed successfully!'
          installLogs.value.push({ type: 'success', message: 'Packages installed successfully!' })

          packagesToInstall.value = []
          await loadPackages()
          emit('updated')

          isInstalling.value = false
          clearActiveJob(props.environment.name)
          ws.close()
        } else if (streamLower === 'error') {
          installStatus.value = `Error: ${outputData}`
          installLogs.value.push({ type: 'error', message: outputData })
          isInstalling.value = false
          clearActiveJob(props.environment.name)
          ws.close()
        } else if (streamLower === 'complete') {
          if (!installStatus.value.includes('successfully')) {
            installStatus.value = 'Job completed'
          }
          isInstalling.value = false
          clearActiveJob(props.environment.name)
          ws.close()
        }

        await nextTick()
        if (logOutput.value) {
          logOutput.value.scrollTop = logOutput.value.scrollHeight
        }
      } else if (data.type === 'error') {
        installStatus.value = `Error: ${data.error}`
        installLogs.value.push({ type: 'error', message: data.error })
        isInstalling.value = false
        clearActiveJob(props.environment.name)
        ws.close()
      }
    } catch (e) {
      console.error('Error parsing WebSocket message:', e)
    }
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    installStatus.value = 'WebSocket connection error'
    installLogs.value.push({ type: 'error', message: 'WebSocket connection error' })
    isInstalling.value = false
  }

  ws.onclose = () => {
    activeWs = null
    if (isInstalling.value && !installStatus.value.includes('successfully')) {
      isInstalling.value = false
    }
  }
}

onMounted(() => {
  loadPackages()

  // Check if there's an active job for this environment and reconnect
  const activeJob = getActiveJob(props.environment.name)
  if (activeJob) {
    isInstalling.value = true
    installStatus.value = activeJob.status
    installLogs.value = [...activeJob.logs]
    packagesToInstall.value = [...activeJob.packages]

    // Reconnect to the WebSocket
    reconnectToJob(activeJob.jobId)
  }
})

onUnmounted(() => {
  // Close WebSocket but don't clear the job - it's still running
  if (activeWs) {
    activeWs.close()
    activeWs = null
  }
})

watch(() => props.environment, () => {
  loadPackages()
})
</script>

<style scoped>
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
  width: 500px;
  max-width: 90%;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
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
  margin-bottom: 0.75rem;
}

.dialog-header h3 {
  margin: 0;
  color: var(--text-primary);
  font-size: 1.1rem;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.env-type-badge {
  font-size: 0.7rem;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-weight: 600;
}

.env-type-badge.python {
  background: rgba(0, 212, 255, 0.2);
  color: var(--accent);
  border: 1px solid rgba(0, 212, 255, 0.4);
}

.env-type-badge.cpp {
  background: rgba(245, 158, 11, 0.2);
  color: #f59e0b;
  border: 1px solid rgba(245, 158, 11, 0.4);
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

.close-btn {
  width: 28px;
  height: 28px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 1.2rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  color: var(--error);
}

.env-details {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
}

.detail {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.add-package-section {
  margin-bottom: 1rem;
}

.package-search {
  position: relative;
  display: flex;
  gap: 0.5rem;
}

.package-search input {
  flex: 1;
  padding: 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 0.85rem;
}

.package-search input:focus {
  outline: none;
  border-color: var(--accent);
}

.add-direct-btn {
  width: 36px;
  padding: 0;
  background: var(--accent);
  border: none;
  color: var(--bg-primary);
  border-radius: 4px;
  cursor: pointer;
  font-size: 1.2rem;
  font-weight: bold;
}

.add-direct-btn:hover {
  opacity: 0.9;
}

.search-results {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 4px 4px;
  max-height: 200px;
  overflow-y: auto;
  z-index: 10;
}

.search-result {
  padding: 0.5rem;
  cursor: pointer;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.search-result:last-child {
  border-bottom: none;
}

.search-result:hover {
  background: rgba(0, 212, 255, 0.1);
}

.pkg-name {
  font-size: 0.85rem;
  color: var(--text-primary);
  font-family: monospace;
}

.pkg-desc {
  font-size: 0.7rem;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pkg-version {
  font-size: 0.7rem;
  color: var(--accent);
}

.pending-packages {
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: rgba(0, 212, 255, 0.1);
  border: 1px solid rgba(0, 212, 255, 0.3);
  border-radius: 4px;
}

.pending-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.pending-header span {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.install-btn {
  padding: 0.35rem 0.75rem;
  font-size: 0.8rem;
  background: var(--accent);
  border: none;
  color: var(--bg-primary);
  border-radius: 4px;
  cursor: pointer;
  font-weight: 600;
}

.install-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.install-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.pending-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.pending-package {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  background: rgba(0, 212, 255, 0.2);
  border: 1px solid rgba(0, 212, 255, 0.4);
  border-radius: 4px;
  font-size: 0.8rem;
  color: var(--accent);
}

.remove-pkg-btn {
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  padding: 0;
  font-size: 0.9rem;
  line-height: 1;
}

.remove-pkg-btn:hover {
  color: var(--error);
}

.install-log {
  margin-bottom: 1rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  display: flex;
  flex-direction: column;
}

.log-header {
  padding: 0.5rem 0.75rem;
  background: var(--bg-primary);
  font-size: 0.8rem;
  color: var(--accent);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.log-output {
  height: 350px;
  min-height: 150px;
  max-height: 70vh;
  overflow-y: auto;
  overflow-x: auto;
  background: #0a0a12;
  font-family: 'Monaco', 'Menlo', 'Consolas', 'Liberation Mono', 'Courier New', monospace;
  font-size: 0.8rem;
  padding: 0.75rem;
  line-height: 1.5;
  resize: vertical;
  border-bottom-left-radius: 4px;
  border-bottom-right-radius: 4px;
}

.dialog.expanded .log-output {
  height: 500px;
}

.log-line {
  padding: 0.1rem 0;
  color: var(--text-secondary);
  white-space: pre;
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

.packages-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.packages-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.packages-header span {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-secondary);
}

.filter-input {
  width: 150px;
  padding: 0.3rem 0.5rem;
  font-size: 0.75rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
}

.filter-input:focus {
  outline: none;
  border-color: var(--accent);
}

.packages-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.package-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.4rem 0.5rem;
  background: var(--bg-primary);
  border-radius: 4px;
}

.package-item:hover {
  background: var(--bg-tertiary);
}

.package-info {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-width: 0;
}

.package-name {
  font-size: 0.8rem;
  color: var(--text-primary);
  font-family: monospace;
}

.package-version {
  font-size: 0.7rem;
  color: var(--text-secondary);
}

.package-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.package-channel {
  font-size: 0.65rem;
  padding: 0.15rem 0.4rem;
  background: var(--bg-tertiary);
  border-radius: 3px;
  color: var(--text-secondary);
}

.remove-btn {
  width: 20px;
  height: 20px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.9rem;
  line-height: 1;
  border-radius: 3px;
  opacity: 0;
  transition: opacity 0.15s;
}

.package-item:hover .remove-btn {
  opacity: 1;
}

.remove-btn:hover {
  background: var(--error);
  color: white;
}

.remove-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.loading {
  padding: 2rem;
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.85rem;
}
</style>
