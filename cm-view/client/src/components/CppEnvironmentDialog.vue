<template>
  <div class="dialog-overlay" @click.self="$emit('close')">
    <div class="dialog" :class="{ expanded: dialogExpanded }">
      <div class="dialog-header">
        <h3>Create C++ Environment</h3>
        <button
          v-if="createLogs.length > 0"
          @click="dialogExpanded = !dialogExpanded"
          class="btn-icon expand-btn"
          :title="dialogExpanded ? 'Collapse' : 'Expand to see full log'"
        >
          <Minimize2 v-if="dialogExpanded" :size="14" />
          <Maximize2 v-else :size="14" />
        </button>
      </div>

      <div class="form-group">
        <label>Environment Name:</label>
        <input
          v-model="envName"
          placeholder="my-cpp-env"
          @keydown.enter="createEnvironment"
        />
      </div>

      <div class="form-group">
        <label>Description (optional):</label>
        <input
          v-model="envDescription"
          placeholder="Environment for CUDA development"
        />
      </div>

      <div class="form-group">
        <label>Dev Packages:</label>
        <div class="package-search">
          <input
            v-model="packageSearch"
            placeholder="Search debian dev packages..."
            @input="searchPackages"
            @keydown.enter="addFirstResult"
          />
          <div class="search-results" v-if="searchResults.length > 0 && packageSearch.length >= 2">
            <div
              v-for="pkg in searchResults"
              :key="pkg.name"
              class="search-result"
              @click="addPackage(pkg)"
            >
              <span class="pkg-name">{{ pkg.name }}</span>
              <span class="pkg-desc">{{ pkg.description }}</span>
            </div>
          </div>
        </div>

        <div class="selected-packages" v-if="selectedPackages.length > 0">
          <div
            v-for="(pkg, index) in selectedPackages"
            :key="pkg"
            class="selected-package"
          >
            <span>{{ pkg }}</span>
            <button @click="removePackage(index)" class="btn-icon btn-danger remove-pkg-btn"><X :size="12" /></button>
          </div>
        </div>
        <span class="hint">Search and select packages to install</span>
      </div>

      <div class="dialog-actions">
        <button @click="$emit('close')" class="btn-secondary cancel-btn" :disabled="isCreating">Cancel</button>
        <button @click="createEnvironment" class="btn-primary create-btn" :disabled="!envName || isCreating">
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
</template>

<script setup>
import { ref, nextTick } from 'vue'
import axios from 'axios'
import { Minimize2, Maximize2, X } from 'lucide-vue-next'

const emit = defineEmits(['close', 'created'])

const envName = ref('')
const envDescription = ref('')
const packageSearch = ref('')
const searchResults = ref([])
const selectedPackages = ref([])
const createStatus = ref('')
const createLogs = ref([])
const isCreating = ref(false)
const logOutput = ref(null)
const dialogExpanded = ref(false)

let searchTimeout = null

async function searchPackages() {
  if (packageSearch.value.length < 2) {
    searchResults.value = []
    return
  }

  // Debounce search
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(async () => {
    try {
      const response = await axios.get('/api/debian-packages/search', {
        params: { q: packageSearch.value, limit: 20 }
      })
      searchResults.value = response.data.packages || []
    } catch (error) {
      console.error('Error searching packages:', error)
      searchResults.value = []
    }
  }, 300)
}

function addPackage(pkg) {
  if (!selectedPackages.value.includes(pkg.name)) {
    selectedPackages.value.push(pkg.name)
  }
  packageSearch.value = ''
  searchResults.value = []
}

function addFirstResult() {
  if (searchResults.value.length > 0) {
    addPackage(searchResults.value[0])
  }
}

function removePackage(index) {
  selectedPackages.value.splice(index, 1)
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
  if (!envName.value || isCreating.value) return

  isCreating.value = true
  createStatus.value = 'Starting...'
  createLogs.value = []

  try {
    const response = await axios.post('/api/cpp-environments', {
      name: envName.value,
      description: envDescription.value,
      packages: selectedPackages.value
    })

    if (response.data.error) {
      createStatus.value = `Error: ${response.data.error}`
      isCreating.value = false
      return
    }

    const { jobId } = response.data
    createStatus.value = response.data.message || `Creating C++ environment '${envName.value}'...`

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

        if (data.type === 'subscribed') {
          createLogs.value.push({ type: 'progress', message: 'Connected to job stream...' })
        } else if (data.type === 'job_output') {
          const { stream, data: outputData } = data
          const streamLower = stream?.toLowerCase()

          if (streamLower === 'stdout' || streamLower === 'stderr') {
            const logType = streamLower === 'stderr' ? 'error' : 'progress'
            appendTerminalOutput(outputData, logType)
          } else if (streamLower === 'result') {
            createStatus.value = `C++ environment '${envName.value}' created successfully!`
            createLogs.value.push({ type: 'success', message: `Environment '${envName.value}' created successfully!` })

            emit('created', envName.value)

            setTimeout(() => {
              emit('close')
            }, 1500)

            ws.close()
          } else if (streamLower === 'error') {
            createStatus.value = `Error: ${outputData}`
            createLogs.value.push({ type: 'error', message: outputData })
            isCreating.value = false
            ws.close()
          } else if (streamLower === 'complete') {
            if (!createStatus.value.includes('successfully')) {
              createStatus.value = 'Job completed'
            }
            isCreating.value = false
            ws.close()
          }

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
  width: 450px;
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
  background: var(--bg-primary);
  border: 1px solid var(--border);
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

.form-group input {
  width: 100%;
  padding: 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.form-group input:focus {
  outline: none;
  border-color: var(--accent);
}

.package-search {
  position: relative;
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

.selected-packages {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.selected-package {
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
  width: 18px;
  height: 18px;
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

.cancel-btn, .create-btn {
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
</style>
