<template>
  <div class="dialog-overlay" @click.self="$emit('close')">
    <div class="dialog" :class="{ expanded: dialogExpanded }">
      <div class="dialog-header">
        <h3>Create Vendor Environment</h3>
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
          v-model="envName"
          placeholder="my-vendor-lib"
          @keydown.enter="createEnvironment"
        />
      </div>

      <div class="form-group">
        <label>Description (optional):</label>
        <input
          v-model="envDescription"
          placeholder="Custom build of library X"
        />
      </div>

      <div class="form-group">
        <label>Git Repository URL:</label>
        <input
          v-model="repoUrl"
          placeholder="https://github.com/user/repo.git"
        />
      </div>

      <div class="form-row">
        <div class="form-group half">
          <label>Branch:</label>
          <input
            v-model="branch"
            placeholder="main"
          />
        </div>

        <div class="form-group half">
          <label>Build System:</label>
          <select v-model="buildType">
            <option value="cmake">CMake</option>
            <option value="make">Make</option>
            <option value="autotools">Autotools</option>
            <option value="auto">Auto-detect</option>
          </select>
        </div>
      </div>

      <div class="form-group" v-if="buildType === 'cmake'">
        <label>CMake Options (optional):</label>
        <input
          v-model="cmakeOptions"
          placeholder="-DBUILD_SHARED_LIBS=ON -DENABLE_CUDA=ON"
        />
        <span class="hint">Additional CMake configuration options</span>
      </div>

      <div class="dialog-actions">
        <button @click="$emit('close')" class="cancel-btn" :disabled="isCreating">Cancel</button>
        <button @click="createEnvironment" class="create-btn" :disabled="!canCreate || isCreating">
          {{ isCreating ? 'Building...' : 'Build & Install' }}
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
import { ref, computed, nextTick } from 'vue'
import axios from 'axios'

const emit = defineEmits(['close', 'created'])

const envName = ref('')
const envDescription = ref('')
const repoUrl = ref('')
const branch = ref('main')
const buildType = ref('cmake')
const cmakeOptions = ref('')
const createStatus = ref('')
const createLogs = ref([])
const isCreating = ref(false)
const logOutput = ref(null)
const dialogExpanded = ref(false)

const canCreate = computed(() => {
  return envName.value && repoUrl.value
})

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
  if (!canCreate.value || isCreating.value) return

  isCreating.value = true
  createStatus.value = 'Starting...'
  createLogs.value = []

  try {
    const response = await axios.post('/api/vendor-environments', {
      name: envName.value,
      description: envDescription.value,
      repo: repoUrl.value,
      branch: branch.value || 'main',
      buildType: buildType.value,
      cmakeOptions: cmakeOptions.value
    })

    if (response.data.error) {
      createStatus.value = `Error: ${response.data.error}`
      isCreating.value = false
      return
    }

    const { jobId } = response.data
    createStatus.value = response.data.message || `Building '${envName.value}' from source...`

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
            createStatus.value = `Vendor environment '${envName.value}' created successfully!`
            createLogs.value.push({ type: 'success', message: `Environment '${envName.value}' built and installed!` })

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
              createStatus.value = 'Build completed'
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
  width: 500px;
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

.form-row {
  display: flex;
  gap: 1rem;
}

.form-group.half {
  flex: 1;
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
  max-height: 350px;
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
