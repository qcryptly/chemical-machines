<template>
  <div class="notebook-view" v-if="notebook">
    <div class="notebook-header">
      <input
        v-model="notebook.name"
        @blur="saveNotebook"
        class="notebook-title"
      />
      <button @click="addCell" class="btn-secondary"><Plus :size="12" /> Cell</button>
    </div>

    <div class="cells">
      <div
        v-for="(cell, index) in cells"
        :key="cell.id"
        class="cell"
      >
        <div class="cell-toolbar">
          <span class="cell-type">{{ cell.type }}</span>
          <button @click="executeCell(index)" v-if="cell.type === 'code'" class="btn-primary">Run</button>
          <button @click="deleteCell(index)" class="btn-danger delete">Delete</button>
        </div>

        <div class="cell-input">
          <textarea
            v-model="cell.content"
            @blur="saveNotebook"
            :placeholder="cell.type === 'code' ? 'Enter Python code...' : 'Enter markdown...'"
          />
        </div>

        <div class="cell-output" v-if="cell.output">
          <pre>{{ cell.output }}</pre>
        </div>

        <div class="cell-status" v-if="cell.status">
          <span :class="cell.status">{{ cell.status }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import { Plus } from 'lucide-vue-next'

const route = useRoute()
const notebook = ref(null)
const cells = ref([])
let ws = null

async function loadNotebook() {
  try {
    const response = await axios.get(`/api/notebooks/${route.params.id}`)
    notebook.value = response.data
    cells.value = JSON.parse(response.data.cells || '[]')

    if (cells.value.length === 0) {
      addCell()
    }
  } catch (error) {
    console.error('Error loading notebook:', error)
  }
}

async function saveNotebook() {
  try {
    await axios.put(`/api/notebooks/${route.params.id}`, {
      name: notebook.value.name,
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
  saveNotebook()
}

async function executeCell(index) {
  const cell = cells.value[index]
  cell.status = 'running'
  cell.output = null

  try {
    const response = await axios.post('/api/compute', {
      type: 'molecular-dynamics',
      params: {
        code: cell.content,
        steps: 1000
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
        cell.output = JSON.stringify(JSON.parse(job.result), null, 2)
        cell.status = 'completed'
        clearInterval(interval)
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

onMounted(() => {
  loadNotebook()
})
</script>

<style scoped>
.notebook-view {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.notebook-header {
  background: var(--bg-secondary);
  padding: 1rem 2rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.notebook-title {
  font-size: 1.5rem;
  font-weight: bold;
  background: transparent;
  border: none;
  color: var(--text-primary);
  flex: 1;
}

.cells {
  flex: 1;
  overflow-y: auto;
  padding: 2rem;
}

.cell {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 1rem;
  overflow: hidden;
}

.cell-toolbar {
  background: var(--bg-tertiary);
  padding: 0.5rem 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border);
}

.cell-type {
  color: var(--text-secondary);
  font-size: 0.85rem;
  text-transform: uppercase;
}

.cell-toolbar button {
  padding: 0.25rem 0.75rem;
}

.cell-input textarea {
  width: 100%;
  min-height: 100px;
  background: transparent;
  border: none;
  color: var(--text-primary);
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 0.9rem;
  padding: 1rem;
  resize: vertical;
}

.cell-output {
  border-top: 1px solid var(--border);
  padding: 1rem;
  background: var(--bg-primary);
}

.cell-output pre {
  margin: 0;
  color: var(--text-secondary);
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 0.85rem;
  white-space: pre-wrap;
}

.cell-status {
  padding: 0.5rem 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.85rem;
}

.cell-status .running {
  color: var(--accent);
}

.cell-status .completed {
  color: var(--success);
}

.cell-status .error {
  color: var(--error);
}
</style>
