<template>
  <div class="workspace-list">
    <div class="header">
      <h2>Workspaces</h2>
    </div>

    <div class="workspaces">
      <!-- Create New Workspace Tile -->
      <div
        class="workspace-card create-card"
        :class="{ editing: isCreating }"
        @click="!isCreating && startCreating()"
      >
        <template v-if="!isCreating">
          <div class="create-icon">+</div>
          <span class="create-text">New Workspace</span>
        </template>
        <template v-else>
          <input
            ref="nameInput"
            v-model="newWorkspaceName"
            type="text"
            placeholder="Workspace name..."
            class="name-input"
            @keyup.enter="createWorkspace"
            @keyup.escape="cancelCreating"
            @blur="handleInputBlur"
          />
          <div class="create-actions">
            <button @click.stop="createWorkspace" class="create-btn" :disabled="!newWorkspaceName.trim()">Create</button>
            <button @click.stop="cancelCreating" class="cancel-btn">Cancel</button>
          </div>
        </template>
      </div>

      <!-- Existing Workspaces -->
      <div
        v-for="workspace in workspaces"
        :key="workspace.id"
        class="workspace-card"
        @click="openWorkspace(workspace.id)"
      >
        <div class="workspace-header">
          <h3>{{ workspace.name }}</h3>
          <button @click.stop="deleteWorkspace(workspace)" class="delete-btn" title="Delete workspace">&times;</button>
        </div>

        <div class="workspace-meta">
          <!-- Environment Stats -->
          <div class="meta-row" v-if="workspace.stats">
            <span class="meta-item" v-if="workspace.stats.pythonEnvs > 0" title="Python environments">
              <span class="meta-icon">Py</span>
              <span class="meta-value">{{ workspace.stats.pythonEnvs }}</span>
            </span>
            <span class="meta-item" v-if="workspace.stats.cppEnvs > 0" title="C++ environments">
              <span class="meta-icon">C++</span>
              <span class="meta-value">{{ workspace.stats.cppEnvs }}</span>
            </span>
            <span class="meta-item" v-if="workspace.stats.vendorEnvs > 0" title="Vendor environments">
              <span class="meta-icon">Lib</span>
              <span class="meta-value">{{ workspace.stats.vendorEnvs }}</span>
            </span>
            <span class="meta-item" v-if="workspace.stats.files > 0" title="Files">
              <span class="meta-icon file-icon">&#128196;</span>
              <span class="meta-value">{{ workspace.stats.files }}</span>
            </span>
          </div>

          <!-- Updated timestamp -->
          <p class="updated">Updated {{ formatRelativeDate(workspace.updated_at) }}</p>
        </div>
      </div>

      <div v-if="workspaces.length === 0 && !isCreating" class="empty">
        <p>No workspaces yet. Create one to get started.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const workspaces = ref([])
const isCreating = ref(false)
const newWorkspaceName = ref('')
const nameInput = ref(null)

async function loadWorkspaces() {
  try {
    const response = await axios.get('/api/workspaces')
    workspaces.value = response.data
  } catch (error) {
    console.error('Error loading workspaces:', error)
  }
}

function startCreating() {
  isCreating.value = true
  newWorkspaceName.value = ''
  nextTick(() => {
    nameInput.value?.focus()
  })
}

function cancelCreating() {
  isCreating.value = false
  newWorkspaceName.value = ''
}

function handleInputBlur(event) {
  // Don't cancel if clicking on Create/Cancel buttons
  if (event.relatedTarget?.closest('.create-actions')) {
    return
  }
  // Small delay to allow button clicks to register
  setTimeout(() => {
    if (!newWorkspaceName.value.trim()) {
      cancelCreating()
    }
  }, 150)
}

async function createWorkspace() {
  const name = newWorkspaceName.value.trim()
  if (!name) return

  try {
    const response = await axios.post('/api/workspaces', {
      name,
      cells: []
    })
    isCreating.value = false
    newWorkspaceName.value = ''
    router.push(`/workspace/${response.data.id}`)
  } catch (error) {
    console.error('Error creating workspace:', error)
    alert(`Failed to create workspace: ${error.response?.data?.error || error.message}`)
  }
}

async function deleteWorkspace(workspace) {
  if (!confirm(`Delete workspace "${workspace.name}"? This cannot be undone.`)) {
    return
  }

  try {
    await axios.delete(`/api/workspaces/${workspace.id}`)
    await loadWorkspaces()
  } catch (error) {
    console.error('Error deleting workspace:', error)
    alert(`Failed to delete workspace: ${error.response?.data?.error || error.message}`)
  }
}

function openWorkspace(id) {
  router.push(`/workspace/${id}`)
}

function formatRelativeDate(dateStr) {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now - date
  const diffSecs = Math.floor(diffMs / 1000)
  const diffMins = Math.floor(diffSecs / 60)
  const diffHours = Math.floor(diffMins / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffSecs < 60) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`

  return date.toLocaleDateString()
}

onMounted(() => {
  loadWorkspaces()
})
</script>

<style scoped>
.workspace-list {
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.header h2 {
  font-size: 2rem;
  color: var(--accent);
}

.workspaces {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
}

.workspace-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  flex-direction: column;
  min-height: 140px;
}

.workspace-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
}

/* Create Card Styles */
.create-card {
  border-style: dashed;
  border-width: 2px;
  justify-content: center;
  align-items: center;
  gap: 0.75rem;
  color: var(--text-secondary);
}

.create-card:not(.editing):hover {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(0, 212, 255, 0.05);
}

.create-card.editing {
  cursor: default;
  align-items: stretch;
  justify-content: flex-start;
  padding: 1rem;
}

.create-icon {
  font-size: 2.5rem;
  font-weight: 300;
  line-height: 1;
}

.create-text {
  font-size: 0.95rem;
}

.name-input {
  width: 100%;
  padding: 0.75rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 1rem;
  margin-bottom: 1rem;
}

.name-input:focus {
  outline: none;
  border-color: var(--accent);
}

.name-input::placeholder {
  color: var(--text-secondary);
}

.create-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: auto;
}

.create-btn, .cancel-btn {
  flex: 1;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: opacity 0.2s;
}

.create-btn {
  background: var(--accent);
  color: var(--bg-primary);
}

.create-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.cancel-btn {
  background: var(--bg-primary);
  color: var(--text-secondary);
  border: 1px solid var(--border);
}

.cancel-btn:hover {
  border-color: var(--text-secondary);
}

/* Workspace Card Content */
.workspace-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.75rem;
}

.workspace-header h3 {
  color: var(--text-primary);
  font-size: 1.1rem;
  margin: 0;
  word-break: break-word;
}

.delete-btn {
  background: none;
  border: none;
  color: var(--text-secondary);
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0 0.25rem;
  line-height: 1;
  opacity: 0;
  transition: opacity 0.2s, color 0.2s;
}

.workspace-card:hover .delete-btn {
  opacity: 1;
}

.delete-btn:hover {
  color: #f87171;
}

.workspace-meta {
  margin-top: auto;
}

.meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  background: var(--bg-primary);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
}

.meta-icon {
  color: var(--accent);
  font-weight: 600;
  font-size: 0.7rem;
}

.meta-icon.file-icon {
  font-size: 0.85rem;
}

.meta-value {
  color: var(--text-secondary);
}

.updated {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin: 0;
}

.empty {
  grid-column: 1 / -1;
  text-align: center;
  padding: 3rem;
  color: var(--text-secondary);
}
</style>
