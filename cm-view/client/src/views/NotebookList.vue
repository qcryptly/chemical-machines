<template>
  <div class="notebook-list">
    <div class="header">
      <h2>Notebooks</h2>
      <button @click="createNotebook">+ New Notebook</button>
    </div>

    <div class="notebooks">
      <div
        v-for="notebook in notebooks"
        :key="notebook.id"
        class="notebook-card"
        @click="openNotebook(notebook.id)"
      >
        <h3>{{ notebook.name }}</h3>
        <p class="meta">
          Updated: {{ formatDate(notebook.updated_at) }}
        </p>
      </div>

      <div v-if="notebooks.length === 0" class="empty">
        <p>No notebooks yet. Create one to get started.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const notebooks = ref([])

async function loadNotebooks() {
  try {
    const response = await axios.get('/api/notebooks')
    notebooks.value = response.data
  } catch (error) {
    console.error('Error loading notebooks:', error)
  }
}

async function createNotebook() {
  const name = prompt('Notebook name:')
  if (!name) return

  try {
    const response = await axios.post('/api/notebooks', {
      name,
      cells: []
    })
    router.push(`/notebook/${response.data.id}`)
  } catch (error) {
    console.error('Error creating notebook:', error)
  }
}

function openNotebook(id) {
  router.push(`/notebook/${id}`)
}

function formatDate(dateStr) {
  const date = new Date(dateStr)
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString()
}

onMounted(() => {
  loadNotebooks()
})
</script>

<style scoped>
.notebook-list {
  padding: 2rem;
  max-width: 1200px;
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

.notebooks {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.notebook-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.2s;
}

.notebook-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
}

.notebook-card h3 {
  margin-bottom: 0.5rem;
  color: var(--text-primary);
}

.notebook-card .meta {
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.empty {
  grid-column: 1 / -1;
  text-align: center;
  padding: 3rem;
  color: var(--text-secondary);
}
</style>
