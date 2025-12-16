<template>
  <div class="file-browser">
    <div class="browser-header">
      <span class="header-title">Files</span>
      <div class="header-actions">
        <button @click="createFile" title="New File" class="icon-btn">+</button>
        <button @click="createFolder" title="New Folder" class="icon-btn">📁</button>
        <button @click="refresh" title="Refresh" class="icon-btn">↻</button>
      </div>
    </div>

    <div
      class="file-tree"
      :class="{ 'drag-over-root': isDragOverRoot }"
      @contextmenu.prevent="handleTreeContextMenu"
      @dragover="handleRootDragOver"
      @dragleave="handleRootDragLeave"
      @drop="handleRootDrop"
    >
      <div v-if="loading" class="loading">Loading...</div>
      <div v-else-if="error" class="error">{{ error }}</div>
      <template v-else>
        <FileTreeNode
          v-for="item in files"
          :key="item.path"
          :item="item"
          :selected-path="selectedPath"
          @select="selectItem"
          @open="openFile"
          @delete="deleteItem"
          @rename="renameItem"
          @create-file="createFileIn"
          @create-folder="createFolderIn"
          @context-menu="handleContextMenu"
          @move="moveItem"
        />
      </template>
    </div>

    <!-- Context Menu -->
    <div
      v-if="contextMenu.show"
      class="context-menu"
      :style="{ top: contextMenu.y + 'px', left: contextMenu.x + 'px' }"
      @click.stop
    >
      <template v-if="contextMenu.item">
        <div v-if="contextMenu.item.type === 'file'" class="menu-item" @click="handleContextAction('open')">Open</div>
        <div class="menu-item" @click="handleContextAction('rename')">Rename</div>
        <div class="menu-divider"></div>
        <div class="menu-item" @click="handleContextAction('newFile')">New File{{ contextMenu.item.type === 'folder' ? ' Here' : '' }}</div>
        <div class="menu-item" @click="handleContextAction('newFolder')">New Folder{{ contextMenu.item.type === 'folder' ? ' Here' : '' }}</div>
        <div class="menu-divider"></div>
        <div class="menu-item danger" @click="handleContextAction('delete')">Delete</div>
      </template>
      <template v-else>
        <div class="menu-item" @click="handleContextAction('newFile')">New File</div>
        <div class="menu-item" @click="handleContextAction('newFolder')">New Folder</div>
      </template>
    </div>

    <!-- Input Dialog -->
    <div v-if="inputDialog.show" class="dialog-overlay" @click.self="cancelInput">
      <div class="input-dialog">
        <h4>{{ inputDialog.title }}</h4>
        <input
          v-model="inputDialog.value"
          @keydown.enter="confirmInput"
          @keydown.escape="cancelInput"
          ref="inputRef"
          :placeholder="inputDialog.placeholder"
        />
        <div class="dialog-buttons">
          <button @click="cancelInput" class="cancel-btn">Cancel</button>
          <button @click="confirmInput" class="confirm-btn">{{ inputDialog.confirmText }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import axios from 'axios'
import FileTreeNode from './FileTreeNode.vue'

const props = defineProps({
  workspaceId: { type: [String, Number], required: true }
})

const emit = defineEmits(['file-open', 'file-select'])

const files = ref([])
const loading = ref(true)
const error = ref(null)
const selectedPath = ref(null)
const inputRef = ref(null)
const isDragOverRoot = ref(false)

const contextMenu = ref({
  show: false,
  x: 0,
  y: 0,
  item: null
})

const inputDialog = ref({
  show: false,
  title: '',
  placeholder: '',
  value: '',
  confirmText: 'Create',
  action: null,
  parentPath: ''
})

// Helper to build workspace file API URL
function fileApiUrl(filePath = '') {
  const base = `/api/workspaces/${props.workspaceId}/files`
  return filePath ? `${base}/${encodeURIComponent(filePath)}` : base
}

async function refresh() {
  loading.value = true
  error.value = null
  try {
    const response = await axios.get(fileApiUrl())
    files.value = response.data.files || []
  } catch (err) {
    error.value = err.response?.data?.error || err.message
    files.value = []
  } finally {
    loading.value = false
  }
}

// Watch for workspace changes and refresh
watch(() => props.workspaceId, () => {
  refresh()
})

function selectItem(item) {
  selectedPath.value = item.path
  emit('file-select', item)
}

function openFile(item) {
  if (item.type === 'file') {
    emit('file-open', item)
  }
}

function showInputDialog(title, placeholder, confirmText, action, parentPath = '') {
  inputDialog.value = {
    show: true,
    title,
    placeholder,
    value: '',
    confirmText,
    action,
    parentPath
  }
  nextTick(() => {
    inputRef.value?.focus()
  })
}

function createFile() {
  showInputDialog('New File', 'filename.py', 'Create', 'createFile', '')
}

function createFolder() {
  showInputDialog('New Folder', 'folder-name', 'Create', 'createFolder', '')
}

function createFileIn(parentPath) {
  showInputDialog('New File', 'filename.py', 'Create', 'createFile', parentPath)
}

function createFolderIn(parentPath) {
  showInputDialog('New Folder', 'folder-name', 'Create', 'createFolder', parentPath)
}

function renameItem(item) {
  inputDialog.value = {
    show: true,
    title: 'Rename',
    placeholder: item.name,
    value: item.name,
    confirmText: 'Rename',
    action: 'rename',
    item
  }
  nextTick(() => {
    inputRef.value?.focus()
    inputRef.value?.select()
  })
}

async function deleteItem(item) {
  if (!confirm(`Delete ${item.type === 'folder' ? 'folder' : 'file'} "${item.name}"?`)) {
    return
  }

  try {
    await axios.delete(fileApiUrl(item.path))
    await refresh()
  } catch (err) {
    alert(`Error deleting: ${err.response?.data?.error || err.message}`)
  }
}

async function confirmInput() {
  const { action, value, parentPath, item } = inputDialog.value

  if (!value.trim()) {
    cancelInput()
    return
  }

  try {
    if (action === 'createFile') {
      const filePath = parentPath ? `${parentPath}/${value}` : value
      await axios.post(fileApiUrl(), { path: filePath, type: 'file', content: '' })
    } else if (action === 'createFolder') {
      const filePath = parentPath ? `${parentPath}/${value}` : value
      await axios.post(fileApiUrl(), { path: filePath, type: 'folder' })
    } else if (action === 'rename') {
      const parentDir = item.path.includes('/') ? item.path.substring(0, item.path.lastIndexOf('/')) : ''
      const newPath = parentDir ? `${parentDir}/${value}` : value
      await axios.put(fileApiUrl(item.path), { newPath })
    }

    await refresh()
  } catch (err) {
    alert(`Error: ${err.response?.data?.error || err.message}`)
  }

  cancelInput()
}

function cancelInput() {
  inputDialog.value.show = false
}

function handleContextAction(action) {
  const item = contextMenu.value.item
  contextMenu.value.show = false

  switch (action) {
    case 'open':
      if (item) openFile(item)
      break
    case 'rename':
      if (item) renameItem(item)
      break
    case 'newFile':
      createFileIn(item?.type === 'folder' ? item.path : '')
      break
    case 'newFolder':
      createFolderIn(item?.type === 'folder' ? item.path : '')
      break
    case 'delete':
      if (item) deleteItem(item)
      break
  }
}

function handleContextMenu({ item, x, y }) {
  contextMenu.value = {
    show: true,
    x,
    y,
    item
  }
}

function handleTreeContextMenu(event) {
  // Only show if clicking on empty space (not on a file/folder node)
  if (event.target.closest('.node-row')) return

  contextMenu.value = {
    show: true,
    x: event.clientX,
    y: event.clientY,
    item: null  // null means root level
  }
}

function hideContextMenu() {
  contextMenu.value.show = false
}

// Move file/folder to a new location
async function moveItem({ sourcePath, sourceName, targetFolder }) {
  const newPath = targetFolder ? `${targetFolder}/${sourceName}` : sourceName

  // Don't move if already in the same location
  if (sourcePath === newPath) return

  try {
    await axios.put(fileApiUrl(sourcePath), { newPath })
    await refresh()
  } catch (err) {
    alert(`Error moving: ${err.response?.data?.error || err.message}`)
  }
}

// Root level drag handlers (for moving to root)
function handleRootDragOver(event) {
  // Only handle if not over a tree node
  if (event.target.closest('.node-row')) {
    isDragOverRoot.value = false
    return
  }
  event.preventDefault()
  event.dataTransfer.dropEffect = 'move'
  isDragOverRoot.value = true
}

function handleRootDragLeave(event) {
  // Only clear if leaving the file-tree entirely
  if (!event.currentTarget.contains(event.relatedTarget)) {
    isDragOverRoot.value = false
  }
}

function handleRootDrop(event) {
  // Only handle if not over a tree node (folders handle their own drops)
  if (event.target.closest('.node-row')) {
    isDragOverRoot.value = false
    return
  }

  event.preventDefault()
  isDragOverRoot.value = false

  try {
    const data = JSON.parse(event.dataTransfer.getData('application/json'))

    // Move to root (empty target folder)
    moveItem({
      sourcePath: data.path,
      sourceName: data.name,
      targetFolder: ''
    })
  } catch (err) {
    console.error('Error parsing drag data:', err)
  }
}

onMounted(() => {
  refresh()
  document.addEventListener('click', hideContextMenu)
})

onUnmounted(() => {
  document.removeEventListener('click', hideContextMenu)
})
</script>

<style scoped>
.file-browser {
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}

.browser-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border);
}

.header-title {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--accent);
}

.header-actions {
  display: flex;
  gap: 0.25rem;
}

.icon-btn {
  width: 22px;
  height: 22px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  border-radius: 3px;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-btn:hover {
  background: var(--bg-primary);
  color: var(--accent);
}

.file-tree {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
}

.file-tree.drag-over-root {
  background: rgba(0, 212, 255, 0.1);
  outline: 2px dashed var(--accent);
  outline-offset: -4px;
}

.loading, .error {
  padding: 1rem;
  text-align: center;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.error {
  color: var(--error);
}

.context-menu {
  position: fixed;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.25rem 0;
  min-width: 120px;
  z-index: 1000;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.menu-item {
  padding: 0.4rem 0.75rem;
  font-size: 0.8rem;
  color: var(--text-primary);
  cursor: pointer;
}

.menu-item:hover {
  background: var(--accent);
  color: var(--bg-primary);
}

.menu-item.danger {
  color: var(--error);
}

.menu-item.danger:hover {
  background: var(--error);
  color: white;
}

.menu-divider {
  height: 1px;
  background: var(--border);
  margin: 0.25rem 0;
}

.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1001;
}

.input-dialog {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  width: 280px;
}

.input-dialog h4 {
  margin: 0 0 0.75rem 0;
  font-size: 0.9rem;
  color: var(--text-primary);
}

.input-dialog input {
  width: 100%;
  padding: 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 0.85rem;
  margin-bottom: 0.75rem;
}

.input-dialog input:focus {
  outline: none;
  border-color: var(--accent);
}

.dialog-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
}

.cancel-btn, .confirm-btn {
  padding: 0.4rem 0.75rem;
  font-size: 0.8rem;
  border-radius: 4px;
  cursor: pointer;
}

.cancel-btn {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-secondary);
}

.cancel-btn:hover {
  border-color: var(--text-secondary);
}

.confirm-btn {
  background: var(--accent);
  border: none;
  color: var(--bg-primary);
}

.confirm-btn:hover {
  opacity: 0.9;
}
</style>
