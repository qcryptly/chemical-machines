<template>
  <div class="tree-node">
    <div
      class="node-row"
      :class="{
        selected: selectedPaths.has(item.path),
        folder: item.type === 'folder',
        'drag-over': isDragOver && item.type === 'folder'
      }"
      draggable="true"
      @click="handleClick"
      @dblclick="handleDoubleClick"
      @contextmenu.prevent="showContextMenu"
      @dragstart="handleDragStart"
      @dragend="handleDragEnd"
      @dragover="handleDragOver"
      @dragleave="handleDragLeave"
      @drop="handleDrop"
    >
      <span
        v-if="item.type === 'folder'"
        class="expand-icon"
        @click.stop="toggleExpand"
      >
        {{ expanded ? '▼' : '▶' }}
      </span>
      <span v-else class="file-spacer"></span>

      <span class="node-icon">{{ getIcon(item) }}</span>
      <span class="node-name">{{ item.name }}</span>
    </div>

    <div v-if="item.type === 'folder' && expanded" class="children">
      <div v-if="loading" class="loading-indicator">Loading...</div>
      <div v-else-if="loadError" class="error-indicator">{{ loadError }}</div>
      <template v-else-if="children && children.length > 0">
        <FileTreeNode
          v-for="child in children"
          :key="child.path"
          :item="child"
          :workspace-id="workspaceId"
          :selected-path="selectedPath"
          :selected-paths="selectedPaths"
          @select="(item, event) => $emit('select', item, event)"
          @open="$emit('open', $event)"
          @delete="$emit('delete', $event)"
          @rename="$emit('rename', $event)"
          @create-file="$emit('create-file', $event)"
          @create-folder="$emit('create-folder', $event)"
          @context-menu="$emit('context-menu', $event)"
          @move="$emit('move', $event)"
        />
        <div v-if="hasMore" class="load-more">
          <button @click="loadMore" :disabled="loadingMore">
            {{ loadingMore ? 'Loading...' : `Load more (${totalCount - children.length} remaining)` }}
          </button>
        </div>
      </template>
      <div v-else-if="item.childCount === 0" class="empty-folder">Empty folder</div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import axios from 'axios'

const props = defineProps({
  item: { type: Object, required: true },
  workspaceId: { type: [String, Number], required: true },
  selectedPath: { type: String, default: null },
  selectedPaths: { type: Set, default: () => new Set() }
})

const emit = defineEmits(['select', 'open', 'delete', 'rename', 'create-file', 'create-folder', 'context-menu', 'move'])

const expanded = ref(false)
const isDragOver = ref(false)
const children = ref(props.item.children || null)
const loading = ref(false)
const loadError = ref(null)
const totalCount = ref(0)
const hasMore = ref(false)
const loadingMore = ref(false)

// Watch for expansion to lazy-load children
watch(expanded, async (isExpanded) => {
  if (isExpanded && props.item.type === 'folder' && !children.value) {
    await loadChildren()
  }
})

async function loadChildren(offset = 0) {
  if (offset === 0) {
    loading.value = true
  } else {
    loadingMore.value = true
  }

  loadError.value = null

  try {
    const url = `/api/workspaces/${props.workspaceId}/files/${encodeURIComponent(props.item.path)}`
    const response = await axios.get(url, {
      params: {
        depth: 0,  // Don't load grandchildren
        limit: 1000,
        offset
      }
    })

    if (offset === 0) {
      children.value = response.data.items || []
    } else {
      children.value = [...children.value, ...(response.data.items || [])]
    }

    totalCount.value = response.data.totalCount || 0
    hasMore.value = response.data.hasMore || false
  } catch (error) {
    loadError.value = error.response?.data?.error || error.message
  } finally {
    loading.value = false
    loadingMore.value = false
  }
}

function loadMore() {
  if (children.value) {
    loadChildren(children.value.length)
  }
}

function getIcon(item) {
  if (item.type === 'folder') {
    return expanded.value ? '📂' : '📁'
  }

  const ext = item.name.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'py':
      return '🐍'  // snake for python
    case 'cpp':
    case 'c':
    case 'h':
    case 'hpp':
      return '⚙'  // gear
    case 'sh':
    case 'bash':
      return '💻'  // terminal
    case 'json':
      return '{}'
    case 'md':
      return '📄'
    default:
      return '📄'  // document
  }
}

function toggleExpand() {
  expanded.value = !expanded.value
}

function handleClick(event) {
  emit('select', props.item, {
    shiftKey: event.shiftKey,
    ctrlKey: event.ctrlKey,
    metaKey: event.metaKey
  })
  if (props.item.type === 'folder' && !event.shiftKey && !event.ctrlKey && !event.metaKey) {
    expanded.value = !expanded.value
  } else if (props.item.type === 'file' && !event.shiftKey && !event.ctrlKey && !event.metaKey) {
    // Open files on single click only when not multi-selecting
    emit('open', props.item)
  }
}

function handleDoubleClick() {
  emit('open', props.item)
}

function showContextMenu(event) {
  // Only emit select if the item is not already selected or there's only one item selected
  // This preserves multi-selection when right-clicking on an already-selected item
  if (!props.selectedPaths.has(props.item.path) || props.selectedPaths.size === 1) {
    emit('select', props.item)
  }
  emit('context-menu', { item: props.item, x: event.clientX, y: event.clientY })
}

// Drag and drop handlers
function handleDragStart(event) {
  event.dataTransfer.setData('application/json', JSON.stringify({
    path: props.item.path,
    name: props.item.name,
    type: props.item.type
  }))
  event.dataTransfer.effectAllowed = 'move'
}

function handleDragEnd() {
  isDragOver.value = false
}

function handleDragOver(event) {
  // Only allow dropping on folders
  if (props.item.type !== 'folder') return

  event.preventDefault()
  event.dataTransfer.dropEffect = 'move'
  isDragOver.value = true
}

function handleDragLeave() {
  isDragOver.value = false
}

function handleDrop(event) {
  event.preventDefault()
  isDragOver.value = false

  // Only allow dropping on folders
  if (props.item.type !== 'folder') return

  try {
    const data = JSON.parse(event.dataTransfer.getData('application/json'))

    // Don't allow dropping on itself or into its own children
    if (data.path === props.item.path) return
    if (props.item.path.startsWith(data.path + '/')) return

    // Emit move event with source and target
    emit('move', {
      sourcePath: data.path,
      sourceName: data.name,
      targetFolder: props.item.path
    })
  } catch (err) {
    console.error('Error parsing drag data:', err)
  }
}
</script>

<style scoped>
.tree-node {
  user-select: none;
}

.node-row {
  display: flex;
  align-items: center;
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  cursor: pointer;
  font-size: 0.8rem;
  color: var(--text-primary);
}

.node-row:hover {
  background: var(--bg-primary);
}

.node-row.selected {
  background: rgba(0, 212, 255, 0.15);
}

.node-row.drag-over {
  background: rgba(0, 212, 255, 0.3);
  outline: 2px dashed var(--accent);
  outline-offset: -2px;
}

.expand-icon {
  width: 14px;
  font-size: 0.6rem;
  color: var(--text-secondary);
  flex-shrink: 0;
}

.file-spacer {
  width: 14px;
  flex-shrink: 0;
}

.node-icon {
  margin-right: 0.4rem;
  font-size: 0.85rem;
}

.node-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.children {
  padding-left: 1rem;
}

.loading-indicator, .error-indicator, .empty-folder {
  padding: 0.5rem;
  font-size: 0.75rem;
  color: var(--text-secondary);
  font-style: italic;
}

.error-indicator {
  color: var(--error);
}

.load-more {
  padding: 0.5rem;
}

.load-more button {
  width: 100%;
  padding: 0.4rem;
  background: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.75rem;
  transition: background 0.2s;
}

.load-more button:hover:not(:disabled) {
  background: var(--bg-tertiary);
}

.load-more button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
