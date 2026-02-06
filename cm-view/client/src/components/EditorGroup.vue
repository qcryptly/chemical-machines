<template>
  <div class="editor-group" :class="{ focused: isFocused }" @click="$emit('focus')">
    <!-- Tab Bar -->
    <div class="group-tab-bar" v-if="group.tabs.length > 0" @dragover.prevent @drop="onTabBarDrop">
      <div
        v-for="(tab, index) in tabsWithDocs"
        :key="tab.path"
        class="tab"
        :class="{
          active: index === group.activeTabIndex,
          dirty: tab.isDirty,
          dragging: index === dragTabIndex,
          'drag-over-left': index === dragOverTabIndex && dragInsertSide === 'left',
          'drag-over-right': index === dragOverTabIndex && dragInsertSide === 'right'
        }"
        draggable="true"
        @dragstart="onTabDragStart($event, index)"
        @dragover="onTabDragOver($event, index)"
        @dragleave="onTabDragLeave($event, index)"
        @drop.stop="onTabDrop($event, index)"
        @dragend="onTabDragEnd($event)"
        @click.stop="$emit('switch-tab', index)"
        :title="tab.path"
      >
        <span class="tab-icon"><component :is="getFileIcon(tab.name)" :size="12" /></span>
        <span class="tab-name">{{ tab.name }}</span>
        <span v-if="tab.isDirty" class="tab-dirty-indicator">●</span>
        <button class="tab-close" @click.stop="$emit('close-tab', index)" title="Close"><X :size="12" /></button>
      </div>
      <div class="tab-bar-spacer"></div>
      <button class="btn-icon split-btn" @click.stop="$emit('split', 'horizontal')" title="Split right">
        <Columns2 :size="12" />
      </button>
      <button class="btn-icon split-btn" @click.stop="$emit('split', 'vertical')" title="Split down">
        <Rows2 :size="12" />
      </button>
    </div>

    <!-- Drop zone for empty groups -->
    <div
      v-if="group.tabs.length === 0"
      class="empty-drop-zone"
      :class="{ 'drag-over': emptyDropActive }"
      @dragover="onEmptyDragOver"
      @dragleave="emptyDropActive = false"
      @drop="onEmptyDrop"
    >
      <p>Drop a tab here or open a file</p>
    </div>

    <!-- Editor Content -->
    <div class="group-editor-content" v-if="activeTab" @dragover.prevent @drop="onContentDrop">
      <!-- Markdown Preview -->
      <div
        class="markdown-preview"
        v-if="activeTab.isMarkdown && activeTab.previewMode"
        v-html="renderedMarkdown"
        ref="markdownPreviewRef"
        @click="handleMarkdownClick"
      ></div>

      <!-- Code Cells -->
      <div class="cells" v-else-if="activeTab.cells.length > 0">
        <CodeCell
          v-for="(cell, index) in activeTab.cells"
          :key="cell.id"
          :cell="cell"
          :index="index"
          :language="activeTab.language"
          :html-output="(activeTab.htmlOutputs || [])[index] || ''"
          @update="$emit('update-cell', { cellIndex: index, data: $event })"
          @run="$emit('run-cell', index)"
          @delete="$emit('delete-cell', index)"
          @create-below="$emit('create-cell-below', index)"
          @reorder="$emit('reorder-cells', $event)"
          @interrupt="$emit('interrupt-cell', index)"
        />
      </div>

      <!-- No content -->
      <div class="no-file-message" v-else>
        <p>Open a file from the Files sidebar to start editing</p>
      </div>
    </div>

    <div class="no-file-message" v-else-if="group.tabs.length > 0">
      <p>Select a tab to start editing</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import CodeCell from './CodeCell.vue'
import { marked } from 'marked'
import {
  X, Columns2, Rows2, FileCode, Settings,
  Terminal as TerminalIcon, Braces, FileText, File as LucideFile
} from 'lucide-vue-next'
import * as documentStore from '../stores/documentStore.js'

const props = defineProps({
  group: { type: Object, required: true },
  groupId: { type: Number, required: true },
  isFocused: { type: Boolean, default: false }
})

const emit = defineEmits([
  'focus', 'switch-tab', 'close-tab', 'split',
  'update-cell', 'run-cell', 'delete-cell', 'create-cell-below',
  'reorder-cells', 'interrupt-cell',
  'tab-drop-from-group'
])

const markdownPreviewRef = ref(null)

// Helper to get merged tab with document data
function getTabWithDocument(tab) {
  if (!tab) return null
  const doc = documentStore.getDocument(tab.path)
  return {
    ...tab,
    // Merge document properties (these override tab props if present)
    cells: doc?.cells || [],
    isDirty: doc?.isDirty || false,
    language: doc?.language || 'python',
    useCells: doc?.useCells !== false,
    isMarkdown: doc?.isMarkdown || false
  }
}

// Get all tabs with document data merged (for tab bar display)
const tabsWithDocs = computed(() => {
  return props.group.tabs.map(tab => getTabWithDocument(tab))
})

// Active tab with document data merged
const activeTab = computed(() => {
  const g = props.group
  if (g.activeTabIndex >= 0 && g.activeTabIndex < g.tabs.length) {
    return getTabWithDocument(g.tabs[g.activeTabIndex])
  }
  return null
})

// Rendered markdown
const renderedMarkdown = computed(() => {
  const tab = activeTab.value
  if (!tab || !tab.isMarkdown || !tab.previewMode) return ''
  return marked(tab.cells[0]?.content || '')
})

// ================== Tab Drag-to-Reorder ==================

const dragTabIndex = ref(-1)
const dragOverTabIndex = ref(-1)
const dragInsertSide = ref(null)
const emptyDropActive = ref(false)

function onTabDragStart(event, index) {
  dragTabIndex.value = index
  event.dataTransfer.effectAllowed = 'move'
  event.dataTransfer.setData('application/x-tab-drag', JSON.stringify({
    sourceGroupId: props.groupId,
    sourceTab: index
  }))
  event.target.style.opacity = '0.4'
}

function onTabDragOver(event, index) {
  event.preventDefault()
  event.dataTransfer.dropEffect = 'move'
  dragOverTabIndex.value = index
  const rect = event.currentTarget.getBoundingClientRect()
  const midpoint = rect.left + rect.width / 2
  dragInsertSide.value = event.clientX < midpoint ? 'left' : 'right'
}

function onTabDragLeave(event, index) {
  if (dragOverTabIndex.value === index) {
    dragOverTabIndex.value = -1
    dragInsertSide.value = null
  }
}

function onTabDrop(event, targetIndex) {
  event.preventDefault()
  const raw = event.dataTransfer.getData('application/x-tab-drag')
  if (!raw) {
    resetDragState()
    return
  }
  const { sourceGroupId, sourceTab } = JSON.parse(raw)

  let effectiveTarget = targetIndex
  if (dragInsertSide.value === 'right') effectiveTarget += 1

  if (sourceGroupId === props.groupId) {
    // Same group: reorder
    let fromIndex = sourceTab
    let toIndex = effectiveTarget
    if (fromIndex < toIndex) toIndex -= 1
    if (fromIndex === toIndex) {
      resetDragState()
      return
    }
    const [movedTab] = props.group.tabs.splice(fromIndex, 1)
    props.group.tabs.splice(toIndex, 0, movedTab)
    if (props.group.activeTabIndex === fromIndex) {
      props.group.activeTabIndex = toIndex
    } else if (fromIndex < props.group.activeTabIndex && toIndex >= props.group.activeTabIndex) {
      props.group.activeTabIndex -= 1
    } else if (fromIndex > props.group.activeTabIndex && toIndex <= props.group.activeTabIndex) {
      props.group.activeTabIndex += 1
    }
  } else {
    // Cross-group: emit event for parent to handle
    emit('tab-drop-from-group', {
      fromGroupId: sourceGroupId,
      fromTab: sourceTab,
      toGroupId: props.groupId,
      toTab: effectiveTarget
    })
  }
  resetDragState()
}

function onTabDragEnd(event) {
  event.target.style.opacity = ''
  resetDragState()
}

function resetDragState() {
  dragTabIndex.value = -1
  dragOverTabIndex.value = -1
  dragInsertSide.value = null
}

// Empty group drop zone
function onEmptyDragOver(event) {
  event.preventDefault()
  event.dataTransfer.dropEffect = 'move'
  emptyDropActive.value = true
}

function onEmptyDrop(event) {
  event.preventDefault()
  emptyDropActive.value = false
  const raw = event.dataTransfer.getData('application/x-tab-drag')
  if (!raw) return
  const { sourceGroupId, sourceTab } = JSON.parse(raw)
  if (sourceGroupId !== props.groupId) {
    emit('tab-drop-from-group', {
      fromGroupId: sourceGroupId,
      fromTab: sourceTab,
      toGroupId: props.groupId,
      toTab: 0
    })
  }
}

// Drop on editor content area (cross-group only)
function onContentDrop(event) {
  const raw = event.dataTransfer.getData('application/x-tab-drag')
  if (!raw) return
  event.preventDefault()
  const { sourceGroupId, sourceTab } = JSON.parse(raw)
  if (sourceGroupId !== props.groupId) {
    emit('tab-drop-from-group', {
      fromGroupId: sourceGroupId,
      fromTab: sourceTab,
      toGroupId: props.groupId,
      toTab: props.group.tabs.length
    })
  }
  resetDragState()
}

// Tab bar background drop (when not dropping on a specific tab)
function onTabBarDrop(event) {
  event.preventDefault()
  const raw = event.dataTransfer.getData('application/x-tab-drag')
  if (!raw) return
  const { sourceGroupId, sourceTab } = JSON.parse(raw)
  if (sourceGroupId !== props.groupId) {
    emit('tab-drop-from-group', {
      fromGroupId: sourceGroupId,
      fromTab: sourceTab,
      toGroupId: props.groupId,
      toTab: props.group.tabs.length
    })
  } else {
    // Same group: move to end
    const fromIndex = sourceTab
    const toIndex = props.group.tabs.length - 1
    if (fromIndex === toIndex) {
      resetDragState()
      return
    }
    const [movedTab] = props.group.tabs.splice(fromIndex, 1)
    props.group.tabs.splice(toIndex, 0, movedTab)
    if (props.group.activeTabIndex === fromIndex) {
      props.group.activeTabIndex = toIndex
    } else if (fromIndex < props.group.activeTabIndex && toIndex >= props.group.activeTabIndex) {
      props.group.activeTabIndex -= 1
    } else if (fromIndex > props.group.activeTabIndex && toIndex <= props.group.activeTabIndex) {
      props.group.activeTabIndex += 1
    }
  }
  resetDragState()
}

// File icon mapping
function getFileIcon(filename) {
  const ext = filename.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'py': return FileCode
    case 'cpp':
    case 'c':
    case 'h':
    case 'hpp': return Settings
    case 'sh':
    case 'bash': return TerminalIcon
    case 'json': return Braces
    case 'md': return FileText
    default: return LucideFile
  }
}

// Markdown anchor click handler
function handleMarkdownClick(event) {
  const target = event.target.closest('a')
  if (!target) return
  const href = target.getAttribute('href')
  if (!href) return
  if (href.startsWith('#')) {
    event.preventDefault()
    const id = href.slice(1)
    const element = markdownPreviewRef.value?.querySelector(`[id="${id}"], [name="${id}"]`)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  } else if (href.startsWith('http://') || href.startsWith('https://')) {
    event.preventDefault()
    window.open(href, '_blank', 'noopener,noreferrer')
  }
}
</script>

<style scoped>
.editor-group {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
  min-width: 200px;
  min-height: 100px;
  border: 1px solid transparent;
}

.editor-group.focused {
  border-color: rgba(0, 212, 255, 0.2);
}

/* Tab Bar */
.group-tab-bar {
  display: flex;
  align-items: center;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border);
  overflow-x: auto;
  flex-shrink: 0;
}

.group-tab-bar::-webkit-scrollbar {
  height: 4px;
}

.group-tab-bar::-webkit-scrollbar-thumb {
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

.tab.dragging {
  opacity: 0.4;
}

.tab.drag-over-left {
  box-shadow: -2px 0 0 0 var(--accent);
}

.tab.drag-over-right {
  box-shadow: 2px 0 0 0 var(--accent);
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

.tab-bar-spacer {
  flex: 1;
}

.split-btn {
  margin: 0 0.25rem;
  flex-shrink: 0;
}

/* Empty drop zone */
.empty-drop-zone {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
  font-size: 0.85rem;
  border: 2px dashed var(--border);
  border-radius: 4px;
  margin: 0.5rem;
  transition: all 0.2s;
}

.empty-drop-zone.drag-over {
  border-color: var(--accent);
  background: rgba(0, 212, 255, 0.05);
  color: var(--accent);
}

/* Editor content */
.group-editor-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.cells {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem;
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

/* Markdown Preview Panel */
.markdown-preview {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem 2rem;
  background: #1a1a24;
  color: #f8f8f2;
  font-family: 'Monaco', 'Menlo', 'Consolas', 'Liberation Mono', 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.7;
  border: 1px solid var(--border);
  border-radius: 6px;
  margin: 0.5rem;
}

.markdown-preview :deep(h1),
.markdown-preview :deep(h2),
.markdown-preview :deep(h3),
.markdown-preview :deep(h4),
.markdown-preview :deep(h5),
.markdown-preview :deep(h6) {
  color: #bd93f9;
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  font-weight: 600;
  line-height: 1.3;
}

.markdown-preview :deep(h1) {
  font-size: 1.8em;
  border-bottom: 1px solid #3a3a4a;
  padding-bottom: 0.3em;
}

.markdown-preview :deep(h2) {
  font-size: 1.4em;
  border-bottom: 1px solid #3a3a4a;
  padding-bottom: 0.3em;
}

.markdown-preview :deep(h3) {
  font-size: 1.2em;
  color: #ff79c6;
}

.markdown-preview :deep(h4),
.markdown-preview :deep(h5),
.markdown-preview :deep(h6) {
  color: #8be9fd;
}

.markdown-preview :deep(p) {
  margin: 0.75em 0;
}

.markdown-preview :deep(a) {
  color: #50fa7b;
  text-decoration: none;
  transition: color 0.15s;
}

.markdown-preview :deep(a:hover) {
  color: #69ff94;
  text-decoration: underline;
}

.markdown-preview :deep(code) {
  background: #12121a;
  padding: 0.2em 0.4em;
  border-radius: 3px;
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.9em;
  color: #f1fa8c;
  border: 1px solid #2a2a3a;
}

.markdown-preview :deep(pre) {
  background: #12121a;
  padding: 1em;
  border-radius: 6px;
  overflow-x: auto;
  margin: 1em 0;
  border: 1px solid #2a2a3a;
}

.markdown-preview :deep(pre code) {
  background: transparent;
  padding: 0;
  font-size: 0.85em;
  line-height: 1.5;
  color: #f8f8f2;
  border: none;
}

.markdown-preview :deep(ul),
.markdown-preview :deep(ol) {
  margin: 0.75em 0;
  padding-left: 2em;
}

.markdown-preview :deep(li) {
  margin: 0.25em 0;
}

.markdown-preview :deep(li::marker) {
  color: #6272a4;
}

.markdown-preview :deep(blockquote) {
  margin: 1em 0;
  padding: 0.5em 1em;
  border-left: 4px solid #bd93f9;
  background: #12121a;
  color: #6272a4;
  font-style: italic;
}

.markdown-preview :deep(blockquote p) {
  margin: 0.5em 0;
}

.markdown-preview :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
}

.markdown-preview :deep(th),
.markdown-preview :deep(td) {
  border: 1px solid #3a3a4a;
  padding: 0.5em 0.75em;
  text-align: left;
}

.markdown-preview :deep(th) {
  background: #12121a;
  font-weight: 600;
  color: #8be9fd;
}

.markdown-preview :deep(tr:nth-child(even)) {
  background: rgba(98, 114, 164, 0.1);
}

.markdown-preview :deep(hr) {
  border: none;
  border-top: 1px solid #3a3a4a;
  margin: 2em 0;
}

.markdown-preview :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  border: 1px solid #3a3a4a;
}

.markdown-preview :deep(strong) {
  font-weight: 600;
  color: #ffb86c;
}

.markdown-preview :deep(em) {
  font-style: italic;
  color: #ff79c6;
}
</style>
