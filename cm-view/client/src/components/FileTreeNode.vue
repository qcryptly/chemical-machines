<template>
  <div class="tree-node">
    <div
      class="node-row"
      :class="{ selected: selectedPath === item.path, folder: item.type === 'folder' }"
      @click="handleClick"
      @dblclick="handleDoubleClick"
      @contextmenu.prevent="showContextMenu"
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

    <div v-if="item.type === 'folder' && expanded && item.children" class="children">
      <FileTreeNode
        v-for="child in item.children"
        :key="child.path"
        :item="child"
        :selected-path="selectedPath"
        @select="$emit('select', $event)"
        @open="$emit('open', $event)"
        @delete="$emit('delete', $event)"
        @rename="$emit('rename', $event)"
        @create-file="$emit('create-file', $event)"
        @create-folder="$emit('create-folder', $event)"
        @context-menu="$emit('context-menu', $event)"
      />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  item: { type: Object, required: true },
  selectedPath: { type: String, default: null }
})

const emit = defineEmits(['select', 'open', 'delete', 'rename', 'create-file', 'create-folder', 'context-menu'])

const expanded = ref(false)

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

function handleClick() {
  emit('select', props.item)
  if (props.item.type === 'folder') {
    expanded.value = !expanded.value
  } else {
    // Open files on single click
    emit('open', props.item)
  }
}

function handleDoubleClick() {
  emit('open', props.item)
}

function showContextMenu(event) {
  emit('select', props.item)
  emit('context-menu', { item: props.item, x: event.clientX, y: event.clientY })
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
</style>
