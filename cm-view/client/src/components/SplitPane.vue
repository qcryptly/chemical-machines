<template>
  <!-- Leaf node: render EditorGroup -->
  <EditorGroup
    v-if="node.type === 'leaf'"
    :group="node"
    :group-id="node.id"
    :is-focused="node.id === focusedLeafId"
    @focus="$emit('focus-leaf', node.id)"
    @switch-tab="$emit('switch-tab', { leafId: node.id, tabIndex: $event })"
    @close-tab="$emit('close-tab', { leafId: node.id, tabIndex: $event })"
    @split="$emit('split', { leafId: node.id, direction: $event })"
    @update-cell="$emit('update-cell', { leafId: node.id, cellIndex: $event.cellIndex, data: $event.data })"
    @run-cell="$emit('run-cell', { leafId: node.id, cellIndex: $event })"
    @delete-cell="$emit('delete-cell', { leafId: node.id, cellIndex: $event })"
    @create-cell-below="$emit('create-cell-below', { leafId: node.id, cellIndex: $event })"
    @reorder-cells="$emit('reorder-cells', { leafId: node.id, payload: $event })"
    @interrupt-cell="$emit('interrupt-cell', { leafId: node.id, cellIndex: $event })"
    @tab-drop-from-group="$emit('tab-drop', $event)"
  />

  <!-- Split node: flex container with children + resize handles -->
  <div
    v-else
    class="split-container"
    :class="node.direction === 'horizontal' ? 'split-h' : 'split-v'"
  >
    <template v-for="(child, idx) in node.children" :key="child.id">
      <!-- Resize handle between children (not before first) -->
      <div
        v-if="idx > 0"
        class="split-resize-handle"
        :class="node.direction === 'horizontal' ? 'handle-col' : 'handle-row'"
        @pointerdown="$emit('resize-start', { event: $event, splitId: node.id, childIndex: idx, direction: node.direction })"
      ></div>
      <!-- Recursive child pane -->
      <div class="split-child" :style="{ flexBasis: node.sizes[idx] + '%' }">
        <SplitPane
          :node="child"
          :focused-leaf-id="focusedLeafId"
          @focus-leaf="$emit('focus-leaf', $event)"
          @switch-tab="$emit('switch-tab', $event)"
          @close-tab="$emit('close-tab', $event)"
          @split="$emit('split', $event)"
          @update-cell="$emit('update-cell', $event)"
          @run-cell="$emit('run-cell', $event)"
          @delete-cell="$emit('delete-cell', $event)"
          @create-cell-below="$emit('create-cell-below', $event)"
          @reorder-cells="$emit('reorder-cells', $event)"
          @interrupt-cell="$emit('interrupt-cell', $event)"
          @tab-drop="$emit('tab-drop', $event)"
          @resize-start="$emit('resize-start', $event)"
        />
      </div>
    </template>
  </div>
</template>

<script setup>
import EditorGroup from './EditorGroup.vue'

defineProps({
  node: { type: Object, required: true },
  focusedLeafId: { type: Number, required: true }
})

defineEmits([
  'focus-leaf', 'switch-tab', 'close-tab', 'split',
  'update-cell', 'run-cell', 'delete-cell', 'create-cell-below',
  'reorder-cells', 'interrupt-cell', 'tab-drop',
  'resize-start'
])
</script>

<style scoped>
.split-container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.split-h {
  flex-direction: row;
}

.split-v {
  flex-direction: column;
}

.split-child {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  flex-shrink: 0;
  flex-grow: 0;
}

.split-resize-handle {
  flex-shrink: 0;
  background: var(--border);
  transition: background 0.2s;
  z-index: 10;
  position: relative;
}

.split-resize-handle.handle-col {
  width: 4px;
  cursor: col-resize;
}

.split-resize-handle.handle-row {
  height: 4px;
  cursor: row-resize;
}

.split-resize-handle:hover {
  background: var(--accent);
}
</style>
