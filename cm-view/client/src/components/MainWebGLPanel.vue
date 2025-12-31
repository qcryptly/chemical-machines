<template>
  <div class="webgl-panel" :class="{ collapsed: !expanded }">
    <div class="webgl-header" @click="toggleExpanded">
      <div class="header-left">
        <span class="expand-icon">{{ expanded ? '▼' : '▶' }}</span>
        <span class="webgl-label">WebGL View</span>
        <span class="status-indicator" :class="{ active: hasContent }"></span>
      </div>
      <div class="header-right" v-if="hasContent">
        <button @click.stop="refreshContent" class="control-btn" title="Refresh">
          <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
            <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
          </svg>
        </button>
        <button v-if="expanded" @click.stop="setExpanded(false)" class="control-btn" title="Collapse">
          <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
            <path d="M7.41 15.41L12 10.83l4.59 4.58L18 14l-6-6-6 6z"/>
          </svg>
        </button>
        <button v-else @click.stop="setExpanded(true)" class="control-btn" title="Expand">
          <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
            <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6z"/>
          </svg>
        </button>
      </div>
    </div>
    <div class="webgl-content" v-if="expanded" :style="contentStyle">
      <iframe
        v-if="hasContent"
        ref="webglFrame"
        class="webgl-frame"
        sandbox="allow-scripts allow-same-origin"
        :srcdoc="htmlContent"
        @load="onFrameLoad"
      ></iframe>
      <div v-else class="no-content">
        <div class="placeholder-scene">
          <div class="cube"></div>
        </div>
        <p>No WebGL content</p>
        <p class="hint">Run a cell with <code>cm.views.webgl()</code> or <code>cm.views.webgl_threejs()</code></p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  htmlContent: { type: String, default: '' },
  height: { type: Number, default: 300 }
})

const emit = defineEmits(['refresh', 'update:expanded'])

const expanded = ref(true)
const webglFrame = ref(null)

const hasContent = computed(() => {
  return props.htmlContent && props.htmlContent.trim().length > 0
})

const contentStyle = computed(() => {
  // Subtract header height (~32px)
  const contentHeight = Math.max(props.height - 32, 100)
  return { height: contentHeight + 'px' }
})

function toggleExpanded() {
  setExpanded(!expanded.value)
}

function setExpanded(value) {
  expanded.value = value
  emit('update:expanded', value)
}

function refreshContent() {
  emit('refresh')
}

function onFrameLoad() {
  // Frame loaded - could add post-processing here if needed
}

// Auto-expand when content arrives
watch(() => props.htmlContent, (newContent) => {
  if (newContent && newContent.trim().length > 0) {
    setExpanded(true)
  }
})
</script>

<style scoped>
.webgl-panel {
  background: #1e1e2e;
  overflow: hidden;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.webgl-panel.collapsed {
  height: auto;
}

.webgl-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: #262637;
  border-bottom: 1px solid var(--border, #313244);
  cursor: pointer;
  user-select: none;
  flex-shrink: 0;
}

.webgl-header:hover {
  background: #2d2d44;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.expand-icon {
  font-size: 0.65rem;
  color: #6c7086;
  width: 12px;
}

.webgl-label {
  font-size: 0.75rem;
  color: #cdd6f4;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.status-indicator {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #45475a;
}

.status-indicator.active {
  background: #a6e3a1;
  box-shadow: 0 0 4px #a6e3a1;
}

.control-btn {
  background: none;
  border: none;
  color: #6c7086;
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.control-btn:hover {
  color: #cdd6f4;
  background: #313244;
}

.webgl-content {
  flex: 1;
  background: #1e1e2e;
  overflow: hidden;
}

.webgl-frame {
  width: 100%;
  height: 100%;
  border: none;
  display: block;
}

.no-content {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #6c7086;
  gap: 0.75rem;
}

.no-content p {
  margin: 0;
  font-size: 0.85rem;
}

.no-content .hint {
  font-size: 0.75rem;
}

.no-content code {
  background: #313244;
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.75rem;
  color: #89b4fa;
}

/* Animated placeholder cube */
.placeholder-scene {
  width: 80px;
  height: 80px;
  perspective: 200px;
  margin-bottom: 1rem;
}

.cube {
  width: 40px;
  height: 40px;
  position: relative;
  transform-style: preserve-3d;
  animation: rotateCube 8s infinite linear;
  margin: 20px auto;
}

.cube::before,
.cube::after {
  content: '';
  position: absolute;
  width: 40px;
  height: 40px;
  border: 2px solid #45475a;
  background: rgba(69, 71, 90, 0.1);
}

.cube::before {
  transform: translateZ(20px);
}

.cube::after {
  transform: translateZ(-20px);
}

@keyframes rotateCube {
  0% { transform: rotateX(0deg) rotateY(0deg); }
  100% { transform: rotateX(360deg) rotateY(360deg); }
}
</style>
