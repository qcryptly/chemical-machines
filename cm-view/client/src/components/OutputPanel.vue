<template>
  <div class="output-panel" v-if="htmlContent">
    <div class="output-header" @click="toggleCollapse">
      <span class="output-label">Output</span>
      <button class="collapse-btn" :title="isCollapsed ? 'Expand output' : 'Collapse output'">
        <span class="chevron" :class="{ collapsed: isCollapsed }">&#9660;</span>
      </button>
    </div>
    <div class="output-content" v-show="!isCollapsed">
      <iframe
        ref="outputFrame"
        class="output-frame"
        sandbox="allow-scripts allow-same-origin"
        :srcdoc="styledHtmlContent"
        @load="onFrameLoad"
      ></iframe>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, computed } from 'vue'

const props = defineProps({
  htmlContent: { type: String, default: '' }
})

const outputFrame = ref(null)
const isCollapsed = ref(false)

function toggleCollapse() {
  isCollapsed.value = !isCollapsed.value
}

// MathJax CDN for math rendering with automatic line breaking
const mathJaxCdn = `
<script>
  MathJax = {
    loader: {
      load: ['[tex]/textmacros']
    },
    tex: {
      inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
      displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
      packages: {'[+]': ['textmacros']}
    },
    chtml: {
      // Match font size to surrounding text
      matchFontHeight: true,
      // Scale factor for math
      scale: 1
    },
    startup: {
      typeset: true
    }
  };
<\/script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js" async><\/script>
`

// Dark mode styles to inject into iframe content
const darkModeStyles = `
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0.75rem;
    background: #1e1e2e;
    color: #cdd6f4;
    font-size: 0.875rem;
    line-height: 1.5;
  }
  pre {
    background: #313244;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    overflow-x: auto;
    margin: 0.25rem 0;
    font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
    font-size: 0.8rem;
    color: #cdd6f4;
    border: 1px solid #45475a;
  }
  code {
    background: #313244;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
    font-size: 0.8rem;
    color: #cdd6f4;
  }
  img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
  }
  table {
    border-collapse: collapse;
    margin: 0.5rem 0;
    width: 100%;
    font-size: 0.8rem;
  }
  th, td {
    border: 1px solid #45475a;
    padding: 0.4rem 0.6rem;
    text-align: left;
  }
  th {
    background: #313244;
    color: #cdd6f4;
    font-weight: 600;
  }
  td {
    background: #1e1e2e;
  }
  tr:hover td {
    background: #262637;
  }
  h1, h2, h3, h4, h5, h6 {
    color: #cdd6f4;
    margin: 0.5rem 0 0.25rem 0;
    font-weight: 600;
  }
  h1 { font-size: 1.5rem; }
  h2 { font-size: 1.25rem; }
  h3 { font-size: 1.1rem; }
  p {
    margin: 0.25rem 0;
  }
  a {
    color: #89b4fa;
  }
  a:hover {
    color: #b4befe;
  }
  .cm-log {
    margin: 0.25rem 0;
    padding: 0.25rem 0.5rem;
    border-left: 3px solid #89b4fa;
    background: #262637;
    border-radius: 0 4px 4px 0;
  }
  .cm-log pre {
    background: transparent;
    border: none;
    padding: 0;
    margin: 0;
  }
  .cm-error {
    border-left-color: #f38ba8;
    background: #302030;
  }
  .cm-warning {
    border-left-color: #f9e2af;
    background: #302a20;
  }
  .cm-success {
    border-left-color: #a6e3a1;
    background: #203020;
  }
  .dataframe {
    border-collapse: collapse;
    font-size: 0.8rem;
  }
  .dataframe th {
    background: #313244;
  }
  .dataframe td {
    background: #1e1e2e;
  }
  /* Math styling */
  .cm-math {
    margin: 0.125rem 0;
    overflow-x: auto;
  }
  .cm-math-labeled {
    display: flex;
    flex-direction: column;
    gap: 0.125rem;
  }
  .cm-math-label {
    font-size: 0.75rem;
    color: #6c7086;
    font-style: italic;
  }
  .cm-equation {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0.125rem 0;
  }
  .cm-equation-content {
    flex: 1;
  }
  .cm-equation-number {
    color: #6c7086;
    font-size: 0.9rem;
    margin-left: 1rem;
  }
  /* Math alignment options */
  .cm-math-left {
    text-align: left;
  }
  .cm-math-center {
    text-align: center;
  }
  .cm-math-right {
    text-align: right;
  }
  /* Math lists */
  .cm-math-list {
    margin: 0.25rem 0;
    padding-left: 1.5rem;
  }
  .cm-math-list li {
    margin: 0.125rem 0;
  }
  .cm-math-list.numbered {
    list-style-type: decimal;
  }
  .cm-math-list.bulleted {
    list-style-type: disc;
  }
  .cm-math-list.none {
    list-style-type: none;
    padding-left: 0;
  }
  /* MathJax dark mode overrides */
  .MathJax svg {
    color: #cdd6f4;
  }
  mjx-container {
    color: #cdd6f4 !important;
    /* Force line breaking */
    display: block !important;
    overflow-wrap: break-word !important;
    word-wrap: break-word !important;
    word-break: break-word !important;
    white-space: normal !important;
    max-width: 100% !important;
    padding: 0.5rem 1rem !important;
  }
  mjx-container[jax="SVG"] > svg {
    color: #cdd6f4;
    max-width: 100%;
    height: auto;
  }
  mjx-container[jax="CHTML"] {
    color: #cdd6f4 !important;
    /* CHTML specific line breaking */
    display: block !important;
    max-width: 100% !important;
    overflow-wrap: break-word !important;
  }
  /* Force MathJax internal elements to allow wrapping */
  mjx-math {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    display: inline-block !important;
    max-width: 100% !important;
  }
  mjx-mrow {
    white-space: normal !important;
    flex-wrap: wrap !important;
  }
  /* Math container with horizontal scroll for long equations */
  .cm-math-scroll {
    overflow-x: auto;
    overflow-y: hidden;
    max-width: 100%;
    padding-bottom: 0.25rem;
  }
  .cm-math-scroll::-webkit-scrollbar {
    height: 6px;
  }
  .cm-math-scroll::-webkit-scrollbar-track {
    background: #313244;
    border-radius: 3px;
  }
  .cm-math-scroll::-webkit-scrollbar-thumb {
    background: #45475a;
    border-radius: 3px;
  }
  .cm-math-scroll::-webkit-scrollbar-thumb:hover {
    background: #585b70;
  }
  /* Multi-line math display using aligned environment */
  .cm-math-multiline mjx-container {
    display: block;
  }
  /* Allow display math to scroll if too wide */
  mjx-container[display="true"] {
    max-width: 100%;
    overflow-x: auto;
    padding: 0.25rem 0;
  }
  /* Force cm-math containers to constrain width */
  .cm-math {
    max-width: 100%;
    overflow-x: auto;
  }
</style>
`

// Inject dark mode styles and MathJax into HTML content
const styledHtmlContent = computed(() => {
  if (!props.htmlContent) return ''

  const headContent = darkModeStyles + mathJaxCdn

  // Check if content already has a <head> tag
  if (props.htmlContent.includes('<head>')) {
    // Replace existing styles by injecting our content after <head>
    return props.htmlContent.replace(/<head>[\s\S]*?<\/head>/i, `<head>${headContent}</head>`)
  } else if (props.htmlContent.includes('<html>')) {
    // Add head with our content
    return props.htmlContent.replace('<html>', `<html><head>${headContent}</head>`)
  } else {
    // Wrap in basic HTML structure
    return `<!DOCTYPE html><html><head>${headContent}</head><body>${props.htmlContent}</body></html>`
  }
})

function adjustFrameHeight() {
  if (outputFrame.value) {
    try {
      const doc = outputFrame.value.contentDocument || outputFrame.value.contentWindow.document
      if (doc && doc.body) {
        // Get the actual content height
        const height = Math.max(
          doc.body.scrollHeight,
          doc.documentElement.scrollHeight,
          40 // minimum height
        )
        // Set height with small padding, no max cap (let container scroll if needed)
        outputFrame.value.style.height = (height + 8) + 'px'
      }
    } catch (e) {
      // Cross-origin or security error, use default height
      console.warn('Could not adjust iframe height:', e)
    }
  }
}

function onFrameLoad() {
  // Initial height adjustment
  adjustFrameHeight()

  // Re-adjust after KaTeX renders (it loads asynchronously)
  setTimeout(adjustFrameHeight, 500)
  setTimeout(adjustFrameHeight, 1000)
}

// Re-adjust height when content changes
watch(() => props.htmlContent, () => {
  if (outputFrame.value) {
    // Small delay to allow content to render
    setTimeout(adjustFrameHeight, 100)
  }
})
</script>

<style scoped>
.output-panel {
  margin-top: 0.5rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: #1e1e2e;
  overflow: hidden;
}

.output-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.25rem 0.5rem;
  background: #262637;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  user-select: none;
}

.output-header:hover {
  background: #2d2d42;
}

.output-label {
  font-size: 0.7rem;
  color: #6c7086;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.collapse-btn {
  background: none;
  border: none;
  color: #6c7086;
  font-size: 0.6rem;
  cursor: pointer;
  padding: 0 0.25rem;
  line-height: 1;
}

.collapse-btn:hover {
  color: #cdd6f4;
}

.chevron {
  display: inline-block;
  transition: transform 0.2s ease;
}

.chevron.collapsed {
  transform: rotate(-90deg);
}

.output-content {
  max-height: 500px;
  overflow: auto;
}

.output-frame {
  width: 100%;
  min-height: 40px;
  border: none;
  background: #1e1e2e;
  display: block;
}
</style>
