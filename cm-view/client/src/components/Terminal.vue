<template>
  <div class="terminal-container" ref="terminalContainer"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Terminal } from 'xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import 'xterm/css/xterm.css'

const props = defineProps({
  workspaceId: { type: [String, Number], required: true },
  active: { type: Boolean, default: true }
})

const emit = defineEmits(['files-changed'])

const terminalContainer = ref(null)
let terminal = null
let fitAddon = null
let ws = null
let resizeObserver = null
let sessionId = null

function connect() {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${wsProtocol}//${window.location.host}/ws`

  ws = new WebSocket(wsUrl)

  ws.onopen = () => {
    // Request a new terminal session for this workspace
    ws.send(JSON.stringify({
      type: 'terminal_create',
      workspaceId: props.workspaceId
    }))
  }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)

      if (data.type === 'terminal_created') {
        sessionId = data.sessionId
        terminal.write('\x1b[32mTerminal connected.\x1b[0m\r\n')
        terminal.write(`\x1b[90mWorkspace: ${props.workspaceId}\x1b[0m\r\n\r\n`)
      } else if (data.type === 'terminal_output') {
        terminal.write(data.data)
      } else if (data.type === 'terminal_exit') {
        terminal.write(`\r\n\x1b[31mSession ended (code: ${data.code})\x1b[0m\r\n`)
        sessionId = null
      } else if (data.type === 'files_changed') {
        // Notify parent that files may have changed
        emit('files-changed')
      } else if (data.type === 'error') {
        terminal.write(`\r\n\x1b[31mError: ${data.error}\x1b[0m\r\n`)
      }
    } catch (e) {
      // Binary data or non-JSON
      terminal.write(event.data)
    }
  }

  ws.onclose = () => {
    terminal.write('\r\n\x1b[33mDisconnected. Reconnecting...\x1b[0m\r\n')
    sessionId = null
    setTimeout(connect, 2000)
  }

  ws.onerror = (error) => {
    console.error('Terminal WebSocket error:', error)
  }
}

function initTerminal() {
  terminal = new Terminal({
    cursorBlink: true,
    cursorStyle: 'bar',
    fontSize: 13,
    fontFamily: "'Monaco', 'Menlo', 'Consolas', 'Liberation Mono', 'Courier New', monospace",
    theme: {
      background: '#0a0a12',
      foreground: '#e0e0e0',
      cursor: '#00d4ff',
      cursorAccent: '#0a0a12',
      selectionBackground: 'rgba(0, 212, 255, 0.3)',
      black: '#1a1a2e',
      red: '#f87171',
      green: '#4ade80',
      yellow: '#fbbf24',
      blue: '#60a5fa',
      magenta: '#c084fc',
      cyan: '#22d3ee',
      white: '#e0e0e0',
      brightBlack: '#6b7280',
      brightRed: '#fca5a5',
      brightGreen: '#86efac',
      brightYellow: '#fcd34d',
      brightBlue: '#93c5fd',
      brightMagenta: '#d8b4fe',
      brightCyan: '#67e8f9',
      brightWhite: '#ffffff'
    },
    scrollback: 5000,
    convertEol: true
  })

  fitAddon = new FitAddon()
  terminal.loadAddon(fitAddon)
  terminal.loadAddon(new WebLinksAddon())

  terminal.open(terminalContainer.value)
  fitAddon.fit()

  // Handle terminal input
  terminal.onData((data) => {
    if (ws && ws.readyState === WebSocket.OPEN && sessionId) {
      ws.send(JSON.stringify({
        type: 'terminal_input',
        sessionId,
        data
      }))
    }
  })

  // Handle resize
  terminal.onResize(({ cols, rows }) => {
    if (ws && ws.readyState === WebSocket.OPEN && sessionId) {
      ws.send(JSON.stringify({
        type: 'terminal_resize',
        sessionId,
        cols,
        rows
      }))
    }
  })

  // Watch for container size changes
  resizeObserver = new ResizeObserver(() => {
    if (fitAddon && props.active) {
      fitAddon.fit()
    }
  })
  resizeObserver.observe(terminalContainer.value)

  // Connect to backend
  connect()
}

// Refit when becoming active
watch(() => props.active, (active) => {
  if (active && fitAddon) {
    setTimeout(() => fitAddon.fit(), 50)
  }
})

onMounted(() => {
  initTerminal()
})

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect()
  }
  if (ws) {
    if (sessionId) {
      ws.send(JSON.stringify({
        type: 'terminal_destroy',
        sessionId
      }))
    }
    ws.close()
  }
  if (terminal) {
    terminal.dispose()
  }
})

// Expose focus method
defineExpose({
  focus: () => terminal?.focus()
})
</script>

<style scoped>
.terminal-container {
  width: 100%;
  height: 100%;
  background: #0a0a12;
  padding: 4px;
}

.terminal-container :deep(.xterm) {
  height: 100%;
}

.terminal-container :deep(.xterm-viewport) {
  overflow-y: auto;
}

.terminal-container :deep(.xterm-viewport::-webkit-scrollbar) {
  width: 8px;
}

.terminal-container :deep(.xterm-viewport::-webkit-scrollbar-track) {
  background: #1a1a2e;
}

.terminal-container :deep(.xterm-viewport::-webkit-scrollbar-thumb) {
  background: #3a3a5e;
  border-radius: 4px;
}

.terminal-container :deep(.xterm-viewport::-webkit-scrollbar-thumb:hover) {
  background: #4a4a7e;
}
</style>
