<template>
  <div class="cell" :class="{ focused: isFocused }">
    <div class="cell-toolbar">
      <span class="cell-number">[{{ index + 1 }}]</span>
      <div class="toolbar-spacer"></div>
      <button @click="$emit('run')" class="run-btn" title="Run cell (Ctrl+Enter)">
        <span v-if="cell.status === 'running'" class="spinner"></span>
        <span v-else>&#9654;</span>
      </button>
      <button @click="$emit('delete')" class="delete-btn" title="Delete cell">&times;</button>
    </div>

    <div class="cell-input" ref="editorContainer"></div>

    <div class="cell-output" v-if="cell.output || cell.status === 'running'">
      <div v-if="cell.status === 'running' && !cell.output" class="running-indicator">
        Running...
      </div>
      <pre v-if="cell.output" :class="{ error: cell.status === 'error' }">{{ cell.output }}</pre>
      <div v-if="cell.status === 'running' && cell.output" class="running-indicator streaming">
        ▶ Running...
      </div>
    </div>

    <!-- HTML Output Panel -->
    <OutputPanel
      v-if="htmlOutput && showHtmlOutput"
      :html-content="htmlOutput"
      @close="showHtmlOutput = false"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from '@codemirror/view'
import OutputPanel from './OutputPanel.vue'
import { EditorState } from '@codemirror/state'
import { defaultKeymap, history, historyKeymap, indentWithTab, insertNewlineAndIndent } from '@codemirror/commands'
import { syntaxHighlighting, HighlightStyle, bracketMatching, foldGutter } from '@codemirror/language'
import { autocompletion, completionKeymap, completeFromList } from '@codemirror/autocomplete'
import { python } from '@codemirror/lang-python'
import { cpp } from '@codemirror/lang-cpp'
import { StreamLanguage } from '@codemirror/language'
import { shell } from '@codemirror/legacy-modes/mode/shell'
import { tags } from '@lezer/highlight'

// Python completions - common builtins, keywords, and scientific computing
const pythonCompletions = [
  // Keywords
  { label: 'import', type: 'keyword', detail: 'import module' },
  { label: 'from', type: 'keyword', detail: 'from module import' },
  { label: 'def', type: 'keyword', detail: 'define function' },
  { label: 'class', type: 'keyword', detail: 'define class' },
  { label: 'return', type: 'keyword' },
  { label: 'yield', type: 'keyword' },
  { label: 'if', type: 'keyword' },
  { label: 'elif', type: 'keyword' },
  { label: 'else', type: 'keyword' },
  { label: 'for', type: 'keyword' },
  { label: 'while', type: 'keyword' },
  { label: 'try', type: 'keyword' },
  { label: 'except', type: 'keyword' },
  { label: 'finally', type: 'keyword' },
  { label: 'with', type: 'keyword' },
  { label: 'as', type: 'keyword' },
  { label: 'lambda', type: 'keyword' },
  { label: 'pass', type: 'keyword' },
  { label: 'break', type: 'keyword' },
  { label: 'continue', type: 'keyword' },
  { label: 'raise', type: 'keyword' },
  { label: 'assert', type: 'keyword' },
  { label: 'async', type: 'keyword' },
  { label: 'await', type: 'keyword' },
  { label: 'global', type: 'keyword' },
  { label: 'nonlocal', type: 'keyword' },
  // Builtins
  { label: 'print', type: 'function', detail: 'print(*args)' },
  { label: 'len', type: 'function', detail: 'len(obj)' },
  { label: 'range', type: 'function', detail: 'range(start, stop, step)' },
  { label: 'list', type: 'function', detail: 'list(iterable)' },
  { label: 'dict', type: 'function', detail: 'dict(**kwargs)' },
  { label: 'set', type: 'function', detail: 'set(iterable)' },
  { label: 'tuple', type: 'function', detail: 'tuple(iterable)' },
  { label: 'str', type: 'function', detail: 'str(obj)' },
  { label: 'int', type: 'function', detail: 'int(x)' },
  { label: 'float', type: 'function', detail: 'float(x)' },
  { label: 'bool', type: 'function', detail: 'bool(x)' },
  { label: 'type', type: 'function', detail: 'type(obj)' },
  { label: 'isinstance', type: 'function', detail: 'isinstance(obj, class)' },
  { label: 'hasattr', type: 'function', detail: 'hasattr(obj, name)' },
  { label: 'getattr', type: 'function', detail: 'getattr(obj, name, default)' },
  { label: 'setattr', type: 'function', detail: 'setattr(obj, name, value)' },
  { label: 'enumerate', type: 'function', detail: 'enumerate(iterable)' },
  { label: 'zip', type: 'function', detail: 'zip(*iterables)' },
  { label: 'map', type: 'function', detail: 'map(func, iterable)' },
  { label: 'filter', type: 'function', detail: 'filter(func, iterable)' },
  { label: 'sorted', type: 'function', detail: 'sorted(iterable, key=None)' },
  { label: 'reversed', type: 'function', detail: 'reversed(seq)' },
  { label: 'sum', type: 'function', detail: 'sum(iterable)' },
  { label: 'min', type: 'function', detail: 'min(iterable)' },
  { label: 'max', type: 'function', detail: 'max(iterable)' },
  { label: 'abs', type: 'function', detail: 'abs(x)' },
  { label: 'round', type: 'function', detail: 'round(x, ndigits)' },
  { label: 'open', type: 'function', detail: 'open(file, mode)' },
  { label: 'input', type: 'function', detail: 'input(prompt)' },
  { label: 'format', type: 'function', detail: 'format(value, spec)' },
  { label: 'repr', type: 'function', detail: 'repr(obj)' },
  { label: 'ord', type: 'function', detail: 'ord(char)' },
  { label: 'chr', type: 'function', detail: 'chr(code)' },
  { label: 'hex', type: 'function', detail: 'hex(x)' },
  { label: 'bin', type: 'function', detail: 'bin(x)' },
  { label: 'all', type: 'function', detail: 'all(iterable)' },
  { label: 'any', type: 'function', detail: 'any(iterable)' },
  { label: 'callable', type: 'function', detail: 'callable(obj)' },
  { label: 'dir', type: 'function', detail: 'dir(obj)' },
  { label: 'vars', type: 'function', detail: 'vars(obj)' },
  { label: 'help', type: 'function', detail: 'help(obj)' },
  { label: 'exec', type: 'function', detail: 'exec(code)' },
  { label: 'eval', type: 'function', detail: 'eval(expr)' },
  // Constants
  { label: 'True', type: 'constant' },
  { label: 'False', type: 'constant' },
  { label: 'None', type: 'constant' },
  // Common imports
  { label: 'numpy', type: 'module', detail: 'numerical computing (as np)' },
  { label: 'pandas', type: 'module', detail: 'data analysis (as pd)' },
  { label: 'matplotlib', type: 'module', detail: 'plotting (.pyplot as plt)' },
  { label: 'scipy', type: 'module', detail: 'scientific computing' },
  { label: 'torch', type: 'module', detail: 'PyTorch' },
  { label: 'tensorflow', type: 'module', detail: 'TensorFlow (as tf)' },
  { label: 'sklearn', type: 'module', detail: 'scikit-learn' },
  { label: 'rdkit', type: 'module', detail: 'chemistry toolkit' },
  { label: 'openmm', type: 'module', detail: 'molecular dynamics' },
  { label: 'mdtraj', type: 'module', detail: 'MD trajectory analysis (as md)' },
  { label: 'Bio', type: 'module', detail: 'Biopython' },
  // NumPy shortcuts
  { label: 'np.array', type: 'function', detail: 'create array' },
  { label: 'np.zeros', type: 'function', detail: 'array of zeros' },
  { label: 'np.ones', type: 'function', detail: 'array of ones' },
  { label: 'np.arange', type: 'function', detail: 'evenly spaced values' },
  { label: 'np.linspace', type: 'function', detail: 'linearly spaced values' },
  { label: 'np.reshape', type: 'function', detail: 'reshape array' },
  { label: 'np.mean', type: 'function', detail: 'mean value' },
  { label: 'np.std', type: 'function', detail: 'standard deviation' },
  { label: 'np.sum', type: 'function', detail: 'sum of elements' },
  { label: 'np.dot', type: 'function', detail: 'dot product' },
  { label: 'np.random', type: 'namespace', detail: 'random number generation' },
  { label: 'np.linalg', type: 'namespace', detail: 'linear algebra' },
]

// C++ completions
const cppCompletions = [
  // Keywords
  { label: '#include', type: 'keyword', detail: '#include <header>' },
  { label: '#define', type: 'keyword', detail: '#define MACRO' },
  { label: 'namespace', type: 'keyword' },
  { label: 'using', type: 'keyword' },
  { label: 'class', type: 'keyword' },
  { label: 'struct', type: 'keyword' },
  { label: 'public', type: 'keyword' },
  { label: 'private', type: 'keyword' },
  { label: 'protected', type: 'keyword' },
  { label: 'virtual', type: 'keyword' },
  { label: 'override', type: 'keyword' },
  { label: 'const', type: 'keyword' },
  { label: 'static', type: 'keyword' },
  { label: 'inline', type: 'keyword' },
  { label: 'template', type: 'keyword' },
  { label: 'typename', type: 'keyword' },
  { label: 'auto', type: 'keyword' },
  { label: 'return', type: 'keyword' },
  { label: 'if', type: 'keyword' },
  { label: 'else', type: 'keyword' },
  { label: 'for', type: 'keyword' },
  { label: 'while', type: 'keyword' },
  { label: 'do', type: 'keyword' },
  { label: 'switch', type: 'keyword' },
  { label: 'case', type: 'keyword' },
  { label: 'break', type: 'keyword' },
  { label: 'continue', type: 'keyword' },
  { label: 'try', type: 'keyword' },
  { label: 'catch', type: 'keyword' },
  { label: 'throw', type: 'keyword' },
  { label: 'new', type: 'keyword' },
  { label: 'delete', type: 'keyword' },
  { label: 'nullptr', type: 'constant' },
  { label: 'true', type: 'constant' },
  { label: 'false', type: 'constant' },
  // Types
  { label: 'int', type: 'type' },
  { label: 'float', type: 'type' },
  { label: 'double', type: 'type' },
  { label: 'char', type: 'type' },
  { label: 'bool', type: 'type' },
  { label: 'void', type: 'type' },
  { label: 'size_t', type: 'type' },
  { label: 'string', type: 'type', detail: 'std::string' },
  { label: 'vector', type: 'type', detail: 'std::vector<T>' },
  { label: 'map', type: 'type', detail: 'std::map<K,V>' },
  { label: 'set', type: 'type', detail: 'std::set<T>' },
  { label: 'unique_ptr', type: 'type', detail: 'std::unique_ptr<T>' },
  { label: 'shared_ptr', type: 'type', detail: 'std::shared_ptr<T>' },
  // Common functions
  { label: 'cout', type: 'variable', detail: 'std::cout' },
  { label: 'cin', type: 'variable', detail: 'std::cin' },
  { label: 'endl', type: 'variable', detail: 'std::endl' },
  { label: 'printf', type: 'function' },
  { label: 'scanf', type: 'function' },
  { label: 'malloc', type: 'function' },
  { label: 'free', type: 'function' },
  { label: 'sizeof', type: 'function' },
  // Headers (for use inside #include <>)
  { label: 'iostream', type: 'text', detail: 'I/O stream header' },
  { label: 'algorithm', type: 'text', detail: 'algorithm header' },
  { label: 'cmath', type: 'text', detail: 'math header' },
]

// Bash completions
const bashCompletions = [
  // Builtins
  { label: 'echo', type: 'function', detail: 'print text' },
  { label: 'cd', type: 'function', detail: 'change directory' },
  { label: 'ls', type: 'function', detail: 'list files' },
  { label: 'pwd', type: 'function', detail: 'print working directory' },
  { label: 'mkdir', type: 'function', detail: 'make directory' },
  { label: 'rm', type: 'function', detail: 'remove files' },
  { label: 'cp', type: 'function', detail: 'copy files' },
  { label: 'mv', type: 'function', detail: 'move files' },
  { label: 'cat', type: 'function', detail: 'concatenate files' },
  { label: 'grep', type: 'function', detail: 'search pattern' },
  { label: 'find', type: 'function', detail: 'find files' },
  { label: 'sed', type: 'function', detail: 'stream editor' },
  { label: 'awk', type: 'function', detail: 'pattern scanning' },
  { label: 'head', type: 'function', detail: 'first lines' },
  { label: 'tail', type: 'function', detail: 'last lines' },
  { label: 'wc', type: 'function', detail: 'word count' },
  { label: 'sort', type: 'function', detail: 'sort lines' },
  { label: 'uniq', type: 'function', detail: 'unique lines' },
  { label: 'cut', type: 'function', detail: 'cut columns' },
  { label: 'xargs', type: 'function', detail: 'build arguments' },
  { label: 'chmod', type: 'function', detail: 'change permissions' },
  { label: 'chown', type: 'function', detail: 'change owner' },
  { label: 'tar', type: 'function', detail: 'archive files' },
  { label: 'gzip', type: 'function', detail: 'compress files' },
  { label: 'curl', type: 'function', detail: 'transfer data' },
  { label: 'wget', type: 'function', detail: 'download files' },
  { label: 'ssh', type: 'function', detail: 'secure shell' },
  { label: 'scp', type: 'function', detail: 'secure copy' },
  { label: 'export', type: 'keyword', detail: 'set env variable' },
  { label: 'source', type: 'keyword', detail: 'run script' },
  { label: 'alias', type: 'keyword', detail: 'create alias' },
  // Control flow
  { label: 'if', type: 'keyword' },
  { label: 'then', type: 'keyword' },
  { label: 'else', type: 'keyword' },
  { label: 'elif', type: 'keyword' },
  { label: 'fi', type: 'keyword' },
  { label: 'for', type: 'keyword' },
  { label: 'do', type: 'keyword' },
  { label: 'done', type: 'keyword' },
  { label: 'while', type: 'keyword' },
  { label: 'case', type: 'keyword' },
  { label: 'esac', type: 'keyword' },
  { label: 'function', type: 'keyword' },
  // Conda
  { label: 'conda', type: 'function', detail: 'conda package manager' },
  { label: 'pip', type: 'function', detail: 'Python package installer' },
  { label: 'python', type: 'function', detail: 'Python interpreter' },
]

// Get completions based on language
function getCompletions(lang) {
  switch (lang) {
    case 'python': return pythonCompletions
    case 'cpp': return cppCompletions
    case 'bash': return bashCompletions
    default: return pythonCompletions
  }
}

// Custom syntax highlighting that's more readable on dark backgrounds
const customHighlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: '#ff79c6' },
  { tag: tags.operator, color: '#ff79c6' },
  { tag: tags.special(tags.variableName), color: '#50fa7b' },
  { tag: tags.typeName, color: '#8be9fd', fontStyle: 'italic' },
  { tag: tags.atom, color: '#bd93f9' },
  { tag: tags.number, color: '#bd93f9' },
  { tag: tags.bool, color: '#bd93f9' },
  { tag: tags.definition(tags.variableName), color: '#50fa7b' },
  { tag: tags.string, color: '#f1fa8c' },
  { tag: tags.special(tags.string), color: '#f1fa8c' },
  { tag: tags.comment, color: '#6272a4' },
  { tag: tags.variableName, color: '#f8f8f2' },
  { tag: tags.bracket, color: '#f8f8f2' },
  { tag: tags.meta, color: '#f8f8f2' },
  { tag: tags.link, color: '#8be9fd', textDecoration: 'underline' },
  { tag: tags.heading, fontWeight: 'bold', color: '#bd93f9' },
  { tag: tags.emphasis, fontStyle: 'italic' },
  { tag: tags.strong, fontWeight: 'bold' },
  { tag: tags.strikethrough, textDecoration: 'line-through' },
  { tag: tags.className, color: '#8be9fd' },
  { tag: tags.propertyName, color: '#66d9ef' },
  { tag: tags.function(tags.variableName), color: '#50fa7b' },
  { tag: tags.function(tags.propertyName), color: '#50fa7b' },
  { tag: tags.constant(tags.variableName), color: '#bd93f9' },
])

const props = defineProps({
  cell: { type: Object, required: true },
  index: { type: Number, required: true },
  language: { type: String, default: 'python' },
  htmlOutput: { type: String, default: '' }
})

const emit = defineEmits(['update', 'run', 'delete', 'blur'])

const editorContainer = ref(null)
const isFocused = ref(false)
const showHtmlOutput = ref(true)
let editorView = null

const languageExtensions = {
  python: () => python(),
  cpp: () => cpp(),
  bash: () => StreamLanguage.define(shell)
}

function getLanguageExtension(lang) {
  const factory = languageExtensions[lang] || languageExtensions.python
  return factory()
}

function createEditor() {
  if (!editorContainer.value) return

  // Create language-specific autocomplete
  const languageCompletions = completeFromList(getCompletions(props.language))

  const extensions = [
    lineNumbers(),
    highlightActiveLine(),
    highlightActiveLineGutter(),
    history(),
    bracketMatching(),
    foldGutter(),
    syntaxHighlighting(customHighlightStyle),
    getLanguageExtension(props.language),
    autocompletion({
      override: [languageCompletions],
      activateOnTyping: true,
      maxRenderedOptions: 20,
    }),
    keymap.of([
      {
        key: 'Mod-Enter',
        run: () => {
          emit('run')
          return true
        },
        preventDefault: true
      },
      // Explicit Enter key handling to prevent autocomplete from interfering
      {
        key: 'Enter',
        run: insertNewlineAndIndent,
      },
      ...defaultKeymap,
      ...historyKeymap,
      indentWithTab,
    ]),
    // Completion keymap separate so it doesn't override Enter behavior
    keymap.of(completionKeymap),

    EditorView.updateListener.of((update) => {
      if (update.docChanged) {
        const content = update.state.doc.toString()
        emit('update', { content })
      }
      if (update.focusChanged) {
        isFocused.value = update.view.hasFocus
        if (!update.view.hasFocus) {
          emit('blur')
        }
      }
    }),
    EditorView.theme({
      '&': {
        fontSize: '13px',
        backgroundColor: '#1a1a24'
      },
      '.cm-content': {
        fontFamily: "'Monaco', 'Menlo', 'Consolas', monospace",
        padding: '8px 0',
        caretColor: '#ffedcb'
      },
      '.cm-gutters': {
        backgroundColor: '#12121a',
        borderRight: '1px solid #2a2a3a'
      },
      '.cm-lineNumbers .cm-gutterElement': {
        padding: '0 8px',
        minWidth: '32px'
      },
      '.cm-scroller': {
        minHeight: '80px'
      },
      '.cm-cursor, .cm-cursor-primary': {
        borderLeftColor: 'white'
      },
      // Autocomplete tooltip styling

      '.cm-tooltip': {
        backgroundColor: '#1e1e2e',
        border: '1px solid #3a3a4a',
        borderRadius: '4px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.4)'
      },
      '.cm-tooltip-autocomplete': {
        '& > ul': {
          fontFamily: "'Monaco', 'Menlo', 'Consolas', monospace",
          fontSize: '12px',
          maxHeight: '250px'
        },
        '& > ul > li': {
          padding: '4px 8px',
          color: '#f8f8f2'
        },
        '& > ul > li[aria-selected]': {
          backgroundColor: '#44475a',
          color: '#f8f8f2'
        }
      },
      '.cm-completionLabel': {
        color: '#f8f8f2'
      },
      '.cm-completionDetail': {
        color: '#6272a4',
        fontStyle: 'italic',
        marginLeft: '8px'
      },
      '.cm-completionIcon': {
        marginRight: '4px'
      },
      '.cm-completionIcon-function': {
        color: '#50fa7b'
      },
      '.cm-completionIcon-keyword': {
        color: '#ff79c6'
      },
      '.cm-completionIcon-type': {
        color: '#8be9fd'
      },
      '.cm-completionIcon-variable': {
        color: '#f8f8f2'
      },
      '.cm-completionIcon-constant': {
        color: '#bd93f9'
      },
      '.cm-completionIcon-module': {
        color: '#ffb86c'
      }
    })
  ]

  const state = EditorState.create({
    doc: props.cell.content || '',
    extensions
  })

  editorView = new EditorView({
    state,
    parent: editorContainer.value
  })
}

function destroyEditor() {
  if (editorView) {
    editorView.destroy()
    editorView = null
  }
}

// Watch for external content changes
watch(() => props.cell.content, (newContent) => {
  if (editorView && newContent !== editorView.state.doc.toString()) {
    editorView.dispatch({
      changes: { from: 0, to: editorView.state.doc.length, insert: newContent || '' }
    })
  }
})

// Rebuild editor when language changes
watch(() => props.language, () => {
  const content = editorView?.state.doc.toString() || props.cell.content || ''
  destroyEditor()
  createEditor()
  if (editorView && content) {
    editorView.dispatch({
      changes: { from: 0, to: editorView.state.doc.length, insert: content }
    })
  }
})

// Show output panel when new HTML output arrives
watch(() => props.htmlOutput, (newOutput) => {
  if (newOutput) {
    showHtmlOutput.value = true
  }
})

onMounted(() => {
  createEditor()
})

onUnmounted(() => {
  destroyEditor()
})
</script>

<style scoped>
.cell {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 0.5rem;
  overflow: hidden;
  transition: border-color 0.2s;
}

.cell.focused {
  border-color: var(--accent);
}

.cell-toolbar {
  background: var(--bg-tertiary);
  padding: 0.25rem 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.cell-number {
  color: var(--text-secondary);
  font-size: 0.7rem;
  font-family: monospace;
  min-width: 25px;
}

.lang-select, .env-select, .compiler-select, .std-select {
  font-size: 0.7rem;
  padding: 0.2rem 0.4rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--text-primary);
  cursor: pointer;
}

.lang-select:hover, .env-select:hover, .compiler-select:hover, .std-select:hover {
  border-color: var(--accent);
}

.compiler-select {
  color: #10b981;
  max-width: 70px;
}

.std-select {
  color: #6366f1;
  max-width: 65px;
}

.env-select {
  color: var(--accent);
  max-width: 100px;
}

.env-select.cpp-env {
  color: #f59e0b;
}

.env-select.vendor-env {
  color: #a855f7;
}

.toolbar-spacer {
  flex: 1;
}

.run-btn, .delete-btn {
  width: 22px;
  height: 22px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  border-radius: 4px;
  border: none;
  cursor: pointer;
}

.run-btn {
  background: var(--success);
  color: white;
}

.run-btn:hover {
  opacity: 0.9;
}

.delete-btn {
  background: transparent;
  color: var(--text-secondary);
}

.delete-btn:hover {
  background: var(--error);
  color: white;
}

.spinner {
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.cell-input {
  min-height: 80px;
  overflow: hidden;
}

.cell-input :deep(.cm-editor) {
  height: 100%;
}

.cell-output {
  border-top: 1px solid var(--border);
  padding: 0.5rem;
  background: var(--bg-tertiary);
  max-height: 200px;
  overflow-y: auto;
}

.cell-output pre {
  margin: 0;
  color: var(--text-secondary);
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.75rem;
  white-space: pre-wrap;
  word-break: break-word;
}

.cell-output pre.error {
  color: var(--error);
}

.running-indicator {
  color: var(--accent);
  font-size: 0.8rem;
  animation: pulse 1.5s ease-in-out infinite;
}

.running-indicator.streaming {
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.7rem;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
