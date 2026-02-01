<template>
  <div
    class="cell"
    :class="{ focused: isFocused, 'drag-over': isDragOver, 'dragging': isDragging }"
    @dragover="handleDragOver"
    @dragleave="handleDragLeave"
    @drop="handleDrop"
  >
    <div class="cell-toolbar">
      <span
        class="drag-handle"
        title="Drag to reorder"
        draggable="true"
        @dragstart="handleDragStart"
        @dragend="handleDragEnd"
      ><GripVertical :size="14" /></span>
      <span class="cell-number">[{{ index + 1 }}]</span>
      <div class="toolbar-spacer"></div>
      <button @click="$emit('create-below')" class="btn-icon create-below-btn" title="Create cell below (Alt+Enter)">
        <Plus :size="14" />
      </button>
      <button
        v-if="cell.status === 'running'"
        @click="$emit('interrupt')"
        class="btn-icon interrupt-btn"
        title="Interrupt execution (Ctrl+C)"
      >
        <Square :size="14" />
      </button>
      <button @click="$emit('run')" class="btn-icon run-btn" title="Run cell (Ctrl+Enter)" v-else>
        <Play :size="14" />
      </button>
      <button @click="$emit('delete')" class="btn-icon btn-danger delete-btn" title="Delete cell"><X :size="14" /></button>
    </div>

    <div class="cell-input" ref="editorContainer"></div>

    <div class="cell-output" v-if="cell.output || cell.status === 'running'">
      <div class="cell-output-header" @click="logCollapsed = !logCollapsed">
        <span class="cell-output-label">Log</span>
        <span class="cell-output-chevron" :class="{ collapsed: logCollapsed }"><ChevronDown :size="12" /></span>
      </div>
      <div class="cell-output-body" v-show="!logCollapsed" :style="{ maxHeight: logHeight + 'px' }">
        <div v-if="cell.status === 'running' && !cell.output" class="running-indicator">
          Running...
        </div>
        <pre v-if="cell.output" :class="{ error: cell.status === 'error' }">{{ cell.output }}</pre>
        <div v-if="cell.status === 'running' && cell.output" class="running-indicator streaming">
          <Play :size="10" style="display:inline-block;vertical-align:middle;margin-right:0.25rem" /> Running...
        </div>
      </div>
      <div
        v-if="!logCollapsed"
        class="cell-output-resize"
        @pointerdown="startLogResize"
      ></div>
    </div>

    <!-- HTML Output Panel -->
    <OutputPanel
      v-if="htmlOutput"
      :html-content="htmlOutput"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from '@codemirror/view'
import OutputPanel from './OutputPanel.vue'
import { GripVertical, Plus, Square, Play, X, ChevronDown } from 'lucide-vue-next'
import { EditorState } from '@codemirror/state'
import { defaultKeymap, history, historyKeymap, insertNewlineAndIndent, indentMore } from '@codemirror/commands'
import { syntaxHighlighting, HighlightStyle, bracketMatching, foldGutter } from '@codemirror/language'
import { autocompletion, completeFromList, acceptCompletion, startCompletion } from '@codemirror/autocomplete'
import { cpp } from '@codemirror/lang-cpp'
import { StreamLanguage } from '@codemirror/language'
import { shell } from '@codemirror/legacy-modes/mode/shell'
import { python as pythonLegacy } from '@codemirror/legacy-modes/mode/python'
import { tags } from '@lezer/highlight'
import axios from 'axios'

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
  // CM library - auto-generated
  { label: 'cm.data', type: 'module', detail: 'Chemical Machines Data Package' },
  { label: 'from cm.data import', type: 'text', detail: 'import CM data' },
  // CM data - classes
  { label: 'BenchmarkMolecule', type: 'class', detail: 'cm.data: BenchmarkMolecule' },
  { label: 'BenchmarkProperty', type: 'class', detail: 'cm.data: BenchmarkProperty' },
  { label: 'ComparisonResult', type: 'class', detail: 'cm.data: ComparisonResult' },
  { label: 'PropertyComparison', type: 'class', detail: 'cm.data: PropertyComparison' },
  { label: 'MoleculeStatus', type: 'class', detail: 'cm.data: MoleculeStatus' },
  { label: 'BenchmarkError', type: 'class', detail: 'cm.data: BenchmarkError' },
  { label: 'ServiceUnavailableError', type: 'class', detail: 'cm.data: ServiceUnavailableError' },
  { label: 'APIError', type: 'class', detail: 'cm.data: APIError' },
  { label: 'JobError', type: 'class', detail: 'cm.data: JobError' },
  { label: 'IndexingInProgressError', type: 'class', detail: 'cm.data: IndexingInProgressError' },
  { label: 'MoleculeNotFoundError', type: 'class', detail: 'cm.data: MoleculeNotFoundError' },
  { label: 'NoIndexError', type: 'class', detail: 'cm.data: NoIndexError' },
  // CM data - functions
  { label: 'search', type: 'function', detail: 'cm.data: search' },
  { label: 'get', type: 'function', detail: 'cm.data: get' },
  { label: 'compare', type: 'function', detail: 'cm.data: compare' },
  { label: 'sync', type: 'function', detail: 'cm.data: sync' },
  { label: 'stats', type: 'function', detail: 'cm.data: stats' },
  { label: 'status', type: 'function', detail: 'cm.data: status' },
  { label: 'cm.qm', type: 'module', detail: 'Chemical Machines Quantum Mechanics Package' },
  { label: 'from cm.qm import', type: 'text', detail: 'import CM qm' },
  // CM qm - constants
  { label: 'ATOMIC_NUMBERS', type: 'constant', detail: 'cm.qm: atomic numbers' },
  { label: 'ELEMENT_SYMBOLS', type: 'constant', detail: 'cm.qm: element symbols' },
  { label: 'AUFBAU_ORDER', type: 'constant', detail: 'cm.qm: aufbau order' },
  // CM qm - classes
  { label: 'CoordinateType', type: 'class', detail: 'cm.qm: CoordinateType' },
  { label: 'Coordinate3D', type: 'class', detail: 'cm.qm: Coordinate3D' },
  { label: 'SpinOrbital', type: 'class', detail: 'cm.qm: SpinOrbital' },
  { label: 'SlaterDeterminant', type: 'class', detail: 'cm.qm: SlaterDeterminant' },
  { label: 'Operator', type: 'class', detail: 'cm.qm: Operator' },
  { label: 'Overlap', type: 'class', detail: 'cm.qm: Overlap' },
  { label: 'MatrixElement', type: 'class', detail: 'cm.qm: MatrixElement' },
  { label: 'DiracSpinor', type: 'class', detail: 'cm.qm: DiracSpinor' },
  { label: 'DiracDeterminant', type: 'class', detail: 'cm.qm: DiracDeterminant' },
  { label: 'RelativisticOperator', type: 'class', detail: 'cm.qm: RelativisticOperator' },
  { label: 'ElectronConfiguration', type: 'class', detail: 'cm.qm: ElectronConfiguration' },
  { label: 'ElectronConfiguration.aufbau', type: 'method', detail: 'ElectronConfiguration method: aufbau()' },
  { label: 'ElectronConfiguration.manual', type: 'method', detail: 'ElectronConfiguration method: manual()' },
  { label: 'ElectronConfiguration.from_string', type: 'method', detail: 'ElectronConfiguration method: from_string()' },
  { label: 'ElectronConfiguration.n_electrons', type: 'method', detail: 'ElectronConfiguration method: n_electrons()' },
  { label: 'ElectronConfiguration.orbitals', type: 'method', detail: 'ElectronConfiguration method: orbitals()' },
  { label: 'ElectronConfiguration.subshell_occupancy', type: 'method', detail: 'ElectronConfiguration method: subshell_occupancy()' },
  { label: 'ElectronConfiguration.shell_occupancy', type: 'method', detail: 'ElectronConfiguration method: shell_occupancy()' },
  { label: 'ElectronConfiguration.label', type: 'method', detail: 'ElectronConfiguration method: label()' },
  { label: 'ElectronConfiguration.latex_label', type: 'method', detail: 'ElectronConfiguration method: latex_label()' },
  { label: 'ElectronConfiguration.to_latex', type: 'method', detail: 'ElectronConfiguration method: to_latex()' },
  { label: 'ElectronConfiguration.excite', type: 'method', detail: 'ElectronConfiguration method: excite()' },
  { label: 'ElectronConfiguration.ionize', type: 'method', detail: 'ElectronConfiguration method: ionize()' },
  { label: 'ElectronConfiguration.orbital_energy_key', type: 'method', detail: 'ElectronConfiguration method: orbital_energy_key()' },
  { label: 'ElectronConfiguration.add_electron', type: 'method', detail: 'ElectronConfiguration method: add_electron()' },
  { label: 'ElectronConfiguration.to_spinors', type: 'method', detail: 'ElectronConfiguration method: to_spinors()' },
  { label: 'ElectronConfiguration.spinors', type: 'method', detail: 'ElectronConfiguration method: spinors()' },
  { label: 'ElectronConfiguration.symbol', type: 'method', detail: 'ElectronConfiguration method: symbol()' },
  { label: 'ElectronConfiguration.name', type: 'method', detail: 'ElectronConfiguration method: name()' },
  { label: 'ElectronConfiguration.charge', type: 'method', detail: 'ElectronConfiguration method: charge()' },
  { label: 'ElectronConfiguration.is_ion', type: 'method', detail: 'ElectronConfiguration method: is_ion()' },
  { label: 'ElectronConfiguration.position', type: 'method', detail: 'ElectronConfiguration method: position()' },
  { label: 'ElectronConfiguration.x', type: 'method', detail: 'ElectronConfiguration method: x()' },
  { label: 'ElectronConfiguration.y', type: 'method', detail: 'ElectronConfiguration method: y()' },
  { label: 'ElectronConfiguration.z', type: 'method', detail: 'ElectronConfiguration method: z()' },
  { label: 'ElectronConfiguration.is_symbolic', type: 'method', detail: 'ElectronConfiguration method: is_symbolic()' },
  { label: 'ElectronConfiguration.numeric_position', type: 'method', detail: 'ElectronConfiguration method: numeric_position()' },
  { label: 'ElectronConfiguration.configuration', type: 'method', detail: 'ElectronConfiguration method: configuration()' },
  { label: 'ElectronConfiguration.relativistic', type: 'method', detail: 'ElectronConfiguration method: relativistic()' },
  { label: 'ElectronConfiguration.vec3', type: 'method', detail: 'ElectronConfiguration method: vec3()' },
  { label: 'ElectronConfiguration.slater_determinant', type: 'method', detail: 'ElectronConfiguration method: slater_determinant()' },
  { label: 'ElectronConfiguration.dirac_determinant', type: 'method', detail: 'ElectronConfiguration method: dirac_determinant()' },
  { label: 'ElectronConfiguration.determinant', type: 'method', detail: 'ElectronConfiguration method: determinant()' },
  { label: 'ElectronConfiguration.to_sympy', type: 'method', detail: 'ElectronConfiguration method: to_sympy()' },
  { label: 'ElectronConfiguration.to_molecule_tuple', type: 'method', detail: 'ElectronConfiguration method: to_molecule_tuple()' },
  { label: 'ElectronConfiguration.render', type: 'method', detail: 'ElectronConfiguration method: render()' },
  { label: 'ElectronConfiguration.render_configuration', type: 'method', detail: 'ElectronConfiguration method: render_configuration()' },
  { label: 'ElectronConfiguration.with_position', type: 'method', detail: 'ElectronConfiguration method: with_position()' },
  { label: 'ElectronConfiguration.with_configuration', type: 'method', detail: 'ElectronConfiguration method: with_configuration()' },
  { label: 'ElectronConfiguration.to_relativistic', type: 'method', detail: 'ElectronConfiguration method: to_relativistic()' },
  { label: 'ElectronConfiguration.to_nonrelativistic', type: 'method', detail: 'ElectronConfiguration method: to_nonrelativistic()' },
  { label: 'ElectronConfiguration.ion_label', type: 'method', detail: 'ElectronConfiguration method: ion_label()' },
  { label: 'ElectronConfiguration.ket_label', type: 'method', detail: 'ElectronConfiguration method: ket_label()' },
  { label: 'ElectronConfiguration.energy', type: 'method', detail: 'ElectronConfiguration method: energy()' },
  { label: 'Atom', type: 'class', detail: 'cm.qm: Atom' },
  { label: 'Atom.aufbau', type: 'method', detail: 'Atom method: aufbau()' },
  { label: 'Atom.manual', type: 'method', detail: 'Atom method: manual()' },
  { label: 'Atom.from_string', type: 'method', detail: 'Atom method: from_string()' },
  { label: 'Atom.n_electrons', type: 'method', detail: 'Atom method: n_electrons()' },
  { label: 'Atom.orbitals', type: 'method', detail: 'Atom method: orbitals()' },
  { label: 'Atom.subshell_occupancy', type: 'method', detail: 'Atom method: subshell_occupancy()' },
  { label: 'Atom.shell_occupancy', type: 'method', detail: 'Atom method: shell_occupancy()' },
  { label: 'Atom.label', type: 'method', detail: 'Atom method: label()' },
  { label: 'Atom.latex_label', type: 'method', detail: 'Atom method: latex_label()' },
  { label: 'Atom.to_latex', type: 'method', detail: 'Atom method: to_latex()' },
  { label: 'Atom.excite', type: 'method', detail: 'Atom method: excite()' },
  { label: 'Atom.ionize', type: 'method', detail: 'Atom method: ionize()' },
  { label: 'Atom.orbital_energy_key', type: 'method', detail: 'Atom method: orbital_energy_key()' },
  { label: 'Atom.add_electron', type: 'method', detail: 'Atom method: add_electron()' },
  { label: 'Atom.to_spinors', type: 'method', detail: 'Atom method: to_spinors()' },
  { label: 'Atom.spinors', type: 'method', detail: 'Atom method: spinors()' },
  { label: 'Atom.symbol', type: 'method', detail: 'Atom method: symbol()' },
  { label: 'Atom.name', type: 'method', detail: 'Atom method: name()' },
  { label: 'Atom.charge', type: 'method', detail: 'Atom method: charge()' },
  { label: 'Atom.is_ion', type: 'method', detail: 'Atom method: is_ion()' },
  { label: 'Atom.position', type: 'method', detail: 'Atom method: position()' },
  { label: 'Atom.x', type: 'method', detail: 'Atom method: x()' },
  { label: 'Atom.y', type: 'method', detail: 'Atom method: y()' },
  { label: 'Atom.z', type: 'method', detail: 'Atom method: z()' },
  { label: 'Atom.is_symbolic', type: 'method', detail: 'Atom method: is_symbolic()' },
  { label: 'Atom.numeric_position', type: 'method', detail: 'Atom method: numeric_position()' },
  { label: 'Atom.configuration', type: 'method', detail: 'Atom method: configuration()' },
  { label: 'Atom.relativistic', type: 'method', detail: 'Atom method: relativistic()' },
  { label: 'Atom.vec3', type: 'method', detail: 'Atom method: vec3()' },
  { label: 'Atom.slater_determinant', type: 'method', detail: 'Atom method: slater_determinant()' },
  { label: 'Atom.dirac_determinant', type: 'method', detail: 'Atom method: dirac_determinant()' },
  { label: 'Atom.determinant', type: 'method', detail: 'Atom method: determinant()' },
  { label: 'Atom.to_sympy', type: 'method', detail: 'Atom method: to_sympy()' },
  { label: 'Atom.to_molecule_tuple', type: 'method', detail: 'Atom method: to_molecule_tuple()' },
  { label: 'Atom.render', type: 'method', detail: 'Atom method: render()' },
  { label: 'Atom.render_configuration', type: 'method', detail: 'Atom method: render_configuration()' },
  { label: 'Atom.with_position', type: 'method', detail: 'Atom method: with_position()' },
  { label: 'Atom.with_configuration', type: 'method', detail: 'Atom method: with_configuration()' },
  { label: 'Atom.to_relativistic', type: 'method', detail: 'Atom method: to_relativistic()' },
  { label: 'Atom.to_nonrelativistic', type: 'method', detail: 'Atom method: to_nonrelativistic()' },
  { label: 'Atom.ion_label', type: 'method', detail: 'Atom method: ion_label()' },
  { label: 'Atom.ket_label', type: 'method', detail: 'Atom method: ket_label()' },
  { label: 'Atom.energy', type: 'method', detail: 'Atom method: energy()' },
  { label: 'Molecule', type: 'class', detail: 'cm.qm: Molecule' },
  { label: 'Molecule.atoms', type: 'method', detail: 'Molecule method: atoms()' },
  { label: 'Molecule.positions', type: 'method', detail: 'Molecule method: positions()' },
  { label: 'Molecule.n_atoms', type: 'method', detail: 'Molecule method: n_atoms()' },
  { label: 'Molecule.n_electrons', type: 'method', detail: 'Molecule method: n_electrons()' },
  { label: 'Molecule.nuclear_charges', type: 'method', detail: 'Molecule method: nuclear_charges()' },
  { label: 'Molecule.total_nuclear_charge', type: 'method', detail: 'Molecule method: total_nuclear_charge()' },
  { label: 'Molecule.geometry_variables', type: 'method', detail: 'Molecule method: geometry_variables()' },
  { label: 'Molecule.is_symbolic', type: 'method', detail: 'Molecule method: is_symbolic()' },
  { label: 'Molecule.atom_at', type: 'method', detail: 'Molecule method: atom_at()' },
  { label: 'Molecule.atoms_with_positions', type: 'method', detail: 'Molecule method: atoms_with_positions()' },
  { label: 'Molecule.slater_determinant', type: 'method', detail: 'Molecule method: slater_determinant()' },
  { label: 'Molecule.ci_basis', type: 'method', detail: 'Molecule method: ci_basis()' },
  { label: 'Molecule.with_geometry', type: 'method', detail: 'Molecule method: with_geometry()' },
  { label: 'Molecule.bond_length', type: 'method', detail: 'Molecule method: bond_length()' },
  { label: 'Molecule.bond_angle', type: 'method', detail: 'Molecule method: bond_angle()' },
  { label: 'Molecule.energy', type: 'method', detail: 'Molecule method: energy()' },
  { label: 'Molecule.render', type: 'method', detail: 'Molecule method: render()' },
  { label: 'Molecule.to_xyz', type: 'method', detail: 'Molecule method: to_xyz()' },
  { label: 'HamiltonianTerm', type: 'class', detail: 'cm.qm: HamiltonianTerm' },
  { label: 'HamiltonianTerm.to_latex', type: 'method', detail: 'HamiltonianTerm method: to_latex()' },
  { label: 'HamiltonianTerm.with_kinetic', type: 'method', detail: 'HamiltonianTerm method: with_kinetic()' },
  { label: 'HamiltonianTerm.with_nuclear_attraction', type: 'method', detail: 'HamiltonianTerm method: with_nuclear_attraction()' },
  { label: 'HamiltonianTerm.with_coulomb', type: 'method', detail: 'HamiltonianTerm method: with_coulomb()' },
  { label: 'HamiltonianTerm.with_spin_orbit', type: 'method', detail: 'HamiltonianTerm method: with_spin_orbit()' },
  { label: 'HamiltonianTerm.with_relativistic', type: 'method', detail: 'HamiltonianTerm method: with_relativistic()' },
  { label: 'HamiltonianTerm.with_external_field', type: 'method', detail: 'HamiltonianTerm method: with_external_field()' },
  { label: 'HamiltonianTerm.with_custom', type: 'method', detail: 'HamiltonianTerm method: with_custom()' },
  { label: 'HamiltonianTerm.scale', type: 'method', detail: 'HamiltonianTerm method: scale()' },
  { label: 'HamiltonianTerm.remove', type: 'method', detail: 'HamiltonianTerm method: remove()' },
  { label: 'HamiltonianTerm.with_basis', type: 'method', detail: 'HamiltonianTerm method: with_basis()' },
  { label: 'HamiltonianTerm.electronic', type: 'method', detail: 'HamiltonianTerm method: electronic()' },
  { label: 'HamiltonianTerm.spin_orbit', type: 'method', detail: 'HamiltonianTerm method: spin_orbit()' },
  { label: 'HamiltonianTerm.relativistic', type: 'method', detail: 'HamiltonianTerm method: relativistic()' },
  { label: 'HamiltonianTerm.build', type: 'method', detail: 'HamiltonianTerm method: build()' },
  { label: 'HamiltonianTerm.terms', type: 'method', detail: 'HamiltonianTerm method: terms()' },
  { label: 'HamiltonianTerm.render', type: 'method', detail: 'HamiltonianTerm method: render()' },
  { label: 'HamiltonianTerm.element', type: 'method', detail: 'HamiltonianTerm method: element()' },
  { label: 'HamiltonianTerm.diagonal', type: 'method', detail: 'HamiltonianTerm method: diagonal()' },
  { label: 'HamiltonianTerm.matrix', type: 'method', detail: 'HamiltonianTerm method: matrix()' },
  { label: 'HamiltonianTerm.term_names', type: 'method', detail: 'HamiltonianTerm method: term_names()' },
  { label: 'HamiltonianTerm.is_relativistic', type: 'method', detail: 'HamiltonianTerm method: is_relativistic()' },
  { label: 'HamiltonianTerm.n_body_max', type: 'method', detail: 'HamiltonianTerm method: n_body_max()' },
  { label: 'HamiltonianTerm.has_term', type: 'method', detail: 'HamiltonianTerm method: has_term()' },
  { label: 'HamiltonianTerm.molecule', type: 'method', detail: 'HamiltonianTerm method: molecule()' },
  { label: 'HamiltonianTerm.basis_name', type: 'method', detail: 'HamiltonianTerm method: basis_name()' },
  { label: 'HamiltonianTerm.uses_gaussian_integrals', type: 'method', detail: 'HamiltonianTerm method: uses_gaussian_integrals()' },
  { label: 'HamiltonianTerm.hartree_fock', type: 'method', detail: 'HamiltonianTerm method: hartree_fock()' },
  { label: 'HamiltonianTerm.to_expr', type: 'method', detail: 'HamiltonianTerm method: to_expr()' },
  { label: 'HamiltonianTerm.to_sympy', type: 'method', detail: 'HamiltonianTerm method: to_sympy()' },
  { label: 'HamiltonianTerm.analytical', type: 'method', detail: 'HamiltonianTerm method: analytical()' },
  { label: 'HamiltonianTerm.numerical', type: 'method', detail: 'HamiltonianTerm method: numerical()' },
  { label: 'HamiltonianTerm.graph', type: 'method', detail: 'HamiltonianTerm method: graph()' },
  { label: 'HamiltonianTerm.compile', type: 'method', detail: 'HamiltonianTerm method: compile()' },
  { label: 'HamiltonianTerm.bra', type: 'method', detail: 'HamiltonianTerm method: bra()' },
  { label: 'HamiltonianTerm.ket', type: 'method', detail: 'HamiltonianTerm method: ket()' },
  { label: 'HamiltonianTerm.hamiltonian', type: 'method', detail: 'HamiltonianTerm method: hamiltonian()' },
  { label: 'HamiltonianTerm.n_excitations', type: 'method', detail: 'HamiltonianTerm method: n_excitations()' },
  { label: 'HamiltonianTerm.is_zero', type: 'method', detail: 'HamiltonianTerm method: is_zero()' },
  { label: 'HamiltonianTerm.is_diagonal', type: 'method', detail: 'HamiltonianTerm method: is_diagonal()' },
  { label: 'HamiltonianTerm.shape', type: 'method', detail: 'HamiltonianTerm method: shape()' },
  { label: 'HamiltonianTerm.basis', type: 'method', detail: 'HamiltonianTerm method: basis()' },
  { label: 'HamiltonianTerm.n_basis', type: 'method', detail: 'HamiltonianTerm method: n_basis()' },
  { label: 'HamiltonianTerm.eigenvalues', type: 'method', detail: 'HamiltonianTerm method: eigenvalues()' },
  { label: 'HamiltonianTerm.eigenvectors', type: 'method', detail: 'HamiltonianTerm method: eigenvectors()' },
  { label: 'HamiltonianTerm.ground_state_energy', type: 'method', detail: 'HamiltonianTerm method: ground_state_energy()' },
  { label: 'HamiltonianTerm.diagonalize', type: 'method', detail: 'HamiltonianTerm method: diagonalize()' },
  { label: 'HamiltonianBuilder', type: 'class', detail: 'cm.qm: HamiltonianBuilder' },
  { label: 'HamiltonianBuilder.to_latex', type: 'method', detail: 'HamiltonianBuilder method: to_latex()' },
  { label: 'HamiltonianBuilder.with_kinetic', type: 'method', detail: 'HamiltonianBuilder method: with_kinetic()' },
  { label: 'HamiltonianBuilder.with_nuclear_attraction', type: 'method', detail: 'HamiltonianBuilder method: with_nuclear_attraction()' },
  { label: 'HamiltonianBuilder.with_coulomb', type: 'method', detail: 'HamiltonianBuilder method: with_coulomb()' },
  { label: 'HamiltonianBuilder.with_spin_orbit', type: 'method', detail: 'HamiltonianBuilder method: with_spin_orbit()' },
  { label: 'HamiltonianBuilder.with_relativistic', type: 'method', detail: 'HamiltonianBuilder method: with_relativistic()' },
  { label: 'HamiltonianBuilder.with_external_field', type: 'method', detail: 'HamiltonianBuilder method: with_external_field()' },
  { label: 'HamiltonianBuilder.with_custom', type: 'method', detail: 'HamiltonianBuilder method: with_custom()' },
  { label: 'HamiltonianBuilder.scale', type: 'method', detail: 'HamiltonianBuilder method: scale()' },
  { label: 'HamiltonianBuilder.remove', type: 'method', detail: 'HamiltonianBuilder method: remove()' },
  { label: 'HamiltonianBuilder.with_basis', type: 'method', detail: 'HamiltonianBuilder method: with_basis()' },
  { label: 'HamiltonianBuilder.electronic', type: 'method', detail: 'HamiltonianBuilder method: electronic()' },
  { label: 'HamiltonianBuilder.spin_orbit', type: 'method', detail: 'HamiltonianBuilder method: spin_orbit()' },
  { label: 'HamiltonianBuilder.relativistic', type: 'method', detail: 'HamiltonianBuilder method: relativistic()' },
  { label: 'HamiltonianBuilder.build', type: 'method', detail: 'HamiltonianBuilder method: build()' },
  { label: 'HamiltonianBuilder.terms', type: 'method', detail: 'HamiltonianBuilder method: terms()' },
  { label: 'HamiltonianBuilder.render', type: 'method', detail: 'HamiltonianBuilder method: render()' },
  { label: 'HamiltonianBuilder.element', type: 'method', detail: 'HamiltonianBuilder method: element()' },
  { label: 'HamiltonianBuilder.diagonal', type: 'method', detail: 'HamiltonianBuilder method: diagonal()' },
  { label: 'HamiltonianBuilder.matrix', type: 'method', detail: 'HamiltonianBuilder method: matrix()' },
  { label: 'HamiltonianBuilder.term_names', type: 'method', detail: 'HamiltonianBuilder method: term_names()' },
  { label: 'HamiltonianBuilder.is_relativistic', type: 'method', detail: 'HamiltonianBuilder method: is_relativistic()' },
  { label: 'HamiltonianBuilder.n_body_max', type: 'method', detail: 'HamiltonianBuilder method: n_body_max()' },
  { label: 'HamiltonianBuilder.has_term', type: 'method', detail: 'HamiltonianBuilder method: has_term()' },
  { label: 'HamiltonianBuilder.molecule', type: 'method', detail: 'HamiltonianBuilder method: molecule()' },
  { label: 'HamiltonianBuilder.basis_name', type: 'method', detail: 'HamiltonianBuilder method: basis_name()' },
  { label: 'HamiltonianBuilder.uses_gaussian_integrals', type: 'method', detail: 'HamiltonianBuilder method: uses_gaussian_integrals()' },
  { label: 'HamiltonianBuilder.hartree_fock', type: 'method', detail: 'HamiltonianBuilder method: hartree_fock()' },
  { label: 'HamiltonianBuilder.to_expr', type: 'method', detail: 'HamiltonianBuilder method: to_expr()' },
  { label: 'HamiltonianBuilder.to_sympy', type: 'method', detail: 'HamiltonianBuilder method: to_sympy()' },
  { label: 'HamiltonianBuilder.analytical', type: 'method', detail: 'HamiltonianBuilder method: analytical()' },
  { label: 'HamiltonianBuilder.numerical', type: 'method', detail: 'HamiltonianBuilder method: numerical()' },
  { label: 'HamiltonianBuilder.graph', type: 'method', detail: 'HamiltonianBuilder method: graph()' },
  { label: 'HamiltonianBuilder.compile', type: 'method', detail: 'HamiltonianBuilder method: compile()' },
  { label: 'HamiltonianBuilder.bra', type: 'method', detail: 'HamiltonianBuilder method: bra()' },
  { label: 'HamiltonianBuilder.ket', type: 'method', detail: 'HamiltonianBuilder method: ket()' },
  { label: 'HamiltonianBuilder.hamiltonian', type: 'method', detail: 'HamiltonianBuilder method: hamiltonian()' },
  { label: 'HamiltonianBuilder.n_excitations', type: 'method', detail: 'HamiltonianBuilder method: n_excitations()' },
  { label: 'HamiltonianBuilder.is_zero', type: 'method', detail: 'HamiltonianBuilder method: is_zero()' },
  { label: 'HamiltonianBuilder.is_diagonal', type: 'method', detail: 'HamiltonianBuilder method: is_diagonal()' },
  { label: 'HamiltonianBuilder.shape', type: 'method', detail: 'HamiltonianBuilder method: shape()' },
  { label: 'HamiltonianBuilder.basis', type: 'method', detail: 'HamiltonianBuilder method: basis()' },
  { label: 'HamiltonianBuilder.n_basis', type: 'method', detail: 'HamiltonianBuilder method: n_basis()' },
  { label: 'HamiltonianBuilder.eigenvalues', type: 'method', detail: 'HamiltonianBuilder method: eigenvalues()' },
  { label: 'HamiltonianBuilder.eigenvectors', type: 'method', detail: 'HamiltonianBuilder method: eigenvectors()' },
  { label: 'HamiltonianBuilder.ground_state_energy', type: 'method', detail: 'HamiltonianBuilder method: ground_state_energy()' },
  { label: 'HamiltonianBuilder.diagonalize', type: 'method', detail: 'HamiltonianBuilder method: diagonalize()' },
  { label: 'MolecularHamiltonian', type: 'class', detail: 'cm.qm: MolecularHamiltonian' },
  { label: 'MolecularHamiltonian.to_latex', type: 'method', detail: 'MolecularHamiltonian method: to_latex()' },
  { label: 'MolecularHamiltonian.with_kinetic', type: 'method', detail: 'MolecularHamiltonian method: with_kinetic()' },
  { label: 'MolecularHamiltonian.with_nuclear_attraction', type: 'method', detail: 'MolecularHamiltonian method: with_nuclear_attraction()' },
  { label: 'MolecularHamiltonian.with_coulomb', type: 'method', detail: 'MolecularHamiltonian method: with_coulomb()' },
  { label: 'MolecularHamiltonian.with_spin_orbit', type: 'method', detail: 'MolecularHamiltonian method: with_spin_orbit()' },
  { label: 'MolecularHamiltonian.with_relativistic', type: 'method', detail: 'MolecularHamiltonian method: with_relativistic()' },
  { label: 'MolecularHamiltonian.with_external_field', type: 'method', detail: 'MolecularHamiltonian method: with_external_field()' },
  { label: 'MolecularHamiltonian.with_custom', type: 'method', detail: 'MolecularHamiltonian method: with_custom()' },
  { label: 'MolecularHamiltonian.scale', type: 'method', detail: 'MolecularHamiltonian method: scale()' },
  { label: 'MolecularHamiltonian.remove', type: 'method', detail: 'MolecularHamiltonian method: remove()' },
  { label: 'MolecularHamiltonian.with_basis', type: 'method', detail: 'MolecularHamiltonian method: with_basis()' },
  { label: 'MolecularHamiltonian.electronic', type: 'method', detail: 'MolecularHamiltonian method: electronic()' },
  { label: 'MolecularHamiltonian.spin_orbit', type: 'method', detail: 'MolecularHamiltonian method: spin_orbit()' },
  { label: 'MolecularHamiltonian.relativistic', type: 'method', detail: 'MolecularHamiltonian method: relativistic()' },
  { label: 'MolecularHamiltonian.build', type: 'method', detail: 'MolecularHamiltonian method: build()' },
  { label: 'MolecularHamiltonian.terms', type: 'method', detail: 'MolecularHamiltonian method: terms()' },
  { label: 'MolecularHamiltonian.render', type: 'method', detail: 'MolecularHamiltonian method: render()' },
  { label: 'MolecularHamiltonian.element', type: 'method', detail: 'MolecularHamiltonian method: element()' },
  { label: 'MolecularHamiltonian.diagonal', type: 'method', detail: 'MolecularHamiltonian method: diagonal()' },
  { label: 'MolecularHamiltonian.matrix', type: 'method', detail: 'MolecularHamiltonian method: matrix()' },
  { label: 'MolecularHamiltonian.term_names', type: 'method', detail: 'MolecularHamiltonian method: term_names()' },
  { label: 'MolecularHamiltonian.is_relativistic', type: 'method', detail: 'MolecularHamiltonian method: is_relativistic()' },
  { label: 'MolecularHamiltonian.n_body_max', type: 'method', detail: 'MolecularHamiltonian method: n_body_max()' },
  { label: 'MolecularHamiltonian.has_term', type: 'method', detail: 'MolecularHamiltonian method: has_term()' },
  { label: 'MolecularHamiltonian.molecule', type: 'method', detail: 'MolecularHamiltonian method: molecule()' },
  { label: 'MolecularHamiltonian.basis_name', type: 'method', detail: 'MolecularHamiltonian method: basis_name()' },
  { label: 'MolecularHamiltonian.uses_gaussian_integrals', type: 'method', detail: 'MolecularHamiltonian method: uses_gaussian_integrals()' },
  { label: 'MolecularHamiltonian.hartree_fock', type: 'method', detail: 'MolecularHamiltonian method: hartree_fock()' },
  { label: 'MolecularHamiltonian.to_expr', type: 'method', detail: 'MolecularHamiltonian method: to_expr()' },
  { label: 'MolecularHamiltonian.to_sympy', type: 'method', detail: 'MolecularHamiltonian method: to_sympy()' },
  { label: 'MolecularHamiltonian.analytical', type: 'method', detail: 'MolecularHamiltonian method: analytical()' },
  { label: 'MolecularHamiltonian.numerical', type: 'method', detail: 'MolecularHamiltonian method: numerical()' },
  { label: 'MolecularHamiltonian.graph', type: 'method', detail: 'MolecularHamiltonian method: graph()' },
  { label: 'MolecularHamiltonian.compile', type: 'method', detail: 'MolecularHamiltonian method: compile()' },
  { label: 'MolecularHamiltonian.bra', type: 'method', detail: 'MolecularHamiltonian method: bra()' },
  { label: 'MolecularHamiltonian.ket', type: 'method', detail: 'MolecularHamiltonian method: ket()' },
  { label: 'MolecularHamiltonian.hamiltonian', type: 'method', detail: 'MolecularHamiltonian method: hamiltonian()' },
  { label: 'MolecularHamiltonian.n_excitations', type: 'method', detail: 'MolecularHamiltonian method: n_excitations()' },
  { label: 'MolecularHamiltonian.is_zero', type: 'method', detail: 'MolecularHamiltonian method: is_zero()' },
  { label: 'MolecularHamiltonian.is_diagonal', type: 'method', detail: 'MolecularHamiltonian method: is_diagonal()' },
  { label: 'MolecularHamiltonian.shape', type: 'method', detail: 'MolecularHamiltonian method: shape()' },
  { label: 'MolecularHamiltonian.basis', type: 'method', detail: 'MolecularHamiltonian method: basis()' },
  { label: 'MolecularHamiltonian.n_basis', type: 'method', detail: 'MolecularHamiltonian method: n_basis()' },
  { label: 'MolecularHamiltonian.eigenvalues', type: 'method', detail: 'MolecularHamiltonian method: eigenvalues()' },
  { label: 'MolecularHamiltonian.eigenvectors', type: 'method', detail: 'MolecularHamiltonian method: eigenvectors()' },
  { label: 'MolecularHamiltonian.ground_state_energy', type: 'method', detail: 'MolecularHamiltonian method: ground_state_energy()' },
  { label: 'MolecularHamiltonian.diagonalize', type: 'method', detail: 'MolecularHamiltonian method: diagonalize()' },
  { label: 'MatrixExpression', type: 'class', detail: 'cm.qm: MatrixExpression' },
  { label: 'MatrixExpression.to_latex', type: 'method', detail: 'MatrixExpression method: to_latex()' },
  { label: 'MatrixExpression.with_kinetic', type: 'method', detail: 'MatrixExpression method: with_kinetic()' },
  { label: 'MatrixExpression.with_nuclear_attraction', type: 'method', detail: 'MatrixExpression method: with_nuclear_attraction()' },
  { label: 'MatrixExpression.with_coulomb', type: 'method', detail: 'MatrixExpression method: with_coulomb()' },
  { label: 'MatrixExpression.with_spin_orbit', type: 'method', detail: 'MatrixExpression method: with_spin_orbit()' },
  { label: 'MatrixExpression.with_relativistic', type: 'method', detail: 'MatrixExpression method: with_relativistic()' },
  { label: 'MatrixExpression.with_external_field', type: 'method', detail: 'MatrixExpression method: with_external_field()' },
  { label: 'MatrixExpression.with_custom', type: 'method', detail: 'MatrixExpression method: with_custom()' },
  { label: 'MatrixExpression.scale', type: 'method', detail: 'MatrixExpression method: scale()' },
  { label: 'MatrixExpression.remove', type: 'method', detail: 'MatrixExpression method: remove()' },
  { label: 'MatrixExpression.with_basis', type: 'method', detail: 'MatrixExpression method: with_basis()' },
  { label: 'MatrixExpression.electronic', type: 'method', detail: 'MatrixExpression method: electronic()' },
  { label: 'MatrixExpression.spin_orbit', type: 'method', detail: 'MatrixExpression method: spin_orbit()' },
  { label: 'MatrixExpression.relativistic', type: 'method', detail: 'MatrixExpression method: relativistic()' },
  { label: 'MatrixExpression.build', type: 'method', detail: 'MatrixExpression method: build()' },
  { label: 'MatrixExpression.terms', type: 'method', detail: 'MatrixExpression method: terms()' },
  { label: 'MatrixExpression.render', type: 'method', detail: 'MatrixExpression method: render()' },
  { label: 'MatrixExpression.element', type: 'method', detail: 'MatrixExpression method: element()' },
  { label: 'MatrixExpression.diagonal', type: 'method', detail: 'MatrixExpression method: diagonal()' },
  { label: 'MatrixExpression.matrix', type: 'method', detail: 'MatrixExpression method: matrix()' },
  { label: 'MatrixExpression.term_names', type: 'method', detail: 'MatrixExpression method: term_names()' },
  { label: 'MatrixExpression.is_relativistic', type: 'method', detail: 'MatrixExpression method: is_relativistic()' },
  { label: 'MatrixExpression.n_body_max', type: 'method', detail: 'MatrixExpression method: n_body_max()' },
  { label: 'MatrixExpression.has_term', type: 'method', detail: 'MatrixExpression method: has_term()' },
  { label: 'MatrixExpression.molecule', type: 'method', detail: 'MatrixExpression method: molecule()' },
  { label: 'MatrixExpression.basis_name', type: 'method', detail: 'MatrixExpression method: basis_name()' },
  { label: 'MatrixExpression.uses_gaussian_integrals', type: 'method', detail: 'MatrixExpression method: uses_gaussian_integrals()' },
  { label: 'MatrixExpression.hartree_fock', type: 'method', detail: 'MatrixExpression method: hartree_fock()' },
  { label: 'MatrixExpression.to_expr', type: 'method', detail: 'MatrixExpression method: to_expr()' },
  { label: 'MatrixExpression.to_sympy', type: 'method', detail: 'MatrixExpression method: to_sympy()' },
  { label: 'MatrixExpression.analytical', type: 'method', detail: 'MatrixExpression method: analytical()' },
  { label: 'MatrixExpression.numerical', type: 'method', detail: 'MatrixExpression method: numerical()' },
  { label: 'MatrixExpression.graph', type: 'method', detail: 'MatrixExpression method: graph()' },
  { label: 'MatrixExpression.compile', type: 'method', detail: 'MatrixExpression method: compile()' },
  { label: 'MatrixExpression.bra', type: 'method', detail: 'MatrixExpression method: bra()' },
  { label: 'MatrixExpression.ket', type: 'method', detail: 'MatrixExpression method: ket()' },
  { label: 'MatrixExpression.hamiltonian', type: 'method', detail: 'MatrixExpression method: hamiltonian()' },
  { label: 'MatrixExpression.n_excitations', type: 'method', detail: 'MatrixExpression method: n_excitations()' },
  { label: 'MatrixExpression.is_zero', type: 'method', detail: 'MatrixExpression method: is_zero()' },
  { label: 'MatrixExpression.is_diagonal', type: 'method', detail: 'MatrixExpression method: is_diagonal()' },
  { label: 'MatrixExpression.shape', type: 'method', detail: 'MatrixExpression method: shape()' },
  { label: 'MatrixExpression.basis', type: 'method', detail: 'MatrixExpression method: basis()' },
  { label: 'MatrixExpression.n_basis', type: 'method', detail: 'MatrixExpression method: n_basis()' },
  { label: 'MatrixExpression.eigenvalues', type: 'method', detail: 'MatrixExpression method: eigenvalues()' },
  { label: 'MatrixExpression.eigenvectors', type: 'method', detail: 'MatrixExpression method: eigenvectors()' },
  { label: 'MatrixExpression.ground_state_energy', type: 'method', detail: 'MatrixExpression method: ground_state_energy()' },
  { label: 'MatrixExpression.diagonalize', type: 'method', detail: 'MatrixExpression method: diagonalize()' },
  { label: 'HamiltonianMatrix', type: 'class', detail: 'cm.qm: HamiltonianMatrix' },
  { label: 'HamiltonianMatrix.to_latex', type: 'method', detail: 'HamiltonianMatrix method: to_latex()' },
  { label: 'HamiltonianMatrix.with_kinetic', type: 'method', detail: 'HamiltonianMatrix method: with_kinetic()' },
  { label: 'HamiltonianMatrix.with_nuclear_attraction', type: 'method', detail: 'HamiltonianMatrix method: with_nuclear_attraction()' },
  { label: 'HamiltonianMatrix.with_coulomb', type: 'method', detail: 'HamiltonianMatrix method: with_coulomb()' },
  { label: 'HamiltonianMatrix.with_spin_orbit', type: 'method', detail: 'HamiltonianMatrix method: with_spin_orbit()' },
  { label: 'HamiltonianMatrix.with_relativistic', type: 'method', detail: 'HamiltonianMatrix method: with_relativistic()' },
  { label: 'HamiltonianMatrix.with_external_field', type: 'method', detail: 'HamiltonianMatrix method: with_external_field()' },
  { label: 'HamiltonianMatrix.with_custom', type: 'method', detail: 'HamiltonianMatrix method: with_custom()' },
  { label: 'HamiltonianMatrix.scale', type: 'method', detail: 'HamiltonianMatrix method: scale()' },
  { label: 'HamiltonianMatrix.remove', type: 'method', detail: 'HamiltonianMatrix method: remove()' },
  { label: 'HamiltonianMatrix.with_basis', type: 'method', detail: 'HamiltonianMatrix method: with_basis()' },
  { label: 'HamiltonianMatrix.electronic', type: 'method', detail: 'HamiltonianMatrix method: electronic()' },
  { label: 'HamiltonianMatrix.spin_orbit', type: 'method', detail: 'HamiltonianMatrix method: spin_orbit()' },
  { label: 'HamiltonianMatrix.relativistic', type: 'method', detail: 'HamiltonianMatrix method: relativistic()' },
  { label: 'HamiltonianMatrix.build', type: 'method', detail: 'HamiltonianMatrix method: build()' },
  { label: 'HamiltonianMatrix.terms', type: 'method', detail: 'HamiltonianMatrix method: terms()' },
  { label: 'HamiltonianMatrix.render', type: 'method', detail: 'HamiltonianMatrix method: render()' },
  { label: 'HamiltonianMatrix.element', type: 'method', detail: 'HamiltonianMatrix method: element()' },
  { label: 'HamiltonianMatrix.diagonal', type: 'method', detail: 'HamiltonianMatrix method: diagonal()' },
  { label: 'HamiltonianMatrix.matrix', type: 'method', detail: 'HamiltonianMatrix method: matrix()' },
  { label: 'HamiltonianMatrix.term_names', type: 'method', detail: 'HamiltonianMatrix method: term_names()' },
  { label: 'HamiltonianMatrix.is_relativistic', type: 'method', detail: 'HamiltonianMatrix method: is_relativistic()' },
  { label: 'HamiltonianMatrix.n_body_max', type: 'method', detail: 'HamiltonianMatrix method: n_body_max()' },
  { label: 'HamiltonianMatrix.has_term', type: 'method', detail: 'HamiltonianMatrix method: has_term()' },
  { label: 'HamiltonianMatrix.molecule', type: 'method', detail: 'HamiltonianMatrix method: molecule()' },
  { label: 'HamiltonianMatrix.basis_name', type: 'method', detail: 'HamiltonianMatrix method: basis_name()' },
  { label: 'HamiltonianMatrix.uses_gaussian_integrals', type: 'method', detail: 'HamiltonianMatrix method: uses_gaussian_integrals()' },
  { label: 'HamiltonianMatrix.hartree_fock', type: 'method', detail: 'HamiltonianMatrix method: hartree_fock()' },
  { label: 'HamiltonianMatrix.to_expr', type: 'method', detail: 'HamiltonianMatrix method: to_expr()' },
  { label: 'HamiltonianMatrix.to_sympy', type: 'method', detail: 'HamiltonianMatrix method: to_sympy()' },
  { label: 'HamiltonianMatrix.analytical', type: 'method', detail: 'HamiltonianMatrix method: analytical()' },
  { label: 'HamiltonianMatrix.numerical', type: 'method', detail: 'HamiltonianMatrix method: numerical()' },
  { label: 'HamiltonianMatrix.graph', type: 'method', detail: 'HamiltonianMatrix method: graph()' },
  { label: 'HamiltonianMatrix.compile', type: 'method', detail: 'HamiltonianMatrix method: compile()' },
  { label: 'HamiltonianMatrix.bra', type: 'method', detail: 'HamiltonianMatrix method: bra()' },
  { label: 'HamiltonianMatrix.ket', type: 'method', detail: 'HamiltonianMatrix method: ket()' },
  { label: 'HamiltonianMatrix.hamiltonian', type: 'method', detail: 'HamiltonianMatrix method: hamiltonian()' },
  { label: 'HamiltonianMatrix.n_excitations', type: 'method', detail: 'HamiltonianMatrix method: n_excitations()' },
  { label: 'HamiltonianMatrix.is_zero', type: 'method', detail: 'HamiltonianMatrix method: is_zero()' },
  { label: 'HamiltonianMatrix.is_diagonal', type: 'method', detail: 'HamiltonianMatrix method: is_diagonal()' },
  { label: 'HamiltonianMatrix.shape', type: 'method', detail: 'HamiltonianMatrix method: shape()' },
  { label: 'HamiltonianMatrix.basis', type: 'method', detail: 'HamiltonianMatrix method: basis()' },
  { label: 'HamiltonianMatrix.n_basis', type: 'method', detail: 'HamiltonianMatrix method: n_basis()' },
  { label: 'HamiltonianMatrix.eigenvalues', type: 'method', detail: 'HamiltonianMatrix method: eigenvalues()' },
  { label: 'HamiltonianMatrix.eigenvectors', type: 'method', detail: 'HamiltonianMatrix method: eigenvectors()' },
  { label: 'HamiltonianMatrix.ground_state_energy', type: 'method', detail: 'HamiltonianMatrix method: ground_state_energy()' },
  { label: 'HamiltonianMatrix.diagonalize', type: 'method', detail: 'HamiltonianMatrix method: diagonalize()' },
  { label: 'BasisSet', type: 'class', detail: 'cm.qm: BasisSet' },
  { label: 'GaussianPrimitive', type: 'class', detail: 'cm.qm: GaussianPrimitive' },
  { label: 'ContractedGaussian', type: 'class', detail: 'cm.qm: ContractedGaussian' },
  { label: 'BasisFunction', type: 'class', detail: 'cm.qm: BasisFunction' },
  { label: 'HartreeFockSolver', type: 'class', detail: 'cm.qm: HartreeFockSolver' },
  { label: 'HFResult', type: 'class', detail: 'cm.qm: HFResult' },
  // CM qm - functions
  { label: 'coord3d', type: 'function', detail: 'cm.qm: coord3d' },
  { label: 'spherical_coord', type: 'function', detail: 'cm.qm: spherical_coord' },
  { label: 'cartesian_coord', type: 'function', detail: 'cm.qm: cartesian_coord' },
  { label: 'spin_orbital', type: 'function', detail: 'cm.qm: spin_orbital' },
  { label: 'basis_orbital', type: 'function', detail: 'cm.qm: basis_orbital' },
  { label: 'basis_orbitals', type: 'function', detail: 'cm.qm: basis_orbitals' },
  { label: 'slater', type: 'function', detail: 'cm.qm: slater' },
  { label: 'hamiltonian', type: 'function', detail: 'cm.qm: hamiltonian' },
  { label: 'one_electron_operator', type: 'function', detail: 'cm.qm: one_electron_operator' },
  { label: 'two_electron_operator', type: 'function', detail: 'cm.qm: two_electron_operator' },
  { label: 'dirac_spinor', type: 'function', detail: 'cm.qm: dirac_spinor' },
  { label: 'dirac_spinor_lj', type: 'function', detail: 'cm.qm: dirac_spinor_lj' },
  { label: 'basis_dirac', type: 'function', detail: 'cm.qm: basis_dirac' },
  { label: 'dirac_slater', type: 'function', detail: 'cm.qm: dirac_slater' },
  { label: 'dirac_hamiltonian', type: 'function', detail: 'cm.qm: dirac_hamiltonian' },
  { label: 'kappa_from_lj', type: 'function', detail: 'cm.qm: kappa_from_lj' },
  { label: 'atom', type: 'function', detail: 'cm.qm: atom' },
  { label: 'atoms', type: 'function', detail: 'cm.qm: atoms' },
  { label: 'ground_state', type: 'function', detail: 'cm.qm: ground_state' },
  { label: 'config_from_string', type: 'function', detail: 'cm.qm: config_from_string' },
  { label: 'molecule', type: 'function', detail: 'cm.qm: molecule' },
  { label: 'spherical_harmonic_orthogonality', type: 'function', detail: 'cm.qm: spherical_harmonic_orthogonality' },
  { label: 'overlap_matrix', type: 'function', detail: 'cm.qm: overlap_matrix' },
  { label: 'kinetic_matrix', type: 'function', detail: 'cm.qm: kinetic_matrix' },
  { label: 'nuclear_attraction_matrix', type: 'function', detail: 'cm.qm: nuclear_attraction_matrix' },
  { label: 'eri_tensor', type: 'function', detail: 'cm.qm: eri_tensor' },
  { label: 'hartree_fock', type: 'function', detail: 'cm.qm: hartree_fock' },
  { label: 'cm.symbols', type: 'module', detail: 'Chemical Machines Symbols Package' },
  { label: 'from cm.symbols import', type: 'text', detail: 'import CM symbols' },
  // CM symbols - classes
  { label: 'Math', type: 'class', detail: 'cm.symbols: Math' },
  { label: 'Math.var', type: 'method', detail: 'Math method: var()' },
  { label: 'Math.const', type: 'method', detail: 'Math method: const()' },
  { label: 'Math.pi', type: 'method', detail: 'Math method: pi()' },
  { label: 'Math.e', type: 'method', detail: 'Math method: e()' },
  { label: 'Math.inf', type: 'method', detail: 'Math method: inf()' },
  { label: 'Math.sqrt', type: 'method', detail: 'Math method: sqrt()' },
  { label: 'Math.sin', type: 'method', detail: 'Math method: sin()' },
  { label: 'Math.cos', type: 'method', detail: 'Math method: cos()' },
  { label: 'Math.tan', type: 'method', detail: 'Math method: tan()' },
  { label: 'Math.exp', type: 'method', detail: 'Math method: exp()' },
  { label: 'Math.log', type: 'method', detail: 'Math method: log()' },
  { label: 'Math.abs', type: 'method', detail: 'Math method: abs()' },
  { label: 'Math.expr', type: 'method', detail: 'Math method: expr()' },
  { label: 'Math.sum', type: 'method', detail: 'Math method: sum()' },
  { label: 'Math.prod', type: 'method', detail: 'Math method: prod()' },
  { label: 'Math.function', type: 'method', detail: 'Math method: function()' },
  { label: 'Math.get_function', type: 'method', detail: 'Math method: get_function()' },
  { label: 'Math.list_functions', type: 'method', detail: 'Math method: list_functions()' },
  { label: 'Math.gamma', type: 'method', detail: 'Math method: gamma()' },
  { label: 'Math.loggamma', type: 'method', detail: 'Math method: loggamma()' },
  { label: 'Math.digamma', type: 'method', detail: 'Math method: digamma()' },
  { label: 'Math.beta', type: 'method', detail: 'Math method: beta()' },
  { label: 'Math.factorial', type: 'method', detail: 'Math method: factorial()' },
  { label: 'Math.factorial2', type: 'method', detail: 'Math method: factorial2()' },
  { label: 'Math.binomial', type: 'method', detail: 'Math method: binomial()' },
  { label: 'Math.erf', type: 'method', detail: 'Math method: erf()' },
  { label: 'Math.erfc', type: 'method', detail: 'Math method: erfc()' },
  { label: 'Math.erfi', type: 'method', detail: 'Math method: erfi()' },
  { label: 'Math.besselj', type: 'method', detail: 'Math method: besselj()' },
  { label: 'Math.bessely', type: 'method', detail: 'Math method: bessely()' },
  { label: 'Math.besseli', type: 'method', detail: 'Math method: besseli()' },
  { label: 'Math.besselk', type: 'method', detail: 'Math method: besselk()' },
  { label: 'Math.jn', type: 'method', detail: 'Math method: jn()' },
  { label: 'Math.yn', type: 'method', detail: 'Math method: yn()' },
  { label: 'Math.hankel1', type: 'method', detail: 'Math method: hankel1()' },
  { label: 'Math.hankel2', type: 'method', detail: 'Math method: hankel2()' },
  { label: 'Math.airyai', type: 'method', detail: 'Math method: airyai()' },
  { label: 'Math.airybi', type: 'method', detail: 'Math method: airybi()' },
  { label: 'Math.airyaiprime', type: 'method', detail: 'Math method: airyaiprime()' },
  { label: 'Math.airybiprime', type: 'method', detail: 'Math method: airybiprime()' },
  { label: 'Math.legendre', type: 'method', detail: 'Math method: legendre()' },
  { label: 'Math.assoc_legendre', type: 'method', detail: 'Math method: assoc_legendre()' },
  { label: 'Math.hermite', type: 'method', detail: 'Math method: hermite()' },
  { label: 'Math.hermite_prob', type: 'method', detail: 'Math method: hermite_prob()' },
  { label: 'Math.laguerre', type: 'method', detail: 'Math method: laguerre()' },
  { label: 'Math.assoc_laguerre', type: 'method', detail: 'Math method: assoc_laguerre()' },
  { label: 'Math.chebyshevt', type: 'method', detail: 'Math method: chebyshevt()' },
  { label: 'Math.chebyshevu', type: 'method', detail: 'Math method: chebyshevu()' },
  { label: 'Math.gegenbauer', type: 'method', detail: 'Math method: gegenbauer()' },
  { label: 'Math.jacobi', type: 'method', detail: 'Math method: jacobi()' },
  { label: 'Math.hyper2f1', type: 'method', detail: 'Math method: hyper2f1()' },
  { label: 'Math.hyper1f1', type: 'method', detail: 'Math method: hyper1f1()' },
  { label: 'Math.hyper0f1', type: 'method', detail: 'Math method: hyper0f1()' },
  { label: 'Math.hyperpfq', type: 'method', detail: 'Math method: hyperpfq()' },
  { label: 'Math.elliptic_k', type: 'method', detail: 'Math method: elliptic_k()' },
  { label: 'Math.elliptic_e', type: 'method', detail: 'Math method: elliptic_e()' },
  { label: 'Math.elliptic_pi', type: 'method', detail: 'Math method: elliptic_pi()' },
  { label: 'Math.zeta', type: 'method', detail: 'Math method: zeta()' },
  { label: 'Math.polylog', type: 'method', detail: 'Math method: polylog()' },
  { label: 'Math.dirac', type: 'method', detail: 'Math method: dirac()' },
  { label: 'Math.heaviside', type: 'method', detail: 'Math method: heaviside()' },
  { label: 'Math.kronecker', type: 'method', detail: 'Math method: kronecker()' },
  { label: 'Math.levi_civita', type: 'method', detail: 'Math method: levi_civita()' },
  { label: 'Math.validate', type: 'method', detail: 'Math method: validate()' },
  { label: 'Math.coerce', type: 'method', detail: 'Math method: coerce()' },
  { label: 'Math.define', type: 'method', detail: 'Math method: define()' },
  { label: 'Math.free_variables', type: 'method', detail: 'Math method: free_variables()' },
  { label: 'Math.expression', type: 'method', detail: 'Math method: expression()' },
  { label: 'Math.hyperparam_names', type: 'method', detail: 'Math method: hyperparam_names()' },
  { label: 'Math.init', type: 'method', detail: 'Math method: init()' },
  { label: 'Math.save', type: 'method', detail: 'Math method: save()' },
  { label: 'Math.to_latex', type: 'method', detail: 'Math method: to_latex()' },
  { label: 'Math.render', type: 'method', detail: 'Math method: render()' },
  { label: 'Math.hyperparam_values', type: 'method', detail: 'Math method: hyperparam_values()' },
  { label: 'Math.run_with', type: 'method', detail: 'Math method: run_with()' },
  { label: 'Math.run', type: 'method', detail: 'Math method: run()' },
  { label: 'Math.notation', type: 'method', detail: 'Math method: notation()' },
  { label: 'Math.evaluate', type: 'method', detail: 'Math method: evaluate()' },
  { label: 'Math.result', type: 'method', detail: 'Math method: result()' },
  { label: 'Math.var_bindings', type: 'method', detail: 'Math method: var_bindings()' },
  { label: 'Math.compile', type: 'method', detail: 'Math method: compile()' },
  { label: 'Math.to_torch', type: 'method', detail: 'Math method: to_torch()' },
  { label: 'Math.grad', type: 'method', detail: 'Math method: grad()' },
  { label: 'Math.device', type: 'method', detail: 'Math method: device()' },
  { label: 'Math.input_vars', type: 'method', detail: 'Math method: input_vars()' },
  { label: 'Math.to', type: 'method', detail: 'Math method: to()' },
  { label: 'Math.cuda', type: 'method', detail: 'Math method: cuda()' },
  { label: 'Math.cpu', type: 'method', detail: 'Math method: cpu()' },
  { label: 'Expr', type: 'class', detail: 'cm.symbols: Expr' },
  { label: 'Expr.to_sympy', type: 'method', detail: 'Expr method: to_sympy()' },
  { label: 'Expr.to_latex', type: 'method', detail: 'Expr method: to_latex()' },
  { label: 'Expr.render', type: 'method', detail: 'Expr method: render()' },
  { label: 'Expr.simplify', type: 'method', detail: 'Expr method: simplify()' },
  { label: 'Expr.expand', type: 'method', detail: 'Expr method: expand()' },
  { label: 'Expr.conjugate', type: 'method', detail: 'Expr method: conjugate()' },
  { label: 'Expr.integrate', type: 'method', detail: 'Expr method: integrate()' },
  { label: 'Expr.diff', type: 'method', detail: 'Expr method: diff()' },
  { label: 'Expr.sum', type: 'method', detail: 'Expr method: sum()' },
  { label: 'Expr.prod', type: 'method', detail: 'Expr method: prod()' },
  { label: 'Expr.evaluate', type: 'method', detail: 'Expr method: evaluate()' },
  { label: 'Expr.subs', type: 'method', detail: 'Expr method: subs()' },
  { label: 'Expr.is_definite', type: 'method', detail: 'Expr method: is_definite()' },
  { label: 'Expr.bound_to_latex', type: 'method', detail: 'Expr method: bound_to_latex()' },
  { label: 'Var', type: 'class', detail: 'cm.symbols: Var' },
  { label: 'Var.to_sympy', type: 'method', detail: 'Var method: to_sympy()' },
  { label: 'Var.to_latex', type: 'method', detail: 'Var method: to_latex()' },
  { label: 'Var.render', type: 'method', detail: 'Var method: render()' },
  { label: 'Var.simplify', type: 'method', detail: 'Var method: simplify()' },
  { label: 'Var.expand', type: 'method', detail: 'Var method: expand()' },
  { label: 'Var.conjugate', type: 'method', detail: 'Var method: conjugate()' },
  { label: 'Var.integrate', type: 'method', detail: 'Var method: integrate()' },
  { label: 'Var.diff', type: 'method', detail: 'Var method: diff()' },
  { label: 'Var.sum', type: 'method', detail: 'Var method: sum()' },
  { label: 'Var.prod', type: 'method', detail: 'Var method: prod()' },
  { label: 'Var.evaluate', type: 'method', detail: 'Var method: evaluate()' },
  { label: 'Var.subs', type: 'method', detail: 'Var method: subs()' },
  { label: 'Var.is_definite', type: 'method', detail: 'Var method: is_definite()' },
  { label: 'Var.bound_to_latex', type: 'method', detail: 'Var method: bound_to_latex()' },
  { label: 'Const', type: 'class', detail: 'cm.symbols: Const' },
  { label: 'Const.to_sympy', type: 'method', detail: 'Const method: to_sympy()' },
  { label: 'Const.to_latex', type: 'method', detail: 'Const method: to_latex()' },
  { label: 'Const.render', type: 'method', detail: 'Const method: render()' },
  { label: 'Const.simplify', type: 'method', detail: 'Const method: simplify()' },
  { label: 'Const.expand', type: 'method', detail: 'Const method: expand()' },
  { label: 'Const.conjugate', type: 'method', detail: 'Const method: conjugate()' },
  { label: 'Const.integrate', type: 'method', detail: 'Const method: integrate()' },
  { label: 'Const.diff', type: 'method', detail: 'Const method: diff()' },
  { label: 'Const.sum', type: 'method', detail: 'Const method: sum()' },
  { label: 'Const.prod', type: 'method', detail: 'Const method: prod()' },
  { label: 'Const.evaluate', type: 'method', detail: 'Const method: evaluate()' },
  { label: 'Const.subs', type: 'method', detail: 'Const method: subs()' },
  { label: 'Const.is_definite', type: 'method', detail: 'Const method: is_definite()' },
  { label: 'Const.bound_to_latex', type: 'method', detail: 'Const method: bound_to_latex()' },
  { label: 'Sum', type: 'class', detail: 'cm.symbols: Sum' },
  { label: 'Sum.to_sympy', type: 'method', detail: 'Sum method: to_sympy()' },
  { label: 'Sum.to_latex', type: 'method', detail: 'Sum method: to_latex()' },
  { label: 'Sum.render', type: 'method', detail: 'Sum method: render()' },
  { label: 'Sum.simplify', type: 'method', detail: 'Sum method: simplify()' },
  { label: 'Sum.expand', type: 'method', detail: 'Sum method: expand()' },
  { label: 'Sum.conjugate', type: 'method', detail: 'Sum method: conjugate()' },
  { label: 'Sum.integrate', type: 'method', detail: 'Sum method: integrate()' },
  { label: 'Sum.diff', type: 'method', detail: 'Sum method: diff()' },
  { label: 'Sum.sum', type: 'method', detail: 'Sum method: sum()' },
  { label: 'Sum.prod', type: 'method', detail: 'Sum method: prod()' },
  { label: 'Sum.evaluate', type: 'method', detail: 'Sum method: evaluate()' },
  { label: 'Sum.subs', type: 'method', detail: 'Sum method: subs()' },
  { label: 'Sum.is_definite', type: 'method', detail: 'Sum method: is_definite()' },
  { label: 'Sum.bound_to_latex', type: 'method', detail: 'Sum method: bound_to_latex()' },
  { label: 'Product', type: 'class', detail: 'cm.symbols: Product' },
  { label: 'Product.to_sympy', type: 'method', detail: 'Product method: to_sympy()' },
  { label: 'Product.to_latex', type: 'method', detail: 'Product method: to_latex()' },
  { label: 'Product.render', type: 'method', detail: 'Product method: render()' },
  { label: 'Product.simplify', type: 'method', detail: 'Product method: simplify()' },
  { label: 'Product.expand', type: 'method', detail: 'Product method: expand()' },
  { label: 'Product.conjugate', type: 'method', detail: 'Product method: conjugate()' },
  { label: 'Product.integrate', type: 'method', detail: 'Product method: integrate()' },
  { label: 'Product.diff', type: 'method', detail: 'Product method: diff()' },
  { label: 'Product.sum', type: 'method', detail: 'Product method: sum()' },
  { label: 'Product.prod', type: 'method', detail: 'Product method: prod()' },
  { label: 'Product.evaluate', type: 'method', detail: 'Product method: evaluate()' },
  { label: 'Product.subs', type: 'method', detail: 'Product method: subs()' },
  { label: 'Product.is_definite', type: 'method', detail: 'Product method: is_definite()' },
  { label: 'Product.bound_to_latex', type: 'method', detail: 'Product method: bound_to_latex()' },
  { label: 'Scalar', type: 'class', detail: 'cm.symbols: Scalar' },
  { label: 'Scalar.var', type: 'method', detail: 'Scalar method: var()' },
  { label: 'Scalar.const', type: 'method', detail: 'Scalar method: const()' },
  { label: 'Scalar.pi', type: 'method', detail: 'Scalar method: pi()' },
  { label: 'Scalar.e', type: 'method', detail: 'Scalar method: e()' },
  { label: 'Scalar.inf', type: 'method', detail: 'Scalar method: inf()' },
  { label: 'Scalar.sqrt', type: 'method', detail: 'Scalar method: sqrt()' },
  { label: 'Scalar.sin', type: 'method', detail: 'Scalar method: sin()' },
  { label: 'Scalar.cos', type: 'method', detail: 'Scalar method: cos()' },
  { label: 'Scalar.tan', type: 'method', detail: 'Scalar method: tan()' },
  { label: 'Scalar.exp', type: 'method', detail: 'Scalar method: exp()' },
  { label: 'Scalar.log', type: 'method', detail: 'Scalar method: log()' },
  { label: 'Scalar.abs', type: 'method', detail: 'Scalar method: abs()' },
  { label: 'Scalar.expr', type: 'method', detail: 'Scalar method: expr()' },
  { label: 'Scalar.sum', type: 'method', detail: 'Scalar method: sum()' },
  { label: 'Scalar.prod', type: 'method', detail: 'Scalar method: prod()' },
  { label: 'Scalar.function', type: 'method', detail: 'Scalar method: function()' },
  { label: 'Scalar.get_function', type: 'method', detail: 'Scalar method: get_function()' },
  { label: 'Scalar.list_functions', type: 'method', detail: 'Scalar method: list_functions()' },
  { label: 'Scalar.gamma', type: 'method', detail: 'Scalar method: gamma()' },
  { label: 'Scalar.loggamma', type: 'method', detail: 'Scalar method: loggamma()' },
  { label: 'Scalar.digamma', type: 'method', detail: 'Scalar method: digamma()' },
  { label: 'Scalar.beta', type: 'method', detail: 'Scalar method: beta()' },
  { label: 'Scalar.factorial', type: 'method', detail: 'Scalar method: factorial()' },
  { label: 'Scalar.factorial2', type: 'method', detail: 'Scalar method: factorial2()' },
  { label: 'Scalar.binomial', type: 'method', detail: 'Scalar method: binomial()' },
  { label: 'Scalar.erf', type: 'method', detail: 'Scalar method: erf()' },
  { label: 'Scalar.erfc', type: 'method', detail: 'Scalar method: erfc()' },
  { label: 'Scalar.erfi', type: 'method', detail: 'Scalar method: erfi()' },
  { label: 'Scalar.besselj', type: 'method', detail: 'Scalar method: besselj()' },
  { label: 'Scalar.bessely', type: 'method', detail: 'Scalar method: bessely()' },
  { label: 'Scalar.besseli', type: 'method', detail: 'Scalar method: besseli()' },
  { label: 'Scalar.besselk', type: 'method', detail: 'Scalar method: besselk()' },
  { label: 'Scalar.jn', type: 'method', detail: 'Scalar method: jn()' },
  { label: 'Scalar.yn', type: 'method', detail: 'Scalar method: yn()' },
  { label: 'Scalar.hankel1', type: 'method', detail: 'Scalar method: hankel1()' },
  { label: 'Scalar.hankel2', type: 'method', detail: 'Scalar method: hankel2()' },
  { label: 'Scalar.airyai', type: 'method', detail: 'Scalar method: airyai()' },
  { label: 'Scalar.airybi', type: 'method', detail: 'Scalar method: airybi()' },
  { label: 'Scalar.airyaiprime', type: 'method', detail: 'Scalar method: airyaiprime()' },
  { label: 'Scalar.airybiprime', type: 'method', detail: 'Scalar method: airybiprime()' },
  { label: 'Scalar.legendre', type: 'method', detail: 'Scalar method: legendre()' },
  { label: 'Scalar.assoc_legendre', type: 'method', detail: 'Scalar method: assoc_legendre()' },
  { label: 'Scalar.hermite', type: 'method', detail: 'Scalar method: hermite()' },
  { label: 'Scalar.hermite_prob', type: 'method', detail: 'Scalar method: hermite_prob()' },
  { label: 'Scalar.laguerre', type: 'method', detail: 'Scalar method: laguerre()' },
  { label: 'Scalar.assoc_laguerre', type: 'method', detail: 'Scalar method: assoc_laguerre()' },
  { label: 'Scalar.chebyshevt', type: 'method', detail: 'Scalar method: chebyshevt()' },
  { label: 'Scalar.chebyshevu', type: 'method', detail: 'Scalar method: chebyshevu()' },
  { label: 'Scalar.gegenbauer', type: 'method', detail: 'Scalar method: gegenbauer()' },
  { label: 'Scalar.jacobi', type: 'method', detail: 'Scalar method: jacobi()' },
  { label: 'Scalar.hyper2f1', type: 'method', detail: 'Scalar method: hyper2f1()' },
  { label: 'Scalar.hyper1f1', type: 'method', detail: 'Scalar method: hyper1f1()' },
  { label: 'Scalar.hyper0f1', type: 'method', detail: 'Scalar method: hyper0f1()' },
  { label: 'Scalar.hyperpfq', type: 'method', detail: 'Scalar method: hyperpfq()' },
  { label: 'Scalar.elliptic_k', type: 'method', detail: 'Scalar method: elliptic_k()' },
  { label: 'Scalar.elliptic_e', type: 'method', detail: 'Scalar method: elliptic_e()' },
  { label: 'Scalar.elliptic_pi', type: 'method', detail: 'Scalar method: elliptic_pi()' },
  { label: 'Scalar.zeta', type: 'method', detail: 'Scalar method: zeta()' },
  { label: 'Scalar.polylog', type: 'method', detail: 'Scalar method: polylog()' },
  { label: 'Scalar.dirac', type: 'method', detail: 'Scalar method: dirac()' },
  { label: 'Scalar.heaviside', type: 'method', detail: 'Scalar method: heaviside()' },
  { label: 'Scalar.kronecker', type: 'method', detail: 'Scalar method: kronecker()' },
  { label: 'Scalar.levi_civita', type: 'method', detail: 'Scalar method: levi_civita()' },
  { label: 'Scalar.validate', type: 'method', detail: 'Scalar method: validate()' },
  { label: 'Scalar.coerce', type: 'method', detail: 'Scalar method: coerce()' },
  { label: 'Scalar.define', type: 'method', detail: 'Scalar method: define()' },
  { label: 'Scalar.free_variables', type: 'method', detail: 'Scalar method: free_variables()' },
  { label: 'Scalar.expression', type: 'method', detail: 'Scalar method: expression()' },
  { label: 'Scalar.hyperparam_names', type: 'method', detail: 'Scalar method: hyperparam_names()' },
  { label: 'Scalar.init', type: 'method', detail: 'Scalar method: init()' },
  { label: 'Scalar.save', type: 'method', detail: 'Scalar method: save()' },
  { label: 'Scalar.to_latex', type: 'method', detail: 'Scalar method: to_latex()' },
  { label: 'Scalar.render', type: 'method', detail: 'Scalar method: render()' },
  { label: 'Scalar.hyperparam_values', type: 'method', detail: 'Scalar method: hyperparam_values()' },
  { label: 'Scalar.run_with', type: 'method', detail: 'Scalar method: run_with()' },
  { label: 'Scalar.run', type: 'method', detail: 'Scalar method: run()' },
  { label: 'Scalar.notation', type: 'method', detail: 'Scalar method: notation()' },
  { label: 'Scalar.evaluate', type: 'method', detail: 'Scalar method: evaluate()' },
  { label: 'Scalar.result', type: 'method', detail: 'Scalar method: result()' },
  { label: 'Scalar.var_bindings', type: 'method', detail: 'Scalar method: var_bindings()' },
  { label: 'Scalar.compile', type: 'method', detail: 'Scalar method: compile()' },
  { label: 'Scalar.to_torch', type: 'method', detail: 'Scalar method: to_torch()' },
  { label: 'Scalar.grad', type: 'method', detail: 'Scalar method: grad()' },
  { label: 'Scalar.device', type: 'method', detail: 'Scalar method: device()' },
  { label: 'Scalar.input_vars', type: 'method', detail: 'Scalar method: input_vars()' },
  { label: 'Scalar.to', type: 'method', detail: 'Scalar method: to()' },
  { label: 'Scalar.cuda', type: 'method', detail: 'Scalar method: cuda()' },
  { label: 'Scalar.cpu', type: 'method', detail: 'Scalar method: cpu()' },
  { label: 'ExprType', type: 'class', detail: 'cm.symbols: ExprType' },
  { label: 'ExprType.var', type: 'method', detail: 'ExprType method: var()' },
  { label: 'ExprType.const', type: 'method', detail: 'ExprType method: const()' },
  { label: 'ExprType.pi', type: 'method', detail: 'ExprType method: pi()' },
  { label: 'ExprType.e', type: 'method', detail: 'ExprType method: e()' },
  { label: 'ExprType.inf', type: 'method', detail: 'ExprType method: inf()' },
  { label: 'ExprType.sqrt', type: 'method', detail: 'ExprType method: sqrt()' },
  { label: 'ExprType.sin', type: 'method', detail: 'ExprType method: sin()' },
  { label: 'ExprType.cos', type: 'method', detail: 'ExprType method: cos()' },
  { label: 'ExprType.tan', type: 'method', detail: 'ExprType method: tan()' },
  { label: 'ExprType.exp', type: 'method', detail: 'ExprType method: exp()' },
  { label: 'ExprType.log', type: 'method', detail: 'ExprType method: log()' },
  { label: 'ExprType.abs', type: 'method', detail: 'ExprType method: abs()' },
  { label: 'ExprType.expr', type: 'method', detail: 'ExprType method: expr()' },
  { label: 'ExprType.sum', type: 'method', detail: 'ExprType method: sum()' },
  { label: 'ExprType.prod', type: 'method', detail: 'ExprType method: prod()' },
  { label: 'ExprType.function', type: 'method', detail: 'ExprType method: function()' },
  { label: 'ExprType.get_function', type: 'method', detail: 'ExprType method: get_function()' },
  { label: 'ExprType.list_functions', type: 'method', detail: 'ExprType method: list_functions()' },
  { label: 'ExprType.gamma', type: 'method', detail: 'ExprType method: gamma()' },
  { label: 'ExprType.loggamma', type: 'method', detail: 'ExprType method: loggamma()' },
  { label: 'ExprType.digamma', type: 'method', detail: 'ExprType method: digamma()' },
  { label: 'ExprType.beta', type: 'method', detail: 'ExprType method: beta()' },
  { label: 'ExprType.factorial', type: 'method', detail: 'ExprType method: factorial()' },
  { label: 'ExprType.factorial2', type: 'method', detail: 'ExprType method: factorial2()' },
  { label: 'ExprType.binomial', type: 'method', detail: 'ExprType method: binomial()' },
  { label: 'ExprType.erf', type: 'method', detail: 'ExprType method: erf()' },
  { label: 'ExprType.erfc', type: 'method', detail: 'ExprType method: erfc()' },
  { label: 'ExprType.erfi', type: 'method', detail: 'ExprType method: erfi()' },
  { label: 'ExprType.besselj', type: 'method', detail: 'ExprType method: besselj()' },
  { label: 'ExprType.bessely', type: 'method', detail: 'ExprType method: bessely()' },
  { label: 'ExprType.besseli', type: 'method', detail: 'ExprType method: besseli()' },
  { label: 'ExprType.besselk', type: 'method', detail: 'ExprType method: besselk()' },
  { label: 'ExprType.jn', type: 'method', detail: 'ExprType method: jn()' },
  { label: 'ExprType.yn', type: 'method', detail: 'ExprType method: yn()' },
  { label: 'ExprType.hankel1', type: 'method', detail: 'ExprType method: hankel1()' },
  { label: 'ExprType.hankel2', type: 'method', detail: 'ExprType method: hankel2()' },
  { label: 'ExprType.airyai', type: 'method', detail: 'ExprType method: airyai()' },
  { label: 'ExprType.airybi', type: 'method', detail: 'ExprType method: airybi()' },
  { label: 'ExprType.airyaiprime', type: 'method', detail: 'ExprType method: airyaiprime()' },
  { label: 'ExprType.airybiprime', type: 'method', detail: 'ExprType method: airybiprime()' },
  { label: 'ExprType.legendre', type: 'method', detail: 'ExprType method: legendre()' },
  { label: 'ExprType.assoc_legendre', type: 'method', detail: 'ExprType method: assoc_legendre()' },
  { label: 'ExprType.hermite', type: 'method', detail: 'ExprType method: hermite()' },
  { label: 'ExprType.hermite_prob', type: 'method', detail: 'ExprType method: hermite_prob()' },
  { label: 'ExprType.laguerre', type: 'method', detail: 'ExprType method: laguerre()' },
  { label: 'ExprType.assoc_laguerre', type: 'method', detail: 'ExprType method: assoc_laguerre()' },
  { label: 'ExprType.chebyshevt', type: 'method', detail: 'ExprType method: chebyshevt()' },
  { label: 'ExprType.chebyshevu', type: 'method', detail: 'ExprType method: chebyshevu()' },
  { label: 'ExprType.gegenbauer', type: 'method', detail: 'ExprType method: gegenbauer()' },
  { label: 'ExprType.jacobi', type: 'method', detail: 'ExprType method: jacobi()' },
  { label: 'ExprType.hyper2f1', type: 'method', detail: 'ExprType method: hyper2f1()' },
  { label: 'ExprType.hyper1f1', type: 'method', detail: 'ExprType method: hyper1f1()' },
  { label: 'ExprType.hyper0f1', type: 'method', detail: 'ExprType method: hyper0f1()' },
  { label: 'ExprType.hyperpfq', type: 'method', detail: 'ExprType method: hyperpfq()' },
  { label: 'ExprType.elliptic_k', type: 'method', detail: 'ExprType method: elliptic_k()' },
  { label: 'ExprType.elliptic_e', type: 'method', detail: 'ExprType method: elliptic_e()' },
  { label: 'ExprType.elliptic_pi', type: 'method', detail: 'ExprType method: elliptic_pi()' },
  { label: 'ExprType.zeta', type: 'method', detail: 'ExprType method: zeta()' },
  { label: 'ExprType.polylog', type: 'method', detail: 'ExprType method: polylog()' },
  { label: 'ExprType.dirac', type: 'method', detail: 'ExprType method: dirac()' },
  { label: 'ExprType.heaviside', type: 'method', detail: 'ExprType method: heaviside()' },
  { label: 'ExprType.kronecker', type: 'method', detail: 'ExprType method: kronecker()' },
  { label: 'ExprType.levi_civita', type: 'method', detail: 'ExprType method: levi_civita()' },
  { label: 'ExprType.validate', type: 'method', detail: 'ExprType method: validate()' },
  { label: 'ExprType.coerce', type: 'method', detail: 'ExprType method: coerce()' },
  { label: 'ExprType.define', type: 'method', detail: 'ExprType method: define()' },
  { label: 'ExprType.free_variables', type: 'method', detail: 'ExprType method: free_variables()' },
  { label: 'ExprType.expression', type: 'method', detail: 'ExprType method: expression()' },
  { label: 'ExprType.hyperparam_names', type: 'method', detail: 'ExprType method: hyperparam_names()' },
  { label: 'ExprType.init', type: 'method', detail: 'ExprType method: init()' },
  { label: 'ExprType.save', type: 'method', detail: 'ExprType method: save()' },
  { label: 'ExprType.to_latex', type: 'method', detail: 'ExprType method: to_latex()' },
  { label: 'ExprType.render', type: 'method', detail: 'ExprType method: render()' },
  { label: 'ExprType.hyperparam_values', type: 'method', detail: 'ExprType method: hyperparam_values()' },
  { label: 'ExprType.run_with', type: 'method', detail: 'ExprType method: run_with()' },
  { label: 'ExprType.run', type: 'method', detail: 'ExprType method: run()' },
  { label: 'ExprType.notation', type: 'method', detail: 'ExprType method: notation()' },
  { label: 'ExprType.evaluate', type: 'method', detail: 'ExprType method: evaluate()' },
  { label: 'ExprType.result', type: 'method', detail: 'ExprType method: result()' },
  { label: 'ExprType.var_bindings', type: 'method', detail: 'ExprType method: var_bindings()' },
  { label: 'ExprType.compile', type: 'method', detail: 'ExprType method: compile()' },
  { label: 'ExprType.to_torch', type: 'method', detail: 'ExprType method: to_torch()' },
  { label: 'ExprType.grad', type: 'method', detail: 'ExprType method: grad()' },
  { label: 'ExprType.device', type: 'method', detail: 'ExprType method: device()' },
  { label: 'ExprType.input_vars', type: 'method', detail: 'ExprType method: input_vars()' },
  { label: 'ExprType.to', type: 'method', detail: 'ExprType method: to()' },
  { label: 'ExprType.cuda', type: 'method', detail: 'ExprType method: cuda()' },
  { label: 'ExprType.cpu', type: 'method', detail: 'ExprType method: cpu()' },
  { label: 'BoundsType', type: 'class', detail: 'cm.symbols: BoundsType' },
  { label: 'BoundsType.var', type: 'method', detail: 'BoundsType method: var()' },
  { label: 'BoundsType.const', type: 'method', detail: 'BoundsType method: const()' },
  { label: 'BoundsType.pi', type: 'method', detail: 'BoundsType method: pi()' },
  { label: 'BoundsType.e', type: 'method', detail: 'BoundsType method: e()' },
  { label: 'BoundsType.inf', type: 'method', detail: 'BoundsType method: inf()' },
  { label: 'BoundsType.sqrt', type: 'method', detail: 'BoundsType method: sqrt()' },
  { label: 'BoundsType.sin', type: 'method', detail: 'BoundsType method: sin()' },
  { label: 'BoundsType.cos', type: 'method', detail: 'BoundsType method: cos()' },
  { label: 'BoundsType.tan', type: 'method', detail: 'BoundsType method: tan()' },
  { label: 'BoundsType.exp', type: 'method', detail: 'BoundsType method: exp()' },
  { label: 'BoundsType.log', type: 'method', detail: 'BoundsType method: log()' },
  { label: 'BoundsType.abs', type: 'method', detail: 'BoundsType method: abs()' },
  { label: 'BoundsType.expr', type: 'method', detail: 'BoundsType method: expr()' },
  { label: 'BoundsType.sum', type: 'method', detail: 'BoundsType method: sum()' },
  { label: 'BoundsType.prod', type: 'method', detail: 'BoundsType method: prod()' },
  { label: 'BoundsType.function', type: 'method', detail: 'BoundsType method: function()' },
  { label: 'BoundsType.get_function', type: 'method', detail: 'BoundsType method: get_function()' },
  { label: 'BoundsType.list_functions', type: 'method', detail: 'BoundsType method: list_functions()' },
  { label: 'BoundsType.gamma', type: 'method', detail: 'BoundsType method: gamma()' },
  { label: 'BoundsType.loggamma', type: 'method', detail: 'BoundsType method: loggamma()' },
  { label: 'BoundsType.digamma', type: 'method', detail: 'BoundsType method: digamma()' },
  { label: 'BoundsType.beta', type: 'method', detail: 'BoundsType method: beta()' },
  { label: 'BoundsType.factorial', type: 'method', detail: 'BoundsType method: factorial()' },
  { label: 'BoundsType.factorial2', type: 'method', detail: 'BoundsType method: factorial2()' },
  { label: 'BoundsType.binomial', type: 'method', detail: 'BoundsType method: binomial()' },
  { label: 'BoundsType.erf', type: 'method', detail: 'BoundsType method: erf()' },
  { label: 'BoundsType.erfc', type: 'method', detail: 'BoundsType method: erfc()' },
  { label: 'BoundsType.erfi', type: 'method', detail: 'BoundsType method: erfi()' },
  { label: 'BoundsType.besselj', type: 'method', detail: 'BoundsType method: besselj()' },
  { label: 'BoundsType.bessely', type: 'method', detail: 'BoundsType method: bessely()' },
  { label: 'BoundsType.besseli', type: 'method', detail: 'BoundsType method: besseli()' },
  { label: 'BoundsType.besselk', type: 'method', detail: 'BoundsType method: besselk()' },
  { label: 'BoundsType.jn', type: 'method', detail: 'BoundsType method: jn()' },
  { label: 'BoundsType.yn', type: 'method', detail: 'BoundsType method: yn()' },
  { label: 'BoundsType.hankel1', type: 'method', detail: 'BoundsType method: hankel1()' },
  { label: 'BoundsType.hankel2', type: 'method', detail: 'BoundsType method: hankel2()' },
  { label: 'BoundsType.airyai', type: 'method', detail: 'BoundsType method: airyai()' },
  { label: 'BoundsType.airybi', type: 'method', detail: 'BoundsType method: airybi()' },
  { label: 'BoundsType.airyaiprime', type: 'method', detail: 'BoundsType method: airyaiprime()' },
  { label: 'BoundsType.airybiprime', type: 'method', detail: 'BoundsType method: airybiprime()' },
  { label: 'BoundsType.legendre', type: 'method', detail: 'BoundsType method: legendre()' },
  { label: 'BoundsType.assoc_legendre', type: 'method', detail: 'BoundsType method: assoc_legendre()' },
  { label: 'BoundsType.hermite', type: 'method', detail: 'BoundsType method: hermite()' },
  { label: 'BoundsType.hermite_prob', type: 'method', detail: 'BoundsType method: hermite_prob()' },
  { label: 'BoundsType.laguerre', type: 'method', detail: 'BoundsType method: laguerre()' },
  { label: 'BoundsType.assoc_laguerre', type: 'method', detail: 'BoundsType method: assoc_laguerre()' },
  { label: 'BoundsType.chebyshevt', type: 'method', detail: 'BoundsType method: chebyshevt()' },
  { label: 'BoundsType.chebyshevu', type: 'method', detail: 'BoundsType method: chebyshevu()' },
  { label: 'BoundsType.gegenbauer', type: 'method', detail: 'BoundsType method: gegenbauer()' },
  { label: 'BoundsType.jacobi', type: 'method', detail: 'BoundsType method: jacobi()' },
  { label: 'BoundsType.hyper2f1', type: 'method', detail: 'BoundsType method: hyper2f1()' },
  { label: 'BoundsType.hyper1f1', type: 'method', detail: 'BoundsType method: hyper1f1()' },
  { label: 'BoundsType.hyper0f1', type: 'method', detail: 'BoundsType method: hyper0f1()' },
  { label: 'BoundsType.hyperpfq', type: 'method', detail: 'BoundsType method: hyperpfq()' },
  { label: 'BoundsType.elliptic_k', type: 'method', detail: 'BoundsType method: elliptic_k()' },
  { label: 'BoundsType.elliptic_e', type: 'method', detail: 'BoundsType method: elliptic_e()' },
  { label: 'BoundsType.elliptic_pi', type: 'method', detail: 'BoundsType method: elliptic_pi()' },
  { label: 'BoundsType.zeta', type: 'method', detail: 'BoundsType method: zeta()' },
  { label: 'BoundsType.polylog', type: 'method', detail: 'BoundsType method: polylog()' },
  { label: 'BoundsType.dirac', type: 'method', detail: 'BoundsType method: dirac()' },
  { label: 'BoundsType.heaviside', type: 'method', detail: 'BoundsType method: heaviside()' },
  { label: 'BoundsType.kronecker', type: 'method', detail: 'BoundsType method: kronecker()' },
  { label: 'BoundsType.levi_civita', type: 'method', detail: 'BoundsType method: levi_civita()' },
  { label: 'BoundsType.validate', type: 'method', detail: 'BoundsType method: validate()' },
  { label: 'BoundsType.coerce', type: 'method', detail: 'BoundsType method: coerce()' },
  { label: 'BoundsType.define', type: 'method', detail: 'BoundsType method: define()' },
  { label: 'BoundsType.free_variables', type: 'method', detail: 'BoundsType method: free_variables()' },
  { label: 'BoundsType.expression', type: 'method', detail: 'BoundsType method: expression()' },
  { label: 'BoundsType.hyperparam_names', type: 'method', detail: 'BoundsType method: hyperparam_names()' },
  { label: 'BoundsType.init', type: 'method', detail: 'BoundsType method: init()' },
  { label: 'BoundsType.save', type: 'method', detail: 'BoundsType method: save()' },
  { label: 'BoundsType.to_latex', type: 'method', detail: 'BoundsType method: to_latex()' },
  { label: 'BoundsType.render', type: 'method', detail: 'BoundsType method: render()' },
  { label: 'BoundsType.hyperparam_values', type: 'method', detail: 'BoundsType method: hyperparam_values()' },
  { label: 'BoundsType.run_with', type: 'method', detail: 'BoundsType method: run_with()' },
  { label: 'BoundsType.run', type: 'method', detail: 'BoundsType method: run()' },
  { label: 'BoundsType.notation', type: 'method', detail: 'BoundsType method: notation()' },
  { label: 'BoundsType.evaluate', type: 'method', detail: 'BoundsType method: evaluate()' },
  { label: 'BoundsType.result', type: 'method', detail: 'BoundsType method: result()' },
  { label: 'BoundsType.var_bindings', type: 'method', detail: 'BoundsType method: var_bindings()' },
  { label: 'BoundsType.compile', type: 'method', detail: 'BoundsType method: compile()' },
  { label: 'BoundsType.to_torch', type: 'method', detail: 'BoundsType method: to_torch()' },
  { label: 'BoundsType.grad', type: 'method', detail: 'BoundsType method: grad()' },
  { label: 'BoundsType.device', type: 'method', detail: 'BoundsType method: device()' },
  { label: 'BoundsType.input_vars', type: 'method', detail: 'BoundsType method: input_vars()' },
  { label: 'BoundsType.to', type: 'method', detail: 'BoundsType method: to()' },
  { label: 'BoundsType.cuda', type: 'method', detail: 'BoundsType method: cuda()' },
  { label: 'BoundsType.cpu', type: 'method', detail: 'BoundsType method: cpu()' },
  { label: 'SymbolicFunction', type: 'class', detail: 'cm.symbols: SymbolicFunction' },
  { label: 'SymbolicFunction.var', type: 'method', detail: 'SymbolicFunction method: var()' },
  { label: 'SymbolicFunction.const', type: 'method', detail: 'SymbolicFunction method: const()' },
  { label: 'SymbolicFunction.pi', type: 'method', detail: 'SymbolicFunction method: pi()' },
  { label: 'SymbolicFunction.e', type: 'method', detail: 'SymbolicFunction method: e()' },
  { label: 'SymbolicFunction.inf', type: 'method', detail: 'SymbolicFunction method: inf()' },
  { label: 'SymbolicFunction.sqrt', type: 'method', detail: 'SymbolicFunction method: sqrt()' },
  { label: 'SymbolicFunction.sin', type: 'method', detail: 'SymbolicFunction method: sin()' },
  { label: 'SymbolicFunction.cos', type: 'method', detail: 'SymbolicFunction method: cos()' },
  { label: 'SymbolicFunction.tan', type: 'method', detail: 'SymbolicFunction method: tan()' },
  { label: 'SymbolicFunction.exp', type: 'method', detail: 'SymbolicFunction method: exp()' },
  { label: 'SymbolicFunction.log', type: 'method', detail: 'SymbolicFunction method: log()' },
  { label: 'SymbolicFunction.abs', type: 'method', detail: 'SymbolicFunction method: abs()' },
  { label: 'SymbolicFunction.expr', type: 'method', detail: 'SymbolicFunction method: expr()' },
  { label: 'SymbolicFunction.sum', type: 'method', detail: 'SymbolicFunction method: sum()' },
  { label: 'SymbolicFunction.prod', type: 'method', detail: 'SymbolicFunction method: prod()' },
  { label: 'SymbolicFunction.function', type: 'method', detail: 'SymbolicFunction method: function()' },
  { label: 'SymbolicFunction.get_function', type: 'method', detail: 'SymbolicFunction method: get_function()' },
  { label: 'SymbolicFunction.list_functions', type: 'method', detail: 'SymbolicFunction method: list_functions()' },
  { label: 'SymbolicFunction.gamma', type: 'method', detail: 'SymbolicFunction method: gamma()' },
  { label: 'SymbolicFunction.loggamma', type: 'method', detail: 'SymbolicFunction method: loggamma()' },
  { label: 'SymbolicFunction.digamma', type: 'method', detail: 'SymbolicFunction method: digamma()' },
  { label: 'SymbolicFunction.beta', type: 'method', detail: 'SymbolicFunction method: beta()' },
  { label: 'SymbolicFunction.factorial', type: 'method', detail: 'SymbolicFunction method: factorial()' },
  { label: 'SymbolicFunction.factorial2', type: 'method', detail: 'SymbolicFunction method: factorial2()' },
  { label: 'SymbolicFunction.binomial', type: 'method', detail: 'SymbolicFunction method: binomial()' },
  { label: 'SymbolicFunction.erf', type: 'method', detail: 'SymbolicFunction method: erf()' },
  { label: 'SymbolicFunction.erfc', type: 'method', detail: 'SymbolicFunction method: erfc()' },
  { label: 'SymbolicFunction.erfi', type: 'method', detail: 'SymbolicFunction method: erfi()' },
  { label: 'SymbolicFunction.besselj', type: 'method', detail: 'SymbolicFunction method: besselj()' },
  { label: 'SymbolicFunction.bessely', type: 'method', detail: 'SymbolicFunction method: bessely()' },
  { label: 'SymbolicFunction.besseli', type: 'method', detail: 'SymbolicFunction method: besseli()' },
  { label: 'SymbolicFunction.besselk', type: 'method', detail: 'SymbolicFunction method: besselk()' },
  { label: 'SymbolicFunction.jn', type: 'method', detail: 'SymbolicFunction method: jn()' },
  { label: 'SymbolicFunction.yn', type: 'method', detail: 'SymbolicFunction method: yn()' },
  { label: 'SymbolicFunction.hankel1', type: 'method', detail: 'SymbolicFunction method: hankel1()' },
  { label: 'SymbolicFunction.hankel2', type: 'method', detail: 'SymbolicFunction method: hankel2()' },
  { label: 'SymbolicFunction.airyai', type: 'method', detail: 'SymbolicFunction method: airyai()' },
  { label: 'SymbolicFunction.airybi', type: 'method', detail: 'SymbolicFunction method: airybi()' },
  { label: 'SymbolicFunction.airyaiprime', type: 'method', detail: 'SymbolicFunction method: airyaiprime()' },
  { label: 'SymbolicFunction.airybiprime', type: 'method', detail: 'SymbolicFunction method: airybiprime()' },
  { label: 'SymbolicFunction.legendre', type: 'method', detail: 'SymbolicFunction method: legendre()' },
  { label: 'SymbolicFunction.assoc_legendre', type: 'method', detail: 'SymbolicFunction method: assoc_legendre()' },
  { label: 'SymbolicFunction.hermite', type: 'method', detail: 'SymbolicFunction method: hermite()' },
  { label: 'SymbolicFunction.hermite_prob', type: 'method', detail: 'SymbolicFunction method: hermite_prob()' },
  { label: 'SymbolicFunction.laguerre', type: 'method', detail: 'SymbolicFunction method: laguerre()' },
  { label: 'SymbolicFunction.assoc_laguerre', type: 'method', detail: 'SymbolicFunction method: assoc_laguerre()' },
  { label: 'SymbolicFunction.chebyshevt', type: 'method', detail: 'SymbolicFunction method: chebyshevt()' },
  { label: 'SymbolicFunction.chebyshevu', type: 'method', detail: 'SymbolicFunction method: chebyshevu()' },
  { label: 'SymbolicFunction.gegenbauer', type: 'method', detail: 'SymbolicFunction method: gegenbauer()' },
  { label: 'SymbolicFunction.jacobi', type: 'method', detail: 'SymbolicFunction method: jacobi()' },
  { label: 'SymbolicFunction.hyper2f1', type: 'method', detail: 'SymbolicFunction method: hyper2f1()' },
  { label: 'SymbolicFunction.hyper1f1', type: 'method', detail: 'SymbolicFunction method: hyper1f1()' },
  { label: 'SymbolicFunction.hyper0f1', type: 'method', detail: 'SymbolicFunction method: hyper0f1()' },
  { label: 'SymbolicFunction.hyperpfq', type: 'method', detail: 'SymbolicFunction method: hyperpfq()' },
  { label: 'SymbolicFunction.elliptic_k', type: 'method', detail: 'SymbolicFunction method: elliptic_k()' },
  { label: 'SymbolicFunction.elliptic_e', type: 'method', detail: 'SymbolicFunction method: elliptic_e()' },
  { label: 'SymbolicFunction.elliptic_pi', type: 'method', detail: 'SymbolicFunction method: elliptic_pi()' },
  { label: 'SymbolicFunction.zeta', type: 'method', detail: 'SymbolicFunction method: zeta()' },
  { label: 'SymbolicFunction.polylog', type: 'method', detail: 'SymbolicFunction method: polylog()' },
  { label: 'SymbolicFunction.dirac', type: 'method', detail: 'SymbolicFunction method: dirac()' },
  { label: 'SymbolicFunction.heaviside', type: 'method', detail: 'SymbolicFunction method: heaviside()' },
  { label: 'SymbolicFunction.kronecker', type: 'method', detail: 'SymbolicFunction method: kronecker()' },
  { label: 'SymbolicFunction.levi_civita', type: 'method', detail: 'SymbolicFunction method: levi_civita()' },
  { label: 'SymbolicFunction.validate', type: 'method', detail: 'SymbolicFunction method: validate()' },
  { label: 'SymbolicFunction.coerce', type: 'method', detail: 'SymbolicFunction method: coerce()' },
  { label: 'SymbolicFunction.define', type: 'method', detail: 'SymbolicFunction method: define()' },
  { label: 'SymbolicFunction.free_variables', type: 'method', detail: 'SymbolicFunction method: free_variables()' },
  { label: 'SymbolicFunction.expression', type: 'method', detail: 'SymbolicFunction method: expression()' },
  { label: 'SymbolicFunction.hyperparam_names', type: 'method', detail: 'SymbolicFunction method: hyperparam_names()' },
  { label: 'SymbolicFunction.init', type: 'method', detail: 'SymbolicFunction method: init()' },
  { label: 'SymbolicFunction.save', type: 'method', detail: 'SymbolicFunction method: save()' },
  { label: 'SymbolicFunction.to_latex', type: 'method', detail: 'SymbolicFunction method: to_latex()' },
  { label: 'SymbolicFunction.render', type: 'method', detail: 'SymbolicFunction method: render()' },
  { label: 'SymbolicFunction.hyperparam_values', type: 'method', detail: 'SymbolicFunction method: hyperparam_values()' },
  { label: 'SymbolicFunction.run_with', type: 'method', detail: 'SymbolicFunction method: run_with()' },
  { label: 'SymbolicFunction.run', type: 'method', detail: 'SymbolicFunction method: run()' },
  { label: 'SymbolicFunction.notation', type: 'method', detail: 'SymbolicFunction method: notation()' },
  { label: 'SymbolicFunction.evaluate', type: 'method', detail: 'SymbolicFunction method: evaluate()' },
  { label: 'SymbolicFunction.result', type: 'method', detail: 'SymbolicFunction method: result()' },
  { label: 'SymbolicFunction.var_bindings', type: 'method', detail: 'SymbolicFunction method: var_bindings()' },
  { label: 'SymbolicFunction.compile', type: 'method', detail: 'SymbolicFunction method: compile()' },
  { label: 'SymbolicFunction.to_torch', type: 'method', detail: 'SymbolicFunction method: to_torch()' },
  { label: 'SymbolicFunction.grad', type: 'method', detail: 'SymbolicFunction method: grad()' },
  { label: 'SymbolicFunction.device', type: 'method', detail: 'SymbolicFunction method: device()' },
  { label: 'SymbolicFunction.input_vars', type: 'method', detail: 'SymbolicFunction method: input_vars()' },
  { label: 'SymbolicFunction.to', type: 'method', detail: 'SymbolicFunction method: to()' },
  { label: 'SymbolicFunction.cuda', type: 'method', detail: 'SymbolicFunction method: cuda()' },
  { label: 'SymbolicFunction.cpu', type: 'method', detail: 'SymbolicFunction method: cpu()' },
  { label: 'BoundFunction', type: 'class', detail: 'cm.symbols: BoundFunction' },
  { label: 'BoundFunction.var', type: 'method', detail: 'BoundFunction method: var()' },
  { label: 'BoundFunction.const', type: 'method', detail: 'BoundFunction method: const()' },
  { label: 'BoundFunction.pi', type: 'method', detail: 'BoundFunction method: pi()' },
  { label: 'BoundFunction.e', type: 'method', detail: 'BoundFunction method: e()' },
  { label: 'BoundFunction.inf', type: 'method', detail: 'BoundFunction method: inf()' },
  { label: 'BoundFunction.sqrt', type: 'method', detail: 'BoundFunction method: sqrt()' },
  { label: 'BoundFunction.sin', type: 'method', detail: 'BoundFunction method: sin()' },
  { label: 'BoundFunction.cos', type: 'method', detail: 'BoundFunction method: cos()' },
  { label: 'BoundFunction.tan', type: 'method', detail: 'BoundFunction method: tan()' },
  { label: 'BoundFunction.exp', type: 'method', detail: 'BoundFunction method: exp()' },
  { label: 'BoundFunction.log', type: 'method', detail: 'BoundFunction method: log()' },
  { label: 'BoundFunction.abs', type: 'method', detail: 'BoundFunction method: abs()' },
  { label: 'BoundFunction.expr', type: 'method', detail: 'BoundFunction method: expr()' },
  { label: 'BoundFunction.sum', type: 'method', detail: 'BoundFunction method: sum()' },
  { label: 'BoundFunction.prod', type: 'method', detail: 'BoundFunction method: prod()' },
  { label: 'BoundFunction.function', type: 'method', detail: 'BoundFunction method: function()' },
  { label: 'BoundFunction.get_function', type: 'method', detail: 'BoundFunction method: get_function()' },
  { label: 'BoundFunction.list_functions', type: 'method', detail: 'BoundFunction method: list_functions()' },
  { label: 'BoundFunction.gamma', type: 'method', detail: 'BoundFunction method: gamma()' },
  { label: 'BoundFunction.loggamma', type: 'method', detail: 'BoundFunction method: loggamma()' },
  { label: 'BoundFunction.digamma', type: 'method', detail: 'BoundFunction method: digamma()' },
  { label: 'BoundFunction.beta', type: 'method', detail: 'BoundFunction method: beta()' },
  { label: 'BoundFunction.factorial', type: 'method', detail: 'BoundFunction method: factorial()' },
  { label: 'BoundFunction.factorial2', type: 'method', detail: 'BoundFunction method: factorial2()' },
  { label: 'BoundFunction.binomial', type: 'method', detail: 'BoundFunction method: binomial()' },
  { label: 'BoundFunction.erf', type: 'method', detail: 'BoundFunction method: erf()' },
  { label: 'BoundFunction.erfc', type: 'method', detail: 'BoundFunction method: erfc()' },
  { label: 'BoundFunction.erfi', type: 'method', detail: 'BoundFunction method: erfi()' },
  { label: 'BoundFunction.besselj', type: 'method', detail: 'BoundFunction method: besselj()' },
  { label: 'BoundFunction.bessely', type: 'method', detail: 'BoundFunction method: bessely()' },
  { label: 'BoundFunction.besseli', type: 'method', detail: 'BoundFunction method: besseli()' },
  { label: 'BoundFunction.besselk', type: 'method', detail: 'BoundFunction method: besselk()' },
  { label: 'BoundFunction.jn', type: 'method', detail: 'BoundFunction method: jn()' },
  { label: 'BoundFunction.yn', type: 'method', detail: 'BoundFunction method: yn()' },
  { label: 'BoundFunction.hankel1', type: 'method', detail: 'BoundFunction method: hankel1()' },
  { label: 'BoundFunction.hankel2', type: 'method', detail: 'BoundFunction method: hankel2()' },
  { label: 'BoundFunction.airyai', type: 'method', detail: 'BoundFunction method: airyai()' },
  { label: 'BoundFunction.airybi', type: 'method', detail: 'BoundFunction method: airybi()' },
  { label: 'BoundFunction.airyaiprime', type: 'method', detail: 'BoundFunction method: airyaiprime()' },
  { label: 'BoundFunction.airybiprime', type: 'method', detail: 'BoundFunction method: airybiprime()' },
  { label: 'BoundFunction.legendre', type: 'method', detail: 'BoundFunction method: legendre()' },
  { label: 'BoundFunction.assoc_legendre', type: 'method', detail: 'BoundFunction method: assoc_legendre()' },
  { label: 'BoundFunction.hermite', type: 'method', detail: 'BoundFunction method: hermite()' },
  { label: 'BoundFunction.hermite_prob', type: 'method', detail: 'BoundFunction method: hermite_prob()' },
  { label: 'BoundFunction.laguerre', type: 'method', detail: 'BoundFunction method: laguerre()' },
  { label: 'BoundFunction.assoc_laguerre', type: 'method', detail: 'BoundFunction method: assoc_laguerre()' },
  { label: 'BoundFunction.chebyshevt', type: 'method', detail: 'BoundFunction method: chebyshevt()' },
  { label: 'BoundFunction.chebyshevu', type: 'method', detail: 'BoundFunction method: chebyshevu()' },
  { label: 'BoundFunction.gegenbauer', type: 'method', detail: 'BoundFunction method: gegenbauer()' },
  { label: 'BoundFunction.jacobi', type: 'method', detail: 'BoundFunction method: jacobi()' },
  { label: 'BoundFunction.hyper2f1', type: 'method', detail: 'BoundFunction method: hyper2f1()' },
  { label: 'BoundFunction.hyper1f1', type: 'method', detail: 'BoundFunction method: hyper1f1()' },
  { label: 'BoundFunction.hyper0f1', type: 'method', detail: 'BoundFunction method: hyper0f1()' },
  { label: 'BoundFunction.hyperpfq', type: 'method', detail: 'BoundFunction method: hyperpfq()' },
  { label: 'BoundFunction.elliptic_k', type: 'method', detail: 'BoundFunction method: elliptic_k()' },
  { label: 'BoundFunction.elliptic_e', type: 'method', detail: 'BoundFunction method: elliptic_e()' },
  { label: 'BoundFunction.elliptic_pi', type: 'method', detail: 'BoundFunction method: elliptic_pi()' },
  { label: 'BoundFunction.zeta', type: 'method', detail: 'BoundFunction method: zeta()' },
  { label: 'BoundFunction.polylog', type: 'method', detail: 'BoundFunction method: polylog()' },
  { label: 'BoundFunction.dirac', type: 'method', detail: 'BoundFunction method: dirac()' },
  { label: 'BoundFunction.heaviside', type: 'method', detail: 'BoundFunction method: heaviside()' },
  { label: 'BoundFunction.kronecker', type: 'method', detail: 'BoundFunction method: kronecker()' },
  { label: 'BoundFunction.levi_civita', type: 'method', detail: 'BoundFunction method: levi_civita()' },
  { label: 'BoundFunction.validate', type: 'method', detail: 'BoundFunction method: validate()' },
  { label: 'BoundFunction.coerce', type: 'method', detail: 'BoundFunction method: coerce()' },
  { label: 'BoundFunction.define', type: 'method', detail: 'BoundFunction method: define()' },
  { label: 'BoundFunction.free_variables', type: 'method', detail: 'BoundFunction method: free_variables()' },
  { label: 'BoundFunction.expression', type: 'method', detail: 'BoundFunction method: expression()' },
  { label: 'BoundFunction.hyperparam_names', type: 'method', detail: 'BoundFunction method: hyperparam_names()' },
  { label: 'BoundFunction.init', type: 'method', detail: 'BoundFunction method: init()' },
  { label: 'BoundFunction.save', type: 'method', detail: 'BoundFunction method: save()' },
  { label: 'BoundFunction.to_latex', type: 'method', detail: 'BoundFunction method: to_latex()' },
  { label: 'BoundFunction.render', type: 'method', detail: 'BoundFunction method: render()' },
  { label: 'BoundFunction.hyperparam_values', type: 'method', detail: 'BoundFunction method: hyperparam_values()' },
  { label: 'BoundFunction.run_with', type: 'method', detail: 'BoundFunction method: run_with()' },
  { label: 'BoundFunction.run', type: 'method', detail: 'BoundFunction method: run()' },
  { label: 'BoundFunction.notation', type: 'method', detail: 'BoundFunction method: notation()' },
  { label: 'BoundFunction.evaluate', type: 'method', detail: 'BoundFunction method: evaluate()' },
  { label: 'BoundFunction.result', type: 'method', detail: 'BoundFunction method: result()' },
  { label: 'BoundFunction.var_bindings', type: 'method', detail: 'BoundFunction method: var_bindings()' },
  { label: 'BoundFunction.compile', type: 'method', detail: 'BoundFunction method: compile()' },
  { label: 'BoundFunction.to_torch', type: 'method', detail: 'BoundFunction method: to_torch()' },
  { label: 'BoundFunction.grad', type: 'method', detail: 'BoundFunction method: grad()' },
  { label: 'BoundFunction.device', type: 'method', detail: 'BoundFunction method: device()' },
  { label: 'BoundFunction.input_vars', type: 'method', detail: 'BoundFunction method: input_vars()' },
  { label: 'BoundFunction.to', type: 'method', detail: 'BoundFunction method: to()' },
  { label: 'BoundFunction.cuda', type: 'method', detail: 'BoundFunction method: cuda()' },
  { label: 'BoundFunction.cpu', type: 'method', detail: 'BoundFunction method: cpu()' },
  { label: 'ComputeGraph', type: 'class', detail: 'cm.symbols: ComputeGraph' },
  { label: 'ComputeGraph.var', type: 'method', detail: 'ComputeGraph method: var()' },
  { label: 'ComputeGraph.const', type: 'method', detail: 'ComputeGraph method: const()' },
  { label: 'ComputeGraph.pi', type: 'method', detail: 'ComputeGraph method: pi()' },
  { label: 'ComputeGraph.e', type: 'method', detail: 'ComputeGraph method: e()' },
  { label: 'ComputeGraph.inf', type: 'method', detail: 'ComputeGraph method: inf()' },
  { label: 'ComputeGraph.sqrt', type: 'method', detail: 'ComputeGraph method: sqrt()' },
  { label: 'ComputeGraph.sin', type: 'method', detail: 'ComputeGraph method: sin()' },
  { label: 'ComputeGraph.cos', type: 'method', detail: 'ComputeGraph method: cos()' },
  { label: 'ComputeGraph.tan', type: 'method', detail: 'ComputeGraph method: tan()' },
  { label: 'ComputeGraph.exp', type: 'method', detail: 'ComputeGraph method: exp()' },
  { label: 'ComputeGraph.log', type: 'method', detail: 'ComputeGraph method: log()' },
  { label: 'ComputeGraph.abs', type: 'method', detail: 'ComputeGraph method: abs()' },
  { label: 'ComputeGraph.expr', type: 'method', detail: 'ComputeGraph method: expr()' },
  { label: 'ComputeGraph.sum', type: 'method', detail: 'ComputeGraph method: sum()' },
  { label: 'ComputeGraph.prod', type: 'method', detail: 'ComputeGraph method: prod()' },
  { label: 'ComputeGraph.function', type: 'method', detail: 'ComputeGraph method: function()' },
  { label: 'ComputeGraph.get_function', type: 'method', detail: 'ComputeGraph method: get_function()' },
  { label: 'ComputeGraph.list_functions', type: 'method', detail: 'ComputeGraph method: list_functions()' },
  { label: 'ComputeGraph.gamma', type: 'method', detail: 'ComputeGraph method: gamma()' },
  { label: 'ComputeGraph.loggamma', type: 'method', detail: 'ComputeGraph method: loggamma()' },
  { label: 'ComputeGraph.digamma', type: 'method', detail: 'ComputeGraph method: digamma()' },
  { label: 'ComputeGraph.beta', type: 'method', detail: 'ComputeGraph method: beta()' },
  { label: 'ComputeGraph.factorial', type: 'method', detail: 'ComputeGraph method: factorial()' },
  { label: 'ComputeGraph.factorial2', type: 'method', detail: 'ComputeGraph method: factorial2()' },
  { label: 'ComputeGraph.binomial', type: 'method', detail: 'ComputeGraph method: binomial()' },
  { label: 'ComputeGraph.erf', type: 'method', detail: 'ComputeGraph method: erf()' },
  { label: 'ComputeGraph.erfc', type: 'method', detail: 'ComputeGraph method: erfc()' },
  { label: 'ComputeGraph.erfi', type: 'method', detail: 'ComputeGraph method: erfi()' },
  { label: 'ComputeGraph.besselj', type: 'method', detail: 'ComputeGraph method: besselj()' },
  { label: 'ComputeGraph.bessely', type: 'method', detail: 'ComputeGraph method: bessely()' },
  { label: 'ComputeGraph.besseli', type: 'method', detail: 'ComputeGraph method: besseli()' },
  { label: 'ComputeGraph.besselk', type: 'method', detail: 'ComputeGraph method: besselk()' },
  { label: 'ComputeGraph.jn', type: 'method', detail: 'ComputeGraph method: jn()' },
  { label: 'ComputeGraph.yn', type: 'method', detail: 'ComputeGraph method: yn()' },
  { label: 'ComputeGraph.hankel1', type: 'method', detail: 'ComputeGraph method: hankel1()' },
  { label: 'ComputeGraph.hankel2', type: 'method', detail: 'ComputeGraph method: hankel2()' },
  { label: 'ComputeGraph.airyai', type: 'method', detail: 'ComputeGraph method: airyai()' },
  { label: 'ComputeGraph.airybi', type: 'method', detail: 'ComputeGraph method: airybi()' },
  { label: 'ComputeGraph.airyaiprime', type: 'method', detail: 'ComputeGraph method: airyaiprime()' },
  { label: 'ComputeGraph.airybiprime', type: 'method', detail: 'ComputeGraph method: airybiprime()' },
  { label: 'ComputeGraph.legendre', type: 'method', detail: 'ComputeGraph method: legendre()' },
  { label: 'ComputeGraph.assoc_legendre', type: 'method', detail: 'ComputeGraph method: assoc_legendre()' },
  { label: 'ComputeGraph.hermite', type: 'method', detail: 'ComputeGraph method: hermite()' },
  { label: 'ComputeGraph.hermite_prob', type: 'method', detail: 'ComputeGraph method: hermite_prob()' },
  { label: 'ComputeGraph.laguerre', type: 'method', detail: 'ComputeGraph method: laguerre()' },
  { label: 'ComputeGraph.assoc_laguerre', type: 'method', detail: 'ComputeGraph method: assoc_laguerre()' },
  { label: 'ComputeGraph.chebyshevt', type: 'method', detail: 'ComputeGraph method: chebyshevt()' },
  { label: 'ComputeGraph.chebyshevu', type: 'method', detail: 'ComputeGraph method: chebyshevu()' },
  { label: 'ComputeGraph.gegenbauer', type: 'method', detail: 'ComputeGraph method: gegenbauer()' },
  { label: 'ComputeGraph.jacobi', type: 'method', detail: 'ComputeGraph method: jacobi()' },
  { label: 'ComputeGraph.hyper2f1', type: 'method', detail: 'ComputeGraph method: hyper2f1()' },
  { label: 'ComputeGraph.hyper1f1', type: 'method', detail: 'ComputeGraph method: hyper1f1()' },
  { label: 'ComputeGraph.hyper0f1', type: 'method', detail: 'ComputeGraph method: hyper0f1()' },
  { label: 'ComputeGraph.hyperpfq', type: 'method', detail: 'ComputeGraph method: hyperpfq()' },
  { label: 'ComputeGraph.elliptic_k', type: 'method', detail: 'ComputeGraph method: elliptic_k()' },
  { label: 'ComputeGraph.elliptic_e', type: 'method', detail: 'ComputeGraph method: elliptic_e()' },
  { label: 'ComputeGraph.elliptic_pi', type: 'method', detail: 'ComputeGraph method: elliptic_pi()' },
  { label: 'ComputeGraph.zeta', type: 'method', detail: 'ComputeGraph method: zeta()' },
  { label: 'ComputeGraph.polylog', type: 'method', detail: 'ComputeGraph method: polylog()' },
  { label: 'ComputeGraph.dirac', type: 'method', detail: 'ComputeGraph method: dirac()' },
  { label: 'ComputeGraph.heaviside', type: 'method', detail: 'ComputeGraph method: heaviside()' },
  { label: 'ComputeGraph.kronecker', type: 'method', detail: 'ComputeGraph method: kronecker()' },
  { label: 'ComputeGraph.levi_civita', type: 'method', detail: 'ComputeGraph method: levi_civita()' },
  { label: 'ComputeGraph.validate', type: 'method', detail: 'ComputeGraph method: validate()' },
  { label: 'ComputeGraph.coerce', type: 'method', detail: 'ComputeGraph method: coerce()' },
  { label: 'ComputeGraph.define', type: 'method', detail: 'ComputeGraph method: define()' },
  { label: 'ComputeGraph.free_variables', type: 'method', detail: 'ComputeGraph method: free_variables()' },
  { label: 'ComputeGraph.expression', type: 'method', detail: 'ComputeGraph method: expression()' },
  { label: 'ComputeGraph.hyperparam_names', type: 'method', detail: 'ComputeGraph method: hyperparam_names()' },
  { label: 'ComputeGraph.init', type: 'method', detail: 'ComputeGraph method: init()' },
  { label: 'ComputeGraph.save', type: 'method', detail: 'ComputeGraph method: save()' },
  { label: 'ComputeGraph.to_latex', type: 'method', detail: 'ComputeGraph method: to_latex()' },
  { label: 'ComputeGraph.render', type: 'method', detail: 'ComputeGraph method: render()' },
  { label: 'ComputeGraph.hyperparam_values', type: 'method', detail: 'ComputeGraph method: hyperparam_values()' },
  { label: 'ComputeGraph.run_with', type: 'method', detail: 'ComputeGraph method: run_with()' },
  { label: 'ComputeGraph.run', type: 'method', detail: 'ComputeGraph method: run()' },
  { label: 'ComputeGraph.notation', type: 'method', detail: 'ComputeGraph method: notation()' },
  { label: 'ComputeGraph.evaluate', type: 'method', detail: 'ComputeGraph method: evaluate()' },
  { label: 'ComputeGraph.result', type: 'method', detail: 'ComputeGraph method: result()' },
  { label: 'ComputeGraph.var_bindings', type: 'method', detail: 'ComputeGraph method: var_bindings()' },
  { label: 'ComputeGraph.compile', type: 'method', detail: 'ComputeGraph method: compile()' },
  { label: 'ComputeGraph.to_torch', type: 'method', detail: 'ComputeGraph method: to_torch()' },
  { label: 'ComputeGraph.grad', type: 'method', detail: 'ComputeGraph method: grad()' },
  { label: 'ComputeGraph.device', type: 'method', detail: 'ComputeGraph method: device()' },
  { label: 'ComputeGraph.input_vars', type: 'method', detail: 'ComputeGraph method: input_vars()' },
  { label: 'ComputeGraph.to', type: 'method', detail: 'ComputeGraph method: to()' },
  { label: 'ComputeGraph.cuda', type: 'method', detail: 'ComputeGraph method: cuda()' },
  { label: 'ComputeGraph.cpu', type: 'method', detail: 'ComputeGraph method: cpu()' },
  { label: 'TorchFunction', type: 'class', detail: 'cm.symbols: TorchFunction' },
  { label: 'TorchFunction.var', type: 'method', detail: 'TorchFunction method: var()' },
  { label: 'TorchFunction.const', type: 'method', detail: 'TorchFunction method: const()' },
  { label: 'TorchFunction.pi', type: 'method', detail: 'TorchFunction method: pi()' },
  { label: 'TorchFunction.e', type: 'method', detail: 'TorchFunction method: e()' },
  { label: 'TorchFunction.inf', type: 'method', detail: 'TorchFunction method: inf()' },
  { label: 'TorchFunction.sqrt', type: 'method', detail: 'TorchFunction method: sqrt()' },
  { label: 'TorchFunction.sin', type: 'method', detail: 'TorchFunction method: sin()' },
  { label: 'TorchFunction.cos', type: 'method', detail: 'TorchFunction method: cos()' },
  { label: 'TorchFunction.tan', type: 'method', detail: 'TorchFunction method: tan()' },
  { label: 'TorchFunction.exp', type: 'method', detail: 'TorchFunction method: exp()' },
  { label: 'TorchFunction.log', type: 'method', detail: 'TorchFunction method: log()' },
  { label: 'TorchFunction.abs', type: 'method', detail: 'TorchFunction method: abs()' },
  { label: 'TorchFunction.expr', type: 'method', detail: 'TorchFunction method: expr()' },
  { label: 'TorchFunction.sum', type: 'method', detail: 'TorchFunction method: sum()' },
  { label: 'TorchFunction.prod', type: 'method', detail: 'TorchFunction method: prod()' },
  { label: 'TorchFunction.function', type: 'method', detail: 'TorchFunction method: function()' },
  { label: 'TorchFunction.get_function', type: 'method', detail: 'TorchFunction method: get_function()' },
  { label: 'TorchFunction.list_functions', type: 'method', detail: 'TorchFunction method: list_functions()' },
  { label: 'TorchFunction.gamma', type: 'method', detail: 'TorchFunction method: gamma()' },
  { label: 'TorchFunction.loggamma', type: 'method', detail: 'TorchFunction method: loggamma()' },
  { label: 'TorchFunction.digamma', type: 'method', detail: 'TorchFunction method: digamma()' },
  { label: 'TorchFunction.beta', type: 'method', detail: 'TorchFunction method: beta()' },
  { label: 'TorchFunction.factorial', type: 'method', detail: 'TorchFunction method: factorial()' },
  { label: 'TorchFunction.factorial2', type: 'method', detail: 'TorchFunction method: factorial2()' },
  { label: 'TorchFunction.binomial', type: 'method', detail: 'TorchFunction method: binomial()' },
  { label: 'TorchFunction.erf', type: 'method', detail: 'TorchFunction method: erf()' },
  { label: 'TorchFunction.erfc', type: 'method', detail: 'TorchFunction method: erfc()' },
  { label: 'TorchFunction.erfi', type: 'method', detail: 'TorchFunction method: erfi()' },
  { label: 'TorchFunction.besselj', type: 'method', detail: 'TorchFunction method: besselj()' },
  { label: 'TorchFunction.bessely', type: 'method', detail: 'TorchFunction method: bessely()' },
  { label: 'TorchFunction.besseli', type: 'method', detail: 'TorchFunction method: besseli()' },
  { label: 'TorchFunction.besselk', type: 'method', detail: 'TorchFunction method: besselk()' },
  { label: 'TorchFunction.jn', type: 'method', detail: 'TorchFunction method: jn()' },
  { label: 'TorchFunction.yn', type: 'method', detail: 'TorchFunction method: yn()' },
  { label: 'TorchFunction.hankel1', type: 'method', detail: 'TorchFunction method: hankel1()' },
  { label: 'TorchFunction.hankel2', type: 'method', detail: 'TorchFunction method: hankel2()' },
  { label: 'TorchFunction.airyai', type: 'method', detail: 'TorchFunction method: airyai()' },
  { label: 'TorchFunction.airybi', type: 'method', detail: 'TorchFunction method: airybi()' },
  { label: 'TorchFunction.airyaiprime', type: 'method', detail: 'TorchFunction method: airyaiprime()' },
  { label: 'TorchFunction.airybiprime', type: 'method', detail: 'TorchFunction method: airybiprime()' },
  { label: 'TorchFunction.legendre', type: 'method', detail: 'TorchFunction method: legendre()' },
  { label: 'TorchFunction.assoc_legendre', type: 'method', detail: 'TorchFunction method: assoc_legendre()' },
  { label: 'TorchFunction.hermite', type: 'method', detail: 'TorchFunction method: hermite()' },
  { label: 'TorchFunction.hermite_prob', type: 'method', detail: 'TorchFunction method: hermite_prob()' },
  { label: 'TorchFunction.laguerre', type: 'method', detail: 'TorchFunction method: laguerre()' },
  { label: 'TorchFunction.assoc_laguerre', type: 'method', detail: 'TorchFunction method: assoc_laguerre()' },
  { label: 'TorchFunction.chebyshevt', type: 'method', detail: 'TorchFunction method: chebyshevt()' },
  { label: 'TorchFunction.chebyshevu', type: 'method', detail: 'TorchFunction method: chebyshevu()' },
  { label: 'TorchFunction.gegenbauer', type: 'method', detail: 'TorchFunction method: gegenbauer()' },
  { label: 'TorchFunction.jacobi', type: 'method', detail: 'TorchFunction method: jacobi()' },
  { label: 'TorchFunction.hyper2f1', type: 'method', detail: 'TorchFunction method: hyper2f1()' },
  { label: 'TorchFunction.hyper1f1', type: 'method', detail: 'TorchFunction method: hyper1f1()' },
  { label: 'TorchFunction.hyper0f1', type: 'method', detail: 'TorchFunction method: hyper0f1()' },
  { label: 'TorchFunction.hyperpfq', type: 'method', detail: 'TorchFunction method: hyperpfq()' },
  { label: 'TorchFunction.elliptic_k', type: 'method', detail: 'TorchFunction method: elliptic_k()' },
  { label: 'TorchFunction.elliptic_e', type: 'method', detail: 'TorchFunction method: elliptic_e()' },
  { label: 'TorchFunction.elliptic_pi', type: 'method', detail: 'TorchFunction method: elliptic_pi()' },
  { label: 'TorchFunction.zeta', type: 'method', detail: 'TorchFunction method: zeta()' },
  { label: 'TorchFunction.polylog', type: 'method', detail: 'TorchFunction method: polylog()' },
  { label: 'TorchFunction.dirac', type: 'method', detail: 'TorchFunction method: dirac()' },
  { label: 'TorchFunction.heaviside', type: 'method', detail: 'TorchFunction method: heaviside()' },
  { label: 'TorchFunction.kronecker', type: 'method', detail: 'TorchFunction method: kronecker()' },
  { label: 'TorchFunction.levi_civita', type: 'method', detail: 'TorchFunction method: levi_civita()' },
  { label: 'TorchFunction.validate', type: 'method', detail: 'TorchFunction method: validate()' },
  { label: 'TorchFunction.coerce', type: 'method', detail: 'TorchFunction method: coerce()' },
  { label: 'TorchFunction.define', type: 'method', detail: 'TorchFunction method: define()' },
  { label: 'TorchFunction.free_variables', type: 'method', detail: 'TorchFunction method: free_variables()' },
  { label: 'TorchFunction.expression', type: 'method', detail: 'TorchFunction method: expression()' },
  { label: 'TorchFunction.hyperparam_names', type: 'method', detail: 'TorchFunction method: hyperparam_names()' },
  { label: 'TorchFunction.init', type: 'method', detail: 'TorchFunction method: init()' },
  { label: 'TorchFunction.save', type: 'method', detail: 'TorchFunction method: save()' },
  { label: 'TorchFunction.to_latex', type: 'method', detail: 'TorchFunction method: to_latex()' },
  { label: 'TorchFunction.render', type: 'method', detail: 'TorchFunction method: render()' },
  { label: 'TorchFunction.hyperparam_values', type: 'method', detail: 'TorchFunction method: hyperparam_values()' },
  { label: 'TorchFunction.run_with', type: 'method', detail: 'TorchFunction method: run_with()' },
  { label: 'TorchFunction.run', type: 'method', detail: 'TorchFunction method: run()' },
  { label: 'TorchFunction.notation', type: 'method', detail: 'TorchFunction method: notation()' },
  { label: 'TorchFunction.evaluate', type: 'method', detail: 'TorchFunction method: evaluate()' },
  { label: 'TorchFunction.result', type: 'method', detail: 'TorchFunction method: result()' },
  { label: 'TorchFunction.var_bindings', type: 'method', detail: 'TorchFunction method: var_bindings()' },
  { label: 'TorchFunction.compile', type: 'method', detail: 'TorchFunction method: compile()' },
  { label: 'TorchFunction.to_torch', type: 'method', detail: 'TorchFunction method: to_torch()' },
  { label: 'TorchFunction.grad', type: 'method', detail: 'TorchFunction method: grad()' },
  { label: 'TorchFunction.device', type: 'method', detail: 'TorchFunction method: device()' },
  { label: 'TorchFunction.input_vars', type: 'method', detail: 'TorchFunction method: input_vars()' },
  { label: 'TorchFunction.to', type: 'method', detail: 'TorchFunction method: to()' },
  { label: 'TorchFunction.cuda', type: 'method', detail: 'TorchFunction method: cuda()' },
  { label: 'TorchFunction.cpu', type: 'method', detail: 'TorchFunction method: cpu()' },
  { label: 'TorchGradFunction', type: 'class', detail: 'cm.symbols: TorchGradFunction' },
  { label: 'TorchGradFunction.var', type: 'method', detail: 'TorchGradFunction method: var()' },
  { label: 'TorchGradFunction.const', type: 'method', detail: 'TorchGradFunction method: const()' },
  { label: 'TorchGradFunction.pi', type: 'method', detail: 'TorchGradFunction method: pi()' },
  { label: 'TorchGradFunction.e', type: 'method', detail: 'TorchGradFunction method: e()' },
  { label: 'TorchGradFunction.inf', type: 'method', detail: 'TorchGradFunction method: inf()' },
  { label: 'TorchGradFunction.sqrt', type: 'method', detail: 'TorchGradFunction method: sqrt()' },
  { label: 'TorchGradFunction.sin', type: 'method', detail: 'TorchGradFunction method: sin()' },
  { label: 'TorchGradFunction.cos', type: 'method', detail: 'TorchGradFunction method: cos()' },
  { label: 'TorchGradFunction.tan', type: 'method', detail: 'TorchGradFunction method: tan()' },
  { label: 'TorchGradFunction.exp', type: 'method', detail: 'TorchGradFunction method: exp()' },
  { label: 'TorchGradFunction.log', type: 'method', detail: 'TorchGradFunction method: log()' },
  { label: 'TorchGradFunction.abs', type: 'method', detail: 'TorchGradFunction method: abs()' },
  { label: 'TorchGradFunction.expr', type: 'method', detail: 'TorchGradFunction method: expr()' },
  { label: 'TorchGradFunction.sum', type: 'method', detail: 'TorchGradFunction method: sum()' },
  { label: 'TorchGradFunction.prod', type: 'method', detail: 'TorchGradFunction method: prod()' },
  { label: 'TorchGradFunction.function', type: 'method', detail: 'TorchGradFunction method: function()' },
  { label: 'TorchGradFunction.get_function', type: 'method', detail: 'TorchGradFunction method: get_function()' },
  { label: 'TorchGradFunction.list_functions', type: 'method', detail: 'TorchGradFunction method: list_functions()' },
  { label: 'TorchGradFunction.gamma', type: 'method', detail: 'TorchGradFunction method: gamma()' },
  { label: 'TorchGradFunction.loggamma', type: 'method', detail: 'TorchGradFunction method: loggamma()' },
  { label: 'TorchGradFunction.digamma', type: 'method', detail: 'TorchGradFunction method: digamma()' },
  { label: 'TorchGradFunction.beta', type: 'method', detail: 'TorchGradFunction method: beta()' },
  { label: 'TorchGradFunction.factorial', type: 'method', detail: 'TorchGradFunction method: factorial()' },
  { label: 'TorchGradFunction.factorial2', type: 'method', detail: 'TorchGradFunction method: factorial2()' },
  { label: 'TorchGradFunction.binomial', type: 'method', detail: 'TorchGradFunction method: binomial()' },
  { label: 'TorchGradFunction.erf', type: 'method', detail: 'TorchGradFunction method: erf()' },
  { label: 'TorchGradFunction.erfc', type: 'method', detail: 'TorchGradFunction method: erfc()' },
  { label: 'TorchGradFunction.erfi', type: 'method', detail: 'TorchGradFunction method: erfi()' },
  { label: 'TorchGradFunction.besselj', type: 'method', detail: 'TorchGradFunction method: besselj()' },
  { label: 'TorchGradFunction.bessely', type: 'method', detail: 'TorchGradFunction method: bessely()' },
  { label: 'TorchGradFunction.besseli', type: 'method', detail: 'TorchGradFunction method: besseli()' },
  { label: 'TorchGradFunction.besselk', type: 'method', detail: 'TorchGradFunction method: besselk()' },
  { label: 'TorchGradFunction.jn', type: 'method', detail: 'TorchGradFunction method: jn()' },
  { label: 'TorchGradFunction.yn', type: 'method', detail: 'TorchGradFunction method: yn()' },
  { label: 'TorchGradFunction.hankel1', type: 'method', detail: 'TorchGradFunction method: hankel1()' },
  { label: 'TorchGradFunction.hankel2', type: 'method', detail: 'TorchGradFunction method: hankel2()' },
  { label: 'TorchGradFunction.airyai', type: 'method', detail: 'TorchGradFunction method: airyai()' },
  { label: 'TorchGradFunction.airybi', type: 'method', detail: 'TorchGradFunction method: airybi()' },
  { label: 'TorchGradFunction.airyaiprime', type: 'method', detail: 'TorchGradFunction method: airyaiprime()' },
  { label: 'TorchGradFunction.airybiprime', type: 'method', detail: 'TorchGradFunction method: airybiprime()' },
  { label: 'TorchGradFunction.legendre', type: 'method', detail: 'TorchGradFunction method: legendre()' },
  { label: 'TorchGradFunction.assoc_legendre', type: 'method', detail: 'TorchGradFunction method: assoc_legendre()' },
  { label: 'TorchGradFunction.hermite', type: 'method', detail: 'TorchGradFunction method: hermite()' },
  { label: 'TorchGradFunction.hermite_prob', type: 'method', detail: 'TorchGradFunction method: hermite_prob()' },
  { label: 'TorchGradFunction.laguerre', type: 'method', detail: 'TorchGradFunction method: laguerre()' },
  { label: 'TorchGradFunction.assoc_laguerre', type: 'method', detail: 'TorchGradFunction method: assoc_laguerre()' },
  { label: 'TorchGradFunction.chebyshevt', type: 'method', detail: 'TorchGradFunction method: chebyshevt()' },
  { label: 'TorchGradFunction.chebyshevu', type: 'method', detail: 'TorchGradFunction method: chebyshevu()' },
  { label: 'TorchGradFunction.gegenbauer', type: 'method', detail: 'TorchGradFunction method: gegenbauer()' },
  { label: 'TorchGradFunction.jacobi', type: 'method', detail: 'TorchGradFunction method: jacobi()' },
  { label: 'TorchGradFunction.hyper2f1', type: 'method', detail: 'TorchGradFunction method: hyper2f1()' },
  { label: 'TorchGradFunction.hyper1f1', type: 'method', detail: 'TorchGradFunction method: hyper1f1()' },
  { label: 'TorchGradFunction.hyper0f1', type: 'method', detail: 'TorchGradFunction method: hyper0f1()' },
  { label: 'TorchGradFunction.hyperpfq', type: 'method', detail: 'TorchGradFunction method: hyperpfq()' },
  { label: 'TorchGradFunction.elliptic_k', type: 'method', detail: 'TorchGradFunction method: elliptic_k()' },
  { label: 'TorchGradFunction.elliptic_e', type: 'method', detail: 'TorchGradFunction method: elliptic_e()' },
  { label: 'TorchGradFunction.elliptic_pi', type: 'method', detail: 'TorchGradFunction method: elliptic_pi()' },
  { label: 'TorchGradFunction.zeta', type: 'method', detail: 'TorchGradFunction method: zeta()' },
  { label: 'TorchGradFunction.polylog', type: 'method', detail: 'TorchGradFunction method: polylog()' },
  { label: 'TorchGradFunction.dirac', type: 'method', detail: 'TorchGradFunction method: dirac()' },
  { label: 'TorchGradFunction.heaviside', type: 'method', detail: 'TorchGradFunction method: heaviside()' },
  { label: 'TorchGradFunction.kronecker', type: 'method', detail: 'TorchGradFunction method: kronecker()' },
  { label: 'TorchGradFunction.levi_civita', type: 'method', detail: 'TorchGradFunction method: levi_civita()' },
  { label: 'TorchGradFunction.validate', type: 'method', detail: 'TorchGradFunction method: validate()' },
  { label: 'TorchGradFunction.coerce', type: 'method', detail: 'TorchGradFunction method: coerce()' },
  { label: 'TorchGradFunction.define', type: 'method', detail: 'TorchGradFunction method: define()' },
  { label: 'TorchGradFunction.free_variables', type: 'method', detail: 'TorchGradFunction method: free_variables()' },
  { label: 'TorchGradFunction.expression', type: 'method', detail: 'TorchGradFunction method: expression()' },
  { label: 'TorchGradFunction.hyperparam_names', type: 'method', detail: 'TorchGradFunction method: hyperparam_names()' },
  { label: 'TorchGradFunction.init', type: 'method', detail: 'TorchGradFunction method: init()' },
  { label: 'TorchGradFunction.save', type: 'method', detail: 'TorchGradFunction method: save()' },
  { label: 'TorchGradFunction.to_latex', type: 'method', detail: 'TorchGradFunction method: to_latex()' },
  { label: 'TorchGradFunction.render', type: 'method', detail: 'TorchGradFunction method: render()' },
  { label: 'TorchGradFunction.hyperparam_values', type: 'method', detail: 'TorchGradFunction method: hyperparam_values()' },
  { label: 'TorchGradFunction.run_with', type: 'method', detail: 'TorchGradFunction method: run_with()' },
  { label: 'TorchGradFunction.run', type: 'method', detail: 'TorchGradFunction method: run()' },
  { label: 'TorchGradFunction.notation', type: 'method', detail: 'TorchGradFunction method: notation()' },
  { label: 'TorchGradFunction.evaluate', type: 'method', detail: 'TorchGradFunction method: evaluate()' },
  { label: 'TorchGradFunction.result', type: 'method', detail: 'TorchGradFunction method: result()' },
  { label: 'TorchGradFunction.var_bindings', type: 'method', detail: 'TorchGradFunction method: var_bindings()' },
  { label: 'TorchGradFunction.compile', type: 'method', detail: 'TorchGradFunction method: compile()' },
  { label: 'TorchGradFunction.to_torch', type: 'method', detail: 'TorchGradFunction method: to_torch()' },
  { label: 'TorchGradFunction.grad', type: 'method', detail: 'TorchGradFunction method: grad()' },
  { label: 'TorchGradFunction.device', type: 'method', detail: 'TorchGradFunction method: device()' },
  { label: 'TorchGradFunction.input_vars', type: 'method', detail: 'TorchGradFunction method: input_vars()' },
  { label: 'TorchGradFunction.to', type: 'method', detail: 'TorchGradFunction method: to()' },
  { label: 'TorchGradFunction.cuda', type: 'method', detail: 'TorchGradFunction method: cuda()' },
  { label: 'TorchGradFunction.cpu', type: 'method', detail: 'TorchGradFunction method: cpu()' },
  { label: 'SpecialFunction', type: 'class', detail: 'cm.symbols: SpecialFunction' },
  { label: 'Gamma', type: 'class', detail: 'cm.symbols: Gamma' },
  { label: 'LogGamma', type: 'class', detail: 'cm.symbols: LogGamma' },
  { label: 'Digamma', type: 'class', detail: 'cm.symbols: Digamma' },
  { label: 'Beta', type: 'class', detail: 'cm.symbols: Beta' },
  { label: 'Factorial', type: 'class', detail: 'cm.symbols: Factorial' },
  { label: 'DoubleFactorial', type: 'class', detail: 'cm.symbols: DoubleFactorial' },
  { label: 'Binomial', type: 'class', detail: 'cm.symbols: Binomial' },
  { label: 'Erf', type: 'class', detail: 'cm.symbols: Erf' },
  { label: 'Erfc', type: 'class', detail: 'cm.symbols: Erfc' },
  { label: 'Erfi', type: 'class', detail: 'cm.symbols: Erfi' },
  { label: 'BesselJ', type: 'class', detail: 'cm.symbols: BesselJ' },
  { label: 'BesselY', type: 'class', detail: 'cm.symbols: BesselY' },
  { label: 'BesselI', type: 'class', detail: 'cm.symbols: BesselI' },
  { label: 'BesselK', type: 'class', detail: 'cm.symbols: BesselK' },
  { label: 'SphericalBesselJ', type: 'class', detail: 'cm.symbols: SphericalBesselJ' },
  { label: 'SphericalBesselY', type: 'class', detail: 'cm.symbols: SphericalBesselY' },
  { label: 'Hankel1', type: 'class', detail: 'cm.symbols: Hankel1' },
  { label: 'Hankel2', type: 'class', detail: 'cm.symbols: Hankel2' },
  { label: 'AiryAi', type: 'class', detail: 'cm.symbols: AiryAi' },
  { label: 'AiryBi', type: 'class', detail: 'cm.symbols: AiryBi' },
  { label: 'AiryAiPrime', type: 'class', detail: 'cm.symbols: AiryAiPrime' },
  { label: 'AiryBiPrime', type: 'class', detail: 'cm.symbols: AiryBiPrime' },
  { label: 'Legendre', type: 'class', detail: 'cm.symbols: Legendre' },
  { label: 'AssocLegendre', type: 'class', detail: 'cm.symbols: AssocLegendre' },
  { label: 'Hermite', type: 'class', detail: 'cm.symbols: Hermite' },
  { label: 'HermiteProb', type: 'class', detail: 'cm.symbols: HermiteProb' },
  { label: 'Laguerre', type: 'class', detail: 'cm.symbols: Laguerre' },
  { label: 'AssocLaguerre', type: 'class', detail: 'cm.symbols: AssocLaguerre' },
  { label: 'Chebyshev1', type: 'class', detail: 'cm.symbols: Chebyshev1' },
  { label: 'Chebyshev2', type: 'class', detail: 'cm.symbols: Chebyshev2' },
  { label: 'Gegenbauer', type: 'class', detail: 'cm.symbols: Gegenbauer' },
  { label: 'Jacobi', type: 'class', detail: 'cm.symbols: Jacobi' },
  { label: 'SphericalHarmonic', type: 'class', detail: 'cm.symbols: SphericalHarmonic' },
  { label: 'RealSphericalHarmonic', type: 'class', detail: 'cm.symbols: RealSphericalHarmonic' },
  { label: 'Hypergeometric2F1', type: 'class', detail: 'cm.symbols: Hypergeometric2F1' },
  { label: 'Hypergeometric1F1', type: 'class', detail: 'cm.symbols: Hypergeometric1F1' },
  { label: 'Hypergeometric0F1', type: 'class', detail: 'cm.symbols: Hypergeometric0F1' },
  { label: 'HypergeometricPFQ', type: 'class', detail: 'cm.symbols: HypergeometricPFQ' },
  { label: 'EllipticK', type: 'class', detail: 'cm.symbols: EllipticK' },
  { label: 'EllipticE', type: 'class', detail: 'cm.symbols: EllipticE' },
  { label: 'EllipticPi', type: 'class', detail: 'cm.symbols: EllipticPi' },
  { label: 'Zeta', type: 'class', detail: 'cm.symbols: Zeta' },
  { label: 'PolyLog', type: 'class', detail: 'cm.symbols: PolyLog' },
  { label: 'DiracDelta', type: 'class', detail: 'cm.symbols: DiracDelta' },
  { label: 'Heaviside', type: 'class', detail: 'cm.symbols: Heaviside' },
  { label: 'KroneckerDelta', type: 'class', detail: 'cm.symbols: KroneckerDelta' },
  { label: 'LeviCivita', type: 'class', detail: 'cm.symbols: LeviCivita' },
  { label: 'ClebschGordan', type: 'class', detail: 'cm.symbols: ClebschGordan' },
  { label: 'Wigner3j', type: 'class', detail: 'cm.symbols: Wigner3j' },
  { label: 'Wigner6j', type: 'class', detail: 'cm.symbols: Wigner6j' },
  { label: 'Wigner9j', type: 'class', detail: 'cm.symbols: Wigner9j' },
  { label: 'DifferentialOperator', type: 'class', detail: 'cm.symbols: DifferentialOperator' },
  { label: 'PartialDerivative', type: 'class', detail: 'cm.symbols: PartialDerivative' },
  { label: 'Gradient', type: 'class', detail: 'cm.symbols: Gradient' },
  { label: 'Laplacian', type: 'class', detail: 'cm.symbols: Laplacian' },
  { label: 'HydrogenRadial', type: 'class', detail: 'cm.symbols: HydrogenRadial' },
  { label: 'HydrogenOrbital', type: 'class', detail: 'cm.symbols: HydrogenOrbital' },
  { label: 'SlaterTypeOrbital', type: 'class', detail: 'cm.symbols: SlaterTypeOrbital' },
  { label: 'GaussianTypeOrbital', type: 'class', detail: 'cm.symbols: GaussianTypeOrbital' },
  { label: 'ContractedGTO', type: 'class', detail: 'cm.symbols: ContractedGTO' },
  // CM symbols - functions
  { label: 'latex', type: 'function', detail: 'cm.symbols: latex' },
  { label: 'equation', type: 'function', detail: 'cm.symbols: equation' },
  { label: 'align', type: 'function', detail: 'cm.symbols: align' },
  { label: 'matrix', type: 'function', detail: 'cm.symbols: matrix' },
  { label: 'bullets', type: 'function', detail: 'cm.symbols: bullets' },
  { label: 'numbered', type: 'function', detail: 'cm.symbols: numbered' },
  { label: 'items', type: 'function', detail: 'cm.symbols: items' },
  { label: 'set_notation', type: 'function', detail: 'cm.symbols: set_notation' },
  { label: 'set_line_height', type: 'function', detail: 'cm.symbols: set_line_height' },
  { label: 'chemical', type: 'function', detail: 'cm.symbols: chemical' },
  { label: 'reaction', type: 'function', detail: 'cm.symbols: reaction' },
  { label: 'fraction', type: 'function', detail: 'cm.symbols: fraction' },
  { label: 'sqrt', type: 'function', detail: 'cm.symbols: sqrt' },
  { label: 'cm.views', type: 'module', detail: 'Chemical Machines Views Package' },
  { label: 'from cm.views import', type: 'text', detail: 'import CM views' },
  // CM views - constants
  { label: 'ELEMENT_DATA', type: 'constant', detail: 'cm.views: element data' },
  // CM views - classes
  { label: 'COLORMAPS', type: 'class', detail: 'cm.views: COLORMAPS' },
  // CM views - functions
  { label: 'html', type: 'function', detail: 'cm.views: html' },
  { label: 'text', type: 'function', detail: 'cm.views: text' },
  { label: 'log', type: 'function', detail: 'cm.views: log' },
  { label: 'clear', type: 'function', detail: 'cm.views: clear' },
  { label: 'image', type: 'function', detail: 'cm.views: image' },
  { label: 'savefig', type: 'function', detail: 'cm.views: savefig' },
  { label: 'dataframe', type: 'function', detail: 'cm.views: dataframe' },
  { label: 'table', type: 'function', detail: 'cm.views: table' },
  { label: 'scatter_3d', type: 'function', detail: 'cm.views: scatter_3d' },
  { label: 'surface', type: 'function', detail: 'cm.views: surface' },
  { label: 'molecule', type: 'function', detail: 'cm.views: molecule' },
  { label: 'crystal', type: 'function', detail: 'cm.views: crystal' },
  { label: 'orbital', type: 'function', detail: 'cm.views: orbital' },
]

// Math builder method completions (shown after typing "m." where m is a Math instance)
const mathMethodCompletions = [
  { label: 'var', type: 'method', detail: 'add variable', apply: 'var(' },
  { label: 'equals', type: 'method', detail: 'add equals sign', apply: 'equals()' },
  { label: 'plus', type: 'method', detail: 'add plus sign', apply: 'plus()' },
  { label: 'minus', type: 'method', detail: 'add minus sign', apply: 'minus()' },
  { label: 'times', type: 'method', detail: 'add times sign', apply: 'times()' },
  { label: 'frac', type: 'method', detail: 'add fraction', apply: 'frac(' },
  { label: 'sqrt', type: 'method', detail: 'add square root', apply: 'sqrt(' },
  { label: 'sum', type: 'method', detail: 'add summation', apply: 'sum(' },
  { label: 'prod', type: 'method', detail: 'add product', apply: 'prod(' },
  { label: 'integral', type: 'method', detail: 'add integral', apply: 'integral(' },
  { label: 'bra', type: 'method', detail: 'add bra ⟨x|', apply: 'bra(' },
  { label: 'ket', type: 'method', detail: 'add ket |x⟩', apply: 'ket(' },
  { label: 'braket', type: 'method', detail: 'add braket ⟨x|y⟩', apply: 'braket(' },
  { label: 'dagger', type: 'method', detail: 'add dagger †', apply: 'dagger()' },
  { label: 'conj', type: 'method', detail: 'add conjugate *', apply: 'conj()' },
  { label: 'op', type: 'method', detail: 'add operator with hat', apply: 'op(' },
  { label: 'expval', type: 'method', detail: 'expectation value', apply: 'expval(' },
  { label: 'comm', type: 'method', detail: 'commutator [A,B]', apply: 'comm(' },
  { label: 'matelem', type: 'method', detail: 'matrix element ⟨a|H|b⟩', apply: 'matelem(' },
  { label: 'render', type: 'method', detail: 'render equation', apply: 'render()' },
  { label: 'sup', type: 'method', detail: 'superscript', apply: 'sup(' },
  { label: 'sub', type: 'method', detail: 'subscript', apply: 'sub(' },
  { label: 'text', type: 'method', detail: 'add text', apply: 'text(' },
  { label: 'space', type: 'method', detail: 'add space', apply: 'space()' },
  { label: 'newline', type: 'method', detail: 'add newline', apply: 'newline()' },
  { label: 'paren', type: 'method', detail: 'add parentheses', apply: 'paren(' },
  { label: 'bracket', type: 'method', detail: 'add brackets', apply: 'bracket(' },
  { label: 'brace', type: 'method', detail: 'add braces', apply: 'brace(' },
  { label: 'abs', type: 'method', detail: 'absolute value', apply: 'abs(' },
  { label: 'norm', type: 'method', detail: 'norm ||x||', apply: 'norm(' },
  { label: 'hbar', type: 'method', detail: 'add ℏ', apply: 'hbar()' },
  { label: 'nabla', type: 'method', detail: 'add ∇', apply: 'nabla()' },
  { label: 'partial', type: 'method', detail: 'partial derivative', apply: 'partial(' },
  { label: 'derivative', type: 'method', detail: 'derivative d/dx', apply: 'derivative(' },
  // Symbolic determinants
  { label: 'determinant_bra', type: 'method', detail: 'det expansion as bras', apply: 'determinant_bra(' },
  { label: 'determinant_ket', type: 'method', detail: 'det expansion as kets', apply: 'determinant_ket(' },
  { label: 'determinant_braket', type: 'method', detail: 'det expansion as brakets', apply: 'determinant_braket(' },
  { label: 'determinant_product', type: 'method', detail: 'det as products', apply: 'determinant_product(' },
  { label: 'determinant_subscript', type: 'method', detail: 'det with subscripts', apply: 'determinant_subscript(' },
  { label: 'slater_determinant', type: 'method', detail: 'Slater determinant', apply: 'slater_determinant(' },
  { label: 'slater_ket', type: 'method', detail: 'Slater ket notation', apply: 'slater_ket(' },
  // Determinant inner products
  { label: 'determinant_inner_product', type: 'method', detail: 'inner product with orthogonality', apply: 'determinant_inner_product(' },
  { label: 'determinant_inner_product_simplified', type: 'method', detail: 'simplified inner product', apply: 'determinant_inner_product_simplified(' },
  { label: 'slater_inner_product', type: 'method', detail: 'Slater det inner product', apply: 'slater_inner_product(' },
  { label: 'determinant_overlap_expansion', type: 'method', detail: 'full overlap expansion', apply: 'determinant_overlap_expansion(' },
  // Special functions - Gamma family
  { label: 'gamma', type: 'method', detail: 'gamma function Γ(z)', apply: 'gamma(' },
  { label: 'loggamma', type: 'method', detail: 'log gamma ln Γ(z)', apply: 'loggamma(' },
  { label: 'digamma', type: 'method', detail: 'digamma ψ(z)', apply: 'digamma(' },
  { label: 'beta', type: 'method', detail: 'beta function B(a,b)', apply: 'beta(' },
  { label: 'factorial', type: 'method', detail: 'factorial n!', apply: 'factorial(' },
  { label: 'factorial2', type: 'method', detail: 'double factorial n!!', apply: 'factorial2(' },
  { label: 'binomial', type: 'method', detail: 'binomial C(n,k)', apply: 'binomial(' },
  // Special functions - Error functions
  { label: 'erf', type: 'method', detail: 'error function', apply: 'erf(' },
  { label: 'erfc', type: 'method', detail: 'complementary error', apply: 'erfc(' },
  { label: 'erfi', type: 'method', detail: 'imaginary error', apply: 'erfi(' },
  // Special functions - Bessel functions
  { label: 'besselj', type: 'method', detail: 'Bessel J_ν(z)', apply: 'besselj(' },
  { label: 'bessely', type: 'method', detail: 'Bessel Y_ν(z)', apply: 'bessely(' },
  { label: 'besseli', type: 'method', detail: 'modified Bessel I_ν(z)', apply: 'besseli(' },
  { label: 'besselk', type: 'method', detail: 'modified Bessel K_ν(z)', apply: 'besselk(' },
  { label: 'jn', type: 'method', detail: 'spherical Bessel j_n(z)', apply: 'jn(' },
  { label: 'yn', type: 'method', detail: 'spherical Bessel y_n(z)', apply: 'yn(' },
  { label: 'hankel1', type: 'method', detail: 'Hankel H_ν^(1)(z)', apply: 'hankel1(' },
  { label: 'hankel2', type: 'method', detail: 'Hankel H_ν^(2)(z)', apply: 'hankel2(' },
  // Special functions - Airy functions
  { label: 'airyai', type: 'method', detail: 'Airy Ai(z)', apply: 'airyai(' },
  { label: 'airybi', type: 'method', detail: 'Airy Bi(z)', apply: 'airybi(' },
  { label: 'airyaiprime', type: 'method', detail: "Airy Ai'(z)", apply: 'airyaiprime(' },
  { label: 'airybiprime', type: 'method', detail: "Airy Bi'(z)", apply: 'airybiprime(' },
  // Special functions - Orthogonal polynomials
  { label: 'legendre', type: 'method', detail: 'Legendre P_n(x)', apply: 'legendre(' },
  { label: 'assoc_legendre', type: 'method', detail: 'Associated Legendre P_n^m(x)', apply: 'assoc_legendre(' },
  { label: 'hermite', type: 'method', detail: 'Hermite H_n(x)', apply: 'hermite(' },
  { label: 'hermite_prob', type: 'method', detail: 'probabilist Hermite He_n(x)', apply: 'hermite_prob(' },
  { label: 'laguerre', type: 'method', detail: 'Laguerre L_n(x)', apply: 'laguerre(' },
  { label: 'assoc_laguerre', type: 'method', detail: 'Associated Laguerre L_n^α(x)', apply: 'assoc_laguerre(' },
  { label: 'chebyshevt', type: 'method', detail: 'Chebyshev T_n(x)', apply: 'chebyshevt(' },
  { label: 'chebyshevu', type: 'method', detail: 'Chebyshev U_n(x)', apply: 'chebyshevu(' },
  { label: 'gegenbauer', type: 'method', detail: 'Gegenbauer C_n^α(x)', apply: 'gegenbauer(' },
  { label: 'jacobi', type: 'method', detail: 'Jacobi P_n^(α,β)(x)', apply: 'jacobi(' },
  // Special functions - Spherical harmonics
  { label: 'Ylm', type: 'method', detail: 'spherical harmonic Y_l^m(θ,φ)', apply: 'Ylm(' },
  { label: 'Ylm_real', type: 'method', detail: 'real spherical harmonic', apply: 'Ylm_real(' },
  // Special functions - Hypergeometric
  { label: 'hyper2f1', type: 'method', detail: 'hypergeometric ₂F₁', apply: 'hyper2f1(' },
  { label: 'hyper1f1', type: 'method', detail: 'confluent ₁F₁', apply: 'hyper1f1(' },
  { label: 'hyper0f1', type: 'method', detail: 'confluent ₀F₁', apply: 'hyper0f1(' },
  { label: 'hyperpfq', type: 'method', detail: 'generalized pFq', apply: 'hyperpfq(' },
  // Special functions - Elliptic integrals
  { label: 'elliptic_k', type: 'method', detail: 'complete elliptic K(m)', apply: 'elliptic_k(' },
  { label: 'elliptic_e', type: 'method', detail: 'complete elliptic E(m)', apply: 'elliptic_e(' },
  { label: 'elliptic_pi', type: 'method', detail: 'complete elliptic Π(n,m)', apply: 'elliptic_pi(' },
  // Special functions - Other
  { label: 'zeta', type: 'method', detail: 'Riemann zeta ζ(s)', apply: 'zeta(' },
  { label: 'polylog', type: 'method', detail: 'polylogarithm Li_s(z)', apply: 'polylog(' },
  { label: 'dirac', type: 'method', detail: 'Dirac delta δ(x)', apply: 'dirac(' },
  { label: 'heaviside', type: 'method', detail: 'Heaviside θ(x)', apply: 'heaviside(' },
  { label: 'kronecker', type: 'method', detail: 'Kronecker δ_ij', apply: 'kronecker(' },
  { label: 'levi_civita', type: 'method', detail: 'Levi-Civita ε_ijk', apply: 'levi_civita(' },
  // Function composition
  { label: 'function', type: 'method', detail: 'create symbolic function', apply: 'function(' },
  { label: 'get_function', type: 'method', detail: 'retrieve saved function', apply: 'get_function(' },
  { label: 'list_functions', type: 'method', detail: 'list all saved functions', apply: 'list_functions()' },
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
  htmlOutput: { type: String, default: '' },
  workspaceId: { type: [String, Number], default: null },
  kernelId: { type: String, default: 'default' }
})

const emit = defineEmits(['update', 'run', 'delete', 'blur', 'create-below', 'reorder', 'interrupt'])

const editorContainer = ref(null)
const isFocused = ref(false)
const isDragOver = ref(false)
const isDragging = ref(false)
let editorView = null

// Log output collapse & resize
const logCollapsed = ref(false)
const logHeight = ref(200)
let logResizing = false
let logStartY = 0
let logStartHeight = 0

let logResizeTarget = null

function startLogResize(e) {
  e.preventDefault()
  logResizing = true
  logStartY = e.clientY
  logStartHeight = logHeight.value
  logResizeTarget = e.target
  // Capture pointer to prevent content from stealing events
  logResizeTarget.setPointerCapture(e.pointerId)
  logResizeTarget.addEventListener('pointermove', doLogResize)
  logResizeTarget.addEventListener('pointerup', stopLogResize)
  logResizeTarget.addEventListener('pointercancel', stopLogResize)
  document.body.style.userSelect = 'none'
  document.body.style.cursor = 'row-resize'
}

function doLogResize(e) {
  if (!logResizing) return
  const delta = e.clientY - logStartY
  logHeight.value = Math.max(60, logStartHeight + delta)
}

function stopLogResize(e) {
  if (!logResizing) return
  logResizing = false
  if (logResizeTarget) {
    if (e && e.pointerId !== undefined) {
      logResizeTarget.releasePointerCapture(e.pointerId)
    }
    logResizeTarget.removeEventListener('pointermove', doLogResize)
    logResizeTarget.removeEventListener('pointerup', stopLogResize)
    logResizeTarget.removeEventListener('pointercancel', stopLogResize)
    logResizeTarget = null
  }
  document.body.style.userSelect = ''
  document.body.style.cursor = ''
}

// Drag-and-drop handlers
function handleDragStart(event) {
  event.dataTransfer.effectAllowed = 'move'
  event.dataTransfer.setData('text/plain', props.index.toString())
  isDragging.value = true
}

function handleDragEnd(event) {
  isDragging.value = false
  isDragOver.value = false
}

function handleDragOver(event) {
  event.preventDefault()
  event.dataTransfer.dropEffect = 'move'
  isDragOver.value = true
}

function handleDragLeave(event) {
  // Only clear if we're actually leaving the cell
  if (event.target.classList.contains('cell')) {
    isDragOver.value = false
  }
}

function handleDrop(event) {
  event.preventDefault()
  isDragOver.value = false

  const fromIndex = parseInt(event.dataTransfer.getData('text/plain'))
  const toIndex = props.index

  if (fromIndex !== toIndex) {
    emit('reorder', { fromIndex, toIndex })
  }
}

const languageExtensions = {
  python: () => StreamLanguage.define(pythonLegacy),
  cpp: () => cpp(),
  bash: () => StreamLanguage.define(shell),
  markdown: () => []  // No syntax highlighting for markdown (uses preview)
}

function getLanguageExtension(lang) {
  const factory = languageExtensions[lang]
  if (!factory) return StreamLanguage.define(pythonLegacy)  // Default to Python
  return factory()
}

// ================== Dynamic Autocomplete Support ==================

// Completion cache to avoid redundant API calls
const completionCache = new Map()
const CACHE_TTL = 5000  // 5 seconds

function getCacheKey(code, pos) {
  // Use a window around cursor position for cache key
  const start = Math.max(0, pos - 100)
  const window = code.substring(start, pos)
  return `${window}_${pos}`
}

// Fetch dynamic completions from backend
async function fetchDynamicCompletions(code, cursorPos) {
  const cacheKey = getCacheKey(code, cursorPos)
  const cached = completionCache.get(cacheKey)

  // Return cached result if still valid
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data
  }

  try {
    const response = await axios.post('/api/complete', {
      code,
      cursor_pos: cursorPos,
      workspace_id: props.workspaceId,
      kernel_id: props.kernelId
    }, {
      timeout: 1500  // 1.5 second timeout
    })

    const completions = response.data.completions || []

    // Cache the result
    completionCache.set(cacheKey, {
      data: completions,
      timestamp: Date.now()
    })

    return completions
  } catch (error) {
    console.warn('Dynamic completion failed:', error.message)
    return []
  }
}

// Convert jedi completion to CodeMirror format
function convertJediCompletion(jediComp) {
  const docPreview = jediComp.docstring ? jediComp.docstring.substring(0, 50) : null

  // Build info text (shown in tooltip)
  let info = ''
  if (jediComp.signature) {
    info = jediComp.signature
    if (jediComp.docstring) {
      info += '\n\n' + jediComp.docstring
    }
  } else if (jediComp.docstring) {
    info = jediComp.docstring
  }

  return {
    label: jediComp.name,
    type: jediComp.type,  // CodeMirror supports: function, variable, type, keyword, etc.
    detail: jediComp.signature || docPreview || jediComp.type,
    info: info || undefined,
    apply: jediComp.name  // Explicitly set what gets inserted
  }
}

// Smart Python completion that detects context and fetches dynamic completions
async function pythonCompleter(context) {
  const doc = context.state.doc.toString()
  const pos = context.pos

  // Get the current line text up to the cursor
  const line = context.state.doc.lineAt(pos)
  const lineTextToCursor = line.text.slice(0, pos - line.from)

  // Detect import context: "from X import " or "from X import Y"
  const importMatch = lineTextToCursor.match(/^(?:from\s+([\w.]+)\s+import\s+)([\w]*)$/)
  if (importMatch && props.language === 'python') {
    const partialName = importMatch[2]  // what user has typed after "import "
    const completionFrom = pos - partialName.length

    try {
      const dynamicResults = await fetchDynamicCompletions(doc, pos)
      if (dynamicResults.length > 0) {
        return {
          from: completionFrom,
          options: dynamicResults.map(convertJediCompletion),
          validFor: /^[\w]*$/
        }
      }
    } catch (error) {
      console.warn('[Autocomplete] Import completion error:', error)
    }
    // Fallback to static completions filtered by partial name
    const completions = getCompletions('python')
    return {
      from: completionFrom,
      options: completions.filter(c =>
        c.label.toLowerCase().startsWith(partialName.toLowerCase())
      ),
      validFor: /^[\w]*$/
    }
  }

  // Detect bare import context: "import cm." or "import cm.sym"
  const bareImportMatch = lineTextToCursor.match(/^import\s+([\w.]*\.)(\w*)$/)
  if (bareImportMatch && props.language === 'python') {
    const partialName = bareImportMatch[2]
    const completionFrom = pos - partialName.length

    try {
      const dynamicResults = await fetchDynamicCompletions(doc, pos)
      if (dynamicResults.length > 0) {
        return {
          from: completionFrom,
          options: dynamicResults.map(convertJediCompletion),
          validFor: /^[\w]*$/
        }
      }
    } catch (error) {
      console.warn('[Autocomplete] Import completion error:', error)
    }
  }

  // Standard word/dot matching
  const word = context.matchBefore(/[\w.]*/)
  if (!word || (word.from === word.to && !context.explicit)) return null

  const text = word.text

  // Detect if we're typing after a dot (instance/module attribute completion)
  const needsDynamic = text.includes('.') && /\w+\.\w*$/.test(text)

  let options = []
  let completionFrom = word.from

  // Try dynamic completion for dot access (obj. or module.)
  if (needsDynamic && props.language === 'python') {
    try {
      const dynamicResults = await fetchDynamicCompletions(doc, pos)

      if (dynamicResults.length > 0) {
        const dynamicOptions = dynamicResults.map(convertJediCompletion)
        options = dynamicOptions

        // Set 'from' to be after the last dot
        const dotIndex = text.lastIndexOf('.')
        if (dotIndex !== -1) {
          completionFrom = word.from + dotIndex + 1
        }
      }
    } catch (error) {
      console.warn('[Autocomplete] Dynamic completion error:', error)
    }
  }

  // Fallback to static completions if no dynamic results
  if (options.length === 0) {
    // Check if we're typing after a dot for static class methods
    if (text.includes('.')) {
      const parts = text.split('.')
      const varName = parts.slice(0, -1).join('.')
      const methodPrefix = parts[parts.length - 1]

      // Look backwards to find if this variable was assigned from Math()
      const beforeCursor = doc.slice(0, pos - text.length)
      const mathPattern = new RegExp(`${varName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*=\\s*Math\\(`, 'm')

      if (mathPattern.test(beforeCursor)) {
        const filtered = mathMethodCompletions.filter(c =>
          c.label.toLowerCase().startsWith(methodPrefix.toLowerCase())
        )
        if (filtered.length > 0) {
          return {
            from: word.from + varName.length + 1,
            options: filtered
          }
        }
      }
    }

    // Default: show general static Python completions
    const completions = getCompletions('python')
    options = completions.filter(c =>
      c.label.toLowerCase().startsWith(text.toLowerCase())
    )
  }

  return {
    from: completionFrom,
    options,
    validFor: /^[\w.]*$/
  }
}

function createEditor() {
  if (!editorContainer.value) return

  // Create language-specific autocomplete
  const languageCompletions = props.language === 'python'
    ? pythonCompleter
    : completeFromList(getCompletions(props.language))

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
    // Trigger autocomplete on space/dot in import contexts (CM6 skips whitespace by default)
    props.language === 'python' ? EditorView.updateListener.of(update => {
      if (!update.docChanged) return
      update.transactions.forEach(tr => {
        if (!tr.isUserEvent('input')) return
        tr.changes.iterChanges((_fromA, _toA, fromB, _toB, inserted) => {
          const text = inserted.toString()
          if (text !== ' ' && text !== '.') return
          const pos = fromB + text.length
          const line = update.state.doc.lineAt(pos)
          const lineText = line.text.slice(0, pos - line.from)
          // Trigger on space after "from X import" or dot in "import X." / "from X."
          if (text === ' ' && /(?:from\s+[\w.]+\s+import|import)\s$/.test(lineText)) {
            setTimeout(() => startCompletion(update.view), 0)
          } else if (text === '.' && /(?:from\s+[\w.]+\.|import\s+[\w.]+\.)$/.test(lineText)) {
            setTimeout(() => startCompletion(update.view), 0)
          }
        })
      })
    }) : [],
    keymap.of([
      {
        key: 'Mod-Enter',
        run: () => {
          emit('run')
          return true
        },
        preventDefault: true
      },
      {
        key: 'Alt-Enter',
        run: () => {
          emit('create-below')
          return true
        },
        preventDefault: true
      },
      // Tab accepts autocomplete, or indents if no completion is active
      {
        key: 'Tab',
        run: (view) => {
          // First try to accept completion
          if (acceptCompletion(view)) {
            return true
          }
          // If no completion to accept, indent instead
          return indentMore(view)
        },
      },
      // Ctrl+Space to manually trigger autocomplete
      {
        key: 'Ctrl-Space',
        run: startCompletion,
      },
      // Enter just inserts a newline, never accepts completion
      {
        key: 'Enter',
        run: insertNewlineAndIndent,
      },
      ...defaultKeymap,
      ...historyKeymap,
    ]),

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


onMounted(() => {
  createEditor()
})

onUnmounted(() => {
  destroyEditor()
  stopLogResize()
})
</script>

<style scoped>
.cell {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 0.5rem;
  overflow: hidden;
  transition: border-color 0.2s, transform 0.2s, opacity 0.2s;
}

.cell.focused {
  border-color: var(--accent);
}

.cell.dragging {
  opacity: 0.5;
}

.cell.drag-over {
  border-color: var(--success);
  transform: scale(1.02);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
}

.cell-toolbar {
  background: var(--bg-tertiary);
  padding: 0.25rem 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.drag-handle {
  color: var(--text-secondary);
  font-size: 0.9rem;
  cursor: grab;
  user-select: none;
  opacity: 0.5;
  transition: opacity 0.2s;
  padding: 0 0.25rem;
}

.drag-handle:hover {
  opacity: 1;
}

.drag-handle:active {
  cursor: grabbing;
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

/* Cell toolbar buttons — override btn-icon size */
.run-btn, .delete-btn, .create-below-btn, .interrupt-btn {
  width: 22px;
  height: 22px;
}

.create-below-btn {
  border: 1px solid var(--border);
}

.create-below-btn:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-color: var(--text-secondary);
}

.run-btn {
  color: var(--success);
}

.run-btn:hover {
  background: rgba(68, 255, 136, 0.1);
}

.interrupt-btn {
  color: #f59e0b;
  animation: pulse-interrupt 1.5s ease-in-out infinite;
}

.interrupt-btn:hover {
  background: rgba(245, 158, 11, 0.15);
  animation: none;
}

@keyframes pulse-interrupt {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.spinner {
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: #f59e0b;
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
  background: var(--bg-tertiary);
}

.cell-output-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.25rem 0.5rem;
  cursor: pointer;
  user-select: none;
  background: #262637;
  border-bottom: 1px solid var(--border);
}

.cell-output-header:hover {
  background: #2d2d42;
}

.cell-output-label {
  font-size: 0.7rem;
  color: #6c7086;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.cell-output-chevron {
  font-size: 0.6rem;
  color: #6c7086;
  display: inline-block;
  transition: transform 0.2s ease;
}

.cell-output-chevron.collapsed {
  transform: rotate(-90deg);
}

.cell-output-body {
  padding: 0.5rem;
  overflow-y: auto;
}

.cell-output-resize {
  height: 6px;
  background: var(--border, #313244);
  cursor: row-resize;
  transition: background 0.2s;
}

.cell-output-resize:hover {
  background: var(--accent, #89b4fa);
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
