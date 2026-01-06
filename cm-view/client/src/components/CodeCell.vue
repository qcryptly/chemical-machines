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
      v-if="htmlOutput"
      :html-content="htmlOutput"
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
import { cpp } from '@codemirror/lang-cpp'
import { StreamLanguage } from '@codemirror/language'
import { shell } from '@codemirror/legacy-modes/mode/shell'
import { python as pythonLegacy } from '@codemirror/legacy-modes/mode/python'
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
  // CM library - views module
  { label: 'cm.views', type: 'module', detail: 'CM visualization module' },
  { label: 'from cm.views import', type: 'text', detail: 'import CM views' },
  { label: 'html', type: 'function', detail: 'cm.views: render HTML' },
  { label: 'text', type: 'function', detail: 'cm.views: render text' },
  { label: 'log', type: 'function', detail: 'cm.views: log message' },
  { label: 'table', type: 'function', detail: 'cm.views: render table' },
  { label: 'image', type: 'function', detail: 'cm.views: display image' },
  { label: 'savefig', type: 'function', detail: 'cm.views: save matplotlib figure' },
  { label: 'dataframe', type: 'function', detail: 'cm.views: render pandas DataFrame' },
  { label: 'scatter_3d', type: 'function', detail: 'cm.views: 3D scatter plot' },
  { label: 'line_3d', type: 'function', detail: 'cm.views: 3D line plot' },
  { label: 'lines_3d', type: 'function', detail: 'cm.views: multiple 3D lines' },
  { label: 'surface', type: 'function', detail: 'cm.views: 3D surface plot' },
  { label: 'vector_field', type: 'function', detail: 'cm.views: 3D vector field' },
  { label: 'molecule', type: 'function', detail: 'cm.views: molecule visualization' },
  { label: 'molecule_xyz', type: 'function', detail: 'cm.views: molecule from XYZ' },
  { label: 'crystal', type: 'function', detail: 'cm.views: crystal structure' },
  { label: 'webgl', type: 'function', detail: 'cm.views: raw WebGL HTML' },
  { label: 'webgl_threejs', type: 'function', detail: 'cm.views: Three.js helper' },
  { label: 'clear', type: 'function', detail: 'cm.views: clear cell output' },
  { label: 'clear_all', type: 'function', detail: 'cm.views: clear all output' },
  // CM library - symbols module
  { label: 'cm.symbols', type: 'module', detail: 'CM LaTeX math module' },
  { label: 'from cm.symbols import', type: 'text', detail: 'import CM symbols' },
  { label: 'latex', type: 'function', detail: 'cm.symbols: render LaTeX' },
  { label: 'equation', type: 'function', detail: 'cm.symbols: numbered equation' },
  { label: 'align', type: 'function', detail: 'cm.symbols: aligned equations' },
  { label: 'matrix', type: 'function', detail: 'cm.symbols: render matrix' },
  { label: 'Math', type: 'class', detail: 'cm.symbols: symbolic math class' },
  { label: 'set_notation', type: 'function', detail: 'cm.symbols: set notation style' },
  { label: 'set_line_height', type: 'function', detail: 'cm.symbols: set line spacing' },
  // CM symbols - hyperparameter types
  { label: 'Scalar', type: 'class', detail: 'cm.symbols: scalar hyperparameter type' },
  { label: 'ExprType', type: 'class', detail: 'cm.symbols: expression hyperparameter type' },
  { label: 'BoundsType', type: 'class', detail: 'cm.symbols: bounds hyperparameter type' },
  // CM symbols - expression classes
  { label: 'Expr', type: 'class', detail: 'cm.symbols: base expression class' },
  { label: 'Var', type: 'class', detail: 'cm.symbols: variable expression' },
  { label: 'Const', type: 'class', detail: 'cm.symbols: constant expression' },
  { label: 'Sum', type: 'class', detail: 'cm.symbols: summation expression' },
  { label: 'Product', type: 'class', detail: 'cm.symbols: product expression' },
  // CM symbols - function composition
  { label: 'SymbolicFunction', type: 'class', detail: 'cm.symbols: reusable function' },
  { label: 'BoundFunction', type: 'class', detail: 'cm.symbols: function with bound params' },
  { label: 'ComputeGraph', type: 'class', detail: 'cm.symbols: lazy evaluation graph' },
  { label: 'TorchFunction', type: 'class', detail: 'cm.symbols: PyTorch compiled function' },
  { label: 'TorchGradFunction', type: 'class', detail: 'cm.symbols: PyTorch gradient function' },
  // CM symbols - special function classes
  { label: 'SpecialFunction', type: 'class', detail: 'cm.symbols: base special function' },
  { label: 'Gamma', type: 'class', detail: 'cm.symbols: gamma function Γ(z)' },
  { label: 'Beta', type: 'class', detail: 'cm.symbols: beta function B(a,b)' },
  { label: 'Factorial', type: 'class', detail: 'cm.symbols: factorial n!' },
  { label: 'Binomial', type: 'class', detail: 'cm.symbols: binomial C(n,k)' },
  { label: 'Erf', type: 'class', detail: 'cm.symbols: error function' },
  { label: 'BesselJ', type: 'class', detail: 'cm.symbols: Bessel J_ν(z)' },
  { label: 'BesselY', type: 'class', detail: 'cm.symbols: Bessel Y_ν(z)' },
  { label: 'Legendre', type: 'class', detail: 'cm.symbols: Legendre P_n(x)' },
  { label: 'AssocLegendre', type: 'class', detail: 'cm.symbols: Associated Legendre' },
  { label: 'Hermite', type: 'class', detail: 'cm.symbols: Hermite H_n(x)' },
  { label: 'Laguerre', type: 'class', detail: 'cm.symbols: Laguerre L_n(x)' },
  { label: 'AssocLaguerre', type: 'class', detail: 'cm.symbols: Associated Laguerre' },
  { label: 'SphericalHarmonic', type: 'class', detail: 'cm.symbols: Y_l^m(θ,φ)' },
  { label: 'AiryAi', type: 'class', detail: 'cm.symbols: Airy Ai(z)' },
  { label: 'AiryBi', type: 'class', detail: 'cm.symbols: Airy Bi(z)' },
  // CM library - qm module
  { label: 'cm.qm', type: 'module', detail: 'CM quantum mechanics module' },
  { label: 'from cm.qm import', type: 'text', detail: 'import CM qm' },
  { label: 'SpinOrbital', type: 'class', detail: 'cm.qm: spin-orbital basis element' },
  { label: 'SlaterDeterminant', type: 'class', detail: 'cm.qm: Slater determinant' },
  { label: 'DiracSpinor', type: 'class', detail: 'cm.qm: relativistic spinor' },
  { label: 'basis_sh', type: 'function', detail: 'cm.qm: create spin-orbital basis' },
  { label: 'basis_sh_element', type: 'function', detail: 'cm.qm: create single spin-orbital' },
  { label: 'slater', type: 'function', detail: 'cm.qm: create Slater determinant' },
  { label: 'hamiltonian', type: 'function', detail: 'cm.qm: create Hamiltonian operator' },
  { label: 'dirac_spinor', type: 'function', detail: 'cm.qm: create Dirac spinor' },
  { label: 'dirac_slater', type: 'function', detail: 'cm.qm: create Dirac Slater det' },
  // Math static methods - core
  { label: 'Math.var', type: 'function', detail: 'create symbolic variable' },
  { label: 'Math.const', type: 'function', detail: 'create symbolic constant' },
  { label: 'Math.sum', type: 'function', detail: 'summation Σ' },
  { label: 'Math.prod', type: 'function', detail: 'product Π' },
  { label: 'Math.sin', type: 'function', detail: 'sine' },
  { label: 'Math.cos', type: 'function', detail: 'cosine' },
  { label: 'Math.tan', type: 'function', detail: 'tangent' },
  { label: 'Math.exp', type: 'function', detail: 'exponential' },
  { label: 'Math.log', type: 'function', detail: 'natural logarithm' },
  { label: 'Math.sqrt', type: 'function', detail: 'square root' },
  { label: 'Math.abs', type: 'function', detail: 'absolute value' },
  // Math static methods - special functions (gamma family)
  { label: 'Math.gamma', type: 'function', detail: 'gamma Γ(z)' },
  { label: 'Math.loggamma', type: 'function', detail: 'log gamma ln Γ(z)' },
  { label: 'Math.digamma', type: 'function', detail: 'digamma ψ(z)' },
  { label: 'Math.beta', type: 'function', detail: 'beta B(a,b)' },
  { label: 'Math.factorial', type: 'function', detail: 'factorial n!' },
  { label: 'Math.factorial2', type: 'function', detail: 'double factorial n!!' },
  { label: 'Math.binomial', type: 'function', detail: 'binomial C(n,k)' },
  // Math static methods - special functions (error)
  { label: 'Math.erf', type: 'function', detail: 'error function' },
  { label: 'Math.erfc', type: 'function', detail: 'complementary error' },
  { label: 'Math.erfi', type: 'function', detail: 'imaginary error' },
  // Math static methods - special functions (Bessel)
  { label: 'Math.besselj', type: 'function', detail: 'Bessel J_ν(z)' },
  { label: 'Math.bessely', type: 'function', detail: 'Bessel Y_ν(z)' },
  { label: 'Math.besseli', type: 'function', detail: 'modified Bessel I_ν(z)' },
  { label: 'Math.besselk', type: 'function', detail: 'modified Bessel K_ν(z)' },
  { label: 'Math.jn', type: 'function', detail: 'spherical Bessel j_n(z)' },
  { label: 'Math.yn', type: 'function', detail: 'spherical Bessel y_n(z)' },
  { label: 'Math.hankel1', type: 'function', detail: 'Hankel H_ν^(1)(z)' },
  { label: 'Math.hankel2', type: 'function', detail: 'Hankel H_ν^(2)(z)' },
  // Math static methods - special functions (Airy)
  { label: 'Math.airyai', type: 'function', detail: 'Airy Ai(z)' },
  { label: 'Math.airybi', type: 'function', detail: 'Airy Bi(z)' },
  { label: 'Math.airyaiprime', type: 'function', detail: "Airy Ai'(z)" },
  { label: 'Math.airybiprime', type: 'function', detail: "Airy Bi'(z)" },
  // Math static methods - special functions (polynomials)
  { label: 'Math.legendre', type: 'function', detail: 'Legendre P_n(x)' },
  { label: 'Math.assoc_legendre', type: 'function', detail: 'Associated Legendre P_n^m(x)' },
  { label: 'Math.hermite', type: 'function', detail: 'Hermite H_n(x)' },
  { label: 'Math.hermite_prob', type: 'function', detail: 'probabilist Hermite He_n(x)' },
  { label: 'Math.laguerre', type: 'function', detail: 'Laguerre L_n(x)' },
  { label: 'Math.assoc_laguerre', type: 'function', detail: 'Associated Laguerre L_n^α(x)' },
  { label: 'Math.chebyshevt', type: 'function', detail: 'Chebyshev T_n(x)' },
  { label: 'Math.chebyshevu', type: 'function', detail: 'Chebyshev U_n(x)' },
  { label: 'Math.gegenbauer', type: 'function', detail: 'Gegenbauer C_n^α(x)' },
  { label: 'Math.jacobi', type: 'function', detail: 'Jacobi P_n^(α,β)(x)' },
  // Math static methods - special functions (spherical)
  { label: 'Math.Ylm', type: 'function', detail: 'spherical harmonic Y_l^m(θ,φ)' },
  { label: 'Math.Ylm_real', type: 'function', detail: 'real spherical harmonic' },
  // Math static methods - special functions (hypergeometric)
  { label: 'Math.hyper2f1', type: 'function', detail: 'hypergeometric ₂F₁' },
  { label: 'Math.hyper1f1', type: 'function', detail: 'confluent ₁F₁' },
  { label: 'Math.hyper0f1', type: 'function', detail: 'confluent ₀F₁' },
  { label: 'Math.hyperpfq', type: 'function', detail: 'generalized pFq' },
  // Math static methods - special functions (elliptic)
  { label: 'Math.elliptic_k', type: 'function', detail: 'complete elliptic K(m)' },
  { label: 'Math.elliptic_e', type: 'function', detail: 'complete elliptic E(m)' },
  { label: 'Math.elliptic_pi', type: 'function', detail: 'complete elliptic Π(n,m)' },
  // Math static methods - special functions (other)
  { label: 'Math.zeta', type: 'function', detail: 'Riemann zeta ζ(s)' },
  { label: 'Math.polylog', type: 'function', detail: 'polylogarithm Li_s(z)' },
  { label: 'Math.dirac', type: 'function', detail: 'Dirac delta δ(x)' },
  { label: 'Math.heaviside', type: 'function', detail: 'Heaviside θ(x)' },
  { label: 'Math.kronecker', type: 'function', detail: 'Kronecker δ_ij' },
  { label: 'Math.levi_civita', type: 'function', detail: 'Levi-Civita ε_ijk' },
  // Math static methods - function composition
  { label: 'Math.function', type: 'function', detail: 'create symbolic function' },
  { label: 'Math.get_function', type: 'function', detail: 'retrieve saved function' },
  { label: 'Math.list_functions', type: 'function', detail: 'list saved functions' },
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
  htmlOutput: { type: String, default: '' }
})

const emit = defineEmits(['update', 'run', 'delete', 'blur'])

const editorContainer = ref(null)
const isFocused = ref(false)
let editorView = null

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

// Smart Python completion that detects context
function pythonCompleter(context) {
  const word = context.matchBefore(/[\w.]*/)
  if (!word || (word.from === word.to && !context.explicit)) return null

  const text = word.text
  const doc = context.state.doc.toString()
  const pos = context.pos

  // Check if we're typing after a dot (method completion)
  if (text.includes('.')) {
    const parts = text.split('.')
    const varName = parts.slice(0, -1).join('.')
    const methodPrefix = parts[parts.length - 1]

    // Look backwards in the document to find if this variable was assigned from Math()
    const beforeCursor = doc.slice(0, pos - text.length)
    const mathPattern = new RegExp(`${varName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*=\\s*Math\\(`, 'm')

    if (mathPattern.test(beforeCursor)) {
      // This is a Math instance - show Math methods
      const filtered = mathMethodCompletions.filter(c =>
        c.label.toLowerCase().startsWith(methodPrefix.toLowerCase())
      )
      if (filtered.length > 0) {
        return {
          from: word.from + varName.length + 1,  // After the dot
          options: filtered
        }
      }
    }
  }

  // Default: show general Python completions
  const completions = getCompletions('python')
  const filtered = completions.filter(c =>
    c.label.toLowerCase().startsWith(text.toLowerCase())
  )

  return {
    from: word.from,
    options: filtered
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
