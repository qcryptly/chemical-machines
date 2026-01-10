<template>
  <div class="benchmark-comparison">
    <div class="comparison-header">
      <h3>Computed vs Benchmark Comparison</h3>
      <p v-if="!comparison">
        Compare your computed molecular properties against experimental or reference data.
      </p>
    </div>

    <div v-if="!comparison && !loading" class="comparison-form">
      <div class="form-group">
        <label>Benchmark Molecule Identifier</label>
        <input
          type="text"
          v-model="identifier"
          placeholder="CAS number, CID, or InChIKey"
          class="form-input"
        />
      </div>

      <div class="form-group">
        <label>Computed Properties (JSON)</label>
        <textarea
          v-model="computedJson"
          placeholder='{"homo_energy": -0.25, "lumo_energy": 0.05, "dipole_moment": 1.85}'
          class="form-textarea"
          rows="5"
        ></textarea>
        <div class="json-help">
          Enter properties as JSON. Property names should match benchmark format:
          homo_energy, lumo_energy, dipole_moment, total_energy, etc.
        </div>
      </div>

      <button
        @click="compare"
        :disabled="!identifier || !computedJson || loading"
        class="compare-button"
      >
        {{ loading ? 'Comparing...' : 'Compare' }}
      </button>

      <div v-if="error" class="error-message">
        {{ error }}
      </div>
    </div>

    <div v-if="loading" class="loading-state">
      <div class="spinner"></div>
      <span>Comparing properties...</span>
    </div>

    <div v-if="comparison" class="comparison-results">
      <div class="results-header">
        <div class="molecule-info">
          <span class="identifier">{{ comparison.identifier }}</span>
          <span class="sources">Sources: {{ comparison.benchmark_sources?.join(', ') || 'N/A' }}</span>
        </div>
        <button @click="reset" class="reset-button">New Comparison</button>
      </div>

      <div class="summary">
        <div class="summary-item">
          <span class="summary-label">Properties Compared</span>
          <span class="summary-value">{{ comparison.summary?.properties_compared || 0 }}</span>
        </div>
        <div class="summary-item">
          <span class="summary-label">Missing Benchmark Data</span>
          <span class="summary-value">{{ comparison.summary?.properties_missing || 0 }}</span>
        </div>
        <div class="summary-item">
          <span class="summary-label">Avg. Percent Difference</span>
          <span class="summary-value" :class="getDifferenceClass(comparison.summary?.avg_percent_diff)">
            {{ formatPercent(comparison.summary?.avg_percent_diff) }}
          </span>
        </div>
      </div>

      <table class="comparison-table">
        <thead>
          <tr>
            <th>Property</th>
            <th>Computed</th>
            <th>Benchmark</th>
            <th>Difference</th>
            <th>% Diff</th>
            <th>Method</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="comp in comparison.comparisons"
            :key="comp.property"
            :class="{ 'missing': comp.benchmark === null }"
          >
            <td>{{ formatPropertyName(comp.property) }}</td>
            <td class="value">{{ formatValue(comp.computed) }}</td>
            <td class="value">
              {{ comp.benchmark !== null ? formatValue(comp.benchmark) : '-' }}
            </td>
            <td class="value" :class="getDifferenceClass(comp.percent_difference)">
              {{ comp.difference !== undefined ? formatValue(comp.difference) : '-' }}
            </td>
            <td class="value" :class="getDifferenceClass(comp.percent_difference)">
              {{ comp.percent_difference !== undefined ? formatPercent(comp.percent_difference) : '-' }}
            </td>
            <td>{{ comp.benchmark_method || comp.note || '-' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const props = defineProps({
  initialIdentifier: {
    type: String,
    default: ''
  },
  initialComputed: {
    type: Object,
    default: null
  }
});

const identifier = ref(props.initialIdentifier);
const computedJson = ref(props.initialComputed ? JSON.stringify(props.initialComputed, null, 2) : '');
const comparison = ref(null);
const loading = ref(false);
const error = ref('');
const currentJobId = ref(null);
let ws = null;

async function compare() {
  if (!identifier.value || !computedJson.value) return;

  let computed;
  try {
    computed = JSON.parse(computedJson.value);
  } catch (e) {
    error.value = 'Invalid JSON: ' + e.message;
    return;
  }

  loading.value = true;
  error.value = '';
  comparison.value = null;

  try {
    const response = await fetch('/api/benchmark/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        computed,
        identifier: identifier.value
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Comparison failed');
    }

    if (data.status === 'comparing' && data.jobId) {
      currentJobId.value = data.jobId;
      subscribeToJob(data.jobId);
    }
  } catch (err) {
    error.value = err.message;
    loading.value = false;
  }
}

function subscribeToJob(jobId) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'subscribe', jobId }));
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.jobId === jobId) {
      if (data.type === 'job_result' || data.stream === 'result') {
        comparison.value = data.result || data.data;
        loading.value = false;
        currentJobId.value = null;
      } else if (data.type === 'error' || data.stream === 'error') {
        error.value = data.error || data.data || 'Comparison failed';
        loading.value = false;
        currentJobId.value = null;
      }
    }
  };

  ws.onerror = () => {
    error.value = 'WebSocket connection error';
    loading.value = false;
  };
}

function reset() {
  comparison.value = null;
  error.value = '';
}

function formatPropertyName(name) {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function formatValue(value) {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') {
    if (Math.abs(value) < 0.001 || Math.abs(value) > 10000) {
      return value.toExponential(4);
    }
    return value.toFixed(6).replace(/\.?0+$/, '');
  }
  return value;
}

function formatPercent(value) {
  if (value === null || value === undefined) return '-';
  return value.toFixed(2) + '%';
}

function getDifferenceClass(percentDiff) {
  if (percentDiff === null || percentDiff === undefined) return '';
  const abs = Math.abs(percentDiff);
  if (abs < 1) return 'diff-excellent';
  if (abs < 5) return 'diff-good';
  if (abs < 10) return 'diff-moderate';
  return 'diff-poor';
}

onUnmounted(() => {
  if (ws) {
    ws.close();
  }
});
</script>

<style scoped>
.benchmark-comparison {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.comparison-header {
  padding: 16px;
  border-bottom: 1px solid var(--border-color, #3c3c3c);
}

.comparison-header h3 {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary, #e0e0e0);
}

.comparison-header p {
  margin: 0;
  font-size: 13px;
  color: var(--text-secondary, #a0a0a0);
}

.comparison-form {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-group label {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary, #e0e0e0);
}

.form-input,
.form-textarea {
  padding: 10px 12px;
  background: var(--bg-primary, #252526);
  border: 1px solid var(--border-color, #3c3c3c);
  border-radius: 4px;
  color: var(--text-primary, #e0e0e0);
  font-size: 13px;
  font-family: inherit;
}

.form-textarea {
  font-family: monospace;
  resize: vertical;
}

.form-input:focus,
.form-textarea:focus {
  outline: none;
  border-color: var(--accent-color, #007acc);
}

.json-help {
  font-size: 11px;
  color: var(--text-secondary, #a0a0a0);
}

.compare-button {
  align-self: flex-start;
  padding: 10px 20px;
  background: var(--accent-color, #007acc);
  border: none;
  border-radius: 4px;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.2s;
}

.compare-button:hover:not(:disabled) {
  background: var(--accent-hover, #005a9e);
}

.compare-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 32px;
  color: var(--text-secondary, #a0a0a0);
}

.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--border-color, #3c3c3c);
  border-top-color: var(--accent-color, #007acc);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-message {
  padding: 12px;
  background: rgba(255, 0, 0, 0.1);
  border: 1px solid rgba(255, 0, 0, 0.3);
  border-radius: 4px;
  color: #ff6b6b;
  font-size: 13px;
}

.comparison-results {
  flex: 1;
  overflow: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.molecule-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.molecule-info .identifier {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary, #e0e0e0);
}

.molecule-info .sources {
  font-size: 12px;
  color: var(--text-secondary, #a0a0a0);
}

.reset-button {
  padding: 6px 12px;
  background: var(--bg-secondary, #1e1e1e);
  border: 1px solid var(--border-color, #3c3c3c);
  border-radius: 4px;
  color: var(--text-primary, #e0e0e0);
  font-size: 12px;
  cursor: pointer;
}

.reset-button:hover {
  background: var(--bg-hover, #2a2a2a);
}

.summary {
  display: flex;
  gap: 24px;
  padding: 16px;
  background: var(--bg-secondary, #1e1e1e);
  border-radius: 8px;
}

.summary-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.summary-label {
  font-size: 11px;
  color: var(--text-secondary, #a0a0a0);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.summary-value {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary, #e0e0e0);
}

.comparison-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.comparison-table th,
.comparison-table td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border-color, #3c3c3c);
}

.comparison-table th {
  background: var(--bg-secondary, #1e1e1e);
  color: var(--text-secondary, #a0a0a0);
  font-weight: 500;
  font-size: 12px;
  position: sticky;
  top: 0;
}

.comparison-table td {
  color: var(--text-primary, #e0e0e0);
}

.comparison-table td.value {
  font-family: monospace;
}

.comparison-table tbody tr:hover {
  background: var(--bg-hover, #2a2a2a);
}

.comparison-table tbody tr.missing {
  opacity: 0.6;
}

.diff-excellent {
  color: #4caf50 !important;
}

.diff-good {
  color: #8bc34a !important;
}

.diff-moderate {
  color: #ff9800 !important;
}

.diff-poor {
  color: #f44336 !important;
}
</style>
