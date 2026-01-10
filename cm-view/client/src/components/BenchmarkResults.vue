<template>
  <div class="benchmark-results">
    <div v-if="loading" class="loading-state">
      <div class="spinner"></div>
      <span>Loading molecule data...</span>
      <span v-if="jobId" class="job-id">Job: {{ jobId }}</span>
    </div>

    <div v-else-if="!molecule" class="empty-state">
      <p>Select a molecule from the search results to view its benchmark data.</p>
    </div>

    <div v-else class="molecule-data">
      <div class="molecule-header">
        <h3>{{ molecule.name || 'Unknown Molecule' }}</h3>
        <div class="molecule-identifiers">
          <span v-if="molecule.formula" class="formula">{{ molecule.formula }}</span>
          <span v-if="molecule.cas" class="identifier">CAS: {{ molecule.cas }}</span>
          <span v-if="molecule.cid" class="identifier">CID: {{ molecule.cid }}</span>
          <span v-if="molecule.inchi_key" class="identifier inchi-key">{{ molecule.inchi_key }}</span>
        </div>
        <div v-if="molecule.sources" class="sources">
          Sources: {{ molecule.sources.join(', ') }}
        </div>
        <div v-if="molecule.cached_at" class="cached-at">
          Cached: {{ formatDate(molecule.cached_at) }}
        </div>
      </div>

      <div class="molecule-sections">
        <!-- Properties Section -->
        <div v-if="molecule.properties && molecule.properties.length > 0" class="section">
          <h4>Properties</h4>
          <table class="properties-table">
            <thead>
              <tr>
                <th>Property</th>
                <th>Value</th>
                <th>Unit</th>
                <th>Source</th>
                <th>Method</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="prop in molecule.properties" :key="`${prop.name}-${prop.source}`">
                <td>{{ formatPropertyName(prop.name) }}</td>
                <td class="value">{{ formatValue(prop.value) }}</td>
                <td>{{ prop.unit || '-' }}</td>
                <td>{{ prop.source }}</td>
                <td>{{ prop.method || '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Geometry Section -->
        <div v-if="molecule.geometry" class="section">
          <h4>
            Geometry
            <button @click="copyGeometry" class="copy-btn" title="Copy as XYZ">
              Copy XYZ
            </button>
          </h4>
          <div class="geometry-info">
            <span>{{ molecule.geometry.atoms?.length || 0 }} atoms</span>
            <span>Unit: {{ molecule.geometry.unit || 'angstrom' }}</span>
            <span v-if="molecule.geometry.source">Source: {{ molecule.geometry.source }}</span>
          </div>
          <div class="geometry-preview">
            <pre>{{ formatGeometry(molecule.geometry) }}</pre>
          </div>
        </div>

        <!-- File Paths Section -->
        <div v-if="molecule.file_paths && Object.keys(molecule.file_paths).length > 0" class="section">
          <h4>Downloaded Files</h4>
          <ul class="file-list">
            <li v-for="(path, name) in molecule.file_paths" :key="name">
              <span class="file-name">{{ name }}:</span>
              <span class="file-path">{{ path }}</span>
            </li>
          </ul>
        </div>

        <!-- SMILES/InChI Section -->
        <div v-if="molecule.smiles || molecule.inchi" class="section">
          <h4>Structure Identifiers</h4>
          <div class="structure-ids">
            <div v-if="molecule.smiles" class="structure-id">
              <label>SMILES:</label>
              <code @click="copyText(molecule.smiles)">{{ molecule.smiles }}</code>
            </div>
            <div v-if="molecule.inchi" class="structure-id">
              <label>InChI:</label>
              <code @click="copyText(molecule.inchi)">{{ molecule.inchi }}</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  molecule: {
    type: Object,
    default: null
  },
  loading: {
    type: Boolean,
    default: false
  },
  jobId: {
    type: String,
    default: null
  }
});

function formatPropertyName(name) {
  // Convert snake_case to Title Case
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function formatValue(value) {
  if (typeof value === 'number') {
    if (Math.abs(value) < 0.001 || Math.abs(value) > 10000) {
      return value.toExponential(4);
    }
    return value.toFixed(6).replace(/\.?0+$/, '');
  }
  return value;
}

function formatDate(dateStr) {
  const date = new Date(dateStr);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function formatGeometry(geometry) {
  if (!geometry?.atoms) return '';

  const lines = [];
  lines.push(geometry.atoms.length.toString());
  lines.push(`Generated from ${geometry.source || 'benchmark'}`);

  for (const atom of geometry.atoms) {
    lines.push(
      `${atom.element.padEnd(2)}  ${atom.x.toFixed(6).padStart(12)}  ${atom.y.toFixed(6).padStart(12)}  ${atom.z.toFixed(6).padStart(12)}`
    );
  }

  return lines.join('\n');
}

async function copyGeometry() {
  if (!props.molecule?.geometry) return;

  const xyz = formatGeometry(props.molecule.geometry);
  await navigator.clipboard.writeText(xyz);
}

async function copyText(text) {
  await navigator.clipboard.writeText(text);
}
</script>

<style scoped>
.benchmark-results {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.loading-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px;
  color: var(--text-secondary, #a0a0a0);
  gap: 12px;
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid var(--border-color, #3c3c3c);
  border-top-color: var(--accent-color, #007acc);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.job-id {
  font-size: 11px;
  font-family: monospace;
  opacity: 0.7;
}

.molecule-data {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
}

.molecule-header {
  padding: 16px;
  background: var(--bg-secondary, #1e1e1e);
  border-bottom: 1px solid var(--border-color, #3c3c3c);
}

.molecule-header h3 {
  margin: 0 0 8px 0;
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary, #e0e0e0);
}

.molecule-identifiers {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}

.formula {
  font-family: monospace;
  font-size: 14px;
  background: var(--accent-color, #007acc);
  color: white;
  padding: 2px 8px;
  border-radius: 4px;
}

.identifier {
  font-size: 12px;
  background: var(--bg-primary, #252526);
  padding: 2px 8px;
  border-radius: 4px;
  color: var(--text-secondary, #a0a0a0);
}

.inchi-key {
  font-family: monospace;
  font-size: 11px;
}

.sources,
.cached-at {
  font-size: 12px;
  color: var(--text-secondary, #a0a0a0);
  margin-top: 4px;
}

.molecule-sections {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.section h4 {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary, #e0e0e0);
}

.copy-btn {
  font-size: 11px;
  padding: 4px 8px;
  background: var(--bg-primary, #252526);
  border: 1px solid var(--border-color, #3c3c3c);
  border-radius: 4px;
  color: var(--text-secondary, #a0a0a0);
  cursor: pointer;
}

.copy-btn:hover {
  background: var(--bg-hover, #2a2a2a);
  color: var(--text-primary, #e0e0e0);
}

.properties-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.properties-table th,
.properties-table td {
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border-color, #3c3c3c);
}

.properties-table th {
  background: var(--bg-secondary, #1e1e1e);
  color: var(--text-secondary, #a0a0a0);
  font-weight: 500;
  font-size: 12px;
}

.properties-table td {
  color: var(--text-primary, #e0e0e0);
}

.properties-table td.value {
  font-family: monospace;
}

.properties-table tbody tr:hover {
  background: var(--bg-hover, #2a2a2a);
}

.geometry-info {
  display: flex;
  gap: 16px;
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--text-secondary, #a0a0a0);
}

.geometry-preview {
  background: var(--bg-primary, #252526);
  border: 1px solid var(--border-color, #3c3c3c);
  border-radius: 4px;
  max-height: 200px;
  overflow: auto;
}

.geometry-preview pre {
  margin: 0;
  padding: 12px;
  font-family: monospace;
  font-size: 12px;
  color: var(--text-primary, #e0e0e0);
  white-space: pre;
}

.file-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.file-list li {
  display: flex;
  gap: 8px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-color, #3c3c3c);
  font-size: 13px;
}

.file-name {
  font-weight: 500;
  color: var(--text-primary, #e0e0e0);
}

.file-path {
  font-family: monospace;
  color: var(--text-secondary, #a0a0a0);
  word-break: break-all;
}

.structure-ids {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.structure-id {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.structure-id label {
  font-size: 12px;
  color: var(--text-secondary, #a0a0a0);
}

.structure-id code {
  font-family: monospace;
  font-size: 12px;
  background: var(--bg-primary, #252526);
  padding: 8px 12px;
  border-radius: 4px;
  color: var(--text-primary, #e0e0e0);
  word-break: break-all;
  cursor: pointer;
}

.structure-id code:hover {
  background: var(--bg-hover, #2a2a2a);
}
</style>
