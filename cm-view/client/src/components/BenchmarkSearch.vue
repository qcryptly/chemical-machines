<template>
  <div class="benchmark-search">
    <div class="search-header">
      <h3>Benchmark Database Search</h3>
      <div class="search-sources">
        <label v-for="source in availableSources" :key="source.id" class="source-checkbox">
          <input
            type="checkbox"
            :value="source.id"
            v-model="selectedSources"
          />
          {{ source.label }}
        </label>
      </div>
    </div>

    <div class="search-input-container">
      <input
        type="text"
        v-model="query"
        @keyup.enter="search"
        placeholder="Search by name, formula, CAS, SMILES, or InChIKey..."
        class="search-input"
        :disabled="status === 'searching'"
      />
      <button
        @click="search"
        :disabled="!query.trim() || status === 'searching'"
        class="search-button"
      >
        {{ status === 'searching' ? 'Searching...' : 'Search' }}
      </button>
    </div>

    <div v-if="status === 'searching'" class="search-status">
      <div class="spinner"></div>
      <span>Searching databases...</span>
    </div>

    <div v-if="error" class="search-error">
      {{ error }}
    </div>

    <div v-if="results.length > 0" class="search-results">
      <div class="results-header">
        <span>{{ results.length }} result{{ results.length !== 1 ? 's' : '' }}</span>
        <span v-if="resultSource" class="result-source">from {{ resultSource }}</span>
      </div>

      <div class="results-list">
        <div
          v-for="result in results"
          :key="result.identifier || result.cid || result.cas"
          class="result-item"
          @click="selectMolecule(result)"
        >
          <div class="result-name">{{ result.name || 'Unknown' }}</div>
          <div class="result-details">
            <span v-if="result.formula" class="result-formula">{{ result.formula }}</span>
            <span v-if="result.cas" class="result-cas">CAS: {{ result.cas }}</span>
            <span v-if="result.cid" class="result-cid">CID: {{ result.cid }}</span>
            <span v-if="result.sources" class="result-sources">
              {{ result.sources.join(', ') }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <div v-if="status === 'idle' && results.length === 0 && hasSearched" class="no-results">
      No results found for "{{ lastQuery }}"
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue';

const props = defineProps({
  workspaceId: {
    type: String,
    default: '1'
  }
});

const emit = defineEmits(['select', 'fetch-started']);

const query = ref('');
const lastQuery = ref('');
const selectedSources = ref(['pubchem', 'nist', 'qm9']);
const status = ref('idle'); // idle, searching, fetching
const results = ref([]);
const resultSource = ref('');
const error = ref('');
const hasSearched = ref(false);
const currentJobId = ref(null);
let ws = null;

const availableSources = [
  { id: 'pubchem', label: 'PubChem' },
  { id: 'nist', label: 'NIST CCCBDB' },
  { id: 'qm9', label: 'QM9' }
];

async function search() {
  if (!query.value.trim()) return;

  status.value = 'searching';
  error.value = '';
  results.value = [];
  lastQuery.value = query.value;
  hasSearched.value = true;

  try {
    const params = new URLSearchParams({
      q: query.value,
      sources: selectedSources.value.join(','),
      limit: '20'
    });

    const response = await fetch(`/api/benchmark/search?${params}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Search failed');
    }

    if (data.status === 'cached') {
      results.value = data.results;
      resultSource.value = 'cache';
      status.value = 'idle';
    } else if (data.status === 'searching' && data.jobId) {
      currentJobId.value = data.jobId;
      subscribeToJob(data.jobId);
    } else {
      results.value = data.results || [];
      resultSource.value = 'live search';
      status.value = 'idle';
    }
  } catch (err) {
    error.value = err.message;
    status.value = 'idle';
  }
}

function subscribeToJob(jobId) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connectWebSocket(() => {
      ws.send(JSON.stringify({ type: 'subscribe', jobId }));
    });
  } else {
    ws.send(JSON.stringify({ type: 'subscribe', jobId }));
  }
}

function connectWebSocket(onOpen) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    if (onOpen) onOpen();
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.jobId === currentJobId.value) {
      if (data.type === 'job_result' || data.stream === 'result') {
        const result = data.result || data.data;
        if (result && result.results) {
          results.value = result.results;
          resultSource.value = result.status || 'search';
        }
        status.value = 'idle';
        currentJobId.value = null;
      } else if (data.type === 'error' || data.stream === 'error') {
        error.value = data.error || data.data || 'Search failed';
        status.value = 'idle';
        currentJobId.value = null;
      }
    }
  };

  ws.onerror = () => {
    error.value = 'WebSocket connection error';
    status.value = 'idle';
  };
}

async function selectMolecule(molecule) {
  const identifier = molecule.identifier || molecule.cas || molecule.cid?.toString();

  if (!identifier) {
    error.value = 'No valid identifier for this molecule';
    return;
  }

  // Check if we have full data
  if (molecule.properties && molecule.properties.length > 0) {
    emit('select', molecule);
    return;
  }

  // Fetch full data
  status.value = 'fetching';
  error.value = '';

  try {
    const params = new URLSearchParams({
      sources: selectedSources.value.join(','),
      workspaceId: props.workspaceId
    });

    const response = await fetch(`/api/benchmark/molecule/${encodeURIComponent(identifier)}?${params}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Fetch failed');
    }

    if (data.status === 'cached') {
      emit('select', data.data);
      status.value = 'idle';
    } else if (data.status === 'fetching' && data.jobId) {
      currentJobId.value = data.jobId;
      emit('fetch-started', { jobId: data.jobId, identifier });
      subscribeToFetchJob(data.jobId);
    }
  } catch (err) {
    error.value = err.message;
    status.value = 'idle';
  }
}

function subscribeToFetchJob(jobId) {
  subscribeToJob(jobId);

  // Override message handler for fetch
  const originalHandler = ws.onmessage;
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.jobId === jobId) {
      if (data.type === 'job_result' || data.stream === 'result') {
        const result = data.result || data.data;
        if (result && result.data) {
          emit('select', result.data);
        }
        status.value = 'idle';
        currentJobId.value = null;
        ws.onmessage = originalHandler;
      } else if (data.type === 'error' || data.stream === 'error') {
        error.value = data.error || data.data || 'Fetch failed';
        status.value = 'idle';
        currentJobId.value = null;
        ws.onmessage = originalHandler;
      }
    }
  };
}

onUnmounted(() => {
  if (ws) {
    ws.close();
  }
});
</script>

<style scoped>
.benchmark-search {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  background: var(--bg-secondary, #1e1e1e);
  border-radius: 8px;
}

.search-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.search-header h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary, #e0e0e0);
}

.search-sources {
  display: flex;
  gap: 12px;
}

.source-checkbox {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--text-secondary, #a0a0a0);
  cursor: pointer;
}

.search-input-container {
  display: flex;
  gap: 8px;
}

.search-input {
  flex: 1;
  padding: 8px 12px;
  background: var(--bg-primary, #252526);
  border: 1px solid var(--border-color, #3c3c3c);
  border-radius: 4px;
  color: var(--text-primary, #e0e0e0);
  font-size: 13px;
}

.search-input:focus {
  outline: none;
  border-color: var(--accent-color, #007acc);
}

.search-button {
  padding: 8px 16px;
  background: var(--accent-color, #007acc);
  border: none;
  border-radius: 4px;
  color: white;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.2s;
}

.search-button:hover:not(:disabled) {
  background: var(--accent-hover, #005a9e);
}

.search-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.search-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: var(--bg-primary, #252526);
  border-radius: 4px;
  font-size: 13px;
  color: var(--text-secondary, #a0a0a0);
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid var(--border-color, #3c3c3c);
  border-top-color: var(--accent-color, #007acc);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.search-error {
  padding: 12px;
  background: rgba(255, 0, 0, 0.1);
  border: 1px solid rgba(255, 0, 0, 0.3);
  border-radius: 4px;
  color: #ff6b6b;
  font-size: 13px;
}

.search-results {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.results-header {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--text-secondary, #a0a0a0);
}

.result-source {
  font-style: italic;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 300px;
  overflow-y: auto;
}

.result-item {
  padding: 10px 12px;
  background: var(--bg-primary, #252526);
  border: 1px solid var(--border-color, #3c3c3c);
  border-radius: 4px;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}

.result-item:hover {
  border-color: var(--accent-color, #007acc);
  background: var(--bg-hover, #2a2a2a);
}

.result-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary, #e0e0e0);
  margin-bottom: 4px;
}

.result-details {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 12px;
  color: var(--text-secondary, #a0a0a0);
}

.result-formula {
  font-family: monospace;
  background: var(--bg-secondary, #1e1e1e);
  padding: 2px 6px;
  border-radius: 3px;
}

.result-sources {
  font-style: italic;
}

.no-results {
  padding: 16px;
  text-align: center;
  color: var(--text-secondary, #a0a0a0);
  font-size: 13px;
}
</style>
