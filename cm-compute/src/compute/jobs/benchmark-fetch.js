/**
 * Benchmark Fetch Job Handler
 *
 * Main dispatcher for fetching molecular benchmark data from various sources.
 * Supports: NIST CCCBDB, PubChem, QM9
 */

const downloadManager = require('./scrapers/download-manager');
const nist = require('./scrapers/nist');
const pubchem = require('./scrapers/pubchem');
const qm9 = require('./scrapers/qm9');

/**
 * Fetch benchmark data for a molecule
 * @param {Object} params - Job parameters
 * @param {string} params.identifier - Molecule identifier (CAS, name, SMILES, CID, etc.)
 * @param {string[]} params.sources - Data sources to query (default: all)
 * @param {string} params.workspaceId - Workspace ID for file storage
 * @param {Object} context - Execution context { pythonPath, jobId, emit }
 * @returns {Promise<Object>} Fetched molecule data
 */
async function benchmarkFetch(params, context) {
  const { identifier, sources = ['pubchem', 'nist', 'qm9'], workspaceId = '1' } = params;
  const { emit = () => {} } = context;

  if (!identifier) {
    throw new Error('Missing required parameter: identifier');
  }

  emit('stdout', `Fetching benchmark data for: ${identifier}\n`);
  emit('stdout', `Sources: ${sources.join(', ')}\n`);
  emit('progress', 5);

  // Check cache first
  const cached = await downloadManager.getCached(identifier, sources);
  if (cached) {
    emit('stdout', `Found cached data (cached at: ${cached.cached_at})\n`);
    emit('progress', 100);
    return {
      status: 'cached',
      data: cached
    };
  }

  emit('stdout', `No cached data, fetching from sources...\n`);
  emit('progress', 10);

  // Fetch from each source
  const results = [];
  const errors = [];
  const progressPerSource = 80 / sources.length;

  for (let i = 0; i < sources.length; i++) {
    const source = sources[i];
    const sourceProgress = 10 + (i * progressPerSource);

    emit('stdout', `\nQuerying ${source.toUpperCase()}...\n`);

    try {
      let data;

      switch (source.toLowerCase()) {
        case 'pubchem':
          data = await pubchem.fetchMolecule(identifier, {
            workspaceId,
            emit: (type, msg) => {
              if (type === 'stdout') emit('stdout', `  [PubChem] ${msg}`);
              if (type === 'progress') {
                emit('progress', Math.round(sourceProgress + (msg / 100) * progressPerSource));
              }
            }
          });
          break;

        case 'nist':
          data = await nist.fetchMolecule(identifier, {
            workspaceId,
            emit: (type, msg) => {
              if (type === 'stdout') emit('stdout', `  [NIST] ${msg}`);
              if (type === 'progress') {
                emit('progress', Math.round(sourceProgress + (msg / 100) * progressPerSource));
              }
            }
          });
          break;

        case 'qm9':
          data = await qm9.fetchMolecule(identifier, {
            workspaceId,
            emit: (type, msg) => {
              if (type === 'stdout') emit('stdout', `  [QM9] ${msg}`);
              if (type === 'progress') {
                emit('progress', Math.round(sourceProgress + (msg / 100) * progressPerSource));
              }
            }
          });
          break;

        default:
          emit('stdout', `Unknown source: ${source}\n`);
          continue;
      }

      if (data) {
        results.push(data);
        emit('stdout', `  Found data from ${source}\n`);
      }
    } catch (error) {
      errors.push({ source, error: error.message });
      emit('stdout', `  No data from ${source}: ${error.message}\n`);
    }
  }

  emit('progress', 95);

  if (results.length === 0) {
    throw new Error(`No benchmark data found for "${identifier}" from any source. Errors: ${errors.map(e => `${e.source}: ${e.error}`).join('; ')}`);
  }

  // Merge results from all sources
  const merged = mergeResults(results);

  emit('progress', 100);
  emit('stdout', `\nFetch complete. Data from ${merged.sources.join(', ')}\n`);

  return {
    status: 'fetched',
    data: merged,
    sources_queried: sources,
    sources_found: merged.sources,
    errors: errors.length > 0 ? errors : undefined
  };
}

/**
 * Merge results from multiple sources
 * @param {Object[]} results - Array of molecule data objects
 * @returns {Object} Merged data
 */
function mergeResults(results) {
  if (results.length === 0) return null;
  if (results.length === 1) return results[0];

  // Start with first result as base
  const merged = { ...results[0] };
  merged.sources = [...(merged.sources || [])];
  merged.properties = [...(merged.properties || [])];
  merged.file_paths = { ...(merged.file_paths || {}) };

  // Merge in additional results
  for (let i = 1; i < results.length; i++) {
    const result = results[i];

    // Merge sources
    for (const source of (result.sources || [])) {
      if (!merged.sources.includes(source)) {
        merged.sources.push(source);
      }
    }

    // Merge properties (avoid duplicates by name+source)
    const existingKeys = new Set(
      merged.properties.map(p => `${p.name}:${p.source}:${p.method || ''}`)
    );

    for (const prop of (result.properties || [])) {
      const key = `${prop.name}:${prop.source}:${prop.method || ''}`;
      if (!existingKeys.has(key)) {
        merged.properties.push(prop);
        existingKeys.add(key);
      }
    }

    // Merge file paths
    merged.file_paths = {
      ...merged.file_paths,
      ...(result.file_paths || {})
    };

    // Fill in missing fields
    for (const field of ['name', 'formula', 'cas', 'smiles', 'inchi', 'inchi_key', 'geometry', 'molecular_weight']) {
      if (!merged[field] && result[field]) {
        merged[field] = result[field];
      }
    }
  }

  return merged;
}

/**
 * Index benchmark data to Elasticsearch (for re-indexing)
 * @param {Object} params - Job parameters
 * @param {Object} params.data - Molecule data to index
 * @param {Object} context - Execution context
 * @returns {Promise<Object>} Index result
 */
async function benchmarkIndex(params, context) {
  const { data } = params;
  const { emit = () => {} } = context;

  if (!data) {
    throw new Error('Missing required parameter: data');
  }

  emit('stdout', `Indexing molecule: ${data.identifier || data.name}\n`);

  await downloadManager.indexMolecule(data);

  emit('stdout', 'Index complete\n');
  emit('progress', 100);

  return { status: 'indexed', identifier: data.identifier };
}

/**
 * Sync benchmark databases (for cron scheduling)
 * @param {Object} params - Job parameters
 * @param {string[]} params.sources - Sources to sync
 * @param {string} params.workspaceId - Workspace ID
 * @param {Object} context - Execution context
 * @returns {Promise<Object>} Sync result
 */
async function benchmarkSync(params, context) {
  const { sources = ['qm9'], workspaceId = '1' } = params;
  const { emit = () => {} } = context;

  emit('stdout', `Starting benchmark sync for: ${sources.join(', ')}\n`);
  emit('progress', 5);

  const results = {};

  for (const source of sources) {
    emit('stdout', `\nSyncing ${source}...\n`);

    switch (source.toLowerCase()) {
      case 'qm9':
        // Download and index QM9 dataset
        results.qm9 = await qm9.downloadDataset({
          workspaceId,
          emit: (type, msg) => {
            if (type === 'stdout') emit('stdout', `  [QM9] ${msg}`);
            if (type === 'progress') emit('progress', msg);
          }
        });
        break;

      case 'pubchem':
        // PubChem is fetched on-demand, just log status
        emit('stdout', '  PubChem data is fetched on-demand\n');
        results.pubchem = { status: 'on-demand' };
        break;

      case 'nist':
        // NIST is fetched on-demand
        emit('stdout', '  NIST data is fetched on-demand\n');
        results.nist = { status: 'on-demand' };
        break;

      default:
        emit('stdout', `  Unknown source: ${source}\n`);
    }
  }

  emit('progress', 100);
  emit('stdout', '\nSync complete\n');

  return { status: 'complete', results };
}

/**
 * Search benchmark databases
 * @param {Object} params - Job parameters
 * @param {string} params.query - Search query
 * @param {string[]} params.sources - Sources to search
 * @param {number} params.limit - Max results
 * @param {Object} context - Execution context
 * @returns {Promise<Object[]>} Search results
 */
async function benchmarkSearch(params, context) {
  const { query, sources = ['pubchem', 'nist', 'qm9'], limit = 20 } = params;
  const { emit = () => {} } = context;

  if (!query) {
    throw new Error('Missing required parameter: query');
  }

  emit('stdout', `Searching for: ${query}\n`);
  emit('progress', 10);

  // Search Elasticsearch first
  const cached = await downloadManager.search(query, { sources, limit });

  if (cached.length > 0) {
    emit('stdout', `Found ${cached.length} cached results\n`);
    emit('progress', 100);
    return {
      status: 'cached',
      results: cached,
      total: cached.length
    };
  }

  // If no cached results, search PubChem API directly
  emit('stdout', 'No cached results, searching PubChem...\n');
  emit('progress', 50);

  const pubchemResults = await pubchem.searchPubChem(query);

  emit('progress', 100);

  return {
    status: 'live',
    results: pubchemResults,
    total: pubchemResults.length,
    note: 'Results from PubChem search. Fetch individual molecules for full data.'
  };
}

/**
 * Compare computed values with benchmark data
 * @param {Object} params - Job parameters
 * @param {Object} params.computed - Computed properties
 * @param {string} params.identifier - Benchmark molecule identifier
 * @param {string[]} params.sources - Data sources to query (default: all)
 * @param {string} params.workspaceId - Workspace ID for file storage
 * @param {Object} context - Execution context
 * @returns {Promise<Object>} Comparison results
 */
async function benchmarkCompare(params, context) {
  const { computed, identifier, sources = ['pubchem', 'nist', 'qm9'], workspaceId = '1' } = params;
  const { emit = () => {} } = context;

  if (!computed || !identifier) {
    throw new Error('Missing required parameters: computed, identifier');
  }

  emit('stdout', `Comparing computed values with benchmark: ${identifier}\n`);
  emit('progress', 10);

  // Get benchmark data - first try cache
  let benchmark = await downloadManager.getCached(identifier, sources);

  // If not found in cache, try to fetch from external sources
  if (!benchmark) {
    emit('stdout', `Not found in cache, fetching from external sources...\n`);
    emit('progress', 20);

    const errors = [];

    // Try each source until we find the molecule
    for (const source of sources) {
      try {
        let data;

        switch (source.toLowerCase()) {
          case 'pubchem':
            data = await pubchem.fetchMolecule(identifier, { workspaceId, emit: () => {} });
            break;
          case 'nist':
            data = await nist.fetchMolecule(identifier, { workspaceId, emit: () => {} });
            break;
          case 'qm9':
            data = await qm9.fetchMolecule(identifier, { workspaceId, emit: () => {} });
            break;
        }

        if (data) {
          benchmark = data;
          emit('stdout', `Found data from ${source}\n`);
          break;
        }
      } catch (error) {
        errors.push({ source, error: error.message });
      }
    }

    if (!benchmark) {
      throw new Error(`No benchmark data found for: ${identifier}. Tried sources: ${sources.join(', ')}`);
    }
  }

  emit('progress', 50);

  // Compare properties
  const comparisons = [];

  for (const [propName, computedValue] of Object.entries(computed)) {
    // Find matching benchmark property
    const benchmarkProp = benchmark.properties?.find(p =>
      p.name === propName || p.name.includes(propName)
    );

    if (benchmarkProp) {
      const diff = computedValue - benchmarkProp.value;
      const percentDiff = (diff / benchmarkProp.value) * 100;

      comparisons.push({
        property: propName,
        computed: computedValue,
        benchmark: benchmarkProp.value,
        benchmark_source: benchmarkProp.source,
        benchmark_method: benchmarkProp.method,
        difference: diff,
        percent_difference: percentDiff,
        unit: benchmarkProp.unit
      });
    } else {
      comparisons.push({
        property: propName,
        computed: computedValue,
        benchmark: null,
        note: 'No benchmark value available'
      });
    }
  }

  emit('progress', 100);
  emit('stdout', `Comparison complete: ${comparisons.length} properties\n`);

  return {
    identifier,
    benchmark_sources: benchmark.sources,
    comparisons,
    summary: {
      properties_compared: comparisons.filter(c => c.benchmark !== null).length,
      properties_missing: comparisons.filter(c => c.benchmark === null).length,
      avg_percent_diff: mean(
        comparisons
          .filter(c => c.percent_difference !== undefined)
          .map(c => Math.abs(c.percent_difference))
      )
    }
  };
}

/**
 * Calculate mean of array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Mean value
 */
function mean(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

module.exports = {
  benchmarkFetch,
  benchmarkIndex,
  benchmarkSync,
  benchmarkSearch,
  benchmarkCompare
};
