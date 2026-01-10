/**
 * Download Manager - Handles file storage and caching for benchmark data
 *
 * Manages the download directory structure:
 * workspace/{id}/downloads/{source}/{identifier}/...
 */

const fs = require('fs').promises;
const path = require('path');
const { Client } = require('@elastic/elasticsearch');

// Elasticsearch client (lazy initialized)
let esClient = null;

const ES_INDEX = 'benchmark_molecules';
const DOWNLOAD_BASE = process.env.DOWNLOAD_DIR || '/app/workspace';

/**
 * Get or create Elasticsearch client
 */
function getElasticsearchClient() {
  if (!esClient) {
    esClient = new Client({
      node: process.env.ELASTICSEARCH_URL || 'http://localhost:9200'
    });
  }
  return esClient;
}

/**
 * Initialize Elasticsearch index with proper mappings
 */
async function initializeIndex() {
  const client = getElasticsearchClient();

  const indexExists = await client.indices.exists({ index: ES_INDEX });

  if (!indexExists) {
    await client.indices.create({
      index: ES_INDEX,
      body: {
        settings: {
          number_of_shards: 1,
          number_of_replicas: 0
        },
        mappings: {
          properties: {
            identifier: { type: 'keyword' },
            cas: { type: 'keyword' },
            inchi: { type: 'keyword' },
            inchi_key: { type: 'keyword' },
            smiles: { type: 'keyword' },
            name: {
              type: 'text',
              analyzer: 'standard',
              fields: {
                keyword: { type: 'keyword' },
                suggest: { type: 'completion' }
              }
            },
            formula: {
              type: 'keyword',
              fields: {
                suggest: { type: 'completion' }
              }
            },
            sources: { type: 'keyword' },
            properties: {
              type: 'nested',
              properties: {
                name: { type: 'keyword' },
                value: { type: 'float' },
                unit: { type: 'keyword' },
                source: { type: 'keyword' },
                method: { type: 'keyword' },
                uncertainty: { type: 'float' }
              }
            },
            geometry: { type: 'object', enabled: false },
            molecular_weight: { type: 'float' },
            charge: { type: 'integer' },
            multiplicity: { type: 'integer' },
            cached_at: { type: 'date' },
            file_paths: { type: 'object', enabled: false }
          }
        }
      }
    });

    console.log(`Created Elasticsearch index: ${ES_INDEX}`);
  }

  return client;
}

/**
 * Get download directory path for a specific source and identifier
 * @param {string} workspaceId - Workspace ID
 * @param {string} source - Data source (nist, pubchem, qm9)
 * @param {string} identifier - Molecule identifier
 * @returns {string} Full path to download directory
 */
function getDownloadPath(workspaceId, source, identifier) {
  // Sanitize identifier for filesystem
  const safeId = identifier.replace(/[^a-zA-Z0-9_-]/g, '_');
  return path.join(DOWNLOAD_BASE, workspaceId, 'downloads', source, safeId);
}

/**
 * Ensure download directory exists
 * @param {string} dirPath - Directory path
 */
async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

/**
 * Check if molecule data is cached in Elasticsearch
 * @param {string} identifier - Molecule identifier (CAS, SMILES, name, etc.)
 * @param {string[]} sources - Sources to check
 * @returns {Object|null} Cached data or null
 */
async function getCached(identifier, sources = ['nist', 'pubchem', 'qm9']) {
  const client = getElasticsearchClient();

  try {
    const result = await client.search({
      index: ES_INDEX,
      body: {
        query: {
          bool: {
            must: [
              {
                bool: {
                  should: [
                    { term: { identifier } },
                    { term: { cas: identifier } },
                    { term: { smiles: identifier } },
                    { term: { inchi_key: identifier } },
                    { term: { formula: identifier.toUpperCase() } },
                    { term: { 'name.keyword': identifier } },
                    { match: { name: identifier } }
                  ],
                  minimum_should_match: 1
                }
              }
            ],
            filter: [
              { terms: { sources } }
            ]
          }
        }
      }
    });

    if (result.hits.hits.length > 0) {
      return result.hits.hits[0]._source;
    }

    return null;
  } catch (error) {
    // Index might not exist yet
    if (error.meta?.statusCode === 404) {
      return null;
    }
    throw error;
  }
}

/**
 * Search for molecules in Elasticsearch
 * @param {string} query - Search query
 * @param {Object} options - Search options
 * @returns {Object[]} Search results
 */
async function search(query, options = {}) {
  const client = getElasticsearchClient();
  const { sources, limit = 20, offset = 0 } = options;

  try {
    const searchBody = {
      query: {
        bool: {
          should: [
            { match: { name: { query, boost: 2 } } },
            { term: { formula: { value: query.toUpperCase(), boost: 1.5 } } },
            { term: { cas: query } },
            { term: { smiles: query } }
          ],
          minimum_should_match: 1
        }
      },
      from: offset,
      size: limit
    };

    if (sources && sources.length > 0) {
      searchBody.query.bool.filter = [{ terms: { sources } }];
    }

    const result = await client.search({
      index: ES_INDEX,
      body: searchBody
    });

    return result.hits.hits.map(hit => ({
      ...hit._source,
      _score: hit._score
    }));
  } catch (error) {
    if (error.meta?.statusCode === 404) {
      return [];
    }
    throw error;
  }
}

/**
 * Index molecule data to Elasticsearch
 * @param {Object} data - Molecule data
 * @returns {Object} Indexed document
 */
async function indexMolecule(data) {
  const client = await initializeIndex();

  // Generate a consistent document ID
  const docId = data.inchi_key || data.cas || data.identifier;

  // Check if document exists and merge sources
  let existingDoc = null;
  try {
    const existing = await client.get({
      index: ES_INDEX,
      id: docId
    });
    existingDoc = existing._source;
  } catch (error) {
    // Document doesn't exist, that's fine
  }

  // Merge data if existing
  const mergedData = existingDoc ? mergeData(existingDoc, data) : data;
  mergedData.cached_at = new Date().toISOString();

  await client.index({
    index: ES_INDEX,
    id: docId,
    body: mergedData,
    refresh: true
  });

  return mergedData;
}

/**
 * Merge existing and new molecule data
 * @param {Object} existing - Existing data
 * @param {Object} newData - New data to merge
 * @returns {Object} Merged data
 */
function mergeData(existing, newData) {
  const merged = { ...existing };

  // Merge sources array
  const sources = new Set([...(existing.sources || []), ...(newData.sources || [])]);
  merged.sources = Array.from(sources);

  // Merge properties (nested array)
  const existingProps = existing.properties || [];
  const newProps = newData.properties || [];

  // Create a key for each property to avoid duplicates
  const propMap = new Map();
  for (const prop of existingProps) {
    const key = `${prop.name}:${prop.source}:${prop.method || ''}`;
    propMap.set(key, prop);
  }
  for (const prop of newProps) {
    const key = `${prop.name}:${prop.source}:${prop.method || ''}`;
    propMap.set(key, prop);
  }
  merged.properties = Array.from(propMap.values());

  // Merge file_paths
  merged.file_paths = {
    ...(existing.file_paths || {}),
    ...(newData.file_paths || {})
  };

  // Prefer non-null values for scalar fields
  for (const field of ['name', 'formula', 'cas', 'smiles', 'inchi', 'inchi_key', 'molecular_weight', 'charge', 'multiplicity', 'geometry']) {
    if (newData[field] && !merged[field]) {
      merged[field] = newData[field];
    }
  }

  return merged;
}

/**
 * Save file to download directory
 * @param {string} workspaceId - Workspace ID
 * @param {string} source - Data source
 * @param {string} identifier - Molecule identifier
 * @param {string} filename - Filename
 * @param {string|Buffer} content - File content
 * @returns {string} Full file path
 */
async function saveFile(workspaceId, source, identifier, filename, content) {
  const dirPath = getDownloadPath(workspaceId, source, identifier);
  await ensureDir(dirPath);

  const filePath = path.join(dirPath, filename);
  await fs.writeFile(filePath, content);

  return filePath;
}

/**
 * Read file from download directory
 * @param {string} workspaceId - Workspace ID
 * @param {string} source - Data source
 * @param {string} identifier - Molecule identifier
 * @param {string} filename - Filename
 * @returns {string|null} File content or null if not found
 */
async function readFile(workspaceId, source, identifier, filename) {
  const dirPath = getDownloadPath(workspaceId, source, identifier);
  const filePath = path.join(dirPath, filename);

  try {
    return await fs.readFile(filePath, 'utf8');
  } catch (error) {
    if (error.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

/**
 * Check if file exists in download directory
 * @param {string} workspaceId - Workspace ID
 * @param {string} source - Data source
 * @param {string} identifier - Molecule identifier
 * @param {string} filename - Filename
 * @returns {boolean}
 */
async function fileExists(workspaceId, source, identifier, filename) {
  const dirPath = getDownloadPath(workspaceId, source, identifier);
  const filePath = path.join(dirPath, filename);

  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

/**
 * List all cached molecules for a source
 * @param {string} source - Data source
 * @returns {Object[]} List of cached molecules
 */
async function listCached(source) {
  const client = getElasticsearchClient();

  try {
    const result = await client.search({
      index: ES_INDEX,
      body: {
        query: {
          term: { sources: source }
        },
        size: 10000,
        _source: ['identifier', 'name', 'formula', 'cas', 'cached_at']
      }
    });

    return result.hits.hits.map(hit => hit._source);
  } catch (error) {
    if (error.meta?.statusCode === 404) {
      return [];
    }
    throw error;
  }
}

/**
 * Count molecules for a source in Elasticsearch
 * @param {string} source - Data source (e.g., 'qm9', 'nist', 'pubchem')
 * @returns {Promise<number>} Count of molecules
 */
async function countBySource(source) {
  const client = getElasticsearchClient();

  try {
    const result = await client.count({
      index: ES_INDEX,
      body: {
        query: {
          term: { sources: source }
        }
      }
    });

    return result.count;
  } catch (error) {
    if (error.meta?.statusCode === 404) {
      return 0;
    }
    throw error;
  }
}

module.exports = {
  ES_INDEX,
  DOWNLOAD_BASE,
  getElasticsearchClient,
  initializeIndex,
  getDownloadPath,
  ensureDir,
  getCached,
  search,
  indexMolecule,
  mergeData,
  saveFile,
  readFile,
  fileExists,
  listCached,
  countBySource
};
