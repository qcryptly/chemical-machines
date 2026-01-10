/**
 * NIST CCCBDB Scraper
 *
 * Fetches computational chemistry data from NIST Computational Chemistry
 * Comparison and Benchmark Database (CCCBDB)
 * https://cccbdb.nist.gov/
 */

const https = require('https');
const http = require('http');
const downloadManager = require('./download-manager');

const NIST_BASE_URL = 'https://cccbdb.nist.gov';

/**
 * Make HTTP/HTTPS request with promise interface
 * @param {string} url - URL to fetch
 * @param {Object} options - Request options
 * @returns {Promise<string>} Response body
 */
function fetch(url, options = {}) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    const req = protocol.get(url, {
      headers: {
        'User-Agent': 'ChemicalMachines/1.0 (Computational Chemistry Research)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        ...options.headers
      },
      timeout: options.timeout || 30000
    }, (res) => {
      // Handle redirects
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        const redirectUrl = res.headers.location.startsWith('http')
          ? res.headers.location
          : new URL(res.headers.location, url).href;
        return fetch(redirectUrl, options).then(resolve).catch(reject);
      }

      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`));
        return;
      }

      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve(data));
    });

    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });
  });
}

/**
 * Search NIST CCCBDB for a molecule
 * @param {string} query - Search query (name, formula, or CAS)
 * @returns {Promise<Object[]>} Search results
 */
async function searchNIST(query) {
  // NIST uses form-based search, we need to construct the right URL
  const searchUrl = `${NIST_BASE_URL}/cccbdb/getformx.asp?formula=${encodeURIComponent(query)}`;

  try {
    const html = await fetch(searchUrl);
    return parseSearchResults(html);
  } catch (error) {
    console.error(`NIST search failed for "${query}":`, error.message);
    return [];
  }
}

/**
 * Parse search results from NIST HTML
 * @param {string} html - Raw HTML response
 * @returns {Object[]} Parsed results
 */
function parseSearchResults(html) {
  const results = [];

  // Match molecule entries - NIST uses tables with specific patterns
  // Example pattern: <a href="exp2x.asp?casno=7732185">Water</a>
  const moleculePattern = /<a\s+href="exp2x\.asp\?casno=(\d+)"[^>]*>([^<]+)<\/a>/gi;
  const formulaPattern = /<td[^>]*>([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)<\/td>/gi;

  let match;
  while ((match = moleculePattern.exec(html)) !== null) {
    const casno = match[1];
    const name = match[2].trim();

    // Format CAS number (e.g., 7732185 -> 7732-18-5)
    const cas = formatCAS(casno);

    results.push({
      cas,
      casno,
      name,
      source: 'nist'
    });
  }

  return results;
}

/**
 * Format CAS number from NIST format
 * @param {string} casno - CAS number without dashes
 * @returns {string} Formatted CAS number
 */
function formatCAS(casno) {
  // CAS format: XXXXXXX-XX-X
  // Last digit is check digit, second-to-last two digits are middle group
  const s = casno.toString().padStart(5, '0');
  const len = s.length;
  return `${s.slice(0, len - 3)}-${s.slice(len - 3, len - 1)}-${s.slice(len - 1)}`;
}

/**
 * Fetch experimental geometry from NIST
 * @param {string} casno - CAS number (without dashes)
 * @returns {Promise<Object|null>} Geometry data
 */
async function fetchGeometry(casno) {
  const url = `${NIST_BASE_URL}/cccbdb/exp2x.asp?casno=${casno}`;

  try {
    const html = await fetch(url);
    return parseGeometry(html);
  } catch (error) {
    console.error(`Failed to fetch geometry for ${casno}:`, error.message);
    return null;
  }
}

/**
 * Parse geometry data from NIST HTML
 * @param {string} html - Raw HTML response
 * @returns {Object|null} Parsed geometry
 */
function parseGeometry(html) {
  // NIST provides geometry in various formats
  // Look for Cartesian coordinates table

  // Match atom lines: Element X Y Z
  const coordPattern = /([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/g;

  // First try to find the geometry section
  const geometrySection = html.match(/experimental\s+geometry/i);
  if (!geometrySection) {
    return null;
  }

  const atoms = [];
  let match;

  // Extract coordinates from tables
  const tableMatch = html.match(/<table[^>]*>[\s\S]*?cartesian[\s\S]*?<\/table>/i);
  if (tableMatch) {
    while ((match = coordPattern.exec(tableMatch[0])) !== null) {
      atoms.push({
        element: match[1],
        x: parseFloat(match[2]),
        y: parseFloat(match[3]),
        z: parseFloat(match[4])
      });
    }
  }

  if (atoms.length === 0) {
    return null;
  }

  return {
    atoms,
    unit: 'angstrom',
    source: 'nist_experimental'
  };
}

/**
 * Fetch calculated properties from NIST
 * @param {string} casno - CAS number (without dashes)
 * @returns {Promise<Object[]>} Properties array
 */
async function fetchProperties(casno) {
  const properties = [];

  // Fetch various property pages
  const propertyUrls = [
    { url: `${NIST_BASE_URL}/cccbdb/expmx.asp?casno=${casno}`, type: 'experimental' },
    { url: `${NIST_BASE_URL}/cccbdb/energy2x.asp?casno=${casno}`, type: 'energy' },
    { url: `${NIST_BASE_URL}/cccbdb/dipole2x.asp?casno=${casno}`, type: 'dipole' }
  ];

  for (const { url, type } of propertyUrls) {
    try {
      const html = await fetch(url);
      const parsed = parseProperties(html, type);
      properties.push(...parsed);
    } catch (error) {
      // Continue with other properties
      console.error(`Failed to fetch ${type} for ${casno}:`, error.message);
    }
  }

  return properties;
}

/**
 * Parse properties from NIST HTML
 * @param {string} html - Raw HTML response
 * @param {string} type - Property type
 * @returns {Object[]} Parsed properties
 */
function parseProperties(html, type) {
  const properties = [];

  // Parse based on property type
  switch (type) {
    case 'experimental':
      // Look for experimental values
      const expPattern = /(\w+(?:\s+\w+)*)\s*[:=]\s*([-\d.]+)\s*([a-zA-Z/]+)?/g;
      let match;
      while ((match = expPattern.exec(html)) !== null) {
        const name = match[1].toLowerCase().replace(/\s+/g, '_');
        const value = parseFloat(match[2]);
        const unit = match[3] || '';

        if (!isNaN(value)) {
          properties.push({
            name,
            value,
            unit,
            source: 'nist',
            method: 'experimental'
          });
        }
      }
      break;

    case 'energy':
      // Parse energy values from tables
      // NIST shows energies in Hartree
      const energyPattern = /(-?\d+\.\d+)\s*(?:hartree|Ha|au)/gi;
      while ((match = energyPattern.exec(html)) !== null) {
        properties.push({
          name: 'total_energy',
          value: parseFloat(match[1]),
          unit: 'hartree',
          source: 'nist',
          method: 'experimental'
        });
        break; // Take first match
      }
      break;

    case 'dipole':
      // Parse dipole moment
      const dipolePattern = /dipole\s*(?:moment)?[^:]*:\s*([\d.]+)\s*(?:debye|D)/i;
      const dipoleMatch = html.match(dipolePattern);
      if (dipoleMatch) {
        properties.push({
          name: 'dipole_moment',
          value: parseFloat(dipoleMatch[1]),
          unit: 'debye',
          source: 'nist',
          method: 'experimental'
        });
      }
      break;
  }

  return properties;
}

/**
 * Fetch vibrational frequencies from NIST
 * @param {string} casno - CAS number (without dashes)
 * @returns {Promise<Object|null>} Frequency data
 */
async function fetchFrequencies(casno) {
  const url = `${NIST_BASE_URL}/cccbdb/vibratex.asp?casno=${casno}`;

  try {
    const html = await fetch(url);
    return parseFrequencies(html);
  } catch (error) {
    console.error(`Failed to fetch frequencies for ${casno}:`, error.message);
    return null;
  }
}

/**
 * Parse vibrational frequencies from NIST HTML
 * @param {string} html - Raw HTML response
 * @returns {Object|null} Parsed frequencies
 */
function parseFrequencies(html) {
  // Match frequency entries: mode number, symmetry, frequency
  const freqPattern = /(\d+)\s+([A-Za-z'\"]+)\s+(\d+(?:\.\d+)?)/g;

  const frequencies = [];
  let match;

  while ((match = freqPattern.exec(html)) !== null) {
    frequencies.push({
      mode: parseInt(match[1]),
      symmetry: match[2],
      frequency: parseFloat(match[3]),
      unit: 'cm-1'
    });
  }

  if (frequencies.length === 0) {
    return null;
  }

  return {
    frequencies,
    unit: 'cm-1',
    source: 'nist_experimental'
  };
}

/**
 * Fetch complete molecule data from NIST
 * @param {string} identifier - CAS number or formula
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} Complete molecule data
 */
async function fetchMolecule(identifier, options = {}) {
  const { workspaceId = '1', emit = () => {} } = options;

  emit('stdout', `Fetching molecule data from NIST for: ${identifier}\n`);
  emit('progress', 10);

  // First search if it's not a CAS number
  let casno = identifier.replace(/-/g, '');

  if (!/^\d+$/.test(casno)) {
    emit('stdout', `Searching NIST for: ${identifier}\n`);
    const results = await searchNIST(identifier);

    if (results.length === 0) {
      throw new Error(`No results found in NIST for: ${identifier}`);
    }

    casno = results[0].casno;
    emit('stdout', `Found: ${results[0].name} (CAS: ${results[0].cas})\n`);
  }

  emit('progress', 20);

  // Fetch all data in parallel
  const [geometry, properties, frequencies] = await Promise.all([
    fetchGeometry(casno),
    fetchProperties(casno),
    fetchFrequencies(casno)
  ]);

  emit('progress', 70);

  // Build molecule data object
  const moleculeData = {
    identifier: formatCAS(casno),
    cas: formatCAS(casno),
    sources: ['nist'],
    properties: properties || [],
    geometry: geometry || null,
    file_paths: {}
  };

  // Save files to workspace
  if (geometry) {
    const xyzContent = formatXYZ(geometry);
    const xyzPath = await downloadManager.saveFile(
      workspaceId, 'nist', formatCAS(casno), 'geometry.xyz', xyzContent
    );
    moleculeData.file_paths.geometry = xyzPath;
    emit('stdout', `Saved geometry to: ${xyzPath}\n`);
  }

  if (frequencies) {
    const freqPath = await downloadManager.saveFile(
      workspaceId, 'nist', formatCAS(casno), 'frequencies.json',
      JSON.stringify(frequencies, null, 2)
    );
    moleculeData.file_paths.frequencies = freqPath;
    emit('stdout', `Saved frequencies to: ${freqPath}\n`);
  }

  // Save metadata
  const metadataPath = await downloadManager.saveFile(
    workspaceId, 'nist', formatCAS(casno), 'metadata.json',
    JSON.stringify(moleculeData, null, 2)
  );
  moleculeData.file_paths.metadata = metadataPath;

  emit('progress', 90);

  // Index to Elasticsearch
  await downloadManager.indexMolecule(moleculeData);
  emit('stdout', `Indexed molecule in Elasticsearch\n`);

  emit('progress', 100);

  return moleculeData;
}

/**
 * Format geometry as XYZ file content
 * @param {Object} geometry - Geometry object with atoms array
 * @returns {string} XYZ file content
 */
function formatXYZ(geometry) {
  const lines = [];
  lines.push(geometry.atoms.length.toString());
  lines.push('Generated by ChemicalMachines from NIST CCCBDB');

  for (const atom of geometry.atoms) {
    lines.push(`${atom.element}  ${atom.x.toFixed(6)}  ${atom.y.toFixed(6)}  ${atom.z.toFixed(6)}`);
  }

  return lines.join('\n');
}

module.exports = {
  searchNIST,
  fetchGeometry,
  fetchProperties,
  fetchFrequencies,
  fetchMolecule,
  formatCAS,
  formatXYZ
};
