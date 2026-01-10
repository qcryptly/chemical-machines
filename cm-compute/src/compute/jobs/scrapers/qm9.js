/**
 * QM9 Dataset Loader
 *
 * Loads and indexes the QM9 dataset for quantum chemistry benchmarks
 * QM9: Quantum chemistry structures and properties of 134k molecules
 *
 * Dataset contains:
 * - 133,885 organic molecules with up to 9 heavy atoms (C, O, N, F)
 * - DFT-computed properties at B3LYP/6-31G(2df,p) level
 * - Properties: HOMO, LUMO, gap, dipole moment, polarizability, etc.
 *
 * Source: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
 */

const https = require('https');
const fs = require('fs').promises;
const path = require('path');
const { createReadStream, createWriteStream } = require('fs');
const { createGunzip } = require('zlib');
const downloadManager = require('./download-manager');

// QM9 dataset URLs (figshare)
const QM9_URLS = {
  // Main dataset (compressed XYZ files)
  dataset: 'https://figshare.com/ndownloader/files/3195389',
  // Atomref: atomic reference energies
  atomref: 'https://figshare.com/ndownloader/files/3195395',
  // Uncharacterized molecules (should be excluded)
  uncharacterized: 'https://figshare.com/ndownloader/files/3195404'
};

// QM9 property indices in the XYZ comment line
const QM9_PROPERTIES = {
  tag: 0,           // gdb9 molecule tag
  idx: 1,           // Index (1-133885)
  A: 2,             // Rotational constant A (GHz)
  B: 3,             // Rotational constant B (GHz)
  C: 4,             // Rotational constant C (GHz)
  mu: 5,            // Dipole moment (Debye)
  alpha: 6,         // Isotropic polarizability (Bohr^3)
  homo: 7,          // HOMO energy (Hartree)
  lumo: 8,          // LUMO energy (Hartree)
  gap: 9,           // HOMO-LUMO gap (Hartree)
  r2: 10,           // Electronic spatial extent (Bohr^2)
  zpve: 11,         // Zero point vibrational energy (Hartree)
  U0: 12,           // Internal energy at 0K (Hartree)
  U: 13,            // Internal energy at 298.15K (Hartree)
  H: 14,            // Enthalpy at 298.15K (Hartree)
  G: 15,            // Free energy at 298.15K (Hartree)
  Cv: 16            // Heat capacity at 298.15K (cal/mol K)
};

// Property metadata
const PROPERTY_INFO = {
  A: { name: 'rotational_constant_A', unit: 'GHz' },
  B: { name: 'rotational_constant_B', unit: 'GHz' },
  C: { name: 'rotational_constant_C', unit: 'GHz' },
  mu: { name: 'dipole_moment', unit: 'debye' },
  alpha: { name: 'polarizability', unit: 'bohr^3' },
  homo: { name: 'homo_energy', unit: 'hartree' },
  lumo: { name: 'lumo_energy', unit: 'hartree' },
  gap: { name: 'homo_lumo_gap', unit: 'hartree' },
  r2: { name: 'electronic_spatial_extent', unit: 'bohr^2' },
  zpve: { name: 'zero_point_energy', unit: 'hartree' },
  U0: { name: 'internal_energy_0K', unit: 'hartree' },
  U: { name: 'internal_energy_298K', unit: 'hartree' },
  H: { name: 'enthalpy_298K', unit: 'hartree' },
  G: { name: 'free_energy_298K', unit: 'hartree' },
  Cv: { name: 'heat_capacity', unit: 'cal/mol/K' }
};

/**
 * Format bytes to human readable string
 * @param {number} bytes - Number of bytes
 * @returns {string} Formatted string
 */
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/**
 * Download file with progress callback
 * @param {string} url - URL to download
 * @param {string} destPath - Destination path
 * @param {Function} onProgress - Progress callback (progress, downloaded, total)
 * @returns {Promise<void>}
 */
function downloadFile(url, destPath, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const makeRequest = (requestUrl) => {
      https.get(requestUrl, (res) => {
        // Handle redirects
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          return makeRequest(res.headers.location);
        }

        if (res.statusCode !== 200) {
          reject(new Error(`Download failed: HTTP ${res.statusCode}`));
          return;
        }

        const totalSize = parseInt(res.headers['content-length'], 10) || 0;
        let downloaded = 0;
        let lastEmit = 0;

        const file = createWriteStream(destPath);
        res.on('data', (chunk) => {
          downloaded += chunk.length;
          const now = Date.now();
          // Emit progress at most every 500ms to avoid flooding
          if (totalSize > 0 && (now - lastEmit > 500 || downloaded === totalSize)) {
            lastEmit = now;
            onProgress(downloaded / totalSize, downloaded, totalSize);
          }
        });

        res.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
        file.on('error', reject);
      }).on('error', reject);
    };

    makeRequest(url);
  });
}

/**
 * Extract gzipped tar file
 * @param {string} gzPath - Path to .tar.gz file
 * @param {string} destDir - Destination directory
 * @returns {Promise<string[]>} List of extracted files
 */
async function extractTarGz(gzPath, destDir) {
  const tar = require('tar');
  await tar.extract({ file: gzPath, cwd: destDir });

  // List extracted files
  const files = [];
  const entries = await fs.readdir(destDir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith('.xyz')) {
      files.push(path.join(destDir, entry.name));
    }
  }
  return files;
}

/**
 * Parse QM9 XYZ file content
 * @param {string} content - XYZ file content
 * @returns {Object} Parsed molecule data
 */
function parseQM9XYZ(content) {
  const lines = content.trim().split('\n');

  if (lines.length < 3) {
    throw new Error('Invalid QM9 XYZ file format');
  }

  const numAtoms = parseInt(lines[0].trim());
  const propertyLine = lines[1].trim();

  // Parse property values from comment line
  // Format: gdb tag idx A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv
  const propValues = propertyLine.split(/\s+/);

  // Parse atoms
  const atoms = [];
  for (let i = 2; i < 2 + numAtoms && i < lines.length; i++) {
    const parts = lines[i].trim().split(/\s+/);
    if (parts.length >= 4) {
      atoms.push({
        element: parts[0],
        x: parseFloat(parts[1]),
        y: parseFloat(parts[2]),
        z: parseFloat(parts[3])
      });
    }
  }

  // Parse SMILES and InChI if present (last two lines)
  let smiles = null;
  let inchi = null;

  if (lines.length > 2 + numAtoms) {
    const smilesLine = lines[2 + numAtoms];
    if (smilesLine) {
      const smilesParts = smilesLine.trim().split(/\s+/);
      smiles = smilesParts[0];
    }
  }

  if (lines.length > 3 + numAtoms) {
    const inchiLine = lines[3 + numAtoms];
    if (inchiLine && inchiLine.startsWith('InChI')) {
      inchi = inchiLine.trim();
    }
  }

  // Build properties array
  const properties = [];
  for (const [key, info] of Object.entries(PROPERTY_INFO)) {
    const idx = QM9_PROPERTIES[key];
    if (idx !== undefined && propValues[idx] !== undefined) {
      const value = parseFloat(propValues[idx]);
      if (!isNaN(value)) {
        properties.push({
          name: info.name,
          value,
          unit: info.unit,
          source: 'qm9',
          method: 'B3LYP/6-31G(2df,p)'
        });
      }
    }
  }

  // Calculate molecular formula from atoms
  const elementCounts = {};
  for (const atom of atoms) {
    elementCounts[atom.element] = (elementCounts[atom.element] || 0) + 1;
  }

  const formula = Object.entries(elementCounts)
    .sort(([a], [b]) => {
      // Hill system: C first, then H, then alphabetical
      if (a === 'C') return -1;
      if (b === 'C') return 1;
      if (a === 'H') return -1;
      if (b === 'H') return 1;
      return a.localeCompare(b);
    })
    .map(([el, count]) => count === 1 ? el : `${el}${count}`)
    .join('');

  const idx = parseInt(propValues[QM9_PROPERTIES.idx]) || 0;

  return {
    identifier: `QM9:${idx}`,
    qm9_idx: idx,
    name: `QM9 molecule ${idx}`,
    formula,
    smiles,
    inchi,
    charge: 0,
    multiplicity: 1,
    sources: ['qm9'],
    properties,
    geometry: {
      atoms,
      unit: 'angstrom',
      source: 'qm9_dft'
    }
  };
}

/**
 * Get QM9 molecule by index
 * @param {number} idx - QM9 index (1-133885)
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} Molecule data
 */
async function getMolecule(idx, options = {}) {
  const { workspaceId = '1', emit = () => {} } = options;

  // Check if already cached
  const cached = await downloadManager.getCached(`QM9:${idx}`, ['qm9']);
  if (cached) {
    emit('stdout', `Found cached QM9 molecule ${idx}\n`);
    return cached;
  }

  emit('stdout', `Loading QM9 molecule ${idx}...\n`);

  // Check if dataset is downloaded
  const qm9Dir = downloadManager.getDownloadPath(workspaceId, 'qm9', 'dataset');
  const xyzPath = path.join(qm9Dir, `dsgdb9nsd_${String(idx).padStart(6, '0')}.xyz`);

  let content;
  try {
    content = await fs.readFile(xyzPath, 'utf8');
  } catch {
    throw new Error(`QM9 dataset not downloaded. Run benchmark_sync to download.`);
  }

  const moleculeData = parseQM9XYZ(content);
  moleculeData.file_paths = { geometry: xyzPath };

  // Index to Elasticsearch
  await downloadManager.indexMolecule(moleculeData);

  return moleculeData;
}

/**
 * Search QM9 dataset by formula or SMILES
 * @param {string} query - Search query
 * @param {Object} options - Search options
 * @returns {Promise<Object[]>} Matching molecules
 */
async function searchQM9(query, options = {}) {
  // Use Elasticsearch to search indexed QM9 molecules
  return downloadManager.search(query, { ...options, sources: ['qm9'] });
}

/**
 * Download and index entire QM9 dataset
 * @param {Object} options - Download options
 * @returns {Promise<Object>} Download result
 */
async function downloadDataset(options = {}) {
  const { workspaceId = '1', emit = () => {}, limit = null } = options;

  const qm9Dir = downloadManager.getDownloadPath(workspaceId, 'qm9', 'dataset');
  await downloadManager.ensureDir(qm9Dir);

  const tarPath = path.join(qm9Dir, 'dsgdb9nsd.xyz.tar.bz2');
  const markerPath = path.join(qm9Dir, '.downloaded');

  // Check if already downloaded
  try {
    await fs.access(markerPath);
    emit('stdout', 'QM9 dataset already downloaded\n');
    emit('progress', 50);
  } catch {
    emit('stdout', 'Downloading QM9 dataset from figshare (~650 MB)...\n');
    emit('progress', 5);

    await downloadFile(QM9_URLS.dataset, tarPath, (progress, downloaded, total) => {
      const pct = 5 + Math.round(progress * 35);
      emit('progress', pct);
      emit('download_progress', {
        phase: 'download',
        progress: pct,
        downloaded: formatBytes(downloaded),
        total: formatBytes(total),
        percent: Math.round(progress * 100)
      });
      emit('stdout', `Downloading: ${formatBytes(downloaded)} / ${formatBytes(total)} (${Math.round(progress * 100)}%)\n`);
    });

    emit('stdout', '\nExtracting dataset (this may take a few minutes)...\n');
    emit('progress', 45);
    emit('download_progress', { phase: 'extracting', progress: 45 });

    // Extract using bz2 and tar
    const { execSync } = require('child_process');
    execSync(`tar -xjf "${tarPath}" -C "${qm9Dir}"`, { stdio: 'pipe' });

    // Create marker file
    await fs.writeFile(markerPath, new Date().toISOString());
    emit('stdout', 'Dataset extracted successfully\n');
    emit('progress', 50);
  }

  // Check how many files we have and how many are already indexed
  emit('stdout', 'Checking index integrity...\n');
  emit('download_progress', { phase: 'checking', progress: 50 });

  const files = await fs.readdir(qm9Dir);
  const xyzFiles = files.filter(f => f.endsWith('.xyz')).sort();
  const toProcess = limit ? xyzFiles.slice(0, limit) : xyzFiles;
  const totalFiles = toProcess.length;

  // Quick integrity check: count indexed QM9 molecules
  const indexedCount = await downloadManager.countBySource('qm9');

  // If we have at least 99% of molecules indexed, consider it complete
  // (allows for a few parsing errors in the original dataset)
  const threshold = Math.floor(totalFiles * 0.99);

  if (indexedCount >= threshold) {
    emit('progress', 100);
    emit('download_progress', { phase: 'complete', progress: 100, indexed: indexedCount, total: totalFiles, skipped: true });
    emit('stdout', `Index integrity check passed: ${indexedCount}/${totalFiles} molecules already indexed\n`);

    return {
      total: totalFiles,
      indexed: indexedCount,
      path: qm9Dir,
      skipped: true
    };
  }

  // Need to index - either fresh or partial
  const needToIndex = indexedCount === 0 ? 'full' : 'partial';
  emit('stdout', `${needToIndex === 'full' ? 'Indexing' : 'Re-indexing'} molecules to Elasticsearch (${indexedCount}/${totalFiles} currently indexed)...\n`);
  emit('download_progress', { phase: 'indexing', progress: 50 });

  let indexed = 0;
  let errors = 0;
  let skipped = 0;
  let lastEmitTime = 0;

  for (let i = 0; i < toProcess.length; i++) {
    const filename = toProcess[i];
    const filePath = path.join(qm9Dir, filename);

    try {
      const content = await fs.readFile(filePath, 'utf8');
      const moleculeData = parseQM9XYZ(content);
      moleculeData.file_paths = { geometry: filePath };

      await downloadManager.indexMolecule(moleculeData);
      indexed++;
    } catch (error) {
      errors++;
      console.error(`Error indexing ${filename}:`, error.message);
    }

    const now = Date.now();
    // Emit progress every 500ms or every 1000 molecules
    if (now - lastEmitTime > 500 || i % 1000 === 0 || i === toProcess.length - 1) {
      lastEmitTime = now;
      const pct = 50 + Math.round((i / totalFiles) * 48);
      emit('progress', pct);
      emit('download_progress', {
        phase: 'indexing',
        progress: pct,
        indexed,
        total: totalFiles,
        errors,
        percent: Math.round((i / totalFiles) * 100)
      });
      emit('stdout', `Indexing: ${indexed}/${totalFiles} molecules (${Math.round((i / totalFiles) * 100)}%)\n`);
    }
  }

  emit('progress', 100);
  emit('download_progress', { phase: 'complete', progress: 100, indexed, total: totalFiles, errors });
  emit('stdout', `\nFinished indexing ${indexed} QM9 molecules (${errors} errors)\n`);

  return {
    total: xyzFiles.length,
    indexed,
    path: qm9Dir
  };
}

/**
 * Get QM9 statistics
 * @param {string} workspaceId - Workspace ID
 * @returns {Promise<Object>} Dataset statistics
 */
async function getStatistics(workspaceId = '1') {
  const qm9Dir = downloadManager.getDownloadPath(workspaceId, 'qm9', 'dataset');

  try {
    await fs.access(path.join(qm9Dir, '.downloaded'));

    const files = await fs.readdir(qm9Dir);
    const xyzCount = files.filter(f => f.endsWith('.xyz')).length;

    // Get indexed count from Elasticsearch
    const indexed = await downloadManager.listCached('qm9');

    return {
      downloaded: true,
      total_files: xyzCount,
      indexed_count: indexed.length,
      path: qm9Dir
    };
  } catch {
    return {
      downloaded: false,
      total_files: 0,
      indexed_count: 0,
      path: null
    };
  }
}

/**
 * Fetch molecule data (wrapper for unified interface)
 * @param {string} identifier - QM9 index or search query
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} Molecule data
 */
async function fetchMolecule(identifier, options = {}) {
  const { emit = () => {} } = options;

  // Check if it's a QM9 index
  const qm9Match = identifier.match(/^(?:QM9:|qm9:)?(\d+)$/i);
  if (qm9Match) {
    const idx = parseInt(qm9Match[1]);
    return getMolecule(idx, options);
  }

  // Otherwise search
  emit('stdout', `Searching QM9 for: ${identifier}\n`);
  const results = await searchQM9(identifier, { limit: 1 });

  if (results.length === 0) {
    throw new Error(`No QM9 results for: ${identifier}`);
  }

  return results[0];
}

module.exports = {
  QM9_PROPERTIES,
  PROPERTY_INFO,
  parseQM9XYZ,
  getMolecule,
  searchQM9,
  downloadDataset,
  getStatistics,
  fetchMolecule
};
