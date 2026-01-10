/**
 * PubChem API Client
 *
 * Fetches molecular data from PubChem REST API
 * https://pubchem.ncbi.nlm.nih.gov/
 *
 * Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
 */

const https = require('https');
const downloadManager = require('./download-manager');

const PUBCHEM_BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug';

/**
 * Make HTTPS request with JSON response
 * @param {string} url - URL to fetch
 * @param {Object} options - Request options
 * @returns {Promise<Object|string>} Response data
 */
function fetch(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: {
        'Accept': options.accept || 'application/json',
        'User-Agent': 'ChemicalMachines/1.0 (Computational Chemistry Research)'
      },
      timeout: options.timeout || 30000
    }, (res) => {
      if (res.statusCode !== 200) {
        let body = '';
        res.on('data', chunk => body += chunk);
        res.on('end', () => {
          reject(new Error(`PubChem API error ${res.statusCode}: ${body}`));
        });
        return;
      }

      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        if (options.accept === 'text/plain' || options.raw) {
          resolve(data);
        } else {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            resolve(data);
          }
        }
      });
    });

    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });
  });
}

/**
 * Search PubChem for compounds
 * @param {string} query - Search query (name, formula, SMILES, etc.)
 * @param {string} searchType - Type of search (name, formula, smiles, inchi)
 * @returns {Promise<Object[]>} Search results with CIDs
 */
async function searchPubChem(query, searchType = 'name') {
  let url;

  switch (searchType) {
    case 'formula':
      url = `${PUBCHEM_BASE}/compound/fastformula/${encodeURIComponent(query)}/cids/JSON`;
      break;
    case 'smiles':
      url = `${PUBCHEM_BASE}/compound/smiles/${encodeURIComponent(query)}/cids/JSON`;
      break;
    case 'inchi':
      url = `${PUBCHEM_BASE}/compound/inchi/${encodeURIComponent(query)}/cids/JSON`;
      break;
    case 'inchikey':
      url = `${PUBCHEM_BASE}/compound/inchikey/${encodeURIComponent(query)}/cids/JSON`;
      break;
    case 'name':
    default:
      url = `${PUBCHEM_BASE}/compound/name/${encodeURIComponent(query)}/cids/JSON`;
      break;
  }

  try {
    const data = await fetch(url);
    const cids = data.IdentifierList?.CID || [];

    // Fetch basic info for each CID (limit to first 10)
    const results = [];
    for (const cid of cids.slice(0, 10)) {
      try {
        const props = await fetchProperties(cid, ['MolecularFormula', 'IUPACName']);
        results.push({
          cid,
          name: props.IUPACName || `CID ${cid}`,
          formula: props.MolecularFormula,
          source: 'pubchem'
        });
      } catch {
        results.push({ cid, source: 'pubchem' });
      }
    }

    return results;
  } catch (error) {
    console.error(`PubChem search failed for "${query}":`, error.message);
    return [];
  }
}

/**
 * Fetch compound properties from PubChem
 * @param {number|string} cid - PubChem Compound ID
 * @param {string[]} propertyList - List of properties to fetch
 * @returns {Promise<Object>} Properties object
 */
async function fetchProperties(cid, propertyList = null) {
  const defaultProperties = [
    'MolecularFormula',
    'MolecularWeight',
    'IUPACName',
    'InChI',
    'InChIKey',
    'CanonicalSMILES',
    'IsomericSMILES',
    'Charge',
    'XLogP',
    'TPSA',
    'HBondDonorCount',
    'HBondAcceptorCount',
    'RotatableBondCount',
    'HeavyAtomCount',
    'AtomStereoCount',
    'BondStereoCount',
    'Complexity',
    'MonoisotopicMass',
    'ExactMass'
  ];

  const props = propertyList || defaultProperties;
  const url = `${PUBCHEM_BASE}/compound/cid/${cid}/property/${props.join(',')}/JSON`;

  const data = await fetch(url);
  const properties = data.PropertyTable?.Properties?.[0] || {};

  return properties;
}

/**
 * Fetch 3D conformer structure from PubChem
 * @param {number|string} cid - PubChem Compound ID
 * @returns {Promise<Object|null>} 3D structure data
 */
async function fetch3DStructure(cid) {
  const url = `${PUBCHEM_BASE}/compound/cid/${cid}/record/JSON?record_type=3d`;

  try {
    const data = await fetch(url);
    const record = data.PC_Compounds?.[0];

    if (!record || !record.coords?.[0]?.conformers?.[0]) {
      return null;
    }

    const conformer = record.coords[0].conformers[0];
    const atoms = record.atoms;

    // Build atom list with coordinates
    const geometry = {
      atoms: [],
      unit: 'angstrom',
      source: 'pubchem_3d'
    };

    // Element lookup from atomic numbers
    const elementSymbols = [
      '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
      'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
      'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
      'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
      'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
      'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba'
    ];

    for (let i = 0; i < atoms.element.length; i++) {
      const atomicNum = atoms.element[i];
      geometry.atoms.push({
        element: elementSymbols[atomicNum] || `X${atomicNum}`,
        x: conformer.x[i],
        y: conformer.y[i],
        z: conformer.z?.[i] || 0
      });
    }

    return geometry;
  } catch (error) {
    console.error(`Failed to fetch 3D structure for CID ${cid}:`, error.message);
    return null;
  }
}

/**
 * Fetch SDF file from PubChem
 * @param {number|string} cid - PubChem Compound ID
 * @param {boolean} threeD - Fetch 3D structure
 * @returns {Promise<string|null>} SDF content
 */
async function fetchSDF(cid, threeD = true) {
  const recordType = threeD ? '3d' : '2d';
  const url = `${PUBCHEM_BASE}/compound/cid/${cid}/SDF?record_type=${recordType}`;

  try {
    const sdf = await fetch(url, { accept: 'text/plain', raw: true });
    return sdf;
  } catch (error) {
    // Try 2D if 3D fails
    if (threeD) {
      return fetchSDF(cid, false);
    }
    console.error(`Failed to fetch SDF for CID ${cid}:`, error.message);
    return null;
  }
}

/**
 * Fetch computed properties from PubChem
 * @param {number|string} cid - PubChem Compound ID
 * @returns {Promise<Object[]>} Array of property objects
 */
async function fetchComputedProperties(cid) {
  const props = await fetchProperties(cid);
  const properties = [];

  // Map PubChem properties to our format
  const propertyMap = {
    MolecularWeight: { name: 'molecular_weight', unit: 'g/mol' },
    MonoisotopicMass: { name: 'monoisotopic_mass', unit: 'Da' },
    ExactMass: { name: 'exact_mass', unit: 'Da' },
    XLogP: { name: 'logp', unit: '' },
    TPSA: { name: 'tpsa', unit: 'angstrom^2' },
    Complexity: { name: 'complexity', unit: '' },
    Charge: { name: 'formal_charge', unit: '' },
    HBondDonorCount: { name: 'hbd_count', unit: '' },
    HBondAcceptorCount: { name: 'hba_count', unit: '' },
    RotatableBondCount: { name: 'rotatable_bonds', unit: '' },
    HeavyAtomCount: { name: 'heavy_atom_count', unit: '' }
  };

  for (const [pubchemKey, mapping] of Object.entries(propertyMap)) {
    if (props[pubchemKey] !== undefined && props[pubchemKey] !== null) {
      properties.push({
        name: mapping.name,
        value: typeof props[pubchemKey] === 'number' ? props[pubchemKey] : parseFloat(props[pubchemKey]),
        unit: mapping.unit,
        source: 'pubchem',
        method: 'computed'
      });
    }
  }

  return properties;
}

/**
 * Fetch synonyms for a compound
 * @param {number|string} cid - PubChem Compound ID
 * @returns {Promise<string[]>} List of synonyms
 */
async function fetchSynonyms(cid) {
  const url = `${PUBCHEM_BASE}/compound/cid/${cid}/synonyms/JSON`;

  try {
    const data = await fetch(url);
    return data.InformationList?.Information?.[0]?.Synonym || [];
  } catch {
    return [];
  }
}

/**
 * Fetch complete molecule data from PubChem
 * @param {string} identifier - CID, name, SMILES, or InChIKey
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} Complete molecule data
 */
async function fetchMolecule(identifier, options = {}) {
  const { workspaceId = '1', emit = () => {} } = options;

  emit('stdout', `Fetching molecule data from PubChem for: ${identifier}\n`);
  emit('progress', 10);

  // Determine search type and find CID
  let cid;

  if (/^\d+$/.test(identifier)) {
    // Pure number = CID
    cid = parseInt(identifier);
  } else if (/^[A-Z]{14}-[A-Z]{10}-[A-Z]$/.test(identifier)) {
    // InChIKey format
    const results = await searchPubChem(identifier, 'inchikey');
    if (results.length === 0) {
      throw new Error(`No PubChem results for InChIKey: ${identifier}`);
    }
    cid = results[0].cid;
  } else if (identifier.startsWith('InChI=')) {
    // InChI
    const results = await searchPubChem(identifier, 'inchi');
    if (results.length === 0) {
      throw new Error(`No PubChem results for InChI: ${identifier}`);
    }
    cid = results[0].cid;
  } else if (/^[A-Za-z0-9@+\-\[\]\(\)\\\/=#$.]+$/.test(identifier) && identifier.includes('C')) {
    // Likely SMILES
    const results = await searchPubChem(identifier, 'smiles');
    if (results.length > 0) {
      cid = results[0].cid;
    }
  }

  // If still no CID, try name search
  if (!cid) {
    const results = await searchPubChem(identifier, 'name');
    if (results.length === 0) {
      throw new Error(`No PubChem results for: ${identifier}`);
    }
    cid = results[0].cid;
    emit('stdout', `Found: ${results[0].name} (CID: ${cid})\n`);
  }

  emit('progress', 30);

  // Fetch all data in parallel
  const [rawProps, geometry, sdf, synonyms] = await Promise.all([
    fetchProperties(cid),
    fetch3DStructure(cid),
    fetchSDF(cid),
    fetchSynonyms(cid)
  ]);

  emit('progress', 60);

  // Convert to our property format
  const properties = await fetchComputedProperties(cid);

  // Build molecule data object
  const moleculeData = {
    identifier: `CID:${cid}`,
    cid,
    name: rawProps.IUPACName || synonyms[0] || `CID ${cid}`,
    formula: rawProps.MolecularFormula,
    smiles: rawProps.CanonicalSMILES,
    inchi: rawProps.InChI,
    inchi_key: rawProps.InChIKey,
    molecular_weight: rawProps.MolecularWeight,
    charge: rawProps.Charge || 0,
    sources: ['pubchem'],
    properties,
    geometry,
    synonyms: synonyms.slice(0, 20), // Keep top 20 synonyms
    file_paths: {}
  };

  emit('progress', 70);

  // Save files to workspace
  if (sdf) {
    const sdfPath = await downloadManager.saveFile(
      workspaceId, 'pubchem', cid.toString(), 'structure.sdf', sdf
    );
    moleculeData.file_paths.sdf = sdfPath;
    emit('stdout', `Saved SDF structure to: ${sdfPath}\n`);
  }

  if (geometry) {
    const xyzContent = formatXYZ(geometry, moleculeData.name);
    const xyzPath = await downloadManager.saveFile(
      workspaceId, 'pubchem', cid.toString(), 'geometry.xyz', xyzContent
    );
    moleculeData.file_paths.geometry = xyzPath;
    emit('stdout', `Saved XYZ geometry to: ${xyzPath}\n`);
  }

  // Save all properties as JSON
  const propsPath = await downloadManager.saveFile(
    workspaceId, 'pubchem', cid.toString(), 'properties.json',
    JSON.stringify({ ...rawProps, computed: properties }, null, 2)
  );
  moleculeData.file_paths.properties = propsPath;

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
 * @param {string} name - Molecule name
 * @returns {string} XYZ file content
 */
function formatXYZ(geometry, name = 'Molecule') {
  const lines = [];
  lines.push(geometry.atoms.length.toString());
  lines.push(`${name} - Generated by ChemicalMachines from PubChem`);

  for (const atom of geometry.atoms) {
    lines.push(`${atom.element}  ${atom.x.toFixed(6)}  ${atom.y.toFixed(6)}  ${atom.z.toFixed(6)}`);
  }

  return lines.join('\n');
}

/**
 * Batch fetch multiple molecules
 * @param {string[]} identifiers - Array of identifiers
 * @param {Object} options - Fetch options
 * @returns {Promise<Object[]>} Array of molecule data
 */
async function fetchBatch(identifiers, options = {}) {
  const { emit = () => {}, concurrency = 3 } = options;
  const results = [];

  for (let i = 0; i < identifiers.length; i += concurrency) {
    const batch = identifiers.slice(i, i + concurrency);
    const batchResults = await Promise.allSettled(
      batch.map(id => fetchMolecule(id, { ...options, emit: () => {} }))
    );

    for (const result of batchResults) {
      if (result.status === 'fulfilled') {
        results.push(result.value);
      } else {
        console.error('Batch fetch error:', result.reason);
      }
    }

    emit('progress', Math.round(((i + batch.length) / identifiers.length) * 100));
  }

  return results;
}

module.exports = {
  searchPubChem,
  fetchProperties,
  fetch3DStructure,
  fetchSDF,
  fetchComputedProperties,
  fetchSynonyms,
  fetchMolecule,
  fetchBatch,
  formatXYZ
};
