/**
 * Shared utility functions for compute service
 */

const fs = require('fs');

const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';

/**
 * Get Python path for a conda environment
 * @param {string} envName - Environment name
 * @returns {string} Path to Python executable
 */
function getPythonPath(envName) {
  if (!envName || envName === 'base') {
    return `${CONDA_PATH}/bin/python`;
  }
  const envPython = `${CONDA_PATH}/envs/${envName}/bin/python`;
  if (fs.existsSync(envPython)) {
    return envPython;
  }
  // Fallback to base if environment doesn't exist
  console.warn(`Environment '${envName}' not found, using base`);
  return `${CONDA_PATH}/bin/python`;
}

module.exports = { getPythonPath, CONDA_PATH };
