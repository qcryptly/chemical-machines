/**
 * Job Registry - Maps job types to their handler functions
 *
 * Each job handler receives (params, context) and returns a result.
 * Context includes: pythonPath, jobId, emit (for progress/output)
 */

const molecularDynamics = require('./molecular-dynamics');
const docking = require('./docking');
const bindingAffinity = require('./binding-affinity');
const structurePrediction = require('./structure-prediction');
const optimization = require('./optimization');
const execute = require('./execute');
const createCondaEnvironment = require('./create-conda-environment');
const createCppEnvironment = require('./create-cpp-environment');
const createVendorEnvironment = require('./create-vendor-environment');
const executeCpp = require('./execute-cpp');
const installCondaPackage = require('./install-conda-package');
const removeCondaPackage = require('./remove-conda-package');

// Benchmark database handlers
const {
  benchmarkFetch,
  benchmarkIndex,
  benchmarkSync,
  benchmarkSearch,
  benchmarkCompare
} = require('./benchmark-fetch');

// Registry of job types to handler functions
const jobRegistry = new Map();

/**
 * Register a job handler
 * @param {string} type - Job type name
 * @param {Function} handler - Handler function (params, context) => result
 */
function registerJob(type, handler) {
  if (typeof handler !== 'function') {
    throw new Error(`Handler for job type '${type}' must be a function`);
  }
  jobRegistry.set(type, handler);
}

/**
 * Get handler for a job type
 * @param {string} type - Job type name
 * @returns {Function|null} Handler function or null if not found
 */
function getJobHandler(type) {
  return jobRegistry.get(type) || null;
}

/**
 * Check if a job type is registered
 * @param {string} type - Job type name
 * @returns {boolean}
 */
function hasJobType(type) {
  return jobRegistry.has(type);
}

/**
 * Get all registered job types
 * @returns {string[]}
 */
function getJobTypes() {
  return Array.from(jobRegistry.keys());
}

// Register built-in job types
registerJob('molecular-dynamics', molecularDynamics);
registerJob('docking', docking);
registerJob('binding-affinity', bindingAffinity);
registerJob('structure-prediction', structurePrediction);
registerJob('optimization', optimization);
registerJob('execute', execute);
registerJob('create_environment', createCondaEnvironment);
registerJob('create_cpp_environment', createCppEnvironment);
registerJob('create_vendor_environment', createVendorEnvironment);
registerJob('execute_cpp', executeCpp);
registerJob('install_conda_package', installCondaPackage);
registerJob('remove_conda_package', removeCondaPackage);

// Benchmark database jobs
registerJob('benchmark_fetch', benchmarkFetch);
registerJob('benchmark_index', benchmarkIndex);
registerJob('benchmark_sync', benchmarkSync);
registerJob('benchmark_search', benchmarkSearch);
registerJob('benchmark_compare', benchmarkCompare);

module.exports = {
  jobRegistry,
  registerJob,
  getJobHandler,
  hasJobType,
  getJobTypes
};
