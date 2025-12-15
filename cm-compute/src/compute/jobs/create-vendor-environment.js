/**
 * Create Vendor Environment Job Handler
 *
 * Creates a new vendor environment by cloning a git repository,
 * building it from source, and installing to an isolated prefix.
 * Supports cmake, make, and autotools build systems.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const database = require('../../database');

const VENDOR_BASE = '/opt/vendor';
const VENDOR_SRC = '/opt/vendor-src';

// Track if database is initialized in this worker
let dbInitialized = false;

/**
 * Ensure database is initialized in the worker process
 */
async function ensureDatabase() {
  if (!dbInitialized) {
    await database.initialize();
    dbInitialized = true;
  }
  return database.getModels();
}

/**
 * Create a new vendor environment from a git repository
 * @param {Object} params - Job parameters
 * @param {string} params.name - Environment name
 * @param {string} params.description - Environment description
 * @param {string} params.repo - Git repository URL
 * @param {string} params.branch - Git branch (default: main)
 * @param {string} params.buildType - Build system type (cmake, make, autotools)
 * @param {string} params.cmakeOptions - Additional CMake options
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Creation result
 */
async function createVendorEnvironment(params, context) {
  const { emit } = context;
  const { name, description = '', repo, branch = 'main', buildType = 'cmake', cmakeOptions = '' } = params;

  if (!name) {
    throw new Error('Environment name is required');
  }

  if (!repo) {
    throw new Error('Repository URL is required');
  }

  // Validate environment name
  if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(name)) {
    throw new Error('Invalid environment name. Use alphanumeric characters, hyphens, and underscores.');
  }

  const installPrefix = path.join(VENDOR_BASE, name);
  const srcDir = path.join(VENDOR_SRC, name);

  emit('stdout', `Creating vendor environment '${name}' from ${repo}...\n`);
  emit('stdout', `Branch: ${branch}\n`);
  emit('stdout', `Build type: ${buildType}\n`);
  emit('stdout', `Install prefix: ${installPrefix}\n`);
  emit('stdout', '─'.repeat(50) + '\n\n');

  // Initialize database in worker and get models
  const { VendorEnvironment } = await ensureDatabase();

  // Check if environment already exists
  const existing = await VendorEnvironment.findByName(name);
  if (existing) {
    throw new Error(`Vendor environment '${name}' already exists`);
  }

  // Ensure directories exist
  if (!fs.existsSync(VENDOR_BASE)) {
    fs.mkdirSync(VENDOR_BASE, { recursive: true });
  }
  if (!fs.existsSync(VENDOR_SRC)) {
    fs.mkdirSync(VENDOR_SRC, { recursive: true });
  }

  // Clean up any existing source directory
  if (fs.existsSync(srcDir)) {
    emit('stdout', `Cleaning existing source directory...\n`);
    fs.rmSync(srcDir, { recursive: true });
  }

  // Step 1: Clone repository
  emit('stdout', `Cloning repository...\n`);
  emit('progress', 10);
  await runCommand('git', ['clone', '--depth', '1', '--branch', branch, repo, srcDir], emit);

  // Step 2: Detect and run build system
  emit('stdout', `\nBuilding from source...\n`);
  emit('progress', 30);

  const detectedBuildType = detectBuildSystem(srcDir, buildType);
  emit('stdout', `Detected build system: ${detectedBuildType}\n\n`);

  switch (detectedBuildType) {
    case 'cmake':
      await buildWithCMake(srcDir, installPrefix, cmakeOptions, emit);
      break;
    case 'make':
      await buildWithMake(srcDir, installPrefix, emit);
      break;
    case 'autotools':
      await buildWithAutotools(srcDir, installPrefix, emit);
      break;
    default:
      throw new Error(`Unknown build type: ${detectedBuildType}`);
  }

  emit('progress', 90);

  // Create environment record in database
  const installation = {
    repo,
    branch,
    build_type: detectedBuildType,
    install_prefix: installPrefix,
    cmake_options: cmakeOptions,
    installed_at: new Date().toISOString()
  };

  const env = await VendorEnvironment.create({
    name,
    description,
    installations: [installation]
  });

  emit('stdout', '\n' + '─'.repeat(50) + '\n');
  emit('stdout', `Vendor environment '${name}' created successfully.\n`);
  emit('stdout', `Libraries installed to: ${installPrefix}\n`);
  emit('progress', 100);

  return {
    status: 'created',
    id: env.id,
    name,
    description,
    installation
  };
}

/**
 * Detect the build system used by the project
 */
function detectBuildSystem(srcDir, hint) {
  if (hint && hint !== 'auto') {
    return hint;
  }

  // Check for CMakeLists.txt
  if (fs.existsSync(path.join(srcDir, 'CMakeLists.txt'))) {
    return 'cmake';
  }

  // Check for configure script (autotools)
  if (fs.existsSync(path.join(srcDir, 'configure'))) {
    return 'autotools';
  }

  // Check for configure.ac (needs autoreconf)
  if (fs.existsSync(path.join(srcDir, 'configure.ac'))) {
    return 'autotools';
  }

  // Check for Makefile
  if (fs.existsSync(path.join(srcDir, 'Makefile'))) {
    return 'make';
  }

  // Default to cmake
  return 'cmake';
}

/**
 * Build with CMake
 */
async function buildWithCMake(srcDir, installPrefix, cmakeOptions, emit) {
  const buildDir = path.join(srcDir, 'build');
  fs.mkdirSync(buildDir, { recursive: true });

  // Configure
  emit('stdout', 'Configuring with CMake...\n');
  const cmakeArgs = [
    '..',
    `-DCMAKE_INSTALL_PREFIX=${installPrefix}`,
    '-DCMAKE_BUILD_TYPE=Release'
  ];

  // Add user-specified options
  if (cmakeOptions) {
    cmakeArgs.push(...cmakeOptions.split(' ').filter(o => o.length > 0));
  }

  await runCommand('cmake', cmakeArgs, emit, buildDir);

  // Build
  emit('stdout', '\nBuilding...\n');
  const cpuCount = require('os').cpus().length;
  await runCommand('make', ['-j', String(cpuCount)], emit, buildDir);

  // Install
  emit('stdout', '\nInstalling...\n');
  await runCommand('make', ['install'], emit, buildDir);
}

/**
 * Build with Make
 */
async function buildWithMake(srcDir, installPrefix, emit) {
  emit('stdout', 'Building with Make...\n');
  const cpuCount = require('os').cpus().length;

  await runCommand('make', ['-j', String(cpuCount), `PREFIX=${installPrefix}`], emit, srcDir);

  emit('stdout', '\nInstalling...\n');
  await runCommand('make', ['install', `PREFIX=${installPrefix}`], emit, srcDir);
}

/**
 * Build with Autotools
 */
async function buildWithAutotools(srcDir, installPrefix, emit) {
  // Run autoreconf if configure doesn't exist
  if (!fs.existsSync(path.join(srcDir, 'configure'))) {
    emit('stdout', 'Running autoreconf...\n');
    await runCommand('autoreconf', ['-fi'], emit, srcDir);
  }

  emit('stdout', 'Configuring...\n');
  await runCommand('./configure', [`--prefix=${installPrefix}`], emit, srcDir);

  emit('stdout', '\nBuilding...\n');
  const cpuCount = require('os').cpus().length;
  await runCommand('make', ['-j', String(cpuCount)], emit, srcDir);

  emit('stdout', '\nInstalling...\n');
  await runCommand('make', ['install'], emit, srcDir);
}

/**
 * Run a shell command and stream output
 */
function runCommand(command, args, emit, cwd = undefined) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        DEBIAN_FRONTEND: 'noninteractive'
      }
    });

    proc.stdout.on('data', (data) => {
      emit('stdout', data.toString());
    });

    proc.stderr.on('data', (data) => {
      const output = data.toString();
      // CMake and make output warnings to stderr
      if (output.toLowerCase().includes('error')) {
        emit('stderr', output);
      } else {
        emit('stdout', output);
      }
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command '${command} ${args.join(' ')}' failed with exit code ${code}`));
      }
    });

    proc.on('error', (error) => {
      reject(error);
    });
  });
}

module.exports = createVendorEnvironment;
