/**
 * Install C++ Packages Job Handler
 *
 * Installs additional debian dev packages into an existing C++ environment.
 * Streams output during installation for real-time progress updates.
 */

const { spawn } = require('child_process');
const database = require('../../database');

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
 * Install packages into an existing C++ environment
 * @param {Object} params - Job parameters
 * @param {string} params.envName - Environment name
 * @param {number} params.envId - Environment ID
 * @param {string[]} params.packages - Debian dev packages to install
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Installation result
 */
async function installCppPackages(params, context) {
  const { emit } = context;
  const { envName, envId, packages } = params;

  if (!packages || packages.length === 0) {
    throw new Error('No packages specified');
  }

  emit('stdout', `Installing packages in '${envName}'...\n`);
  emit('stdout', `Packages: ${packages.join(', ')}\n`);

  // Initialize database in worker and get models
  const { CppEnvironment } = await ensureDatabase();

  // Verify environment exists
  const env = await CppEnvironment.findById(envId);
  if (!env) {
    throw new Error(`Environment '${envName}' not found`);
  }

  emit('stdout', '\n' + '─'.repeat(50) + '\n');

  // Update apt cache first
  emit('stdout', 'Updating package cache...\n');
  await runCommand('apt-get', ['update'], emit);

  // Install packages
  emit('stdout', `\nInstalling ${packages.length} package(s)...\n`);
  await runCommand('apt-get', ['install', '-y', ...packages], emit);

  emit('stdout', '\n' + '─'.repeat(50) + '\n');

  // Update environment record with new packages
  const existingPackages = env.packages || [];
  const newPackages = [...new Set([...existingPackages, ...packages])];
  await CppEnvironment.addPackages(envId, packages);

  emit('stdout', `\nPackages installed successfully in '${envName}'.\n`);
  emit('stdout', `Total packages: ${newPackages.length}\n`);

  return {
    status: 'installed',
    envName,
    installedPackages: packages,
    totalPackages: newPackages.length
  };
}

/**
 * Run a shell command and stream output
 * @param {string} command - Command to run
 * @param {string[]} args - Command arguments
 * @param {Function} emit - Emit function for streaming output
 * @returns {Promise<void>}
 */
function runCommand(command, args, emit) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      env: {
        ...process.env,
        DEBIAN_FRONTEND: 'noninteractive'
      }
    });

    proc.stdout.on('data', (data) => {
      const output = data.toString();
      emit('stdout', output);

      // Parse progress from apt output
      const progressMatch = output.match(/(\d+)%/);
      if (progressMatch) {
        const progress = parseInt(progressMatch[1], 10);
        emit('progress', progress);
      }
    });

    proc.stderr.on('data', (data) => {
      const output = data.toString();
      // apt-get outputs progress to stderr, so we'll treat it as stdout
      // unless it looks like an actual error
      if (output.includes('E:') || output.includes('error:')) {
        emit('stderr', output);
      } else {
        emit('stdout', output);
      }
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command '${command}' failed with exit code ${code}`));
      }
    });

    proc.on('error', (error) => {
      reject(error);
    });
  });
}

module.exports = installCppPackages;
