/**
 * Remove C++ Package Job Handler
 *
 * Removes a debian dev package from an existing C++ environment.
 * Streams output during removal for real-time progress updates.
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
 * Remove a package from an existing C++ environment
 * @param {Object} params - Job parameters
 * @param {string} params.envName - Environment name
 * @param {number} params.envId - Environment ID
 * @param {string} params.packageName - Debian package to remove
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Removal result
 */
async function removeCppPackage(params, context) {
  const { emit } = context;
  const { envName, envId, packageName } = params;

  if (!packageName) {
    throw new Error('No package specified');
  }

  emit('stdout', `Removing package '${packageName}' from '${envName}'...\n`);

  // Initialize database in worker and get models
  const { CppEnvironment } = await ensureDatabase();

  // Verify environment exists
  const env = await CppEnvironment.findById(envId);
  if (!env) {
    throw new Error(`Environment '${envName}' not found`);
  }

  // Check if package is in the environment
  const existingPackages = env.packages || [];
  if (!existingPackages.includes(packageName)) {
    throw new Error(`Package '${packageName}' is not in environment '${envName}'`);
  }

  emit('stdout', '\n' + '─'.repeat(50) + '\n');

  // Remove the package
  emit('stdout', `Removing ${packageName}...\n`);
  await runCommand('apt-get', ['remove', '-y', packageName], emit);

  // Optionally autoremove unused dependencies
  emit('stdout', '\nCleaning up unused dependencies...\n');
  await runCommand('apt-get', ['autoremove', '-y'], emit);

  emit('stdout', '\n' + '─'.repeat(50) + '\n');

  // Update environment record - remove package from list
  const updatedPackages = existingPackages.filter(p => p !== packageName);
  await CppEnvironment.update(envId, { packages: updatedPackages });

  emit('stdout', `\nPackage '${packageName}' removed from '${envName}'.\n`);
  emit('stdout', `Remaining packages: ${updatedPackages.length}\n`);

  return {
    status: 'removed',
    envName,
    removedPackage: packageName,
    remainingPackages: updatedPackages.length
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
      // apt-get outputs progress to stderr
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

module.exports = removeCppPackage;
