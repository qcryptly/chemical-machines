/**
 * Remove Conda Package Job Handler
 *
 * Removes a package from a conda environment.
 * Streams output during removal for real-time progress updates.
 */

const { spawn } = require('child_process');

const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';

/**
 * Remove a package from a conda environment
 * @param {Object} params - Job parameters
 * @param {string} params.envName - Environment name
 * @param {string} params.packageName - Package to remove
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Removal result
 */
async function removeCondaPackage(params, context) {
  const { emit } = context;
  const { envName, packageName } = params;

  if (!packageName) {
    throw new Error('No package specified');
  }

  emit('stdout', `Removing package '${packageName}' from '${envName}'...\n`);
  emit('stdout', '\n' + '─'.repeat(50) + '\n');

  // Remove the package
  await runCondaCommand(['remove', '-n', envName, '-y', packageName], emit);

  emit('stdout', '\n' + '─'.repeat(50) + '\n');
  emit('stdout', `\nPackage '${packageName}' removed from '${envName}'.\n`);

  return {
    status: 'removed',
    envName,
    removedPackage: packageName
  };
}

/**
 * Run a conda command and stream output
 * @param {string[]} args - Command arguments
 * @param {Function} emit - Emit function for streaming output
 * @returns {Promise<void>}
 */
function runCondaCommand(args, emit) {
  return new Promise((resolve, reject) => {
    const conda = spawn(`${CONDA_PATH}/bin/conda`, args, {
      env: {
        ...process.env,
        CONDA_YES: '1'
      }
    });

    conda.stdout.on('data', (data) => {
      const output = data.toString();
      emit('stdout', output);

      // Parse progress from conda output
      const progressMatch = output.match(/(\d+)%/);
      if (progressMatch) {
        const progress = parseInt(progressMatch[1], 10);
        emit('progress', progress);
      }
    });

    conda.stderr.on('data', (data) => {
      const output = data.toString();
      emit('stderr', output);
    });

    conda.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Conda command failed with exit code ${code}`));
      }
    });

    conda.on('error', (error) => {
      reject(error);
    });
  });
}

module.exports = removeCondaPackage;
