/**
 * Install Conda Package Job Handler
 *
 * Installs packages into a conda environment.
 * Streams output during installation for real-time progress updates.
 */

const { spawn } = require('child_process');

const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';

/**
 * Install packages into a conda environment
 * @param {Object} params - Job parameters
 * @param {string} params.envName - Environment name
 * @param {string[]} params.packages - Packages to install
 * @param {string} [params.channel] - Optional conda channel
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Installation result
 */
async function installCondaPackage(params, context) {
  const { emit } = context;
  const { envName, packages, channel } = params;

  if (!packages || !Array.isArray(packages) || packages.length === 0) {
    throw new Error('No packages specified');
  }

  emit('stdout', `Installing packages in '${envName}': ${packages.join(', ')}...\n`);
  emit('stdout', '\n' + '─'.repeat(50) + '\n');

  // Build conda install command
  const args = ['install', '-n', envName, '-y', ...packages];
  if (channel) {
    args.splice(1, 0, '-c', channel);
  }

  // Install the packages
  await runCondaCommand(args, emit);

  emit('stdout', '\n' + '─'.repeat(50) + '\n');
  emit('stdout', `\nPackages installed successfully in '${envName}'.\n`);

  return {
    status: 'installed',
    envName,
    installedPackages: packages
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

module.exports = installCondaPackage;
