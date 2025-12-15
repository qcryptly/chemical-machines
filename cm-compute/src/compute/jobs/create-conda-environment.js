/**
 * Create Conda Environment Job Handler
 *
 * Creates a new conda environment with specified Python version and packages.
 * Streams output during creation for real-time progress updates.
 */

const { spawn } = require('child_process');

const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';

/**
 * Create a new conda environment
 * @param {Object} params - Job parameters
 * @param {string} params.name - Environment name
 * @param {string} params.pythonVersion - Python version (e.g., '3.12')
 * @param {string[]} params.packages - Additional packages to install
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Creation result
 */
async function createCondaEnvironment(params, context) {
  const { emit } = context;
  const { name, pythonVersion = '3.12', packages = [] } = params;

  if (!name) {
    throw new Error('Environment name is required');
  }

  // Validate environment name
  if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(name)) {
    throw new Error('Invalid environment name. Use alphanumeric characters, hyphens, and underscores.');
  }

  emit('stdout', `Creating conda environment '${name}' with Python ${pythonVersion}...\n`);

  // Step 1: Create the environment
  await runCondaCommand(
    ['create', '-n', name, `python=${pythonVersion}`, '-y'],
    emit
  );

  emit('stdout', `\nEnvironment '${name}' created successfully.\n`);

  // Step 2: Install additional packages if specified
  if (packages.length > 0) {
    emit('stdout', `\nInstalling packages: ${packages.join(', ')}...\n`);

    await runCondaCommand(
      ['install', '-n', name, ...packages, '-y'],
      emit
    );

    emit('stdout', `\nPackages installed successfully.\n`);
  }

  return {
    status: 'created',
    name,
    pythonVersion,
    packages
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
        // Ensure conda doesn't ask for user input
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

module.exports = createCondaEnvironment;
