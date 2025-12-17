/**
 * Create Conda Environment Job Handler
 *
 * Creates a new conda environment with specified Python version and packages.
 * Streams output during creation for real-time progress updates.
 *
 * PyTorch packages (torch, torchvision, torchaudio, etc.) are installed from
 * nightly builds using pip for the latest features and fixes.
 */

const { spawn, execSync } = require('child_process');

const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';

/**
 * Detect CUDA version from nvidia-smi
 * @returns {string|null} CUDA version (e.g., '12.4') or null if not available
 */
function detectCudaVersion() {
  try {
    // nvidia-smi displays CUDA version in its output header
    const output = execSync('nvidia-smi', { encoding: 'utf-8', timeout: 5000 });

    // Parse CUDA Version from output like "CUDA Version: 12.4"
    const cudaMatch = output.match(/CUDA Version:\s*(\d+\.\d+)/);
    if (cudaMatch) {
      return cudaMatch[1];
    }

    return null;
  } catch {
    // nvidia-smi not available or failed
    return null;
  }
}

// PyTorch-related packages that should use nightly builds
const PYTORCH_PACKAGES = [
  'torch',
  'torchvision',
  'torchaudio',
  'torchtext',
  'torchdata',
  'torchserve',
  'torch-model-archiver',
  'functorch',
  'torcharrow',
  'torchrec',
  'pytorch-lightning'
];

/**
 * Check if a package is a PyTorch-related package
 * @param {string} pkg - Package name (may include version specifier)
 * @returns {boolean}
 */
function isPyTorchPackage(pkg) {
  const pkgName = pkg.split(/[=<>!]/)[0].toLowerCase();
  return PYTORCH_PACKAGES.some(p => pkgName === p || pkgName.startsWith(`${p}-`) || pkgName.startsWith(`${p}[`));
}

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
    // Separate PyTorch packages from regular packages
    const pytorchPackages = packages.filter(isPyTorchPackage);
    const regularPackages = packages.filter(pkg => !isPyTorchPackage(pkg));

    // Install regular packages via conda
    if (regularPackages.length > 0) {
      emit('stdout', `\nInstalling packages via conda: ${regularPackages.join(', ')}...\n`);

      await runCondaCommand(
        ['install', '-n', name, ...regularPackages, '-y'],
        emit
      );

      emit('stdout', `\nConda packages installed successfully.\n`);
    }

    // Install PyTorch packages via pip with nightly builds
    if (pytorchPackages.length > 0) {
      // Detect CUDA version and get appropriate index URL
      const cudaVersion = detectCudaVersion().replace('.','');
      const indexUrl = `https://download.pytorch.org/whl/nightly/cu${cudaVersion}`;

      if (cudaVersion) {
        emit('stdout', `\nDetected CUDA version: ${cudaVersion}\n`);
      } else {
        emit('stdout', `\nNo CUDA detected, using CPU-only PyTorch\n`);
      }

      emit('stdout', `Installing PyTorch nightly packages via pip: ${pytorchPackages.join(', ')}...\n`);
      emit('stdout', `Using index: ${indexUrl}\n\n`);

      await runPipCommand(
        name,
        [
          'install',
          '--pre',
          ...pytorchPackages,
          '--index-url', indexUrl
        ],
        emit
      );

      emit('stdout', `\nPyTorch nightly packages installed successfully.\n`);
    }
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

/**
 * Run a pip command within a conda environment
 * @param {string} envName - Conda environment name
 * @param {string[]} args - Pip command arguments
 * @param {Function} emit - Emit function for streaming output
 * @returns {Promise<void>}
 */
function runPipCommand(envName, args, emit) {
  return new Promise((resolve, reject) => {
    // Use conda run to execute pip within the environment
    const conda = spawn(`${CONDA_PATH}/bin/conda`, ['run', '-n', envName, 'pip', ...args], {
      env: {
        ...process.env
      }
    });

    conda.stdout.on('data', (data) => {
      const output = data.toString();
      emit('stdout', output);
    });

    conda.stderr.on('data', (data) => {
      const output = data.toString();
      // Pip often writes progress to stderr, show as stdout for better UX
      emit('stdout', output);
    });

    conda.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Pip command failed with exit code ${code}`));
      }
    });

    conda.on('error', (error) => {
      reject(error);
    });
  });
}

module.exports = createCondaEnvironment;
