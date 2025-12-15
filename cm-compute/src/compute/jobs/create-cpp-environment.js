/**
 * Create C++ Environment Job Handler
 *
 * Creates a new C++ environment by installing debian dev packages.
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
 * Create a new C++ environment with debian dev packages
 * @param {Object} params - Job parameters
 * @param {string} params.name - Environment name
 * @param {string} params.description - Environment description
 * @param {string[]} params.packages - Debian dev packages to install
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Creation result
 */
async function createCppEnvironment(params, context) {
  const { emit } = context;
  const { name, description = '', packages = [] } = params;

  if (!name) {
    throw new Error('Environment name is required');
  }

  // Validate environment name
  if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(name)) {
    throw new Error('Invalid environment name. Use alphanumeric characters, hyphens, and underscores.');
  }

  emit('stdout', `Creating C++ environment '${name}'...\n`);

  // Initialize database in worker and get models
  const { CppEnvironment } = await ensureDatabase();

  // Check if environment already exists
  const existing = await CppEnvironment.findByName(name);
  if (existing) {
    throw new Error(`Environment '${name}' already exists`);
  }

  // Install packages if specified
  if (packages.length > 0) {
    emit('stdout', `\nInstalling packages: ${packages.join(', ')}...\n`);
    emit('stdout', '─'.repeat(50) + '\n');

    // Update apt cache first
    emit('stdout', 'Updating package cache...\n');
    await runCommand('apt-get', ['update'], emit);

    // Install packages
    emit('stdout', `\nInstalling ${packages.length} package(s)...\n`);
    await runCommand('apt-get', ['install', '-y', ...packages], emit);

    emit('stdout', '\n' + '─'.repeat(50) + '\n');
    emit('stdout', `Packages installed successfully.\n`);
  }

  // Create environment record in database
  const env = await CppEnvironment.create({
    name,
    description,
    packages
  });

  emit('stdout', `\nC++ environment '${name}' created successfully.\n`);

  return {
    status: 'created',
    id: env.id,
    name,
    description,
    packages
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

module.exports = createCppEnvironment;
