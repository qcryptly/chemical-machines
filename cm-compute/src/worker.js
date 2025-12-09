const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { createWorkerLogger } = require('./logger');

// Worker process that executes compute jobs
// This process is forked by the queue and runs Python/C++ compute tasks

const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';

// Worker ID passed from queue.js
let workerId = process.env.WORKER_ID || '0';
let logger = null;

// Initialize logger when worker ID is set
function initLogger() {
  if (!logger) {
    logger = createWorkerLogger(workerId);
    logger.info('Worker started', { workerId, pid: process.pid });
  }
}

const COMPUTE_TYPES = {
  'molecular-dynamics': 'compute/molecular_dynamics.py',
  'docking': 'compute/docking.py',
  'binding-affinity': 'compute/binding_affinity.py',
  'structure-prediction': 'compute/structure_prediction.py',
  'optimization': 'compute/optimization.py',
  'execute': null, // Direct code execution
};

// Get Python path for a conda environment
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

process.on('message', async (message) => {
  if (message.type === 'init') {
    // Receive worker ID from queue
    workerId = message.workerId;
    initLogger();
    return;
  }

  if (message.type === 'execute') {
    const { job } = message;
    initLogger(); // Ensure logger is initialized

    logger.info('Executing job', { jobId: job.id, type: job.type, params: job.params });

    try {
      const result = await executeJob(job);
      logger.info('Job completed successfully', { jobId: job.id, type: job.type });
      process.send({
        type: 'result',
        result: result
      });
    } catch (error) {
      logger.error('Job failed', { jobId: job.id, type: job.type, error: error.message, stack: error.stack });
      process.send({
        type: 'result',
        result: {
          error: error.message,
          stack: error.stack
        }
      });
    }
  }
});

async function executeJob(job) {
  const scriptPath = COMPUTE_TYPES[job.type];

  // Check if this is a known compute type (except 'execute' which is direct code)
  if (scriptPath === undefined) {
    throw new Error(`Unknown compute type: ${job.type}`);
  }

  // Get Python path for the specified environment
  const environment = job.params?.environment || 'chemcomp';
  const pythonPath = getPythonPath(environment);

  // Handle direct code execution
  if (job.type === 'execute') {
    return executeCode(pythonPath, job.params);
  }

  const fullPath = path.join(__dirname, scriptPath);

  return new Promise((resolve, reject) => {
    // Use specified conda environment Python
    const python = spawn(pythonPath, [
      fullPath,
      JSON.stringify(job.params)
    ]);

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      const output = data.toString();
      stdout += output;

      // Check for progress updates
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.startsWith('PROGRESS:')) {
          const progress = parseFloat(line.split(':')[1]);
          process.send({
            type: 'progress',
            progress: progress
          });
        }
      }
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (error) {
          resolve({ output: stdout });
        }
      } else {
        reject(new Error(`Python process exited with code ${code}: ${stderr}`));
      }
    });

    python.on('error', (error) => {
      reject(error);
    });
  });
}

// Execute Python code directly (for notebook cells)
async function executeCode(pythonPath, params) {
  const { code } = params;

  if (!code) {
    throw new Error('No code provided');
  }

  logger.info('Executing Python code', { pythonPath, codeLength: code.length });

  return new Promise((resolve, reject) => {
    // Execute code via python -c
    const python = spawn(pythonPath, ['-c', code], {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1'
      }
    });

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      const output = data.toString();
      stdout += output;

      // Check for progress updates
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.startsWith('PROGRESS:')) {
          const progress = parseFloat(line.split(':')[1]);
          process.send({
            type: 'progress',
            progress: progress
          });
        }
      }
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        logger.info('Python code executed successfully', { exitCode: code });
        // Try to parse as JSON first
        try {
          const result = JSON.parse(stdout.trim());
          resolve(result);
        } catch (e) {
          // Return raw output
          resolve({ output: stdout, stderr: stderr || undefined });
        }
      } else {
        logger.warn('Python code exited with error', { exitCode: code, stderr });
        resolve({
          error: `Python exited with code ${code}`,
          output: stdout,
          stderr: stderr
        });
      }
    });

    python.on('error', (error) => {
      logger.error('Python spawn error', { error: error.message });
      reject(error);
    });
  });
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  if (logger) {
    logger.error('Uncaught exception in worker', { error: error.message, stack: error.stack });
  }
  console.error('Uncaught exception in worker:', error);
  process.send({
    type: 'result',
    result: {
      error: error.message,
      stack: error.stack
    }
  });
  process.exit(1);
});
