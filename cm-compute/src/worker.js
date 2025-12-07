const { spawn } = require('child_process');
const path = require('path');

// Worker process that executes compute jobs
// This process is forked by the queue and runs Python/C++ compute tasks

const COMPUTE_TYPES = {
  'molecular-dynamics': 'compute/molecular_dynamics.py',
  'docking': 'compute/docking.py',
  'binding-affinity': 'compute/binding_affinity.py',
  'structure-prediction': 'compute/structure_prediction.py',
  'optimization': 'compute/optimization.py',
};

process.on('message', async (message) => {
  if (message.type === 'execute') {
    const { job } = message;

    try {
      const result = await executeJob(job);
      process.send({
        type: 'result',
        result: result
      });
    } catch (error) {
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

  if (!scriptPath) {
    throw new Error(`Unknown compute type: ${job.type}`);
  }

  const fullPath = path.join(__dirname, scriptPath);

  return new Promise((resolve, reject) => {
    // Use conda environment Python
    const python = spawn('/opt/conda/envs/chemcomp/bin/python', [
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

// Handle uncaught errors
process.on('uncaughtException', (error) => {
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
