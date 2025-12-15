/**
 * Molecular Dynamics Job Handler
 *
 * Runs molecular dynamics simulations using Python/PyTorch
 */

const { spawn } = require('child_process');
const path = require('path');

const SCRIPT_PATH = path.join(__dirname, '../scripts/molecular_dynamics.py');

/**
 * Execute molecular dynamics simulation
 * @param {Object} params - Job parameters
 * @param {Object} context - Execution context { pythonPath, jobId, emit }
 * @returns {Promise<Object>} Simulation results
 */
async function molecularDynamics(params, context) {
  const { pythonPath, jobId, emit } = context;

  return new Promise((resolve, reject) => {
    const python = spawn(pythonPath, [
      SCRIPT_PATH,
      JSON.stringify(params)
    ]);

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      const output = data.toString();
      stdout += output;

      // Emit stdout to job listener
      emit('stdout', output);

      // Parse progress updates
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.startsWith('PROGRESS:')) {
          const progress = parseFloat(line.split(':')[1]);
          emit('progress', progress);
        }
      }
    });

    python.stderr.on('data', (data) => {
      const output = data.toString();
      stderr += output;

      // Emit stderr to job listener
      emit('stderr', output);
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
        reject(new Error(`Molecular dynamics failed with code ${code}: ${stderr}`));
      }
    });

    python.on('error', (error) => {
      reject(error);
    });
  });
}

module.exports = molecularDynamics;
