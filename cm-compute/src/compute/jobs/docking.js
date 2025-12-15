/**
 * Molecular Docking Job Handler
 *
 * Runs molecular docking calculations
 */

const { spawn } = require('child_process');
const path = require('path');

const SCRIPT_PATH = path.join(__dirname, '../scripts/docking.py');

/**
 * Execute molecular docking
 * @param {Object} params - Job parameters
 * @param {Object} context - Execution context { pythonPath, jobId, emit }
 * @returns {Promise<Object>} Docking results
 */
async function docking(params, context) {
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

      emit('stdout', output);

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
        reject(new Error(`Docking failed with code ${code}: ${stderr}`));
      }
    });

    python.on('error', (error) => {
      reject(error);
    });
  });
}

module.exports = docking;
