/**
 * Execute Job Handler
 *
 * Executes arbitrary Python code (for notebook cells)
 */

const { spawn } = require('child_process');

/**
 * Execute Python code directly
 * @param {Object} params - Job parameters { code, environment }
 * @param {Object} context - Execution context { pythonPath, jobId, emit }
 * @returns {Promise<Object>} Execution results
 */
async function execute(params, context) {
  const { pythonPath, jobId, emit } = context;
  const { code } = params;

  if (!code) {
    throw new Error('No code provided');
  }

  return new Promise((resolve, reject) => {
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
      emit('stderr', output);
    });

    python.on('close', (code) => {
      if (code === 0) {
        // Try to parse as JSON first
        try {
          const result = JSON.parse(stdout.trim());
          resolve(result);
        } catch (e) {
          // Return raw output
          resolve({ output: stdout, stderr: stderr || undefined });
        }
      } else {
        resolve({
          error: `Python exited with code ${code}`,
          output: stdout,
          stderr: stderr
        });
      }
    });

    python.on('error', (error) => {
      reject(error);
    });
  });
}

module.exports = execute;
