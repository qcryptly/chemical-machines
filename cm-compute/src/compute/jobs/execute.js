/**
 * Execute Job Handler
 *
 * Executes arbitrary Python code (for notebook cells)
 * Supports workspace-relative imports via `from workspace import module`
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Workspace directory (matches cm-view configuration)
const WORKSPACE_DIR = process.env.WORKSPACE_DIR || path.join(__dirname, '../../../../workspace');

/**
 * Ensure workspace has an __init__.py for package imports
 */
function ensureWorkspacePackage() {
  const initPath = path.join(WORKSPACE_DIR, '__init__.py');
  if (!fs.existsSync(initPath)) {
    try {
      fs.writeFileSync(initPath, '# Workspace package - auto-generated\n');
    } catch (e) {
      // Ignore - workspace might not exist yet
    }
  }
}

/**
 * Execute Python code directly
 * @param {Object} params - Job parameters { code, environment, sourceDir, cellInfo }
 * @param {Object} context - Execution context { pythonPath, jobId, emit }
 * @returns {Promise<Object>} Execution results
 */
async function execute(params, context) {
  const { pythonPath, jobId, emit } = context;
  const { code, sourceDir = '', cellInfo } = params;

  if (!code) {
    throw new Error('No code provided');
  }

  // Ensure workspace can be imported as a package
  ensureWorkspacePackage();

  // Build PYTHONPATH to include:
  // 1. Parent of workspace (so `import workspace` works)
  // 2. The workspace itself (so `import module` works for files in workspace root)
  // 3. The source file's directory (so relative imports work)
  // 4. cm-libraries for cm_output module
  const workspaceParent = path.dirname(WORKSPACE_DIR);
  const sourceFullDir = sourceDir ? path.join(WORKSPACE_DIR, sourceDir) : WORKSPACE_DIR;
  const cmLibrariesPath = path.join(__dirname, '../../../../cm-libraries/python');

  const pythonPathParts = [
    cmLibrariesPath,  // For cm_output module
    workspaceParent,  // For `from workspace import module`
    WORKSPACE_DIR,    // For direct `import module` from workspace root
    sourceFullDir     // For relative imports from current file's directory
  ];

  const existingPythonPath = process.env.PYTHONPATH || '';
  const newPythonPath = [...pythonPathParts, existingPythonPath].filter(Boolean).join(':');

  // Build cell output environment variables
  const cellEnv = {};
  if (cellInfo) {
    const { filePath, cellIndex, isCellFile } = cellInfo;
    // Output file path: .out/filename.html in the workspace
    const outputFile = filePath
      ? path.join(sourceFullDir, '.out', `${filePath}.html`)
      : '';

    cellEnv.CM_OUTPUT_FILE = outputFile;
    cellEnv.CM_CELL_INDEX = String(cellIndex ?? -1);
    cellEnv.CM_IS_CELL_FILE = isCellFile ? 'true' : 'false';
    cellEnv.CM_WORKSPACE_DIR = sourceFullDir;
  }

  return new Promise((resolve, reject) => {
    const python = spawn(pythonPath, ['-c', code], {
      cwd: sourceFullDir,  // Run from the source file's directory
      env: {
        ...process.env,
        ...cellEnv,
        PYTHONUNBUFFERED: '1',
        PYTHONPATH: newPythonPath
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

    python.on('close', (exitCode) => {
      if (exitCode === 0) {
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
          error: `Python exited with code ${exitCode}`,
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
