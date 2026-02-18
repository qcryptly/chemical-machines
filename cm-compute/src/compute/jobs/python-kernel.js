/**
 * Python Kernel Manager
 *
 * Maintains persistent Python processes to enable Jupyter-like behavior
 * where imports, variables, and function definitions persist across cell executions.
 */

const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');

// Store active kernels by workspace/file identifier
const activeKernels = new Map();

// Unique delimiter to separate cell outputs
const OUTPUT_DELIMITER = '<<<CELL_OUTPUT_END>>>';

class PythonKernel {
  constructor(pythonPath, workspaceDir, sourceDir, cellInfo) {
    this.pythonPath = pythonPath;
    this.workspaceDir = workspaceDir;
    this.sourceDir = sourceDir;
    this.cellInfo = cellInfo;
    this.process = null;
    this.isReady = false;
    this.currentResolve = null;
    this.currentReject = null;
    this.currentEmit = null;
    this.stdout = '';
    this.stderr = '';
    this.executionTimeout = null;
    this.isExecuting = false;
  }

  /**
   * Start the persistent Python process
   */
  async start() {
    if (this.process) {
      return;
    }

    const workspaceParent = path.dirname(this.workspaceDir);
    const cmLibrariesPath = path.join(__dirname, '../../../../cm-libraries/python');

    const pythonPathParts = [
      cmLibrariesPath,
      workspaceParent,
      this.workspaceDir,
      this.sourceDir
    ];

    const existingPythonPath = process.env.PYTHONPATH || '';
    const newPythonPath = [...pythonPathParts, existingPythonPath].filter(Boolean).join(':');

    // Build cell output environment variables
    const cellEnv = {};
    if (this.cellInfo) {
      const { filePath, isCellFile } = this.cellInfo;
      const fileName = filePath ? path.basename(filePath) : '';
      const outputFile = fileName
        ? path.join(this.sourceDir, '.out', `${fileName}.html`)
        : '';

      cellEnv.CM_OUTPUT_FILE = outputFile;
      cellEnv.CM_IS_CELL_FILE = isCellFile ? 'true' : 'false';
      cellEnv.CM_WORKSPACE_DIR = this.sourceDir;
    }

    // Start Python in interactive mode with code execution wrapper
    const pythonCode = `
import sys
import traceback
import signal
import importlib

# Print delimiter after each cell execution
DELIMITER = "${OUTPUT_DELIMITER}"
INTERRUPTED_MARKER = "<<<CELL_INTERRUPTED>>>"

# Flag to track if execution was interrupted
interrupted = False

def handle_interrupt(signum, frame):
    """Handle SIGINT (Ctrl+C) during cell execution"""
    global interrupted
    interrupted = True
    raise KeyboardInterrupt("Cell execution interrupted by user")

def _reload_cm_modules():
    """Clear cached cm.* modules so source changes are picked up on next import."""
    stale = [name for name in sys.modules if name == 'cm' or name.startswith('cm.')]
    for name in stale:
        del sys.modules[name]
    importlib.invalidate_caches()

def execute_cell(code):
    """Execute a cell of code and capture output"""
    global interrupted
    interrupted = False

    # Auto-reload cm library modules to pick up source changes
    _reload_cm_modules()

    # Set up interrupt handler for this cell execution
    old_handler = signal.signal(signal.SIGINT, handle_interrupt)

    try:
        # Use exec to run the code in the global namespace
        exec(code, globals())
        print(DELIMITER, flush=True)
        return True
    except KeyboardInterrupt:
        # Cell was interrupted
        print("\\nKeyboardInterrupt: Cell execution interrupted", file=sys.stderr, flush=True)
        print(INTERRUPTED_MARKER, flush=True)
        print(DELIMITER, flush=True)
        return False
    except Exception as e:
        # Print the full traceback
        traceback.print_exc()
        print(DELIMITER, flush=True)
        return False
    finally:
        # Restore original interrupt handler
        signal.signal(signal.SIGINT, old_handler)
        interrupted = False

# Signal that kernel is ready
print("KERNEL_READY", flush=True)

# Read and execute cells from stdin
while True:
    try:
        # Read until we get the magic marker
        lines = []
        while True:
            line = sys.stdin.readline()
            if not line:  # EOF
                sys.exit(0)
            if line.strip() == "<<<EXECUTE_CELL>>>":
                break
            lines.append(line)

        code = ''.join(lines)
        if code.strip():
            execute_cell(code)
        else:
            print(DELIMITER, flush=True)
    except KeyboardInterrupt:
        # Interrupted while waiting for input - continue
        print("\\nInterrupted while idle", file=sys.stderr, flush=True)
        print(DELIMITER, flush=True)
        continue
    except Exception as e:
        traceback.print_exc()
        print(DELIMITER, flush=True)
`;

    this.process = spawn(this.pythonPath, ['-u', '-c', pythonCode], {
      cwd: this.sourceDir,
      env: {
        ...process.env,
        ...cellEnv,
        PYTHONUNBUFFERED: '1',
        PYTHONDONTWRITEBYTECODE: '1',
        PYTHONPATH: newPythonPath
      }
    });

    // Handle stdin errors (EPIPE if Python process dies unexpectedly)
    this.process.stdin.on('error', (err) => {
      if (err.code === 'EPIPE' || err.code === 'ERR_STREAM_DESTROYED') {
        // Python process died — handled by 'close' event
        return;
      }
      console.error('Kernel stdin error:', err);
    });

    // Set up output handling
    const rl = readline.createInterface({
      input: this.process.stdout,
      crlfDelay: Infinity
    });

    rl.on('line', (line) => {
      if (line === 'KERNEL_READY') {
        this.isReady = true;
      } else if (line === '<<<CELL_INTERRUPTED>>>') {
        this.wasInterrupted = true;
      } else if (line === OUTPUT_DELIMITER) {
        // Cell execution completed
        if (this.currentResolve) {
          // Clear timeout if set
          if (this.executionTimeout) {
            clearTimeout(this.executionTimeout);
            this.executionTimeout = null;
          }

          this.isExecuting = false;

          const result = {
            output: this.stdout,
            stderr: this.stderr || undefined,
            interrupted: this.wasInterrupted || false
          };

          this.currentResolve(result);
          this.currentResolve = null;
          this.currentReject = null;
          this.currentEmit = null;
          this.stdout = '';
          this.stderr = '';
          this.wasInterrupted = false;
        }
      } else {
        this.stdout += line + '\n';
        if (this.currentEmit) {
          this.currentEmit('stdout', line + '\n');
        }
      }
    });

    // Capture stderr
    this.process.stderr.on('data', (data) => {
      const output = data.toString();
      this.stderr += output;
      if (this.currentEmit) {
        this.currentEmit('stderr', output);
      }
    });

    this.process.on('close', (exitCode) => {
      this.isReady = false;
      this.isExecuting = false;
      this.process = null;
      if (this.executionTimeout) {
        clearTimeout(this.executionTimeout);
        this.executionTimeout = null;
      }
      if (this.currentReject) {
        this.currentReject(new Error(`Kernel exited with code ${exitCode}`));
        this.currentResolve = null;
        this.currentReject = null;
        this.currentEmit = null;
      }
    });

    this.process.on('error', (error) => {
      this.isReady = false;
      if (this.currentReject) {
        this.currentReject(error);
        this.currentResolve = null;
        this.currentReject = null;
        this.currentEmit = null;
      }
    });

    // Wait for kernel to be ready
    await new Promise((resolve, reject) => {
      const checkReady = setInterval(() => {
        if (this.isReady) {
          clearInterval(checkReady);
          resolve();
        }
      }, 50);

      // Timeout after 5 seconds
      setTimeout(() => {
        clearInterval(checkReady);
        if (!this.isReady) {
          reject(new Error('Kernel failed to start'));
        }
      }, 5000);
    });
  }

  /**
   * Execute code in the persistent kernel
   * @param {string} code - Python code to execute
   * @param {number} cellIndex - Cell index for tracking
   * @param {Function} emit - Function to emit output events
   * @param {number} timeout - Optional timeout in milliseconds (0 or undefined = no timeout)
   */
  async execute(code, cellIndex, emit, timeout) {
    if (!this.process || !this.isReady) {
      await this.start();
    }

    if (this.isExecuting) {
      throw new Error('Kernel is already executing a cell. Please wait or interrupt the current execution.');
    }

    return new Promise((resolve, reject) => {
      this.currentResolve = resolve;
      this.currentReject = reject;
      this.currentEmit = emit;
      this.stdout = '';
      this.stderr = '';
      this.isExecuting = true;
      this.wasInterrupted = false;

      // Set up timeout if specified
      if (timeout && timeout > 0) {
        this.executionTimeout = setTimeout(() => {
          this.interrupt();
          if (this.currentResolve) {
            this.currentResolve({
              output: this.stdout + '\n\nExecution timed out after ' + timeout + 'ms',
              stderr: (this.stderr || '') + '\nTimeoutError: Cell execution exceeded timeout',
              interrupted: true,
              timedOut: true
            });
            this.currentResolve = null;
            this.currentReject = null;
            this.currentEmit = null;
            this.isExecuting = false;
          }
        }, timeout);
      }

      // Guard against writing to a dead process
      if (!this.process || !this.process.stdin || this.process.stdin.destroyed) {
        this.isExecuting = false;
        reject(new Error('Kernel process is not available'));
        return;
      }

      try {
        // Update cell index and clear previous HTML output for this cell
        if (cellIndex !== undefined) {
          const cellIndexUpdate = `import os; os.environ['CM_CELL_INDEX'] = '${cellIndex}'\n`;
          this.process.stdin.write(cellIndexUpdate);
          // Clear stale HTML output so removed cm.views calls don't linger
          const clearOutput = `try:\n from cm.views.output import clear as _cm_clear; _cm_clear()\nexcept Exception:\n pass\n`;
          this.process.stdin.write(clearOutput);
        }

        // Send code followed by execution marker
        this.process.stdin.write(code + '\n');
        this.process.stdin.write('<<<EXECUTE_CELL>>>\n');
      } catch (err) {
        this.isExecuting = false;
        reject(new Error(`Failed to write to kernel: ${err.message}`));
      }
    });
  }

  /**
   * Interrupt the currently executing cell
   * Sends SIGINT to the Python process
   */
  interrupt() {
    if (this.process && this.isExecuting) {
      // Send SIGINT (Ctrl+C) to the Python process
      this.process.kill('SIGINT');
      return true;
    }
    return false;
  }

  /**
   * Check if the kernel is currently executing a cell
   */
  isBusy() {
    return this.isExecuting;
  }

  /**
   * Stop the kernel process
   */
  stop() {
    if (this.process) {
      this.process.kill();
      this.process = null;
      this.isReady = false;
    }
  }

  /**
   * Reset the kernel by stopping and clearing it
   */
  async reset() {
    this.stop();
    this.stdout = '';
    this.stderr = '';
  }
}

/**
 * Get or create a kernel for a specific context
 * @param {string} kernelId - Unique identifier for the kernel
 * @param {string} pythonPath - Path to Python executable
 * @param {string} workspaceDir - Workspace directory
 * @param {string} sourceDir - Source file directory
 * @param {Object} cellInfo - Cell metadata
 * @returns {PythonKernel}
 */
function getKernel(kernelId, pythonPath, workspaceDir, sourceDir, cellInfo) {
  if (!activeKernels.has(kernelId)) {
    const kernel = new PythonKernel(pythonPath, workspaceDir, sourceDir, cellInfo);
    activeKernels.set(kernelId, kernel);
  }
  return activeKernels.get(kernelId);
}

/**
 * Stop and remove a kernel
 * @param {string} kernelId - Kernel identifier
 */
function stopKernel(kernelId) {
  const kernel = activeKernels.get(kernelId);
  if (kernel) {
    kernel.stop();
    activeKernels.delete(kernelId);
  }
}

/**
 * Reset a kernel (stop and restart with clean state)
 * @param {string} kernelId - Kernel identifier
 */
async function resetKernel(kernelId) {
  const kernel = activeKernels.get(kernelId);
  if (kernel) {
    await kernel.reset();
    activeKernels.delete(kernelId);
  }
}

/**
 * Interrupt a running kernel
 * @param {string} kernelId - Kernel identifier
 * @returns {boolean} True if kernel was interrupted, false if not found or not executing
 */
function interruptKernel(kernelId) {
  const kernel = activeKernels.get(kernelId);
  if (kernel) {
    return kernel.interrupt();
  }
  return false;
}

/**
 * Check if a kernel is busy executing
 * @param {string} kernelId - Kernel identifier
 * @returns {boolean} True if kernel is executing
 */
function isKernelBusy(kernelId) {
  const kernel = activeKernels.get(kernelId);
  if (kernel) {
    return kernel.isBusy();
  }
  return false;
}

/**
 * Stop all active kernels
 */
function stopAllKernels() {
  for (const kernel of activeKernels.values()) {
    kernel.stop();
  }
  activeKernels.clear();
}

module.exports = {
  getKernel,
  stopKernel,
  resetKernel,
  interruptKernel,
  isKernelBusy,
  stopAllKernels
};
