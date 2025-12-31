/**
 * Execute C++ Code Job Handler
 *
 * Compiles and executes C++ code with automatic dependency detection.
 * Supports CUDA, OpenMP, and various libraries.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { analyzeCode, generateCompileCommand, parseIncludes } = require('../lib/include-parser');
const database = require('../../database');

// Workspace directory (matches cm-view configuration)
const WORKSPACE_DIR = process.env.WORKSPACE_DIR || path.join(__dirname, '../../../../workspace');

// Track if database is initialized in this worker
let dbInitialized = false;

/**
 * Extract local include paths from code (includes with ./ or ../ or just filename)
 * @param {string} code - C++ source code
 * @returns {string[]} Array of local include paths
 */
function extractLocalIncludes(code) {
  const includes = parseIncludes(code);
  return includes.filter(inc => {
    // Local includes: start with ./ or ../, or end with .h/.hpp/.hxx and don't contain /
    return inc.startsWith('./') ||
           inc.startsWith('../') ||
           (inc.match(/\.(h|hpp|hxx)$/i) && !inc.includes('/'));
  });
}

/**
 * Copy local include files to target directory, preserving relative paths
 * @param {string} sourceDir - Source directory (relative to workspace)
 * @param {string} targetDir - Target directory to copy files to
 * @param {string[]} localIncludes - Array of local include paths
 * @param {Function} emit - Emit function for logging
 * @returns {string[]} Array of successfully copied files
 */
function copyLocalIncludes(sourceDir, targetDir, localIncludes, emit) {
  const copiedFiles = [];
  const workspaceSourceDir = path.join(WORKSPACE_DIR, sourceDir);

  for (const include of localIncludes) {
    try {
      // Resolve the include path relative to the source directory
      const sourcePath = path.resolve(workspaceSourceDir, include);

      // Security check: ensure the resolved path is within workspace
      const resolvedWorkspace = path.resolve(WORKSPACE_DIR);
      if (!sourcePath.startsWith(resolvedWorkspace)) {
        emit('stderr', `Warning: Include path '${include}' is outside workspace, skipping\n`);
        continue;
      }

      // Check if file exists
      if (!fs.existsSync(sourcePath)) {
        emit('stderr', `Warning: Include file '${include}' not found at ${sourcePath}\n`);
        continue;
      }

      // Determine target path (preserve relative structure)
      const targetPath = path.join(targetDir, include);

      // Create parent directories if needed
      const targetParent = path.dirname(targetPath);
      if (!fs.existsSync(targetParent)) {
        fs.mkdirSync(targetParent, { recursive: true });
      }

      // Copy the file
      fs.copyFileSync(sourcePath, targetPath);
      copiedFiles.push(include);

      // Recursively check the included file for more includes
      const fileContent = fs.readFileSync(sourcePath, 'utf-8');
      const nestedIncludes = extractLocalIncludes(fileContent);
      if (nestedIncludes.length > 0) {
        // Resolve nested includes relative to the included file's directory
        const includeDir = path.dirname(include);
        const nestedResolved = nestedIncludes.map(nested => {
          if (includeDir && includeDir !== '.') {
            return path.join(includeDir, nested);
          }
          return nested;
        });
        const nestedCopied = copyLocalIncludes(sourceDir, targetDir, nestedResolved, emit);
        copiedFiles.push(...nestedCopied);
      }
    } catch (e) {
      emit('stderr', `Warning: Could not copy '${include}': ${e.message}\n`);
    }
  }

  return [...new Set(copiedFiles)]; // Remove duplicates
}

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
 * Execute C++ code
 * @param {Object} params - Job parameters
 * @param {string} params.code - C++ source code
 * @param {string} params.sourceDir - Source directory (relative to workspace) for resolving local includes
 * @param {string} params.cppEnvironment - Name of C++ environment to use
 * @param {string} params.vendorEnvironment - Name of vendor environment to use
 * @param {string} params.compiler - Compiler to use (g++, clang++)
 * @param {string} params.cppStandard - C++ standard (c++11, c++14, c++17, c++20, c++23)
 * @param {string[]} params.extraFlags - Additional compiler flags
 * @param {Object} params.cellInfo - Cell output information { filePath, cellIndex, isCellFile }
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Execution result
 */
async function executeCpp(params, context) {
  const { emit, jobId } = context;
  const { code, sourceDir = '', cppEnvironment, vendorEnvironment, compiler = 'clang++', cppStandard = 'c++23', extraFlags = [], cellInfo } = params;

  if (!code || code.trim().length === 0) {
    throw new Error('No code provided');
  }

  // Get environment details from database
  let cppEnv = null;
  let vendorEnv = null;

  try {
    // Initialize database in worker and get models
    const models = await ensureDatabase();

    if (cppEnvironment) {
      cppEnv = await models.CppEnvironment.findByName(cppEnvironment);
      if (cppEnv) {
        // Get linked vendor environments
        const linkedVendors = await models.CppEnvironment.getLinkedVendorEnvironments(cppEnv.id);
        if (linkedVendors.length > 0 && !vendorEnvironment) {
          vendorEnv = linkedVendors[0];
        }
      }
    }

    if (vendorEnvironment) {
      vendorEnv = await models.VendorEnvironment.findByName(vendorEnvironment);
    }
  } catch (e) {
    // Log but continue - can still compile without environment details
    emit('stderr', `Warning: Could not load environment details: ${e.message}\n`);
  }

  // Create temporary directory for compilation
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cpp-'));
  const sourceFile = path.join(tmpDir, 'main.cpp');
  const outputFile = path.join(tmpDir, 'main');

  // Build cell output environment variables
  const sourceFullDir = sourceDir ? path.join(WORKSPACE_DIR, sourceDir) : WORKSPACE_DIR;
  const cellEnv = {};
  if (cellInfo) {
    const { filePath, cellIndex, isCellFile } = cellInfo;
    // Output file path: .out/filename.html in the workspace
    const outputHtmlFile = filePath
      ? path.join(sourceFullDir, '.out', `${filePath}.html`)
      : '';

    cellEnv.CM_OUTPUT_FILE = outputHtmlFile;
    cellEnv.CM_CELL_INDEX = String(cellIndex ?? -1);
    cellEnv.CM_IS_CELL_FILE = isCellFile ? 'true' : 'false';
    cellEnv.CM_WORKSPACE_DIR = sourceFullDir;
  }

  try {
    // Write source code to file
    fs.writeFileSync(sourceFile, code);

    // Extract and copy local includes if sourceDir is provided
    const localIncludes = extractLocalIncludes(code);
    if (localIncludes.length > 0 && sourceDir) {
      emit('stdout', `Copying local includes: ${localIncludes.join(', ')}\n`);
      const copiedFiles = copyLocalIncludes(sourceDir, tmpDir, localIncludes, emit);
      if (copiedFiles.length > 0) {
        emit('stdout', `Copied ${copiedFiles.length} local file(s)\n`);
      }
    } else if (localIncludes.length > 0 && !sourceDir) {
      emit('stderr', `Warning: Local includes detected but source directory not provided\n`);
      emit('stderr', `Local includes may not be resolved: ${localIncludes.join(', ')}\n`);
    }

    // Analyze code for dependencies
    emit('stdout', `Using ${compiler} with ${cppStandard}...\n`);
    const analysis = analyzeCode(code, cppEnv, vendorEnv, { compiler, cppStandard });

    // Show detected dependencies (exclude local includes we already handled)
    if (analysis.includes.length > 0) {
      const nonStdIncludes = analysis.includes.filter(i =>
        !i.startsWith('c') &&
        !['iostream', 'vector', 'string', 'map', 'set', 'algorithm', 'memory', 'functional', 'fstream', 'sstream'].includes(i) &&
        !localIncludes.includes(i)
      );
      if (nonStdIncludes.length > 0) {
        emit('stdout', `Detected includes: ${nonStdIncludes.slice(0, 5).join(', ')}${nonStdIncludes.length > 5 ? '...' : ''}\n`);
      }
    }

    if (analysis.usesCuda) {
      emit('stdout', 'CUDA detected, using nvcc compiler\n');
    }

    if (analysis.usesOpenMP) {
      emit('stdout', 'OpenMP detected, enabling parallel support\n');
    }

    // Generate compile command
    const { compiler: finalCompiler, args } = generateCompileCommand(sourceFile, outputFile, analysis);

    // Add cm-libraries include path for cm_output.hpp
    const cmLibrariesCppPath = path.join(__dirname, '../../../../cm-libraries/cpp');
    args.unshift(`-I${cmLibrariesCppPath}`);

    // Add extra flags
    if (extraFlags.length > 0) {
      args.push(...extraFlags);
    }

    emit('stdout', `\nCompiling with ${finalCompiler}...\n`);
    emit('stdout', `$ ${finalCompiler} ${args.join(' ')}\n\n`);

    // Compile
    const compileResult = await runProcess(finalCompiler, args, emit, tmpDir);

    if (compileResult.exitCode !== 0) {
      emit('stderr', '\nCompilation failed.\n');
      return {
        status: 'error',
        stage: 'compile',
        exitCode: compileResult.exitCode,
        output: compileResult.stderr
      };
    }

    emit('stdout', 'Compilation successful.\n\n');
    emit('stdout', '─'.repeat(40) + '\n');
    emit('stdout', 'Running program...\n');
    emit('stdout', '─'.repeat(40) + '\n\n');

    // Execute with cell environment variables
    const execResult = await runProcess(outputFile, [], emit, tmpDir, cellEnv);

    emit('stdout', '\n' + '─'.repeat(40) + '\n');
    emit('stdout', `Program exited with code ${execResult.exitCode}\n`);

    return {
      status: execResult.exitCode === 0 ? 'success' : 'error',
      stage: 'execute',
      exitCode: execResult.exitCode,
      compiler: finalCompiler,
      analysis: {
        usesCuda: analysis.usesCuda,
        usesOpenMP: analysis.usesOpenMP,
        suggestedPackages: analysis.suggestedPackages
      }
    };

  } finally {
    // Cleanup
    try {
      fs.rmSync(tmpDir, { recursive: true });
    } catch (e) {
      // Ignore cleanup errors
    }
  }
}

/**
 * Run a process and stream output
 * @param {string} command - Command to run
 * @param {string[]} args - Command arguments
 * @param {Function} emit - Emit function for stdout/stderr
 * @param {string} cwd - Working directory
 * @param {Object} extraEnv - Extra environment variables
 */
function runProcess(command, args, emit, cwd, extraEnv = {}) {
  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';

    const proc = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        ...extraEnv,
        // Add CUDA paths if available
        PATH: `${process.env.PATH}:/usr/local/cuda/bin`,
        LD_LIBRARY_PATH: `${process.env.LD_LIBRARY_PATH || ''}:/usr/local/cuda/lib64:/opt/vendor/lib`
      }
    });

    proc.stdout.on('data', (data) => {
      const text = data.toString();
      stdout += text;
      emit('stdout', text);
    });

    proc.stderr.on('data', (data) => {
      const text = data.toString();
      stderr += text;
      emit('stderr', text);
    });

    proc.on('close', (code) => {
      resolve({
        exitCode: code || 0,
        stdout,
        stderr
      });
    });

    proc.on('error', (error) => {
      emit('stderr', `Error: ${error.message}\n`);
      resolve({
        exitCode: 1,
        stdout,
        stderr: error.message
      });
    });
  });
}

module.exports = executeCpp;
