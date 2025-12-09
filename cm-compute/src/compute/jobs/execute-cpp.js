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
const { analyzeCode, generateCompileCommand } = require('../lib/include-parser');
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
 * Execute C++ code
 * @param {Object} params - Job parameters
 * @param {string} params.code - C++ source code
 * @param {string} params.cppEnvironment - Name of C++ environment to use
 * @param {string} params.vendorEnvironment - Name of vendor environment to use
 * @param {string} params.compiler - Compiler to use (g++, clang++)
 * @param {string} params.cppStandard - C++ standard (c++11, c++14, c++17, c++20, c++23)
 * @param {string[]} params.extraFlags - Additional compiler flags
 * @param {Object} context - Execution context { jobId, emit }
 * @returns {Promise<Object>} Execution result
 */
async function executeCpp(params, context) {
  const { emit, jobId } = context;
  const { code, cppEnvironment, vendorEnvironment, compiler = 'clang++', cppStandard = 'c++23', extraFlags = [] } = params;

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

  try {
    // Write source code to file
    fs.writeFileSync(sourceFile, code);

    // Analyze code for dependencies
    emit('stdout', `Using ${compiler} with ${cppStandard}...\n`);
    const analysis = analyzeCode(code, cppEnv, vendorEnv, { compiler, cppStandard });

    // Show detected dependencies
    if (analysis.includes.length > 0) {
      const nonStdIncludes = analysis.includes.filter(i => !i.startsWith('c') && !['iostream', 'vector', 'string', 'map', 'set', 'algorithm', 'memory', 'functional', 'fstream', 'sstream'].includes(i));
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

    // Execute
    const execResult = await runProcess(outputFile, [], emit, tmpDir);

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
 */
function runProcess(command, args, emit, cwd) {
  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';

    const proc = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
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
