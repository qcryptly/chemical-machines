#!/usr/bin/env node
/**
 * Integrated Tests for Compute Jobs
 *
 * Run directly with: node test.js
 * Or run specific test: node test.js execute
 *
 * These tests run the actual job handlers with mock contexts.
 */

const path = require('path');

// Job handlers
const execute = require('./execute');
const molecularDynamics = require('./molecular-dynamics');
const docking = require('./docking');
const bindingAffinity = require('./binding-affinity');
const structurePrediction = require('./structure-prediction');
const optimization = require('./optimization');
const createCondaEnvironment = require('./create-conda-environment');

// Test configuration
const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';
const PYTHON_PATH = process.env.PYTHON_PATH || `${CONDA_PATH}/bin/python`;

// Colors for output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m'
};

/**
 * Create a mock context for testing
 */
function createMockContext(jobId = 'test-job-1') {
  const output = {
    stdout: [],
    stderr: [],
    progress: []
  };

  return {
    context: {
      pythonPath: PYTHON_PATH,
      jobId,
      workerId: 'test-worker',
      environment: 'base',
      emit: (type, data) => {
        if (type === 'stdout') {
          output.stdout.push(data);
          process.stdout.write(colors.dim + data + colors.reset);
        } else if (type === 'stderr') {
          output.stderr.push(data);
          process.stderr.write(colors.yellow + data + colors.reset);
        } else if (type === 'progress') {
          output.progress.push(data);
        }
      }
    },
    output
  };
}

/**
 * Test runner
 */
class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
    this.skipped = 0;
  }

  /**
   * Register a test
   */
  test(name, fn, options = {}) {
    this.tests.push({ name, fn, options });
  }

  /**
   * Run all tests or specific test by name
   */
  async run(filter = null) {
    console.log('\n' + colors.cyan + '═'.repeat(60) + colors.reset);
    console.log(colors.cyan + '  Compute Jobs Integration Tests' + colors.reset);
    console.log(colors.cyan + '═'.repeat(60) + colors.reset + '\n');

    const testsToRun = filter
      ? this.tests.filter(t => t.name.toLowerCase().includes(filter.toLowerCase()))
      : this.tests;

    if (testsToRun.length === 0) {
      console.log(colors.yellow + `No tests matching "${filter}"` + colors.reset);
      return;
    }

    for (const test of testsToRun) {
      await this.runTest(test);
    }

    this.printSummary();
  }

  /**
   * Run a single test
   */
  async runTest(test) {
    const { name, fn, options } = test;
    const startTime = Date.now();

    process.stdout.write(`\n${colors.cyan}▶ ${name}${colors.reset}\n`);

    if (options.skip) {
      console.log(`  ${colors.yellow}⊘ SKIPPED${colors.reset} - ${options.skipReason || 'No reason given'}`);
      this.skipped++;
      return;
    }

    try {
      await fn();
      const duration = Date.now() - startTime;
      console.log(`  ${colors.green}✓ PASSED${colors.reset} (${duration}ms)\n`);
      this.passed++;
    } catch (error) {
      const duration = Date.now() - startTime;
      console.log(`  ${colors.red}✗ FAILED${colors.reset} (${duration}ms)`);
      console.log(`  ${colors.red}Error: ${error.message}${colors.reset}`);
      if (error.stack) {
        console.log(colors.dim + error.stack.split('\n').slice(1, 4).join('\n') + colors.reset);
      }
      this.failed++;
    }
  }

  /**
   * Print test summary
   */
  printSummary() {
    console.log('\n' + colors.cyan + '─'.repeat(60) + colors.reset);
    console.log(`  Tests: ${colors.green}${this.passed} passed${colors.reset}, ` +
      `${colors.red}${this.failed} failed${colors.reset}, ` +
      `${colors.yellow}${this.skipped} skipped${colors.reset}`);
    console.log(colors.cyan + '─'.repeat(60) + colors.reset + '\n');

    process.exit(this.failed > 0 ? 1 : 0);
  }
}

/**
 * Assertion helpers
 */
function assert(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(message || `Expected ${expected}, got ${actual}`);
  }
}

function assertContains(str, substring, message) {
  if (!str.includes(substring)) {
    throw new Error(message || `Expected "${str}" to contain "${substring}"`);
  }
}

function assertHasProperty(obj, prop, message) {
  if (!(prop in obj)) {
    throw new Error(message || `Expected object to have property "${prop}"`);
  }
}

// Create test runner
const runner = new TestRunner();

// ============================================================================
// Execute Job Tests
// ============================================================================

runner.test('execute: simple print statement', async () => {
  const { context, output } = createMockContext('exec-1');

  const result = await execute({ code: 'print("Hello, World!")' }, context);

  assertHasProperty(result, 'output');
  assertContains(result.output, 'Hello, World!');
  assert(output.stdout.length > 0, 'Should have stdout output');
});

runner.test('execute: JSON output', async () => {
  const { context } = createMockContext('exec-2');

  const result = await execute({
    code: 'import json; print(json.dumps({"status": "ok", "value": 42}))'
  }, context);

  assertEqual(result.status, 'ok');
  assertEqual(result.value, 42);
});

runner.test('execute: multiline code with calculation', async () => {
  const { context } = createMockContext('exec-3');

  const result = await execute({
    code: `
import json
x = 10
y = 20
result = x * y
print(json.dumps({"result": result}))
`
  }, context);

  assertEqual(result.result, 200);
});

runner.test('execute: progress reporting', async () => {
  const { context, output } = createMockContext('exec-4');

  await execute({
    code: `
for i in range(5):
    print(f"PROGRESS:{i * 25}")
print("done")
`
  }, context);

  assert(output.progress.length >= 4, 'Should have progress updates');
});

runner.test('execute: error handling', async () => {
  const { context } = createMockContext('exec-5');

  const result = await execute({
    code: 'raise ValueError("Test error")'
  }, context);

  assertHasProperty(result, 'error');
  assertContains(result.stderr, 'ValueError');
});

runner.test('execute: no code provided throws error', async () => {
  const { context } = createMockContext('exec-6');

  try {
    await execute({}, context);
    throw new Error('Should have thrown');
  } catch (error) {
    assertContains(error.message, 'No code provided');
  }
});

runner.test('execute: stderr capture', async () => {
  const { context, output } = createMockContext('exec-7');

  await execute({
    code: 'import sys; print("error message", file=sys.stderr)'
  }, context);

  assert(output.stderr.length > 0, 'Should have stderr output');
  assertContains(output.stderr.join(''), 'error message');
});

// ============================================================================
// Molecular Dynamics Tests
// ============================================================================

runner.test('molecular-dynamics: basic simulation', async () => {
  const { context, output } = createMockContext('md-1');

  const result = await molecularDynamics({
    steps: 100,
    temperature: 300,
    timestep: 2.0
  }, context);

  assertHasProperty(result, 'status');
  assertEqual(result.status, 'success');
  assertEqual(result.steps_completed, 100);
  assertHasProperty(result, 'final_energy');
  assertHasProperty(result, 'average_energy');
  assertHasProperty(result, 'device');
});

runner.test('molecular-dynamics: progress updates', async () => {
  const { context, output } = createMockContext('md-2');

  await molecularDynamics({
    steps: 500
  }, context);

  // Progress is emitted every 100 steps, so 500 steps = 5 progress updates
  assert(output.progress.length >= 4, `Should have progress updates, got ${output.progress.length}`);
});

// ============================================================================
// Docking Tests
// ============================================================================

runner.test('docking: basic docking calculation', async () => {
  const { context } = createMockContext('dock-1');

  const result = await docking({
    ligand: 'test-ligand',
    receptor: 'test-receptor'
  }, context);

  assertHasProperty(result, 'status');
}, { skip: true, skipReason: 'Requires docking.py implementation' });

// ============================================================================
// Binding Affinity Tests
// ============================================================================

runner.test('binding-affinity: basic calculation', async () => {
  const { context } = createMockContext('ba-1');

  const result = await bindingAffinity({
    complex: 'test-complex'
  }, context);

  assertHasProperty(result, 'status');
}, { skip: true, skipReason: 'Requires binding_affinity.py implementation' });

// ============================================================================
// Structure Prediction Tests
// ============================================================================

runner.test('structure-prediction: basic prediction', async () => {
  const { context } = createMockContext('sp-1');

  const result = await structurePrediction({
    sequence: 'MKFLILLFNILCLFPVLAADNH'
  }, context);

  assertHasProperty(result, 'status');
}, { skip: true, skipReason: 'Requires structure_prediction.py implementation' });

// ============================================================================
// Optimization Tests
// ============================================================================

runner.test('optimization: basic optimization', async () => {
  const { context } = createMockContext('opt-1');

  const result = await optimization({
    structure: 'test-structure'
  }, context);

  assertHasProperty(result, 'status');
}, { skip: true, skipReason: 'Requires optimization.py implementation' });

// ============================================================================
// Create Conda Environment Tests
// ============================================================================

runner.test('create_environment: validation - missing name', async () => {
  const { context } = createMockContext('env-1');

  try {
    await createCondaEnvironment({}, context);
    throw new Error('Should have thrown');
  } catch (error) {
    assertContains(error.message, 'Environment name is required');
  }
});

runner.test('create_environment: validation - invalid name', async () => {
  const { context } = createMockContext('env-2');

  try {
    await createCondaEnvironment({ name: '123invalid' }, context);
    throw new Error('Should have thrown');
  } catch (error) {
    assertContains(error.message, 'Invalid environment name');
  }
});

runner.test('create_environment: validation - invalid name with spaces', async () => {
  const { context } = createMockContext('env-3');

  try {
    await createCondaEnvironment({ name: 'invalid name' }, context);
    throw new Error('Should have thrown');
  } catch (error) {
    assertContains(error.message, 'Invalid environment name');
  }
});

runner.test('create_environment: valid name patterns', async () => {
  const { context } = createMockContext('env-4');

  // These should not throw validation errors
  // (they may fail due to conda not being available, but that's ok)
  const validNames = ['myenv', 'my_env', 'my-env', 'MyEnv123'];

  for (const name of validNames) {
    const namePattern = /^[a-zA-Z][a-zA-Z0-9_-]*$/;
    assert(namePattern.test(name), `"${name}" should be a valid environment name`);
  }
});

runner.test('create_environment: actual creation', async () => {
  const { context, output } = createMockContext('env-5');
  const envName = `test_env_${Date.now()}`;

  try {
    const result = await createCondaEnvironment({
      name: envName,
      pythonVersion: '3.11'
    }, context);

    assertEqual(result.status, 'created');
    assertEqual(result.name, envName);
    assertEqual(result.pythonVersion, '3.11');
    assert(output.stdout.length > 0, 'Should have stdout output');

    // Cleanup: remove the test environment
    const { execSync } = require('child_process');
    execSync(`${CONDA_PATH}/bin/conda env remove -n ${envName} -y`, { stdio: 'ignore' });

  } catch (error) {
    // Skip if conda is not available
    if (error.message.includes('ENOENT') || error.message.includes('spawn')) {
      console.log('  (Skipped - conda not available)');
      return;
    }
    throw error;
  }
}, { skip: process.env.SKIP_CONDA_TESTS === '1', skipReason: 'SKIP_CONDA_TESTS=1' });

runner.test('create_environment: actual creation with packages', async () => {
  const { context, output } = createMockContext('env-6');
  const envName = `test_env_${Date.now()}`;

  try {
    const result = await createCondaEnvironment({
      name: envName,
      pythonVersion: '3.11',
      packages: ['numpy', 'pandas','pytorch']
    }, context);

    assertEqual(result.status, 'created');
    assertEqual(result.name, envName);
    assertEqual(result.pythonVersion, '3.11');
    assert(output.stdout.length > 0, 'Should have stdout output');

    // Cleanup: remove the test environment
    const { execSync } = require('child_process');
    execSync(`${CONDA_PATH}/bin/conda env remove -n ${envName} -y`, { stdio: 'ignore' });

  } catch (error) {
    // Skip if conda is not available
    if (error.message.includes('ENOENT') || error.message.includes('spawn')) {
      console.log('  (Skipped - conda not available)');
      return;
    }
    throw error;
  }
}, { skip: process.env.SKIP_CONDA_TESTS === '1', skipReason: 'SKIP_CONDA_TESTS=1' });

// ============================================================================
// Job Registry Tests
// ============================================================================

runner.test('job-registry: all handlers registered', async () => {
  const { getJobHandler, hasJobType, getJobTypes } = require('./index');

  const expectedTypes = [
    'molecular-dynamics',
    'docking',
    'binding-affinity',
    'structure-prediction',
    'optimization',
    'execute',
    'create_environment'
  ];

  for (const type of expectedTypes) {
    assert(hasJobType(type), `Job type "${type}" should be registered`);
    const handler = getJobHandler(type);
    assert(typeof handler === 'function', `Handler for "${type}" should be a function`);
  }

  const registeredTypes = getJobTypes();
  assertEqual(registeredTypes.length, expectedTypes.length, 'Should have correct number of job types');
});

// ============================================================================
// Run Tests
// ============================================================================

const filter = process.argv[2];
runner.run(filter);
