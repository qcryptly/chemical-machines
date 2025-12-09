const express = require('express');
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');
const { Client } = require('elasticsearch');
const { v4: uuidv4 } = require('uuid');
const { fork, spawn, execSync } = require('child_process');
const ComputeQueue = require('./queue');
const { createMainLogger } = require('./logger');

// Initialize main logger
const logger = createMainLogger();
const ComputeWorker = require('./worker');

// Conda environment management
const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';

function getCondaEnvs() {
  try {
    const result = execSync(`${CONDA_PATH}/bin/conda env list --json`, { encoding: 'utf-8' });
    const data = JSON.parse(result);
    return data.envs.map(envPath => {
      const name = path.basename(envPath);
      const isBase = envPath === CONDA_PATH;

      // Get Python version for this environment
      let pythonVersion = null;
      try {
        const pythonPath = isBase
          ? `${CONDA_PATH}/bin/python`
          : `${envPath}/bin/python`;
        if (fs.existsSync(pythonPath)) {
          pythonVersion = execSync(`${pythonPath} --version 2>&1`, { encoding: 'utf-8' }).trim().replace('Python ', '');
        }
      } catch (e) {
        pythonVersion = 'unknown';
      }

      // Get installed packages count
      let packageCount = 0;
      try {
        const pkgResult = execSync(`${CONDA_PATH}/bin/conda list -n ${isBase ? 'base' : name} --json`, { encoding: 'utf-8' });
        packageCount = JSON.parse(pkgResult).length;
      } catch (e) {}

      return {
        name: isBase ? 'base' : name,
        path: envPath,
        pythonVersion,
        packageCount,
        isBase
      };
    });
  } catch (error) {
    console.error('Error listing conda environments:', error);
    return [];
  }
}

function getAvailablePythonVersions() {
  try {
    const result = execSync(`${CONDA_PATH}/bin/conda search python --json`, { encoding: 'utf-8' });
    const data = JSON.parse(result);
    const versions = [...new Set(data.python.map(p => p.version))];
    // Filter to major.minor versions and sort descending
    const majorMinor = [...new Set(versions.map(v => v.split('.').slice(0, 2).join('.')))];
    return majorMinor.sort((a, b) => {
      const [aMaj, aMin] = a.split('.').map(Number);
      const [bMaj, bMin] = b.split('.').map(Number);
      return bMaj - aMaj || bMin - aMin;
    }).slice(0, 10); // Return top 10 versions
  } catch (error) {
    console.error('Error getting Python versions:', error);
    return ['3.12', '3.11', '3.10', '3.9', '3.8'];
  }
}

const SOCKET_PATH = process.env.COMPUTE_SOCKET_PATH || '/var/run/cm-compute.sock';
const HTTP_PORT = process.env.COMPUTE_HTTP_PORT || 3001;

// Database connections
const pgPool = new Pool({
  host: process.env.POSTGRES_HOST || 'localhost',
  database: process.env.POSTGRES_DB || 'chemicalmachines',
  user: process.env.POSTGRES_USER || 'cmuser',
  password: process.env.POSTGRES_PASSWORD || 'changeme',
  port: 5432,
});

const esClient = new Client({
  node: `http://${process.env.ELASTICSEARCH_HOST || 'localhost'}:${process.env.ELASTICSEARCH_PORT || 9200}`,
});

// Initialize compute queue
const computeQueue = new ComputeQueue();

// Express app
const app = express();
app.use(express.json({ limit: '50mb' }));

// Request logging middleware
app.use((req, res, next) => {
  const startTime = Date.now();
  logger.request(req.method, req.path, req.body && Object.keys(req.body).length > 0 ? req.body : null);

  // Capture response
  const originalSend = res.send;
  res.send = function(body) {
    const duration = Date.now() - startTime;
    logger.response(req.method, req.path, res.statusCode, { duration: `${duration}ms` });
    return originalSend.call(this, body);
  };

  next();
});

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    queue: computeQueue.getStats(),
    timestamp: new Date().toISOString()
  });
});

// Submit compute job
app.post('/compute', async (req, res) => {
  const { type, params, priority = 5 } = req.body;

  if (!type) {
    return res.status(400).json({ error: 'Compute type required' });
  }

  const jobId = uuidv4();
  const job = {
    id: jobId,
    type,
    params,
    priority,
    status: 'queued',
    createdAt: new Date().toISOString(),
  };

  try {
    // Store job in database
    await pgPool.query(
      'INSERT INTO compute_jobs (id, type, params, priority, status, created_at) VALUES ($1, $2, $3, $4, $5, $6)',
      [job.id, job.type, JSON.stringify(job.params), job.priority, job.status, job.createdAt]
    );

    logger.job('QUEUED', jobId, { type, params, priority });

    // Add to queue
    computeQueue.enqueue(job, async (result) => {
      const status = result.error ? 'failed' : 'completed';
      logger.job(status.toUpperCase(), jobId, result.error ? { error: result.error } : { success: true });

      // Update job status in database
      await pgPool.query(
        'UPDATE compute_jobs SET status = $1, result = $2, completed_at = $3 WHERE id = $4',
        [status, JSON.stringify(result), new Date().toISOString(), jobId]
      );
    });

    res.json({
      jobId,
      status: 'queued',
      position: computeQueue.getPosition(jobId)
    });
  } catch (error) {
    logger.error('Error submitting job', { jobId, error: error.message, stack: error.stack });
    console.error('Error submitting job:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get job status
app.get('/compute/:jobId', async (req, res) => {
  const { jobId } = req.params;

  try {
    const result = await pgPool.query(
      'SELECT * FROM compute_jobs WHERE id = $1',
      [jobId]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const job = result.rows[0];
    const queuePosition = computeQueue.getPosition(jobId);

    res.json({
      ...job,
      queuePosition,
    });
  } catch (error) {
    console.error('Error fetching job:', error);
    res.status(500).json({ error: error.message });
  }
});

// ================== Conda Environment Management ==================

// List all conda environments
app.get('/environments', (req, res) => {
  try {
    const envs = getCondaEnvs();
    res.json({ environments: envs });
  } catch (error) {
    console.error('Error listing environments:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get available Python versions
app.get('/environments/python-versions', (req, res) => {
  try {
    const versions = getAvailablePythonVersions();
    res.json({ versions });
  } catch (error) {
    console.error('Error getting Python versions:', error);
    res.status(500).json({ error: error.message });
  }
});

// Create new conda environment
app.post('/environments', (req, res) => {
  const { name, pythonVersion = '3.12', packages = [] } = req.body;

  if (!name) {
    return res.status(400).json({ error: 'Environment name required' });
  }

  if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(name)) {
    return res.status(400).json({ error: 'Invalid environment name. Use alphanumeric characters, hyphens, and underscores.' });
  }

  // Check if environment already exists
  const existingEnvs = getCondaEnvs();
  if (existingEnvs.some(e => e.name === name)) {
    return res.status(409).json({ error: `Environment '${name}' already exists` });
  }

  logger.info('Creating conda environment', { name, pythonVersion, packages });

  // Create environment in background
  const createProcess = spawn(
    `${CONDA_PATH}/bin/conda`,
    ['create', '-n', name, `python=${pythonVersion}`, '-y', ...packages],
    { stdio: ['ignore', 'pipe', 'pipe'] }
  );

  let stdout = '';
  let stderr = '';

  createProcess.stdout.on('data', (data) => { stdout += data.toString(); });
  createProcess.stderr.on('data', (data) => { stderr += data.toString(); });

  createProcess.on('close', (code) => {
    if (code === 0) {
      logger.info('Created conda environment', { name, pythonVersion });
      console.log(`Created conda environment: ${name}`);
    } else {
      logger.error('Failed to create conda environment', { name, pythonVersion, stderr, exitCode: code });
      console.error(`Failed to create environment ${name}:`, stderr);
    }
  });

  // Return immediately with creation status
  res.json({
    status: 'creating',
    name,
    pythonVersion,
    message: `Creating environment '${name}' with Python ${pythonVersion}...`
  });
});

// Delete conda environment
app.delete('/environments/:name', (req, res) => {
  const { name } = req.params;

  if (name === 'base') {
    return res.status(400).json({ error: 'Cannot delete base environment' });
  }

  // Check if environment exists
  const existingEnvs = getCondaEnvs();
  if (!existingEnvs.some(e => e.name === name)) {
    return res.status(404).json({ error: `Environment '${name}' not found` });
  }

  try {
    logger.info('Deleting conda environment', { name });
    execSync(`${CONDA_PATH}/bin/conda env remove -n ${name} -y`, { encoding: 'utf-8' });
    logger.info('Deleted conda environment', { name });
    res.json({ status: 'deleted', name });
  } catch (error) {
    logger.error('Error deleting environment', { name, error: error.message });
    console.error(`Error deleting environment ${name}:`, error);
    res.status(500).json({ error: error.message });
  }
});

// Install packages in environment
app.post('/environments/:name/packages', (req, res) => {
  const { name } = req.params;
  const { packages, channel = null } = req.body;

  if (!packages || !Array.isArray(packages) || packages.length === 0) {
    return res.status(400).json({ error: 'Packages array required' });
  }

  logger.info('Installing packages', { environment: name, packages, channel });

  const args = ['install', '-n', name, '-y', ...packages];
  if (channel) {
    args.splice(1, 0, '-c', channel);
  }

  const installProcess = spawn(`${CONDA_PATH}/bin/conda`, args, {
    stdio: ['ignore', 'pipe', 'pipe']
  });

  let stderr = '';
  installProcess.stderr.on('data', (data) => { stderr += data.toString(); });

  installProcess.on('close', (code) => {
    if (code === 0) {
      logger.info('Installed packages', { environment: name, packages });
    } else {
      logger.error('Failed to install packages', { environment: name, packages, stderr, exitCode: code });
      console.error(`Failed to install packages in ${name}:`, stderr);
    }
  });

  res.json({
    status: 'installing',
    environment: name,
    packages,
    message: `Installing ${packages.join(', ')} in '${name}'...`
  });
});

// Get packages in environment
app.get('/environments/:name/packages', (req, res) => {
  const { name } = req.params;

  try {
    const result = execSync(`${CONDA_PATH}/bin/conda list -n ${name} --json`, { encoding: 'utf-8' });
    const packages = JSON.parse(result);
    res.json({ packages });
  } catch (error) {
    console.error(`Error listing packages for ${name}:`, error);
    res.status(500).json({ error: error.message });
  }
});

// ================== Job Management ==================

// Cancel job
app.delete('/compute/:jobId', async (req, res) => {
  const { jobId } = req.params;

  try {
    const cancelled = computeQueue.cancel(jobId);

    if (cancelled) {
      await pgPool.query(
        'UPDATE compute_jobs SET status = $1 WHERE id = $2',
        ['cancelled', jobId]
      );
      res.json({ status: 'cancelled' });
    } else {
      res.status(404).json({ error: 'Job not found or already completed' });
    }
  } catch (error) {
    console.error('Error cancelling job:', error);
    res.status(500).json({ error: error.message });
  }
});

// Initialize database schema
async function initDatabase() {
  await pgPool.query(`
    CREATE TABLE IF NOT EXISTS compute_jobs (
      id VARCHAR(36) PRIMARY KEY,
      type VARCHAR(255) NOT NULL,
      params JSONB,
      priority INTEGER DEFAULT 5,
      status VARCHAR(50) DEFAULT 'queued',
      result JSONB,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      started_at TIMESTAMP,
      completed_at TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_compute_jobs_status ON compute_jobs(status);
    CREATE INDEX IF NOT EXISTS idx_compute_jobs_created_at ON compute_jobs(created_at);
  `);
}

// Start the server
async function start() {
  try {
    logger.info('Starting cm-compute daemon');

    // Initialize database
    await initDatabase();
    logger.info('Database initialized');
    console.log('Database initialized');

    // Test Elasticsearch connection
    await esClient.ping();
    logger.info('Connected to Elasticsearch');
    console.log('Connected to Elasticsearch');

    // Start compute workers
    computeQueue.startWorkers(4); // 4 parallel workers
    logger.info('Compute workers started', { count: 4 });
    console.log('Compute workers started');

    // Remove existing socket if it exists
    if (fs.existsSync(SOCKET_PATH)) {
      fs.unlinkSync(SOCKET_PATH);
    }

    // Start HTTP server (for compatibility)
    app.listen(HTTP_PORT, () => {
      logger.info('HTTP server started', { port: HTTP_PORT });
      console.log(`cm-compute HTTP server listening on port ${HTTP_PORT}`);
    });

    // Start Unix socket server
    const server = app.listen(SOCKET_PATH, () => {
      fs.chmodSync(SOCKET_PATH, '0666');
      logger.info('Unix socket server started', { path: SOCKET_PATH });
      console.log(`cm-compute daemon listening on ${SOCKET_PATH}`);
    });

    // Graceful shutdown
    process.on('SIGTERM', async () => {
      logger.info('SIGTERM received, shutting down gracefully');
      console.log('SIGTERM received, shutting down gracefully');
      server.close();
      await pgPool.end();
      await esClient.close();
      logger.info('Shutdown complete');
      process.exit(0);
    });

  } catch (error) {
    logger.error('Failed to start cm-compute', { error: error.message, stack: error.stack });
    console.error('Failed to start cm-compute:', error);
    process.exit(1);
  }
}

start();

module.exports = app;
