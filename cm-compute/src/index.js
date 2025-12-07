const express = require('express');
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');
const { Client } = require('elasticsearch');
const { v4: uuidv4 } = require('uuid');
const { fork } = require('child_process');
const ComputeQueue = require('./queue');
const ComputeWorker = require('./worker');

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

    // Add to queue
    computeQueue.enqueue(job, async (result) => {
      // Update job status in database
      await pgPool.query(
        'UPDATE compute_jobs SET status = $1, result = $2, completed_at = $3 WHERE id = $4',
        [result.error ? 'failed' : 'completed', JSON.stringify(result), new Date().toISOString(), jobId]
      );
    });

    res.json({
      jobId,
      status: 'queued',
      position: computeQueue.getPosition(jobId)
    });
  } catch (error) {
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
    // Initialize database
    await initDatabase();
    console.log('Database initialized');

    // Test Elasticsearch connection
    await esClient.ping();
    console.log('Connected to Elasticsearch');

    // Start compute workers
    computeQueue.startWorkers(4); // 4 parallel workers
    console.log('Compute workers started');

    // Remove existing socket if it exists
    if (fs.existsSync(SOCKET_PATH)) {
      fs.unlinkSync(SOCKET_PATH);
    }

    // Start HTTP server (for compatibility)
    app.listen(HTTP_PORT, () => {
      console.log(`cm-compute HTTP server listening on port ${HTTP_PORT}`);
    });

    // Start Unix socket server
    const server = app.listen(SOCKET_PATH, () => {
      fs.chmodSync(SOCKET_PATH, '0666');
      console.log(`cm-compute daemon listening on ${SOCKET_PATH}`);
    });

    // Graceful shutdown
    process.on('SIGTERM', async () => {
      console.log('SIGTERM received, shutting down gracefully');
      server.close();
      await pgPool.end();
      await esClient.close();
      process.exit(0);
    });

  } catch (error) {
    console.error('Failed to start cm-compute:', error);
    process.exit(1);
  }
}

start();

module.exports = app;
