const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const cors = require('cors');
const { Pool } = require('pg');
const { Client } = require('elasticsearch');
const axios = require('axios');

const PORT = process.env.PORT || 3000;
const COMPUTE_URL = process.env.COMPUTE_URL || 'http://localhost:3001';

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

// Express app
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Serve static files from Vue build
app.use(express.static(path.join(__dirname, '../client/dist')));

// API Routes

// Notebooks
app.get('/api/notebooks', async (req, res) => {
  try {
    const result = await pgPool.query(
      'SELECT id, name, created_at, updated_at FROM notebooks ORDER BY updated_at DESC'
    );
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching notebooks:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/notebooks/:id', async (req, res) => {
  try {
    const result = await pgPool.query(
      'SELECT * FROM notebooks WHERE id = $1',
      [req.params.id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Notebook not found' });
    }

    res.json(result.rows[0]);
  } catch (error) {
    console.error('Error fetching notebook:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/notebooks', async (req, res) => {
  const { name, cells } = req.body;

  try {
    const result = await pgPool.query(
      'INSERT INTO notebooks (name, cells) VALUES ($1, $2) RETURNING *',
      [name, JSON.stringify(cells || [])]
    );
    res.json(result.rows[0]);
  } catch (error) {
    console.error('Error creating notebook:', error);
    res.status(500).json({ error: error.message });
  }
});

app.put('/api/notebooks/:id', async (req, res) => {
  const { name, cells } = req.body;

  try {
    const result = await pgPool.query(
      'UPDATE notebooks SET name = $1, cells = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $3 RETURNING *',
      [name, JSON.stringify(cells), req.params.id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Notebook not found' });
    }

    res.json(result.rows[0]);
  } catch (error) {
    console.error('Error updating notebook:', error);
    res.status(500).json({ error: error.message });
  }
});

// Search
app.get('/api/search', async (req, res) => {
  const { q, type } = req.query;

  if (!q) {
    return res.status(400).json({ error: 'Query required' });
  }

  try {
    const result = await esClient.search({
      index: type || 'molecules',
      body: {
        query: {
          multi_match: {
            query: q,
            fields: ['name^2', 'description', 'formula', 'smiles']
          }
        }
      }
    });

    res.json({
      hits: result.hits.hits.map(hit => ({
        id: hit._id,
        ...hit._source,
        score: hit._score
      }))
    });
  } catch (error) {
    console.error('Error searching:', error);
    res.status(500).json({ error: error.message });
  }
});

// Autocomplete
app.get('/api/autocomplete', async (req, res) => {
  const { q, field = 'name' } = req.query;

  if (!q) {
    return res.json({ suggestions: [] });
  }

  try {
    const result = await esClient.search({
      index: 'molecules',
      body: {
        suggest: {
          autocomplete: {
            prefix: q,
            completion: {
              field: field,
              size: 10
            }
          }
        }
      }
    });

    res.json({
      suggestions: result.suggest.autocomplete[0].options.map(opt => opt.text)
    });
  } catch (error) {
    console.error('Error autocomplete:', error);
    res.status(500).json({ error: error.message });
  }
});

// Compute proxy
app.post('/api/compute', async (req, res) => {
  try {
    const response = await axios.post(`${COMPUTE_URL}/compute`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error submitting compute job:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/compute/:jobId', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/compute/${req.params.jobId}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching compute job:', error);
    res.status(500).json({ error: error.message });
  }
});

// WebSocket for real-time updates
wss.on('connection', (ws) => {
  console.log('Client connected');

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message);

      if (data.type === 'execute_cell') {
        // Execute cell and stream results
        const { cellId, code } = data;

        // Submit to compute
        const response = await axios.post(`${COMPUTE_URL}/compute`, {
          type: 'execute',
          params: { code }
        });

        ws.send(JSON.stringify({
          type: 'cell_result',
          cellId,
          jobId: response.data.jobId
        }));
      }
    } catch (error) {
      ws.send(JSON.stringify({
        type: 'error',
        error: error.message
      }));
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

// Initialize database schema
async function initDatabase() {
  await pgPool.query(`
    CREATE TABLE IF NOT EXISTS notebooks (
      id SERIAL PRIMARY KEY,
      name VARCHAR(255) NOT NULL,
      cells JSONB DEFAULT '[]',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS molecules (
      id SERIAL PRIMARY KEY,
      name VARCHAR(255) NOT NULL,
      formula VARCHAR(255),
      smiles TEXT,
      structure JSONB,
      properties JSONB,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_notebooks_updated ON notebooks(updated_at);
    CREATE INDEX IF NOT EXISTS idx_molecules_name ON molecules(name);
  `);
}

// Catch-all route - serve Vue app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../client/dist/index.html'));
});

// Start server
async function start() {
  try {
    await initDatabase();
    console.log('Database initialized');

    await esClient.ping();
    console.log('Connected to Elasticsearch');

    server.listen(PORT, () => {
      console.log(`cm-view server listening on port ${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start cm-view:', error);
    process.exit(1);
  }
}

start();

module.exports = app;
