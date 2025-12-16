const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const fs = require('fs').promises;
const cors = require('cors');
const { Pool } = require('pg');
const { Client } = require('elasticsearch');
const axios = require('axios');

// Workspace directory for user files
const WORKSPACE_DIR = process.env.WORKSPACE_DIR || path.join(__dirname, '../../workspace');

const PORT = process.env.PORT || 3000;
const COMPUTE_URL = process.env.COMPUTE_URL || 'http://0.0.0.0:3001';
const COMPUTE_WS_URL = process.env.COMPUTE_WS_URL || 'ws://0.0.0.0:3001/ws';

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
const wss = new WebSocket.Server({ server, path: '/ws' });

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Serve static files from Vue build
app.use(express.static(path.join(__dirname, '../client/dist')));

// API Routes

// ================== Workspaces (replaces Notebooks) ==================

// Helper to count files in workspace directory
async function countFilesInDir(dirPath) {
  let count = 0;
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
      if (entry.isDirectory()) {
        count += await countFilesInDir(path.join(dirPath, entry.name));
      } else {
        count++;
      }
    }
  } catch {
    // Directory doesn't exist or not accessible
  }
  return count;
}

// Helper to get workspace stats (environments + files)
async function getWorkspaceStats(workspaceId) {
  const stats = {
    pythonEnvs: 0,
    cppEnvs: 0,
    vendorEnvs: 0,
    files: 0
  };

  try {
    // Get environment counts from compute service
    const [pythonRes, cppRes, vendorRes] = await Promise.allSettled([
      axios.get(`${COMPUTE_URL}/environments`),
      axios.get(`${COMPUTE_URL}/cpp-environments`),
      axios.get(`${COMPUTE_URL}/vendor-environments`)
    ]);

    if (pythonRes.status === 'fulfilled') {
      stats.pythonEnvs = pythonRes.value.data?.environments?.length || 0;
    }
    if (cppRes.status === 'fulfilled') {
      stats.cppEnvs = cppRes.value.data?.environments?.length || 0;
    }
    if (vendorRes.status === 'fulfilled') {
      stats.vendorEnvs = vendorRes.value.data?.environments?.length || 0;
    }

    // Count files in this workspace's directory
    if (workspaceId) {
      const workspaceDir = getWorkspaceDir(workspaceId);
      stats.files = await countFilesInDir(workspaceDir);
    }
  } catch (error) {
    console.error('Error getting workspace stats:', error.message);
  }

  return stats;
}

// List all workspaces with stats
app.get('/api/workspaces', async (req, res) => {
  try {
    const result = await pgPool.query(
      'SELECT id, name, created_at, updated_at FROM notebooks ORDER BY updated_at DESC'
    );

    // Get stats for each workspace
    const workspaces = await Promise.all(
      result.rows.map(async (workspace) => ({
        ...workspace,
        stats: await getWorkspaceStats(workspace.id)
      }))
    );

    res.json(workspaces);
  } catch (error) {
    console.error('Error fetching workspaces:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get single workspace
app.get('/api/workspaces/:id', async (req, res) => {
  try {
    const result = await pgPool.query(
      'SELECT * FROM notebooks WHERE id = $1',
      [req.params.id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Workspace not found' });
    }

    const workspace = result.rows[0];
    workspace.stats = await getWorkspaceStats(workspace.id);

    res.json(workspace);
  } catch (error) {
    console.error('Error fetching workspace:', error);
    res.status(500).json({ error: error.message });
  }
});

// Create workspace
app.post('/api/workspaces', async (req, res) => {
  const { name, cells } = req.body;

  try {
    const result = await pgPool.query(
      'INSERT INTO notebooks (name, cells) VALUES ($1, $2) RETURNING *',
      [name, JSON.stringify(cells || [])]
    );

    const workspace = result.rows[0];

    // Create workspace directory with example files
    await createExampleFiles(workspace.id);

    res.json(workspace);
  } catch (error) {
    console.error('Error creating workspace:', error);
    res.status(500).json({ error: error.message });
  }
});

// Update workspace
app.put('/api/workspaces/:id', async (req, res) => {
  const { name, cells } = req.body;

  try {
    const result = await pgPool.query(
      'UPDATE notebooks SET name = $1, cells = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $3 RETURNING *',
      [name, JSON.stringify(cells), req.params.id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Workspace not found' });
    }

    res.json(result.rows[0]);
  } catch (error) {
    console.error('Error updating workspace:', error);
    res.status(500).json({ error: error.message });
  }
});

// Delete workspace
app.delete('/api/workspaces/:id', async (req, res) => {
  try {
    const result = await pgPool.query(
      'DELETE FROM notebooks WHERE id = $1 RETURNING id',
      [req.params.id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Workspace not found' });
    }

    // Delete workspace directory
    const workspaceDir = getWorkspaceDir(req.params.id);
    try {
      await fs.rm(workspaceDir, { recursive: true });
    } catch (e) {
      // Ignore if directory doesn't exist
    }

    res.json({ success: true, id: result.rows[0].id });
  } catch (error) {
    console.error('Error deleting workspace:', error);
    res.status(500).json({ error: error.message });
  }
});

// Legacy notebook endpoints (kept for backwards compatibility)
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

// ================== File Browser API ==================

// Example files for new workspaces
const EXAMPLE_FILES = {
  'file.hpp': `#pragma once

// Example header file
// This can be included by both example.cpp and example.cell.cpp

#include <string>
#include <vector>

namespace example {

inline std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

inline std::vector<int> range(int n) {
    std::vector<int> result;
    for (int i = 0; i < n; ++i) {
        result.push_back(i);
    }
    return result;
}

} // namespace example
`,

  'example.cpp': `// Single-file C++ example
// This file is executed as a single unit (no cells)
// It includes file.hpp from the same directory

#include <iostream>
#include "file.hpp"

int main() {
    // Use the greet function from our header
    std::cout << example::greet("World") << std::endl;

    // Use the range function
    auto numbers = example::range(5);
    std::cout << "Numbers: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    return 0;
}
`,

  'example.cell.cpp': `// %% Cell 1 - Include and Setup
// Cell-based C++ example
// Each cell is compiled and executed independently

#include <iostream>
#include "file.hpp"

int main() {
    std::cout << example::greet("Cell 1") << std::endl;
    return 0;
}

// %% Cell 2 - Using Range
#include <iostream>
#include "file.hpp"

int main() {
    auto nums = example::range(10);
    int sum = 0;
    for (int n : nums) sum += n;
    std::cout << "Sum of 0-9: " << sum << std::endl;
    return 0;
}
`,

  'utils.py': `"""
Example Python module
This can be imported by other Python files in the workspace
"""

def greet(name: str) -> str:
    """Return a greeting message"""
    return f"Hello, {name}!"

def factorial(n: int) -> int:
    """Calculate factorial recursively"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> list[int]:
    """Generate first n Fibonacci numbers"""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    fibs = [0, 1]
    for _ in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

class Counter:
    """Simple counter class"""
    def __init__(self, start: int = 0):
        self.value = start

    def increment(self, by: int = 1) -> int:
        self.value += by
        return self.value

    def decrement(self, by: int = 1) -> int:
        self.value -= by
        return self.value
`,

  'example.py': `# Single-file Python example
# This file is executed as a single unit (no cells)
# It imports utils.py from the same directory

from utils import greet, factorial, fibonacci, Counter

# Use the greet function
print(greet("World"))

# Calculate some factorials
for i in range(1, 6):
    print(f"{i}! = {factorial(i)}")

# Generate Fibonacci sequence
fibs = fibonacci(10)
print(f"First 10 Fibonacci numbers: {fibs}")

# Use the Counter class
counter = Counter(10)
print(f"Counter starts at: {counter.value}")
counter.increment(5)
print(f"After +5: {counter.value}")
counter.decrement(3)
print(f"After -3: {counter.value}")
`,

  'example.cell.py': `# %% Cell 1 - Basic Import
# Cell-based Python example
# Each cell can be run independently

from utils import greet

print(greet("Cell 1"))
print("This is the first cell")

# %% Cell 2 - Math Functions
from utils import factorial, fibonacci

# Calculate factorials
for n in [5, 7, 10]:
    print(f"{n}! = {factorial(n)}")

# Show Fibonacci sequence
print(f"Fibonacci(15): {fibonacci(15)}")

# %% Cell 3 - Using Classes
from utils import Counter

# Create and use a counter
c = Counter(100)
print(f"Start: {c.value}")

for i in range(5):
    c.increment(i * 2)
    print(f"After +{i*2}: {c.value}")
`
};

// Ensure workspace directory exists for a specific workspace
async function ensureWorkspaceDir(workspaceId = null) {
  const dir = workspaceId ? path.join(WORKSPACE_DIR, String(workspaceId)) : WORKSPACE_DIR;
  try {
    await fs.access(dir);
  } catch {
    await fs.mkdir(dir, { recursive: true });
  }
  return dir;
}

// Create example files for a new workspace
async function createExampleFiles(workspaceId) {
  const workspaceDir = await ensureWorkspaceDir(workspaceId);

  for (const [filename, content] of Object.entries(EXAMPLE_FILES)) {
    const filePath = path.join(workspaceDir, filename);
    try {
      // Only create if doesn't exist
      await fs.access(filePath);
    } catch {
      await fs.writeFile(filePath, content, 'utf-8');
    }
  }

  // Create __init__.py for Python package imports
  const initPath = path.join(workspaceDir, '__init__.py');
  try {
    await fs.access(initPath);
  } catch {
    await fs.writeFile(initPath, '# Workspace package - auto-generated\n', 'utf-8');
  }
}

// Get workspace directory path
function getWorkspaceDir(workspaceId) {
  return path.join(WORKSPACE_DIR, String(workspaceId));
}

// Build file tree recursively
async function buildFileTree(dirPath, basePath = '') {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const items = [];

  for (const entry of entries) {
    // Skip hidden files and node_modules
    if (entry.name.startsWith('.') || entry.name === 'node_modules') {
      continue;
    }

    const relativePath = basePath ? `${basePath}/${entry.name}` : entry.name;
    const fullPath = path.join(dirPath, entry.name);

    if (entry.isDirectory()) {
      const children = await buildFileTree(fullPath, relativePath);
      items.push({
        name: entry.name,
        path: relativePath,
        type: 'folder',
        children
      });
    } else {
      const stats = await fs.stat(fullPath);
      items.push({
        name: entry.name,
        path: relativePath,
        type: 'file',
        size: stats.size,
        modified: stats.mtime
      });
    }
  }

  // Sort: folders first, then files, alphabetically
  return items.sort((a, b) => {
    if (a.type !== b.type) return a.type === 'folder' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

// Validate path to prevent directory traversal (workspace-specific)
function validateWorkspacePath(workspaceId, userPath) {
  const workspaceDir = getWorkspaceDir(workspaceId);
  const resolved = path.resolve(workspaceDir, userPath);
  if (!resolved.startsWith(path.resolve(workspaceDir))) {
    throw new Error('Invalid path');
  }
  return resolved;
}

// ================== Workspace-specific File API ==================

// List files in a workspace
app.get('/api/workspaces/:workspaceId/files', async (req, res) => {
  try {
    const workspaceDir = await ensureWorkspaceDir(req.params.workspaceId);
    const files = await buildFileTree(workspaceDir);
    res.json({ files });
  } catch (error) {
    console.error('Error listing files:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get file content in a workspace
app.get('/api/workspaces/:workspaceId/files/:path(*)', async (req, res) => {
  try {
    const filePath = validateWorkspacePath(req.params.workspaceId, req.params.path);
    const stats = await fs.stat(filePath);

    if (stats.isDirectory()) {
      const children = await buildFileTree(filePath, req.params.path);
      res.json({ type: 'folder', children });
    } else {
      const content = await fs.readFile(filePath, 'utf-8');
      res.json({
        type: 'file',
        content,
        size: stats.size,
        modified: stats.mtime
      });
    }
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'File not found' });
    } else {
      console.error('Error reading file:', error);
      res.status(500).json({ error: error.message });
    }
  }
});

// Create file or folder in a workspace
app.post('/api/workspaces/:workspaceId/files', async (req, res) => {
  const { path: filePath, type, content = '' } = req.body;

  if (!filePath) {
    return res.status(400).json({ error: 'Path required' });
  }

  try {
    await ensureWorkspaceDir(req.params.workspaceId);
    const fullPath = validateWorkspacePath(req.params.workspaceId, filePath);

    // Check if already exists
    try {
      await fs.access(fullPath);
      return res.status(409).json({ error: 'File or folder already exists' });
    } catch {
      // Does not exist, proceed
    }

    if (type === 'folder') {
      await fs.mkdir(fullPath, { recursive: true });
      res.json({ success: true, path: filePath, type: 'folder' });
    } else {
      // Ensure parent directory exists
      const parentDir = path.dirname(fullPath);
      await fs.mkdir(parentDir, { recursive: true });
      await fs.writeFile(fullPath, content, 'utf-8');
      res.json({ success: true, path: filePath, type: 'file' });
    }
  } catch (error) {
    console.error('Error creating file:', error);
    res.status(500).json({ error: error.message });
  }
});

// Update file content or rename in a workspace
app.put('/api/workspaces/:workspaceId/files/:path(*)', async (req, res) => {
  const { content, newPath } = req.body;

  try {
    const fullPath = validateWorkspacePath(req.params.workspaceId, req.params.path);

    if (newPath !== undefined) {
      // Rename/move
      const newFullPath = validateWorkspacePath(req.params.workspaceId, newPath);
      const parentDir = path.dirname(newFullPath);
      await fs.mkdir(parentDir, { recursive: true });
      await fs.rename(fullPath, newFullPath);
      res.json({ success: true, path: newPath });
    } else if (content !== undefined) {
      // Update content
      await fs.writeFile(fullPath, content, 'utf-8');
      res.json({ success: true, path: req.params.path });
    } else {
      res.status(400).json({ error: 'Content or newPath required' });
    }
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'File not found' });
    } else {
      console.error('Error updating file:', error);
      res.status(500).json({ error: error.message });
    }
  }
});

// Delete file or folder in a workspace
app.delete('/api/workspaces/:workspaceId/files/:path(*)', async (req, res) => {
  try {
    const fullPath = validateWorkspacePath(req.params.workspaceId, req.params.path);
    const stats = await fs.stat(fullPath);

    if (stats.isDirectory()) {
      await fs.rm(fullPath, { recursive: true });
    } else {
      await fs.unlink(fullPath);
    }

    res.json({ success: true });
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'File not found' });
    } else {
      console.error('Error deleting file:', error);
      res.status(500).json({ error: error.message });
    }
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

// ================== Conda Environment Proxy ==================

// List environments
app.get('/api/environments', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/environments`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching environments:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get available Python versions
app.get('/api/environments/python-versions', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/environments/python-versions`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching Python versions:', error);
    res.status(500).json({ error: error.message });
  }
});

// Create environment - returns job ID for WebSocket subscription
app.post('/api/environments', async (req, res) => {
  try {
    // Make request to compute service
    const response = await axios({
      method: 'POST',
      url: `${COMPUTE_URL}/environments`,
      data: req.body,
      headers: { 'Content-Type': 'application/json' }
    });

    // Return job info - client subscribes via WebSocket for progress
    res.json(response.data);
  } catch (error) {
    console.error('Error creating environment:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Delete environment
app.delete('/api/environments/:name', async (req, res) => {
  try {
    const response = await axios.delete(`${COMPUTE_URL}/environments/${req.params.name}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting environment:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get packages in environment
app.get('/api/environments/:name/packages', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/environments/${req.params.name}/packages`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching packages:', error);
    res.status(500).json({ error: error.message });
  }
});

// Install packages
app.post('/api/environments/:name/packages', async (req, res) => {
  try {
    const response = await axios.post(`${COMPUTE_URL}/environments/${req.params.name}/packages`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error installing packages:', error);
    res.status(500).json({ error: error.message });
  }
});

// ================== C++ Environment Proxy ==================

// List C++ environments
app.get('/api/cpp-environments', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/cpp-environments`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching C++ environments:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get C++ environment details
app.get('/api/cpp-environments/:name', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/cpp-environments/${req.params.name}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching C++ environment:', error);
    if (error.response?.status === 404) {
      res.status(404).json({ error: 'Environment not found' });
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Create C++ environment - returns job ID for WebSocket subscription
app.post('/api/cpp-environments', async (req, res) => {
  try {
    const response = await axios.post(`${COMPUTE_URL}/cpp-environments`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error creating C++ environment:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Delete C++ environment
app.delete('/api/cpp-environments/:name', async (req, res) => {
  try {
    const response = await axios.delete(`${COMPUTE_URL}/cpp-environments/${req.params.name}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting C++ environment:', error);
    res.status(500).json({ error: error.message });
  }
});

// ================== Vendor Environment Proxy ==================

// List vendor environments
app.get('/api/vendor-environments', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/vendor-environments`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching vendor environments:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get vendor environment details
app.get('/api/vendor-environments/:name', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/vendor-environments/${req.params.name}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching vendor environment:', error);
    if (error.response?.status === 404) {
      res.status(404).json({ error: 'Vendor environment not found' });
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Create vendor environment - returns job ID for WebSocket subscription
app.post('/api/vendor-environments', async (req, res) => {
  try {
    const response = await axios.post(`${COMPUTE_URL}/vendor-environments`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error creating vendor environment:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Delete vendor environment
app.delete('/api/vendor-environments/:name', async (req, res) => {
  try {
    const response = await axios.delete(`${COMPUTE_URL}/vendor-environments/${req.params.name}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting vendor environment:', error);
    res.status(500).json({ error: error.message });
  }
});

// ================== Debian Package Search ==================

// Search for available debian dev packages
app.get('/api/debian-packages/search', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/debian-packages/search`, {
      params: req.query
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error searching debian packages:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get C++ compiler versions
app.get('/api/compilers', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/compilers`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching compiler versions:', error);
    res.status(500).json({ error: error.message });
  }
});

// WebSocket connection to cm-compute
let computeWs = null;
const clientToCompute = new Map(); // Map client cell requests to compute responses
const jobSubscriptions = new Map(); // Map jobId -> Set of subscribed WebSocket clients

function connectToCompute() {
  computeWs = new WebSocket(COMPUTE_WS_URL);

  computeWs.on('open', () => {
    console.log('Connected to cm-compute WebSocket');
    // Re-subscribe to any pending job subscriptions
    for (const jobId of jobSubscriptions.keys()) {
      computeWs.send(JSON.stringify({ type: 'subscribe', jobId }));
    }
  });

  computeWs.on('message', (message) => {
    try {
      const data = JSON.parse(message);

      // Forward compute responses to the appropriate client (cell execution)
      if (data.cellId && clientToCompute.has(data.cellId)) {
        const clientWs = clientToCompute.get(data.cellId);
        if (clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify(data));
        }

        // Clean up on completion
        if (data.type === 'job_result' || data.type === 'error') {
          clientToCompute.delete(data.cellId);
        }
      }

      // Forward job output to subscribed clients
      if (data.jobId && jobSubscriptions.has(data.jobId)) {
        const subscribers = jobSubscriptions.get(data.jobId);
        for (const clientWs of subscribers) {
          if (clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(JSON.stringify(data));
          }
        }

        // Clean up completed/failed jobs (stream types are lowercase from JobListener)
        const streamLower = data.stream?.toLowerCase();
        if (streamLower === 'result' || streamLower === 'error' || streamLower === 'complete') {
          jobSubscriptions.delete(data.jobId);
        }
      }
    } catch (error) {
      console.error('Error parsing compute message:', error);
    }
  });

  computeWs.on('close', () => {
    console.log('Disconnected from cm-compute, reconnecting in 3s...');
    setTimeout(connectToCompute, 3000);
  });

  computeWs.on('error', (error) => {
    console.error('Compute WebSocket error:', error.message);
  });
}

// Connect to compute service
connectToCompute();

// WebSocket for client connections
wss.on('connection', (ws) => {
  console.log('Client connected');

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message);

      if (data.type === 'execute_cell') {
        const { cellId, code, environment } = data;

        // Track this request
        clientToCompute.set(cellId, ws);

        // Forward to compute via WebSocket
        if (computeWs && computeWs.readyState === WebSocket.OPEN) {
          computeWs.send(JSON.stringify({
            type: 'execute',
            cellId,
            code,
            environment
          }));
        } else {
          // Fallback to HTTP if WebSocket not connected
          const response = await axios.post(`${COMPUTE_URL}/compute`, {
            type: 'execute',
            params: { code, environment }
          });

          ws.send(JSON.stringify({
            type: 'job_accepted',
            cellId,
            jobId: response.data.jobId
          }));
        }
      } else if (data.type === 'subscribe') {
        // Subscribe to job output
        const { jobId } = data;
        if (!jobSubscriptions.has(jobId)) {
          jobSubscriptions.set(jobId, new Set());
        }
        jobSubscriptions.get(jobId).add(ws);

        // Forward subscription to cm-compute
        if (computeWs && computeWs.readyState === WebSocket.OPEN) {
          computeWs.send(JSON.stringify({
            type: 'subscribe',
            jobId
          }));
        }

        // Acknowledge subscription
        ws.send(JSON.stringify({
          type: 'subscribed',
          jobId
        }));

      } else if (data.type === 'unsubscribe') {
        // Unsubscribe from job output
        const { jobId } = data;
        if (jobSubscriptions.has(jobId)) {
          jobSubscriptions.get(jobId).delete(ws);
          if (jobSubscriptions.get(jobId).size === 0) {
            jobSubscriptions.delete(jobId);
          }
        }

      } else if (data.type === 'cancel') {
        // Forward cancel request
        if (computeWs && computeWs.readyState === WebSocket.OPEN) {
          computeWs.send(JSON.stringify({
            type: 'cancel',
            jobId: data.jobId
          }));
        }
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
    // Clean up any pending requests for this client
    for (const [cellId, clientWs] of clientToCompute.entries()) {
      if (clientWs === ws) {
        clientToCompute.delete(cellId);
      }
    }
    // Clean up job subscriptions for this client
    for (const [jobId, subscribers] of jobSubscriptions.entries()) {
      subscribers.delete(ws);
      if (subscribers.size === 0) {
        jobSubscriptions.delete(jobId);
      }
    }
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
