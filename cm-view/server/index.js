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
// Workspace templates directory
const TEMPLATES_DIR = process.env.TEMPLATES_DIR || path.join(__dirname, '../../workspaces');

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

// ================== Health Check ==================

// Check health of all backend services
app.get('/api/health', async (req, res) => {
  const services = {
    database: { status: 'checking', message: null },
    compute: { status: 'checking', message: null },
    elasticsearch: { status: 'checking', message: null }
  };

  // Check PostgreSQL
  try {
    await pgPool.query('SELECT 1');
    services.database.status = 'healthy';
  } catch (error) {
    services.database.status = 'unhealthy';
    services.database.message = error.message;
  }

  // Check cm-compute
  try {
    const response = await axios.get(`${COMPUTE_URL}/environments`, { timeout: 3000 });
    if (response.status === 200) {
      services.compute.status = 'healthy';
    } else {
      services.compute.status = 'unhealthy';
      services.compute.message = `Unexpected status: ${response.status}`;
    }
  } catch (error) {
    services.compute.status = 'unhealthy';
    services.compute.message = error.code === 'ECONNREFUSED'
      ? 'Compute service not running'
      : error.message;
  }

  // Check Elasticsearch
  try {
    await esClient.ping();
    services.elasticsearch.status = 'healthy';
  } catch (error) {
    services.elasticsearch.status = 'unhealthy';
    services.elasticsearch.message = error.message;
  }

  // Overall status - healthy only if all services are healthy
  const allHealthy = Object.values(services).every(s => s.status === 'healthy');
  const anyHealthy = Object.values(services).some(s => s.status === 'healthy');

  const overallStatus = allHealthy ? 'healthy' : (anyHealthy ? 'degraded' : 'unhealthy');

  res.status(allHealthy ? 200 : 503).json({
    status: overallStatus,
    services,
    timestamp: new Date().toISOString()
  });
});

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

// List available workspace templates
app.get('/api/templates', async (req, res) => {
  try {
    const entries = await fs.readdir(TEMPLATES_DIR, { withFileTypes: true });
    const templates = entries
      .filter(entry => entry.isDirectory())
      .map(entry => entry.name)
      .sort((a, b) => {
        // 'default' always comes first
        if (a === 'default') return -1;
        if (b === 'default') return 1;
        return a.localeCompare(b);
      });

    res.json({ templates });
  } catch (error) {
    console.error('Error listing templates:', error);
    // Return empty list if templates dir doesn't exist
    res.json({ templates: [] });
  }
});

// Create workspace
app.post('/api/workspaces', async (req, res) => {
  const { name, cells, template = 'default' } = req.body;

  try {
    const result = await pgPool.query(
      'INSERT INTO notebooks (name, cells) VALUES ($1, $2) RETURNING *',
      [name, JSON.stringify(cells || [])]
    );

    const workspace = result.rows[0];

    // Copy template files to workspace directory
    await copyTemplateToWorkspace(workspace.id, template);

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

// Copy template directory to a new workspace
async function copyTemplateToWorkspace(workspaceId, templateName = 'default') {
  const workspaceDir = await ensureWorkspaceDir(workspaceId);
  const templateDir = path.join(TEMPLATES_DIR, templateName);

  try {
    // Check if template exists
    await fs.access(templateDir);

    // Recursively copy template to workspace
    await copyDirRecursive(templateDir, workspaceDir);
  } catch (error) {
    // If template doesn't exist, try 'default', or create empty workspace
    if (templateName !== 'default') {
      console.warn(`Template '${templateName}' not found, falling back to 'default'`);
      const defaultDir = path.join(TEMPLATES_DIR, 'default');
      try {
        await fs.access(defaultDir);
        await copyDirRecursive(defaultDir, workspaceDir);
      } catch {
        // Default doesn't exist either - create minimal workspace
        console.warn('No templates found, creating empty workspace');
        const initPath = path.join(workspaceDir, '__init__.py');
        await fs.writeFile(initPath, '# Workspace package - auto-generated\n', 'utf-8');
      }
    } else {
      // Default template doesn't exist - create minimal workspace
      console.warn('No templates found, creating empty workspace');
      const initPath = path.join(workspaceDir, '__init__.py');
      await fs.writeFile(initPath, '# Workspace package - auto-generated\n', 'utf-8');
    }
  }

  // Always copy the library reference README.md to the workspace root
  const readmePath = path.join(TEMPLATES_DIR, 'README.md');
  const destReadmePath = path.join(workspaceDir, 'README.md');

  try {
    await fs.copyFile(readmePath, destReadmePath);
    console.log(`Copied README.md to workspace ${workspaceId}`);
  } catch (error) {
    console.warn(`Failed to copy README.md to workspace ${workspaceId}:`, error.message);
  }
}

// Recursively copy a directory
async function copyDirRecursive(src, dest) {
  await fs.mkdir(dest, { recursive: true });
  const entries = await fs.readdir(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      await copyDirRecursive(srcPath, destPath);
    } else {
      await fs.copyFile(srcPath, destPath);
    }
  }
}

// Get workspace directory path
function getWorkspaceDir(workspaceId) {
  return path.join(WORKSPACE_DIR, String(workspaceId));
}

// Build file tree with lazy loading support
// maxDepth: 0 = no recursion (lazy load), 1 = one level, -1 = full recursion
// limit: max items per directory (for pagination)
async function buildFileTree(dirPath, basePath = '', options = {}) {
  const { maxDepth = 0, limit = 1000, offset = 0 } = options;

  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const items = [];

  // Filter and categorize entries
  const filtered = entries.filter(entry =>
    !entry.name.startsWith('.') && entry.name !== 'node_modules'
  );

  // Sort: folders first, then files, alphabetically
  const sorted = filtered.sort((a, b) => {
    if (a.isDirectory() !== b.isDirectory()) {
      return a.isDirectory() ? -1 : 1;
    }
    return a.name.localeCompare(b.name);
  });

  // Apply pagination
  const paginated = sorted.slice(offset, offset + limit);
  const hasMore = sorted.length > offset + limit;

  for (const entry of paginated) {
    const relativePath = basePath ? `${basePath}/${entry.name}` : entry.name;
    const fullPath = path.join(dirPath, entry.name);

    if (entry.isDirectory()) {
      // Count immediate children for folder
      let childCount = 0;
      try {
        const dirEntries = await fs.readdir(fullPath, { withFileTypes: true });
        childCount = dirEntries.filter(e =>
          !e.name.startsWith('.') && e.name !== 'node_modules'
        ).length;
      } catch {
        childCount = 0;
      }

      const folderItem = {
        name: entry.name,
        path: relativePath,
        type: 'folder',
        childCount,
        // Only load children if maxDepth allows
        children: maxDepth > 0
          ? (await buildFileTree(fullPath, relativePath, { ...options, maxDepth: maxDepth - 1 })).items
          : null
      };

      items.push(folderItem);
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

  return {
    items,
    totalCount: sorted.length,
    hasMore,
    offset,
    limit
  };
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

    // Parse query parameters for lazy loading
    const options = {
      maxDepth: parseInt(req.query.depth) || 1,  // Default: load one level
      limit: parseInt(req.query.limit) || 1000,
      offset: parseInt(req.query.offset) || 0
    };

    const result = await buildFileTree(workspaceDir, '', options);
    res.json(result);
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
      // Parse query parameters for lazy loading
      const options = {
        maxDepth: parseInt(req.query.depth) || 0,  // Default: no recursion (lazy)
        limit: parseInt(req.query.limit) || 1000,
        offset: parseInt(req.query.offset) || 0
      };

      const result = await buildFileTree(filePath, req.params.path, options);
      res.json({ type: 'folder', ...result });
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

// Copy file or folder in a workspace
app.post('/api/workspaces/:workspaceId/files/copy', async (req, res) => {
  const { sourcePath, targetPath } = req.body;

  if (!sourcePath || !targetPath) {
    return res.status(400).json({ error: 'Source and target paths required' });
  }

  try {
    const fullSourcePath = validateWorkspacePath(req.params.workspaceId, sourcePath);
    const fullTargetPath = validateWorkspacePath(req.params.workspaceId, targetPath);

    // Check if source exists
    const stats = await fs.stat(fullSourcePath);

    // Check if target already exists
    try {
      await fs.access(fullTargetPath);
      return res.status(409).json({ error: 'Target already exists' });
    } catch {
      // Target doesn't exist, proceed
    }

    // Ensure parent directory exists
    const parentDir = path.dirname(fullTargetPath);
    await fs.mkdir(parentDir, { recursive: true });

    if (stats.isDirectory()) {
      // Recursively copy directory
      await copyDir(fullSourcePath, fullTargetPath);
    } else {
      // Copy file
      await fs.copyFile(fullSourcePath, fullTargetPath);
    }

    res.json({ success: true, path: targetPath });
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'Source not found' });
    } else {
      console.error('Error copying:', error);
      res.status(500).json({ error: error.message });
    }
  }
});

// Helper function to recursively copy a directory
async function copyDir(src, dest) {
  await fs.mkdir(dest, { recursive: true });
  const entries = await fs.readdir(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      await copyDir(srcPath, destPath);
    } else {
      await fs.copyFile(srcPath, destPath);
    }
  }
}

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

// Get HTML output for a file from .out/ directory
// For .cell.py or .cell.cpp files, returns array of cell outputs (split by delimiter)
// For regular .py or .cpp files, returns single output
app.get('/api/workspaces/:workspaceId/output/:path(*)', async (req, res) => {
  try {
    const workspaceDir = getWorkspaceDir(req.params.workspaceId);
    const requestedPath = req.params.path;

    // Determine the HTML output path
    // For files in subdirectories, the .out folder is inside the subdirectory
    // e.g., subdir/myfile.cell.py -> subdir/.out/myfile.cell.py.html
    // For files in root: myfile.cell.py -> .out/myfile.cell.py.html
    const dirPart = path.dirname(requestedPath);
    const fileName = path.basename(requestedPath);
    const htmlFileName = fileName + '.html';

    let outputPath;
    if (dirPart && dirPart !== '.') {
      // File is in a subdirectory - .out is inside that directory
      outputPath = path.join(workspaceDir, dirPart, '.out', htmlFileName);
    } else {
      // File is in root - .out is at workspace root
      outputPath = path.join(workspaceDir, '.out', htmlFileName);
    }

    // Validate path to prevent directory traversal
    const resolvedPath = path.resolve(outputPath);
    const resolvedWorkspace = path.resolve(workspaceDir);
    if (!resolvedPath.startsWith(resolvedWorkspace)) {
      return res.status(403).json({ error: 'Invalid path' });
    }

    // Check if output file exists, create .out directory if needed
    try {
      await fs.access(outputPath);
    } catch {
      // Output doesn't exist - ensure .out directory exists for future writes
      const outDir = path.dirname(outputPath);
      try {
        await fs.mkdir(outDir, { recursive: true });
      } catch {
        // Ignore mkdir errors
      }
      return res.json({ exists: false, outputs: [] });
    }

    const htmlContent = await fs.readFile(outputPath, 'utf-8');

    // Check if this is a cell-based file (.cell.py or .cell.cpp)
    const isCellFile = /\.cell\.(py|cpp)$/i.test(requestedPath);

    if (isCellFile) {
      // Split HTML by cell delimiter comments
      // Expected format: <!-- CELL_DELIMITER --> or similar marker
      const cellDelimiter = /<!--\s*CELL[_-]?DELIMITER\s*-->/gi;
      const cellOutputs = htmlContent.split(cellDelimiter).map(s => s.trim());

      res.json({
        exists: true,
        isCellFile: true,
        outputs: cellOutputs
      });
    } else {
      // Single file, return as single output
      res.json({
        exists: true,
        isCellFile: false,
        outputs: [htmlContent.trim()]
      });
    }
  } catch (error) {
    console.error('Error reading output file:', error);
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

// Autocomplete proxy
app.post('/api/complete', async (req, res) => {
  try {
    const response = await axios.post(`${COMPUTE_URL}/api/complete`, req.body, {
      timeout: 2000  // 2 second timeout for autocomplete
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching autocomplete:', error);
    // Return empty completions on error (graceful degradation)
    res.status(200).json({ completions: [], error: error.message });
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

// ================== Profile Proxy ==================

// Get active profile
app.get('/api/profile', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/profile`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching profile:', error);
    res.status(500).json({ error: error.message });
  }
});

// Create or update profile
app.post('/api/profile', async (req, res) => {
  try {
    const response = await axios.post(`${COMPUTE_URL}/profile`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error saving profile:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Generate SSH key
app.post('/api/profile/:id/generate-ssh-key', async (req, res) => {
  try {
    const response = await axios.post(`${COMPUTE_URL}/profile/${req.params.id}/generate-ssh-key`);
    res.json(response.data);
  } catch (error) {
    console.error('Error generating SSH key:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Get public SSH key
app.get('/api/profile/:id/ssh-key', async (req, res) => {
  try {
    const response = await axios.get(`${COMPUTE_URL}/profile/${req.params.id}/ssh-key`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching SSH key:', error);
    if (error.response?.status === 404) {
      res.status(404).json({ error: 'No SSH key found' });
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// ================== Benchmark Database API ==================

// Search benchmark databases (Elasticsearch cache + PubChem)
app.get('/api/benchmark/search', async (req, res) => {
  const { q, sources, limit = 20 } = req.query;

  if (!q) {
    return res.status(400).json({ error: 'Query parameter "q" is required' });
  }

  try {
    // First search Elasticsearch cache
    const sourceList = sources ? sources.split(',') : ['pubchem', 'nist', 'qm9'];

    const esResult = await esClient.search({
      index: 'benchmark_molecules',
      body: {
        query: {
          bool: {
            should: [
              { match: { name: { query: q, boost: 2 } } },
              { term: { formula: { value: q.toUpperCase(), boost: 1.5 } } },
              { term: { cas: q } },
              { term: { smiles: q } },
              { term: { inchi_key: q } }
            ],
            minimum_should_match: 1,
            filter: [{ terms: { sources: sourceList } }]
          }
        },
        size: parseInt(limit)
      }
    });

    const cached = esResult.hits.hits.map(hit => ({
      ...hit._source,
      _score: hit._score
    }));

    if (cached.length > 0) {
      return res.json({
        status: 'cached',
        results: cached,
        total: cached.length
      });
    }

    // If no cached results, submit search job to cm-compute
    const response = await axios.post(`${COMPUTE_URL}/compute`, {
      type: 'benchmark_search',
      params: { query: q, sources: sourceList, limit: parseInt(limit) },
      priority: 7
    });

    res.json({
      status: 'searching',
      jobId: response.data.jobId,
      message: 'Search job submitted. Subscribe via WebSocket for results.'
    });
  } catch (error) {
    // Elasticsearch index might not exist yet
    if (error.statusCode === 404 || error.meta?.statusCode === 404) {
      // No cache, submit search job
      try {
        const response = await axios.post(`${COMPUTE_URL}/compute`, {
          type: 'benchmark_search',
          params: { query: q, sources: sources?.split(',') || ['pubchem', 'nist', 'qm9'], limit: parseInt(limit) },
          priority: 7
        });

        return res.json({
          status: 'searching',
          jobId: response.data.jobId,
          message: 'Search job submitted. Subscribe via WebSocket for results.'
        });
      } catch (jobError) {
        return res.status(500).json({ error: jobError.message });
      }
    }

    console.error('Benchmark search error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get specific molecule data (returns cached or queues fetch)
app.get('/api/benchmark/molecule/:identifier', async (req, res) => {
  const { identifier } = req.params;
  const { sources, workspaceId = '1' } = req.query;

  try {
    // Build query clauses - only include numeric fields if identifier is numeric
    const shouldClauses = [
      { term: { identifier } },
      { term: { cas: identifier } },
      { term: { smiles: identifier } },
      { term: { inchi_key: identifier } },
      { term: { 'name.keyword': identifier } },
      { match: { name: identifier } }
    ];

    // Only search cid if identifier looks like a number
    if (/^\d+$/.test(identifier)) {
      shouldClauses.push({ term: { cid: parseInt(identifier, 10) } });
    }

    // Check Elasticsearch cache first
    const esResult = await esClient.search({
      index: 'benchmark_molecules',
      body: {
        query: {
          bool: {
            should: shouldClauses,
            minimum_should_match: 1
          }
        },
        size: 1
      }
    });

    if (esResult.hits.hits.length > 0) {
      const molecule = esResult.hits.hits[0]._source;
      return res.json({
        status: 'cached',
        data: molecule,
        cached_at: molecule.cached_at
      });
    }

    // Not cached, submit fetch job
    const response = await axios.post(`${COMPUTE_URL}/compute`, {
      type: 'benchmark_fetch',
      params: {
        identifier,
        sources: sources?.split(',') || ['pubchem', 'nist', 'qm9'],
        workspaceId
      },
      priority: 8
    });

    res.json({
      status: 'fetching',
      jobId: response.data.jobId,
      message: `Fetching data for ${identifier}. Subscribe via WebSocket for results.`
    });
  } catch (error) {
    // Elasticsearch index might not exist
    if (error.statusCode === 404 || error.meta?.statusCode === 404) {
      try {
        const response = await axios.post(`${COMPUTE_URL}/compute`, {
          type: 'benchmark_fetch',
          params: {
            identifier,
            sources: sources?.split(',') || ['pubchem', 'nist', 'qm9'],
            workspaceId: workspaceId || '1'
          },
          priority: 8
        });

        return res.json({
          status: 'fetching',
          jobId: response.data.jobId,
          message: `Fetching data for ${identifier}. Subscribe via WebSocket for results.`
        });
      } catch (jobError) {
        return res.status(500).json({ error: jobError.message });
      }
    }

    console.error('Benchmark fetch error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Compare computed values with benchmark data
app.post('/api/benchmark/compare', async (req, res) => {
  const { computed, identifier } = req.body;

  if (!computed || !identifier) {
    return res.status(400).json({
      error: 'Request body must include "computed" properties and "identifier"'
    });
  }

  try {
    const response = await axios.post(`${COMPUTE_URL}/compute`, {
      type: 'benchmark_compare',
      params: { computed, identifier },
      priority: 8
    });

    res.json({
      status: 'comparing',
      jobId: response.data.jobId,
      message: 'Comparison job submitted. Subscribe via WebSocket for results.'
    });
  } catch (error) {
    console.error('Benchmark compare error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Track active indexing jobs for status reporting
const activeIndexingJobs = new Map(); // source -> { jobId, startedAt, progress }

// Trigger benchmark database sync
app.post('/api/benchmark/sync', async (req, res) => {
  const { sources = ['qm9'], workspaceId = '1' } = req.body;

  try {
    const response = await axios.post(`${COMPUTE_URL}/compute`, {
      type: 'benchmark_sync',
      params: { sources, workspaceId },
      priority: 3
    });

    const jobId = response.data.jobId;

    // Track this as an active indexing job
    for (const source of sources) {
      activeIndexingJobs.set(source, {
        source,
        jobId,
        startedAt: new Date().toISOString(),
        progress: 0,
        phase: 'starting',
        details: {}
      });
    }

    // Subscribe to job updates to track progress
    if (computeWs && computeWs.readyState === WebSocket.OPEN) {
      computeWs.send(JSON.stringify({ type: 'subscribe', jobId }));

      // Set up handler to update progress and clean up when done
      const handleSyncProgress = (message) => {
        try {
          const data = JSON.parse(message);
          if (data.jobId === jobId) {
            // Handle progress updates
            if (data.stream === 'progress' || data.type === 'progress') {
              for (const source of sources) {
                const job = activeIndexingJobs.get(source);
                if (job) {
                  job.progress = data.data || data.progress || 0;
                }
              }
            }
            // Handle download_progress stream with detailed info
            if (data.stream === 'download_progress') {
              for (const source of sources) {
                const job = activeIndexingJobs.get(source);
                if (job && data.data) {
                  job.phase = data.data.phase || job.phase;
                  job.progress = data.data.progress || job.progress;
                  job.details = data.data;
                }
              }
            }
            // Handle stdout for progress parsing
            if (data.stream === 'stdout' && typeof data.data === 'string') {
              for (const source of sources) {
                const job = activeIndexingJobs.get(source);
                if (job) {
                  job.lastOutput = data.data;
                }
              }
            }
            // Cleanup on completion
            if (data.stream === 'complete' || data.stream === 'result' || data.stream === 'error') {
              // Clean up indexing jobs
              for (const source of sources) {
                activeIndexingJobs.delete(source);
              }
              computeWs.off('message', handleSyncProgress);
            }
          }
        } catch {}
      };
      computeWs.on('message', handleSyncProgress);

      // Timeout cleanup after 2 hours
      setTimeout(() => {
        for (const source of sources) {
          activeIndexingJobs.delete(source);
        }
        computeWs.off('message', handleSyncProgress);
      }, 2 * 60 * 60 * 1000);
    }

    res.json({
      status: 'syncing',
      jobId,
      sources,
      message: `Sync job submitted for: ${sources.join(', ')}. Use /api/benchmark/stats to check progress.`
    });
  } catch (error) {
    console.error('Benchmark sync error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get benchmark database statistics
app.get('/api/benchmark/stats', async (req, res) => {
  try {
    const stats = {
      sources: {},
      total_molecules: 0,
      indexing: {},
      index_exists: true
    };

    // Check for active indexing jobs
    for (const [source, job] of activeIndexingJobs) {
      stats.indexing[source] = {
        jobId: job.jobId,
        startedAt: job.startedAt,
        progress: job.progress || 0,
        phase: job.phase || 'starting',
        details: job.details || {},
        status: 'indexing'
      };
    }

    // Count molecules per source
    for (const source of ['pubchem', 'nist', 'qm9']) {
      try {
        const countResult = await esClient.count({
          index: 'benchmark_molecules',
          body: {
            query: { term: { sources: source } }
          }
        });
        stats.sources[source] = countResult.count;
        stats.total_molecules += countResult.count;
      } catch {
        stats.sources[source] = 0;
      }
    }

    res.json(stats);
  } catch (error) {
    // Index might not exist
    if (error.statusCode === 404 || error.meta?.statusCode === 404) {
      // Check if we're currently indexing
      const indexingStatus = {};
      for (const [source, job] of activeIndexingJobs) {
        indexingStatus[source] = {
          jobId: job.jobId,
          startedAt: job.startedAt,
          progress: job.progress || 0,
          phase: job.phase || 'starting',
          details: job.details || {},
          status: 'indexing'
        };
      }

      return res.json({
        sources: { pubchem: 0, nist: 0, qm9: 0 },
        total_molecules: 0,
        index_exists: false,
        indexing: indexingStatus,
        note: Object.keys(indexingStatus).length > 0
          ? 'Index is being created. Data will be available shortly.'
          : 'No benchmark data indexed yet. Run benchmark.sync() to populate.'
      });
    }

    console.error('Benchmark stats error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Check status of a specific molecule (is it indexed, indexing, or not found)
app.get('/api/benchmark/status/:identifier', async (req, res) => {
  const { identifier } = req.params;

  try {
    // Build query clauses - only include numeric fields if identifier is numeric
    const shouldClauses = [
      { term: { identifier } },
      { term: { cas: identifier } },
      { term: { smiles: identifier } },
      { term: { inchi_key: identifier } },
      { term: { 'name.keyword': identifier } },
      { match: { name: identifier } }
    ];

    // Only search cid if identifier looks like a number
    if (/^\d+$/.test(identifier)) {
      shouldClauses.push({ term: { cid: parseInt(identifier, 10) } });
    }

    // Check if molecule exists in index
    const esResult = await esClient.search({
      index: 'benchmark_molecules',
      body: {
        query: {
          bool: {
            should: shouldClauses,
            minimum_should_match: 1
          }
        },
        size: 1,
        _source: ['identifier', 'name', 'formula', 'sources', 'cached_at']
      }
    });

    if (esResult.hits.hits.length > 0) {
      const molecule = esResult.hits.hits[0]._source;
      return res.json({
        status: 'indexed',
        exists: true,
        molecule: {
          identifier: molecule.identifier,
          name: molecule.name,
          formula: molecule.formula,
          sources: molecule.sources,
          cached_at: molecule.cached_at
        }
      });
    }

    // Not in index - check if we're currently syncing
    const indexingJobs = Array.from(activeIndexingJobs.values());
    if (indexingJobs.length > 0) {
      return res.json({
        status: 'indexing',
        exists: false,
        message: 'Database is currently being indexed. The molecule may become available soon.',
        indexing_jobs: indexingJobs.map(j => ({
          source: j.source,
          jobId: j.jobId,
          progress: j.progress,
          phase: j.phase,
          details: j.details,
          startedAt: j.startedAt
        }))
      });
    }

    // Not found and not indexing
    res.json({
      status: 'not_found',
      exists: false,
      message: `Molecule "${identifier}" is not in the benchmark database. It can be fetched from external sources.`,
      can_fetch: true
    });

  } catch (error) {
    if (error.statusCode === 404 || error.meta?.statusCode === 404) {
      // Index doesn't exist
      const indexingJobs = Array.from(activeIndexingJobs.values());
      if (indexingJobs.length > 0) {
        return res.json({
          status: 'indexing',
          exists: false,
          index_exists: false,
          message: 'Benchmark index is being created. Please wait.',
          indexing_jobs: indexingJobs.map(j => ({
            source: j.source,
            jobId: j.jobId,
            progress: j.progress,
            phase: j.phase,
            details: j.details,
            startedAt: j.startedAt
          }))
        });
      }

      return res.json({
        status: 'no_index',
        exists: false,
        index_exists: false,
        message: 'Benchmark database has not been initialized. Run benchmark.sync() first.',
        can_fetch: true
      });
    }

    console.error('Benchmark status error:', error);
    res.status(500).json({ error: error.message });
  }
});

// WebSocket connection to cm-compute
let computeWs = null;
const clientToCompute = new Map(); // Map client cell requests to compute responses
const jobSubscriptions = new Map(); // Map jobId -> Set of subscribed WebSocket clients
const terminalSessions = new Map(); // Map sessionId -> client WebSocket

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

      // Handle terminal_created - map session to pending client
      if (data.type === 'terminal_created') {
        // Find the client that requested this terminal
        for (const client of wss.clients) {
          if (client.pendingTerminalCreate && client.readyState === WebSocket.OPEN) {
            terminalSessions.set(data.sessionId, client);
            client.pendingTerminalCreate = false;
            client.send(JSON.stringify(data));
            break;
          }
        }
      }

      // Forward terminal messages to the appropriate client
      if (data.sessionId && terminalSessions.has(data.sessionId)) {
        const clientWs = terminalSessions.get(data.sessionId);
        if (clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify(data));
        }

        // Clean up on terminal exit
        if (data.type === 'terminal_exit') {
          terminalSessions.delete(data.sessionId);
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

      } else if (data.type === 'terminal_create') {
        // Create terminal session - forward to compute and track response
        ws.pendingTerminalCreate = true;
        if (computeWs && computeWs.readyState === WebSocket.OPEN) {
          computeWs.send(JSON.stringify(data));
        }

      } else if (data.type === 'terminal_input' || data.type === 'terminal_resize' || data.type === 'terminal_destroy') {
        // Forward terminal commands to compute
        if (computeWs && computeWs.readyState === WebSocket.OPEN) {
          computeWs.send(JSON.stringify(data));
        }
        // Clean up session tracking on destroy
        if (data.type === 'terminal_destroy' && data.sessionId) {
          terminalSessions.delete(data.sessionId);
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
    // Clean up terminal sessions for this client
    for (const [sessionId, clientWs] of terminalSessions.entries()) {
      if (clientWs === ws) {
        // Tell compute to destroy the session
        if (computeWs && computeWs.readyState === WebSocket.OPEN) {
          computeWs.send(JSON.stringify({
            type: 'terminal_destroy',
            sessionId
          }));
        }
        terminalSessions.delete(sessionId);
      }
    }
  });
});

// Retry helper with exponential backoff
async function retryWithBackoff(fn, name, maxRetries = 30, initialDelay = 1000) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt === maxRetries) {
        throw error;
      }
      const delay = Math.min(initialDelay * Math.pow(1.5, attempt - 1), 10000);
      console.log(`Waiting for ${name}... (attempt ${attempt}/${maxRetries}, retry in ${Math.round(delay/1000)}s)`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

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
    // Initialize database with retry
    await retryWithBackoff(
      () => initDatabase(),
      'PostgreSQL'
    );
    console.log('Database initialized');

    // Connect to Elasticsearch with retry
    await retryWithBackoff(
      () => esClient.ping(),
      'Elasticsearch'
    );
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
