const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const { Client } = require('elasticsearch');
const { execSync } = require('child_process');
const { ComputeQueue, JobListener } = require('./compute');
const { createMainLogger } = require('./logger');
const database = require('./database');
const terminal = require('./terminal');

// Initialize main logger
const logger = createMainLogger();

// Initialize job listener for demuxing stdout/stderr by job ID
const jobListener = new JobListener();

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

// Database models (initialized in start())
let Job = null;

const esClient = new Client({
  node: `http://${process.env.ELASTICSEARCH_HOST || 'localhost'}:${process.env.ELASTICSEARCH_PORT || 9200}`,
});

// Initialize compute queue with job listener for output demuxing
const computeQueue = new ComputeQueue({ jobListener });

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

  try {
    // Create job in database
    const job = await Job.create({ type, params, priority });
    const jobId = job.id.toString();

    logger.job('QUEUED', jobId, { type, params, priority });

    // Add to queue
    computeQueue.enqueue({ ...job, id: jobId }, async (result) => {
      const status = result.error ? 'failed' : 'completed';
      logger.job(status.toUpperCase(), jobId, result.error ? { error: result.error } : { success: true });

      await Job.updateStatus(job.id, status, result);
    });

    res.json({
      jobId,
      status: 'queued',
      position: computeQueue.getPosition(jobId)
    });
  } catch (error) {
    logger.error('Error submitting job', { error: error.message, stack: error.stack });
    console.error('Error submitting job:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get job status
app.get('/compute/:jobId', async (req, res) => {
  const { jobId } = req.params;

  try {
    const job = await Job.findById(jobId);

    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const queuePosition = computeQueue.getPosition(jobId);

    res.json({
      ...job,
      id: job.id.toString(),
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
app.post('/environments', async (req, res) => {
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

  try {
    logger.info('Creating conda environment', { name, pythonVersion, packages });
    const job = await Job.create({
      type: 'create_environment',
      params: { name, pythonVersion, packages },
      priority: 5
    });

    const jobId = job.id.toString();
    computeQueue.enqueue({ ...job, id: jobId });

    // Return job info - client can subscribe to WebSocket for progress
    res.json({
      jobId,
      status: 'queued',
      message: `Creating environment '${name}' with Python ${pythonVersion}...`,
      params: { name, pythonVersion, packages }
    });
  } catch (error) {
    logger.error('Error creating environment job', { name, error: error.message });
    res.status(500).json({ error: error.message });
  }
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
app.post('/environments/:name/packages', async (req, res) => {
  const { name } = req.params;
  const { packages, channel = null } = req.body;

  if (!packages || !Array.isArray(packages) || packages.length === 0) {
    return res.status(400).json({ error: 'Packages array required' });
  }

  try {
    logger.info('Installing packages', { environment: name, packages, channel });

    const job = await Job.create({
      type: 'install_conda_package',
      params: { envName: name, packages, channel },
      priority: 5
    });

    const jobId = job.id.toString();
    computeQueue.enqueue({ ...job, id: jobId });

    res.json({
      jobId,
      status: 'queued',
      environment: name,
      packages,
      message: `Installing ${packages.join(', ')} in '${name}'...`
    });
  } catch (error) {
    logger.error('Failed to queue package installation', { environment: name, packages, error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// Remove package from environment
app.delete('/environments/:name/packages/:packageName', async (req, res) => {
  const { name, packageName } = req.params;

  if (!packageName) {
    return res.status(400).json({ error: 'Package name required' });
  }

  try {
    logger.info('Removing package', { environment: name, packageName });

    const job = await Job.create({
      type: 'remove_conda_package',
      params: { envName: name, packageName },
      priority: 5
    });

    const jobId = job.id.toString();
    computeQueue.enqueue({ ...job, id: jobId });

    res.json({
      jobId,
      status: 'queued',
      environment: name,
      packageName,
      message: `Removing '${packageName}' from '${name}'...`
    });
  } catch (error) {
    logger.error('Failed to queue package removal', { environment: name, packageName, error: error.message });
    res.status(500).json({ error: error.message });
  }
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

// ================== C++ Environment Management ==================

// Database models for C++ environments (initialized in start())
let CppEnvironment = null;
let VendorEnvironment = null;

// Search for available debian dev packages
// Get C++ compiler versions
app.get('/compilers', (req, res) => {
  const compilers = [];

  // Check g++
  try {
    const gppVersion = execSync('g++ --version 2>&1 | head -1', { encoding: 'utf-8' }).trim();
    const versionMatch = gppVersion.match(/(\d+\.\d+\.\d+)/);
    compilers.push({
      name: 'g++',
      version: versionMatch ? versionMatch[1] : 'unknown',
      fullVersion: gppVersion,
      available: true
    });
  } catch (e) {
    compilers.push({ name: 'g++', version: null, available: false });
  }

  // Check clang++
  try {
    const clangVersion = execSync('clang++ --version 2>&1 | head -1', { encoding: 'utf-8' }).trim();
    const versionMatch = clangVersion.match(/(\d+\.\d+\.\d+)/);
    compilers.push({
      name: 'clang++',
      version: versionMatch ? versionMatch[1] : 'unknown',
      fullVersion: clangVersion,
      available: true
    });
  } catch (e) {
    compilers.push({ name: 'clang++', version: null, available: false });
  }

  res.json({ compilers });
});

app.get('/debian-packages/search', async (req, res) => {
  const { q, limit = 50 } = req.query;

  if (!q || q.length < 2) {
    return res.json({ packages: [] });
  }

  try {
    // Search for dev packages using apt-cache
    const result = execSync(
      `apt-cache search "${q}" | grep -E "^lib.*-dev |^.*-dev " | head -n ${limit}`,
      { encoding: 'utf-8', timeout: 10000 }
    );

    const packages = result.trim().split('\n')
      .filter(line => line.length > 0)
      .map(line => {
        const [name, ...descParts] = line.split(' - ');
        return {
          name: name.trim(),
          description: descParts.join(' - ').trim()
        };
      });

    res.json({ packages });
  } catch (error) {
    // If no results or error, return empty
    if (error.status === 1) {
      return res.json({ packages: [] });
    }
    console.error('Error searching debian packages:', error);
    res.status(500).json({ error: error.message });
  }
});

// List all C++ environments
app.get('/cpp-environments', async (req, res) => {
  try {
    if (!CppEnvironment) {
      return res.json({ environments: [] });
    }
    const envs = await CppEnvironment.findAll();
    res.json({ environments: envs });
  } catch (error) {
    console.error('Error listing C++ environments:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get C++ environment details
app.get('/cpp-environments/:name', async (req, res) => {
  try {
    if (!CppEnvironment) {
      return res.status(404).json({ error: 'Environment not found' });
    }
    const env = await CppEnvironment.findByName(req.params.name);
    if (!env) {
      return res.status(404).json({ error: 'Environment not found' });
    }

    // Get linked vendor environments
    const vendorEnvs = await CppEnvironment.getLinkedVendorEnvironments(env.id);

    res.json({ ...env, vendorEnvironments: vendorEnvs });
  } catch (error) {
    console.error('Error fetching C++ environment:', error);
    res.status(500).json({ error: error.message });
  }
});

// Create C++ environment
app.post('/cpp-environments', async (req, res) => {
  const { name, description = '', packages = [] } = req.body;

  if (!name) {
    return res.status(400).json({ error: 'Environment name required' });
  }

  if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(name)) {
    return res.status(400).json({ error: 'Invalid environment name. Use alphanumeric characters, hyphens, and underscores.' });
  }

  try {
    // Check if environment already exists
    const existing = await CppEnvironment.findByName(name);
    if (existing) {
      return res.status(409).json({ error: `Environment '${name}' already exists` });
    }

    logger.info('Creating C++ environment', { name, packages });

    // Create job to install packages
    const job = await Job.create({
      type: 'create_cpp_environment',
      params: { name, description, packages },
      priority: 5
    });

    const jobId = job.id.toString();
    computeQueue.enqueue({ ...job, id: jobId });

    res.json({
      jobId,
      status: 'queued',
      message: `Creating C++ environment '${name}'...`,
      params: { name, description, packages }
    });
  } catch (error) {
    logger.error('Error creating C++ environment', { name, error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// Delete C++ environment
app.delete('/cpp-environments/:name', async (req, res) => {
  const { name } = req.params;

  try {
    const env = await CppEnvironment.findByName(name);
    if (!env) {
      return res.status(404).json({ error: `Environment '${name}' not found` });
    }

    logger.info('Deleting C++ environment', { name });
    await CppEnvironment.deleteByName(name);
    logger.info('Deleted C++ environment', { name });

    res.json({ status: 'deleted', name });
  } catch (error) {
    logger.error('Error deleting C++ environment', { name, error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// ================== Vendor Environment Management ==================

// List all vendor environments
app.get('/vendor-environments', async (req, res) => {
  try {
    if (!VendorEnvironment) {
      return res.json({ environments: [] });
    }
    const envs = await VendorEnvironment.findAll();
    res.json({ environments: envs });
  } catch (error) {
    console.error('Error listing vendor environments:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get vendor environment details
app.get('/vendor-environments/:name', async (req, res) => {
  try {
    if (!VendorEnvironment) {
      return res.status(404).json({ error: 'Vendor environment not found' });
    }
    const env = await VendorEnvironment.findByName(req.params.name);
    if (!env) {
      return res.status(404).json({ error: 'Vendor environment not found' });
    }
    res.json(env);
  } catch (error) {
    console.error('Error fetching vendor environment:', error);
    res.status(500).json({ error: error.message });
  }
});

// Create vendor environment
app.post('/vendor-environments', async (req, res) => {
  const { name, description = '', repo, branch = 'main', buildType = 'cmake', cmakeOptions = '' } = req.body;

  if (!name) {
    return res.status(400).json({ error: 'Environment name required' });
  }

  if (!repo) {
    return res.status(400).json({ error: 'Repository URL required' });
  }

  if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(name)) {
    return res.status(400).json({ error: 'Invalid environment name. Use alphanumeric characters, hyphens, and underscores.' });
  }

  try {
    // Check if environment already exists
    const existing = await VendorEnvironment.findByName(name);
    if (existing) {
      return res.status(409).json({ error: `Vendor environment '${name}' already exists` });
    }

    logger.info('Creating vendor environment', { name, repo, branch, buildType });

    // Create job to clone and build
    const job = await Job.create({
      type: 'create_vendor_environment',
      params: { name, description, repo, branch, buildType, cmakeOptions },
      priority: 5
    });

    const jobId = job.id.toString();
    computeQueue.enqueue({ ...job, id: jobId });

    res.json({
      jobId,
      status: 'queued',
      message: `Creating vendor environment '${name}' from ${repo}...`,
      params: { name, description, repo, branch, buildType }
    });
  } catch (error) {
    logger.error('Error creating vendor environment', { name, error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// Delete vendor environment
app.delete('/vendor-environments/:name', async (req, res) => {
  const { name } = req.params;

  try {
    const env = await VendorEnvironment.findByName(name);
    if (!env) {
      return res.status(404).json({ error: `Vendor environment '${name}' not found` });
    }

    logger.info('Deleting vendor environment', { name });

    // Remove installation directory
    const installPrefix = `/opt/vendor/${name}`;
    try {
      execSync(`rm -rf ${installPrefix}`, { encoding: 'utf-8' });
    } catch (e) {
      // Directory may not exist
    }

    await VendorEnvironment.deleteByName(name);
    logger.info('Deleted vendor environment', { name });

    res.json({ status: 'deleted', name });
  } catch (error) {
    logger.error('Error deleting vendor environment', { name, error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// ================== Profile Management ==================

let Profile = null;

// Get active profile
app.get('/profile', async (req, res) => {
  try {
    const profile = await Profile.getActive();
    if (profile) {
      // Don't send private key to client
      const { ssh_private_key, ...safeProfile } = profile;
      res.json(safeProfile);
    } else {
      res.json(null);
    }
  } catch (error) {
    logger.error('Error getting profile', { error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// Create or update profile
app.post('/profile', async (req, res) => {
  const { name, email } = req.body;

  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email are required' });
  }

  try {
    let profile = await Profile.findByEmail(email);

    if (profile) {
      // Update existing profile
      profile = await Profile.update(profile.id, { name, email });
    } else {
      // Create new profile
      profile = await Profile.create({ name, email });
    }

    const { ssh_private_key, ...safeProfile } = profile;
    res.json(safeProfile);
  } catch (error) {
    logger.error('Error saving profile', { error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// Generate SSH key pair
app.post('/profile/:id/generate-ssh-key', async (req, res) => {
  const { id } = req.params;

  try {
    const profile = await Profile.findById(id);
    if (!profile) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    const os = require('os');

    // Create a temporary directory for key generation
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ssh-'));
    const keyPath = path.join(tmpDir, 'id_ed25519');

    try {
      // Generate ED25519 key (more secure and shorter than RSA)
      execSync(`ssh-keygen -t ed25519 -C "${profile.email}" -f "${keyPath}" -N ""`, {
        encoding: 'utf-8',
        stdio: 'pipe'
      });

      // Read the generated keys
      const privateKey = fs.readFileSync(keyPath, 'utf-8');
      const publicKey = fs.readFileSync(`${keyPath}.pub`, 'utf-8').trim();

      // Generate fingerprint
      const fingerprintOutput = execSync(`ssh-keygen -lf "${keyPath}.pub"`, { encoding: 'utf-8' });
      const fingerprint = fingerprintOutput.split(' ')[1];

      // Save keys to profile
      await Profile.setSSHKeys(id, { publicKey, privateKey, fingerprint });

      // Configure git to use this key
      const sshDir = path.join(os.homedir(), '.ssh');
      if (!fs.existsSync(sshDir)) {
        fs.mkdirSync(sshDir, { mode: 0o700 });
      }

      // Write the private key to ~/.ssh/id_ed25519_chemicalmachines
      const sshKeyPath = path.join(sshDir, 'id_ed25519_chemicalmachines');
      fs.writeFileSync(sshKeyPath, privateKey, { mode: 0o600 });
      fs.writeFileSync(`${sshKeyPath}.pub`, publicKey, { mode: 0o644 });

      // Add SSH config entry for GitHub
      const sshConfigPath = path.join(sshDir, 'config');
      let sshConfig = '';
      if (fs.existsSync(sshConfigPath)) {
        sshConfig = fs.readFileSync(sshConfigPath, 'utf-8');
      }

      // Check if we already have a config for github.com
      if (!sshConfig.includes('Host github.com') || !sshConfig.includes('id_ed25519_chemicalmachines')) {
        const githubConfig = `
# Chemical Machines - GitHub
Host github.com
  HostName github.com
  User git
  IdentityFile ${sshKeyPath}
  IdentitiesOnly yes
`;
        // Only add if not already present
        if (!sshConfig.includes('id_ed25519_chemicalmachines')) {
          fs.appendFileSync(sshConfigPath, githubConfig);
          fs.chmodSync(sshConfigPath, 0o600);
        }
      }

      // Configure git user
      try {
        execSync(`git config --global user.name "${profile.name}"`, { stdio: 'pipe' });
        execSync(`git config --global user.email "${profile.email}"`, { stdio: 'pipe' });
      } catch (e) {
        logger.warn('Could not configure git user', { error: e.message });
      }

      // Clean up temp files
      fs.unlinkSync(keyPath);
      fs.unlinkSync(`${keyPath}.pub`);
      fs.rmdirSync(tmpDir);

      res.json({
        publicKey,
        fingerprint,
        message: 'SSH key generated and configured successfully'
      });
    } catch (keyError) {
      // Clean up on error
      try {
        if (fs.existsSync(keyPath)) fs.unlinkSync(keyPath);
        if (fs.existsSync(`${keyPath}.pub`)) fs.unlinkSync(`${keyPath}.pub`);
        if (fs.existsSync(tmpDir)) fs.rmdirSync(tmpDir);
      } catch (e) {}
      throw keyError;
    }
  } catch (error) {
    logger.error('Error generating SSH key', { error: error.message });
    res.status(500).json({ error: error.message });
  }
});

// Get public SSH key
app.get('/profile/:id/ssh-key', async (req, res) => {
  const { id } = req.params;

  try {
    const publicKey = await Profile.getPublicKey(id);
    if (!publicKey) {
      return res.status(404).json({ error: 'No SSH key found' });
    }
    res.json({ publicKey });
  } catch (error) {
    logger.error('Error getting SSH key', { error: error.message });
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
      await Job.updateStatus(jobId, 'cancelled');
      res.json({ status: 'cancelled' });
    } else {
      res.status(404).json({ error: 'Job not found or already completed' });
    }
  } catch (error) {
    console.error('Error cancelling job:', error);
    res.status(500).json({ error: error.message });
  }
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

// Start the server
async function start() {
  try {
    logger.info('Starting cm-compute daemon');

    // Initialize database with retry
    const db = await retryWithBackoff(
      () => database.initialize(),
      'PostgreSQL'
    );
    Job = db.models.Job;
    CppEnvironment = db.models.CppEnvironment;
    VendorEnvironment = db.models.VendorEnvironment;
    Profile = db.models.Profile;
    logger.info('Database initialized');
    console.log('Database initialized');

    // Test Elasticsearch connection with retry
    await retryWithBackoff(
      () => esClient.ping(),
      'Elasticsearch'
    );
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

    // Start HTTP server with WebSocket support
    const httpServer = http.createServer(app);
    const wss = new WebSocket.Server({ server: httpServer, path: '/ws' });

    // Track connected clients
    const clients = new Set();

    wss.on('connection', (ws) => {
      clients.add(ws);
      logger.info('WebSocket client connected', { totalClients: clients.size });

      ws.on('message', async (message) => {
        try {
          const data = JSON.parse(message);

          if (data.type === 'execute') {
            const { code, cellId, environment } = data;

            // Create job in database
            const job = await Job.create({
              type: 'execute',
              params: { code, environment },
              priority: 5
            });

            const jobId = job.id.toString();

            // Send job accepted
            ws.send(JSON.stringify({
              type: 'job_accepted',
              jobId,
              cellId
            }));

            // Enqueue with callback for real-time updates
            computeQueue.enqueue({ ...job, id: jobId }, async (result) => {
              const status = result.error ? 'failed' : 'completed';

              await Job.updateStatus(job.id, status, result);

              // Send result via WebSocket
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                  type: 'job_result',
                  jobId,
                  cellId,
                  status,
                  result
                }));
              }
            });

            logger.job('QUEUED', jobId, { type: 'execute', cellId });

          } else if (data.type === 'subscribe') {
            // Subscribe to job output streams via job listener
            ws.subscribedJobs = ws.subscribedJobs || new Set();
            ws.subscribedJobs.add(data.jobId);

            // Send any buffered output first
            const buffered = jobListener.getBuffer(data.jobId);
            if (buffered.stdout) {
              ws.send(JSON.stringify({
                type: 'job_output',
                jobId: data.jobId,
                stream: 'stdout',
                data: buffered.stdout,
                timestamp: Date.now()
              }));
            }
            if (buffered.stderr) {
              ws.send(JSON.stringify({
                type: 'job_output',
                jobId: data.jobId,
                stream: 'stderr',
                data: buffered.stderr,
                timestamp: Date.now()
              }));
            }

            // Subscribe to job output demuxed by job ID
            const unsubscribe = jobListener.subscribe(data.jobId, (message) => {
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                  type: 'job_output',
                  jobId: data.jobId,
                  stream: message.type,
                  data: message.data || message.progress || message.result || message.error,
                  timestamp: message.timestamp
                }));
              }
            });

            // Store unsubscribe function for cleanup
            ws.unsubscribers = ws.unsubscribers || new Map();
            ws.unsubscribers.set(data.jobId, unsubscribe);

            // Send subscription acknowledgement
            ws.send(JSON.stringify({
              type: 'subscribed',
              jobId: data.jobId
            }));

          } else if (data.type === 'cancel') {
            const cancelled = computeQueue.cancel(data.jobId);
            ws.send(JSON.stringify({
              type: 'job_cancelled',
              jobId: data.jobId,
              success: cancelled
            }));

          } else if (data.type === 'terminal_create') {
            // Create a new terminal session for a workspace
            const { workspaceId, cols, rows } = data;
            const emit = (type, payload) => {
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type, ...payload }));
              }
            };

            const sessionId = terminal.createSession(workspaceId, emit, { cols, rows });
            ws.terminalSessions = ws.terminalSessions || new Set();
            ws.terminalSessions.add(sessionId);

            ws.send(JSON.stringify({
              type: 'terminal_created',
              sessionId,
              workspaceId
            }));

            logger.info('Terminal session created', { sessionId, workspaceId });

          } else if (data.type === 'terminal_input') {
            // Write input to terminal
            const { sessionId, data: inputData } = data;
            terminal.writeToSession(sessionId, inputData);

          } else if (data.type === 'terminal_resize') {
            // Resize terminal
            const { sessionId, cols, rows } = data;
            terminal.resizeSession(sessionId, cols, rows);

          } else if (data.type === 'terminal_destroy') {
            // Destroy terminal session
            const { sessionId } = data;
            terminal.destroySession(sessionId);
            if (ws.terminalSessions) {
              ws.terminalSessions.delete(sessionId);
            }
            logger.info('Terminal session destroyed', { sessionId });
          }
        } catch (error) {
          logger.error('WebSocket message error', { error: error.message });
          ws.send(JSON.stringify({
            type: 'error',
            error: error.message
          }));
        }
      });

      ws.on('close', () => {
        // Clean up job subscriptions
        if (ws.unsubscribers) {
          ws.unsubscribers.forEach(unsubscribe => unsubscribe());
          ws.unsubscribers.clear();
        }
        // Clean up terminal sessions
        if (ws.terminalSessions) {
          ws.terminalSessions.forEach(sessionId => {
            terminal.destroySession(sessionId);
          });
          ws.terminalSessions.clear();
        }
        clients.delete(ws);
        logger.info('WebSocket client disconnected', { totalClients: clients.size });
      });

      ws.on('error', (error) => {
        logger.error('WebSocket error', { error: error.message });
        clients.delete(ws);
      });
    });

    // Broadcast function for job updates
    function broadcast(message) {
      const data = JSON.stringify(message);
      clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(data);
        }
      });
    }

    httpServer.listen(HTTP_PORT, () => {
      logger.info('HTTP/WebSocket server started', { port: HTTP_PORT });
      console.log(`cm-compute HTTP/WebSocket server listening on port ${HTTP_PORT}`);
    });

    // Start Unix socket server
    const server = app.listen(SOCKET_PATH, () => {
      fs.chmodSync(SOCKET_PATH, '0666');
      logger.info('Unix socket server started', { path: SOCKET_PATH });
      console.log(`cm-compute daemon listening on ${SOCKET_PATH}`);
    });

    // Schedule benchmark database sync (every 24 hours)
    const BENCHMARK_SYNC_INTERVAL = 24 * 60 * 60 * 1000; // 24 hours
    setInterval(async () => {
      logger.info('Starting scheduled benchmark sync');
      try {
        const job = await Job.create({
          type: 'benchmark_sync',
          params: { sources: ['qm9'], workspaceId: '1' },
          priority: 1 // Low priority
        });
        computeQueue.enqueue({
          id: job.id.toString(),
          type: 'benchmark_sync',
          params: job.params,
          priority: job.priority
        });
        logger.info('Benchmark sync job enqueued', { jobId: job.id });
      } catch (error) {
        logger.error('Failed to schedule benchmark sync', { error: error.message });
      }
    }, BENCHMARK_SYNC_INTERVAL);

    logger.info('Benchmark sync scheduled', { intervalHours: 24 });

    // Graceful shutdown
    process.on('SIGTERM', async () => {
      logger.info('SIGTERM received, shutting down gracefully');
      console.log('SIGTERM received, shutting down gracefully');
      server.close();
      await database.close();
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
