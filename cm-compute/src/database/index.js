const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');
const Job = require('./job');
const CppEnvironment = require('./cpp-environment');
const VendorEnvironment = require('./vendor-environment');
const Profile = require('./profile');

let pool = null;
let models = null;

function createPool(config) {
  return new Pool({
    host: config.host || process.env.POSTGRES_HOST || 'localhost',
    database: config.database || process.env.POSTGRES_DB || 'chemicalmachines',
    user: config.user || process.env.POSTGRES_USER || 'cmuser',
    password: config.password || process.env.POSTGRES_PASSWORD || 'changeme',
    port: config.port || 5432,
  });
}

async function migrate(pgPool) {
  const schemaDir = path.join(__dirname, '../schema');

  // Get all .sql files sorted by name
  const files = fs.readdirSync(schemaDir)
    .filter(f => f.endsWith('.sql'))
    .sort();

  for (const file of files) {
    const filePath = path.join(schemaDir, file);
    const sql = fs.readFileSync(filePath, 'utf-8');

    try {
      await pgPool.query(sql);
      console.log(`Migrated: ${file}`);
    } catch (error) {
      console.error(`Migration failed for ${file}:`, error.message);
      throw error;
    }
  }
}

async function connect(config = {}) {
  pool = createPool(config);

  // Test connection
  await pool.query('SELECT 1');

  return pool;
}

async function initialize(config = {}) {
  const pgPool = await connect(config);
  await migrate(pgPool);

  // Initialize models
  models = {
    Job: new Job(pgPool),
    CppEnvironment: new CppEnvironment(pgPool),
    VendorEnvironment: new VendorEnvironment(pgPool),
    Profile: new Profile(pgPool)
  };

  return { pool: pgPool, models };
}

function getPool() {
  if (!pool) {
    throw new Error('Database not initialized. Call initialize() first.');
  }
  return pool;
}

function getModels() {
  if (!models) {
    throw new Error('Database not initialized. Call initialize() first.');
  }
  return models;
}

async function close() {
  if (pool) {
    await pool.end();
    pool = null;
    models = null;
  }
}

module.exports = {
  initialize,
  connect,
  migrate,
  getPool,
  getModels,
  close
};
