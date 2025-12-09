const fs = require('fs');
const path = require('path');

const LOG_DIR = process.env.CM_LOG_DIR || '/var/log/cm-compute';

// Ensure log directory exists
function ensureLogDir() {
  if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
  }
}

// Format timestamp for logs
function timestamp() {
  return new Date().toISOString();
}

// Format log entry
function formatEntry(level, message, data = null) {
  const entry = {
    timestamp: timestamp(),
    level,
    message
  };
  if (data !== null) {
    entry.data = data;
  }
  return JSON.stringify(entry);
}

// Create a logger for a specific log file
function createLogger(filename) {
  ensureLogDir();
  const logPath = path.join(LOG_DIR, filename);

  // Create/open the log file in append mode
  const stream = fs.createWriteStream(logPath, { flags: 'a' });

  const logger = {
    info(message, data = null) {
      stream.write(formatEntry('INFO', message, data) + '\n');
    },

    warn(message, data = null) {
      stream.write(formatEntry('WARN', message, data) + '\n');
    },

    error(message, data = null) {
      stream.write(formatEntry('ERROR', message, data) + '\n');
    },

    request(method, path, payload = null) {
      stream.write(formatEntry('REQUEST', `${method} ${path}`, payload) + '\n');
    },

    response(method, path, status, data = null) {
      stream.write(formatEntry('RESPONSE', `${method} ${path} -> ${status}`, data) + '\n');
    },

    job(action, jobId, details = null) {
      stream.write(formatEntry('JOB', `${action} [${jobId}]`, details) + '\n');
    },

    close() {
      stream.end();
    }
  };

  return logger;
}

// Main process logger
function createMainLogger() {
  return createLogger('main.out');
}

// Worker process logger
function createWorkerLogger(workerId) {
  return createLogger(`worker.${workerId}.out`);
}

module.exports = {
  createMainLogger,
  createWorkerLogger,
  createLogger,
  LOG_DIR
};
