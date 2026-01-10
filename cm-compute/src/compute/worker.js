/**
 * Worker Process - Executes compute jobs
 *
 * This process is forked by the queue and runs job handlers.
 * Job types are mapped to handler functions via the job registry.
 * All output (stdout/stderr) is sent back to the main process
 * where it is demuxed by job ID via the JobListener.
 */

const fs = require('fs');
const path = require('path');
const { createWorkerLogger } = require('../logger');
const { getJobHandler, hasJobType, getJobTypes } = require('./jobs');

const CONDA_PATH = process.env.CONDA_PATH || '/opt/conda';

// Worker state
let workerId = process.env.WORKER_ID || '0';
let logger = null;
let isChannelOpen = true;

/**
 * Initialize the logger
 */
function initLogger() {
  if (!logger) {
    logger = createWorkerLogger(workerId);
    logger.info('Worker started', { workerId, pid: process.pid });
  }
}

/**
 * Get Python path for a conda environment
 * @param {string} envName - Environment name
 * @returns {string} Path to Python executable
 */
function getPythonPath(envName) {
  if (!envName || envName === 'base') {
    return `${CONDA_PATH}/bin/python`;
  }
  const envPython = `${CONDA_PATH}/envs/${envName}/bin/python`;
  if (fs.existsSync(envPython)) {
    return envPython;
  }
  // Fallback to base if environment doesn't exist
  console.warn(`Environment '${envName}' not found, using base`);
  return `${CONDA_PATH}/bin/python`;
}

/**
 * Send message to parent process
 * @param {string} type - Message type
 * @param {*} data - Message data
 * @returns {boolean} True if message was sent successfully
 */
function send(type, data) {
  if (!isChannelOpen || !process.send) {
    return false;
  }

  try {
    process.send({ type, ...data });
    return true;
  } catch (error) {
    // EPIPE means parent closed the channel - mark as closed
    if (error.code === 'EPIPE' || error.code === 'ERR_IPC_CHANNEL_CLOSED') {
      isChannelOpen = false;
    }
    return false;
  }
}

/**
 * Exit codes for job completion
 */
const ExitCode = {
  SUCCESS: 0,
  GENERAL_ERROR: 1,
  INVALID_INPUT: 2,
  EXECUTION_ERROR: 3,
  TIMEOUT: 4,
  CANCELLED: 5
};

/**
 * Create an emit function for a job that sends output back to parent
 * @param {string} jobId - Job ID
 * @returns {Function} Emit function (streamType, data) => void
 */
function createJobEmitter(jobId) {
  return (streamType, data) => {
    switch (streamType) {
      case 'stdout':
        send('stdout', { jobId, data });
        break;
      case 'stderr':
        send('stderr', { jobId, data });
        break;
      case 'progress':
        send('progress', { progress: data });
        break;
      case 'download_progress':
        // Custom stream type for detailed download progress
        send('custom', { jobId, stream: 'download_progress', data });
        break;
      default:
        // For other custom stream types, send as custom message
        if (typeof data === 'object') {
          send('custom', { jobId, stream: streamType, data });
        } else {
          send('stdout', { jobId, data: String(data) });
        }
    }
  };
}

/**
 * Execute a job using the registered handler
 * @param {Object} job - Job to execute { id, type, params }
 * @returns {Promise<Object>} Job result
 */
async function executeJob(job) {
  const { id: jobId, type, params } = job;

  // Check if job type is registered
  if (!hasJobType(type)) {
    const availableTypes = getJobTypes().join(', ');
    throw new Error(`Unknown job type: '${type}'. Available types: ${availableTypes}`);
  }

  // Get the handler function for this job type
  const handler = getJobHandler(type);

  // Determine Python environment
  const environment = params?.environment || 'torch';
  const pythonPath = getPythonPath(environment);

  // Create context for the handler
  const context = {
    pythonPath,
    jobId,
    emit: createJobEmitter(jobId),
    workerId,
    environment
  };

  logger.info('Executing job', { jobId, type, environment, pythonPath });

  // Execute the handler
  const result = await handler(params, context);

  return result;
}

// Handle messages from parent process
process.on('message', async (message) => {
  if (message.type === 'init') {
    // Receive worker ID from queue
    workerId = message.workerId;
    initLogger();
    return;
  }

  if (message.type === 'execute') {
    const { job } = message;
    initLogger(); // Ensure logger is initialized

    let exitCode = ExitCode.SUCCESS;
    let result = null;

    try {
      result = await executeJob(job);

      // Check if the result contains an error (from handler returning error object)
      if (result && result.error) {
        exitCode = ExitCode.EXECUTION_ERROR;
      }

      logger.info('Job completed', { jobId: job.id, type: job.type, exitCode });
    } catch (error) {
      logger.error('Job failed', {
        jobId: job.id,
        type: job.type,
        error: error.message,
        stack: error.stack
      });

      // Determine exit code based on error type
      if (error.message.includes('Unknown job type') || error.message.includes('No code provided')) {
        exitCode = ExitCode.INVALID_INPUT;
      } else if (error.message.includes('timeout') || error.message.includes('TIMEOUT')) {
        exitCode = ExitCode.TIMEOUT;
      } else if (error.message.includes('cancelled') || error.message.includes('CANCELLED')) {
        exitCode = ExitCode.CANCELLED;
      } else {
        exitCode = ExitCode.GENERAL_ERROR;
      }

      result = {
        error: error.message,
        stack: error.stack
      };
    }

    // Send result with exit code
    send('result', { result, exitCode });

    // Send completion signal with final exit code
    send('complete', { jobId: job.id, exitCode });
  }
});

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  if (logger) {
    logger.error('Uncaught exception in worker', {
      error: error.message,
      stack: error.stack
    });
  }
  console.error('Uncaught exception in worker:', error);
  send('result', {
    result: {
      error: error.message,
      stack: error.stack
    },
    exitCode: ExitCode.GENERAL_ERROR
  });
  send('complete', { exitCode: ExitCode.GENERAL_ERROR });
  process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  const error = reason instanceof Error ? reason : new Error(String(reason));
  if (logger) {
    logger.error('Unhandled rejection in worker', {
      error: error.message,
      stack: error.stack
    });
  }
  console.error('Unhandled rejection in worker:', error);
});

// Handle parent disconnect
process.on('disconnect', () => {
  isChannelOpen = false;
  if (logger) {
    logger.info('Parent disconnected, worker exiting');
  }
  process.exit(0);
});

// Export ExitCode for use by queue
module.exports = { ExitCode };
