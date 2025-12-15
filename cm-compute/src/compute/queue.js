/**
 * ComputeQueue - Job queue with worker pool management
 *
 * Jobs are enqueued and then picked up by forked worker processes.
 * Each worker uses the job type to map to a handler function.
 * All output is routed through the JobListener for demuxing.
 */

const { fork } = require('child_process');
const path = require('path');
const JobListener = require('./job-listener');
const { StreamType } = require('./job-listener');

/**
 * Exit codes for job completion (mirrored from worker.js)
 */
const ExitCode = {
  SUCCESS: 0,
  GENERAL_ERROR: 1,
  INVALID_INPUT: 2,
  EXECUTION_ERROR: 3,
  TIMEOUT: 4,
  CANCELLED: 5
};

class ComputeQueue {
  /**
   * @param {Object} options - Configuration options
   * @param {number} options.maxWorkers - Maximum parallel workers (default: 4)
   * @param {JobListener} options.jobListener - Job listener instance
   */
  constructor(options = {}) {
    this.queue = [];
    this.running = new Map();
    this.workers = [];
    this.maxWorkers = options.maxWorkers || 4;
    this.workerIdCounter = 0;

    // Job listener for demuxing stdout/stderr by job ID
    this.jobListener = options.jobListener || new JobListener();
  }

  /**
   * Enqueue a job for execution
   * @param {Object} job - Job object { id, type, params, priority }
   * @param {Function} callback - Optional callback when job completes (result, exitCode) => void
   */
  enqueue(job, callback) {
    // Attach callback to job object if provided
    if (callback) {
      job.callback = callback;
    }

    this.queue.push(job);
    // Sort by priority (higher priority first)
    this.queue.sort((a, b) => b.priority - a.priority);

    // Register job with listener for output demuxing
    this.jobListener.registerJob(job.id);

    this.processNext();
  }

  /**
   * Dequeue the next job
   * @returns {Object|undefined} Next job or undefined
   */
  dequeue() {
    return this.queue.shift();
  }

  /**
   * Process next job if capacity available
   */
  processNext() {
    if (this.queue.length === 0 || this.running.size >= this.maxWorkers) {
      return;
    }

    const job = this.dequeue();
    if (!job) return;

    this.execute(job);
  }

  /**
   * Execute a job by forking a worker process
   * @param {Object} job - Job to execute
   */
  execute(job) {
    const workerId = this.workerIdCounter++;
    const workerPath = path.join(__dirname, 'worker.js');
    const worker = fork(workerPath);

    this.running.set(job.id, { worker, job, workerId });

    // Initialize worker with its ID
    worker.send({
      type: 'init',
      workerId: workerId
    });

    // Send the job to execute
    worker.send({
      type: 'execute',
      job: {
        id: job.id,
        type: job.type,
        params: job.params
      }
    });

    // Handle messages from worker
    worker.on('message', (message) => {
      switch (message.type) {
        case 'result':
          this._handleResult(job, message.result, message.exitCode, worker);
          break;

        case 'complete':
          // Final completion signal with exit code
          this.jobListener.complete(job.id, message.exitCode);
          break;

        case 'progress':
          // Route progress through job listener
          this.jobListener.progress(job.id, message.progress);
          break;

        case 'stdout':
          // Route stdout through job listener
          this.jobListener.write(job.id, StreamType.STDOUT, message.data);
          break;

        case 'stderr':
          // Route stderr through job listener
          this.jobListener.write(job.id, StreamType.STDERR, message.data);
          break;
      }
    });

    worker.on('error', (error) => {
      console.error(`Worker error for job ${job.id}:`, error);
      this._handleError(job, error, worker);
    });

    worker.on('exit', (code) => {
      if (code !== 0 && this.running.has(job.id)) {
        console.error(`Worker exited with code ${code} for job ${job.id}`);
        this._handleError(job, new Error(`Worker exited with code ${code}`), worker);
      }
    });
  }

  /**
   * Handle job result
   * @private
   * @param {Object} job - Job object
   * @param {Object} result - Job result
   * @param {number} exitCode - Exit code (0 = success)
   * @param {ChildProcess} worker - Worker process
   */
  _handleResult(job, result, exitCode = ExitCode.SUCCESS, worker) {
    this.running.delete(job.id);

    // Send result through job listener with exit code
    if (result && result.error) {
      this.jobListener.error(job.id, result.error, exitCode);
    } else {
      this.jobListener.result(job.id, result, exitCode);
    }

    if (job.callback) {
      job.callback(result, exitCode);
    }

    worker.kill();
    this.processNext();
  }

  /**
   * Handle job error
   * @private
   * @param {Object} job - Job object
   * @param {Error} error - Error object
   * @param {ChildProcess} worker - Worker process
   * @param {number} exitCode - Exit code
   */
  _handleError(job, error, worker, exitCode = ExitCode.GENERAL_ERROR) {
    this.running.delete(job.id);

    // Send error through job listener with exit code
    this.jobListener.error(job.id, error, exitCode);
    this.jobListener.complete(job.id, exitCode);

    if (job.callback) {
      job.callback({ error: error.message }, exitCode);
    }

    worker.kill();
    this.processNext();
  }

  /**
   * Cancel a job
   * @param {string} jobId - Job ID to cancel
   * @returns {boolean} True if job was cancelled
   */
  cancel(jobId) {
    // Check if in queue
    const queueIndex = this.queue.findIndex(j => j.id === jobId);
    if (queueIndex !== -1) {
      this.queue.splice(queueIndex, 1);
      this.jobListener.unregisterJob(jobId);
      return true;
    }

    // Check if running
    const running = this.running.get(jobId);
    if (running) {
      running.worker.kill();
      this.running.delete(jobId);
      this.jobListener.unregisterJob(jobId);
      this.processNext();
      return true;
    }

    return false;
  }

  /**
   * Get queue position for a job
   * @param {string} jobId - Job ID
   * @returns {number|null} Queue position (1-indexed) or null if not in queue
   */
  getPosition(jobId) {
    const index = this.queue.findIndex(j => j.id === jobId);
    return index === -1 ? null : index + 1;
  }

  /**
   * Get queue statistics
   * @returns {Object} Queue stats
   */
  getStats() {
    return {
      queued: this.queue.length,
      running: this.running.size,
      maxWorkers: this.maxWorkers,
      listenerStats: this.jobListener.getStats()
    };
  }

  /**
   * Set maximum number of workers
   * @param {number} count - Worker count
   */
  startWorkers(count) {
    this.maxWorkers = count;
  }

  /**
   * Subscribe to a job's output
   * @param {string} jobId - Job ID
   * @param {Function} callback - Callback for messages
   * @returns {Function} Unsubscribe function
   */
  subscribe(jobId, callback) {
    return this.jobListener.subscribe(jobId, callback);
  }

  /**
   * Get buffered output for a job
   * @param {string} jobId - Job ID
   * @returns {Object} Buffered output
   */
  getJobOutput(jobId) {
    return this.jobListener.getBuffer(jobId);
  }

  /**
   * Get the job listener instance
   * @returns {JobListener}
   */
  getJobListener() {
    return this.jobListener;
  }
}

module.exports = { ComputeQueue, ExitCode };
