/**
 * Job Listener - Demultiplexes stdout/stderr streams by job ID
 *
 * All job executions write back to this listener on the main process.
 * Messages are tagged with job ID and stream type (stdout/stderr).
 */

const EventEmitter = require('events');

/**
 * Stream types for job output
 */
const StreamType = {
  STDOUT: 'stdout',
  STDERR: 'stderr',
  PROGRESS: 'progress',
  RESULT: 'result',
  ERROR: 'error',
  COMPLETE: 'complete'
};

/**
 * JobListener - Central listener for all job output streams
 *
 * Provides demultiplexing of output by job ID, allowing subscribers
 * to receive only output for specific jobs.
 */
class JobListener extends EventEmitter {
  constructor() {
    super();
    // Map of jobId -> { stdout: Buffer[], stderr: Buffer[], subscribers: Set }
    this.jobs = new Map();
    // Maximum buffer size per stream (in characters)
    this.maxBufferSize = 1024 * 1024; // 1MB
  }

  /**
   * Register a new job for listening
   * @param {string} jobId - Job identifier
   */
  registerJob(jobId) {
    if (this.jobs.has(jobId)) {
      return;
    }

    this.jobs.set(jobId, {
      stdout: [],
      stderr: [],
      progress: 0,
      subscribers: new Set(),
      startTime: Date.now()
    });

    this.emit('job:registered', jobId);
  }

  /**
   * Unregister a job and clean up resources
   * @param {string} jobId - Job identifier
   */
  unregisterJob(jobId) {
    const job = this.jobs.get(jobId);
    if (job) {
      // Notify subscribers that job is being removed
      job.subscribers.forEach(callback => {
        callback({ type: 'unregistered', jobId });
      });
      this.jobs.delete(jobId);
      this.emit('job:unregistered', jobId);
    }
  }

  /**
   * Write data to a job's stream
   * @param {string} jobId - Job identifier
   * @param {string} streamType - Stream type (stdout/stderr)
   * @param {string} data - Data to write
   */
  write(jobId, streamType, data) {
    const job = this.jobs.get(jobId);
    if (!job) {
      // Auto-register if not exists
      this.registerJob(jobId);
      return this.write(jobId, streamType, data);
    }

    // Store in buffer
    if (streamType === StreamType.STDOUT) {
      job.stdout.push(data);
      // Trim buffer if too large
      this._trimBuffer(job.stdout);
    } else if (streamType === StreamType.STDERR) {
      job.stderr.push(data);
      this._trimBuffer(job.stderr);
    }

    // Create message for subscribers
    const message = {
      type: streamType,
      jobId,
      data,
      timestamp: Date.now()
    };

    // Notify subscribers
    job.subscribers.forEach(callback => {
      try {
        callback(message);
      } catch (err) {
        console.error(`Error in job listener callback for ${jobId}:`, err);
      }
    });

    // Emit global event
    this.emit(`job:${streamType}`, jobId, data);
    this.emit('job:output', message);
  }

  /**
   * Report progress for a job
   * @param {string} jobId - Job identifier
   * @param {number} progress - Progress value (0-100)
   */
  progress(jobId, progress) {
    const job = this.jobs.get(jobId);
    if (!job) {
      this.registerJob(jobId);
      return this.progress(jobId, progress);
    }

    job.progress = progress;

    const message = {
      type: StreamType.PROGRESS,
      jobId,
      progress,
      timestamp: Date.now()
    };

    job.subscribers.forEach(callback => {
      try {
        callback(message);
      } catch (err) {
        console.error(`Error in progress callback for ${jobId}:`, err);
      }
    });

    this.emit('job:progress', jobId, progress);
  }

  /**
   * Report job result
   * @param {string} jobId - Job identifier
   * @param {Object} result - Job result
   * @param {number} exitCode - Exit code (0 = success)
   */
  result(jobId, result, exitCode = 0) {
    const job = this.jobs.get(jobId);
    if (!job) return;

    const message = {
      type: StreamType.RESULT,
      jobId,
      result,
      exitCode,
      timestamp: Date.now(),
      duration: Date.now() - job.startTime
    };

    job.subscribers.forEach(callback => {
      try {
        callback(message);
      } catch (err) {
        console.error(`Error in result callback for ${jobId}:`, err);
      }
    });

    this.emit('job:result', jobId, result, exitCode);
  }

  /**
   * Report job error
   * @param {string} jobId - Job identifier
   * @param {Error|string} error - Error object or message
   * @param {number} exitCode - Exit code (non-zero for errors)
   */
  error(jobId, error, exitCode = 1) {
    const job = this.jobs.get(jobId);
    if (!job) return;

    const errorMessage = error instanceof Error ? error.message : error;
    const errorStack = error instanceof Error ? error.stack : null;

    const message = {
      type: StreamType.ERROR,
      jobId,
      error: errorMessage,
      stack: errorStack,
      exitCode,
      timestamp: Date.now(),
      duration: Date.now() - job.startTime
    };

    job.subscribers.forEach(callback => {
      try {
        callback(message);
      } catch (err) {
        console.error(`Error in error callback for ${jobId}:`, err);
      }
    });

    this.emit('job:error', jobId, error, exitCode);
  }

  /**
   * Signal job completion with final exit code
   * @param {string} jobId - Job identifier
   * @param {number} exitCode - Exit code (0 = success, non-zero = failure)
   */
  complete(jobId, exitCode = 0) {
    const job = this.jobs.get(jobId);
    if (!job) return;

    const message = {
      type: StreamType.COMPLETE,
      jobId,
      exitCode,
      timestamp: Date.now(),
      duration: Date.now() - job.startTime
    };

    job.subscribers.forEach(callback => {
      try {
        callback(message);
      } catch (err) {
        console.error(`Error in complete callback for ${jobId}:`, err);
      }
    });

    this.emit('job:complete', jobId, exitCode);
  }

  /**
   * Subscribe to a job's output
   * @param {string} jobId - Job identifier
   * @param {Function} callback - Callback function (message) => void
   * @returns {Function} Unsubscribe function
   */
  subscribe(jobId, callback) {
    const job = this.jobs.get(jobId);
    if (!job) {
      this.registerJob(jobId);
      return this.subscribe(jobId, callback);
    }

    job.subscribers.add(callback);

    // Return unsubscribe function
    return () => {
      job.subscribers.delete(callback);
    };
  }

  /**
   * Get buffered output for a job
   * @param {string} jobId - Job identifier
   * @param {string} streamType - Optional stream type filter
   * @returns {Object} Buffered output { stdout, stderr, progress }
   */
  getBuffer(jobId, streamType = null) {
    const job = this.jobs.get(jobId);
    if (!job) {
      return { stdout: '', stderr: '', progress: 0 };
    }

    if (streamType === StreamType.STDOUT) {
      return job.stdout.join('');
    } else if (streamType === StreamType.STDERR) {
      return job.stderr.join('');
    }

    return {
      stdout: job.stdout.join(''),
      stderr: job.stderr.join(''),
      progress: job.progress
    };
  }

  /**
   * Create an emitter function for a specific job
   * Used by workers to send output back to the listener
   * @param {string} jobId - Job identifier
   * @returns {Function} Emit function (streamType, data) => void
   */
  createEmitter(jobId) {
    this.registerJob(jobId);

    return (streamType, data) => {
      switch (streamType) {
        case StreamType.STDOUT:
        case StreamType.STDERR:
          this.write(jobId, streamType, data);
          break;
        case StreamType.PROGRESS:
          this.progress(jobId, data);
          break;
        case StreamType.RESULT:
          this.result(jobId, data);
          break;
        case StreamType.ERROR:
          this.error(jobId, data);
          break;
        default:
          // Treat unknown types as stdout
          this.write(jobId, StreamType.STDOUT, data);
      }
    };
  }

  /**
   * Get statistics about active jobs
   * @returns {Object} Statistics
   */
  getStats() {
    const stats = {
      activeJobs: this.jobs.size,
      totalSubscribers: 0,
      jobs: []
    };

    this.jobs.forEach((job, jobId) => {
      stats.totalSubscribers += job.subscribers.size;
      stats.jobs.push({
        jobId,
        subscribers: job.subscribers.size,
        stdoutSize: job.stdout.join('').length,
        stderrSize: job.stderr.join('').length,
        progress: job.progress,
        age: Date.now() - job.startTime
      });
    });

    return stats;
  }

  /**
   * Trim buffer to stay under max size
   * @private
   */
  _trimBuffer(buffer) {
    let totalSize = buffer.reduce((sum, chunk) => sum + chunk.length, 0);

    while (totalSize > this.maxBufferSize && buffer.length > 1) {
      const removed = buffer.shift();
      totalSize -= removed.length;
    }
  }

  /**
   * Clear all jobs and subscribers
   */
  clear() {
    this.jobs.forEach((job, jobId) => {
      job.subscribers.forEach(callback => {
        callback({ type: 'cleared', jobId });
      });
    });
    this.jobs.clear();
    this.emit('cleared');
  }
}

// Export singleton instance and class
const defaultListener = new JobListener();

module.exports = JobListener;
module.exports.default = defaultListener;
module.exports.StreamType = StreamType;
