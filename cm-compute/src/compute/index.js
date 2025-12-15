/**
 * Compute Module - Main entry point
 *
 * Exports the compute queue, worker management, and job listener
 * for the cm-compute daemon.
 */

const { ComputeQueue, ExitCode } = require('./queue');
const JobListener = require('./job-listener');
const { StreamType } = require('./job-listener');
const { jobRegistry, registerJob, getJobHandler } = require('./jobs');

module.exports = {
  ComputeQueue,
  ExitCode,
  JobListener,
  StreamType,
  jobRegistry,
  registerJob,
  getJobHandler
};
