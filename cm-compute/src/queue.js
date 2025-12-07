const { fork } = require('child_process');
const path = require('path');

class ComputeQueue {
  constructor() {
    this.queue = [];
    this.running = new Map();
    this.workers = [];
    this.maxWorkers = 4;
  }

  enqueue(job, callback) {
    this.queue.push({ ...job, callback });
    this.queue.sort((a, b) => b.priority - a.priority);
    this.processNext();
  }

  dequeue() {
    return this.queue.shift();
  }

  processNext() {
    if (this.queue.length === 0 || this.running.size >= this.maxWorkers) {
      return;
    }

    const job = this.dequeue();
    if (!job) return;

    this.execute(job);
  }

  execute(job) {
    const worker = fork(path.join(__dirname, 'worker.js'));

    this.running.set(job.id, { worker, job });

    worker.send({
      type: 'execute',
      job: {
        id: job.id,
        type: job.type,
        params: job.params
      }
    });

    worker.on('message', (message) => {
      if (message.type === 'result') {
        this.running.delete(job.id);

        if (job.callback) {
          job.callback(message.result);
        }

        worker.kill();
        this.processNext();
      } else if (message.type === 'progress') {
        console.log(`Job ${job.id} progress:`, message.progress);
      }
    });

    worker.on('error', (error) => {
      console.error(`Worker error for job ${job.id}:`, error);
      this.running.delete(job.id);

      if (job.callback) {
        job.callback({ error: error.message });
      }

      worker.kill();
      this.processNext();
    });

    worker.on('exit', (code) => {
      if (code !== 0 && this.running.has(job.id)) {
        console.error(`Worker exited with code ${code} for job ${job.id}`);
        this.running.delete(job.id);

        if (job.callback) {
          job.callback({ error: `Worker exited with code ${code}` });
        }

        this.processNext();
      }
    });
  }

  cancel(jobId) {
    // Check if in queue
    const queueIndex = this.queue.findIndex(j => j.id === jobId);
    if (queueIndex !== -1) {
      this.queue.splice(queueIndex, 1);
      return true;
    }

    // Check if running
    const running = this.running.get(jobId);
    if (running) {
      running.worker.kill();
      this.running.delete(jobId);
      this.processNext();
      return true;
    }

    return false;
  }

  getPosition(jobId) {
    const index = this.queue.findIndex(j => j.id === jobId);
    return index === -1 ? null : index + 1;
  }

  getStats() {
    return {
      queued: this.queue.length,
      running: this.running.size,
      maxWorkers: this.maxWorkers
    };
  }

  startWorkers(count) {
    this.maxWorkers = count;
  }
}

module.exports = ComputeQueue;
