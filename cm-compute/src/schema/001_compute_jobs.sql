-- Compute jobs table for tracking job execution
-- DROP INDEX IF EXISTS idx_compute_jobs_status;
-- DROP INDEX IF EXISTS idx_compute_jobs_created_at;
-- DROP INDEX IF EXISTS idx_compute_jobs_deleted_at;
-- DROP TABLE IF EXISTS compute_jobs;

CREATE TABLE IF NOT EXISTS compute_jobs (
  id BIGSERIAL PRIMARY KEY,
  type VARCHAR(255) NOT NULL,
  params JSONB,
  priority INTEGER DEFAULT 5,
  status VARCHAR(50) DEFAULT 'queued',
  result JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  deleted_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_compute_jobs_status ON compute_jobs(status);
CREATE INDEX IF NOT EXISTS idx_compute_jobs_created_at ON compute_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_compute_jobs_deleted_at ON compute_jobs(deleted_at);
