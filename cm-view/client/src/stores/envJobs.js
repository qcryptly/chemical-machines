/**
 * Simple reactive store for tracking active environment installation jobs.
 * This allows the EnvironmentDetailDialog to reconnect to running jobs
 * when closed and reopened.
 */
import { reactive } from 'vue'

// Store structure: { [envName]: { jobId, status, logs, packages } }
const activeJobs = reactive({})

export function getActiveJob(envName) {
  return activeJobs[envName] || null
}

export function setActiveJob(envName, jobData) {
  activeJobs[envName] = {
    jobId: jobData.jobId,
    status: jobData.status || 'Installing...',
    logs: jobData.logs || [],
    packages: jobData.packages || [],
    startTime: jobData.startTime || Date.now()
  }
}

export function updateJobStatus(envName, status) {
  if (activeJobs[envName]) {
    activeJobs[envName].status = status
  }
}

export function appendJobLog(envName, log) {
  if (activeJobs[envName]) {
    activeJobs[envName].logs.push(log)
  }
}

export function clearActiveJob(envName) {
  delete activeJobs[envName]
}

export function hasActiveJob(envName) {
  return !!activeJobs[envName]
}
