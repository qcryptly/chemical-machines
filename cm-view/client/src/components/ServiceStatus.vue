<template>
  <div class="service-status" :class="statusClass">
    <div class="status-content">
      <div class="status-icon">
        <!-- Show spinner during loading OR during startup phase -->
        <div v-if="isLoading || (isStartingUp && overallStatus !== 'healthy')" class="spinner"></div>
        <svg v-else-if="overallStatus === 'healthy'" viewBox="0 0 24 24" fill="currentColor">
          <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
        </svg>
        <svg v-else-if="overallStatus === 'degraded'" viewBox="0 0 24 24" fill="currentColor">
          <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
        </svg>
        <svg v-else viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
      </div>

      <div class="status-text">
        <span class="status-title">{{ statusTitle }}</span>
        <span class="status-detail" v-if="statusDetail">{{ statusDetail }}</span>
      </div>

      <button v-if="showRetry" @click="checkHealth" class="retry-btn" :disabled="isLoading">
        {{ isLoading ? 'Checking...' : 'Retry' }}
      </button>
    </div>

    <div v-if="showDetails && services" class="service-details">
      <div
        v-for="(service, name) in services"
        :key="name"
        class="service-item"
        :class="getServiceClass(service.status)"
      >
        <span class="service-name">{{ formatServiceName(name) }}</span>
        <span class="service-status-badge">{{ getServiceBadgeText(service.status) }}</span>
        <span v-if="service.message && !isStartingUp" class="service-message">{{ service.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import axios from 'axios'

const props = defineProps({
  showDetails: {
    type: Boolean,
    default: false
  },
  autoRetry: {
    type: Boolean,
    default: true
  },
  retryInterval: {
    type: Number,
    default: 5000 // 5 seconds
  }
})

const emit = defineEmits(['status-change', 'healthy'])

const isLoading = ref(true)
const overallStatus = ref('checking')
const services = ref(null)
const retryTimer = ref(null)
const retryCount = ref(0)

// During startup (first 30 seconds / 6 retries), show loading state instead of error
const isStartingUp = computed(() => retryCount.value < 6)

const statusClass = computed(() => ({
  loading: isLoading.value || (isStartingUp.value && overallStatus.value !== 'healthy'),
  healthy: overallStatus.value === 'healthy',
  degraded: !isStartingUp.value && overallStatus.value === 'degraded',
  unhealthy: !isStartingUp.value && overallStatus.value === 'unhealthy'
}))

const showRetry = computed(() =>
  !isLoading.value && !isStartingUp.value && overallStatus.value !== 'healthy'
)

const statusTitle = computed(() => {
  if (isLoading.value) return 'Connecting to services...'
  if (overallStatus.value === 'healthy') return 'All services ready'

  // During startup, show friendly "starting" message
  if (isStartingUp.value) return 'Starting services...'

  if (overallStatus.value === 'degraded') return 'Some services unavailable'
  return 'Services unavailable'
})

const statusDetail = computed(() => {
  if (isLoading.value) return null
  if (overallStatus.value === 'healthy') return null

  const unhealthyServices = services.value
    ? Object.entries(services.value)
        .filter(([, s]) => s.status !== 'healthy')
        .map(([name]) => formatServiceName(name))
    : []

  if (unhealthyServices.length > 0) {
    // During startup, say "waiting" instead of listing as errors
    if (isStartingUp.value) {
      return `Waiting for: ${unhealthyServices.join(', ')}`
    }
    return `Unavailable: ${unhealthyServices.join(', ')}`
  }
  return isStartingUp.value ? 'Initializing...' : null
})

function formatServiceName(name) {
  const names = {
    database: 'Database',
    compute: 'Compute Engine',
    elasticsearch: 'Search'
  }
  return names[name] || name
}

function getServiceClass(status) {
  if (status === 'healthy') return 'healthy'
  // During startup, show "starting" style instead of "unhealthy"
  if (isStartingUp.value) return 'starting'
  return status
}

function getServiceBadgeText(status) {
  if (status === 'healthy') return 'ready'
  // During startup, show "starting" instead of "unhealthy"
  if (isStartingUp.value) return 'starting'
  return status
}

async function checkHealth() {
  isLoading.value = true

  try {
    const response = await axios.get('/api/health', { timeout: 10000 })
    overallStatus.value = response.data.status
    services.value = response.data.services

    emit('status-change', response.data)

    if (response.data.status === 'healthy') {
      emit('healthy')
      stopRetry()
      retryCount.value = 0
    } else if (props.autoRetry) {
      retryCount.value++
      scheduleRetry()
    }
  } catch (error) {
    overallStatus.value = 'unhealthy'
    services.value = null

    emit('status-change', { status: 'unhealthy', error: error.message })

    if (props.autoRetry) {
      retryCount.value++
      scheduleRetry()
    }
  } finally {
    isLoading.value = false
  }
}

function scheduleRetry() {
  stopRetry()
  retryTimer.value = setTimeout(checkHealth, props.retryInterval)
}

function stopRetry() {
  if (retryTimer.value) {
    clearTimeout(retryTimer.value)
    retryTimer.value = null
  }
}

onMounted(() => {
  checkHealth()
})

onUnmounted(() => {
  stopRetry()
})

defineExpose({ checkHealth })
</script>

<style scoped>
.service-status {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.service-status.healthy {
  border-color: #22c55e;
  background: rgba(34, 197, 94, 0.05);
}

.service-status.degraded {
  border-color: #f59e0b;
  background: rgba(245, 158, 11, 0.05);
}

.service-status.unhealthy {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.05);
}

.status-content {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.status-icon {
  width: 32px;
  height: 32px;
  flex-shrink: 0;
}

.status-icon svg {
  width: 100%;
  height: 100%;
}

.service-status.healthy .status-icon {
  color: #22c55e;
}

.service-status.degraded .status-icon {
  color: #f59e0b;
}

.service-status.unhealthy .status-icon {
  color: #ef4444;
}

.service-status.loading .status-icon {
  color: var(--accent);
}

.spinner {
  width: 100%;
  height: 100%;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.status-text {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.status-title {
  font-size: 1rem;
  font-weight: 500;
  color: var(--text-primary);
}

.status-detail {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.retry-btn {
  padding: 0.5rem 1rem;
  background: var(--accent);
  color: var(--bg-primary);
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: opacity 0.2s;
}

.retry-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.retry-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.service-details {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.service-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem;
  background: var(--bg-primary);
  border-radius: 4px;
  font-size: 0.85rem;
}

.service-name {
  font-weight: 500;
  color: var(--text-primary);
  min-width: 120px;
}

.service-status-badge {
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  font-size: 0.75rem;
  text-transform: uppercase;
  font-weight: 600;
}

.service-item.healthy .service-status-badge {
  background: rgba(34, 197, 94, 0.15);
  color: #22c55e;
}

.service-item.starting .service-status-badge {
  background: rgba(0, 212, 255, 0.15);
  color: var(--accent);
}

.service-item.unhealthy .service-status-badge {
  background: rgba(239, 68, 68, 0.15);
  color: #ef4444;
}

.service-message {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin-left: auto;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
