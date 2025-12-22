<template>
  <div class="dialog-overlay" @click.self="$emit('close')">
    <div class="dialog">
      <div class="dialog-header">
        <h3>{{ profile ? 'Profile' : 'Create Profile' }}</h3>
        <button @click="$emit('close')" class="close-btn">&times;</button>
      </div>

      <!-- Profile Form -->
      <div class="form-section">
        <div class="form-group">
          <label>Name</label>
          <input v-model="name" placeholder="Your name" />
        </div>
        <div class="form-group">
          <label>Email</label>
          <input v-model="email" type="email" placeholder="your.email@example.com" />
        </div>
        <button @click="saveProfile" class="save-btn" :disabled="saving || !name || !email">
          {{ saving ? 'Saving...' : (profile ? 'Update Profile' : 'Create Profile') }}
        </button>
      </div>

      <!-- SSH Key Section (only show if profile exists) -->
      <div class="ssh-section" v-if="profile">
        <div class="section-header">
          <h4>SSH Key</h4>
          <span class="fingerprint" v-if="profile.ssh_key_fingerprint">
            {{ profile.ssh_key_fingerprint }}
          </span>
        </div>

        <div v-if="profile.ssh_public_key" class="ssh-key-display">
          <div class="key-header">
            <span>Public Key</span>
            <button @click="copyPublicKey" class="copy-btn" :class="{ copied }">
              {{ copied ? 'Copied!' : 'Copy' }}
            </button>
          </div>
          <textarea readonly :value="profile.ssh_public_key" class="key-text"></textarea>
          <p class="key-hint">
            Add this key to your GitHub account at
            <a href="https://github.com/settings/ssh/new" target="_blank">github.com/settings/ssh/new</a>
          </p>
        </div>

        <div v-else class="no-key">
          <p>No SSH key configured. Generate one to clone repositories via SSH.</p>
        </div>

        <button
          @click="generateSSHKey"
          class="generate-btn"
          :disabled="generating"
        >
          {{ generating ? 'Generating...' : (profile.ssh_public_key ? 'Regenerate SSH Key' : 'Generate SSH Key') }}
        </button>

        <p class="warning" v-if="profile.ssh_public_key">
          Regenerating will replace your current key. You'll need to update it on GitHub.
        </p>
      </div>

      <!-- Status Messages -->
      <div class="status" v-if="statusMessage" :class="statusType">
        {{ statusMessage }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const emit = defineEmits(['close', 'updated'])

const profile = ref(null)
const name = ref('')
const email = ref('')
const saving = ref(false)
const generating = ref(false)
const copied = ref(false)
const statusMessage = ref('')
const statusType = ref('info')

async function loadProfile() {
  try {
    const response = await axios.get('/api/profile')
    if (response.data) {
      profile.value = response.data
      name.value = response.data.name
      email.value = response.data.email
    }
  } catch (error) {
    console.error('Error loading profile:', error)
  }
}

async function saveProfile() {
  if (!name.value || !email.value) return

  saving.value = true
  statusMessage.value = ''

  try {
    const response = await axios.post('/api/profile', {
      name: name.value,
      email: email.value
    })
    profile.value = response.data
    statusMessage.value = 'Profile saved successfully'
    statusType.value = 'success'
    emit('updated', response.data)
  } catch (error) {
    statusMessage.value = error.response?.data?.error || 'Failed to save profile'
    statusType.value = 'error'
  } finally {
    saving.value = false
  }
}

async function generateSSHKey() {
  if (!profile.value) return

  generating.value = true
  statusMessage.value = ''

  try {
    const response = await axios.post(`/api/profile/${profile.value.id}/generate-ssh-key`)
    profile.value.ssh_public_key = response.data.publicKey
    profile.value.ssh_key_fingerprint = response.data.fingerprint
    statusMessage.value = 'SSH key generated successfully'
    statusType.value = 'success'
    emit('updated', profile.value)
  } catch (error) {
    statusMessage.value = error.response?.data?.error || 'Failed to generate SSH key'
    statusType.value = 'error'
  } finally {
    generating.value = false
  }
}

async function copyPublicKey() {
  if (!profile.value?.ssh_public_key) return

  try {
    await navigator.clipboard.writeText(profile.value.ssh_public_key)
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  } catch (error) {
    // Fallback for older browsers
    const textarea = document.createElement('textarea')
    textarea.value = profile.value.ssh_public_key
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    document.body.removeChild(textarea)
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  }
}

onMounted(() => {
  loadProfile()
})
</script>

<style scoped>
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.dialog {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  width: 480px;
  max-width: 90%;
  max-height: 85vh;
  overflow-y: auto;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.dialog-header h3 {
  margin: 0;
  color: var(--text-primary);
  font-size: 1.1rem;
}

.close-btn {
  width: 28px;
  height: 28px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 1.2rem;
}

.close-btn:hover {
  color: var(--error);
}

.form-section {
  margin-bottom: 1.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.35rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.form-group input {
  width: 100%;
  padding: 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.form-group input:focus {
  outline: none;
  border-color: var(--accent);
}

.save-btn {
  width: 100%;
  padding: 0.6rem;
  background: var(--accent);
  border: none;
  border-radius: 4px;
  color: var(--bg-primary);
  font-weight: 600;
  cursor: pointer;
  font-size: 0.9rem;
}

.save-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.save-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.ssh-section {
  margin-bottom: 1rem;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.section-header h4 {
  margin: 0;
  font-size: 0.95rem;
  color: var(--text-primary);
}

.fingerprint {
  font-family: monospace;
  font-size: 0.7rem;
  color: var(--text-secondary);
  background: var(--bg-primary);
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
}

.ssh-key-display {
  margin-bottom: 1rem;
}

.key-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.key-header span {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.copy-btn {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--text-secondary);
  cursor: pointer;
}

.copy-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.copy-btn.copied {
  background: rgba(74, 222, 128, 0.2);
  border-color: #4ade80;
  color: #4ade80;
}

.key-text {
  width: 100%;
  height: 80px;
  padding: 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
  font-family: monospace;
  font-size: 0.75rem;
  resize: none;
}

.key-hint {
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.key-hint a {
  color: var(--accent);
  text-decoration: none;
}

.key-hint a:hover {
  text-decoration: underline;
}

.no-key {
  padding: 1rem;
  background: var(--bg-primary);
  border-radius: 4px;
  margin-bottom: 1rem;
}

.no-key p {
  margin: 0;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.generate-btn {
  width: 100%;
  padding: 0.6rem;
  background: var(--bg-primary);
  border: 1px solid var(--accent);
  border-radius: 4px;
  color: var(--accent);
  font-weight: 600;
  cursor: pointer;
  font-size: 0.9rem;
}

.generate-btn:hover:not(:disabled) {
  background: rgba(0, 212, 255, 0.1);
}

.generate-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.warning {
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: #f59e0b;
}

.status {
  margin-top: 1rem;
  padding: 0.5rem 0.75rem;
  border-radius: 4px;
  font-size: 0.85rem;
}

.status.success {
  background: rgba(74, 222, 128, 0.15);
  color: #4ade80;
  border: 1px solid rgba(74, 222, 128, 0.3);
}

.status.error {
  background: rgba(248, 113, 113, 0.15);
  color: #f87171;
  border: 1px solid rgba(248, 113, 113, 0.3);
}

.status.info {
  background: rgba(0, 212, 255, 0.15);
  color: var(--accent);
  border: 1px solid rgba(0, 212, 255, 0.3);
}
</style>
