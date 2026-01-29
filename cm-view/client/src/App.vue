<template>
  <div id="app">
    <header v-if="!isWorkspaceRoute" class="app-header">
      <h1>Chemical Machines</h1>
      <nav>
        <router-link to="/">Notebooks</router-link>
        <router-link to="/workspace">New Workspace</router-link>
      </nav>
    </header>
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const isWorkspaceRoute = computed(() => /^\/workspace\/\d+/.test(route.path))
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  background: #1a1a1a;
  color: #e0e0e0;
}

#app {
  width: 100vw;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: #2a2a2a;
  padding: 1rem 2rem;
  border-bottom: 1px solid #3a3a3a;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.app-header h1 {
  font-size: 1.5rem;
  color: #00d4ff;
}

.app-header nav {
  display: flex;
  gap: 1.5rem;
}

.app-header nav a {
  color: #e0e0e0;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background 0.2s;
}

.app-header nav a:hover {
  background: #3a3a3a;
}

.app-header nav a.router-link-active {
  background: rgba(0, 212, 255, 0.1);
  color: #00d4ff;
}

.app-main {
  flex: 1;
  overflow: hidden;
}
</style>
