<template>
  <button
    class="theme-toggle"
    @click="toggleTheme"
    :aria-label="isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'"
  >
    <i :class="isDarkMode ? 'fas fa-sun' : 'fas fa-moon'"></i>
  </button>
</template>

<script setup lang="ts">
import { inject } from 'vue'

const { isDarkMode, toggleTheme } = inject('theme', { isDarkMode: false, toggleTheme: () => {} })
</script>

<style scoped>
.theme-toggle {
  position: fixed;
  top: 20px;
  right: 60px;
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  cursor: pointer;
  font-size: 16px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: all 0.3s ease;
  color: var(--text-color);
  z-index: 1000;
  box-shadow: 0 2px 8px var(--shadow-color);
}

.theme-toggle:hover {
  background: var(--hover-bg);
  transform: scale(1.05);
}

.theme-toggle i {
  transition: transform 0.3s;
}

.theme-toggle:hover i {
  transform: scale(1.1);
}
</style>
