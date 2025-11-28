<template>
  <button class="history-button" @click="goHistory" aria-label="View generated repositories">
    <span class="label">History</span>
  </button>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'

const router = useRouter()

const goHistory = () => {
  router.push({ name: 'History' })
}
</script>

<style scoped>
.history-button {
  position: fixed;
  top: 20px;
  right: 60px;
  height: 40px;
  padding: 0 18px;
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 999px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-color);
  box-shadow: 0 2px 8px var(--shadow-color);
  transition: all 0.3s ease;
  z-index: 1000;
}

.history-button:hover {
  background: var(--hover-bg);
  transform: translateY(-1px);
}
</style>
