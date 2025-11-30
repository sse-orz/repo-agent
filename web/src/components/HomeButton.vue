<template>
  <button class="home-button" @click="goHome" aria-label="Return to home">
    <span class="label">Home</span>
  </button>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'

const router = useRouter()

const goHome = () => {
  router.push({ name: 'Home' })
}
</script>

<style scoped>
.home-button {
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
}

.home-button:hover {
  background: var(--hover-bg);
  transform: translateY(-1px);
}
</style>
