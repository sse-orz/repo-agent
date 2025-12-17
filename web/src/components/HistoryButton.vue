<template>
  <button
    class="history-button"
    @click="goHistory"
    aria-label="View generated repositories"
    title="History"
  >
    <i class="fas fa-clock-rotate-left"></i>
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
  width: 40px;
  height: 40px;
  background: transparent;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-color);
  box-shadow: none;
  transition:
    background 0.25s ease,
    transform 0.25s ease,
    color 0.25s ease;
}

.history-button:hover {
  background: rgba(0, 0, 0, 0.06);
  transform: translateY(-1px);
}

.history-button i {
  font-size: 16px;
  line-height: 1;
}
</style>
