<template>
  <div class="top-controls">
    <ThemeToggle />
    <!-- <HomeButton /> -->
    <HistoryButton />
  </div>
</template>

<script setup lang="ts">
import ThemeToggle from './ThemeToggle.vue'
import HistoryButton from './HistoryButton.vue'
</script>

<style scoped>
.top-controls {
  position: fixed;
  top: 20px;
  right: 20px;
  display: flex;
  align-items: center;
  gap: 5px;
  z-index: 1000;
}
</style>
