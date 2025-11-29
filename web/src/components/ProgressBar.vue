<template>
  <div class="progress-container">
    <div class="progress-header">
      <span class="progress-label">Progress</span>
      <span class="progress-percent">{{ progressPercent }}%</span>
    </div>
    <div class="progress-bar-wrapper">
      <div class="progress-bar" :style="{ width: progressPercent + '%' }"></div>
    </div>
    <div v-if="logs.length > 0" class="stream-log">
      <p v-for="(log, index) in logs" :key="index">{{ log }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  progress?: number
  logs?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  progress: 0,
  logs: () => [],
})

const progressPercent = computed(() => {
  return Math.min(100, Math.max(0, props.progress))
})
</script>

<style scoped>
.progress-container {
  padding: 24px 0;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.progress-label {
  font-size: 15px;
  font-weight: 600;
  color: var(--title-color);
  letter-spacing: 0.3px;
  transition: color 0.3s;
}

.progress-percent {
  font-size: 15px;
  font-weight: 600;
  color: var(--title-color);
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.3px;
  transition: color 0.3s;
}

.progress-bar-wrapper {
  width: 100%;
  height: 6px;
  background: var(--border-color);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 28px;
  transition: background-color 0.3s;
}

.progress-bar {
  height: 100%;
  background: var(--title-color);
  border-radius: 3px;
  transition:
    width 0.6s cubic-bezier(0.4, 0, 0.2, 1),
    background-color 0.3s;
  position: relative;
  overflow: hidden;
  opacity: 0.85;
}

.progress-bar::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.2) 50%,
    transparent 100%
  );
  animation: shimmer 2.5s infinite;
  opacity: 0.6;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

.stream-log {
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  font-size: 13px;
  line-height: 1.8;
  color: var(--secondary-text);
  transition: color 0.3s;
}

.stream-log p {
  margin: 0 0 10px;
  padding: 8px 0;
  position: relative;
  padding-left: 20px;
}

.stream-log p::before {
  content: '•';
  position: absolute;
  left: 0;
  color: var(--title-color);
  font-weight: 600;
  opacity: 0.6;
  transition:
    color 0.3s,
    opacity 0.3s;
}

.stream-log p:last-child {
  margin-bottom: 0;
}
</style>
