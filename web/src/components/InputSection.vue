<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  loading: boolean
}>()

const emit = defineEmits(['submit'])

const modelUrl = defineModel('url', { type: String })
const modelSpeed = defineModel('speed', { type: String })

const showSpeedMenu = ref(false)
const selectedMode = ref('Smart')

const speedOptions = ['Smart', 'Fast', 'Detailed', 'Custom']

const selectMode = (mode: string) => {
  selectedMode.value = mode
  modelSpeed.value = mode
  showSpeedMenu.value = false
}
</script>

<template>
  <div class="input-section">
    <div class="input-group">
      <!-- Link Icon -->
      <div class="icon-wrapper">
        <i class="fas fa-link"></i>
      </div>

      <!-- Input Field -->
      <input
        v-model="modelUrl"
        type="text"
        class="input-url"
        placeholder="Paste Github or Gitee URL link here..."
        @keyup.enter="emit('submit')"
      />

      <!-- Speed Selector -->
      <div class="speed-selector">
        <button
          class="speed-button"
          @click="showSpeedMenu = !showSpeedMenu"
          title="Select analysis speed"
        >
          <span class="speed-label">{{ selectedMode }}</span>
          <i class="fas fa-chevron-down"></i>
        </button>
        <div v-if="showSpeedMenu" class="speed-menu">
          <div
            v-for="option in speedOptions"
            :key="option"
            class="speed-option"
            :class="{ active: option === selectedMode }"
            @click="selectMode(option)"
          >
            {{ option }}
          </div>
        </div>
      </div>

      <!-- Submit Button -->
      <button class="btn-submit" @click="emit('submit')" :disabled="loading" title="Submit">
        <i class="fas fa-arrow-right"></i>
      </button>
    </div>
  </div>
</template>

<style scoped>
.input-section {
  width: 100%;
  max-width: 700px;
  margin: 0 auto 60px;
}

.input-group {
  display: flex;
  gap: 0;
  align-items: center;
  background: var(--input-bg, white);
  border: 1px solid var(--border-color, #e8e8e8);
  border-radius: 12px;
  padding: 0;
  overflow: visible;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.input-group:focus-within {
  border-color: var(--border-color);
  box-shadow: 0 4px 16px var(--focus-color);
}

.icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 16px;
  color: var(--icon-color, #999);
  font-size: 16px;
  flex-shrink: 0;
}

.input-url {
  flex: 1;
  padding: 14px 12px;
  border: none;
  font-size: 14px;
  background: transparent;
  color: var(--text-color);
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

.input-url::placeholder {
  color: var(--placeholder-color, #bbb);
}

.input-url:focus {
  outline: none;
}

.speed-selector {
  position: relative;
  border-left: 1px solid var(--border-color, #f0f0f0);
  padding: 0 12px;
  flex-shrink: 0;
}

.speed-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: transparent;
  border: none;
  cursor: pointer;
  font-size: 12px;
  color: var(--secondary-text, #666);
  transition: all 0.2s ease;
  font-weight: 500;
  white-space: nowrap;
}

.speed-button:hover {
  color: var(--text-color);
  background: var(--hover-bg);
  border-radius: 6px;
}

.speed-button i {
  font-size: 11px;
  transition: transform 0.2s ease;
}

.speed-menu {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 8px;
  background: var(--menu-bg, white);
  border: 1px solid var(--border-color, #e8e8e8);
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  z-index: 10;
  min-width: 120px;
  overflow: hidden;
}

.speed-option {
  padding: 10px 16px;
  cursor: pointer;
  font-size: 13px;
  color: var(--secondary-text, #666);
  transition: all 0.2s ease;
  border-bottom: 1px solid var(--border-color, #f5f5f5);
}

.speed-option:last-child {
  border-bottom: none;
}

.speed-option:hover {
  background: var(--hover-bg);
  color: var(--text-color);
}

.speed-option.active {
  background: var(--hover-bg);
  color: var(--text-color);
  font-weight: 600;
}

.btn-submit {
  padding: 12px 18px;
  background: transparent;
  border: none;
  border-left: 1px solid var(--border-color, #f0f0f0);
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--secondary-text, #666);
}

.btn-submit i {
  font-size: 18px;
}

.btn-submit:hover:not(:disabled) {
  color: var(--text-color);
  background: var(--hover-bg);
}

.btn-submit:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
