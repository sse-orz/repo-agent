<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  loading: boolean
}>()

const emit = defineEmits(['submit'])

const modelUrl = defineModel('url', { type: String })
const modelMode = defineModel('mode', { type: String })

const showModeMenu = ref(false)
const selectedMode = ref<'sub' | 'moe'>(
  modelMode.value === 'sub' || modelMode.value === 'moe' ? modelMode.value : 'sub'
)

const modeOptions = [
  { value: 'sub' as const, label: 'Sub' },
  { value: 'moe' as const, label: 'Moe' },
]

// Initialize modelMode if not set
if (!modelMode.value || (modelMode.value !== 'sub' && modelMode.value !== 'moe')) {
  modelMode.value = 'sub'
}

const selectMode = (mode: 'sub' | 'moe') => {
  selectedMode.value = mode
  modelMode.value = mode
  showModeMenu.value = false
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

      <!-- Mode Selector -->
      <div class="mode-selector">
        <button
          class="mode-button"
          @click="showModeMenu = !showModeMenu"
          title="Select generation mode"
        >
          <span class="mode-label">{{
            modeOptions.find((opt) => opt.value === selectedMode)?.label || selectedMode
          }}</span>
          <i class="fas fa-chevron-down"></i>
        </button>
        <div v-if="showModeMenu" class="mode-menu">
          <div
            v-for="option in modeOptions"
            :key="option.value"
            class="mode-option"
            :class="{ active: option.value === selectedMode }"
            @click="selectMode(option.value)"
          >
            {{ option.label }}
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

.mode-selector {
  position: relative;
  border-left: 1px solid var(--border-color, #f0f0f0);
  padding: 0 12px;
  flex-shrink: 0;
}

.mode-button {
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

.mode-button:hover {
  color: var(--text-color);
  background: var(--hover-bg);
  border-radius: 6px;
}

.mode-button i {
  font-size: 11px;
  transition: transform 0.2s ease;
}

.mode-menu {
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

.mode-option {
  padding: 10px 16px;
  cursor: pointer;
  font-size: 13px;
  color: var(--secondary-text, #666);
  transition: all 0.2s ease;
  border-bottom: 1px solid var(--border-color, #f5f5f5);
}

.mode-option:last-child {
  border-bottom: none;
}

.mode-option:hover {
  background: var(--hover-bg);
  color: var(--text-color);
}

.mode-option.active {
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
