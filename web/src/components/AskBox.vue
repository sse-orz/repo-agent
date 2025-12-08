<template>
  <div class="ask-row">
    <button class="new-repo" @click="handleNewRepo" aria-label="New repo">
      <span class="plus">+</span>
    </button>

    <div class="ask-box">
      <div class="ask-icon-wrapper">
        <i class="fas fa-comment-dots"></i>
      </div>
      <input
        :value="modelValue"
        class="ask-input"
        type="text"
        :placeholder="placeholder"
        @keyup.enter="handleSend"
        @input="handleInput"
      />
      <div class="mode-selector">
        <button
          class="mode-button"
          type="button"
          @click="showModeMenu = !showModeMenu"
          title="Select RAG mode"
        >
          <span class="mode-label">{{ currentModeLabel }}</span>
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
      <button
        class="send-btn"
        @click="props.isLoading ? handleAbort() : handleSend()"
        :title="props.isLoading ? 'Abort request' : 'Send question'"
      >
        <i :class="props.isLoading ? 'fas fa-stop-circle' : 'fas fa-arrow-right'"></i>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

interface Props {
  modelValue: string
  placeholder?: string
  mode?: 'fast' | 'smart'
  isLoading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: 'Try to ask me...',
  mode: 'fast',
  isLoading: false,
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'send'): void
  (e: 'new-repo'): void
  (e: 'update:mode', value: 'fast' | 'smart'): void
  (e: 'mode-change', value: 'fast' | 'smart'): void
  (e: 'abort'): void
}>()

const showModeMenu = ref(false)
const selectedMode = ref<'fast' | 'smart'>(props.mode === 'smart' ? 'smart' : 'fast')

// sync the external mode to the local selected state
watch(
  () => props.mode,
  (newMode) => {
    if (newMode === 'fast' || newMode === 'smart') {
      selectedMode.value = newMode
    }
  }
)

const modeOptions = [
  { value: 'fast' as const, label: 'Fast' },
  { value: 'smart' as const, label: 'Smart' },
]

const currentModeLabel = computed(
  () => modeOptions.find((opt) => opt.value === selectedMode.value)?.label || selectedMode.value
)

const handleSend = () => {
  if (!props.modelValue) return
  // emit send first, then clear input
  emit('send')
  emit('update:modelValue', '')
}

const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement
  emit('update:modelValue', target.value)
}

const handleNewRepo = () => {
  emit('new-repo')
}

const selectMode = (value: 'fast' | 'smart') => {
  // if the value is the same as the selected mode, don't do anything
  if (value === selectedMode.value) {
    showModeMenu.value = false
    return
  }

  selectedMode.value = value
  emit('update:mode', value)
  emit('mode-change', value)
  showModeMenu.value = false
}

const handleAbort = () => {
  emit('abort')
}
</script>

<style scoped>
.ask-row {
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  bottom: 24px;
  width: min(720px, 90%);
  display: flex;
  align-items: center;
  gap: 10px;
  z-index: 1200;
}

.ask-box {
  flex: 1 1 auto;
  display: flex;
  align-items: center;
  background: var(--input-bg, #ffffff);
  border: 1px solid var(--border-color, #e8e8e8);
  border-radius: 12px;
  padding: 0 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
  height: 44px;
}

.ask-box:focus-within {
  border-color: var(--border-color);
  box-shadow: 0 4px 16px var(--focus-color);
}

.ask-icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 10px;
  color: var(--icon-color, #999);
  font-size: 16px;
  flex-shrink: 0;
}

.ask-input {
  flex: 1 1 auto;
  border: none;
  outline: none;
  background: transparent;
  font-family: inherit;
  font-size: 14px;
  color: var(--text-color);
  padding: 0 6px;
  height: 100%;
  line-height: 44px;
}

.ask-input::placeholder {
  color: var(--placeholder-color, #bbb);
}

.mode-selector {
  position: relative;
  border-left: 1px solid var(--border-color, #f0f0f0);
  padding: 0 8px;
  flex-shrink: 0;
}

.mode-button {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
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
  bottom: 100%;
  right: 0;
  margin-bottom: 6px;
  background: var(--menu-bg, white);
  border: 1px solid var(--border-color, #e8e8e8);
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  z-index: 10;
  min-width: 110px;
  overflow: hidden;
}

.mode-option {
  padding: 8px 14px;
  cursor: pointer;
  font-size: 12px;
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

.send-btn {
  padding: 10px 14px;
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

.send-btn i {
  font-size: 18px;
}

.send-btn:hover {
  color: var(--text-color);
  background: var(--hover-bg);
}

.new-repo {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--card-bg);
  border: 2px solid var(--border-color);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 2px 8px var(--shadow-color);
  transition: all 0.2s ease;
  color: var(--text-color);
}

.new-repo:hover {
  background: var(--hover-bg);
  border-color: var(--border-color);
  transform: scale(1.05);
}

.new-repo .plus {
  font-size: 20px;
  line-height: 1;
  color: var(--text-color);
}
</style>
