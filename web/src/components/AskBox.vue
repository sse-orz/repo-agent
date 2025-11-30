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
        :model-value="modelValue"
        class="ask-input"
        type="text"
        :placeholder="placeholder"
        @keyup.enter="handleSend"
        @input="handleInput"
      />
      <button class="send-btn" @click="handleSend" title="Send question">
        <i class="fas fa-arrow-right"></i>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  modelValue: string
  placeholder?: string
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: 'Try to ask me...',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'send'): void
  (e: 'new-repo'): void
}>()

const handleSend = () => {
  if (!props.modelValue) return
  emit('send')
}

const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement
  emit('update:modelValue', target.value)
}

const handleNewRepo = () => {
  emit('new-repo')
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
