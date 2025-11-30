<template>
  <div v-if="repoName" class="repo-header">
    <div class="repo-link" @click="handleClick">
      <i :class="platform === 'github' ? 'fab fa-github' : 'fab fa-git-alt'"></i>
      <span class="repo-link-text">{{ repoName }}</span>
      <i class="fas fa-external-link-alt small-icon"></i>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  repoName: string
  platform?: string
}

const props = withDefaults(defineProps<Props>(), {
  platform: 'github',
})

const emit = defineEmits<{
  (e: 'click'): void
}>()

const handleClick = () => {
  emit('click')
}
</script>

<style scoped>
.repo-header {
  position: fixed;
  top: 20px;
  left: 60px;
  z-index: 950;
}

.repo-link {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--text-color);
  cursor: pointer;
  font-size: 16px;
  font-weight: 600;
  transition: color 0.2s ease;
  height: 40px;
  line-height: 40px;
}

.repo-link:hover {
  color: var(--title-color);
}

.repo-link-text {
  max-width: 320px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.small-icon {
  font-size: 13px;
}
</style>
