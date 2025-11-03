<script setup lang="ts">
import { ref, onMounted } from 'vue'
import Header from './components/Header.vue'
import InputSection from './components/InputSection.vue'
import InfoCard from './components/InfoCard.vue'

const repoUrl = ref('')
const speed = ref('fast')
const isLoading = ref(false)
const isDarkMode = ref(false)

const handleSubmit = async () => {
  if (!repoUrl.value) return
  isLoading.value = true
  // TODO: 发送请求到后端
  setTimeout(() => {
    isLoading.value = false
  }, 1000)
}

const toggleTheme = () => {
  isDarkMode.value = !isDarkMode.value
  document.body.classList.toggle('dark', isDarkMode.value)
  localStorage.setItem('theme', isDarkMode.value ? 'dark' : 'light')
}

onMounted(() => {
  const savedTheme = localStorage.getItem('theme')
  if (savedTheme === 'dark') {
    isDarkMode.value = true
    document.body.classList.add('dark')
  }
})
</script>

<template>
  <div class="app">
    <button
      class="theme-toggle"
      @click="toggleTheme"
      :aria-label="isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'"
    >
      <i :class="isDarkMode ? 'fas fa-sun' : 'fas fa-moon'"></i>
    </button>
    <div class="container">
      <Header />
      <InputSection
        v-model:url="repoUrl"
        v-model:speed="speed"
        :loading="isLoading"
        @submit="handleSubmit"
      />
      <InfoCard />
    </div>
  </div>
</template>

<style scoped>
.app {
  position: relative;
  min-height: 100vh;
}

.theme-toggle {
  position: fixed;
  top: 20px;
  right: 20px;
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  cursor: pointer;
  font-size: 16px;
  padding: 10px;
  border-radius: 50%;
  transition: all 0.3s ease;
  color: var(--text-color);
  z-index: 1000;
  box-shadow: 0 2px 8px var(--shadow-color);
}

.theme-toggle:hover {
  background: var(--hover-bg);
  transform: scale(1.05);
}

.theme-toggle i {
  transition: transform 0.3s;
}

.theme-toggle:hover i {
  transform: scale(1.1);
}

.container {
  min-height: 100vh;
  background: var(--container-bg);
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s;
}
</style>
