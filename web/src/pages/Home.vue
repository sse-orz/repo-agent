<template>
  <div class="home-page">
    <ThemeToggle />
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

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import Header from '../components/Header.vue'
import InputSection from '../components/InputSection.vue'
import InfoCard from '../components/InfoCard.vue'
import ThemeToggle from '../components/ThemeToggle.vue'
import { generateDocStream } from '../utils/request'

const router = useRouter()
const repoUrl = ref('')
const speed = ref('fast')
const isLoading = ref(false)

const handleSubmit = async () => {
  if (!repoUrl.value) return
  isLoading.value = true
  try {
    const url = repoUrl.value.trim()
    let owner: string, repo: string
    if (url.includes('github.com')) {
      const parts = url.split('/')
      if (parts.length < 2) throw new Error('Invalid URL')
      owner = parts[parts.length - 2]!
      repo = parts[parts.length - 1]!
    } else {
      const parts = url.split('/')
      if (parts.length < 2) throw new Error('Invalid repo format')
      owner = parts[0]!
      repo = parts[1]!
    }
    if (!owner || !repo) {
      alert('Invalid repo URL')
      isLoading.value = false
      return
    }

    // Always start streaming generation and navigate to the RepoDetail page
    // The RepoDetail page will handle the streaming results and display progress
    const controller = new AbortController()
    // Fire-and-forget the stream; RepoDetail will create its own stream when mounted.
    generateDocStream({ owner, repo }, () => {}, { signal: controller.signal }).catch((e) => {
      console.error('Stream start error:', e)
    })
    router.push({ name: 'RepoDetail', params: { repoId: `${owner}_${repo}` } })
  } catch (e) {
    console.error(e)
    alert('Error occurred')
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
.home-page {
  min-height: 100vh;
  box-sizing: border-box;
}

.app {
  position: relative;
  min-height: 100vh;
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
