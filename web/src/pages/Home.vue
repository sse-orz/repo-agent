<template>
  <div class="home-page">
    <TopControls />
    <div class="container">
      <Header />
      <InputSection
        v-model:url="repoUrl"
        v-model:mode="mode"
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
import TopControls from '../components/TopControls.vue'
import { generateDocStream } from '../utils/request'

const router = useRouter()
const repoUrl = ref('')
const mode = ref<'sub' | 'moe'>('sub')
const isLoading = ref(false)

const handleSubmit = async () => {
  if (!repoUrl.value) return
  isLoading.value = true
  try {
    const url = repoUrl.value.trim()
    let owner: string,
      repo: string,
      platform: string = 'github'

    // Parse URL to extract owner, repo, and platform
    if (url.includes('github.com')) {
      platform = 'github'
      // Extract path after github.com
      const match = url.match(/github\.com\/([^/]+)\/([^/\s]+)/)
      if (!match || match.length < 3) throw new Error('Invalid GitHub URL')
      owner = match[1]!
      repo = match[2]!.replace(/\.git$/, '') // Remove .git suffix if present
    } else if (url.includes('gitee.com')) {
      platform = 'gitee'
      // Extract path after gitee.com
      const match = url.match(/gitee\.com\/([^/]+)\/([^/\s]+)/)
      if (!match || match.length < 3) throw new Error('Invalid Gitee URL')
      owner = match[1]!
      repo = match[2]!.replace(/\.git$/, '') // Remove .git suffix if present
    } else {
      // Try to parse as owner/repo format
      const parts = url.split('/').filter((part) => part)
      if (parts.length < 2) throw new Error('Invalid repo format')
      owner = parts[0]!
      repo = parts[1]!.replace(/\.git$/, '')
      // Default to github for owner/repo format
      platform = 'github'
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
    generateDocStream({ mode: mode.value, request: { owner, repo, platform } }, () => {}, {
      signal: controller.signal,
    }).catch((e) => {
      console.error('Stream start error:', e)
    })
    router.push({
      name: 'RepoDetail',
      params: { repoId: `${owner}_${repo}` },
      query: { mode: mode.value, platform },
    })
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
