<template>
  <div class="history-page">
    <ThemeToggle />
    <HistoryButton />

    <div class="history-container">
      <header class="history-header">
        <h1 class="title">Which repo would you like to understand?</h1>
        <p class="subtitle">Browse repositories that already have generated documentation.</p>
      </header>

      <div v-if="loading" class="status-text">Loading history...</div>
      <div v-else-if="error" class="status-text error">{{ error }}</div>
      <div v-else-if="!wikis.length" class="status-text">No repositories generated yet.</div>

      <div v-else class="grid">
        <!-- Add repo tile -->
        <div class="repo-card add-card" @click="goHome">
          <div class="add-main">
            <div class="add-icon">
              <i class="fas fa-plus"></i>
            </div>
            <div>
              <h2 class="repo-name">Add repo</h2>
              <p class="repo-meta">Paste a new repo link to generate docs</p>
            </div>
          </div>
        </div>

        <!-- History tiles -->
        <div
          v-for="item in wikis"
          :key="`${item.owner}/${item.repo}`"
          class="repo-card"
          @click="goDetail(item)"
        >
          <div class="repo-main">
            <h2 class="repo-name">{{ item.owner }} / {{ item.repo }}</h2>
            <p class="repo-meta">
              {{ item.total_files }} files ·
              <span class="time">{{ formatTime(item.generated_at) }}</span>
            </p>
          </div>
          <button class="enter-btn" aria-label="Open repo">
            <i class="fas fa-arrow-right"></i>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import ThemeToggle from '../components/ThemeToggle.vue'
import HistoryButton from '../components/HistoryButton.vue'
import { listDocs, type BaseResponse } from '../utils/request'

interface WikiItem {
  owner: string
  repo: string
  wiki_path: string
  wiki_url: string
  total_files: number
  generated_at: string
}

interface ListData {
  wikis: WikiItem[]
  total_wikis: number
}

const router = useRouter()
const wikis = ref<WikiItem[]>([])
const loading = ref(false)
const error = ref('')

const fetchHistory = async () => {
  loading.value = true
  error.value = ''
  try {
    const res: BaseResponse<ListData> = await listDocs()
    if (res.code !== 200) {
      throw new Error(res.message || 'Failed to load history')
    }
    wikis.value = res.data?.wikis || []
  } catch (err) {
    console.error(err)
    error.value = '加载历史仓库失败，请稍后重试。'
  } finally {
    loading.value = false
  }
}

const goDetail = (item: WikiItem) => {
  router.push({ name: 'RepoDetail', params: { repoId: `${item.owner}_${item.repo}` } })
}

const goHome = () => {
  router.push({ name: 'Home' })
}

const formatTime = (iso: string) => {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

onMounted(() => {
  void fetchHistory()
})
</script>

<style scoped>
.history-page {
  min-height: 100vh;
  background: var(--container-bg);
  color: var(--text-color);
}

.history-container {
  max-width: 1120px;
  margin: 0 auto;
  padding: 80px 24px 40px;
}

.title {
  font-size: 24px;
  font-weight: 600;
  margin: 0 0 8px;
}

.subtitle {
  margin: 0;
  font-size: 14px;
  color: var(--subtitle-color);
}

.history-header {
  margin-bottom: 24px;
}

.status-text {
  font-size: 14px;
  color: var(--subtitle-color);
  padding: 40px 0;
}

.status-text.error {
  color: #ff6b6b;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}

.repo-card {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 14px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px var(--shadow-color);
}

.repo-card:hover {
  transform: translateY(-2px);
  border-color: var(--border-color);
  background: var(--hover-bg);
}

.repo-name {
  font-size: 15px;
  margin: 0 0 6px;
}

.repo-meta {
  margin: 0;
  font-size: 12px;
  color: var(--subtitle-color);
}

.repo-meta .time {
  opacity: 0.9;
}

.enter-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 1px solid var(--border-color);
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: var(--text-color);
  transition: all 0.2s ease;
}

.repo-card:hover .enter-btn {
  background: var(--card-bg);
}

.add-card {
  justify-content: flex-start;
}

.add-main {
  display: flex;
  align-items: center;
  gap: 12px;
}

.add-icon {
  width: 32px;
  height: 32px;
  border-radius: 999px;
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}
</style>
