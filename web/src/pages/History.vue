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
      <div v-else-if="!groupedWikis.length" class="status-text">No repositories generated yet.</div>

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
        <div v-for="item in groupedWikis" :key="`${item.owner}/${item.repo}`" class="repo-card">
          <div class="repo-main">
            <div class="repo-header">
              <h2 class="repo-name">{{ item.owner }} / {{ item.repo }}</h2>
              <div class="mode-selector-wrapper">
                <div class="mode-toggle">
                  <button
                    v-for="mode in ['sub', 'moe']"
                    :key="mode"
                    class="mode-btn"
                    :class="{
                      active: item.selectedMode === mode,
                      available: item.modes.includes(mode),
                    }"
                    @click.stop="selectMode(item, mode)"
                    :disabled="!item.modes.includes(mode)"
                  >
                    {{ mode.toUpperCase() }}
                  </button>
                </div>
              </div>
            </div>
            <p class="repo-meta">
              {{ item.currentInfo?.total_files || 0 }} files ·
              <span class="time">{{ formatTime(item.currentInfo?.generated_at) }}</span>
            </p>
          </div>
          <button class="enter-btn" aria-label="Open repo" @click.stop="goDetail(item)">
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
  mode: string
}

interface GroupedWikiItem {
  owner: string
  repo: string
  modes: string[]
  selectedMode: 'sub' | 'moe'
  currentInfo: WikiItem | null
  allModes: {
    sub?: WikiItem
    moe?: WikiItem
  }
}

interface ListData {
  wikis: WikiItem[]
  total_wikis: number
}

const router = useRouter()
const wikis = ref<WikiItem[]>([])
const groupedWikis = ref<GroupedWikiItem[]>([])
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
    groupWikisByRepo()
  } catch (err) {
    console.error(err)
    error.value = '加载历史仓库失败，请稍后重试。'
  } finally {
    loading.value = false
  }
}

const groupWikisByRepo = () => {
  const grouped = new Map<string, GroupedWikiItem>()

  for (const wiki of wikis.value) {
    const key = `${wiki.owner}_${wiki.repo}`

    if (!grouped.has(key)) {
      grouped.set(key, {
        owner: wiki.owner,
        repo: wiki.repo,
        modes: [],
        selectedMode: 'sub',
        currentInfo: null,
        allModes: {},
      })
    }

    const item = grouped.get(key)!

    // Only add modes that have documents (total_files > 0)
    if (wiki.total_files > 0) {
      item.modes.push(wiki.mode)
      item.allModes[wiki.mode as 'sub' | 'moe'] = wiki
    }
  }

  // Set default selected mode for each grouped item (prefer sub if available, otherwise moe)
  grouped.forEach((item) => {
    // Only select from modes that have documents
    if (item.modes.length > 0) {
      // Prefer 'sub' if available, otherwise use the first available mode
      if (item.modes.includes('sub')) {
        item.selectedMode = 'sub'
      } else if (item.modes.includes('moe')) {
        item.selectedMode = 'moe'
      } else {
        // Fallback to first available mode
        item.selectedMode = item.modes[0] as 'sub' | 'moe'
      }
    } else {
      // No modes with documents, reset to default
      item.selectedMode = 'sub'
    }

    // Set current info for the selected mode
    item.currentInfo = item.allModes[item.selectedMode] || null
  })

  groupedWikis.value = Array.from(grouped.values())
}

const selectMode = (item: GroupedWikiItem, mode: string) => {
  if (mode !== 'sub' && mode !== 'moe') return
  if (!item.modes.includes(mode)) return
  item.selectedMode = mode as 'sub' | 'moe'
  item.currentInfo = item.allModes[mode as 'sub' | 'moe'] || null
}

const goDetail = (item: GroupedWikiItem) => {
  router.push({
    name: 'RepoDetail',
    params: { repoId: `${item.owner}_${item.repo}` },
    query: { mode: item.selectedMode },
  })
}

const goHome = () => {
  router.push({ name: 'Home' })
}

const formatTime = (iso: string | undefined) => {
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

.repo-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 6px;
}

.repo-name {
  font-size: 15px;
  margin: 0;
  flex: 1;
  min-width: 0;
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

.mode-selector-wrapper {
  flex-shrink: 0;
}

.mode-toggle {
  display: flex;
  gap: 4px;
  background: var(--input-bg, rgba(0, 0, 0, 0.05));
  border-radius: 6px;
  padding: 2px;
}

.mode-btn {
  flex: 1;
  padding: 4px 8px;
  border: none;
  background: transparent;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  color: var(--subtitle-color);
  opacity: 0.5;
}

.mode-btn.available {
  opacity: 1;
}

.mode-btn.active {
  background: var(--card-bg);
  color: var(--text-color);
  box-shadow: 0 1px 3px var(--shadow-color);
}

.mode-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.mode-btn:not(:disabled):hover {
  opacity: 0.8;
}
</style>
