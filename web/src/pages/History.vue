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
        <div class="repo-card add-repo-card" @click="goHome">
          <div class="repo-main">
            <div>
              <h2 class="repo-name">Add repo</h2>
            </div>
            <div class="repo-footer add-repo-footer">
              <div class="mode-selector-wrapper">
                <div class="mode-toggle placeholder-mode-toggle">
                  <button class="mode-btn placeholder-btn">SUB</button>
                  <button class="mode-btn placeholder-btn">MOE</button>
                </div>
              </div>
              <p class="repo-meta">Paste a new repo link to generate docs</p>
            </div>
          </div>
          <button class="enter-btn add-btn" aria-label="Add repo" @click.stop="goHome">
            <i class="fas fa-plus"></i>
          </button>
        </div>

        <!-- History tiles -->
        <div v-for="item in groupedWikis" :key="`${item.owner}/${item.repo}`" class="repo-card">
          <div class="repo-main">
            <h2 class="repo-name" :title="`${item.owner} / ${item.repo}`">
              {{ item.owner }} / {{ item.repo }}
            </h2>
            <div class="repo-footer">
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
              <p class="repo-meta">
                {{ item.currentInfo?.total_files || 0 }} files ·
                <span class="time">{{ formatTime(item.currentInfo?.generated_at) }}</span>
              </p>
            </div>
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
  max-width: 1280px;
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
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 20px;
}

.repo-card {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 20px 24px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px var(--shadow-color);
  min-height: 120px;
}

.repo-card:hover {
  transform: translateY(-2px);
  border-color: var(--border-color);
  background: var(--hover-bg);
}

.add-repo-card {
  background: linear-gradient(
    135deg,
    rgba(99, 102, 241, 0.1) 0%,
    rgba(139, 92, 246, 0.1) 50%,
    rgba(59, 130, 246, 0.1) 100%
  );
  border-color: rgba(99, 102, 241, 0.3);
}

.add-repo-card:hover {
  background: linear-gradient(
    135deg,
    rgba(99, 102, 241, 0.15) 0%,
    rgba(139, 92, 246, 0.15) 50%,
    rgba(59, 130, 246, 0.15) 100%
  );
  border-color: rgba(99, 102, 241, 0.4);
}

.repo-main {
  flex: 1;
  min-width: 0;
  margin-right: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  justify-content: space-between;
  min-height: 100%;
}

.repo-name {
  font-size: 15px;
  margin: 0;
  padding: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.4;
  font-weight: 600;
  color: var(--text-color);
}

.repo-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  min-height: 24px;
}

.add-repo-footer {
  justify-content: space-between;
}

.placeholder-mode-toggle {
  visibility: hidden;
  pointer-events: none;
  height: 20px;
  display: inline-flex;
  gap: 3px;
  background: var(--input-bg, rgba(0, 0, 0, 0.04));
  border-radius: 8px;
  padding: 3px;
  border: 1px solid var(--border-color);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.placeholder-btn {
  visibility: hidden;
  pointer-events: none;
}

.repo-meta {
  margin: 0;
  font-size: 12px;
  color: var(--subtitle-color);
  flex-shrink: 0;
  line-height: 1.4;
  display: inline-flex;
  align-items: center;
}

.repo-meta .time {
  opacity: 0.9;
}

.enter-btn {
  width: 32px;
  height: 32px;
  min-width: 32px;
  flex-shrink: 0;
  border-radius: 50%;
  border: 1px solid var(--border-color);
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: var(--text-color);
  transition: all 0.2s ease;
  margin-top: 0;
}

.repo-card:hover .enter-btn {
  background: var(--card-bg);
}

.add-btn {
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

.repo-card:hover .add-btn {
  background: var(--card-bg);
}

.mode-selector-wrapper {
  flex-shrink: 0;
}

.mode-toggle {
  display: inline-flex;
  gap: 3px;
  background: var(--input-bg, rgba(0, 0, 0, 0.04));
  border-radius: 8px;
  padding: 3px;
  border: 1px solid var(--border-color);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.mode-btn {
  flex: 1;
  padding: 4px 12px;
  border: none;
  background: transparent;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.3px;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  color: var(--subtitle-color);
  min-width: 44px;
  text-align: center;
  line-height: 1.4;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.mode-btn.available {
  color: var(--text-color);
  opacity: 0.65;
}

.mode-btn.active {
  background: var(--card-bg);
  color: var(--text-color);
  opacity: 1;
  font-weight: 700;
  transform: translateY(-0.5px);
}

.mode-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
  color: var(--subtitle-color);
}

.mode-btn:not(:disabled):hover {
  opacity: 0.85;
  background: var(--hover-bg);
  transform: translateY(-0.5px);
}

.mode-btn.active:not(:disabled):hover {
  opacity: 1;
  background: var(--card-bg);
  transform: translateY(-0.5px);
}
</style>
