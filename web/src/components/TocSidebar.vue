<template>
  <aside class="toc">
    <div class="toc-tabs">
      <button
        class="toc-tab-btn"
        :class="{ active: activeTab === 'files' }"
        @click="activeTab = 'files'"
      >
        文件
      </button>
      <button
        class="toc-tab-btn"
        :class="{ active: activeTab === 'outline' }"
        @click="activeTab = 'outline'"
      >
        大纲
      </button>
    </div>

    <div class="toc-content">
      <!-- Files Tab -->
      <nav v-show="activeTab === 'files'">
        <div v-for="(section, si) in sections" :key="`sec-${si}`" class="toc-section">
          <div
            v-for="(item, ii) in displayItems(section)"
            :key="`item-${si}-${ii}-${item.fullPath || item.name}`"
          >
            <div
              class="toc-item"
              @click="handleItemClick(item)"
              :aria-current="item.url && item.url === selectedUrl ? 'true' : undefined"
              :style="{ paddingLeft: 8 + (item.depth || 0) * 12 + 'px' }"
            >
              <span class="toc-item-name">
                <span v-if="item.depth > 0" class="toc-pipes">
                  <span v-for="n in item.depth" :key="n" class="toc-pipe">|</span>
                  <span class="toc-pipe-connector">─</span>
                </span>
                <span class="toc-icon">
                  <i
                    v-if="item.isDir"
                    class="fas"
                    :class="item.expanded ? 'fa-folder-open' : 'fa-folder'"
                  ></i>
                  <i v-else class="fas fa-file-alt"></i>
                </span>
                <span class="toc-label">
                  {{ item.name }}
                </span>
              </span>
              <button
                v-if="item.isDir"
                class="toc-toggle"
                @click.stop="handleToggle(item)"
                aria-label="Toggle"
              >
                {{ item.expanded ? '▾' : '▸' }}
              </button>
            </div>
          </div>
        </div>
      </nav>

      <!-- Outline Tab -->
      <nav v-show="activeTab === 'outline'">
        <div v-if="activeFileItem && activeFileItem.headings && activeFileItem.headings.length">
          <div class="toc-headings">
            <div
              v-for="(h, hi) in activeFileItem.headings"
              :key="`heading-${hi}`"
              class="toc-heading"
              :style="{ paddingLeft: (h.level - 1) * 12 + 8 + 'px' }"
              @click="handleHeadingClick(activeFileItem!, h.id)"
            >
              {{ h.title }}
            </div>
          </div>
        </div>
        <div v-else class="toc-empty">
          <span v-if="!activeFileItem">请选择一个文件</span>
          <span v-else>当前文件无大纲</span>
        </div>
      </nav>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface HeadingItem {
  level: number
  title: string
  id: string
}

interface FileItem {
  name: string
  url?: string
  headings?: HeadingItem[]
  expanded?: boolean
  depth: number
  isDir: boolean
  children?: FileItem[]
  fullPath: string
}

interface TocSection {
  id: string
  title: string
  owner: string
  repo: string
  items?: FileItem[]
}

interface Props {
  sections: TocSection[]
  selectedUrl: string | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'item-click', item: FileItem): void
  (e: 'heading-click', item: FileItem, headingId: string): void
  (e: 'toggle', item: FileItem): void
}>()

const activeTab = ref<'files' | 'outline'>('files')

const flattenItems = (items?: FileItem[]): FileItem[] => {
  const result: FileItem[] = []
  const walk = (nodes: FileItem[]) => {
    for (const n of nodes) {
      result.push(n)
      if (n.isDir && n.expanded && n.children?.length) {
        walk(n.children)
      }
    }
  }
  if (items?.length) {
    walk(items)
  }
  return result
}

const displayItems = (section: TocSection) => {
  const items = flattenItems(section.items)
  return items || []
}

// Find the currently active file item from the sections tree
const activeFileItem = computed(() => {
  if (!props.selectedUrl) return null

  for (const section of props.sections) {
    if (!section.items) continue

    // Helper to search recursively in items
    const findInItems = (items: FileItem[]): FileItem | null => {
      for (const item of items) {
        if (item.url === props.selectedUrl) return item
        if (item.children && item.children.length) {
          const found = findInItems(item.children)
          if (found) return found
        }
      }
      return null
    }

    const found = findInItems(section.items)
    if (found) return found
  }
  return null
})

const handleItemClick = (item: FileItem) => {
  emit('item-click', item)
}

const handleHeadingClick = (item: FileItem, headingId: string) => {
  emit('heading-click', item, headingId)
}

const handleToggle = (item: FileItem) => {
  emit('toggle', item)
}
</script>

<style scoped>
.toc {
  /* 固定目录列宽度 */
  flex: 0 0 300px;
  max-width: 300px;
  min-width: 300px;
  padding: 24px 16px;
  margin-bottom: 100px;
  overflow: hidden; /* Changed to hidden to manage scroll in content */
  display: flex;
  flex-direction: column;
}

.toc-tabs {
  display: flex;
  gap: 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 16px;
}

.toc-tab-btn {
  background: none;
  border: none;
  font-size: 14px;
  font-weight: 600;
  color: var(--subtitle-color);
  cursor: pointer;
  padding: 4px 8px;
  position: relative;
  transition: color 0.2s;
}

.toc-tab-btn:hover {
  color: var(--text-color);
}

.toc-tab-btn.active {
  color: var(--text-color);
}

.toc-tab-btn.active::after {
  content: '';
  position: absolute;
  bottom: -17px; /* Adjust to align with border-bottom of container */
  left: 0;
  width: 100%;
  height: 2px;
  background: var(--text-color);
}

.toc-content {
  flex: 1;
  overflow-y: auto;
  /* Hide scrollbar */
  scrollbar-width: none;
}

.toc-content::-webkit-scrollbar {
  width: 0;
  height: 0;
}

.toc nav {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.toc-section {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.toc-item {
  padding: 6px 8px;
  cursor: pointer;
  color: var(--text-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-width: 0;
}

.toc-toggle {
  background: transparent;
  border: none;
  color: var(--muted-text-color, var(--text-color));
  cursor: pointer;
  margin-left: 6px;
  flex: 0 0 auto;
}

.toc-item-name {
  vertical-align: middle;
  flex: 1 1 auto;
  min-width: 0;
  display: flex; /* Changed to flex for icon alignment */
  align-items: center;
  gap: 6px;
  overflow: hidden;
  padding-right: 8px;
}

.toc-icon {
  flex-shrink: 0;
  width: 16px;
  text-align: center;
  color: var(--subtitle-color);
  font-size: 13px;
}

.toc-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.toc-pipes {
  display: inline-flex;
  align-items: center;
  margin-right: 4px;
  color: var(--muted-text-color, var(--text-color));
  opacity: 0.7;
  font-size: 11px;
}

.toc-pipe {
  margin-right: 1px;
}

.toc-pipe-connector {
  margin-right: 3px;
}

.toc-headings {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.toc-heading {
  padding: 4px 8px;
  cursor: pointer;
  font-size: 13px;
  color: var(--muted-text-color, var(--text-color));
  border-radius: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.toc-heading:hover {
  background: var(--card-bg);
  color: var(--text-color);
}

.toc-item[aria-current='true'] {
  background: var(--card-bg);
  border-radius: 6px;
  box-shadow: 0 1px 4px var(--shadow-color);
}

.toc-empty {
  padding: 16px;
  text-align: center;
  color: var(--subtitle-color);
  font-size: 14px;
}
</style>
