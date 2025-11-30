<template>
  <aside class="toc">
    <nav>
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
              <span class="toc-label">
                {{ item.isDir ? item.name + '/' : item.name }}
              </span>
            </span>
            <button
              v-if="item.isDir || (item.headings || []).length"
              class="toc-toggle"
              @click.stop="handleToggle(item)"
              aria-label="Toggle"
            >
              {{ item.expanded ? '▾' : '▸' }}
            </button>
          </div>

          <!-- headings for the file (expandable) -->
          <div
            v-if="!item.isDir && (item.headings || []).length && item.expanded"
            class="toc-headings"
          >
            <div
              v-for="(h, hi) in item.headings"
              :key="`heading-${si}-${ii}-${hi}`"
              class="toc-heading"
              @click.stop="handleHeadingClick(item, h.id)"
            >
              {{ h.title }}
            </div>
          </div>
        </div>
      </div>
    </nav>
  </aside>
</template>

<script setup lang="ts">
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
  overflow: auto;
}

/* 隐藏左侧导航滚动条但保持可滚动 */
.toc {
  scrollbar-width: none; /* Firefox */
}
.toc::-webkit-scrollbar {
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

.toc-title {
  font-weight: 600;
  padding-left: 8px;
  /* 多行截断：最多显示两行，超出部分以省略号表示 */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}

.toc-item {
  padding: 6px 8px;
  cursor: pointer;
  color: var(--text-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-width: 0; /* allow children to shrink inside flex */
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
  /* allow name to shrink and apply multi-line clamp inside this element */
  flex: 1 1 auto;
  min-width: 0; /* important for overflow handling in flex */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  word-break: break-word;
  padding-right: 8px;
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

.toc-label {
  display: inline-block;
  max-width: 100%;
}

.toc-headings {
  padding-left: 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 8px;
}

.toc-heading {
  padding: 4px 8px;
  cursor: pointer;
  font-size: 13px;
  color: var(--muted-text-color, var(--text-color));
  border-radius: 4px;
}

.toc-heading:hover {
  background: var(--card-bg);
}

.toc-item[aria-current='true'] {
  background: var(--card-bg);
  border-radius: 6px;
  box-shadow: 0 1px 4px var(--shadow-color);
}

.toc-title[aria-current='true'] {
  background: var(--card-bg);
  border-radius: 6px;
  box-shadow: 0 1px 4px var(--shadow-color);
}
</style>
