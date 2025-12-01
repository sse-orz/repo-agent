<template>
  <main class="doc">
    <div class="doc-inner" ref="docInnerRef">
      <ProgressBar
        v-if="isStreaming && progressLogs.length > 0"
        :progress="progress"
        :logs="progressLogs"
      />
      <div v-html="content"></div>
    </div>
    <div v-if="showTop" class="fade fade-top" aria-hidden="true"></div>
    <div v-if="showBottom" class="fade fade-bottom" aria-hidden="true"></div>
  </main>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import ProgressBar from './ProgressBar.vue'

interface Props {
  content: string
  isStreaming?: boolean
  progressLogs?: string[]
  progress?: number
}

const props = withDefaults(defineProps<Props>(), {
  isStreaming: false,
  progressLogs: () => [],
  progress: 0,
})

const docInnerRef = ref<HTMLElement | null>(null)
const showTop = ref(false)
const showBottom = ref(false)

const updateFades = () => {
  const el = docInnerRef.value
  if (!el) return
  showTop.value = el.scrollTop > 0
  showBottom.value =
    el.scrollHeight > el.clientHeight && el.scrollTop + el.clientHeight < el.scrollHeight - 1
}

const emit = defineEmits<{
  (e: 'scroll', event: Event): void
}>()

onMounted(() => {
  nextTick(() => {
    updateFades()
    const el = docInnerRef.value
    if (!el) return
    const scrollHandler = () => {
      updateFades()
      emit('scroll', new Event('scroll'))
    }
    el.addEventListener('scroll', scrollHandler, { passive: true })
    window.addEventListener('resize', updateFades)
  })
})

onUnmounted(() => {
  const el = docInnerRef.value
  if (el) {
    // Note: We can't remove the anonymous function, but this is fine for cleanup
    window.removeEventListener('resize', updateFades)
  }
})

watch(
  () => props.content,
  () => {
    nextTick(() => updateFades())
  }
)

defineExpose({
  scrollToTop: () => {
    if (docInnerRef.value) {
      docInnerRef.value.scrollTop = 0
    }
  },
  scrollToHeading: async (id: string) => {
    await nextTick()
    const container = docInnerRef.value
    if (!container) return
    try {
      const el = container.querySelector(`#${CSS.escape(id)}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    } catch (e) {
      // fallback: try find by name
      const el = container.querySelector(`[id='${id}']`)
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  },
  getElement: () => docInnerRef.value,
})
</script>

<style scoped>
.doc {
  /* 固定文档列宽度（包含左右 margin 和内边距） */
  flex: 0 0 900px;
  max-width: 900px;
  border-left: 1px solid var(--border-color);
  min-height: 0;
  margin: 20px 40px 100px 0;
  position: relative;
}

.doc-inner {
  /* make the inner doc area the only scrollable region */
  height: 100%;
  overflow-y: auto;
  overflow-x: auto;
  /* 左右内边距形成 markdown 内容与文档列边界之间的 margin */
  padding: 0 40px;
  background: transparent;
  line-height: 1.3;
  /* preserve newlines inside v-html content and allow long words to wrap */
  white-space: pre-wrap;
  overflow-wrap: break-word;
  word-break: break-word;
  scrollbar-width: none; /* Firefox */
}

.doc-inner::-webkit-scrollbar {
  width: 0;
  height: 0;
}

/* 限制 Markdown 实际内容宽度并居中显示 */
.doc-inner :deep(*) {
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}

/* 收紧 markdown 元素的上下间距，使版面更紧凑 */
.doc-inner :deep(h1),
.doc-inner :deep(h2),
.doc-inner :deep(h3),
.doc-inner :deep(h4),
.doc-inner :deep(h5),
.doc-inner :deep(h6) {
  margin-top: 1.2em;
  margin-bottom: 0.4em;
  color: var(--title-color);
}

.doc-inner :deep(p) {
  margin-top: 0.15em;
  margin-bottom: 0.35em;
}

.doc-inner :deep(ul),
.doc-inner :deep(ol) {
  margin-top: 0.25em;
  margin-bottom: 0.5em;
  padding-left: 1.5em;
}

.doc-inner :deep(li) {
  margin-top: 0.05em;
  margin-bottom: 0.05em;
}

.doc-inner :deep(pre) {
  background: var(--hover-bg);
  padding: 16px;
  border-radius: 6px;
  overflow: auto;
  margin-bottom: 1em;
  border: 1px solid var(--border-color);
}

.doc-inner :deep(code) {
  font-family:
    ui-monospace,
    SFMono-Regular,
    SF Mono,
    Menlo,
    Consolas,
    Liberation Mono,
    monospace;
  font-size: 85%;
  background: var(--hover-bg);
  padding: 0.2em 0.4em;
  border-radius: 6px;
}

.doc-inner :deep(pre code) {
  background: transparent;
  padding: 0;
  font-size: 100%;
  white-space: pre;
}

.doc-inner :deep(blockquote) {
  padding: 0 1em;
  color: var(--secondary-text);
  border-left: 0.25em solid var(--border-color);
  margin-bottom: 1em;
}

.doc-inner :deep(table) {
  border-spacing: 0;
  border-collapse: collapse;
  margin-bottom: 1em;
  width: 100%;
}

.doc-inner :deep(th),
.doc-inner :deep(td) {
  padding: 6px 13px;
  border: 1px solid var(--border-color);
}

.doc-inner :deep(tr:nth-child(2n)) {
  background-color: var(--bg-color);
}

.doc-inner :deep(img) {
  max-width: 100%;
  box-sizing: content-box;
  background-color: var(--bg-color);
}

.doc-inner :deep(a) {
  color: var(--link-color, #0969da);
  text-decoration: none;
}

.doc-inner :deep(a:hover) {
  text-decoration: underline;
}

/* Mermaid styles */
.doc-inner :deep(.mermaid-container) {
  position: relative;
  margin: 1em 0;
  text-align: center;
}

.doc-inner :deep(.mermaid) {
  background: var(--placeholder-color, #bbb);
  padding: 10px;
  border-radius: 8px;
  overflow-x: auto;
}

.doc-inner :deep(.mermaid .node rect),
.doc-inner :deep(.mermaid .node circle),
.doc-inner :deep(.mermaid .node polygon),
.doc-inner :deep(.mermaid .cluster rect),
.doc-inner :deep(.mermaid .label-container),
.doc-inner :deep(.mermaid .actor) {
  fill: var(--card-bg, #ffffff) !important;
  stroke: var(--border-color, #d1d5db) !important;
}

.doc-inner :deep(.mermaid text),
.doc-inner :deep(.mermaid span) {
  fill: var(--text-color, #111827) !important;
  color: var(--text-color, #111827) !important;
}

.doc-inner :deep(.mermaid-zoom-btn) {
  position: absolute;
  top: 10px;
  right: 10px;
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 5px 10px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s;
  z-index: 10;
  color: var(--text-color);
}

.doc-inner :deep(.mermaid-container:hover .mermaid-zoom-btn) {
  opacity: 1;
}

.fade {
  position: absolute;
  left: 0;
  right: 0;
  height: 48px;
  pointer-events: none;
  z-index: 8;
}

.fade-top {
  top: 0;
  background: linear-gradient(to bottom, var(--container-bg), transparent);
}

.fade-bottom {
  bottom: 0;
  background: linear-gradient(to top, var(--container-bg), transparent);
}
</style>
