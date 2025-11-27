<template>
  <div class="repo-detail">
    <ThemeToggle />
    <div class="content-wrapper">
      <aside class="toc">
        <nav>
          <div v-for="(section, si) in tocSections" :key="`sec-${si}`" class="toc-section">
            <div v-for="(item, ii) in section.items || []" :key="`item-${si}-${ii}`">
              <div
                class="toc-item"
                @click="selectItem(si, ii)"
                :aria-current="selected.section === si && selected.item === ii ? 'true' : undefined"
              >
                <span class="toc-item-name">{{ item.name }}</span>
                <button
                  class="toc-toggle"
                  @click.stop="toggleExpand(si, ii)"
                  aria-label="Toggle headings"
                >
                  {{ item.expanded ? '▾' : '▸' }}
                </button>
              </div>

              <!-- headings for the file (expandable) -->
              <div v-if="(item.headings || []).length && item.expanded" class="toc-headings">
                <div
                  v-for="(h, hi) in item.headings"
                  :key="`heading-${si}-${ii}-${hi}`"
                  class="toc-heading"
                  @click.stop="scrollToHeading(h.id)"
                >
                  {{ h.title }}
                </div>
              </div>
            </div>
          </div>
        </nav>
      </aside>

      <main class="doc">
        <div class="doc-inner" ref="docInnerRef" v-html="docContent"></div>
        <div v-if="showTop" class="fade fade-top" aria-hidden="true"></div>
        <div v-if="showBottom" class="fade fade-bottom" aria-hidden="true"></div>
      </main>
    </div>

    <div class="ask-row">
      <div class="ask-box">
        <textarea
          v-model="query"
          class="ask-input"
          :placeholder="placeholder"
          @keydown.enter.prevent="handleSend"
          rows="3"
        ></textarea>
        <button class="send-btn" @click="handleSend">Send-&gt;</button>
      </div>

      <button class="new-repo" @click="router.push('/')" aria-label="New repo">
        <span class="plus">+</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import ThemeToggle from '../components/ThemeToggle.vue'
import { generateDocStream, resolveBackendStaticUrl, type BaseResponse } from '../utils/request'
import MarkdownIt from 'markdown-it'
import anchor from 'markdown-it-anchor'

interface HeadingItem {
  level: number
  title: string
  id: string
}

interface FileItem {
  name: string
  url: string
  headings?: HeadingItem[]
  expanded?: boolean
}

interface TocSection {
  id: string
  title: string
  owner: string
  repo: string
  items?: FileItem[]
}

interface StreamData {
  stage?: string
  message?: string
  progress?: number
  wiki_url?: string
  error?: string
  [key: string]: unknown
}

const route = useRoute()
const router = useRouter()

const repoId = ref<string>(typeof route.params.repoId === 'string' ? route.params.repoId : '')
const placeholder = ref('Try to ask me...')
const docInnerRef = ref<HTMLElement | null>(null)
const showTop = ref(false)
const showBottom = ref(false)
const docContent = ref('<p>Select a repository to view documentation.</p>')
const tocSections = ref<TocSection[]>([])
const query = ref('')
const selected = ref<{ section: number | null; item: number | null }>({ section: null, item: null })
const isStreaming = ref(false)
const progressLogs = ref<string[]>([])
const streamController = ref<AbortController | null>(null)

const escapeHtml = (value: string) =>
  String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const extractTitleFromMarkdown = (content: string) => {
  const lines = content.split('\n')
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('# ')) {
      return trimmed.slice(2).trim()
    }
  }
  return ''
}

// Markdown renderer + TOC extractor (markdown-it + markdown-it-anchor)
const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
}).use(anchor, {
  permalink: false,
  // keep default slugify behavior
})

const renderMarkdown = (markdown: string) => {
  const env: Record<string, unknown> = {}
  const html = md.render(markdown, env)

  // parse tokens to extract headings and their ids
  const tokens = md.parse(markdown, env)
  const headings: { level: number; title: string; id: string }[] = []
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]
    if (t.type === 'heading_open') {
      const level = Number(t.tag.replace('h', '')) || 1
      // try to find id in attrs
      let id = ''
      if (Array.isArray(t.attrs)) {
        const attrs = t.attrs as [string, string][]
        const idAttr = attrs.find((a) => a[0] === 'id')
        if (idAttr) id = String(idAttr[1])
      }
      // next token should be inline with content
      const inline = tokens[i + 1]
      const title = inline && inline.type === 'inline' ? String(inline.content || '') : ''
      if (id && title) headings.push({ level, title, id })
    }
  }

  return { html, headings }
}

const attachHeadingsToResolvedUrl = (
  section: TocSection,
  fileUrl: string,
  headings: { level: number; title: string; id: string }[] | undefined
) => {
  if (!section || !section.items || !fileUrl || !headings || !headings.length) return
  const resolved = resolveBackendStaticUrl(String(fileUrl))
  for (const it of section.items) {
    if (it.url === resolved) {
      ;(it as FileItem).headings = headings.map((h) => ({
        level: h.level,
        title: h.title,
        id: h.id,
      }))
      break
    }
  }
}

const scrollToHeading = async (id: string) => {
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
}

const renderProgress = () => {
  if (!progressLogs.value.length) {
    docContent.value = '<p>Waiting for updates...</p>'
    return
  }
  docContent.value = `<div class="stream-log">${progressLogs.value
    .map((msg) => `<p>${escapeHtml(msg)}</p>`)
    .join('')}</div>`
}

const updateFades = () => {
  const el = docInnerRef.value
  if (!el) return
  showTop.value = el.scrollTop > 0
  showBottom.value =
    el.scrollHeight > el.clientHeight && el.scrollTop + el.clientHeight < el.scrollHeight - 1
}

const findSectionIndexById = (id: string) =>
  tocSections.value.findIndex((section) => section.id === id)

const createSectionFromId = (id: string): TocSection | null => {
  const separator = id.indexOf('_')
  if (separator === -1) return null
  const owner = id.slice(0, separator)
  const repo = id.slice(separator + 1)
  if (!owner || !repo) return null
  return {
    id,
    title: `${owner}/${repo}`,
    owner,
    repo,
    items: [],
  }
}

async function loadDocumentation(section: TocSection, needUpdate = false) {
  if (!section) return

  if (streamController.value) {
    streamController.value.abort()
    streamController.value = null
  }

  const controller = new AbortController()
  streamController.value = controller
  isStreaming.value = true
  progressLogs.value = ['Connecting to documentation stream...']
  renderProgress()

  let loadedFromStream = false

  try {
    const lastEvent = await generateDocStream(
      { owner: section.owner, repo: section.repo, need_update: needUpdate },
      async (event: BaseResponse<unknown>) => {
        if (!event) return

        if (event.code !== 200) {
          const message = event.message || 'Unexpected response from stream.'
          progressLogs.value.push(message)
          const errorDetail = (event.data as StreamData | undefined)?.error
          if (errorDetail) {
            progressLogs.value.push(errorDetail)
          }
          renderProgress()
          return
        }

        const data = event.data as StreamData

        if (data?.stage) {
          const message = data.message || event.message || `Processing ${data.stage}`
          progressLogs.value.push(message)
          renderProgress()
          await nextTick()
          if (docInnerRef.value) {
            docInnerRef.value.scrollTop = docInnerRef.value.scrollHeight
          }
          return
        }

        if (data?.error) {
          progressLogs.value.push(data.error)
          renderProgress()
          return
        }

        if (data?.wiki_url) {
          progressLogs.value.push('Documentation ready.')
          renderProgress()
          // Render generated file list (do not fetch from /wikis)
          const files = (data.files as unknown[]) || []
          if (files.length === 1) {
            // Directly load the single file
            const file = files[0] as Record<string, unknown>
            const url = typeof file.url === 'string' ? file.url : undefined
            let fileHeadings: HeadingItem[] | undefined = undefined
            let resolvedUrl = ''
            if (url) {
              try {
                const resolved = resolveBackendStaticUrl(url)
                resolvedUrl = resolved
                progressLogs.value.push(`Loading file from ${resolved}`)
                renderProgress()
                const response = await fetch(resolved)
                if (response.ok) {
                  const content = await response.text()
                  // If looks like markdown, render HTML + extract headings
                  const isMd = typeof url === 'string' && /\.md$/i.test(url)
                  if (isMd || content.trim().startsWith('#') || content.includes('\n# ')) {
                    const { html, headings } = renderMarkdown(content)
                    docContent.value = html
                    fileHeadings = headings.map((h) => ({
                      level: h.level,
                      title: h.title,
                      id: h.id,
                    }))
                    const title = extractTitleFromMarkdown(content)
                    section.title = title || section.repo
                  } else {
                    docContent.value = `<pre>${escapeHtml(content)}</pre>`
                    const title = extractTitleFromMarkdown(content)
                    if (title) {
                      section.title = title
                    } else {
                      section.title = section.repo
                    }
                  }
                } else {
                  docContent.value = `<p>Failed to load document: ${response.status}</p>`
                }
              } catch (error) {
                docContent.value = `<p>Error loading document: ${String(error)}</p>`
              }
            }
            // Populate TOC with the single file (resolve to backend full URL)
            section.items = (files as unknown[]).map((f) => {
              const rec = f as Record<string, unknown>
              const urlVal = typeof rec.url === 'string' ? rec.url : String(rec.url ?? '')
              const u = resolveBackendStaticUrl(String(urlVal))
              return {
                name: String(rec.path ?? rec.name ?? rec.url ?? 'file'),
                url: u,
                headings: u === resolvedUrl ? fileHeadings : undefined,
              }
            })
          } else if (files.length > 1) {
            const listHtml = `<h3>Generated Files (${files.length})</h3><ul>${(files as unknown[])
              .map((f) => {
                const rec = f as Record<string, unknown>
                return `<li><a href="#" data-url="${escapeHtml(String(rec.url ?? ''))}">${escapeHtml(
                  String(rec.path ?? rec.name ?? rec.url ?? 'file')
                )}</a></li>`
              })
              .join('')}</ul>`
            docContent.value = listHtml
            // Update TOC with files (resolve to backend full URL immediately)
            section.items = (files as unknown[]).map((f) => {
              const rec = f as Record<string, unknown>
              return {
                name: String(rec.path ?? rec.name ?? rec.url ?? 'file'),
                url: resolveBackendStaticUrl(String(rec.url ?? '')),
              }
            })
          } else {
            docContent.value = `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`
          }
          loadedFromStream = true
          await nextTick()
          if (docInnerRef.value) {
            docInnerRef.value.scrollTop = 0
          }
          updateFades()

          // Fetch title from first file if available (only for multiple files)
          if (files.length > 1 && files.length > 0) {
            const firstFile = files[0] as Record<string, unknown>
            const url = typeof firstFile.url === 'string' ? firstFile.url : undefined
            if (url) {
              try {
                const resolved = resolveBackendStaticUrl(String(url))
                const response = await fetch(resolved)
                if (response.ok) {
                  const content = await response.text()
                  const title = extractTitleFromMarkdown(content)
                  if (title) {
                    section.title = title
                  } else {
                    section.title = section.repo
                  }
                }
              } catch (e) {
                console.error('Failed to fetch title:', e)
                section.title = section.repo
              }
            }
          }
        }
      },
      { signal: controller.signal }
    )

    if (!loadedFromStream && lastEvent?.code === 200) {
      const data = lastEvent.data as StreamData | undefined
      // Render files or data summary instead of fetching /wikis
      const files = (data as unknown as Record<string, unknown>)?.files || []
      if (Array.isArray(files) && files.length === 1) {
        // Directly load the single file
        const file = files[0] as Record<string, unknown>
        const url = typeof file.url === 'string' ? file.url : undefined
        if (url) {
          try {
            const response = await fetch(resolveBackendStaticUrl(url))
            if (response.ok) {
              const content = await response.text()
              const isMd = typeof url === 'string' && /\.md$/i.test(url)
              let fileHeadings: HeadingItem[] | undefined = undefined
              let resolvedUrl = ''
              if (url) resolvedUrl = resolveBackendStaticUrl(url)
              if (isMd || content.trim().startsWith('#') || content.includes('\n# ')) {
                const { html, headings } = renderMarkdown(content)
                docContent.value = html
                fileHeadings = headings.map((h) => ({ level: h.level, title: h.title, id: h.id }))
                const title = extractTitleFromMarkdown(content)
                section.title = title || section.repo
              } else {
                docContent.value = `<pre>${escapeHtml(content)}</pre>`
                const title = extractTitleFromMarkdown(content)
                if (title) section.title = title
                else section.title = section.repo
              }
              // populate TOC linking and attach headings if present
              section.items = (files as unknown[]).map((f) => {
                const rec = f as Record<string, unknown>
                const urlVal = typeof rec.url === 'string' ? rec.url : String(rec.url ?? '')
                const u = resolveBackendStaticUrl(String(urlVal))
                return {
                  name: String(rec.path ?? rec.name ?? rec.url ?? 'file'),
                  url: u,
                  headings: u === resolvedUrl ? fileHeadings : undefined,
                }
              })
            } else {
              docContent.value = `<p>Failed to load document: ${response.status}</p>`
            }
          } catch (error) {
            docContent.value = `<p>Error loading document: ${String(error)}</p>`
          }
        }
        // Populate TOC with the single file (resolve to backend full URL)
        section.items = (files as unknown[]).map((f) => {
          const rec = f as Record<string, unknown>
          return {
            name: String(rec.path ?? rec.name ?? rec.url ?? 'file'),
            url: resolveBackendStaticUrl(String(rec.url ?? '')),
          }
        })
      } else if (Array.isArray(files) && files.length > 1) {
        docContent.value = `<h3>Generated Files (${files.length})</h3><ul>${(files as unknown[])
          .map((f) => {
            const rec = f as Record<string, unknown>
            return `<li>${escapeHtml(String(rec.path ?? rec.name ?? rec.url ?? 'file'))}</li>`
          })
          .join('')}</ul>`
        // Update TOC with files (resolve to backend full URL immediately)
        section.items = (files as unknown[]).map((f) => {
          const rec = f as Record<string, unknown>
          return {
            name: String(rec.path ?? rec.name ?? rec.url ?? 'file'),
            url: resolveBackendStaticUrl(String(rec.url ?? '')),
          }
        })
      } else {
        docContent.value = `<pre>${escapeHtml(JSON.stringify(data || lastEvent, null, 2))}</pre>`
      }
      await nextTick()
      if (docInnerRef.value) {
        docInnerRef.value.scrollTop = 0
      }
      updateFades()

      // Fetch title from first file if available (only for multiple files)
      if (Array.isArray(files) && files.length > 1 && files.length > 0) {
        const firstFile = files[0] as Record<string, unknown>
        const url = typeof firstFile.url === 'string' ? firstFile.url : undefined
        if (url) {
          try {
            const response = await fetch(resolveBackendStaticUrl(String(url)))
            if (response.ok) {
              const content = await response.text()
              const title = extractTitleFromMarkdown(content)
              if (title) {
                section.title = title
              } else {
                section.title = section.repo
              }
            }
          } catch (e) {
            console.error('Failed to fetch title:', e)
            section.title = section.repo
          }
        }
      }
    }
  } catch (error) {
    if (controller.signal.aborted) {
      return
    }
    const message = error instanceof Error ? error.message : 'Failed to stream documentation.'
    progressLogs.value.push(message)
    renderProgress()
    console.error('Failed to stream documentation:', error)
  } finally {
    if (!controller.signal.aborted) {
      isStreaming.value = false
    }
    if (streamController.value === controller) {
      streamController.value = null
    }
    nextTick(() => updateFades())
  }
}

const selectRepoById = (id: string, needUpdate = false) => {
  if (!id) return
  let index = findSectionIndexById(id)
  let section = index >= 0 ? tocSections.value[index] : undefined

  if (!section) {
    const fallback = createSectionFromId(id)
    if (!fallback) return
    tocSections.value = [...tocSections.value, fallback]
    index = tocSections.value.length - 1
    section = fallback
  }

  selected.value = { section: index, item: null }
  void loadDocumentation(section, needUpdate)
}

let lastHandledRepoId: string | null = null

watch(
  () => route.params.repoId,
  (newRepoId) => {
    const id = typeof newRepoId === 'string' ? newRepoId : ''
    if (id === lastHandledRepoId) {
      repoId.value = id
      return
    }

    repoId.value = id

    if (!id) {
      progressLogs.value = []
      docContent.value = '<p>Select a repository to view documentation.</p>'
      lastHandledRepoId = id
      return
    }

    lastHandledRepoId = id
    selectRepoById(id)
  }
)

watch(docContent, () => {
  nextTick(() => updateFades())
})

onMounted(async () => {
  nextTick(() => {
    updateFades()
    const el = docInnerRef.value
    if (!el) return
    el.addEventListener('scroll', updateFades, { passive: true })
    window.addEventListener('resize', updateFades)
  })

  if (repoId.value && lastHandledRepoId !== repoId.value) {
    lastHandledRepoId = repoId.value
    selectRepoById(repoId.value)
  }
})

onUnmounted(() => {
  const el = docInnerRef.value
  if (el) el.removeEventListener('scroll', updateFades)
  window.removeEventListener('resize', updateFades)
  if (streamController.value) {
    streamController.value.abort()
    streamController.value = null
  }
})

const selectItem = async (si: number, ii: number) => {
  selected.value = { section: si, item: ii }
  const section = tocSections.value[si]
  if (!section || !section.items) return
  const item = section.items[ii]
  if (!item || !item.url) return

  try {
    const resolved = resolveBackendStaticUrl(item.url)
    progressLogs.value.push(`Loading file from ${resolved}`)
    renderProgress()
    const response = await fetch(resolved)
    if (response.ok) {
      const content = await response.text()
      // If this item is an intra-doc link (#id), scroll to it instead
      if (item.url.startsWith('#')) {
        const id = item.url.slice(1)
        await nextTick()
        const container = docInnerRef.value
        if (container) {
          const el = container.querySelector(`#${CSS.escape(id)}`)
          if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }
        }
        return
      }
      // Render markdown or plain text
      const isMd = typeof item.url === 'string' && /\.md$/i.test(item.url)
      if (isMd || content.trim().startsWith('#') || content.includes('\n# ')) {
        const { html, headings } = renderMarkdown(content)
        docContent.value = html
        // store headings on the file item so sidebar can show them
        ;(item as FileItem).headings = headings.map((h) => ({
          level: h.level,
          title: h.title,
          id: h.id,
        }))
      } else {
        docContent.value = `<pre>${escapeHtml(content)}</pre>`
      }
      await nextTick()
      if (docInnerRef.value) {
        docInnerRef.value.scrollTop = 0
      }
      updateFades()
    } else {
      docContent.value = `<p>Failed to load document: ${response.status}</p>`
    }
  } catch (error) {
    docContent.value = `<p>Error loading document: ${String(error)}</p>`
  }
}

const toggleExpand = (si: number, ii: number) => {
  const section = tocSections.value[si]
  if (!section || !section.items) return
  const item = section.items[ii]
  if (!item) return
  item.expanded = !item.expanded
}

const selectTitle = (si: number) => {
  selected.value = { section: si, item: null }
  const section = tocSections.value[si]
  if (!section) return
  if (route.params.repoId !== section.id) {
    router.replace({ name: 'RepoDetail', params: { repoId: section.id } })
    return
  }
  void loadDocumentation(section)
}

const emit = defineEmits<{
  (e: 'send', payload: string): void
  (e: 'navigateNewRepo'): void
}>()

const handleSend = () => {
  if (!query.value) return
  emit('send', query.value)
  query.value = ''
}
</script>

<style scoped>
.repo-detail {
  padding: 20px;
  box-sizing: border-box;
  height: 100vh;
  overflow: hidden;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
  background: var(--container-bg);
  color: var(--text-color);
}

.content-wrapper {
  display: flex;
  gap: 24px;
  align-items: stretch;
  flex: 1 1 auto;
  overflow: hidden;
}

.toc {
  min-width: 100px;
  padding: 24px 24px;
  margin-bottom: 100px;
  overflow: auto;
  flex: 0 0 200px;
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

.doc {
  flex: 1 1 auto;
  border-left: 1px solid var(--border-color);
  min-height: 0;
  margin: 20px 100px 100px 0;
  position: relative;
}

.doc-inner {
  /* make the inner doc area the only scrollable region */
  height: 100%;
  overflow-y: auto;
  overflow-x: auto;
  padding: 0 40px;
  background: transparent;
  line-height: 1.7;
  /* preserve newlines inside v-html content and allow long words to wrap */
  white-space: pre-wrap;
  overflow-wrap: break-word;
  word-break: break-word;
}

.stream-log {
  font-family: var(--font-mono, monospace);
  font-size: 14px;
  line-height: 1.5;
  color: var(--text-color);
}

.stream-log p {
  margin: 0 0 8px;
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

.ask-row {
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  bottom: 24px;
  width: min(960px, 90%);
  display: flex;
  align-items: center;
  gap: 16px;
  z-index: 1200;
}

.ask-box {
  flex: 1 1 auto;
  display: flex;
  min-height: 80px;
  align-items: flex-start;
  background: var(--card-bg);
  border: 2px solid var(--border-color);
  border-radius: 20px;
  padding: 12px 18px;
  box-shadow: 0 2px 8px var(--shadow-color);
}

.ask-input {
  flex: 1 1 auto;
  border: none;
  outline: none;
  background: transparent;
  font-family: inherit;
  font-size: 16px;
  color: var(--text-color);
  max-height: 200px;
  resize: none;
  overflow-y: auto;
}

.send-btn {
  background: transparent;
  border: none;
  color: var(--text-color);
  cursor: pointer;
  font-weight: 600;
  align-self: center;
}

.new-repo {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: var(--card-bg);
  border: 2px solid var(--border-color);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 2px 8px var(--shadow-color);
}

.new-repo .plus {
  font-size: 24px;
  line-height: 1;
}
</style>
