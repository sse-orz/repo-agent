<template>
  <div class="repo-detail">
    <ThemeToggle />
    <HistoryButton />

    <div v-if="currentRepoName" class="repo-header">
      <div class="repo-link" @click="openRepoInNewTab">
        <i :class="repoPlatform === 'github' ? 'fab fa-github' : 'fab fa-git-alt'"></i>
        <span class="repo-link-text">{{ currentRepoName }}</span>
        <i class="fas fa-external-link-alt small-icon"></i>
      </div>
    </div>
    <div class="content-wrapper">
      <aside class="toc">
        <nav>
          <div v-for="(section, si) in tocSections" :key="`sec-${si}`" class="toc-section">
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
                  @click.stop="toggleItemExpand(item)"
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

      <main class="doc">
        <div class="doc-inner" ref="docInnerRef">
          <ProgressBar
            v-if="isStreaming && progressLogs.length > 0"
            :progress="currentProgress"
            :logs="progressLogs"
          />
          <div v-html="docContent"></div>
        </div>
        <div v-if="showTop" class="fade fade-top" aria-hidden="true"></div>
        <div v-if="showBottom" class="fade fade-bottom" aria-hidden="true"></div>
      </main>
    </div>

    <div class="ask-row">
      <button class="new-repo" @click="router.push('/')" aria-label="New repo">
        <span class="plus">+</span>
      </button>

      <div class="ask-box">
        <div class="ask-icon-wrapper">
          <i class="fas fa-comment-dots"></i>
        </div>
        <input
          v-model="query"
          class="ask-input"
          type="text"
          :placeholder="placeholder"
          @keyup.enter="handleSend"
        />
        <button class="send-btn" @click="handleSend" title="Send question">
          <i class="fas fa-arrow-right"></i>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, watch, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import ThemeToggle from '../components/ThemeToggle.vue'
import HistoryButton from '../components/HistoryButton.vue'
import ProgressBar from '../components/ProgressBar.vue'
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
const selectedUrl = ref<string | null>(null)
const isStreaming = ref(false)
const progressLogs = ref<string[]>([])
const streamController = ref<AbortController | null>(null)
const currentProgress = ref<number>(0)

const repoPlatform = computed(() => {
  const platform = route.query.platform
  if (typeof platform === 'string') return platform
  return 'github'
})

const generationMode = computed<'sub' | 'moe'>(() => {
  const mode = route.query.mode
  if (mode === 'sub' || mode === 'moe') return mode
  return 'sub'
})

const owner = computed(() => {
  if (!repoId.value) return ''
  const parts = String(repoId.value).split('_')
  return parts[0] || ''
})

const repoName = computed(() => {
  if (!repoId.value) return ''
  const parts = String(repoId.value).split('_')
  return parts[1] || ''
})

const currentRepoName = computed(() => {
  if (!owner.value || !repoName.value) return ''
  return `${owner.value} / ${repoName.value}`
})

const externalRepoUrl = computed(() => {
  if (!owner.value || !repoName.value) return ''
  if (repoPlatform.value === 'gitee') {
    return `https://gitee.com/${owner.value}/${repoName.value}`
  }
  // 默认 GitHub
  return `https://github.com/${owner.value}/${repoName.value}`
})

const buildTocItems = (
  files: unknown[],
  options: { resolvedUrlWithHeadings?: { url: string; headings?: HeadingItem[] } } = {}
): FileItem[] => {
  const rootChildren: FileItem[] = []

  const ensureDir = (
    children: FileItem[],
    name: string,
    depth: number,
    parentPath: string
  ): FileItem => {
    let node = children.find((it) => it.isDir && it.name === name)
    if (!node) {
      node = {
        name,
        isDir: true,
        depth,
        expanded: true,
        children: [],
        fullPath: parentPath ? `${parentPath}/${name}` : name,
      }
      children.push(node)
    }
    return node
  }

  for (const f of files as Record<string, unknown>[]) {
    const rec = f || {}
    const rawPath = String(rec.path ?? rec.name ?? rec.url ?? '').trim()
    const rawUrl = String(rec.url ?? '').trim()
    if (!rawPath && !rawUrl) continue

    // 只保留 markdown 文件
    const isMd = /\.md$/i.test(rawPath || rawUrl)
    if (!isMd) continue

    const segments = (rawPath || rawUrl).split('/').filter(Boolean)
    if (!segments.length) continue

    let children = rootChildren
    let depth = 0
    let parentPath = ''

    // 中间段作为目录节点
    for (let i = 0; i < segments.length - 1; i++) {
      const seg = segments[i]
      if (!seg) continue
      const dir = ensureDir(children, seg, depth, parentPath || '')
      children = dir.children || (dir.children = [])
      parentPath = dir.fullPath ?? seg
      depth += 1
    }

    const baseName = segments[segments.length - 1] || rawPath || rawUrl
    // 统一处理形如 "xxxxxx.ext_doc.md" 的文件名，只显示 "xxxxxx.ext"
    let displayName = baseName.replace(/\.md$/i, '')
    displayName = displayName.replace(/_doc$/i, '')
    const url = resolveBackendStaticUrl(rawUrl || rawPath)

    const item: FileItem = {
      name: displayName,
      url,
      depth,
      isDir: false,
      expanded: false,
      fullPath: rawPath || rawUrl,
    }

    children.push(item)
  }

  // 附加标题信息到指定 URL 的文件节点
  if (options.resolvedUrlWithHeadings) {
    const { url, headings } = options.resolvedUrlWithHeadings
    const attach = (nodes: FileItem[]) => {
      for (const node of nodes) {
        if (!node.isDir && node.url === url && headings?.length) {
          node.headings = headings.map((h) => ({
            level: h.level,
            title: h.title,
            id: h.id,
          }))
          return true
        }
        if (node.children && node.children.length && attach(node.children)) return true
      }
      return false
    }
    attach(rootChildren)
  }

  return rootChildren
}

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

const displayItems = (section: TocSection) => flattenItems(section.items)

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

// 进度条现在由 ProgressBar 组件处理，无需手动渲染

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
  currentProgress.value = 5
  progressLogs.value = ['Connecting to documentation stream...']
  docContent.value = ''

  let loadedFromStream = false

  try {
    const lastEvent = await generateDocStream(
      {
        mode: generationMode.value,
        request: {
          owner: section.owner,
          repo: section.repo,
          platform: repoPlatform.value,
          need_update: needUpdate,
        },
      },
      async (event: BaseResponse<unknown>) => {
        if (!event) return

        if (event.code !== 200) {
          const message = event.message || 'Unexpected response from stream.'
          progressLogs.value.push(message)
          const errorDetail = (event.data as StreamData | undefined)?.error
          if (errorDetail) {
            progressLogs.value.push(errorDetail)
          }
          return
        }

        const data = event.data as StreamData

        if (data?.stage) {
          const message = data.message || event.message || `Processing ${data.stage}`
          progressLogs.value.push(message)
          // 更新进度值
          if (typeof data.progress === 'number') {
            currentProgress.value = data.progress
          }
          await nextTick()
          if (docInnerRef.value) {
            docInnerRef.value.scrollTop = docInnerRef.value.scrollHeight
          }
          return
        }

        if (data?.error) {
          progressLogs.value.push(data.error)
          return
        }

        if (data?.wiki_url) {
          currentProgress.value = 100
          progressLogs.value.push('Documentation ready.')
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
            // Populate TOC with the single file (only markdown, with optional headings)
            section.items = buildTocItems(files as unknown[], {
              resolvedUrlWithHeadings: resolvedUrl
                ? {
                    url: resolvedUrl,
                    headings: fileHeadings,
                  }
                : undefined,
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
            // Update TOC with markdown files only
            section.items = buildTocItems(files as unknown[])
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
              // populate TOC linking and attach headings if present (markdown only)
              section.items = buildTocItems(files as unknown[], {
                resolvedUrlWithHeadings: resolvedUrl
                  ? { url: resolvedUrl, headings: fileHeadings }
                  : undefined,
              })
            } else {
              docContent.value = `<p>Failed to load document: ${response.status}</p>`
            }
          } catch (error) {
            docContent.value = `<p>Error loading document: ${String(error)}</p>`
          }
        }
        // Populate TOC with markdown files only
        section.items = buildTocItems(files as unknown[])
      } else if (Array.isArray(files) && files.length > 1) {
        docContent.value = `<h3>Generated Files (${files.length})</h3><ul>${(files as unknown[])
          .map((f) => {
            const rec = f as Record<string, unknown>
            return `<li>${escapeHtml(String(rec.path ?? rec.name ?? rec.url ?? 'file'))}</li>`
          })
          .join('')}</ul>`
        // Update TOC with markdown files only
        section.items = buildTocItems(files as unknown[])
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
      currentProgress.value = 0
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

const selectItemByNode = async (item: FileItem) => {
  if (!item || item.isDir || !item.url) return
  selectedUrl.value = item.url

  try {
    const resolved = resolveBackendStaticUrl(item.url)
    progressLogs.value.push(`Loading file from ${resolved}`)
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

const toggleItemExpand = (item: FileItem) => {
  item.expanded = !item.expanded
}

const handleHeadingClick = async (item: FileItem, headingId: string) => {
  if (item.isDir || !headingId) return
  const needSwitchFile = item.url && item.url !== selectedUrl.value
  if (needSwitchFile) {
    await selectItemByNode(item)
  }
  await scrollToHeading(headingId)
}

const handleItemClick = (item: FileItem) => {
  if (item.isDir) {
    toggleItemExpand(item)
  } else {
    void selectItemByNode(item)
  }
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

const openRepoInNewTab = () => {
  if (!externalRepoUrl.value) return
  window.open(externalRepoUrl.value, '_blank', 'noopener')
}
</script>

<style scoped>
.repo-detail {
  padding: 72px 20px 20px;
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
  border-top: 1px solid var(--border-color);
  padding-top: 16px;
  margin-top: 8px;
  /* 固定整体内容区域宽度并居中 */
  max-width: 1340px;
  margin-left: auto;
  margin-right: auto;
}

.repo-header {
  position: fixed;
  top: 20px;
  left: 60px;
  z-index: 950;
}

.repo-link {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--text-color);
  cursor: pointer;
  font-size: 16px;
  font-weight: 600;
  transition: color 0.2s ease;
  height: 40px;
  line-height: 40px;
}

.repo-link:hover {
  color: var(--title-color);
}

.repo-link-text {
  max-width: 320px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.small-icon {
  font-size: 13px;
}

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
}

/* 限制 Markdown 实际内容宽度并居中显示 */
.doc-inner > * {
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}

/* 收紧 markdown 元素的上下间距，使版面更紧凑 */
.doc-inner h1,
.doc-inner h2,
.doc-inner h3,
.doc-inner h4,
.doc-inner h5,
.doc-inner h6 {
  margin-top: 1.2em;
  margin-bottom: 0.4em;
}

.doc-inner p {
  margin-top: 0.15em;
  margin-bottom: 0.35em;
}

.doc-inner ul,
.doc-inner ol {
  margin-top: 0.25em;
  margin-bottom: 0.5em;
  padding-left: 1.5em;
}

.doc-inner li {
  margin-top: 0.05em;
  margin-bottom: 0.05em;
}

/* 隐藏正文区域滚动条但保持可滚动 */
.doc-inner {
  scrollbar-width: none; /* Firefox */
}
.doc-inner::-webkit-scrollbar {
  width: 0;
  height: 0;
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
  width: min(720px, 90%);
  display: flex;
  align-items: center;
  gap: 10px;
  z-index: 1200;
}

.ask-box {
  flex: 1 1 auto;
  display: flex;
  align-items: center;
  background: var(--input-bg, #ffffff);
  border: 1px solid var(--border-color, #e8e8e8);
  border-radius: 12px;
  padding: 0 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
  height: 44px;
}

.ask-box:focus-within {
  border-color: var(--border-color);
  box-shadow: 0 4px 16px var(--focus-color);
}

.ask-icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 10px;
  color: var(--icon-color, #999);
  font-size: 16px;
  flex-shrink: 0;
}

.ask-input {
  flex: 1 1 auto;
  border: none;
  outline: none;
  background: transparent;
  font-family: inherit;
  font-size: 14px;
  color: var(--text-color);
  padding: 0 6px;
  height: 100%;
  line-height: 44px;
}

.ask-input::placeholder {
  color: var(--placeholder-color, #bbb);
}

.send-btn {
  padding: 10px 14px;
  background: transparent;
  border: none;
  border-left: 1px solid var(--border-color, #f0f0f0);
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--secondary-text, #666);
}

.send-btn i {
  font-size: 18px;
}

.send-btn:hover {
  color: var(--text-color);
  background: var(--hover-bg);
}

.new-repo {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--card-bg);
  border: 2px solid var(--border-color);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 2px 8px var(--shadow-color);
  transition: all 0.2s ease;
  color: var(--text-color);
}

.new-repo:hover {
  background: var(--hover-bg);
  border-color: var(--border-color);
  transform: scale(1.05);
}

.new-repo .plus {
  font-size: 20px;
  line-height: 1;
  color: var(--text-color);
}
</style>
