<template>
  <div class="repo-detail">
    <TopControls />

    <RepoHeader
      v-if="currentRepoName"
      :repo-name="currentRepoName"
      :platform="repoPlatform"
      @click="openRepoInNewTab"
    />

    <div class="content-wrapper">
      <TocSidebar
        :sections="tocSections"
        :selected-url="selectedUrl"
        @item-click="handleItemClick"
        @heading-click="handleHeadingClick"
        @toggle="toggleItemExpand"
      />

      <DocContent
        ref="docContentRef"
        :content="docContent"
        :is-streaming="isStreaming"
        :progress-logs="progressLogs"
        :progress="currentProgress"
      />
    </div>

    <AskBox
      v-model="query"
      :placeholder="placeholder"
      @send="handleSend"
      @new-repo="handleNewRepo"
    />
    <div
      v-if="zoomModalVisible"
      class="zoom-modal"
      @click.self="closeZoomModal"
      @mousemove="onDrag"
      @mouseup="stopDrag"
      @mouseleave="stopDrag"
      @keydown.esc="closeZoomModal"
      tabindex="0"
    >
      <div
        class="zoom-content"
        :style="zoomStyle"
        @wheel.prevent="handleZoom"
        @mousedown="startDrag"
      >
        <div v-html="zoomSvgHtml"></div>
      </div>
      <button class="close-zoom" @click="closeZoomModal" aria-label="Close zoom modal">
        <i class="fas fa-times"></i>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, watch, computed, inject, type Ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import TopControls from '../components/TopControls.vue'
import RepoHeader from '../components/RepoHeader.vue'
import TocSidebar from '../components/TocSidebar.vue'
import DocContent from '../components/DocContent.vue'
import AskBox from '../components/AskBox.vue'
import { generateDocStream, resolveBackendStaticUrl, type BaseResponse } from '../utils/request'
import MarkdownIt from 'markdown-it'
import anchor from 'markdown-it-anchor'
import mermaid from 'mermaid'

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
const docContentRef = ref<InstanceType<typeof DocContent> | null>(null)
const docContent = ref('<p>Select a repository to view documentation.</p>')
const tocSections = ref<TocSection[]>([])
const query = ref('')
const selected = ref<{ section: number | null; item: number | null }>({ section: null, item: null })
const selectedUrl = ref<string | null>(null)
const isStreaming = ref(false)
const progressLogs = ref<string[]>([])
const streamController = ref<AbortController | null>(null)
const currentProgress = ref<number>(0)
const themeContext = inject<{ isDarkMode: Ref<boolean> } | null>('theme', null)

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

    const isMd = /\.md$/i.test(rawPath || rawUrl)
    if (!isMd) continue

    const segments = (rawPath || rawUrl).split('/').filter(Boolean)
    if (!segments.length) continue

    let children = rootChildren
    let depth = 0
    let parentPath = ''

    for (let i = 0; i < segments.length - 1; i++) {
      const seg = segments[i]
      if (!seg) continue
      const dir = ensureDir(children, seg, depth, parentPath || '')
      children = dir.children || (dir.children = [])
      parentPath = dir.fullPath ?? seg
      depth += 1
    }

    const baseName = segments[segments.length - 1] || rawPath || rawUrl
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

type FenceRule = Exclude<typeof md.renderer.rules.fence, undefined>
type FenceParams = Parameters<FenceRule>

const builtInFence = md.renderer.rules.fence?.bind(md.renderer.rules)
const defaultFence: FenceRule = builtInFence
  ? (builtInFence as FenceRule)
  : (((
      tokens: FenceParams[0],
      idx: FenceParams[1],
      options: FenceParams[2],
      env: FenceParams[3],
      self: FenceParams[4]
    ) => self.renderToken(tokens, idx, options)) as FenceRule)

md.renderer.rules.fence = (
  tokens: FenceParams[0],
  idx: FenceParams[1],
  options: FenceParams[2],
  env: FenceParams[3],
  self: FenceParams[4]
) => {
  const token = tokens[idx]
  if (!token) {
    return defaultFence(tokens, idx, options, env, self)
  }
  const info = (token.info || '').trim()
  if (info === 'mermaid') {
    const content = token.content || ''
    return `<div class="mermaid-container"><div class="mermaid">${escapeHtml(content)}</div><button class="mermaid-zoom-btn" title="Zoom" aria-label="Zoom diagram"><i class="fas fa-search-plus"></i></button></div>`
  }
  return defaultFence(tokens, idx, options, env, self)
}

const renderMarkdown = (markdown: string) => {
  const env: Record<string, unknown> = {}
  const html = md.render(markdown, env)

  // parse tokens to extract headings and their ids
  const tokens = md.parse(markdown, env)
  const headings: { level: number; title: string; id: string }[] = []
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]
    if (!t) continue
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

const getMermaidThemeVariables = () => {
  if (typeof window === 'undefined') {
    return {
      background: '#bbb',
      primaryColor: '#ffffff',
      primaryBorderColor: '#d1d5db',
      primaryTextColor: '#111827',
      secondaryColor: '#ffffff',
      secondaryBorderColor: '#d1d5db',
      secondaryTextColor: '#111827',
      tertiaryColor: '#ffffff',
      tertiaryBorderColor: '#d1d5db',
      tertiaryTextColor: '#111827',
      lineColor: '#9ca3af',
      textColor: '#111827',
      edgeLabelBackground: '#ffffff',
    }
  }

  const styles = getComputedStyle(document.documentElement)
  const pick = (name: string, fallback: string) => {
    const value = styles.getPropertyValue(name).trim()
    return value || fallback
  }

  return {
    background: pick('--placeholder-color', '#bbb'),
    primaryColor: pick('--card-bg', '#ffffff'),
    primaryBorderColor: pick('--border-color', '#d1d5db'),
    primaryTextColor: pick('--text-color', '#111827'),
    secondaryColor: pick('--card-bg', '#ffffff'),
    secondaryBorderColor: pick('--border-color', '#d1d5db'),
    secondaryTextColor: pick('--text-color', '#111827'),
    tertiaryColor: pick('--card-bg', '#ffffff'),
    tertiaryBorderColor: pick('--border-color', '#d1d5db'),
    tertiaryTextColor: pick('--text-color', '#111827'),
    lineColor: pick('--border-color', '#9ca3af'),
    textColor: pick('--text-color', '#111827'),
    edgeLabelBackground: pick('--card-bg', '#ffffff'),
  }
}

const ensureMermaid = () => {
  if (typeof window === 'undefined') return false
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: 'base',
    themeVariables: getMermaidThemeVariables(),
  })
  return true
}

const zoomModalVisible = ref(false)
const zoomSvgHtml = ref('')
const zoomScale = ref(1)
const zoomTranslate = ref({ x: 0, y: 0 })
const isDragging = ref(false)
const dragStart = ref({ x: 0, y: 0 })
const lastTranslate = ref({ x: 0, y: 0 })

const openZoomModal = (html: string) => {
  zoomSvgHtml.value = html
  zoomModalVisible.value = true
  zoomScale.value = 1
  zoomTranslate.value = { x: 0, y: 0 }
  lastTranslate.value = { x: 0, y: 0 }
  document.body.style.overflow = 'hidden'
}

const closeZoomModal = () => {
  zoomModalVisible.value = false
  document.body.style.overflow = ''
}

const handleZoom = (e: WheelEvent) => {
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  zoomScale.value = Math.max(0.1, Math.min(5, zoomScale.value * delta))
}

const startDrag = (e: MouseEvent) => {
  isDragging.value = true
  dragStart.value = { x: e.clientX, y: e.clientY }
  lastTranslate.value = { ...zoomTranslate.value }
}

const onDrag = (e: MouseEvent) => {
  if (!isDragging.value) return
  const dx = e.clientX - dragStart.value.x
  const dy = e.clientY - dragStart.value.y
  zoomTranslate.value = {
    x: lastTranslate.value.x + dx,
    y: lastTranslate.value.y + dy,
  }
}

const stopDrag = () => {
  isDragging.value = false
}

const zoomStyle = computed(() => ({
  transform: `translate(${zoomTranslate.value.x}px, ${zoomTranslate.value.y}px) scale(${zoomScale.value})`,
  transformOrigin: 'center center',
  cursor: isDragging.value ? 'grabbing' : 'grab',
}))

const attachZoomListeners = () => {
  if (!docContentRef.value) return
  const container = docContentRef.value.getElement()
  if (!container) return
  const btns = container.querySelectorAll('.mermaid-zoom-btn')
  btns.forEach((btn) => {
    const newBtn = btn.cloneNode(true)
    if (btn.parentNode) {
      btn.parentNode.replaceChild(newBtn, btn)
      newBtn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement
        const container = target.closest('.mermaid-container')
        const mermaidDiv = container?.querySelector('.mermaid')
        if (mermaidDiv) {
          openZoomModal(mermaidDiv.innerHTML)
        }
      })
    }
  })
}

const renderMermaidDiagrams = async () => {
  if (!docContentRef.value) return
  await nextTick()
  const container = docContentRef.value.getElement()
  if (!container) return
  const nodes = container.querySelectorAll<HTMLElement>('.mermaid')
  if (!nodes.length || !ensureMermaid()) return
  try {
    await mermaid.run({ nodes })
    attachZoomListeners()
  } catch (error) {
    console.warn('Failed to render mermaid diagrams', error)
  }
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
  if (docContentRef.value) {
    await docContentRef.value.scrollToHeading(id)
  }
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
          if (typeof data.progress === 'number') {
            currentProgress.value = data.progress
          }
          await nextTick()
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
            const index = findSectionIndexById(section.id)
            if (index >= 0 && tocSections.value[index]) {
              tocSections.value[index].items = buildTocItems(files as unknown[], {
                resolvedUrlWithHeadings: resolvedUrl
                  ? {
                      url: resolvedUrl,
                      headings: fileHeadings,
                    }
                  : undefined,
              })
            }
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
            const index = findSectionIndexById(section.id)
            if (index >= 0 && tocSections.value[index]) {
              tocSections.value[index].items = buildTocItems(files as unknown[])
            }
          } else {
            docContent.value = `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`
          }
          loadedFromStream = true
          await nextTick()
          if (docContentRef.value) {
            docContentRef.value.scrollToTop()
          }

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
      if (docContentRef.value) {
        docContentRef.value.scrollToTop()
      }

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
    section = tocSections.value[index]
  }

  if (!section) return

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
  nextTick(() => {
    void renderMermaidDiagrams()
  })
})

if (themeContext?.isDarkMode) {
  watch(
    () => themeContext.isDarkMode.value,
    () => {
      // Re-render diagrams so mermaid picks up the latest CSS variable values per theme
      void renderMermaidDiagrams()
    }
  )
}

onMounted(async () => {
  if (repoId.value && lastHandledRepoId !== repoId.value) {
    lastHandledRepoId = repoId.value
    selectRepoById(repoId.value)
  }
})

onUnmounted(() => {
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
        await scrollToHeading(id)
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
      if (docContentRef.value) {
        docContentRef.value.scrollToTop()
      }
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

const handleNewRepo = () => {
  router.push('/')
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
  max-width: 1340px;
  margin-left: auto;
  margin-right: auto;
}

.zoom-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.8);
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.zoom-content {
  display: inline-block;
  transition: transform 0.1s ease-out;
}

.zoom-content :deep(svg) {
  max-width: none;
  height: auto;
  background: var(--placeholder-color, #bbb);
  padding: 20px;
  border-radius: 12px;
}

.zoom-content :deep(.node rect),
.zoom-content :deep(.node circle),
.zoom-content :deep(.node polygon),
.zoom-content :deep(.cluster rect),
.zoom-content :deep(.label-container),
.zoom-content :deep(.actor) {
  fill: var(--card-bg, #ffffff) !important;
  stroke: var(--border-color, #d1d5db) !important;
}

.zoom-content :deep(text),
.zoom-content :deep(span) {
  fill: var(--text-color, #111827) !important;
  color: var(--text-color, #111827) !important;
}

.close-zoom {
  position: absolute;
  top: 20px;
  right: 20px;
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  color: white;
  font-size: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.close-zoom:hover {
  background: rgba(255, 255, 255, 0.4);
}
</style>
