<template>
  <div class="large-chat">
    <div class="chat-container">
      <div class="messages" ref="messagesEl">
        <div
          v-for="(m, idx) in messages"
          :key="idx"
          :class="['message-row', m.role === 'user' ? 'user' : 'assistant']"
        >
          <template v-if="m.role === 'assistant' && idx === lastAssistantIndex && isStreaming">
            <div class="progress-row">
              <div class="spinner" aria-hidden="true"></div>
              <div class="node-text">{{ currentNode ?? 'Initializing' }}</div>
            </div>
          </template>

          <div class="bubble">
            <div
              v-if="m.role === 'assistant'"
              class="bubble-text"
              v-html="(!m.text || !m.text.trim()) ? 'Loading...' : md.render(m.text)"
            ></div>
            <div
              v-else
              class="bubble-text"
            >{{ m.text }}</div>
          </div>
          <div
            v-if="m.role === 'assistant' && canRetry && idx === lastAssistantIndex"
            class="retry-row"
          >
            <button class="retry-btn" @click="retryStream" title="Retry">
              <i class="fas fa-redo"></i>
            </button>
            <span class="retry-text">Retry</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onUnmounted, onMounted, computed, watch } from 'vue'
import { askRagStream } from '../utils/request'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'

const md = new MarkdownIt({
  highlight: function (str, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(str, { language: lang }).value
      } catch (__) {
        return md.utils.escapeHtml(str)
      }
    }
    return md.utils.escapeHtml(str)
  },
})

const props = withDefaults(
  defineProps<{
    placeholder?: string
    owner?: string
    repo?: string
    platform?: string
    mode?: 'fast' | 'smart'
  }>(),
  { placeholder: 'Try to ask me...', platform: 'github', mode: 'fast' }
)
const emit = defineEmits<{
  (e: 'send', text: string): void
  (e: 'new-repo'): void
  (e: 'streaming', v: boolean): void
}>()

type Msg = { role: 'user' | 'assistant'; text: string }

const messages = ref<Msg[]>([
  { role: 'assistant', text: 'Hello, is there anything I can help with? ' },
])

// Storage key derived from owner/repo so messages persist while RepoDetail stays mounted
const storageKey = computed(() => {
  if (!props.owner || !props.repo) return null
  return `largechat:${props.owner}:${props.repo}`
})

// Restore messages from sessionStorage when component mounts (so toggling v-if preserves history)
onMounted(() => {
  try {
    const key = storageKey.value
    if (key) {
      const raw = sessionStorage.getItem(key)
      if (raw) {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed) && parsed.length > 0) {
          messages.value = parsed
        }
      }
    }
  } catch (e) {
    console.warn('[LargeChat] restore messages failed', e)
  }
})

// Persist messages to sessionStorage whenever they change
watch(
  messages,
  (v) => {
    try {
      const key = storageKey.value
      if (key) {
        sessionStorage.setItem(key, JSON.stringify(v))
      }
    } catch (e) {
      console.warn('[LargeChat] persist messages failed', e)
    }
  },
  { deep: true }
)
const input = ref('')
const messagesEl = ref<HTMLElement | null>(null)
let abortController: AbortController | null = null
const currentNode = ref<string | null>(null)
const isStreaming = ref(false)
const lastUserMessage = ref<string>('')
const canRetry = ref(false)

const lastAssistantIndex = computed(() => {
  for (let i = messages.value.length - 1; i >= 0; i--) {
    const msg = messages.value[i]
    if (!msg) continue
    if (msg.role === 'assistant') return i
  }
  return -1
})

const scrollToBottom = async () => {
  await nextTick()
  const el = messagesEl.value
  if (el) el.scrollTop = el.scrollHeight
}

const appendAssistantPlaceholder = () => {
  messages.value.push({ role: 'assistant', text: 'Loading...' })
}

const updateLastAssistant = (text: string) => {
  for (let i = messages.value.length - 1; i >= 0; i--) {
    const msg = messages.value[i]
    if (!msg) continue
    if (msg.role === 'assistant') {
      // Avoid overwriting placeholder with empty/whitespace text from stream
      if (text && text.trim()) {
        msg.text = text
      }
      return
    }
  }
  // no assistant found, push new
  messages.value.push({ role: 'assistant', text: text && text.trim() ? text : 'Loading...' })
}

onUnmounted(() => {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
  // persist final state on unmount (watch typically already did)
  try {
    const key = storageKey.value
    if (key) {
      sessionStorage.setItem(key, JSON.stringify(messages.value))
    }
  } catch (e) {
    console.warn('[LargeChat] save on unmount failed', e)
  }
})

const streamRagAnswer = async (question: string) => {
  if (!props.owner || !props.repo) {
    updateLastAssistant('Owner/Repo required. Please set them and try again.')
    return
  }

  if (abortController) {
    abortController.abort()
  }
  abortController = new AbortController()
  isStreaming.value = true
  emit('streaming', true)

  try {
    await askRagStream(
      {
        owner: props.owner,
        repo: props.repo,
        platform: props.platform || 'github',
        mode: props.mode || 'fast',
        question,
      },
      async (event) => {
        try {
          const d = event.data ?? event

          // 如果 data 中包含 node 字段，作为当前进度显示
          let node: unknown = null
          if (d && typeof d === 'object') {
            const obj = d as unknown as Record<string, unknown>
            if ('node' in obj) node = obj.node ?? null
            else if ('data' in obj && obj.data && typeof obj.data === 'object') {
              node = (obj.data as Record<string, unknown>).node ?? null
            }
          }
          if (node) {
            currentNode.value = String(node)
          }

          // prefer answer field
          let ans = ''
          if (d && typeof d === 'object') {
            const obj = d as unknown as Record<string, unknown>
            const candidate = 'answer' in obj ? obj.answer : 'text' in obj ? obj.text : ''
            ans = candidate === undefined || candidate === null ? '' : String(candidate)
          } else {
            ans = String(d || '')
          }

          if (ans !== undefined && ans !== null) {
            updateLastAssistant(String(ans))
            scrollToBottom()
          }
        } catch (err) {
          // fallback to message
          try {
            // event may be a MessageEvent with message property
            // access safely without using `any`
            const maybe = event as unknown as Record<string, unknown>
            const msg = 'message' in maybe ? maybe.message : JSON.stringify(event)
            updateLastAssistant(String(msg))
          } catch (e) {
            updateLastAssistant(String(event))
          }
          scrollToBottom()
        }
      },
      { signal: abortController.signal }
    )
  } catch (err) {
    const errName =
      err && typeof err === 'object' && 'name' in err ? (err as { name?: string }).name : undefined
    if (errName === 'AbortError') {
      updateLastAssistant('Request aborted.')
      currentNode.value = null
      isStreaming.value = false
      canRetry.value = true
      emit('streaming', false)
    } else {
      updateLastAssistant(`Request error: ${(err as Error).message}`)
      currentNode.value = null
      isStreaming.value = false
      canRetry.value = true
      emit('streaming', false)
    }
  } finally {
    abortController = null
    // Clean up progress after stream ends
    currentNode.value = null
    isStreaming.value = false
    emit('streaming', false)
  }
}

// Expose a method to allow parent to programmatically send a message
const receiveMessage = (text: string) => {
  if (!text) return
  console.debug('[LargeChat] receiveMessage:', text)
  messages.value.push({ role: 'user', text })
  lastUserMessage.value = text
  appendAssistantPlaceholder()
  // mark streaming active so spinner appears immediately; clear previous node
  currentNode.value = null
  isStreaming.value = true
  canRetry.value = false
  emit('streaming', true)
  scrollToBottom()
  void streamRagAnswer(text)
}

const abortStream = () => {
  if (abortController) {
    abortController.abort()
    // ensure parent knows streaming stopped
    emit('streaming', false)
    // mark that retry is available and update assistant message so UI shows retry row
    canRetry.value = true
    updateLastAssistant('Request aborted.')
  }
}

const retryStream = () => {
  if (!lastUserMessage.value) return
  appendAssistantPlaceholder()
  currentNode.value = null
  isStreaming.value = true
  emit('streaming', true)
  canRetry.value = false
  scrollToBottom()
  void streamRagAnswer(lastUserMessage.value)
}

defineExpose({ receiveMessage, abortStream })
</script>

<style scoped>
.large-chat {
  position: relative;
  /* Use viewport width so the chat occupies ~80% of the page horizontally */
  width: 80vw;
  max-width: 1200px;
  margin: 0 auto;
  height: calc(100vh - 120px);
  display: flex;
  flex-direction: column;
  z-index: 1100;
  padding-bottom: 96px; /* leave space for page AskBox */
}
.chat-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
  /* let this flex item grow/shrink and allow children to scroll (min-height:0 needed for overflow in flex) */
  flex: 1 1 auto;
  min-height: 0;
}
.messages {
  flex: 1 1 auto;
  overflow: auto;
  padding: 20px 24px;
  background: transparent;
  /* ensure this element can shrink below its content so overflow works inside flex layout */
  min-height: 0;
}
.message-row {
  display: flex;
  margin: 8px 0;
}
.message-row.assistant {
  justify-content: flex-start;
  /* Stack progress indicator above the bubble */
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}
.message-row.user {
  justify-content: flex-end;
}
.bubble {
  /* allow bubbles to take more of the wider chat area */
  max-width: 85%;
  padding: 12px 16px;
  border-radius: 14px;
  box-shadow: 0 6px 14px rgba(0, 0, 0, 0.06);
}
.message-row.assistant .bubble {
  background: var(--card-bg);
  color: var(--text-color);
  border-top-left-radius: 6px;
  border-top-right-radius: 14px;
  border-bottom-right-radius: 14px;
  border-bottom-left-radius: 14px;
}
.message-row.user .bubble {
  background: var(--subcard-bg);
  color: var(--text-color);
  border-top-right-radius: 6px;
}
.bubble-text {
  white-space: normal;
  overflow-wrap: break-word;
  word-break: break-word;
  line-height: 1.3;
  font-family: 'Myriad', 'Noto Serif SC', serif !important;
}

.bubble-text :deep(*):not(code):not(pre):not(.mermaid) {
  font-family: 'Myriad', 'Noto Serif SC', serif !important;
}

.bubble-text :deep(*) {
  max-width: 100%;
}

/* 收紧 markdown 元素的上下间距，使版面更紧凑 */
.bubble-text :deep(h1),
.bubble-text :deep(h2),
.bubble-text :deep(h3),
.bubble-text :deep(h4),
.bubble-text :deep(h5),
.bubble-text :deep(h6) {
  margin-top: 1.1em;
  margin-bottom: 0.45em;
  color: var(--title-color);
  font-weight: 600;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 4px;
  text-align: left;
  letter-spacing: 0.4px;
}

.bubble-text :deep(p) {
  margin-top: 0.15em;
  margin-bottom: 0.35em;
}

.bubble-text :deep(ul),
.bubble-text :deep(ol) {
  margin-top: 0.25em;
  margin-bottom: 0.5em;
  padding-left: 1.5em;
}

.bubble-text :deep(li) {
  margin-top: 0.05em;
  margin-bottom: 0.05em;
}

.bubble-text :deep(pre) {
  background: var(--hover-bg);
  padding: 16px;
  border-radius: 6px;
  overflow: auto;
  margin-bottom: 1em;
  border: 1px solid var(--border-color);
}

.bubble-text :deep(code) {
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

.bubble-text :deep(pre code) {
  background: transparent;
  padding: 0;
  font-size: 100%;
  white-space: pre;
}

.bubble-text :deep(blockquote) {
  padding: 0 1em;
  color: var(--secondary-text);
  border-left: 0.25em solid var(--border-color);
  margin-bottom: 1em;
}

.bubble-text :deep(table) {
  border-spacing: 0;
  border-collapse: collapse;
  margin-bottom: 1em;
  width: 100%;
}

.bubble-text :deep(th),
.bubble-text :deep(td) {
  padding: 6px 13px;
  border: 1px solid var(--border-color);
}

.bubble-text :deep(tr:nth-child(2n)) {
  background-color: var(--bg-color);
}

.bubble-text :deep(img) {
  max-width: 100%;
  box-sizing: content-box;
  background-color: var(--bg-color);
}

.bubble-text :deep(a) {
  color: var(--link-color, #0969da);
  text-decoration: none;
}

.bubble-text :deep(a:hover) {
  text-decoration: underline;
}

/* Progress indicator shown above assistant's message */
.progress-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 6px 0;
}
.spinner {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 3px solid var(--subtitle-color);
  border-right-color: transparent; /* create a gap on the right */
  box-sizing: border-box;
  animation: spin 0.95s linear infinite;
}
.node-text {
  color: var(--subtitle-color);
  font-size: 13px;
  opacity: 0.95;
}

.retry-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0 0 0;
  padding-left: 16px;
}
.retry-btn {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 2px solid var(--border-color);
  background: transparent;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  color: var(--secondary-text);
}
.retry-btn:hover {
  background: var(--hover-bg);
  color: var(--text-color);
}
.retry-btn i {
  font-size: 12px;
}
.retry-text {
  font-size: 12px;
  color: var(--secondary-text);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
