<template>
  <div class="large-chat">
    <div class="chat-container">
      <div class="messages" ref="messagesEl">
        <div
          v-for="(m, idx) in messages"
          :key="idx"
          :class="['message-row', m.role === 'user' ? 'user' : 'assistant']"
        >
          <div class="bubble">
            <div class="bubble-text">{{ m.text }}</div>
          </div>
        </div>
      </div>

      <!-- input handled by RepoDetail AskBox when LargeChat is visible -->
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onUnmounted } from 'vue'
import { askRagStream } from '../utils/request'

const props = withDefaults(
  defineProps<{ placeholder?: string; owner?: string; repo?: string; platform?: string; mode?: 'fast' | 'smart' }>(),
  { placeholder: 'Try to ask me...', platform: 'github', mode: 'fast' }
)
const emit = defineEmits<{
  (e: 'send', text: string): void
  (e: 'new-repo'): void
}>()

type Msg = { role: 'user' | 'assistant'; text: string }

const messages = ref<Msg[]>([
  { role: 'assistant', text: '你好，有什么我可以帮忙的吗？' },
])
const input = ref('')
const messagesEl = ref<HTMLElement | null>(null)
let abortController: AbortController | null = null

const scrollToBottom = async () => {
  await nextTick()
  const el = messagesEl.value
  if (el) el.scrollTop = el.scrollHeight
}

const appendAssistantPlaceholder = () => {
  messages.value.push({ role: 'assistant', text: '正在思考……' })
}

const updateLastAssistant = (text: string) => {
  for (let i = messages.value.length - 1; i >= 0; i--) {
    const msg = messages.value[i]
    if (!msg) continue
    if (msg.role === 'assistant') {
      msg.text = text
      return
    }
  }
  // no assistant found, push new
  messages.value.push({ role: 'assistant', text })
}

onUnmounted(() => {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
})

const streamRagAnswer = async (question: string) => {
  if (!props.owner || !props.repo) {
    updateLastAssistant('缺少 owner/repo 信息，无法发起 RAG 请求。')
    return
  }

  if (abortController) {
    abortController.abort()
  }
  abortController = new AbortController()

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
          // prefer answer field
          const ans = d && typeof d === 'object' ? (d as any).answer ?? (d as any).text ?? '' : String(d || '')
          if (ans !== undefined && ans !== null) {
            updateLastAssistant(String(ans))
            scrollToBottom()
          }
        } catch (err) {
          // fallback to message
          updateLastAssistant(event.message ?? JSON.stringify(event))
          scrollToBottom()
        }
      },
      { signal: abortController.signal }
    )
  } catch (err) {
    if ((err as any)?.name === 'AbortError') {
      updateLastAssistant('已取消请求。')
    } else {
      updateLastAssistant(`请求出错：${(err as Error).message}`)
    }
  } finally {
    abortController = null
  }
}

const handleSend = () => {
  const text = input.value && input.value.trim()
  if (!text) return
  messages.value.push({ role: 'user', text })
  emit('send', text)
  input.value = ''
  scrollToBottom()

  appendAssistantPlaceholder()
  scrollToBottom()
  void streamRagAnswer(text)
}

// Expose a method to allow parent to programmatically send a message
const receiveMessage = (text: string) => {
  if (!text) return
  console.debug('[LargeChat] receiveMessage:', text)
  messages.value.push({ role: 'user', text })
  appendAssistantPlaceholder()
  scrollToBottom()
  void streamRagAnswer(text)
}

defineExpose({ receiveMessage })
</script>

<style scoped>
.large-chat {
  position: relative;
  /* Use viewport width so the chat occupies ~80% of the page horizontally */
  width: 80vw;
  max-width: 1200px;
  margin: 0 auto;
  height: 100%;
  display: flex;
  flex-direction: column;
  z-index: 1200;
  padding-bottom: 96px; /* leave space for page AskBox */
}
.chat-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.messages {
  flex: 1 1 auto;
  overflow: auto;
  padding: 20px 24px;
  background: transparent;
}
.message-row {
  display: flex;
  margin: 8px 0;
}
.message-row.assistant {
  justify-content: flex-start;
}
.message-row.user {
  justify-content: flex-end;
}
.bubble {
  /* allow bubbles to take more of the wider chat area */
  max-width: 85%;
  padding: 12px 16px;
  border-radius: 14px;
  box-shadow: 0 6px 14px rgba(0,0,0,0.06);
}
.message-row.assistant .bubble {
  background: #9bd6ef; /* light blue */
  color: #0b2b34;
  border-top-left-radius: 6px;
  border-top-right-radius: 14px;
  border-bottom-right-radius: 14px;
  border-bottom-left-radius: 14px;
}
.message-row.user .bubble {
  background: #e9ecef; /* light gray */
  color: #1f2d36;
  border-top-right-radius: 6px;
}
.bubble-text {
  white-space: pre-wrap;
  line-height: 1.45;
}
</style>
