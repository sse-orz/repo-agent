<template>
  <div class="header">
    <h1 class="title">Repo Agent</h1>
    <p class="subtitle">
      <span class="text">{{ displayText }}</span>
      <!-- 只有当没有结束，或者结束但保留光标时显示 -->
      <span class="cursor" :class="{ 'cursor-stop': isFinished }">|</span>
    </p>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// --- 配置项 ---
// 这里放你想展示的多段文本
const textArray = [
  'Help you read and understand repositories quickly',
  "Now let's explore your code. ^ ^", // 优化后的引导语
]
const typingSpeed = 60 // 打字基础速度 (ms)
const deletingSpeed = 30 // 删除速度 (ms) - 稍微调快了一点，让过渡更利索
const pauseDuration = 1500 // 句子打完后的停留时间 (ms)

// --- 状态管理 ---
const displayText = ref('')
const isFinished = ref(false) // 标记是否彻底完成了所有对话
let textIndex = 0 // 当前在打第几句话
let charIndex = 0 // 当前字符索引
let isDeleting = false // 是否处于删除状态
let timer = null

// --- 打字机核心逻辑 ---
const typeEffect = () => {
  // 获取当前要处理的那一句话
  const currentFullText = textArray[textIndex]

  if (isDeleting) {
    // 删除逻辑
    displayText.value = currentFullText.substring(0, charIndex - 1)
    charIndex--
  } else {
    // 输入逻辑
    displayText.value = currentFullText.substring(0, charIndex + 1)
    charIndex++
  }

  // 计算下一次的延迟
  let delta = typingSpeed
  if (!isDeleting) delta += Math.random() * 30 // 打字随机延迟
  if (isDeleting) delta = deletingSpeed // 删除固定速度

  // --- 关键的状态流转逻辑 ---

  // 1. 如果当前句子打完了
  if (!isDeleting && displayText.value === currentFullText) {
    // 判断是否是最后一句
    if (textIndex === textArray.length - 1) {
      // 是最后一句 -> 停止！
      isFinished.value = true
      return // 直接返回，不再调用 setTimeout，逻辑结束
    } else {
      // 不是最后一句 -> 停顿后开始删除
      delta = pauseDuration
      isDeleting = true
    }
  } else if (isDeleting && displayText.value === '') {
    // 2. 如果当前句子删完了
    isDeleting = false
    textIndex++ // 切换到下一句话
    delta = 500 // 稍微停顿再开始打下一句
  }

  // 递归调用
  timer = setTimeout(typeEffect, delta)
}

onMounted(() => {
  typeEffect()
})

onUnmounted(() => {
  if (timer) clearTimeout(timer)
})
</script>

<style scoped>
.header {
  text-align: center;
  margin-bottom: 40px;
  opacity: 0;
  animation: slideUpFade 0.8s ease-out forwards;
}

.title {
  font-size: 38px;
  font-weight: 700;
  margin: 0 0 12px 0;
  color: var(--title-color, #333);
  transition: color 0.3s;
}

.subtitle {
  font-size: 14px;
  color: var(--subtitle-color, #666);
  margin: 0;
  line-height: 1.4;
  transition: color 0.3s;
  min-height: 1.4em; /* 防止高度塌陷 */
}

/* 光标通用样式 */
.cursor {
  display: inline-block;
  margin-left: 2px;
  font-weight: 400;
  color: inherit;
  opacity: 1;
  animation: blink 1s step-end infinite;
}

.cursor-stop {
  animation: none;
  opacity: 0;
}

@keyframes blink {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0;
  }
}
</style>
