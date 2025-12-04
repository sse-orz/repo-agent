<template>
  <div class="progress-container" :class="{ 'is-mini': isMini }">
    <!-- 顶部状态栏：包含阶段文字和百分比 -->
    <div class="progress-header">
      <div class="stages-wrapper">
        <span 
          v-for="(stage, index) in stages" 
          :key="index"
          class="stage-text"
          :class="{ 'active-stage': currentStageIndex === index, 'passed-stage': currentStageIndex > index }"
        >
          {{ stage }}
        </span>
      </div>
      <span class="progress-percent">{{ displayPercent }}%</span>
    </div>

    <!-- 进度条主体 -->
    <div class="progress-bar-wrapper">
      <div class="progress-bar" :style="{ width: displayProgress + '%' }">
        <div class="progress-glow"></div>
      </div>
    </div>

    <!-- 日志区域：Full模式显示列表，Mini模式只显示最新一条 -->
    <div class="log-container">
      <div v-if="!isMini && logs.length > 0" class="stream-log full-log">
        <p v-for="(log, index) in logs" :key="index">{{ log }}</p>
      </div>
      <div v-else-if="isMini && logs.length > 0" class="stream-log mini-log">
        <span class="mini-log-dot">●</span>
        <span class="mini-log-text">{{ logs[logs.length - 1] }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, onUnmounted, onMounted } from 'vue'

interface Props {
  progress?: number
  logs?: string[]
  // 新增：用于控制是否进入迷你悬浮模式 (当文档有内容时设为 true)
  isMini?: boolean 
}

const props = withDefaults(defineProps<Props>(), {
  progress: 0,
  logs: () => [],
  isMini: false,
})

// --- 阶段定义 ---
const stages = ['Init', 'Analyzing', 'Generating', 'Finalizing']

// --- 核心状态 ---
const displayProgress = ref(0)
let animationFrame: number | null = null

// --- 计算属性 ---

// 格式化显示的百分比
const displayPercent = computed(() => Math.floor(displayProgress.value))

// 计算当前处于哪个阶段 
const currentStageIndex = computed(() => {
  const p = displayProgress.value
  if (p < 20) return 0
  if (p < 50) return 1
  if (p < 90) return 2
  return 3
})

// --- 动画与虚假进度逻辑 ---

// 平滑动画更新函数
const updateProgress = () => {
  const target = Math.min(100, Math.max(0, props.progress))
  const current = displayProgress.value

  if (current < target) {
    const diff = target - current
    displayProgress.value += diff * 0.1 + 0.05
  } 
  //虚假进度逻辑
  else if (current >= target && target < 99 && current < target + 5) {
    displayProgress.value += 0.03
  }

  // 边界修正
  if (displayProgress.value > 99.5 && props.progress < 100) {
    displayProgress.value = 99.5 // 只要没收到100%，就卡在 99.5%
  } else if (displayProgress.value > 100) {
    displayProgress.value = 100
  }

  animationFrame = requestAnimationFrame(updateProgress)
}

// 监听真实进度变化，唤醒动画
watch(() => props.progress, (newVal) => {
  if (newVal >= 100) {
    displayProgress.value = 100
  }
})

onMounted(() => {
  displayProgress.value = Math.max(0, props.progress) // 初始同步
  updateProgress()
})

onUnmounted(() => {
  if (animationFrame) cancelAnimationFrame(animationFrame)
})
</script>

<style scoped>
/* --- 容器基础样式 --- */
.progress-container {
  padding: 16px 0;
  width: 100%;
  max-width: none;
  margin: 0; 
  transition: all 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
  background: transparent;
  z-index: 100;
}

/* --- Mini 模式 (右上角悬浮) --- */
.progress-container.is-mini {
  position: fixed;
  top: 90px;  
  right: 20px;

  width: 180px;
  padding: 16px;

  background: var(--bg-color, #ffffff);
  border: 1px solid var(--border-color, #eee);
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
  backdrop-filter: blur(10px);
}

/* --- 头部区域 --- */
.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end; /* 底部对齐，为了字体放大时的基线 */
  margin-bottom: 12px;
}

.stages-wrapper {
  display: flex;
  gap: 20px;
}

.stage-text {
  font-size: 15px;
  color: var(--secondary-text, #999);
  transition: all 0.4s ease;
  transform-origin: left bottom;
  position: relative;
}

/* 当前阶段：变大、加粗、高亮 */
.stage-text.active-stage {
  font-size: 19px; /* 字体明显变大 */
  font-weight: 700;
  color: var(--title-color, #38916e);
  transform: scale(1.1);
  text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* 已过阶段：略微变暗 */
.stage-text.passed-stage {
  color: var(--success-color, #54745b);
  opacity: 0.8;
  font-size: 15px;
}

/* 在 Mini 模式下隐藏非当前的阶段文字，节省空间 */
.is-mini .stage-text:not(.active-stage) {
  display: none;
}
.is-mini .stage-text.active-stage{
  font-size: 14px;
  color: #d6d2d2;
}

.progress-percent {
  font-size: 14px;
  font-weight: 700;
  color: var(--title-color, #333);
  font-variant-numeric: tabular-nums;
}

/* --- 进度条 --- */
.progress-bar-wrapper {
  width: 100%;
  height: 8px; 
  background: var(--border-color, #eee);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 16px;
  position: relative;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #9facb6, #709493);
  border-radius: 4px;
  position: relative;
}

/* 进度条光效动画 */
.progress-glow {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.4) 50%,
    transparent 100%
  );
  animation: shimmer 1.5s infinite linear;
  transform: translateX(-100%);
}

@keyframes shimmer {
  100% { transform: translateX(100%); }
}

/* --- 日志区域 --- */
.stream-log {
  font-family: -apple-system, monospace;
  font-size: 13px;
  color: var(--secondary-text, #666);
  line-height: 1.6;
}

/* Full 模式下的日志 */
.full-log p {
  margin: 4px 0;
  padding-left: 16px;
  position: relative;
}
.full-log p::before {
  content: '>';
  position: absolute;
  left: 0;
  color: #709493;
  font-weight: bold;
}

/* Mini 模式下的日志 */
.mini-log {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  align-items: flex-start; /* 顶部对齐 更好看 */
  font-size: 12px;
  color: #888;
  border-top: 1px dashed var(--border-color, #eee);
  padding-top: 8px;
  margin-top: 8px;
}

.mini-log-dot {
  color: #50d8b2;
  margin-right: 4px;
  animation: pulse 1s infinite;
}

.mini-log-text {
  white-space: normal; /* 允许换行 */
  overflow: visible; 
  word-break: break-word;
}

@keyframes pulse {
  0% { opacity: 0.5; transform: scale(0.8); }
  50% { opacity: 1; transform: scale(1.2); }
  100% { opacity: 0.5; transform: scale(0.8); }
}

/* 暗色模式简单适配 (如果父组件提供了CSS变量) */
@media (prefers-color-scheme: dark) {
  .progress-container.is-mini {
    background: #1e1e1e;
    border-color: #333;
  }
}
</style>