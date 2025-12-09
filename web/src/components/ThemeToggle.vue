<template>
  <button
    class="theme-toggle"
    @click="handleToggle"
    :aria-label="isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'"
  >
    <i :class="isDarkMode ? 'fas fa-sun' : 'fas fa-moon'"></i>
  </button>
</template>

<script setup lang="ts">
import { inject, nextTick, type Ref } from 'vue'

const { isDarkMode, toggleTheme } = inject('theme') as {
  isDarkMode: Ref<boolean>
  toggleTheme: () => void
}

const handleToggle = async (event: MouseEvent) => {
  // @ts-ignore
  if (!document.startViewTransition) {
    toggleTheme()
    return
  }

  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const x = rect.left + rect.width / 2
  const y = rect.top + rect.height / 2
  const endRadius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y))

  const isDarkBefore = isDarkMode.value

  if (isDarkBefore) {
    document.documentElement.classList.add('theme-transition-back')
  }

  // @ts-ignore
  const transition = document.startViewTransition(async () => {
    toggleTheme()
    await nextTick()
  })

  try {
    await transition.ready

    const clipPath = [
      `circle(0px at ${x}px ${y}px)`,
      `circle(${endRadius}px at ${x}px ${y}px)`,
    ]

    // If was Light (false) -> Going Dark (true)
    // New view (Dark) expands
    if (!isDarkBefore) {
      document.documentElement.animate(
        {
          clipPath: clipPath,
        },
        {
          duration: 400,
          easing: 'ease-in',
          pseudoElement: '::view-transition-new(root)',
        },
      )
    } else {
      // Was Dark (true) -> Going Light (false)
      // Old view (Dark) shrinks
      document.documentElement.animate(
        {
          clipPath: clipPath.reverse(),
        },
        {
          duration: 400,
          easing: 'ease-out',
          pseudoElement: '::view-transition-old(root)',
          fill: 'forwards',
        },
      )
    }
  } finally {
    transition.finished.then(() => {
      document.documentElement.classList.remove('theme-transition-back')
    })
  }
}
</script>

<style scoped>
.theme-toggle {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  cursor: pointer;
  font-size: 16px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: all 0.3s ease;
  color: var(--text-color);
  box-shadow: 0 2px 8px var(--shadow-color);
}

.theme-toggle:hover {
  background: var(--hover-bg);
  transform: scale(1.05);
}

.theme-toggle i {
  transition: transform 0.3s;
}

.theme-toggle:hover i {
  transform: scale(1.1);
}
</style>
