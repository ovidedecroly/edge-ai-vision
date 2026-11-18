import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useThemeStore = defineStore('theme', () => {
  const saved = localStorage.getItem('unsgate_theme') as 'dark' | 'light' | null
  const currentTheme = ref<'dark' | 'light'>(saved || 'dark')

  function applyTheme(theme: 'dark' | 'light') {
    currentTheme.value = theme
    localStorage.setItem('unsgate_theme', theme)
    if (typeof document !== 'undefined') {
      if (theme === 'dark') {
        document.documentElement.classList.add('dark')
        document.documentElement.classList.remove('light')
      } else {
        document.documentElement.classList.remove('dark')
        document.documentElement.classList.add('light')
      }
    }
  }

  function toggleTheme() {
    applyTheme(currentTheme.value === 'dark' ? 'light' : 'dark')
  }

  // Initial apply
  applyTheme(currentTheme.value)

  return {
    currentTheme,
    applyTheme,
    toggleTheme
  }
})
