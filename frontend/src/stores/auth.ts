import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import apiClient from '@/api/client'
import type { User } from '@/types'

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(localStorage.getItem('access_token'))
  const user = ref<User | null>(null)
  const isAuthenticated = computed(() => !!token.value)
  const roles = computed(() => user.value?.roles.map(r => r.name) || [])

  const hasRole = (roleName: string) => {
    if (user.value?.is_superuser) return true
    return roles.value.includes(roleName)
  }

  async function login(username: string, password: string): Promise<boolean> {
    try {
      const res = await apiClient.post('/auth/login/json', { username, password })
      token.value = res.data.access_token
      localStorage.setItem('access_token', res.data.access_token)
      if (res.data.refresh_token) {
        localStorage.setItem('refresh_token', res.data.refresh_token)
      }
      await fetchMe()
      return true
    } catch (err) {
      console.error('Login failed', err)
      throw err
    }
  }

  async function fetchMe() {
    try {
      const res = await apiClient.get<User>('/auth/me')
      user.value = res.data
    } catch (err) {
      logout()
    }
  }

  function logout() {
    token.value = null
    user.value = null
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
  }

  return {
    token,
    user,
    isAuthenticated,
    roles,
    hasRole,
    login,
    fetchMe,
    logout
  }
})
