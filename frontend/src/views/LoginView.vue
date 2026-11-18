<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { Lock, User, AlertCircle, Shield, Crown, Sparkles } from 'lucide-vue-next'

const router = useRouter()
const authStore = useAuthStore()

const username = ref('admin')
const password = ref('Admin@123456')
const error = ref('')
const loading = ref(false)

function setPreset(u: string, p: string) {
  username.value = u
  password.value = p
}

async function handleLogin() {
  error.value = ''
  loading.value = true
  try {
    await authStore.login(username.value, password.value)
    router.push('/dashboard')
  } catch (err: any) {
    error.value = err.response?.data?.detail || 'Username atau password salah. Silakan coba lagi.'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center p-4">
    <div class="w-full max-w-md bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl shadow-xl p-8 space-y-6">
      <!-- Header -->
      <div class="text-center space-y-2">
        <div class="w-14 h-14 rounded-2xl bg-linear-to-br from-[#f97316] to-[#ea580c] text-white flex items-center justify-center font-black text-xl mx-auto shadow-md">
          UNS
        </div>
        <div>
          <h2 class="text-2xl font-black text-gray-900 dark:text-white tracking-tight">UNSGATE</h2>
          <p class="text-xs text-gray-500 dark:text-gray-400">Sistem Otomasi Gerbang Kampus ANPR</p>
        </div>
      </div>

      <!-- Quick Role Select Presets -->
      <div class="p-3 bg-gray-50 dark:bg-gray-800/60 rounded-xl border border-gray-200 dark:border-gray-700/60 space-y-2 text-xs">
        <span class="text-[10px] uppercase font-bold text-gray-500 dark:text-gray-400 block tracking-wider">
          Pilihan Akses Cepat Login:
        </span>
        <div class="grid grid-cols-2 gap-2">
          <button
            type="button"
            @click="setPreset('admin', 'Admin@123456')"
            class="p-2 rounded-lg text-left border transition cursor-pointer"
            :class="username === 'admin'
              ? 'bg-purple-50 dark:bg-purple-950/40 border-purple-300 dark:border-purple-700 text-purple-700 dark:text-purple-300'
              : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50'"
          >
            <div class="flex items-center space-x-1 font-bold">
              <Crown class="w-3.5 h-3.5 text-purple-500" />
              <span>Super Admin</span>
            </div>
            <p class="text-[10px] text-gray-400 mt-0.5 font-mono">admin / Admin@123456</p>
          </button>

          <button
            type="button"
            @click="setPreset('operator', 'Operator@123')"
            class="p-2 rounded-lg text-left border transition cursor-pointer"
            :class="username === 'operator'
              ? 'bg-blue-50 dark:bg-blue-950/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300'
              : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50'"
          >
            <div class="flex items-center space-x-1 font-bold">
              <Shield class="w-3.5 h-3.5 text-blue-500" />
              <span>Pos Satpam</span>
            </div>
            <p class="text-[10px] text-gray-400 mt-0.5 font-mono">operator / Operator@123</p>
          </button>
        </div>
      </div>

      <!-- Error Alert -->
      <div v-if="error" class="p-3 bg-rose-50 dark:bg-rose-950/40 border border-rose-300 dark:border-rose-800 rounded-lg text-rose-800 dark:text-rose-300 text-xs flex items-center space-x-2">
        <AlertCircle class="w-4 h-4 shrink-0" />
        <span>{{ error }}</span>
      </div>

      <!-- Form -->
      <form @submit.prevent="handleLogin" class="space-y-4">
        <div class="space-y-1">
          <label class="text-xs font-semibold text-gray-700 dark:text-gray-300">Username</label>
          <div class="relative">
            <User class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
            <input
              v-model="username"
              type="text"
              required
              placeholder="Username"
              class="w-full pl-9 pr-3 py-2 text-xs bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#173B65] text-gray-900 dark:text-white"
            />
          </div>
        </div>

        <div class="space-y-1">
          <label class="text-xs font-semibold text-gray-700 dark:text-gray-300">Password</label>
          <div class="relative">
            <Lock class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
            <input
              v-model="password"
              type="password"
              required
              placeholder="Password"
              class="w-full pl-9 pr-3 py-2 text-xs bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#173B65] text-gray-900 dark:text-white"
            />
          </div>
        </div>

        <button
          type="submit"
          :disabled="loading"
          class="w-full py-2.5 px-4 bg-[#173B65] hover:bg-[#0f2942] text-white rounded-xl text-xs font-bold tracking-wide transition-colors flex items-center justify-center space-x-2 disabled:opacity-50 shadow-md cursor-pointer"
        >
          <span v-if="loading">Memverifikasi kredensial...</span>
          <span v-else>Masuk ke Sistem UNSGate</span>
        </button>
      </form>

      <!-- Footer Info -->
      <div class="pt-4 border-t border-gray-100 dark:border-gray-800 flex items-center justify-center space-x-1.5 text-[11px] text-gray-500 dark:text-gray-400">
        <Shield class="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
        <span>Kamera Hikvision DS-2CD1043G2-LIU • Localhost LAN UNS</span>
      </div>
    </div>
  </div>
</template>
