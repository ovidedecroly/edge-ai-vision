<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useThemeStore } from '@/stores/theme'
import apiClient from '@/api/client'
import {
  LayoutDashboard,
  FileText,
  Users,
  ShieldCheck,
  Settings,
  DoorClosed,
  ChevronDown,
  Sun,
  Moon,
  Zap,
  LogOut,
  Sparkles,
  Shield,
  Crown
} from 'lucide-vue-next'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const themeStore = useThemeStore()

const userMenuOpen = ref(false)
const activeAiModel = ref('yolo26_ocr') // internal ID only

// Simulated or real role mode toggle for demo/operational use
// If user has Administrator role, they can switch view between Super Admin and Admin (Pos Satpam)
const activeRoleView = ref<'super_admin' | 'admin'>('super_admin')

const isSuperAdmin = computed(() => {
  return authStore.user?.is_superuser ||
    authStore.roles.includes('Administrator') ||
    activeRoleView.value === 'super_admin'
})

const availableModels = ref([
  { id: 'yolo26_ocr', name: 'Model 1', badge: 'Akurasi Tinggi' },
  { id: 'yolo26_lprnet', name: 'Model 2', badge: 'Ultra-Cepat' }
])

// Base Operator (Admin) Navigation
const baseNavItems = [
  { name: 'Dashboard ANPR', path: '/dashboard', icon: LayoutDashboard },
  { name: 'Riwayat Kendaraan', path: '/events', icon: FileText },
  { name: 'Sivitas Whitelist', path: '/whitelist', icon: ShieldCheck },
]

// Super Admin Navigation
const superAdminNavItems = [
  { name: 'Kelola Perangkat', path: '/devices', icon: DoorClosed },
  { name: 'Manajemen Pengguna', path: '/users', icon: Users },
  { name: 'Pengaturan Sistem', path: '/settings', icon: Settings },
  { name: 'Audit Log', path: '/audit', icon: Shield },
]

const visibleNavItems = computed(() => {
  if (activeRoleView.value === 'super_admin') {
    return {
      base: baseNavItems,
      admin: superAdminNavItems
    }
  }
  return {
    base: baseNavItems,
    admin: []
  }
})

async function fetchAiModel() {
  try {
    const res = await apiClient.get('/gate/model')
    activeAiModel.value = res.data.active_model
  } catch (err) {
    // fallback
  }
}

async function handleSwitchModel(modelId: string) {
  try {
    await apiClient.post('/gate/model', { model_id: modelId })
    activeAiModel.value = modelId
  } catch (err) {
    console.error('Failed to switch model', err)
  }
}

function toggleRoleView() {
  activeRoleView.value = activeRoleView.value === 'super_admin' ? 'admin' : 'super_admin'
}

onMounted(async () => {
  await fetchAiModel()
  if (authStore.user) {
    if (authStore.user.is_superuser || authStore.roles.includes('Administrator')) {
      activeRoleView.value = 'super_admin'
    } else {
      activeRoleView.value = 'admin'
    }
  }
})

function handleLogout() {
  authStore.logout()
  router.push('/login')
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 flex flex-col font-sans transition-colors duration-200">
    <!-- Topbar -->
    <header class="h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between px-6 z-20 sticky top-0 shadow-xs transition-colors duration-200">
      <!-- Left side: Platform Branding (UNSGATE) -->
      <div class="flex items-center space-x-5">
        <div class="flex items-center space-x-3 cursor-pointer" @click="router.push('/dashboard')">
          <div class="w-9 h-9 rounded-xl bg-linear-to-br from-[#f97316] to-[#ea580c] text-white flex items-center justify-center font-black text-xs tracking-wider shadow-md">
            UNS
          </div>
          <div>
            <div class="flex items-center space-x-1.5">
              <h1 class="font-black text-[#173B65] dark:text-blue-400 text-lg leading-tight tracking-tight">UNSGATE</h1>
              <span class="text-[9px] font-bold px-1.5 py-0.2 bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-300 rounded uppercase">
                Campus ANPR
              </span>
            </div>
            <p class="text-[10px] text-gray-500 dark:text-gray-400 leading-tight font-medium">Sistem Otomasi Gerbang Kampus UNS</p>
          </div>
        </div>

        <!-- Role Indicator & View Switcher -->
        <button
          @click="toggleRoleView"
          :title="'Klik untuk beralih tampilan (Sekarang: ' + (activeRoleView === 'super_admin' ? 'Super Admin' : 'Admin Operator') + ')'"
          class="hidden lg:flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-bold border transition cursor-pointer"
          :class="activeRoleView === 'super_admin'
            ? 'bg-purple-50 dark:bg-purple-950/40 border-purple-200 dark:border-purple-800 text-purple-700 dark:text-purple-300 hover:bg-purple-100'
            : 'bg-blue-50 dark:bg-blue-950/40 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 hover:bg-blue-100'"
        >
          <Crown v-if="activeRoleView === 'super_admin'" class="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
          <Shield v-else class="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
          <span>{{ activeRoleView === 'super_admin' ? 'Super Admin' : 'Admin (Pos Satpam)' }}</span>
          <span class="text-[9px] opacity-70 ml-1">⇄ Ganti</span>
        </button>
      </div>

      <!-- Center: Model 1 vs Model 2 Selector -->
      <div class="hidden md:flex items-center space-x-3 text-xs">
        <div class="flex items-center space-x-1.5 bg-gray-100 dark:bg-gray-800/80 border border-gray-200 dark:border-gray-700 rounded-xl p-1">
          <div class="flex items-center space-x-1.5 px-2 text-xs font-bold text-gray-700 dark:text-gray-300">
            <Zap class="w-3.5 h-3.5 text-amber-500" />
            <span class="text-[11px] uppercase tracking-wider">Pilih AI:</span>
          </div>
          <div class="flex items-center space-x-1">
            <button
              @click="handleSwitchModel('yolo26_ocr')"
              :class="activeAiModel === 'yolo26_ocr'
                ? 'bg-[#173B65] text-white shadow-xs font-bold'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-white dark:hover:bg-gray-700'"
              class="px-3 py-1 rounded-lg text-xs transition flex items-center space-x-1.5"
              title="Model 1 - Akurasi Tinggi"
            >
              <span>Model 1</span>
              <span class="text-[10px] opacity-80 font-normal">(Akurasi Tinggi)</span>
            </button>
            <button
              @click="handleSwitchModel('yolo26_lprnet')"
              :class="activeAiModel === 'yolo26_lprnet'
                ? 'bg-[#f97316] text-white shadow-xs font-bold'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-white dark:hover:bg-gray-700'"
              class="px-3 py-1 rounded-lg text-xs transition flex items-center space-x-1.5"
              title="Model 2 - Ultra-Cepat"
            >
              <span>Model 2</span>
              <span class="text-[10px] opacity-80 font-normal">(Ultra-Cepat)</span>
            </button>
          </div>
        </div>

        <!-- Active Camera Indicator -->
        <div class="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-gray-100 dark:bg-gray-800/80 border border-gray-200 dark:border-gray-700">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span class="font-medium text-gray-600 dark:text-gray-400 text-xs">Device:</span>
          <span class="font-bold text-[11px] text-emerald-600 dark:text-emerald-400">DS-2CD1043G2-LIU</span>
        </div>
      </div>

      <!-- Right side: Theme Toggle & User Menu -->
      <div class="flex items-center space-x-3">
        <!-- Mode Siang / Mode Malam Switcher (Segmented Switch) -->
        <div class="flex items-center bg-gray-100 dark:bg-gray-800 p-0.5 rounded-xl border border-gray-200 dark:border-gray-700">
          <button
            @click="themeStore.applyTheme('light')"
            class="flex items-center space-x-1 px-2.5 py-1 rounded-lg text-xs font-bold transition-all"
            :class="themeStore.currentTheme === 'light'
              ? 'bg-white text-amber-600 shadow-xs'
              : 'text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white'"
            title="Aktifkan Mode Siang"
          >
            <Sun class="w-3.5 h-3.5" :class="themeStore.currentTheme === 'light' ? 'text-amber-500' : ''" />
            <span class="hidden sm:inline">Siang</span>
          </button>
          <button
            @click="themeStore.applyTheme('dark')"
            class="flex items-center space-x-1 px-2.5 py-1 rounded-lg text-xs font-bold transition-all"
            :class="themeStore.currentTheme === 'dark'
              ? 'bg-[#173B65] text-white shadow-xs'
              : 'text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white'"
            title="Aktifkan Mode Malam"
          >
            <Moon class="w-3.5 h-3.5" :class="themeStore.currentTheme === 'dark' ? 'text-blue-300' : ''" />
            <span class="hidden sm:inline">Malam</span>
          </button>
        </div>

        <!-- User Profile Dropdown -->
        <div class="relative">
          <button
            @click="userMenuOpen = !userMenuOpen"
            class="flex items-center space-x-2 text-xs text-gray-800 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 px-2.5 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 transition"
          >
            <div class="w-6 h-6 rounded-lg bg-[#173B65]/10 dark:bg-blue-900/40 text-[#173B65] dark:text-blue-300 font-bold flex items-center justify-center text-xs">
              {{ (authStore.user?.username || 'A')[0].toUpperCase() }}
            </div>
            <div class="text-left hidden sm:block">
              <span class="font-bold block leading-none">{{ authStore.user?.username || 'admin' }}</span>
              <span class="text-[9px] text-gray-400 capitalize block leading-none mt-0.5">
                {{ activeRoleView === 'super_admin' ? 'Super Admin' : 'Admin Operator' }}
              </span>
            </div>
            <ChevronDown class="w-3.5 h-3.5 text-gray-500" />
          </button>

          <!-- Dropdown menu -->
          <div
            v-if="userMenuOpen"
            class="absolute right-0 mt-2 w-56 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-xl py-2 z-30 space-y-1"
          >
            <div class="px-3 py-2 border-b border-gray-100 dark:border-gray-700 text-xs">
              <p class="font-bold text-gray-900 dark:text-white">{{ authStore.user?.full_name || 'Petugas Gerbang UNS' }}</p>
              <p class="text-gray-500 dark:text-gray-400 text-[11px] truncate">{{ authStore.user?.email || 'admin@uns.ac.id' }}</p>
              <div class="mt-1.5 inline-block px-2 py-0.5 rounded text-[10px] font-bold"
                :class="activeRoleView === 'super_admin' ? 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300' : 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'">
                {{ activeRoleView === 'super_admin' ? '👑 Mode Super Admin' : '🛡️ Mode Admin (Pos Satpam)' }}
              </div>
            </div>

            <button
              @click="toggleRoleView"
              class="w-full text-left px-3 py-2 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 flex items-center space-x-2"
            >
              <Sparkles class="w-3.5 h-3.5 text-purple-500" />
              <span>Ganti ke {{ activeRoleView === 'super_admin' ? 'Admin Pos Satpam' : 'Super Admin' }}</span>
            </button>

            <button
              @click="handleLogout"
              class="w-full text-left px-3 py-2 text-xs text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/40 flex items-center space-x-2"
            >
              <LogOut class="w-3.5 h-3.5" />
              <span>Keluar (Sign Out)</span>
            </button>
          </div>
        </div>
      </div>
    </header>

    <div class="flex-1 flex overflow-hidden">
      <!-- Sidebar -->
      <aside class="w-60 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col justify-between shrink-0 transition-colors duration-200">
        <nav class="p-3 space-y-4 overflow-y-auto">
          <!-- Operasional Section -->
          <div>
            <p class="px-3 text-[10px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-1.5">
              Operasional Gate
            </p>
            <div class="space-y-1">
              <router-link
                v-for="item in visibleNavItems.base"
                :key="item.path"
                :to="item.path"
                class="flex items-center space-x-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all"
                :class="route.path === item.path
                  ? 'bg-[#173B65] text-white shadow-xs'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-[#173B65] dark:hover:text-white'"
              >
                <component :is="item.icon" class="w-4 h-4" />
                <span>{{ item.name }}</span>
              </router-link>
            </div>
          </div>

          <!-- Super Admin Section -->
          <div v-if="visibleNavItems.admin.length > 0">
            <div class="flex items-center justify-between px-3 mb-1.5">
              <p class="text-[10px] font-bold uppercase tracking-wider text-purple-600 dark:text-purple-400">
                Super Admin
              </p>
              <Crown class="w-3 h-3 text-purple-500" />
            </div>
            <div class="space-y-1">
              <router-link
                v-for="item in visibleNavItems.admin"
                :key="item.path"
                :to="item.path"
                class="flex items-center space-x-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all"
                :class="route.path === item.path
                  ? 'bg-purple-700 text-white shadow-xs'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-purple-600 dark:hover:text-purple-300'"
              >
                <component :is="item.icon" class="w-4 h-4" />
                <span>{{ item.name }}</span>
              </router-link>
            </div>
          </div>
        </nav>

        <!-- Sidebar footer: Hardware & AI Engine Status -->
        <div class="p-4 border-t border-gray-200 dark:border-gray-800 bg-gray-50/70 dark:bg-gray-950/60 text-[11px] text-gray-500 dark:text-gray-400 space-y-1.5">
          <div class="flex justify-between items-center">
            <span>Server Gate</span>
            <span class="font-mono text-gray-900 dark:text-gray-200 font-semibold">Localhost (RTX Edge)</span>
          </div>
          <div class="flex justify-between items-center">
            <span>Model Aktif</span>
            <span class="font-semibold" :class="activeAiModel === 'yolo26_ocr' ? 'text-blue-600 dark:text-blue-400' : 'text-amber-600 dark:text-amber-400'">
              {{ activeAiModel === 'yolo26_ocr' ? 'Model 1 (Akurasi)' : 'Model 2 (Cepat)' }}
            </span>
          </div>
          <div class="flex justify-between items-center">
            <span>Kamera Gate</span>
            <span class="font-mono text-emerald-600 dark:text-emerald-400 font-medium">DS-2CD1043G2</span>
          </div>
        </div>
      </aside>

      <!-- Main content view -->
      <main class="flex-1 overflow-y-auto p-6 bg-gray-50 dark:bg-gray-950 transition-colors duration-200">
        <router-view />
      </main>
    </div>
  </div>
</template>
