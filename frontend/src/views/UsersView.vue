<script setup lang="ts">
import { ref, onMounted } from 'vue'
import apiClient from '@/api/client'
import {
  Users,
  Shield,
  ShieldAlert,
  UserPlus,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  X
} from 'lucide-vue-next'

interface UserItem {
  id: string
  username: string
  email: string
  full_name: string
  is_active: boolean
  is_superuser: boolean
  roles: { id: string; name: string; description: string }[]
}

const users = ref<UserItem[]>([])
const loading = ref(true)
const showModal = ref(false)
const submitting = ref(false)
const statusAlert = ref<{ type: 'success' | 'error'; message: string } | null>(null)

const formData = ref({
  username: '',
  email: '',
  full_name: '',
  password: '',
  role: 'Operator'
})

function showAlert(type: 'success' | 'error', message: string) {
  statusAlert.value = { type, message }
  setTimeout(() => {
    if (statusAlert.value?.message === message) {
      statusAlert.value = null
    }
  }, 5000)
}

async function fetchUsers() {
  loading.value = true
  try {
    const res = await apiClient.get<UserItem[]>('/users')
    users.value = res.data
  } catch (err: any) {
    showAlert('error', 'Gagal memuat daftar pengguna: ' + (err.message || 'Unauthorized'))
  } finally {
    loading.value = false
  }
}

function openAddModal() {
  formData.value = {
    username: '',
    email: '',
    full_name: '',
    password: '',
    role: 'Operator'
  }
  showModal.value = true
}

async function submitUser() {
  submitting.value = true
  try {
    await apiClient.post('/users', formData.value)
    showAlert('success', `Akun ${formData.value.username} berhasil dibuat.`)
    showModal.value = false
    await fetchUsers()
  } catch (err: any) {
    showAlert('error', err.response?.data?.detail || 'Gagal membuat pengguna baru')
  } finally {
    submitting.value = false
  }
}

async function deleteUser(u: UserItem) {
  if (u.is_superuser && users.value.filter(x => x.is_superuser).length <= 1) {
    showAlert('error', 'Tidak dapat menghapus satu-satunya akun Super Admin!')
    return
  }
  if (!confirm(`Hapus akun pengguna ${u.username} (${u.full_name || 'Tanpa Nama'})?`)) {
    return
  }
  try {
    await apiClient.delete(`/users/${u.id}`)
    showAlert('success', `Pengguna ${u.username} berhasil dihapus.`)
    await fetchUsers()
  } catch (err: any) {
    showAlert('error', err.response?.data?.detail || 'Gagal menghapus pengguna')
  }
}

onMounted(fetchUsers)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <div class="flex items-center space-x-2">
          <h2 class="text-xl font-black text-gray-900 dark:text-white tracking-tight">Manajemen Pengguna & Hak Akses</h2>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-[#173B65]/10 dark:bg-blue-900/40 text-[#173B65] dark:text-blue-300 border border-[#173B65]/20">
            Super Admin
          </span>
        </div>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Pengaturan akun Super Admin (Master Control) dan Admin Operator (Petugas Pos Satpam Gate UNS)
        </p>
      </div>

      <div class="flex items-center space-x-2.5">
        <button
          @click="fetchUsers"
          :disabled="loading"
          class="px-3 py-2 rounded-lg text-xs font-semibold bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-750 transition flex items-center space-x-1.5 shadow-xs"
        >
          <RefreshCw class="w-3.5 h-3.5 text-gray-500" :class="{ 'animate-spin': loading }" />
          <span>Segarkan</span>
        </button>

        <button
          @click="openAddModal"
          class="px-3.5 py-2 rounded-lg text-xs font-bold bg-[#173B65] hover:bg-[#0f2942] text-white shadow-xs transition flex items-center space-x-1.5"
        >
          <UserPlus class="w-4 h-4" />
          <span>Tambah Pengguna Baru</span>
        </button>
      </div>
    </div>

    <!-- Alert Banner -->
    <div
      v-if="statusAlert"
      class="p-3.5 rounded-lg text-xs font-medium flex items-center space-x-2.5 transition-all"
      :class="statusAlert.type === 'success'
        ? 'bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-300 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300'
        : 'bg-rose-50 dark:bg-rose-950/40 border border-rose-300 dark:border-rose-800 text-rose-800 dark:text-rose-300'"
    >
      <CheckCircle2 v-if="statusAlert.type === 'success'" class="w-4 h-4 shrink-0" />
      <AlertTriangle v-else class="w-4 h-4 shrink-0" />
      <span class="flex-1">{{ statusAlert.message }}</span>
      <button @click="statusAlert = null" class="text-xs opacity-70 hover:opacity-100">&times;</button>
    </div>

    <!-- Role Info Cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-xs flex items-start space-x-3.5">
        <div class="w-10 h-10 rounded-xl bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 flex items-center justify-center font-bold shrink-0">
          <ShieldAlert class="w-5 h-5" />
        </div>
        <div>
          <h3 class="font-bold text-sm text-gray-900 dark:text-white">Super Admin (Master Control)</h3>
          <p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            Hak akses penuh: Manajemen Kamera & Barrier Gate, Tambah/Hapus Pengguna, Konfigurasi AI Model & Engine ANPR, serta Audit Log.
          </p>
        </div>
      </div>

      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-xs flex items-start space-x-3.5">
        <div class="w-10 h-10 rounded-xl bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 flex items-center justify-center font-bold shrink-0">
          <Shield class="w-5 h-5" />
        </div>
        <div>
          <h3 class="font-bold text-sm text-gray-900 dark:text-white">Admin Operator (Pos Satpam Gate UNS)</h3>
          <p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            Hak akses operasional harian: Monitoring Live ANPR Kamera Masuk/Keluar, Trigger Buka Palang Manual, Verifikasi Sivitas UNS Whitelist, dan Riwayat Kendaraan.
          </p>
        </div>
      </div>
    </div>

    <!-- Users Table -->
    <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-xs overflow-hidden">
      <table class="w-full text-left text-xs">
        <thead class="bg-gray-50 dark:bg-gray-900/60 text-gray-500 dark:text-gray-400 font-semibold border-b border-gray-200 dark:border-gray-700">
          <tr>
            <th class="px-6 py-3.5">Pengguna & Nama Lengkap</th>
            <th class="px-6 py-3.5">Email</th>
            <th class="px-6 py-3.5">Level Hak Akses</th>
            <th class="px-6 py-3.5">Status Akun</th>
            <th class="px-6 py-3.5 text-right">Aksi</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-100 dark:divide-gray-700">
          <tr v-for="u in users" :key="u.id" class="hover:bg-gray-50/50 dark:hover:bg-gray-750/50 transition">
            <td class="px-6 py-4 font-semibold text-gray-900 dark:text-white">
              <div class="flex items-center space-x-3">
                <div class="w-8 h-8 rounded-full bg-[#173B65]/10 dark:bg-blue-900/30 text-[#173B65] dark:text-blue-300 flex items-center justify-center font-bold text-xs uppercase">
                  {{ u.username.substring(0, 2) }}
                </div>
                <div>
                  <p class="font-bold text-gray-900 dark:text-white">{{ u.username }}</p>
                  <p class="text-[11px] text-gray-500 dark:text-gray-400 font-normal">{{ u.full_name || 'Petugas Gerbang' }}</p>
                </div>
              </div>
            </td>
            <td class="px-6 py-4 font-mono text-gray-600 dark:text-gray-300">
              {{ u.email }}
            </td>
            <td class="px-6 py-4">
              <span
                v-if="u.is_superuser || u.roles.some(r => r.name === 'Administrator')"
                class="px-2.5 py-1 rounded text-[10px] font-bold bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800"
              >
                👑 Super Admin
              </span>
              <span
                v-else
                class="px-2.5 py-1 rounded text-[10px] font-bold bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800"
              >
                🛡️ Admin (Operator Pos Satpam)
              </span>
            </td>
            <td class="px-6 py-4">
              <span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 uppercase">
                Aktif
              </span>
            </td>
            <td class="px-6 py-4 text-right">
              <button
                @click="deleteUser(u)"
                class="p-1.5 text-gray-400 hover:text-rose-600 dark:hover:text-rose-400 rounded hover:bg-rose-50 dark:hover:bg-rose-950/40 transition"
                title="Hapus Pengguna"
              >
                <Trash2 class="w-4 h-4" />
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Modal Tambah Pengguna Baru -->
    <div
      v-if="showModal"
      class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs"
    >
      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl w-full max-w-md shadow-2xl p-6 space-y-4">
        <div class="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
          <div>
            <h3 class="font-black text-base text-gray-900 dark:text-white">Tambah Pengguna Baru</h3>
            <p class="text-xs text-gray-500 dark:text-gray-400">Buat akun untuk Super Admin atau Petugas Pos Satpam</p>
          </div>
          <button @click="showModal = false" class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">
            <X class="w-5 h-5" />
          </button>
        </div>

        <form @submit.prevent="submitUser" class="space-y-3.5 text-xs">
          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Username Login</label>
            <input
              v-model="formData.username"
              required
              type="text"
              placeholder="Contoh: satpam_depan"
              class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
          </div>

          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Password</label>
            <input
              v-model="formData.password"
              required
              type="password"
              placeholder="••••••••"
              class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
          </div>

          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Nama Lengkap</label>
            <input
              v-model="formData.full_name"
              type="text"
              placeholder="Contoh: Budi Santoso"
              class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
          </div>

          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Alamat Email</label>
            <input
              v-model="formData.email"
              required
              type="email"
              placeholder="satpam@uns.ac.id"
              class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
          </div>

          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Level Hak Akses (Role)</label>
            <select
              v-model="formData.role"
              class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            >
              <option value="Operator">Admin (Operator Pos Satpam Gate UNS)</option>
              <option value="Administrator">Super Admin (Master Control)</option>
            </select>
          </div>

          <div class="flex items-center justify-end space-x-2 pt-3 border-t border-gray-100 dark:border-gray-700">
            <button
              type="button"
              @click="showModal = false"
              class="px-4 py-2 rounded-lg text-xs font-semibold text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              Batal
            </button>
            <button
              type="submit"
              :disabled="submitting"
              class="px-5 py-2 rounded-lg text-xs font-bold bg-[#173B65] hover:bg-[#0f2942] text-white shadow-xs"
            >
              {{ submitting ? 'Menyimpan...' : 'Buat Akun' }}
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>
