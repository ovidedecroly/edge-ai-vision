<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import apiClient from '@/api/client'
import {
  DoorClosed,
  DoorOpen,
  Camera,
  Cpu,
  Plus,
  Trash2,
  Edit2,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Zap,
  Activity,
  Server,
  Wifi,
  ShieldCheck,
  X
} from 'lucide-vue-next'

interface GateDevice {
  id: string
  name: string
  type: string // camera_in, camera_out, relay, etc.
  direction: string // masuk, keluar, both
  model: string
  ip_address: string
  http_port: number
  rtsp_port: number
  channel: number
  username: string
  password?: string
  isapi_url?: string
  rtsp_url?: string
  status: string
  is_active: boolean
  last_seen?: string
  latency_ms?: number
}

const devices = ref<GateDevice[]>([])
const loading = ref(true)
const testingId = ref<string | null>(null)
const statusAlert = ref<{ type: 'success' | 'error'; message: string } | null>(null)

// Modal states
const showModal = ref(false)
const isEditing = ref(false)
const modalSubmitting = ref(false)
const modalTesting = ref(false)
const modalTestResult = ref<{ success: boolean; message: string; latency_ms?: number } | null>(null)

const formData = ref<Partial<GateDevice>>({
  name: '',
  type: 'camera_in',
  direction: 'masuk',
  model: 'Hikvision DS-2CD1043G2-LIU',
  ip_address: '192.168.1.64',
  http_port: 80,
  rtsp_port: 554,
  channel: 1,
  username: 'admin',
  password: 'pitUNS@1103',
  isapi_url: '',
  rtsp_url: '',
  is_active: true
})

function showAlert(type: 'success' | 'error', message: string) {
  statusAlert.value = { type, message }
  setTimeout(() => {
    if (statusAlert.value?.message === message) {
      statusAlert.value = null
    }
  }, 5000)
}

async function fetchDevices() {
  loading.value = true
  try {
    const res = await apiClient.get('/gate/devices')
    devices.value = res.data.items || []
  } catch (err: any) {
    showAlert('error', 'Gagal memuat daftar perangkat: ' + (err.message || 'Error'))
  } finally {
    loading.value = false
  }
}

async function testConnection(device: GateDevice) {
  testingId.value = device.id
  try {
    const res = await apiClient.post(`/gate/devices/${device.id}/test`)
    if (res.data.success) {
      showAlert('success', `Koneksi berhasil ke ${device.name} (${device.ip_address}): Latensi ${res.data.latency_ms} ms`)
      device.status = 'online'
      device.latency_ms = res.data.latency_ms
    } else {
      showAlert('error', `Gagal terhubung ke ${device.name} (${device.ip_address}): ${res.data.message}`)
      device.status = 'offline'
    }
  } catch (err: any) {
    showAlert('error', `Error saat menguji ${device.name}: ` + (err.message || 'Timeout'))
  } finally {
    testingId.value = null
  }
}

async function testAllDevices() {
  for (const d of devices.value) {
    await testConnection(d)
  }
}

function openAddModal() {
  isEditing.value = false
  modalTestResult.value = null
  formData.value = {
    name: 'Kamera Entrance Gate Baru',
    type: 'camera_in',
    direction: 'masuk',
    model: 'Hikvision DS-2CD1043G2-LIU',
    ip_address: '192.168.1.64',
    http_port: 80,
    rtsp_port: 554,
    channel: 1,
    username: 'admin',
    password: 'pitUNS@1103',
    isapi_url: '',
    rtsp_url: '',
    is_active: true
  }
  showModal.value = true
}

function openEditModal(d: GateDevice) {
  isEditing.value = true
  modalTestResult.value = null
  formData.value = { ...d }
  showModal.value = true
}

async function testModalConnection() {
  modalTesting.value = true
  modalTestResult.value = null
  try {
    // Quick probe using temporary endpoint or check device
    const targetIp = formData.value.ip_address || '192.168.1.64'
    const targetPort = formData.value.http_port || 80
    // If editing existing device:
    if (isEditing.value && formData.value.id) {
      const res = await apiClient.post(`/gate/devices/${formData.value.id}/test`)
      modalTestResult.value = res.data
    } else {
      // test first
      const res = await apiClient.post('/gate/devices', {
        ...formData.value,
        name: formData.value.name || 'Test Probe'
      })
      if (res.data.success && res.data.device) {
        const testRes = await apiClient.post(`/gate/devices/${res.data.device.id}/test`)
        modalTestResult.value = testRes.data
        // Cleanup probe
        await apiClient.delete(`/gate/devices/${res.data.device.id}`)
      }
    }
  } catch (err: any) {
    modalTestResult.value = {
      success: false,
      message: 'Uji koneksi gagal: ' + (err.response?.data?.detail || err.message)
    }
  } finally {
    modalTesting.value = false
  }
}

async function submitModal() {
  modalSubmitting.value = true
  try {
    if (isEditing.value && formData.value.id) {
      await apiClient.put(`/gate/devices/${formData.value.id}`, formData.value)
      showAlert('success', `Perangkat ${formData.value.name} berhasil diperbarui.`)
    } else {
      await apiClient.post('/gate/devices', formData.value)
      showAlert('success', `Perangkat baru ${formData.value.name} berhasil didaftarkan.`)
    }
    showModal.value = false
    await fetchDevices()
  } catch (err: any) {
    showAlert('error', 'Gagal menyimpan konfigurasi perangkat: ' + (err.response?.data?.detail || err.message))
  } finally {
    modalSubmitting.value = false
  }
}

async function deleteDevice(d: GateDevice) {
  if (!confirm(`Hapus perangkat "${d.name}" (${d.ip_address}) dari sistem UNSGate?`)) {
    return
  }
  try {
    await apiClient.delete(`/gate/devices/${d.id}`)
    showAlert('success', `Perangkat ${d.name} telah dihapus.`)
    await fetchDevices()
  } catch (err: any) {
    showAlert('error', 'Gagal menghapus perangkat: ' + (err.message || 'Error'))
  }
}

async function toggleActive(d: GateDevice) {
  try {
    await apiClient.put(`/gate/devices/${d.id}`, { is_active: !d.is_active })
    d.is_active = !d.is_active
    showAlert('success', `Status ${d.name} diubah menjadi ${d.is_active ? 'Aktif' : 'Non-Aktif'}`)
  } catch (err: any) {
    showAlert('error', 'Gagal memperbarui status: ' + (err.message || 'Error'))
  }
}

onMounted(fetchDevices)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <div class="flex items-center space-x-2">
          <h2 class="text-xl font-black text-gray-900 dark:text-white tracking-tight">Manajemen Perangkat Gate UNS</h2>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-[#173B65]/10 dark:bg-blue-900/40 text-[#173B65] dark:text-blue-300 border border-[#173B65]/20">
            Super Admin
          </span>
        </div>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Kelola kamera ANPR Hikvision DS-2CD1043G2-LIU, controller palang, dan penambahan device tanpa perlu edit backend code.
        </p>
      </div>

      <div class="flex items-center space-x-2.5">
        <button
          @click="testAllDevices"
          :disabled="loading"
          class="px-3 py-2 rounded-lg text-xs font-semibold bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-750 transition flex items-center space-x-1.5 shadow-xs"
        >
          <Activity class="w-3.5 h-3.5 text-blue-500" />
          <span>Ping Semua</span>
        </button>

        <button
          @click="fetchDevices"
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
          <Plus class="w-4 h-4" />
          <span>Tambah Perangkat Baru</span>
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

    <!-- System Highlights Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-xs">
        <div class="flex items-center justify-between">
          <span class="text-[11px] font-semibold text-gray-500 dark:text-gray-400">Total Perangkat Gate</span>
          <Server class="w-4 h-4 text-blue-500" />
        </div>
        <p class="text-2xl font-black text-gray-900 dark:text-white mt-1">{{ devices.length }}</p>
        <span class="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">Terkoneksi ke LAN UNS</span>
      </div>

      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-xs">
        <div class="flex items-center justify-between">
          <span class="text-[11px] font-semibold text-gray-500 dark:text-gray-400">Kamera Masuk (Hikvision)</span>
          <Camera class="w-4 h-4 text-emerald-500" />
        </div>
        <p class="text-2xl font-black text-emerald-600 dark:text-emerald-400 mt-1">
          {{ devices.filter(d => d.type === 'camera_in').length }} Unit
        </p>
        <span class="text-[10px] text-gray-500 dark:text-gray-400">DS-2CD1043G2-LIU (Active)</span>
      </div>

      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-xs">
        <div class="flex items-center justify-between">
          <span class="text-[11px] font-semibold text-gray-500 dark:text-gray-400">Kamera Keluar (Hikvision)</span>
          <Camera class="w-4 h-4 text-blue-500" />
        </div>
        <p class="text-2xl font-black text-blue-600 dark:text-blue-400 mt-1">
          {{ devices.filter(d => d.type === 'camera_out').length }} Unit
        </p>
        <span class="text-[10px] text-gray-500 dark:text-gray-400">DS-2CD1043G2-LIU (Standby)</span>
      </div>

      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-xs">
        <div class="flex items-center justify-between">
          <span class="text-[11px] font-semibold text-gray-500 dark:text-gray-400">Relay Barrier Controller</span>
          <Zap class="w-4 h-4 text-amber-500" />
        </div>
        <p class="text-2xl font-black text-amber-600 dark:text-amber-400 mt-1">
          {{ devices.filter(d => d.type === 'relay').length }} Unit
        </p>
        <span class="text-[10px] text-gray-500 dark:text-gray-400">Trigger Otomatis Palang Aktif</span>
      </div>
    </div>

    <!-- Notice info -->
    <div class="p-3 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded-lg text-xs text-blue-800 dark:text-blue-300 flex items-center space-x-2">
      <ShieldCheck class="w-4 h-4 shrink-0 text-blue-600 dark:text-blue-400" />
      <span>
        <strong>Konfigurasi Dinamis:</strong> Setiap perangkat yang ditambahkan, diubah, atau dihapus langsung tersimpan ke penyimpanan konfigurasi UNSGate tanpa harus menyentuh kode python backend ataupun restart server.
      </span>
    </div>

    <!-- Devices Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div
        v-for="d in devices"
        :key="d.id"
        class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-5 shadow-xs hover:border-gray-300 dark:hover:border-gray-600 transition flex flex-col justify-between"
      >
        <div>
          <!-- Top Row: Icon + Name + Badge + Toggle -->
          <div class="flex items-start justify-between gap-3">
            <div class="flex items-start space-x-3">
              <div
                class="w-10 h-10 rounded-xl flex items-center justify-center font-bold shrink-0 mt-0.5"
                :class="d.type === 'relay'
                  ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
                  : d.direction === 'keluar'
                    ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                    : 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400'"
              >
                <Zap v-if="d.type === 'relay'" class="w-5 h-5" />
                <Camera v-else class="w-5 h-5" />
              </div>

              <div>
                <div class="flex items-center space-x-2">
                  <h3 class="font-bold text-sm text-gray-900 dark:text-white leading-snug">{{ d.name }}</h3>
                </div>
                <p class="text-xs font-mono text-gray-500 dark:text-gray-400 mt-0.5">{{ d.model }}</p>
              </div>
            </div>

            <!-- Type badge & Status -->
            <div class="flex items-center space-x-2 shrink-0">
              <span
                class="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
                :class="d.status === 'online'
                  ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300'
                  : 'bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300'"
              >
                {{ d.status === 'online' ? '● Online' : '○ Offline' }}
              </span>

              <button
                @click="toggleActive(d)"
                :title="d.is_active ? 'Klik untuk non-aktifkan' : 'Klik untuk aktifkan'"
                class="px-2 py-0.5 rounded text-[10px] font-bold transition"
                :class="d.is_active
                  ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 hover:bg-blue-200'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-200'"
              >
                {{ d.is_active ? 'Aktif' : 'Non-Aktif' }}
              </button>
            </div>
          </div>

          <!-- Parameter Table -->
          <div class="mt-4 grid grid-cols-2 gap-2 text-xs bg-gray-50 dark:bg-gray-900/50 p-3 rounded-lg border border-gray-100 dark:border-gray-800">
            <div>
              <span class="text-gray-400 dark:text-gray-500 text-[10px] uppercase font-semibold">Alamat IP & Port</span>
              <p class="font-mono text-gray-800 dark:text-gray-200 font-semibold">{{ d.ip_address }}:{{ d.http_port }}</p>
            </div>
            <div>
              <span class="text-gray-400 dark:text-gray-500 text-[10px] uppercase font-semibold">Tipe & Jalur</span>
              <p class="text-gray-800 dark:text-gray-200 font-medium capitalize">{{ d.type.replace('_', ' ') }} • Gate {{ d.direction }}</p>
            </div>
            <div v-if="d.rtsp_port">
              <span class="text-gray-400 dark:text-gray-500 text-[10px] uppercase font-semibold">Port RTSP & Channel</span>
              <p class="font-mono text-gray-800 dark:text-gray-200 font-medium">{{ d.rtsp_port }} (CH {{ d.channel }})</p>
            </div>
            <div>
              <span class="text-gray-400 dark:text-gray-500 text-[10px] uppercase font-semibold">Otentikasi Device</span>
              <p class="font-mono text-gray-800 dark:text-gray-200 font-medium">User: {{ d.username }}</p>
            </div>
            <div v-if="d.latency_ms !== undefined" class="col-span-2">
              <span class="text-gray-400 dark:text-gray-500 text-[10px] uppercase font-semibold">Hasil Ping Terakhir</span>
              <p class="text-emerald-600 dark:text-emerald-400 font-semibold text-xs">
                Latency: {{ d.latency_ms }} ms • Status Terverifikasi
              </p>
            </div>
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="mt-5 pt-3 border-t border-gray-100 dark:border-gray-700/60 flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <button
              @click="testConnection(d)"
              :disabled="testingId === d.id"
              class="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 dark:hover:bg-emerald-900/60 border border-emerald-200 dark:border-emerald-800 transition flex items-center space-x-1"
            >
              <RefreshCw class="w-3 h-3" :class="{ 'animate-spin': testingId === d.id }" />
              <span>{{ testingId === d.id ? 'Menguji...' : 'Test Koneksi' }}</span>
            </button>

            <button
              @click="openEditModal(d)"
              class="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-gray-50 dark:bg-gray-750 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-600 transition flex items-center space-x-1"
            >
              <Edit2 class="w-3 h-3 text-gray-500" />
              <span>Edit</span>
            </button>
          </div>

          <button
            @click="deleteDevice(d)"
            class="px-2.5 py-1.5 rounded-lg text-xs font-semibold text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/40 border border-transparent hover:border-rose-200 dark:hover:border-rose-800 transition flex items-center space-x-1"
            title="Hapus Perangkat"
          >
            <Trash2 class="w-3 h-3" />
            <span>Hapus</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Modal Form Tambah / Edit Perangkat -->
    <div
      v-if="showModal"
      class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs overflow-y-auto"
    >
      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl w-full max-w-xl shadow-2xl p-6 space-y-5 my-8">
        <div class="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
          <div>
            <h3 class="font-black text-base text-gray-900 dark:text-white">
              {{ isEditing ? 'Edit Konfigurasi Perangkat' : 'Tambah Perangkat Gate Baru' }}
            </h3>
            <p class="text-xs text-gray-500 dark:text-gray-400">
              Konfigurasi IP kamera ANPR, stream RTSP, ISAPI Snapshot, atau Controller Relay
            </p>
          </div>
          <button @click="showModal = false" class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">
            <X class="w-5 h-5" />
          </button>
        </div>

        <form @submit.prevent="submitModal" class="space-y-4 text-xs">
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div class="sm:col-span-2">
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Nama Perangkat</label>
              <input
                v-model="formData.name"
                required
                type="text"
                placeholder="Contoh: Kamera Entrance Gate Masuk - Gerbang Utama"
                class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-[#173B65]"
              />
            </div>

            <div>
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Tipe Perangkat</label>
              <select
                v-model="formData.type"
                class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              >
                <option value="camera_in">Kamera ANPR Gerbang Masuk</option>
                <option value="camera_out">Kamera ANPR Gerbang Keluar</option>
                <option value="relay">Barrier Gate Controller (Relay)</option>
              </select>
            </div>

            <div>
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Arah Jalur</label>
              <select
                v-model="formData.direction"
                class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              >
                <option value="masuk">Masuk Kampus (Inbound)</option>
                <option value="keluar">Keluar Kampus (Outbound)</option>
                <option value="both">Dua Arah (Bidirectional)</option>
              </select>
            </div>

            <div>
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Model Perangkat</label>
              <input
                v-model="formData.model"
                type="text"
                placeholder="Hikvision DS-2CD1043G2-LIU"
                class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              />
            </div>

            <div>
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">IP Address</label>
              <input
                v-model="formData.ip_address"
                required
                type="text"
                placeholder="192.168.1.64"
                class="w-full px-3 py-2 rounded-lg font-mono border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              />
            </div>

            <div>
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">HTTP / ISAPI Port</label>
              <input
                v-model.number="formData.http_port"
                type="number"
                placeholder="80"
                class="w-full px-3 py-2 rounded-lg font-mono border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              />
            </div>

            <div>
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">RTSP Port</label>
              <input
                v-model.number="formData.rtsp_port"
                type="number"
                placeholder="554"
                class="w-full px-3 py-2 rounded-lg font-mono border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              />
            </div>

            <div>
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Username Kamera/Relay</label>
              <input
                v-model="formData.username"
                type="text"
                placeholder="admin"
                class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              />
            </div>

            <div>
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Password</label>
              <input
                v-model="formData.password"
                type="password"
                placeholder="pitUNS@1103"
                class="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              />
            </div>

            <div class="sm:col-span-2">
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">
                Custom ISAPI Snapshot URL <span class="text-gray-400 font-normal">(opsional)</span>
              </label>
              <input
                v-model="formData.isapi_url"
                type="text"
                placeholder="http://192.168.1.64:80/ISAPI/Streaming/channels/1/picture"
                class="w-full px-3 py-2 rounded-lg font-mono text-[11px] border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              />
            </div>

            <div class="sm:col-span-2">
              <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">
                Custom RTSP Stream URL <span class="text-gray-400 font-normal">(opsional)</span>
              </label>
              <input
                v-model="formData.rtsp_url"
                type="text"
                placeholder="rtsp://admin:pitUNS%401103@192.168.1.64:554/Streaming/Channels/101"
                class="w-full px-3 py-2 rounded-lg font-mono text-[11px] border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
              />
            </div>
          </div>

          <!-- Test result in modal -->
          <div
            v-if="modalTestResult"
            class="p-2.5 rounded-lg text-xs"
            :class="modalTestResult.success ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300' : 'bg-rose-50 dark:bg-rose-950/40 text-rose-800 dark:text-rose-300'"
          >
            {{ modalTestResult.message }}
          </div>

          <div class="flex items-center justify-between pt-3 border-t border-gray-100 dark:border-gray-700">
            <button
              type="button"
              @click="testModalConnection"
              :disabled="modalTesting"
              class="px-3 py-2 rounded-lg text-xs font-semibold bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 transition flex items-center space-x-1"
            >
              <RefreshCw class="w-3.5 h-3.5" :class="{ 'animate-spin': modalTesting }" />
              <span>{{ modalTesting ? 'Menguji...' : 'Uji Koneksi IP' }}</span>
            </button>

            <div class="flex items-center space-x-2">
              <button
                type="button"
                @click="showModal = false"
                class="px-4 py-2 rounded-lg text-xs font-semibold text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
                Batal
              </button>
              <button
                type="submit"
                :disabled="modalSubmitting"
                class="px-5 py-2 rounded-lg text-xs font-bold bg-[#173B65] hover:bg-[#0f2942] text-white shadow-xs transition"
              >
                {{ modalSubmitting ? 'Menyimpan...' : 'Simpan Perangkat' }}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>
