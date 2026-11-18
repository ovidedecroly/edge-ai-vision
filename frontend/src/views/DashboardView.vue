<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import apiClient from '@/api/client'
import {
  Car,
  Camera,
  CheckCircle2,
  Clock,
  Cpu,
  BarChart3,
  Wifi,
  WifiOff,
  RefreshCw,
  ChevronsLeft,
  ToggleRight,
  Square,
  ScanLine,
  Nfc,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  X,
  Eye,
  Loader2,
  ZoomIn,
  Sparkles,
  Zap,
  ArrowRight,
  UserCheck,
  Plus,
  Search
} from 'lucide-vue-next'

const router = useRouter()

// ── State UNSGate ──
const activeModel = ref('yolo26_ocr')
const availableModels = ref<any[]>([])
const modelSwitching = ref(false)

const summary = ref({
  total_events: 15660,
  today_events: 240,
  vehicles_inside: 86,
  total_nodes: 2,
  online_nodes: 2,
  offline_nodes: 0,
})

const streamTick = ref(Date.now())
let streamTimer: any = null

const gates = ref([
  {
    id: 1,
    title: 'Gate Masuk',
    lane: 'Lane 1 - Gerbang Depan (Ir. Sutami)',
    direction: 'masuk',
    barrierStatus: 'TERTUTUP',
    is_locked_open: false,
    last_plate: 'AD 2914 JG',
    last_card: 'UNS-FT-88219',
    confidence: 94.5,
    timestamp: '',
    image: '/datasets/plates/demo_car_plate.png',
    crop_image: '/datasets/plates/crop_ad2914jg.jpg',
    dailyCount: 142,
    cameraActive: true,
    capturing: false,
    relayLoading: false,
    feedback: '',
    auto_detect: true,
    viewMode: 'live',
    useWebRTC: false, // Default to Fast HTTP Snapshot for robust local/remote display
    cameraModel: 'Hikvision DS-2CD1043G2-LIU (192.168.1.64)'
  },
  {
    id: 2,
    title: 'Gate Keluar',
    lane: 'Lane 2 - Gerbang Depan (Ir. Sutami)',
    direction: 'keluar',
    barrierStatus: 'TERTUTUP',
    is_locked_open: false,
    last_plate: 'K 5919 DL',
    last_card: 'UNS-FKIP-55104',
    confidence: 91.2,
    timestamp: '',
    image: '/datasets/plates/demo_motor_plate.png',
    crop_image: '/datasets/plates/demo_motor_plate.png',
    dailyCount: 98,
    cameraActive: true,
    capturing: false,
    relayLoading: false,
    feedback: '',
    auto_detect: true,
    viewMode: 'live',
    useWebRTC: false,
    cameraModel: 'Hikvision DS-2CD1043G2-LIU (192.168.1.65)'
  }
])

const recentPlates = ref<any[]>([])
const loading = ref(false)
const selectedPlate = ref<any>(null)
const previewImage = ref<string | null>(null)
const showWhitelistModal = ref(false)
const whitelistItems = ref<any[]>([])
const whitelistSearch = ref('')

const currentTime = ref('')
let timer: any = null
let pollTimer: any = null

// Form Whitelist Baru
const newWhitelist = ref({
  plate_number: '',
  owner_name: '',
  category: 'Dosen / Pengajar',
  unit: 'Fakultas Teknik',
  rfid_uid: '',
  vehicle_type: 'Mobil'
})
const savingWhitelist = ref(false)

function updateClock() {
  const now = new Date()
  currentTime.value = now.toLocaleDateString('id-ID', {
    weekday: 'long',
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  }) + ' - ' + now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// ── API Calls ──
async function fetchModels() {
  try {
    const res = await apiClient.get('/gate/model')
    activeModel.value = res.data.active_model
    availableModels.value = res.data.models || []
  } catch (err) {
    console.error('Failed to load gate models', err)
  }
}

async function selectModel(modelId: string) {
  if (modelSwitching.value || activeModel.value === modelId) return
  modelSwitching.value = true
  try {
    const res = await apiClient.post('/gate/model', { model_id: modelId })
    activeModel.value = res.data.active_model
  } catch (err) {
    console.error('Failed to switch model', err)
  } finally {
    modelSwitching.value = false
  }
}

async function fetchStatus() {
  try {
    const [statusRes, summaryRes, platesRes] = await Promise.all([
      apiClient.get('/gate/status'),
      apiClient.get('/gate/summary'),
      apiClient.get('/gate/plates?limit=15')
    ])

    if (statusRes.data.gates) {
      statusRes.data.gates.forEach((g: any) => {
        const localGate = gates.value.find(item => item.direction === g.direction)
        if (localGate) {
          localGate.barrierStatus = g.barrierStatus
          localGate.is_locked_open = g.is_locked_open
          localGate.last_plate = g.last_plate
          localGate.last_card = g.last_card
          localGate.confidence = g.confidence
          localGate.timestamp = g.timestamp
          localGate.image = g.image
          localGate.crop_image = g.crop_image
          localGate.dailyCount = g.dailyCount
        }
      })
    }

    if (summaryRes.data) {
      summary.value = summaryRes.data
    }

    if (platesRes.data.items) {
      recentPlates.value = platesRes.data.items
    }
  } catch (err) {
    // Keep cached
  }
}

async function handleRelay(gate: any, action: 'open' | 'close' | 'toggle') {
  gate.relayLoading = true
  try {
    const res = await apiClient.post('/gate/relay', {
      direction: gate.direction,
      action
    })
    gate.barrierStatus = res.data.barrierStatus
    gate.is_locked_open = res.data.is_locked_open
    gate.feedback = `Relay ${action.toUpperCase()} Berhasil`
    setTimeout(() => { gate.feedback = '' }, 3000)
  } catch (err: any) {
    gate.feedback = 'Gagal mengirim sinyal relay'
  } finally {
    gate.relayLoading = false
  }
}

async function handleCapture(gate: any) {
  gate.capturing = true
  gate.feedback = 'Memproses Deteksi Kendaraan & ANPR...'
  try {
    const res = await apiClient.post(`/gate/capture/${gate.direction}`)
    if (res.data.success) {
      gate.last_plate = res.data.plate
      gate.confidence = res.data.confidence
      gate.barrierStatus = res.data.barrierStatus
      gate.image = res.data.scene_image_url
      gate.crop_image = res.data.plate_image_url
      gate.timestamp = res.data.timestamp
      const vtype = res.data.vehicle_type ? (res.data.vehicle_type === 'car' ? 'Mobil' : 'Motor') : 'Kendaraan'
      gate.feedback = `Terdeteksi: ${vtype} -> ${res.data.plate} (${res.data.confidence}%)`
      await fetchStatus()
    } else {
      gate.feedback = res.data.message || 'Tidak ada kendaraan terdeteksi'
    }
    setTimeout(() => { gate.feedback = '' }, 4500)
  } catch (err) {
    gate.feedback = 'Gagal trigger capture'
  } finally {
    gate.capturing = false
  }
}

async function toggleAutoDetect(gate: any) {
  const newVal = !gate.auto_detect
  try {
    const res = await apiClient.post('/gate/auto-detect', {
      direction: gate.direction,
      enabled: newVal
    })
    gate.auto_detect = res.data.auto_detect
    gate.feedback = res.data.message
    setTimeout(() => { gate.feedback = '' }, 3500)
  } catch (err) {
    gate.feedback = 'Gagal mengubah mode auto-detect'
  }
}

async function runAutoDetectCheck() {
  for (const g of gates.value) {
    if (g.auto_detect && !g.capturing) {
      try {
        const res = await apiClient.post(`/gate/check-auto/${g.direction}`)
        if (res.data && res.data.triggered) {
          g.last_plate = res.data.plate
          g.confidence = res.data.confidence
          g.barrierStatus = res.data.barrierStatus
          g.image = res.data.scene_image_url
          g.crop_image = res.data.plate_image_url
          g.timestamp = res.data.timestamp
          g.dailyCount += 1
          g.feedback = `⚡ AUTO-DETECT: Terdeteksi ${res.data.plate} (${res.data.confidence}%) -> ${res.data.barrierStatus}`
          await fetchStatus()
          setTimeout(() => { g.feedback = '' }, 6000)
        }
      } catch (e) {
        // Silent polling
      }
    }
  }
}

async function fetchWhitelist() {
  try {
    const res = await apiClient.get('/gate/whitelist')
    whitelistItems.value = res.data.items || []
  } catch (err) {
    console.error('Failed to load whitelist', err)
  }
}

async function submitWhitelist() {
  if (!newWhitelist.value.plate_number || !newWhitelist.value.owner_name) return
  savingWhitelist.value = true
  try {
    await apiClient.post('/gate/whitelist', newWhitelist.value)
    newWhitelist.value = {
      plate_number: '',
      owner_name: '',
      category: 'Dosen / Pengajar',
      unit: 'Fakultas Teknik',
      rfid_uid: '',
      vehicle_type: 'Mobil'
    }
    await fetchWhitelist()
  } catch (err) {
    console.error('Failed to save whitelist', err)
  } finally {
    savingWhitelist.value = false
  }
}

function openPlateDetail(plate: any) {
  selectedPlate.value = plate
}

function closePlateDetail() {
  selectedPlate.value = null
}

function openZoom(imgUrl: string) {
  previewImage.value = imgUrl
}

function closeZoom() {
  previewImage.value = null
}

const filteredWhitelist = computed(() => {
  if (!whitelistSearch.value.trim()) return whitelistItems.value
  const q = whitelistSearch.value.toUpperCase()
  return whitelistItems.value.filter(item =>
    item.plate_number.includes(q) ||
    item.owner_name.toUpperCase().includes(q) ||
    item.unit.toUpperCase().includes(q)
  )
})

onMounted(async () => {
  updateClock()
  timer = setInterval(updateClock, 1000)
  streamTimer = setInterval(() => {
    streamTick.value = Date.now()
  }, 1200)
  await Promise.all([fetchModels(), fetchStatus(), fetchWhitelist()])
  pollTimer = setInterval(async () => {
    await fetchStatus()
    await runAutoDetectCheck()
  }, 3500)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
  if (pollTimer) clearInterval(pollTimer)
  if (streamTimer) clearInterval(streamTimer)
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header Pos Satpam UNSGate -->
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] p-5 rounded-2xl shadow-xl shadow-gray-200/50 dark:shadow-black/30 transition-colors">
      <div>
        <div class="flex items-center space-x-3">
          <div class="w-10 h-10 rounded-xl bg-[#f97316] text-white flex items-center justify-center font-black text-sm tracking-wider shadow-lg shadow-[#f97316]/30">
            UNS
          </div>
          <div>
            <h2 class="text-xl font-black text-gray-900 dark:text-[#E5E9F0] tracking-tight flex items-center gap-2">
              UNSGate — Pos Satpam Gerbang Utama
            </h2>
            <p class="text-xs text-gray-500 dark:text-[#8B95A7]">
              Sistem Otomasi Gerbang Kampus & Intelligent ANPR (Jl. Ir. Sutami No. 36A Surakarta)
            </p>
          </div>
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-3">
        <!-- Live Clock -->
        <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-100 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] text-gray-800 dark:text-[#E5E9F0] text-xs font-mono font-medium shadow-xs">
          <Clock class="w-3.5 h-3.5 text-[#f97316]" />
          <span>{{ currentTime }}</span>
        </div>

        <!-- Whitelist Button -->
        <button
          @click="showWhitelistModal = true"
          class="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-[#f97316] hover:bg-[#ea6a0a] text-white text-xs font-bold transition shadow-md shadow-[#f97316]/20 active:scale-95"
        >
          <UserCheck class="w-3.5 h-3.5" />
          <span>Sivitas Terdaftar (Whitelist)</span>
        </button>
      </div>
    </div>

    <!-- AI Model Selector (Dual Engine Choice) -->
    <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] p-4 rounded-xl shadow-xl shadow-gray-200/50 dark:shadow-black/30 transition-colors">
      <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-3">
        <div class="flex items-center gap-2">
          <Zap class="w-4 h-4 text-[#f97316]" />
          <h3 class="font-bold text-xs uppercase tracking-wider text-gray-900 dark:text-[#E5E9F0]">Pilih Model Deteksi & Pembacaan Plat (ANPR Pipeline)</h3>
        </div>
        <span class="text-[11px] text-gray-500 dark:text-[#8B95A7]">Dua model siap pakai — klik untuk langsung mengaktifkan</span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Option 1: Model 1 -->
        <div
          @click="selectModel('yolo26_ocr')"
          :class="[
            'p-4 rounded-xl border transition cursor-pointer relative overflow-hidden',
            activeModel === 'yolo26_ocr'
              ? 'bg-emerald-50/70 dark:bg-[#1A2438] border-emerald-500 dark:border-[#22c55e] shadow-lg shadow-emerald-500/10 dark:shadow-[#22c55e]/10'
              : 'bg-gray-50/50 dark:bg-[#131C2E] border-gray-200 dark:border-[#24304A] hover:border-gray-400 dark:hover:border-[#8B95A7]/40 opacity-80'
          ]"
        >
          <div class="flex justify-between items-start mb-2">
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full" :class="activeModel === 'yolo26_ocr' ? 'bg-[#22c55e] animate-ping' : 'bg-gray-400 dark:bg-gray-500'"></span>
              <h4 class="font-bold text-sm text-gray-900 dark:text-[#E5E9F0]">Model 1</h4>
            </div>
            <span
              class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider"
              :class="activeModel === 'yolo26_ocr' ? 'bg-[#22c55e]/20 text-emerald-700 dark:text-[#22c55e] border border-[#22c55e]/40' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'"
            >
              {{ activeModel === 'yolo26_ocr' ? 'AKTIF DIGUNAKAN' : 'PILIH MODEL' }}
            </span>
          </div>
          <p class="text-xs text-gray-600 dark:text-[#8B95A7] mb-3">
            Model inferensi berakurasi tinggi dengan pengenalan plat presisi dan ekstraksi ganda nomor registrasi kendaraan.
          </p>
          <div class="flex flex-wrap gap-2 text-[10px] font-mono">
            <span class="px-2 py-0.5 rounded bg-emerald-100 dark:bg-black/40 text-emerald-800 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-500/20">Akurasi Nopol: 94.5%</span>
            <span class="px-2 py-0.5 rounded bg-amber-100 dark:bg-black/40 text-amber-800 dark:text-amber-400 border border-amber-300 dark:border-amber-500/20">Ekstraksi Pajak: Ya</span>
            <span class="px-2 py-0.5 rounded bg-blue-100 dark:bg-black/40 text-blue-800 dark:text-blue-400 border border-blue-300 dark:border-blue-500/20">Latensi: ~40 ms</span>
          </div>
        </div>

        <!-- Option 2: Model 2 -->
        <div
          @click="selectModel('yolo26_lprnet')"
          :class="[
            'p-4 rounded-xl border transition cursor-pointer relative overflow-hidden',
            activeModel === 'yolo26_lprnet'
              ? 'bg-orange-50/70 dark:bg-[#1A2438] border-orange-500 dark:border-[#f97316] shadow-lg shadow-orange-500/10 dark:shadow-[#f97316]/10'
              : 'bg-gray-50/50 dark:bg-[#131C2E] border-gray-200 dark:border-[#24304A] hover:border-gray-400 dark:hover:border-[#8B95A7]/40 opacity-80'
          ]"
        >
          <div class="flex justify-between items-start mb-2">
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full" :class="activeModel === 'yolo26_lprnet' ? 'bg-[#f97316] animate-ping' : 'bg-gray-400 dark:bg-gray-500'"></span>
              <h4 class="font-bold text-sm text-gray-900 dark:text-[#E5E9F0]">Model 2</h4>
            </div>
            <span
              class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider"
              :class="activeModel === 'yolo26_lprnet' ? 'bg-[#f97316]/20 text-orange-700 dark:text-[#f97316] border border-[#f97316]/40' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'"
            >
              {{ activeModel === 'yolo26_lprnet' ? 'AKTIF DIGUNAKAN' : 'PILIH MODEL' }}
            </span>
          </div>
          <p class="text-xs text-gray-600 dark:text-[#8B95A7] mb-3">
            Model inferensi ultra-cepat berbasis neural network sekuensial yang dioptimalkan untuk respons seketika palang gerbang.
          </p>
          <div class="flex flex-wrap gap-2 text-[10px] font-mono">
            <span class="px-2 py-0.5 rounded bg-emerald-100 dark:bg-black/40 text-emerald-800 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-500/20">Akurasi Nopol: 91.2%</span>
            <span class="px-2 py-0.5 rounded bg-orange-100 dark:bg-black/40 text-orange-800 dark:text-orange-400 border border-orange-300 dark:border-orange-500/20">Inferensi Cepat: Ya</span>
            <span class="px-2 py-0.5 rounded bg-cyan-100 dark:bg-black/40 text-cyan-800 dark:text-cyan-400 border border-cyan-300 dark:border-cyan-500/20">Latensi: ~25 ms (Ultra-Fast)</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Summary KPI Cards (4 Cards matching AutoGateUNS) -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <!-- 1. Total Event -->
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-4 shadow-xl shadow-gray-200/50 dark:shadow-black/30 transition-colors">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center shrink-0">
            <BarChart3 class="w-5 h-5 text-blue-500 dark:text-blue-400" />
          </div>
          <div class="min-w-0">
            <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider truncate">Total Event</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-[#E5E9F0] font-mono">{{ summary.total_events.toLocaleString() }}</p>
            <p class="text-[10px] text-gray-400 dark:text-[#8B95A7]">Sepanjang waktu</p>
          </div>
        </div>
      </div>

      <!-- 2. Hari Ini -->
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-4 shadow-xl shadow-gray-200/50 dark:shadow-black/30 transition-colors">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-[#f97316]/10 border border-[#f97316]/20 flex items-center justify-center shrink-0">
            <Car class="w-5 h-5 text-[#f97316]" />
          </div>
          <div class="min-w-0">
            <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider truncate">Hari Ini</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-[#E5E9F0] font-mono">{{ summary.today_events }}</p>
            <p class="text-[10px] text-gray-400 dark:text-[#8B95A7]">Sejak 00:00 WIB</p>
          </div>
        </div>
      </div>

      <!-- 3. Di Dalam Kampus -->
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-4 shadow-xl shadow-gray-200/50 dark:shadow-black/30 transition-colors">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-[#22c55e]/10 border border-[#22c55e]/20 flex items-center justify-center shrink-0">
            <Car class="w-5 h-5 text-[#22c55e]" />
          </div>
          <div class="min-w-0">
            <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider truncate">Di Dalam Kampus</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-[#E5E9F0] font-mono">{{ summary.vehicles_inside }}</p>
            <p class="text-[10px] text-gray-400 dark:text-[#8B95A7]">Saat ini berada di UNS</p>
          </div>
        </div>
      </div>

      <!-- 4. Status Gate Online -->
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-4 shadow-xl shadow-gray-200/50 dark:shadow-black/30 transition-colors">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-[#22c55e]/10 border border-[#22c55e]/20 flex items-center justify-center shrink-0">
            <Wifi class="w-5 h-5 text-[#22c55e]" />
          </div>
          <div class="min-w-0">
            <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider truncate">Node Gate Online</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-[#E5E9F0] font-mono">{{ summary.online_nodes }} / 2</p>
            <p class="text-[10px] text-[#22c55e] font-semibold">Lane 1 & 2 Aktif</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Dual Gate Cards (Gate Masuk Lane 1 & Gate Keluar Lane 2) -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div
        v-for="gate in gates"
        :key="gate.id"
        class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-2xl p-5 shadow-xl shadow-gray-200/50 dark:shadow-black/40 flex flex-col justify-between transition-colors"
      >
        <!-- Top Info Header -->
        <div>
          <div class="flex justify-between items-center mb-3">
            <div>
              <div class="flex items-center gap-2">
                <h3 class="text-lg font-bold text-gray-900 dark:text-[#E5E9F0] tracking-tight">{{ gate.title }}</h3>
                <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-gray-100 dark:bg-[#1A2438] text-gray-600 dark:text-[#8B95A7] border border-gray-200 dark:border-[#24304A]">
                  {{ gate.direction.toUpperCase() }}
                </span>
              </div>
              <p class="text-xs text-gray-500 dark:text-[#8B95A7]">{{ gate.lane }}</p>
            </div>

            <!-- Status Palang Badge -->
            <div
              class="px-3 py-1 rounded-full text-xs font-bold font-mono tracking-wider flex items-center gap-1.5 border"
              :class="gate.barrierStatus === 'TERBUKA' ? 'bg-emerald-50 dark:bg-[#22c55e]/15 text-emerald-700 dark:text-[#22c55e] border-emerald-300 dark:border-[#22c55e]/40' : 'bg-rose-50 dark:bg-[#ef4444]/15 text-rose-700 dark:text-[#ef4444] border-rose-300 dark:border-[#ef4444]/40'"
            >
              <span class="w-2 h-2 rounded-full" :class="gate.barrierStatus === 'TERBUKA' ? 'bg-[#22c55e] animate-ping' : 'bg-[#ef4444]'"></span>
              <span>PALANG: {{ gate.barrierStatus }}</span>
            </div>
          </div>

          <!-- Counter Daily -->
          <div class="flex items-center gap-2 mb-3 text-xs text-gray-500 dark:text-[#8B95A7]">
            <Car class="w-3.5 h-3.5 text-[#f97316]" />
            <span class="font-mono font-bold text-gray-900 dark:text-[#E5E9F0]">{{ gate.dailyCount }}</span>
            <span>kendaraan lewat hari ini</span>
          </div>

          <!-- CCTV Feed Preview (Professional Live CCTV Player with HUD) -->
          <div
            class="relative rounded-2xl overflow-hidden bg-black border-2 transition-all duration-300 aspect-video flex items-center justify-center group mb-4"
            :class="gate.barrierStatus === 'TERBUKA' ? 'border-emerald-500 dark:border-[#22c55e] shadow-xl shadow-emerald-500/20' : 'border-gray-200 dark:border-[#24304A]'"
          >
            <!-- 1. Mode Live CCTV -->
            <template v-if="gate.viewMode === 'live'">
              <!-- WebRTC Player (Go2RTC) if enabled for Gate Masuk -->
              <iframe
                v-if="gate.useWebRTC && gate.direction === 'masuk'"
                src="http://localhost:1984/stream.html?src=kamera_hikvision"
                class="w-full h-full border-0 bg-black pointer-events-auto"
                allow="autoplay; fullscreen"
                allowfullscreen
              ></iframe>

              <!-- Fast Snapshot / HTTP Stream (Auto-Refreshing Live Feed) -->
              <img
                v-else
                :src="'/api/v1/gate/stream/' + gate.direction + '?t=' + streamTick"
                :alt="gate.title"
                class="w-full h-full object-cover transition-opacity duration-200"
                @error="gate.cameraActive = false"
                @load="gate.cameraActive = true"
              />
            </template>

            <!-- 2. Mode Snapshot Hasil ANPR -->
            <template v-else>
              <img
                :src="gate.image"
                :alt="gate.title"
                class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              />
            </template>

            <!-- Top Left: Live Status HUD -->
            <div class="absolute top-3 left-3 flex items-center gap-2 pointer-events-none z-10">
              <div class="bg-black/85 backdrop-blur-md px-2.5 py-1 rounded-md text-[11px] font-mono text-gray-200 border border-white/10 shadow flex items-center gap-2">
                <span class="w-2.5 h-2.5 rounded-full" :class="gate.viewMode === 'live' ? 'bg-red-500 animate-ping' : 'bg-emerald-400'"></span>
                <span class="font-bold text-red-400 tracking-wider" v-if="gate.viewMode === 'live'">● LIVE REC</span>
                <span class="font-bold text-emerald-400 tracking-wider" v-else>📸 HASIL ANPR</span>
                <span class="text-gray-500">|</span>
                <span class="text-gray-300 font-sans">{{ currentTime }}</span>
              </div>
            </div>

            <!-- Top Right: View Controls (Switch Live vs Hasil ANPR, WebRTC toggle, Zoom) -->
            <div class="absolute top-3 right-3 flex items-center gap-1.5 z-10">
              <!-- Switch Live vs Hasil ANPR -->
              <div class="bg-black/85 backdrop-blur-md p-0.5 rounded-lg border border-white/10 flex items-center shadow">
                <button
                  @click="gate.viewMode = 'live'"
                  :class="[
                    'px-2 py-0.5 rounded text-[10px] font-bold font-mono transition flex items-center gap-1',
                    gate.viewMode === 'live' ? 'bg-red-600 text-white shadow' : 'text-gray-400 hover:text-white'
                  ]"
                  title="Tampilkan Live CCTV Kamera Realtime"
                >
                  <span class="w-1.5 h-1.5 rounded-full bg-white animate-pulse" v-if="gate.viewMode === 'live'"></span>
                  LIVE
                </button>
                <button
                  @click="gate.viewMode = 'snapshot'"
                  :class="[
                    'px-2 py-0.5 rounded text-[10px] font-bold font-mono transition flex items-center gap-1',
                    gate.viewMode === 'snapshot' ? 'bg-[#2563EB] text-white shadow' : 'text-gray-400 hover:text-white'
                  ]"
                  title="Tampilkan Hasil Deteksi Terakhir (Bounding Box Plat)"
                >
                  HASIL ANPR
                </button>
              </div>

              <!-- WebRTC / ISAPI Toggle for Gate Masuk -->
              <button
                v-if="gate.direction === 'masuk'"
                @click="gate.useWebRTC = !gate.useWebRTC"
                class="bg-black/75 hover:bg-black/90 backdrop-blur-md px-2 py-1 rounded-md text-[10px] font-mono text-amber-300 border border-amber-500/30 shadow transition"
                :title="gate.useWebRTC ? 'Mode: WebRTC (Klik untuk beralih ke Fast HTTP)' : 'Mode: Fast HTTP (Klik untuk beralih ke WebRTC)'"
              >
                {{ gate.useWebRTC ? 'WebRTC' : 'ISAPI' }}
              </button>

              <!-- Zoom Button -->
              <button
                @click="openZoom(gate.viewMode === 'live' ? ('/api/v1/gate/stream/' + gate.direction + '?t=' + streamTick) : gate.image)"
                class="bg-black/75 hover:bg-black/90 backdrop-blur-md p-1.5 rounded-md text-gray-300 hover:text-white border border-white/10 shadow transition"
                title="Perbesar Tampilan Kamera"
              >
                <ZoomIn class="w-3.5 h-3.5" />
              </button>
            </div>

            <!-- Bottom Left: Floating Plate Detected HUD -->
            <div class="absolute bottom-3 left-3 bg-black/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-[#22c55e]/40 flex items-center gap-2.5 shadow-lg z-10">
              <ScanLine class="w-4 h-4 text-[#22c55e]" />
              <div>
                <div class="flex items-center gap-1.5">
                  <span class="text-[9px] text-gray-400 font-sans uppercase">Nopol Terbaca</span>
                  <span class="text-[9px] px-1 py-0.2 rounded bg-emerald-500/20 text-emerald-300 font-mono" v-if="gate.confidence">
                    {{ gate.confidence }}%
                  </span>
                </div>
                <span class="text-base font-black font-mono tracking-widest text-[#22c55e] leading-none block">
                  {{ gate.last_plate }}
                </span>
              </div>
            </div>

            <!-- Bottom Right: Live Crop Plate ROI Mini Preview -->
            <div
              v-if="gate.crop_image"
              @click="openZoom(gate.crop_image)"
              class="absolute bottom-3 right-3 bg-black/90 backdrop-blur-md p-1 rounded-xl border border-white/20 shadow-lg cursor-pointer hover:scale-105 transition group/crop z-10"
              title="Klik untuk perbesar Crop Plat Nomor"
            >
              <img :src="gate.crop_image" class="h-8 w-24 object-cover rounded-lg border border-white/10" alt="ROI Plat" />
              <div class="text-[8px] text-center text-gray-300 font-mono mt-0.5">ROI Plat (Zoom)</div>
            </div>
          </div>

          <!-- Mode Selector: Auto-Detect vs Manual -->
          <div class="flex items-center justify-between p-2.5 rounded-xl bg-gray-50 dark:bg-[#0E1524] border border-gray-200 dark:border-[#24304A] mb-3 transition-colors">
            <div class="flex items-center gap-2">
              <Zap class="w-4 h-4 shrink-0" :class="gate.auto_detect ? 'text-amber-500 animate-pulse' : 'text-gray-400'" />
              <div>
                <p class="text-xs font-bold text-gray-900 dark:text-[#E5E9F0]">
                  {{ gate.auto_detect ? 'Mode Otomatis (Auto-Detect Saat Lewat)' : 'Mode Manual (Pemicu Tombol)' }}
                </p>
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7]">
                  {{ gate.auto_detect ? 'Kamera memantau & otomatis baca plat saat mobil/motor lewat' : 'Hanya memindai saat tombol Scan diklik satpam' }}
                </p>
              </div>
            </div>
            <button
              @click="toggleAutoDetect(gate)"
              :class="[
                'px-2.5 py-1 rounded-lg text-[11px] font-mono font-bold transition flex items-center gap-1.5 border shrink-0',
                gate.auto_detect
                  ? 'bg-amber-50 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300 border-amber-300 dark:border-amber-500/40 hover:bg-amber-100 shadow'
                  : 'bg-gray-100 dark:bg-[#1A2438] text-gray-600 dark:text-gray-400 border-gray-200 dark:border-[#24304A] hover:bg-gray-200 dark:hover:bg-[#24304A]'
              ]"
              :title="gate.auto_detect ? 'Klik untuk beralih ke Mode Manual' : 'Klik untuk mengaktifkan Auto-Detect'"
            >
              <span class="w-2 h-2 rounded-full" :class="gate.auto_detect ? 'bg-amber-500' : 'bg-gray-400'"></span>
              <span>{{ gate.auto_detect ? 'AUTO: ON' : 'AUTO: OFF' }}</span>
            </button>
          </div>

          <!-- Read-only Fields: Plat Nomor & Kartu -->
          <div class="grid grid-cols-2 gap-3 mb-4">
            <div class="bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl px-3 py-2 transition-colors">
              <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider mb-0.5">Plat Nomor</p>
              <div class="flex items-center gap-2">
                <ScanLine class="w-4 h-4 text-[#f97316] shrink-0" />
                <p class="text-base font-black font-mono text-gray-900 dark:text-[#E5E9F0] truncate">{{ gate.last_plate }}</p>
              </div>
            </div>

            <div class="bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl px-3 py-2 transition-colors">
              <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider mb-0.5">Kartu / RFID</p>
              <div class="flex items-center gap-2">
                <Nfc class="w-4 h-4 text-[#22c55e] shrink-0" />
                <p class="text-sm font-bold font-mono text-gray-900 dark:text-[#E5E9F0] truncate">{{ gate.last_card }}</p>
              </div>
            </div>
          </div>

          <!-- Notification feedback if any -->
          <div v-if="gate.feedback" class="mb-3 px-3 py-2 rounded-lg bg-emerald-50 dark:bg-[#22c55e]/15 border border-emerald-200 dark:border-[#22c55e]/30 text-emerald-800 dark:text-[#22c55e] text-xs font-semibold flex items-center justify-between">
            <div class="flex items-center gap-2">
              <ShieldCheck class="w-4 h-4" />
              <span>{{ gate.feedback }}</span>
            </div>
            <button @click="gate.feedback = ''"><X class="w-3.5 h-3.5" /></button>
          </div>
        </div>

        <!-- 4 Action Buttons matching AutoGateUNS -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-3 border-t border-gray-200 dark:border-[#24304A]">
          <button
            @click="handleRelay(gate, 'open')"
            :disabled="gate.relayLoading"
            class="flex items-center justify-center gap-1.5 bg-gray-100 hover:bg-emerald-50 text-emerald-700 dark:bg-[#1A2438] dark:hover:bg-[#24304A] dark:text-[#22c55e] border border-gray-200 dark:border-[#24304A] font-bold py-2 px-2.5 rounded-lg text-xs transition active:scale-95 disabled:opacity-50"
            title="Kirim sinyal buka palang ke relay"
          >
            <ChevronsLeft class="w-4 h-4" />
            <span>Buka Manual</span>
          </button>

          <button
            @click="handleRelay(gate, 'toggle')"
            :disabled="gate.relayLoading"
            class="flex items-center justify-center gap-1.5 bg-gray-100 hover:bg-orange-50 text-orange-700 dark:bg-[#1A2438] dark:hover:bg-[#24304A] dark:text-[#f97316] border border-gray-200 dark:border-[#24304A] font-bold py-2 px-2.5 rounded-lg text-xs transition active:scale-95 disabled:opacity-50"
            title="Tahan palang tetap terbuka"
          >
            <ToggleRight class="w-4 h-4" />
            <span>Selalu Buka</span>
          </button>

          <button
            @click="handleRelay(gate, 'close')"
            :disabled="gate.relayLoading"
            class="flex items-center justify-center gap-1.5 bg-gray-100 hover:bg-rose-50 text-rose-700 dark:bg-[#1A2438] dark:hover:bg-[#24304A] dark:text-[#ef4444] border border-gray-200 dark:border-[#24304A] font-bold py-2 px-2.5 rounded-lg text-xs transition active:scale-95 disabled:opacity-50"
            title="Kirim sinyal tutup palang ke relay"
          >
            <Square class="w-4 h-4" />
            <span>Tutup Manual</span>
          </button>

          <button
            @click="handleCapture(gate)"
            :disabled="gate.capturing"
            class="flex items-center justify-center gap-1.5 bg-[#f97316] hover:bg-[#ea6a0a] text-white font-bold py-2 px-2.5 rounded-lg text-xs transition shadow-md shadow-[#f97316]/30 active:scale-95 disabled:opacity-50"
            title="Ambil gambar dan jalankan AI ANPR"
          >
            <Loader2 v-if="gate.capturing" class="w-4 h-4 animate-spin" />
            <Camera v-else class="w-4 h-4" />
            <span>Scan Manual</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Recent Plates History Table (matching AutoGateUNS) -->
    <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-2xl shadow-xl shadow-gray-200/50 dark:shadow-black/40 overflow-hidden transition-colors">
      <div class="px-6 py-4 border-b border-gray-200 dark:border-[#24304A] bg-gray-50/80 dark:bg-[#1A2438]/50 flex justify-between items-center transition-colors">
        <div>
          <h3 class="font-bold text-sm text-gray-900 dark:text-[#E5E9F0]">Log Riwayat Kendaraan Masuk & Keluar Terkini</h3>
          <p class="text-xs text-gray-500 dark:text-[#8B95A7]">Data deteksi ANPR realtime dari seluruh pos gerbang UNS</p>
        </div>
        <button
          @click="fetchStatus"
          class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white dark:bg-[#1A2438] hover:bg-gray-100 dark:hover:bg-[#24304A] text-gray-700 dark:text-[#E5E9F0] text-xs font-semibold border border-gray-200 dark:border-[#24304A] transition"
        >
          <RefreshCw class="w-3.5 h-3.5" />
          <span>Refresh</span>
        </button>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-gray-50 dark:bg-[#1A2438] text-gray-600 dark:text-[#8B95A7] font-semibold border-b border-gray-200 dark:border-[#24304A]">
            <tr>
              <th class="px-5 py-3">Snapshot</th>
              <th class="px-5 py-3">Waktu</th>
              <th class="px-5 py-3">Plat Nomor</th>
              <th class="px-5 py-3">Arah</th>
              <th class="px-5 py-3">Model AI</th>
              <th class="px-5 py-3">Pemilik / Unit</th>
              <th class="px-5 py-3">Aksi Palang</th>
              <th class="px-5 py-3 text-center">Aksi</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-200 dark:divide-[#24304A]">
            <tr v-if="recentPlates.length === 0" class="text-center py-8 text-gray-500 dark:text-[#8B95A7]">
              <td colspan="8" class="py-12">
                <p class="font-medium text-gray-900 dark:text-[#E5E9F0]">Belum ada data deteksi plat</p>
                <p class="text-[11px]">Klik tombol "Scan Manual" pada salah satu gerbang untuk menjalankan ANPR</p>
              </td>
            </tr>
            <tr
              v-for="p in recentPlates"
              :key="p.id"
              class="hover:bg-gray-50 dark:hover:bg-[#1A2438]/50 transition cursor-pointer"
              @click="openPlateDetail(p)"
            >
              <!-- Thumbnail -->
              <td class="px-5 py-2.5">
                <div
                  class="w-16 h-10 rounded bg-black overflow-hidden border border-gray-200 dark:border-[#24304A] relative group/thumb shadow"
                  @click.stop="openZoom(p.scene_image_url || p.plate_image_url)"
                >
                  <img :src="p.scene_image_url || p.plate_image_url" class="w-full h-full object-cover group-hover/thumb:scale-110 transition" />
                  <div class="absolute inset-0 bg-black/40 opacity-0 group-hover/thumb:opacity-100 flex items-center justify-center transition">
                    <ZoomIn class="w-3.5 h-3.5 text-white" />
                  </div>
                </div>
              </td>
              <td class="px-5 py-3 font-mono text-[11px] text-gray-600 dark:text-[#8B95A7]">
                {{ new Date(p.captured_at).toLocaleTimeString('id-ID') }}<br />
                <span class="text-[10px] text-gray-400 dark:text-gray-500">{{ new Date(p.captured_at).toLocaleDateString('id-ID') }}</span>
              </td>
              <td class="px-5 py-3">
                <span class="font-mono font-black text-sm text-gray-900 dark:text-[#E5E9F0] tracking-wider bg-gray-100 dark:bg-black/40 px-2 py-1 rounded border border-gray-300 dark:border-white/10">
                  {{ p.plate_number }}
                </span>
              </td>
              <td class="px-5 py-3">
                <span
                  class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border"
                  :class="p.direction === 'masuk' ? 'bg-emerald-50 dark:bg-[#22c55e]/15 text-emerald-700 dark:text-[#22c55e] border-emerald-300 dark:border-[#22c55e]/30' : 'bg-orange-50 dark:bg-[#f97316]/15 text-orange-700 dark:text-[#f97316] border-orange-300 dark:border-[#f97316]/30'"
                >
                  {{ p.direction }}
                </span>
              </td>
              <td class="px-5 py-3 text-[11px] text-gray-600 dark:text-[#8B95A7] font-medium">
                {{ p.model_used === 'yolo26_lprnet' || p.model_used === 'Model 2' ? 'Model 2' : 'Model 1' }}
              </td>
              <td class="px-5 py-3 text-xs font-semibold text-gray-900 dark:text-[#E5E9F0]">
                {{ p.owner_name || 'Pengunjung / Tamu Umum' }}
              </td>
              <td class="px-5 py-3">
                <span class="px-2 py-0.5 rounded text-[10px] font-bold text-emerald-700 dark:text-[#22c55e] bg-emerald-50 dark:bg-[#22c55e]/10 border border-emerald-300 dark:border-[#22c55e]/20">
                  {{ p.barrier_action }}
                </span>
              </td>
              <td class="px-5 py-3 text-center" @click.stop>
                <button
                  @click="openPlateDetail(p)"
                  class="px-2.5 py-1 bg-gray-100 dark:bg-[#1A2438] hover:bg-gray-200 dark:hover:bg-[#24304A] text-gray-800 dark:text-[#E5E9F0] rounded text-[11px] font-semibold border border-gray-200 dark:border-[#24304A] transition inline-flex items-center gap-1"
                >
                  <Eye class="w-3 h-3 text-[#f97316]" />
                  <span>Detail</span>
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Modal Detail Kendaraan (PlateDetailModal matching AutoGateUNS) -->
    <Teleport to="body">
      <div
        v-if="selectedPlate"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
        @click.self="closePlateDetail"
      >
        <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-2xl shadow-2xl shadow-black/60 w-full max-w-2xl max-h-[90vh] overflow-y-auto animate-in zoom-in-95 duration-150 transition-colors">
          <div class="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-[#24304A] bg-gray-50/80 dark:bg-[#1A2438]/60 transition-colors">
            <div class="flex items-center gap-3">
              <Car class="w-5 h-5 text-[#f97316]" />
              <h3 class="text-base font-bold text-gray-900 dark:text-[#E5E9F0] tracking-tight">Detail Kendaraan & Hasil ANPR</h3>
            </div>
            <button @click="closePlateDetail" class="p-1 rounded-lg text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-[#24304A] transition">
              <X class="w-5 h-5" />
            </button>
          </div>

          <div class="p-6 space-y-6">
            <!-- Side-by-side Scene & Crop Images -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div class="space-y-1.5">
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider">Tangkapan Kamera Utuh</p>
                <div class="relative rounded-xl overflow-hidden bg-black border border-gray-200 dark:border-[#24304A] aspect-video flex items-center justify-center">
                  <img :src="selectedPlate.scene_image_url || selectedPlate.plate_image_url" class="w-full h-full object-cover" />
                  <button @click="openZoom(selectedPlate.scene_image_url || selectedPlate.plate_image_url)" class="absolute bottom-2 right-2 p-1.5 bg-black/75 rounded text-white text-xs">
                    <ZoomIn class="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <div class="space-y-1.5">
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider">Crop Plat Nomor</p>
                <div class="relative rounded-xl overflow-hidden bg-black border border-gray-200 dark:border-[#24304A] aspect-video flex items-center justify-center p-3">
                  <img :src="selectedPlate.plate_image_url || selectedPlate.scene_image_url" class="max-w-full max-h-full object-contain" />
                  <button @click="openZoom(selectedPlate.plate_image_url || selectedPlate.scene_image_url)" class="absolute bottom-2 right-2 p-1.5 bg-black/75 rounded text-white text-xs">
                    <ZoomIn class="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            <!-- Detail Grid Info -->
            <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div class="bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl px-3 py-2 transition-colors">
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider">Plat Terbaca</p>
                <p class="text-base font-black font-mono text-gray-900 dark:text-[#E5E9F0] mt-0.5">{{ selectedPlate.plate_number }}</p>
              </div>

              <div class="bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl px-3 py-2 transition-colors">
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider">Arah</p>
                <span
                  class="inline-block mt-0.5 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border"
                  :class="selectedPlate.direction === 'masuk' ? 'bg-emerald-50 dark:bg-[#22c55e]/15 text-emerald-700 dark:text-[#22c55e] border-emerald-300 dark:border-[#22c55e]/30' : 'bg-orange-50 dark:bg-[#f97316]/15 text-orange-700 dark:text-[#f97316] border-orange-300 dark:border-[#f97316]/30'"
                >
                  {{ selectedPlate.direction }}
                </span>
              </div>

              <div class="bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl px-3 py-2 transition-colors">
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider">Akurasi AI</p>
                <p class="text-sm font-bold font-mono text-emerald-600 dark:text-[#22c55e] mt-0.5">{{ selectedPlate.confidence }}%</p>
              </div>

              <div class="bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl px-3 py-2 transition-colors">
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider">Model Digunakan</p>
                <p class="text-xs font-semibold text-gray-900 dark:text-[#E5E9F0] mt-0.5">{{ selectedPlate.model_used }}</p>
              </div>

              <div class="bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl px-3 py-2 transition-colors">
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider">RFID / KTM</p>
                <p class="text-xs font-mono font-bold text-[#f97316] mt-0.5">{{ selectedPlate.rfid_uid || '-' }}</p>
              </div>

              <div class="bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl px-3 py-2 transition-colors">
                <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase tracking-wider">Keputusan Barrier</p>
                <p class="text-xs font-bold text-emerald-600 dark:text-[#22c55e] mt-0.5">{{ selectedPlate.barrier_action }}</p>
              </div>
            </div>

            <!-- Pemilik Info Banner -->
            <div class="p-3 rounded-xl bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] flex items-center justify-between transition-colors">
              <div>
                <span class="text-[10px] text-gray-500 dark:text-[#8B95A7] block font-semibold uppercase">Identitas Sivitas Akademika UNS</span>
                <span class="text-sm font-bold text-gray-900 dark:text-[#E5E9F0]">{{ selectedPlate.owner_name }}</span>
              </div>
              <span class="px-2.5 py-1 rounded-full text-[10px] font-bold bg-emerald-50 dark:bg-[#22c55e]/20 text-emerald-700 dark:text-[#22c55e] border border-emerald-300 dark:border-[#22c55e]/40">
                TERVERIFIKASI
              </span>
            </div>

            <div class="flex justify-end pt-2">
              <button
                @click="closePlateDetail"
                class="px-5 py-2 rounded-xl bg-gray-100 dark:bg-[#1A2438] hover:bg-gray-200 dark:hover:bg-[#24304A] text-gray-800 dark:text-[#E5E9F0] text-xs font-bold border border-gray-200 dark:border-[#24304A] transition"
              >
                Tutup
              </button>
            </div>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Modal Whitelist Sivitas UNS -->
    <Teleport to="body">
      <div
        v-if="showWhitelistModal"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
        @click.self="showWhitelistModal = false"
      >
        <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-2xl shadow-2xl shadow-black/60 w-full max-w-3xl max-h-[90vh] overflow-y-auto animate-in zoom-in-95 duration-150 transition-colors">
          <div class="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-[#24304A] bg-gray-50/80 dark:bg-[#1A2438]/60 transition-colors">
            <div class="flex items-center gap-3">
              <UserCheck class="w-5 h-5 text-[#f97316]" />
              <h3 class="text-base font-bold text-gray-900 dark:text-[#E5E9F0] tracking-tight">Daftar Kendaraan Sivitas Akademika UNS (Whitelist)</h3>
            </div>
            <button @click="showWhitelistModal = false" class="p-1 rounded-lg text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-[#24304A] transition">
              <X class="w-5 h-5" />
            </button>
          </div>

          <div class="p-6 space-y-6">
            <!-- Search & Count -->
            <div class="flex justify-between items-center gap-3">
              <div class="relative flex-1">
                <Search class="w-3.5 h-3.5 text-gray-400 dark:text-[#8B95A7] absolute left-3 top-3" />
                <input
                  v-model="whitelistSearch"
                  placeholder="Cari nopol, nama dosen, mahasiswa, atau fakultas..."
                  class="w-full pl-9 pr-3 py-2 bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl text-xs text-gray-900 dark:text-[#E5E9F0] placeholder-gray-400 focus:outline-none focus:border-[#f97316]"
                />
              </div>
              <span class="text-xs text-gray-500 dark:text-[#8B95A7] font-mono">Total: {{ filteredWhitelist.length }}</span>
            </div>

            <!-- Whitelist Table -->
            <div class="border border-gray-200 dark:border-[#24304A] rounded-xl overflow-hidden">
              <table class="w-full text-left text-xs">
                <thead class="bg-gray-50 dark:bg-[#1A2438] text-gray-600 dark:text-[#8B95A7] font-semibold border-b border-gray-200 dark:border-[#24304A]">
                  <tr>
                    <th class="px-4 py-2.5">Plat Nomor</th>
                    <th class="px-4 py-2.5">Nama Pemilik</th>
                    <th class="px-4 py-2.5">Kategori / Unit</th>
                    <th class="px-4 py-2.5">RFID / KTM</th>
                    <th class="px-4 py-2.5">Akses Barrier</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-gray-200 dark:divide-[#24304A]">
                  <tr v-for="(item, idx) in filteredWhitelist" :key="idx" class="hover:bg-gray-50 dark:hover:bg-[#1A2438]/40 transition">
                    <td class="px-4 py-2.5 font-mono font-bold text-gray-900 dark:text-[#E5E9F0]">{{ item.plate_number }}</td>
                    <td class="px-4 py-2.5 font-semibold text-gray-900 dark:text-[#E5E9F0]">{{ item.owner_name }}</td>
                    <td class="px-4 py-2.5 text-gray-500 dark:text-[#8B95A7]">{{ item.category }} ({{ item.unit }})</td>
                    <td class="px-4 py-2.5 font-mono text-[#f97316] font-bold">{{ item.rfid_uid }}</td>
                    <td class="px-4 py-2.5">
                      <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 dark:bg-[#22c55e]/15 text-emerald-700 dark:text-[#22c55e] border border-emerald-300 dark:border-[#22c55e]/30">
                        {{ item.status }}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- Form Tambah Kendaraan Baru -->
            <div class="p-4 bg-gray-50 dark:bg-[#1A2438] border border-gray-200 dark:border-[#24304A] rounded-xl space-y-3 transition-colors">
              <h4 class="font-bold text-xs uppercase tracking-wider text-gray-900 dark:text-[#E5E9F0] flex items-center gap-1.5">
                <Plus class="w-4 h-4 text-[#f97316]" />
                <span>Daftarkan Kendaraan Sivitas Baru</span>
              </h4>
              <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <input
                  v-model="newWhitelist.plate_number"
                  placeholder="Nomor Plat (cth: AD 1234 XY)"
                  class="px-3 py-2 bg-white dark:bg-[#131C2E] border border-gray-300 dark:border-[#24304A] rounded-lg text-xs text-gray-900 dark:text-[#E5E9F0] placeholder-gray-400 focus:outline-none focus:border-[#f97316]"
                />
                <input
                  v-model="newWhitelist.owner_name"
                  placeholder="Nama Lengkap Dosen / Mhs"
                  class="px-3 py-2 bg-white dark:bg-[#131C2E] border border-gray-300 dark:border-[#24304A] rounded-lg text-xs text-gray-900 dark:text-[#E5E9F0] placeholder-gray-400 focus:outline-none focus:border-[#f97316]"
                />
                <input
                  v-model="newWhitelist.unit"
                  placeholder="Fakultas / Unit Kerja"
                  class="px-3 py-2 bg-white dark:bg-[#131C2E] border border-gray-300 dark:border-[#24304A] rounded-lg text-xs text-gray-900 dark:text-[#E5E9F0] placeholder-gray-400 focus:outline-none focus:border-[#f97316]"
                />
              </div>
              <div class="flex justify-end">
                <button
                  @click="submitWhitelist"
                  :disabled="savingWhitelist"
                  class="px-4 py-2 rounded-lg bg-[#f97316] hover:bg-[#ea6a0a] text-white text-xs font-bold transition shadow-md shadow-[#f97316]/20 active:scale-95"
                >
                  Simpan ke Database Whitelist
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Zoom Image Modal -->
    <Teleport to="body">
      <div
        v-if="previewImage"
        class="fixed inset-0 z-[9999] bg-black/90 backdrop-blur-md flex items-center justify-center p-4"
        @click.self="closeZoom"
      >
        <div class="relative max-w-5xl w-full max-h-[90vh] flex flex-col items-center">
          <button @click="closeZoom" class="absolute -top-10 right-0 text-white p-1 rounded-full bg-black/50 hover:bg-black/80">
            <X class="w-6 h-6" />
          </button>
          <img :src="previewImage" class="max-w-full max-h-[85vh] object-contain rounded-xl border border-white/20 shadow-2xl" />
        </div>
      </div>
    </Teleport>
  </div>
</template>
