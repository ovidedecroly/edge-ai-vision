<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import apiClient from '@/api/client'
import type { RecognitionEvent } from '@/types'
import {
  FileText,
  Search,
  CheckCircle2,
  AlertTriangle,
  ZoomIn,
  X,
  ExternalLink,
  Car,
  Clock,
  ShieldCheck,
  Sparkles,
  Filter,
  RefreshCw
} from 'lucide-vue-next'

const events = ref<RecognitionEvent[]>([])
const loading = ref(true)
const searchQuery = ref('')
const selectedEvent = ref<RecognitionEvent | null>(null)
const previewImage = ref<string | null>(null)

// Spotlight demo showcase
const demoSpotlights = ref([
  {
    title: 'Mobil Kampus - Rektorat',
    plate: 'AD 1234 UNS',
    tax: '08.28',
    conf: '96.8%',
    vehicle: 'Toyota Fortuner',
    region: 'Surakarta / UNS',
    gate: 'Gate Masuk Utama',
    img: '/datasets/plates/demo_car_plate.png'
  },
  {
    title: 'Sepeda Motor Sivitas - FKIP',
    plate: 'AD 5678 FK',
    tax: '11.27',
    conf: '94.2%',
    vehicle: 'Honda Vario 160',
    region: 'Surakarta',
    gate: 'Gate Masuk Motor',
    img: '/datasets/plates/demo_motorcycle_plate.png'
  },
  {
    title: 'Mobil Dinas Operasional - FT',
    plate: 'AD 9999 UNS',
    tax: '03.29',
    conf: '98.1%',
    vehicle: 'Toyota HiAce',
    region: 'Surakarta / UNS',
    gate: 'Gate Masuk Timur',
    img: '/datasets/plates/demo_car_plate.png'
  }
])

async function fetchEvents() {
  loading.value = true
  try {
    const [resEvents, resPlates] = await Promise.all([
      apiClient.get<RecognitionEvent[]>('/events').catch(() => ({ data: [] })),
      apiClient.get('/gate/plates?limit=50').catch(() => ({ data: { items: [] } }))
    ])
    const dbEvents = resEvents.data || []
    const gateItems = (resPlates.data.items || []).map((p: any) => ({
      id: p.id,
      camera_id: 'gate-cam-1',
      pipeline_id: p.model_used === 'yolo26_lprnet' || p.model_used === 'Model 2' ? 'Model 2' : 'Model 1',
      timestamp: p.captured_at,
      vehicle_class: p.vehicle_type || 'car',
      vehicle_confidence: 0.95,
      vehicle_bbox: [100, 100, 800, 600],
      plate_text: p.plate_number,
      plate_confidence: (p.confidence || 90) / 100,
      plate_detector_confidence: 0.92,
      authorization_status: p.barrier_action?.includes('TERBUKA') ? 'authorized' : 'unrecognized',
      gate_action: p.barrier_action,
      review_status: 'approved',
      detection_latency_ms: 18.0,
      recognition_latency_ms: p.model_used?.includes('Model 2') ? 8.5 : 23.2,
      total_latency_ms: p.model_used?.includes('Model 2') ? 26.5 : 41.2,
      annotated_frame_path: p.scene_image_url,
      raw_frame_path: p.scene_image_url,
      plate_crop_path: p.plate_image_url,
      model_versions: { "pipeline": p.model_used === 'yolo26_lprnet' || p.model_used === 'Model 2' ? 'Model 2' : 'Model 1' }
    }))

    const seen = new Set()
    const merged = []
    for (const item of [...gateItems, ...dbEvents]) {
      const key = `${item.plate_text}-${item.timestamp}`
      if (!seen.has(key)) {
        seen.add(key)
        merged.push(item)
      }
    }
    events.value = merged
  } catch (err) {
    console.error('Failed to fetch events', err)
  } finally {
    loading.value = false
  }
}

function openDetail(ev: RecognitionEvent) {
  selectedEvent.value = ev
}

function closeDetail() {
  selectedEvent.value = null
}

function openZoom(imgUrl: string) {
  previewImage.value = imgUrl
}

function closeZoom() {
  previewImage.value = null
}

const filteredEvents = computed(() => {
  if (!searchQuery.value) return events.value
  const q = searchQuery.value.toLowerCase()
  return events.value.filter(e =>
    (e.plate_text && e.plate_text.toLowerCase().includes(q)) ||
    (e.vehicle_class && e.vehicle_class.toLowerCase().includes(q)) ||
    (e.gate_action && e.gate_action.toLowerCase().includes(q))
  )
})

onMounted(fetchEvents)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <div class="flex items-center space-x-2">
          <h2 class="text-xl font-black text-gray-900 dark:text-white tracking-tight">Riwayat & Log Kendaraan Gate UNS</h2>
          <span class="px-2 py-0.5 text-[10px] font-bold bg-[#173B65]/10 dark:bg-blue-900/40 text-[#173B65] dark:text-blue-300 rounded-full border border-[#173B65]/20">
            Sistem ANPR UNSGate
          </span>
        </div>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Hasil tangkapan kamera Hikvision, pengenalan plat nomor kendaraan, dan aksi palang barrier gate
        </p>
      </div>

      <div class="flex items-center space-x-2.5">
        <div class="relative">
          <Search class="w-3.5 h-3.5 text-gray-400 absolute left-2.5 top-2.5" />
          <input
            v-model="searchQuery"
            placeholder="Cari plat atau kendaraan..."
            class="pl-8 pr-3 py-1.5 text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-[#173B65]"
          />
        </div>

        <button
          @click="fetchEvents"
          :disabled="loading"
          class="p-2 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700 transition"
          title="Segarkan Data"
        >
          <RefreshCw class="w-3.5 h-3.5" :class="{ 'animate-spin': loading }" />
        </button>
      </div>
    </div>

    <!-- Spotlight Cards (Sample Capture Visual) -->
    <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-4 rounded-xl shadow-xs">
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center space-x-2">
          <Sparkles class="w-4 h-4 text-amber-500" />
          <h3 class="font-bold text-xs uppercase tracking-wider text-gray-800 dark:text-gray-200">
            Sampel Citra Deteksi Plat & Hasil Pembacaan
          </h3>
        </div>
        <span class="text-[11px] text-gray-400">Klik gambar untuk perbesar</span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div
          v-for="(spot, sIdx) in demoSpotlights"
          :key="sIdx"
          class="bg-gray-50 dark:bg-gray-900/80 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden shadow-xs hover:shadow-md transition cursor-pointer group"
          @click="openZoom(spot.img)"
        >
          <!-- Image Box with Overlay -->
          <div class="relative h-44 bg-gray-950 overflow-hidden flex items-center justify-center">
            <img :src="spot.img" :alt="spot.plate" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
            <div class="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/20 flex flex-col justify-between p-3">
              <div class="flex justify-between items-start">
                <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-[#173B65] text-white shadow">
                  {{ spot.vehicle }}
                </span>
                <span class="p-1 rounded bg-black/60 text-white/90 hover:text-white">
                  <ZoomIn class="w-3.5 h-3.5" />
                </span>
              </div>
              <div class="flex justify-between items-end">
                <div>
                  <span class="text-[10px] text-gray-300 block">{{ spot.title }}</span>
                  <span class="text-base font-black font-mono tracking-widest text-white drop-shadow">
                    {{ spot.plate }}
                  </span>
                </div>
                <div class="text-right">
                  <span class="text-[9px] text-gray-300 block">Pajak (MM.YY)</span>
                  <span class="text-xs font-mono font-bold text-amber-300">{{ spot.tax }}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Card Details -->
          <div class="p-3 text-xs space-y-1.5 bg-white dark:bg-gray-800">
            <div class="flex justify-between items-center text-[11px]">
              <span class="text-gray-500 dark:text-gray-400">Wilayah Samsat:</span>
              <span class="font-semibold text-gray-800 dark:text-gray-200">{{ spot.region }}</span>
            </div>
            <div class="flex justify-between items-center text-[11px]">
              <span class="text-gray-500 dark:text-gray-400">Akurasi Pengenalan:</span>
              <span class="font-mono font-bold text-emerald-600 dark:text-emerald-400">{{ spot.conf }}</span>
            </div>
            <div class="pt-1.5 border-t border-gray-100 dark:border-gray-700 flex justify-between items-center">
              <span class="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                <CheckCircle2 class="w-3 h-3" />
                {{ spot.gate }}
              </span>
              <span class="text-[10px] text-blue-600 dark:text-blue-400 font-semibold">
                Lihat Detail →
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Events Table -->
    <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-xs overflow-hidden">
      <div class="px-5 py-3.5 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60 flex justify-between items-center">
        <h3 class="font-bold text-xs text-gray-900 dark:text-white uppercase tracking-wider">
          Log Verifikasi Kamera Gate UNS
        </h3>
        <span class="text-xs text-gray-500 dark:text-gray-400">Total: {{ filteredEvents.length }} Rekaman</span>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-gray-50 dark:bg-gray-900/40 text-gray-500 dark:text-gray-400 font-semibold border-b border-gray-200 dark:border-gray-700">
            <tr>
              <th class="px-4 py-3">Visual Snapshot</th>
              <th class="px-4 py-3">Waktu</th>
              <th class="px-4 py-3">Jenis Kendaraan</th>
              <th class="px-4 py-3">Hasil Plat Nomor</th>
              <th class="px-4 py-3">Model AI</th>
              <th class="px-4 py-3">Akurasi</th>
              <th class="px-4 py-3">Status Izin</th>
              <th class="px-4 py-3">Aksi Palang</th>
              <th class="px-4 py-3 text-center">Aksi</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100 dark:divide-gray-700">
            <tr v-if="filteredEvents.length === 0" class="text-center py-8 text-gray-500">
              <td colspan="9" class="py-12">
                <p class="font-medium text-gray-900 dark:text-white">Tidak ada data rekaman deteksi</p>
                <p class="text-[11px] text-gray-400">Event akan masuk secara realtime saat kamera aktif</p>
              </td>
            </tr>
            <tr
              v-for="ev in filteredEvents"
              :key="ev.id"
              class="hover:bg-gray-50/60 dark:hover:bg-gray-750/50 transition cursor-pointer"
              @click="openDetail(ev)"
            >
              <!-- Thumbnail Visual -->
              <td class="px-4 py-2.5">
                <div
                  class="w-20 h-12 rounded-lg bg-black overflow-hidden border border-gray-200 dark:border-gray-700 relative group/thumb shadow-xs"
                  @click.stop="openZoom(ev.annotated_frame_path || ev.raw_frame_path || '/datasets/plates/demo_car_plate.png')"
                >
                  <img
                    :src="ev.annotated_frame_path || ev.raw_frame_path || '/datasets/plates/demo_car_plate.png'"
                    class="w-full h-full object-cover group-hover/thumb:scale-110 transition"
                  />
                  <div class="absolute inset-0 bg-black/40 opacity-0 group-hover/thumb:opacity-100 flex items-center justify-center transition">
                    <ZoomIn class="w-3.5 h-3.5 text-white" />
                  </div>
                </div>
              </td>
              <td class="px-4 py-3 font-mono text-[11px] text-gray-600 dark:text-gray-300">
                {{ ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString('id-ID') : '-' }}<br />
                <span class="text-[10px] text-gray-400">{{ ev.timestamp ? new Date(ev.timestamp).toLocaleDateString('id-ID') : '' }}</span>
              </td>
              <td class="px-4 py-3 font-semibold text-gray-800 dark:text-gray-200 capitalize">
                {{ ev.vehicle_class || 'Mobil' }}
              </td>
              <td class="px-4 py-3">
                <span class="px-2 py-1 rounded bg-gray-100 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 font-mono font-black text-xs tracking-wider text-gray-900 dark:text-white">
                  {{ ev.plate_text || 'UNRECOGNIZED' }}
                </span>
              </td>
              <td class="px-4 py-3">
                <span class="px-2 py-0.5 rounded text-[10px] font-bold"
                  :class="ev.pipeline_id === 'Model 2' ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300' : 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'">
                  {{ ev.pipeline_id || 'Model 1' }}
                </span>
              </td>
              <td class="px-4 py-3 font-mono font-bold text-emerald-600 dark:text-emerald-400">
                {{ Math.round((ev.plate_confidence || 0.9) * 100) }}%
              </td>
              <td class="px-4 py-3">
                <span
                  class="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
                  :class="ev.authorization_status === 'authorized'
                    ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300'
                    : 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300'"
                >
                  {{ ev.authorization_status === 'authorized' ? 'Sivitas Terdaftar' : 'Tamu / Umum' }}
                </span>
              </td>
              <td class="px-4 py-3">
                <span
                  class="px-2 py-0.5 rounded text-[10px] font-bold"
                  :class="ev.gate_action?.includes('TERBUKA')
                    ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'"
                >
                  {{ ev.gate_action || 'Manual Trigger' }}
                </span>
              </td>
              <td class="px-4 py-3 text-center" @click.stop>
                <button
                  @click="openDetail(ev)"
                  class="px-2.5 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 rounded-lg text-xs font-semibold transition"
                >
                  Detail
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Detail Modal -->
    <div
      v-if="selectedEvent"
      class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs overflow-y-auto"
      @click.self="closeDetail"
    >
      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl w-full max-w-2xl shadow-2xl p-6 space-y-4 my-8">
        <div class="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
          <div>
            <h3 class="font-black text-base text-gray-900 dark:text-white">Detail Verifikasi Kendaraan</h3>
            <p class="text-xs text-gray-500 dark:text-gray-400 font-mono">{{ selectedEvent.id }}</p>
          </div>
          <button @click="closeDetail" class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">
            <X class="w-5 h-5" />
          </button>
        </div>

        <!-- Images Section -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div class="rounded-xl overflow-hidden bg-black aspect-video flex items-center justify-center relative">
            <img
              :src="selectedEvent.annotated_frame_path || selectedEvent.raw_frame_path || '/datasets/plates/demo_car_plate.png'"
              class="w-full h-full object-cover"
            />
            <span class="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 text-white text-[10px] font-bold">
              Kamera Penuh
            </span>
          </div>

          <div class="rounded-xl overflow-hidden bg-black aspect-video flex items-center justify-center relative p-3">
            <img
              :src="selectedEvent.plate_crop_path || selectedEvent.annotated_frame_path || '/datasets/plates/demo_car_plate.png'"
              class="max-w-full max-h-full object-contain"
            />
            <span class="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 text-white text-[10px] font-bold">
              Crop Plat Nomor
            </span>
          </div>
        </div>

        <!-- Meta Details -->
        <div class="grid grid-cols-2 sm:grid-cols-3 gap-2.5 text-xs">
          <div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
            <span class="text-gray-400 text-[10px] uppercase font-semibold block">Plat Terbaca</span>
            <span class="font-mono font-black text-sm text-gray-900 dark:text-white">{{ selectedEvent.plate_text }}</span>
          </div>

          <div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
            <span class="text-gray-400 text-[10px] uppercase font-semibold block">Akurasi AI</span>
            <span class="font-mono font-bold text-emerald-600 dark:text-emerald-400 text-sm">
              {{ Math.round((selectedEvent.plate_confidence || 0.9) * 100) }}%
            </span>
          </div>

          <div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
            <span class="text-gray-400 text-[10px] uppercase font-semibold block">Model AI</span>
            <span class="font-semibold text-gray-900 dark:text-white">{{ selectedEvent.pipeline_id || 'Model 1' }}</span>
          </div>

          <div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
            <span class="text-gray-400 text-[10px] uppercase font-semibold block">Aksi Palang</span>
            <span class="font-semibold text-emerald-600 dark:text-emerald-400">{{ selectedEvent.gate_action }}</span>
          </div>

          <div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
            <span class="text-gray-400 text-[10px] uppercase font-semibold block">Waktu Tangkapan</span>
            <span class="font-mono text-gray-800 dark:text-gray-200 text-[11px]">
              {{ selectedEvent.timestamp ? new Date(selectedEvent.timestamp).toLocaleString('id-ID') : '-' }}
            </span>
          </div>

          <div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
            <span class="text-gray-400 text-[10px] uppercase font-semibold block">Latensi Pemrosesan</span>
            <span class="font-mono text-gray-800 dark:text-gray-200 text-[11px]">{{ selectedEvent.total_latency_ms || 32 }} ms</span>
          </div>
        </div>

        <div class="flex justify-end pt-2 border-t border-gray-100 dark:border-gray-700">
          <button
            @click="closeDetail"
            class="px-4 py-2 rounded-lg bg-[#173B65] text-white text-xs font-bold hover:bg-[#0f2942]"
          >
            Tutup
          </button>
        </div>
      </div>
    </div>

    <!-- Zoom Modal -->
    <div
      v-if="previewImage"
      class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-xs"
      @click="closeZoom"
    >
      <div class="relative max-w-4xl max-h-[90vh]">
        <img :src="previewImage" class="max-w-full max-h-[85vh] rounded-xl object-contain shadow-2xl" />
        <button
          @click="closeZoom"
          class="absolute top-3 right-3 p-2 rounded-full bg-black/70 text-white hover:bg-black"
        >
          <X class="w-5 h-5" />
        </button>
      </div>
    </div>
  </div>
</template>
