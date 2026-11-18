<script setup lang="ts">
import { ref, onMounted } from 'vue'
import apiClient from '@/api/client'
import { useCameraStore } from '@/stores/camera'
import type { Camera, CameraHealth } from '@/types'
import {
  Camera as CameraIcon,
  Plus,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Video,
  Activity,
  ShieldCheck,
  Eye
} from 'lucide-vue-next'

const cameraStore = useCameraStore()
const healthData = ref<Record<string, CameraHealth>>({})
const probing = ref(false)
const snapshotPreview = ref<string | null>(null)

async function refreshHealth(camId: string) {
  probing.value = true
  try {
    const res = await apiClient.get<CameraHealth>(`/cameras/${camId}/health`)
    healthData.value[camId] = res.data
  } catch (err) {
    console.error('Failed to probe camera health', err)
  } finally {
    probing.value = false
  }
}

async function loadAllHealth() {
  for (const cam of cameraStore.cameras) {
    await refreshHealth(cam.id)
  }
}

function showSnapshot(camId: string) {
  snapshotPreview.value = `/api/v1/cameras/${camId}/snapshot?t=${Date.now()}`
}

onMounted(async () => {
  await cameraStore.fetchCameras()
  await loadAllHealth()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <h2 class="text-xl font-bold text-[#17202A] tracking-tight">Camera Management</h2>
        <p class="text-xs text-[#667085]">Configured IP video streams and hardware telemetry</p>
      </div>
      <button
        class="px-3 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded text-xs font-semibold flex items-center space-x-1.5 shadow-sm"
      >
        <Plus class="w-3.5 h-3.5" />
        <span>Add New Camera</span>
      </button>
    </div>

    <!-- Camera List Table -->
    <div class="bg-white border border-[#DDE3EA] rounded-lg shadow-thin overflow-hidden">
      <table class="w-full text-left text-xs">
        <thead class="bg-[#F6F8FB] text-[#667085] font-semibold border-b border-[#DDE3EA]">
          <tr>
            <th class="px-6 py-3">Camera Name & Model</th>
            <th class="px-6 py-3">IP Address</th>
            <th class="px-6 py-3">RTSP Stream</th>
            <th class="px-6 py-3">RTSP Health</th>
            <th class="px-6 py-3">ISAPI Health</th>
            <th class="px-6 py-3">Measured FPS</th>
            <th class="px-6 py-3">Actions</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-[#DDE3EA]">
          <tr v-for="cam in cameraStore.cameras" :key="cam.id" class="hover:bg-[#F6F8FB]/50">
            <td class="px-6 py-4">
              <div class="flex items-center space-x-3">
                <div class="w-8 h-8 rounded bg-[#173B65]/10 text-[#173B65] flex items-center justify-center font-bold">
                  <CameraIcon class="w-4 h-4" />
                </div>
                <div>
                  <p class="font-bold text-[#17202A]">{{ cam.name }}</p>
                  <p class="text-[11px] text-[#667085]">{{ cam.model }} • {{ cam.location }}</p>
                </div>
              </div>
            </td>
            <td class="px-6 py-4 font-mono font-medium text-[#17202A]">
              {{ cam.ip_address }}:{{ cam.rtsp_port }}
            </td>
            <td class="px-6 py-4 font-mono text-[11px] text-[#667085] max-w-[200px] truncate" :title="cam.rtsp_main_stream">
              {{ cam.rtsp_main_stream }}
            </td>
            <td class="px-6 py-4">
              <span
                class="px-2 py-0.5 rounded text-[10px] font-semibold uppercase flex items-center space-x-1 w-max"
                :class="healthData[cam.id]?.rtsp_health === 'ok' ? 'bg-[#238A57]/10 text-[#238A57]' : 'bg-[#D58A16]/10 text-[#D58A16]'"
              >
                <CheckCircle2 v-if="healthData[cam.id]?.rtsp_health === 'ok'" class="w-3 h-3" />
                <AlertCircle v-else class="w-3 h-3" />
                <span>{{ healthData[cam.id]?.rtsp_health || 'Checking...' }}</span>
              </span>
            </td>
            <td class="px-6 py-4">
              <span
                class="px-2 py-0.5 rounded text-[10px] font-semibold uppercase flex items-center space-x-1 w-max"
                :class="healthData[cam.id]?.snapshot_health === 'ok' ? 'bg-[#238A57]/10 text-[#238A57]' : 'bg-[#C94141]/10 text-[#C94141]'"
              >
                <CheckCircle2 v-if="healthData[cam.id]?.snapshot_health === 'ok'" class="w-3 h-3" />
                <AlertCircle v-else class="w-3 h-3" />
                <span>{{ healthData[cam.id]?.snapshot_health || 'Checking...' }}</span>
              </span>
            </td>
            <td class="px-6 py-4 font-mono tabular-nums text-[#17202A]">
              {{ healthData[cam.id]?.measured_fps || 25.0 }} fps
            </td>
            <td class="px-6 py-4">
              <div class="flex items-center space-x-2">
                <button
                  @click="refreshHealth(cam.id)"
                  :disabled="probing"
                  title="Probe Health"
                  class="p-1 text-[#173B65] hover:bg-[#F6F8FB] rounded transition-colors"
                >
                  <RefreshCw class="w-4 h-4" :class="{ 'animate-spin': probing }" />
                </button>
                <button
                  @click="showSnapshot(cam.id)"
                  title="View Snapshot"
                  class="p-1 text-[#147D83] hover:bg-[#F6F8FB] rounded transition-colors"
                >
                  <Eye class="w-4 h-4" />
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Snapshot Preview Modal / Box if opened -->
    <div v-if="snapshotPreview" class="bg-white border border-[#DDE3EA] rounded-lg p-4 shadow-thin space-y-3">
      <div class="flex justify-between items-center">
        <h3 class="font-bold text-xs text-[#17202A]">Snapshot Preview (ISAPI Direct)</h3>
        <button @click="snapshotPreview = null" class="text-xs text-[#C94141] hover:underline font-semibold">Close</button>
      </div>
      <div class="max-w-2xl bg-black rounded overflow-hidden">
        <img :src="snapshotPreview" alt="Preview" class="w-full h-auto object-contain" />
      </div>
    </div>
  </div>
</template>
