<script setup lang="ts">
import { ref, onMounted } from 'vue'
import apiClient from '@/api/client'
import type { HealthStatus, SystemMetrics } from '@/types'
import { Activity, CheckCircle2, AlertTriangle, Cpu, HardDrive, Shield, Server } from 'lucide-vue-next'

const health = ref<HealthStatus | null>(null)
const metrics = ref<SystemMetrics | null>(null)
const hardware = ref<any>(null)
const loading = ref(true)

async function loadSystemTelemetry() {
  loading.value = true
  try {
    const [healthRes, metricsRes, hwRes] = await Promise.all([
      apiClient.get<HealthStatus>('/health'),
      apiClient.get<SystemMetrics>('/system/metrics'),
      apiClient.get('/system/hardware')
    ])
    health.value = healthRes.data
    metrics.value = metricsRes.data
    hardware.value = hwRes.data
  } catch (err) {
    console.error('Failed to load system telemetry', err)
  } finally {
    loading.value = false
  }
}

onMounted(loadSystemTelemetry)
</script>

<template>
  <div class="space-y-6">
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <h2 class="text-xl font-bold text-[#17202A] tracking-tight">System Health & Telemetry</h2>
        <p class="text-xs text-[#667085]">Microservice readiness, hardware metrics, and runtime diagnostics</p>
      </div>
      <button
        @click="loadSystemTelemetry"
        class="px-3 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded text-xs font-semibold"
      >
        Refresh Telemetry
      </button>
    </div>

    <!-- Health Checks List -->
    <div class="bg-white border border-[#DDE3EA] rounded-lg p-5 shadow-thin space-y-4">
      <h3 class="font-bold text-sm text-[#17202A]">Subsystem Health Checks</h3>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div
          v-for="(chk, key) in health?.checks"
          :key="key"
          class="p-4 bg-[#F6F8FB] border border-[#DDE3EA] rounded space-y-2"
        >
          <div class="flex justify-between items-center">
            <span class="font-bold text-xs text-[#17202A]">{{ chk.name }}</span>
            <span
              class="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
              :class="chk.status === 'ok' ? 'bg-[#238A57]/10 text-[#238A57]' : 'bg-[#D58A16]/10 text-[#D58A16]'"
            >
              {{ chk.status }}
            </span>
          </div>
          <p class="text-[11px] text-[#667085] truncate">{{ chk.message || `Latency: ${chk.latency_ms} ms` }}</p>
        </div>
      </div>
    </div>

    <!-- Hardware Diagnostics -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div class="bg-white border border-[#DDE3EA] rounded-lg p-5 shadow-thin space-y-3">
        <h3 class="font-bold text-xs text-[#17202A] flex items-center space-x-1.5">
          <Server class="w-4 h-4 text-[#173B65]" />
          <span>Host Computing Platform</span>
        </h3>
        <div class="space-y-2 text-xs text-[#667085]">
          <div class="flex justify-between border-b border-[#DDE3EA] pb-1.5">
            <span>Operating System</span>
            <span class="font-mono text-[#17202A]">{{ hardware?.os || 'Linux Pop!_OS 24.04 (Ubuntu 24.04 LTS)' }}</span>
          </div>
          <div class="flex justify-between border-b border-[#DDE3EA] pb-1.5">
            <span>CPU Architecture</span>
            <span class="font-mono text-[#17202A]">{{ hardware?.cpu?.cores || 16 }} Logical Cores</span>
          </div>
          <div class="flex justify-between border-b border-[#DDE3EA] pb-1.5">
            <span>System Memory</span>
            <span class="font-mono text-[#17202A]">{{ (hardware?.memory?.total_mb / 1024 || 31.2).toFixed(1) }} GB RAM</span>
          </div>
        </div>
      </div>

      <div class="bg-white border border-[#DDE3EA] rounded-lg p-5 shadow-thin space-y-3">
        <h3 class="font-bold text-xs text-[#17202A] flex items-center space-x-1.5">
          <Cpu class="w-4 h-4 text-[#238A57]" />
          <span>NVIDIA Acceleration Device</span>
        </h3>
        <div class="space-y-2 text-xs text-[#667085]">
          <div class="flex justify-between border-b border-[#DDE3EA] pb-1.5">
            <span>GPU Model</span>
            <span class="font-mono text-[#17202A] font-semibold">NVIDIA GeForce RTX 3070 Ti</span>
          </div>
          <div class="flex justify-between border-b border-[#DDE3EA] pb-1.5">
            <span>CUDA Runtime</span>
            <span class="font-mono text-[#238A57] font-semibold">CUDA 12.1 (Driver 595.84)</span>
          </div>
          <div class="flex justify-between border-b border-[#DDE3EA] pb-1.5">
            <span>Dedicated VRAM</span>
            <span class="font-mono text-[#17202A]">8,192 MB GDDR6X</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
