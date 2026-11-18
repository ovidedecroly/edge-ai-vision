<script setup lang="ts">
import { ref, onMounted } from 'vue'
import apiClient from '@/api/client'
import type { AuditLog } from '@/types'
import { ShieldCheck, Clock, User, Filter, RefreshCw } from 'lucide-vue-next'

const logs = ref<AuditLog[]>([])
const loading = ref(true)

async function fetchLogs() {
  loading.value = true
  try {
    const res = await apiClient.get<AuditLog[]>('/audit-logs')
    logs.value = res.data
  } catch (err) {
    console.error('Failed to fetch audit logs', err)
  } finally {
    loading.value = false
  }
}

onMounted(fetchLogs)
</script>

<template>
  <div class="space-y-6">
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <div class="flex items-center space-x-2">
          <h2 class="text-xl font-black text-gray-900 dark:text-white tracking-tight">Audit Log Keamanan & Palang</h2>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-[#173B65]/10 dark:bg-blue-900/40 text-[#173B65] dark:text-blue-300 border border-[#173B65]/20">
            Super Admin
          </span>
        </div>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Catatan riwayat aktivitas operasional palang, aktivasi model AI, dan perintah pembukaan darurat
        </p>
      </div>

      <button
        @click="fetchLogs"
        :disabled="loading"
        class="px-3 py-2 rounded-lg text-xs font-semibold bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-750 transition flex items-center space-x-1.5 shadow-xs"
      >
        <RefreshCw class="w-3.5 h-3.5 text-gray-500" :class="{ 'animate-spin': loading }" />
        <span>Segarkan Log</span>
      </button>
    </div>

    <!-- Audit Logs Table -->
    <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-xs overflow-hidden">
      <table class="w-full text-left text-xs">
        <thead class="bg-gray-50 dark:bg-gray-900/60 text-gray-500 dark:text-gray-400 font-semibold border-b border-gray-200 dark:border-gray-700">
          <tr>
            <th class="px-6 py-3.5">Waktu Kejadian</th>
            <th class="px-6 py-3.5">Operator</th>
            <th class="px-6 py-3.5">Aksi Operasi</th>
            <th class="px-6 py-3.5">Objek Target</th>
            <th class="px-6 py-3.5">Keterangan</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-100 dark:divide-gray-700">
          <tr v-for="log in logs" :key="log.id" class="hover:bg-gray-50/50 dark:hover:bg-gray-750/50 transition">
            <td class="px-6 py-3.5 font-mono text-[11px] text-gray-500 dark:text-gray-400 tabular-nums">
              {{ new Date(log.timestamp).toLocaleString('id-ID') }}
            </td>
            <td class="px-6 py-3.5 font-semibold text-gray-900 dark:text-white">
              {{ log.username }}
            </td>
            <td class="px-6 py-3.5">
              <span class="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-[#173B65]/10 dark:bg-blue-900/40 text-[#173B65] dark:text-blue-300">
                {{ log.action }}
              </span>
            </td>
            <td class="px-6 py-3.5 capitalize text-gray-600 dark:text-gray-300">
              {{ log.target_type }}
            </td>
            <td class="px-6 py-3.5 font-mono text-[11px] text-gray-500 dark:text-gray-400 max-w-[300px] truncate" :title="JSON.stringify(log.details)">
              {{ JSON.stringify(log.details) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
