<script setup lang="ts">
import { ref } from 'vue'
import { Settings, Shield, HardDrive, Key, Save, CheckCircle2, Sliders, Bell, Cpu } from 'lucide-vue-next'

const savedMsg = ref('')

const settingsState = ref({
  autoOpenDelaySec: 5,
  minConfidence: 85,
  defaultModel: 'model_1',
  gateTimeoutSec: 15,
  soundAlertEnabled: true,
  autoSnapshotRetentionDays: 30,
  campusSubnet: '192.168.1.0/24'
})

function saveSettings() {
  localStorage.setItem('unsgate_settings', JSON.stringify(settingsState.value))
  savedMsg.value = 'Pengaturan sistem UNSGate berhasil disimpan.'
  setTimeout(() => { savedMsg.value = '' }, 3500)
}
</script>

<template>
  <div class="space-y-6 max-w-4xl">
    <div>
      <div class="flex items-center space-x-2">
        <h2 class="text-xl font-black text-gray-900 dark:text-white tracking-tight">Pengaturan Sistem UNSGate</h2>
        <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-[#173B65]/10 dark:bg-blue-900/40 text-[#173B65] dark:text-blue-300 border border-[#173B65]/20">
          Super Admin
        </span>
      </div>
      <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
        Parameter operasional gerbang kampus, waktu relay palang otomatis, toleransi AI ANPR, dan subnet jaringan
      </p>
    </div>

    <div v-if="savedMsg" class="p-3 bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-300 dark:border-emerald-800 rounded-lg text-emerald-800 dark:text-emerald-300 text-xs font-semibold flex items-center space-x-2">
      <CheckCircle2 class="w-4 h-4 shrink-0" />
      <span>{{ savedMsg }}</span>
    </div>

    <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6 shadow-xs space-y-6">
      <!-- ANPR & Palang Otomatis -->
      <div class="space-y-4">
        <div class="flex items-center space-x-2 border-b border-gray-100 dark:border-gray-700 pb-2">
          <Sliders class="w-4 h-4 text-[#173B65] dark:text-blue-400" />
          <h3 class="font-bold text-sm text-gray-900 dark:text-white">Otomasi Palang & ANPR</h3>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Durasi Buka Palang Otomatis (Detik)</label>
            <input
              v-model.number="settingsState.autoOpenDelaySec"
              type="number"
              min="2"
              max="30"
              class="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white"
            />
            <span class="text-[11px] text-gray-400">Palang akan menutup otomatis setelah durasi ini jika safety loop aman.</span>
          </div>

          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Threshold Akurasi ANPR (%)</label>
            <input
              v-model.number="settingsState.minConfidence"
              type="number"
              min="50"
              max="99"
              class="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white"
            />
            <span class="text-[11px] text-gray-400">Minimal kepastian deteksi sebelum trigger palang otomatis dieksekusi.</span>
          </div>

          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Model Default ANPR</label>
            <select
              v-model="settingsState.defaultModel"
              class="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white"
            >
              <option value="model_1">Model 1 (Akurasi Tinggi - Rekomendasi)</option>
              <option value="model_2">Model 2 (Ultra-Cepat Edge AI)</option>
            </select>
          </div>

          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Timeout Sensor Palang (Detik)</label>
            <input
              v-model.number="settingsState.gateTimeoutSec"
              type="number"
              min="5"
              max="60"
              class="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white"
            />
          </div>
        </div>
      </div>

      <!-- Kamera & Network Section -->
      <div class="space-y-4">
        <div class="flex items-center space-x-2 border-b border-gray-100 dark:border-gray-700 pb-2">
          <HardDrive class="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          <h3 class="font-bold text-sm text-gray-900 dark:text-white">Jaringan & Gateway Streaming</h3>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">WebRTC Gateway (go2rtc)</label>
            <input
              value="http://localhost:1984"
              readonly
              class="w-full px-3 py-2 bg-gray-100 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 rounded-lg font-mono text-gray-700 dark:text-gray-300"
            />
          </div>

          <div>
            <label class="block font-semibold text-gray-700 dark:text-gray-300 mb-1">Subnet LAN Gerbang</label>
            <input
              v-model="settingsState.campusSubnet"
              type="text"
              class="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg font-mono text-gray-900 dark:text-white"
            />
          </div>
        </div>
      </div>

      <div class="pt-4 border-t border-gray-100 dark:border-gray-700 flex justify-end">
        <button
          @click="saveSettings"
          class="px-5 py-2.5 bg-[#173B65] hover:bg-[#0f2942] text-white rounded-lg text-xs font-bold flex items-center space-x-2 shadow-xs transition"
        >
          <Save class="w-4 h-4" />
          <span>Simpan Perubahan</span>
        </button>
      </div>
    </div>
  </div>
</template>
