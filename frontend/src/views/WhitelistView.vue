<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import apiClient from '@/api/client'
import {
  UserCheck,
  Plus,
  Search,
  Filter,
  Trash2,
  Edit2,
  ShieldCheck,
  CheckCircle2,
  Car,
  Bike,
  Sparkles,
  Download,
  Upload
} from 'lucide-vue-next'

const whitelist = ref<any[]>([])
const loading = ref(true)
const searchQuery = ref('')
const selectedCategory = ref('ALL')
const showAddModal = ref(false)
const saving = ref(false)
const feedback = ref('')

const newForm = ref({
  plate_number: '',
  owner_name: '',
  category: 'Dosen / Pengajar',
  unit: 'Fakultas Teknik UNS',
  rfid_uid: '',
  vehicle_type: 'Mobil'
})

const categories = ['ALL', 'Dosen / Pengajar', 'Tenaga Kependidikan', 'Mahasiswa Pascasarjana', 'Tamu Khusus']

async function fetchWhitelist() {
  loading.value = true
  try {
    const res = await apiClient.get('/gate/whitelist')
    whitelist.value = res.data.items || []
  } catch (err) {
    console.error('Failed fetching whitelist', err)
  } finally {
    loading.value = false
  }
}

async function handleAdd() {
  if (!newForm.value.plate_number.trim() || !newForm.value.owner_name.trim()) return
  saving.value = true
  try {
    await apiClient.post('/gate/whitelist', newForm.value)
    showAddModal.value = false
    newForm.value = {
      plate_number: '',
      owner_name: '',
      category: 'Dosen / Pengajar',
      unit: 'Fakultas Teknik UNS',
      rfid_uid: '',
      vehicle_type: 'Mobil'
    }
    feedback.value = 'Data kendaraan sivitas berhasil ditambahkan!'
    setTimeout(() => { feedback.value = '' }, 4000)
    await fetchWhitelist()
  } catch (err) {
    feedback.value = 'Gagal menyimpan kendaraan.'
  } finally {
    saving.value = false
  }
}

const filteredItems = computed(() => {
  return whitelist.value.filter(item => {
    const matchCat = selectedCategory.value === 'ALL' || item.category === selectedCategory.value
    const q = searchQuery.value.toUpperCase().trim()
    const matchSearch = !q ||
      item.plate_number.includes(q) ||
      item.owner_name.toUpperCase().includes(q) ||
      item.unit.toUpperCase().includes(q)
    return matchCat && matchSearch
  })
})

const stats = computed(() => {
  return {
    total: whitelist.value.length,
    dosen: whitelist.value.filter(i => i.category.includes('Dosen')).length,
    tendik: whitelist.value.filter(i => i.category.includes('Kependidikan')).length,
    mhs: whitelist.value.filter(i => i.category.includes('Mahasiswa')).length,
    tamu: whitelist.value.filter(i => i.category.includes('Tamu')).length,
  }
})

onMounted(fetchWhitelist)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] p-5 rounded-2xl shadow-xl">
      <div>
        <div class="flex items-center space-x-3">
          <div class="w-10 h-10 rounded-xl bg-sky-500/10 dark:bg-sky-500/20 text-sky-600 dark:text-sky-400 border border-sky-500/30 flex items-center justify-center font-bold">
            <UserCheck class="w-5 h-5" />
          </div>
          <div>
            <h2 class="text-xl font-black text-gray-900 dark:text-[#E5E9F0] tracking-tight">
              Master Whitelist Sivitas Akademika UNS
            </h2>
            <p class="text-xs text-gray-500 dark:text-[#8B95A7]">
              Database kendaraan resmi dosen, tenaga kependidikan, mahasiswa, dan tamu VIP untuk auto-open barrier portal
            </p>
          </div>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <button
          @click="showAddModal = true"
          class="flex items-center gap-2 px-4 py-2 rounded-xl bg-orange-600 hover:bg-orange-500 text-white font-bold text-xs transition shadow-md shadow-orange-600/20 active:scale-95"
        >
          <Plus class="w-4 h-4" />
          <span>Tambah Sivitas Baru</span>
        </button>
      </div>
    </div>

    <!-- Feedback Banner -->
    <div v-if="feedback" class="p-3.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 text-xs font-semibold flex items-center gap-2">
      <CheckCircle2 class="w-4 h-4 shrink-0" />
      <span>{{ feedback }}</span>
    </div>

    <!-- KPI Metric Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-5 gap-3">
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-3 shadow">
        <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase">Total Sivitas</p>
        <p class="text-xl font-black text-gray-900 dark:text-white font-mono mt-0.5">{{ stats.total }}</p>
      </div>
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-3 shadow">
        <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase">Dosen / Guru Besar</p>
        <p class="text-xl font-black text-sky-600 dark:text-sky-400 font-mono mt-0.5">{{ stats.dosen }}</p>
      </div>
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-3 shadow">
        <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase">Tendik / Pegawai</p>
        <p class="text-xl font-black text-emerald-600 dark:text-emerald-400 font-mono mt-0.5">{{ stats.tendik }}</p>
      </div>
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-3 shadow">
        <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase">Mahasiswa</p>
        <p class="text-xl font-black text-amber-600 dark:text-amber-400 font-mono mt-0.5">{{ stats.mhs }}</p>
      </div>
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-xl p-3 shadow">
        <p class="text-[10px] text-gray-500 dark:text-[#8B95A7] font-semibold uppercase">Tamu VIP Rektorat</p>
        <p class="text-xl font-black text-purple-600 dark:text-purple-400 font-mono mt-0.5">{{ stats.tamu }}</p>
      </div>
    </div>

    <!-- Filter & Search Toolbar -->
    <div class="flex flex-col sm:flex-row justify-between items-center gap-3 bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] p-3.5 rounded-xl">
      <div class="relative w-full sm:w-80">
        <Search class="w-4 h-4 absolute left-3 top-2.5 text-gray-400" />
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Cari nopol, nama, atau fakultas..."
          class="w-full pl-9 pr-3 py-1.5 rounded-lg bg-gray-50 dark:bg-[#1A2438] border border-gray-300 dark:border-[#24304A] text-xs text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:border-orange-500"
        />
      </div>

      <!-- Categories Pills -->
      <div class="flex flex-wrap items-center gap-1.5 w-full sm:w-auto">
        <button
          v-for="cat in categories"
          :key="cat"
          @click="selectedCategory = cat"
          :class="[
            'px-2.5 py-1 rounded-lg text-xs font-semibold transition',
            selectedCategory === cat
              ? 'bg-orange-600 text-white shadow'
              : 'bg-gray-100 dark:bg-[#1A2438] text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-[#24304A]'
          ]"
        >
          {{ cat === 'ALL' ? 'Semua' : cat }}
        </button>
      </div>
    </div>

    <!-- Whitelist Table -->
    <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-2xl shadow-xl overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-gray-50 dark:bg-[#1A2438] text-gray-500 dark:text-[#8B95A7] uppercase font-bold text-[10px] tracking-wider border-b border-gray-200 dark:border-[#24304A]">
            <tr>
              <th class="py-3 px-4">No.</th>
              <th class="py-3 px-4">Plat Nomor</th>
              <th class="py-3 px-4">Nama Pemilik</th>
              <th class="py-3 px-4">Kategori Sivitas</th>
              <th class="py-3 px-4">Fakultas / Unit</th>
              <th class="py-3 px-4">UID Kartu / RFID</th>
              <th class="py-3 px-4">Kendaraan</th>
              <th class="py-3 px-4 text-center">Status Akses</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100 dark:divide-[#24304A]">
            <tr
              v-for="(item, idx) in filteredItems"
              :key="idx"
              class="hover:bg-gray-50 dark:hover:bg-[#1A2438]/50 transition"
            >
              <td class="py-3 px-4 text-gray-400 font-mono">{{ idx + 1 }}</td>
              <td class="py-3 px-4 font-mono font-black text-sm text-gray-900 dark:text-white">
                <span class="px-2.5 py-1 rounded-md bg-gray-100 dark:bg-black/60 border border-gray-300 dark:border-white/10 tracking-widest text-emerald-600 dark:text-emerald-400">
                  {{ item.plate_number }}
                </span>
              </td>
              <td class="py-3 px-4 font-bold text-gray-800 dark:text-[#E5E9F0]">
                {{ item.owner_name }}
              </td>
              <td class="py-3 px-4">
                <span class="px-2 py-0.5 rounded text-[11px] font-semibold bg-sky-500/10 text-sky-700 dark:text-sky-300 border border-sky-500/20">
                  {{ item.category }}
                </span>
              </td>
              <td class="py-3 px-4 text-gray-600 dark:text-gray-300 font-medium">
                {{ item.unit }}
              </td>
              <td class="py-3 px-4 font-mono text-gray-500 dark:text-gray-400">
                {{ item.rfid_uid || '-' }}
              </td>
              <td class="py-3 px-4">
                <div class="flex items-center gap-1.5 text-gray-700 dark:text-gray-300">
                  <Car v-if="item.vehicle_type === 'Mobil' || item.vehicle_type === 'Mobil Dinas'" class="w-3.5 h-3.5 text-orange-500" />
                  <Bike v-else class="w-3.5 h-3.5 text-emerald-500" />
                  <span>{{ item.vehicle_type }}</span>
                </div>
              </td>
              <td class="py-3 px-4 text-center">
                <span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-bold bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                  <CheckCircle2 class="w-3 h-3" />
                  <span>Auto-Open</span>
                </span>
              </td>
            </tr>
            <tr v-if="filteredItems.length === 0">
              <td colspan="8" class="py-8 text-center text-gray-400">
                Tidak ada data kendaraan yang cocok dengan pencarian.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Modal Tambah Sivitas -->
    <div v-if="showAddModal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div class="bg-white dark:bg-[#131C2E] border border-gray-200 dark:border-[#24304A] rounded-2xl w-full max-w-lg p-6 shadow-2xl space-y-4">
        <div class="flex justify-between items-center border-b border-gray-200 dark:border-[#24304A] pb-3">
          <h3 class="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Plus class="w-4 h-4 text-orange-500" />
            <span>Tambah Data Kendaraan Sivitas</span>
          </h3>
          <button @click="showAddModal = false" class="text-gray-400 hover:text-gray-200 text-sm font-bold">&times;</button>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div>
            <label class="block text-gray-600 dark:text-[#8B95A7] font-semibold mb-1">Nomor Plat Polisi *</label>
            <input
              v-model="newForm.plate_number"
              type="text"
              placeholder="Contoh: AD 1234 XY"
              class="w-full px-3 py-2 rounded-lg bg-gray-50 dark:bg-[#1A2438] border border-gray-300 dark:border-[#24304A] text-gray-900 dark:text-white font-mono uppercase font-bold focus:outline-none focus:border-orange-500"
            />
          </div>

          <div>
            <label class="block text-gray-600 dark:text-[#8B95A7] font-semibold mb-1">Nama Pemilik *</label>
            <input
              v-model="newForm.owner_name"
              type="text"
              placeholder="Nama Lengkap & Gelar"
              class="w-full px-3 py-2 rounded-lg bg-gray-50 dark:bg-[#1A2438] border border-gray-300 dark:border-[#24304A] text-gray-900 dark:text-white focus:outline-none focus:border-orange-500"
            />
          </div>

          <div>
            <label class="block text-gray-600 dark:text-[#8B95A7] font-semibold mb-1">Kategori Sivitas</label>
            <select
              v-model="newForm.category"
              class="w-full px-3 py-2 rounded-lg bg-gray-50 dark:bg-[#1A2438] border border-gray-300 dark:border-[#24304A] text-gray-900 dark:text-white focus:outline-none focus:border-orange-500"
            >
              <option value="Dosen / Pengajar">Dosen / Pengajar</option>
              <option value="Tenaga Kependidikan">Tenaga Kependidikan</option>
              <option value="Mahasiswa Pascasarjana">Mahasiswa Pascasarjana</option>
              <option value="Mahasiswa Sarjana">Mahasiswa Sarjana</option>
              <option value="Tamu Khusus">Tamu Khusus / Rektorat</option>
            </select>
          </div>

          <div>
            <label class="block text-gray-600 dark:text-[#8B95A7] font-semibold mb-1">Fakultas / Unit Kerja</label>
            <input
              v-model="newForm.unit"
              type="text"
              placeholder="Contoh: Fakultas Teknik"
              class="w-full px-3 py-2 rounded-lg bg-gray-50 dark:bg-[#1A2438] border border-gray-300 dark:border-[#24304A] text-gray-900 dark:text-white focus:outline-none focus:border-orange-500"
            />
          </div>

          <div>
            <label class="block text-gray-600 dark:text-[#8B95A7] font-semibold mb-1">Jenis Kendaraan</label>
            <select
              v-model="newForm.vehicle_type"
              class="w-full px-3 py-2 rounded-lg bg-gray-50 dark:bg-[#1A2438] border border-gray-300 dark:border-[#24304A] text-gray-900 dark:text-white focus:outline-none focus:border-orange-500"
            >
              <option value="Mobil">Mobil Pribadi</option>
              <option value="Sepeda Motor">Sepeda Motor</option>
              <option value="Mobil Dinas">Mobil Dinas Kampus</option>
              <option value="Bus / Truk Kampus">Bus / Operasional</option>
            </select>
          </div>

          <div>
            <label class="block text-gray-600 dark:text-[#8B95A7] font-semibold mb-1">UID Kartu / E-KTP / KTM</label>
            <input
              v-model="newForm.rfid_uid"
              type="text"
              placeholder="Contoh: UNS-FT-00123"
              class="w-full px-3 py-2 rounded-lg bg-gray-50 dark:bg-[#1A2438] border border-gray-300 dark:border-[#24304A] text-gray-900 dark:text-white font-mono focus:outline-none focus:border-orange-500"
            />
          </div>
        </div>

        <div class="flex justify-end gap-2 pt-3 border-t border-gray-200 dark:border-[#24304A]">
          <button
            @click="showAddModal = false"
            class="px-4 py-2 rounded-xl bg-gray-100 dark:bg-[#1A2438] text-gray-700 dark:text-gray-300 text-xs font-bold hover:bg-gray-200"
          >
            Batal
          </button>
          <button
            @click="handleAdd"
            :disabled="saving"
            class="px-5 py-2 rounded-xl bg-orange-600 hover:bg-orange-500 text-white text-xs font-bold transition shadow-md shadow-orange-600/20 disabled:opacity-50"
          >
            {{ saving ? 'Menyimpan...' : 'Simpan Kendaraan' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
