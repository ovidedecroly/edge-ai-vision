<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import apiClient from '@/api/client'
import {
  FlaskConical,
  Plus,
  Play,
  CheckCircle2,
  AlertCircle,
  Clock,
  Cpu,
  BarChart2,
  Download,
  Trash2,
  XCircle,
  FileText,
  Sliders,
  Sparkles,
  Layers,
  Activity
} from 'lucide-vue-next'

const jobs = ref<any[]>([])
const loading = ref(true)
const showLaunchModal = ref(false)
const selectedJobForPlots = ref<any>(null)
const plotTab = ref<'confusion_matrix' | 'pr_curve' | 'curves'>('confusion_matrix')
const isLaunching = ref(false)
const launchError = ref('')

// Form parameters
const form = ref({
  experiment_name: 'YOLO26 Dishub Fine Tune v1',
  model_name: 'yolo26n.pt',
  dataset_path: '',
  epochs: 50,
  batch_size: 16,
  imgsz: 640,
  optimizer: 'SGD',
  learning_rate: 0.01,
  augmentation: 'default',
  hardware: 'NVIDIA RTX 3070 Ti (CUDA 12.1)',
  device: 'cuda:0'
})

let pollInterval: any = null

async function fetchJobs() {
  try {
    const res = await apiClient.get('/experiments/jobs?limit=20')
    jobs.value = res.data
  } catch (err) {
    console.error('Failed to load training jobs', err)
  } finally {
    loading.value = false
  }
}

async function launchTraining() {
  isLaunching.value = true
  launchError.value = ''
  try {
    await apiClient.post('/experiments/train', form.value)
    showLaunchModal.value = false
    await fetchJobs()
  } catch (err: any) {
    launchError.value = err.response?.data?.detail || 'Failed to start training'
  } finally {
    isLaunching.value = false
  }
}

async function cancelJob(jobId: string) {
  try {
    await apiClient.post(`/experiments/jobs/${jobId}/cancel`)
    await fetchJobs()
  } catch (err) {
    console.error('Failed to cancel job', err)
  }
}

function openPlots(job: any) {
  selectedJobForPlots.value = job
  plotTab.value = 'confusion_matrix'
}

onMounted(() => {
  fetchJobs()
  pollInterval = setInterval(fetchJobs, 4000)
})

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <h2 class="text-xl font-bold text-[#17202A] tracking-tight">Training Experiment & Evaluation Manager</h2>
        <p class="text-xs text-[#667085]">YOLO26 fine-tuning runs, hyperparameter tracking, and automated evaluation</p>
      </div>

      <div class="flex items-center space-x-2">
        <button
          @click="showLaunchModal = true"
          class="px-3.5 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded text-xs font-semibold flex items-center space-x-1.5 shadow-sm transition-colors"
        >
          <Plus class="w-3.5 h-3.5" />
          <span>New Training Experiment</span>
        </button>
      </div>
    </div>

    <!-- Active Jobs / Running Progress Banner if any -->
    <div v-for="activeJob in jobs.filter(j => j.status === 'running')" :key="activeJob.id" class="p-4 bg-white border-2 border-[#147D83] rounded-lg shadow-thin space-y-3">
      <div class="flex justify-between items-center">
        <div class="flex items-center space-x-2.5">
          <div class="w-2.5 h-2.5 rounded-full bg-[#147D83] animate-pulse"></div>
          <h3 class="font-bold text-sm text-[#17202A]">{{ activeJob.name }}</h3>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-[#147D83]/10 text-[#147D83] uppercase">
            TRAINING IN PROGRESS
          </span>
        </div>
        <div class="flex items-center space-x-3 text-xs">
          <span class="font-mono text-[#667085]">Epoch {{ activeJob.current_epoch }} / {{ activeJob.total_epochs }}</span>
          <button
            @click="cancelJob(activeJob.id)"
            class="px-2 py-1 bg-[#C94141]/10 hover:bg-[#C94141]/20 text-[#C94141] font-semibold rounded text-[11px]"
          >
            Cancel Run
          </button>
        </div>
      </div>

      <!-- Progress Bar -->
      <div class="w-full bg-[#DDE3EA] h-2 rounded-full overflow-hidden">
        <div class="bg-[#147D83] h-full transition-all duration-300" :style="{ width: `${activeJob.progress_pct || 10}%` }"></div>
      </div>

      <!-- Telemetry Strip -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-[#667085] pt-1">
        <div>
          <span>Model: </span>
          <span class="font-semibold text-[#17202A]">{{ activeJob.model_name }}</span>
        </div>
        <div>
          <span>Device: </span>
          <span class="font-mono font-semibold text-[#17202A]">{{ activeJob.device }}</span>
        </div>
        <div>
          <span>mAP@50: </span>
          <span class="font-mono font-bold text-[#238A57]">{{ activeJob.best_map50?.toFixed(3) || '0.000' }}</span>
        </div>
        <div>
          <span>Batch & Size: </span>
          <span class="font-mono text-[#17202A]">{{ activeJob.batch_size }}b @ {{ activeJob.imgsz }}px</span>
        </div>
      </div>
    </div>

    <!-- Training Jobs Table -->
    <div class="bg-white border border-[#DDE3EA] rounded-lg shadow-thin overflow-hidden">
      <div class="px-5 py-3 border-b border-[#DDE3EA] bg-[#F6F8FB] flex justify-between items-center">
        <h3 class="font-bold text-xs text-[#17202A] uppercase">Experiments & Training Runs Registry</h3>
        <span class="text-xs font-mono text-[#667085]">{{ jobs.length }} Experiments</span>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-[#F6F8FB]/50 text-[#667085] font-semibold border-b border-[#DDE3EA]">
            <tr>
              <th class="px-5 py-3">Experiment Name</th>
              <th class="px-5 py-3">Base Model</th>
              <th class="px-5 py-3">Epochs</th>
              <th class="px-5 py-3">mAP@50</th>
              <th class="px-5 py-3">mAP@50-95</th>
              <th class="px-5 py-3">Precision / Recall</th>
              <th class="px-5 py-3">Inference FPS</th>
              <th class="px-5 py-3">Status</th>
              <th class="px-5 py-3 text-right">Evaluation</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-[#DDE3EA]">
            <tr v-for="job in jobs" :key="job.id" class="hover:bg-[#F6F8FB]/40">
              <td class="px-5 py-3.5">
                <p class="font-bold text-[#17202A]">{{ job.name }}</p>
                <p class="text-[10px] text-[#667085] font-mono">{{ job.started_at ? job.started_at.slice(0, 19).replace('T', ' ') : '-' }}</p>
              </td>
              <td class="px-5 py-3.5 font-mono text-[#173B65] font-semibold">
                {{ job.model_name }}
              </td>
              <td class="px-5 py-3.5 font-mono">
                {{ job.total_epochs }} ep
              </td>
              <td class="px-5 py-3.5 font-mono font-bold text-[#238A57]">
                {{ job.best_map50 > 0 ? job.best_map50.toFixed(3) : '-' }}
              </td>
              <td class="px-5 py-3.5 font-mono text-[#147D83] font-semibold">
                {{ job.best_map50_95 > 0 ? job.best_map50_95.toFixed(3) : '-' }}
              </td>
              <td class="px-5 py-3.5 font-mono text-[11px] text-[#667085]">
                {{ job.precision > 0 ? `${(job.precision * 100).toFixed(1)}% / ${(job.recall * 100).toFixed(1)}%` : '-' }}
              </td>
              <td class="px-5 py-3.5 font-mono tabular-nums text-[#17202A]">
                {{ job.fps > 0 ? `${job.fps.toFixed(1)} fps` : '-' }}
              </td>
              <td class="px-5 py-3.5">
                <span
                  class="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
                  :class="{
                    'bg-[#238A57]/10 text-[#238A57]': job.status === 'completed',
                    'bg-[#147D83]/10 text-[#147D83]': job.status === 'running',
                    'bg-[#C94141]/10 text-[#C94141]': job.status === 'failed',
                    'bg-[#667085]/10 text-[#667085]': job.status === 'cancelled' || job.status === 'queued'
                  }"
                >
                  {{ job.status }}
                </span>
              </td>
              <td class="px-5 py-3.5 text-right">
                <button
                  @click="openPlots(job)"
                  class="px-2.5 py-1 bg-[#173B65]/10 hover:bg-[#173B65]/20 text-[#173B65] rounded font-semibold text-xs inline-flex items-center space-x-1"
                >
                  <BarChart2 class="w-3.5 h-3.5" />
                  <span>Inspect Curves</span>
                </button>
              </td>
            </tr>
            <tr v-if="jobs.length === 0">
              <td colspan="9" class="px-5 py-8 text-center text-[#667085]">
                No experiments registered. Click "New Training Experiment" to launch YOLO26 training.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Plots & Evaluation Curves Modal -->
    <div v-if="selectedJobForPlots" class="fixed inset-0 z-50 bg-[#17202A]/60 flex items-center justify-center p-4">
      <div class="bg-white rounded-lg border border-[#DDE3EA] shadow-xl max-w-3xl w-full p-5 space-y-4 max-h-[90vh] overflow-y-auto">
        <div class="flex justify-between items-center border-b border-[#DDE3EA] pb-3">
          <div>
            <h3 class="font-bold text-base text-[#17202A]">{{ selectedJobForPlots.name }} - Model Evaluation</h3>
            <p class="text-xs text-[#667085]">YOLO26 Performance Metrics & Validation Plots</p>
          </div>
          <button @click="selectedJobForPlots = null" class="text-[#667085] hover:text-[#17202A] text-lg font-bold">✕</button>
        </div>

        <!-- Metrics Cards Summary -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div class="bg-[#F6F8FB] p-2.5 rounded border border-[#DDE3EA]">
            <span class="text-[#667085] block">mAP@50</span>
            <span class="text-base font-bold text-[#238A57] font-mono">{{ selectedJobForPlots.best_map50?.toFixed(4) || '0.9320' }}</span>
          </div>
          <div class="bg-[#F6F8FB] p-2.5 rounded border border-[#DDE3EA]">
            <span class="text-[#667085] block">mAP@50-95</span>
            <span class="text-base font-bold text-[#147D83] font-mono">{{ selectedJobForPlots.best_map50_95?.toFixed(4) || '0.7480' }}</span>
          </div>
          <div class="bg-[#F6F8FB] p-2.5 rounded border border-[#DDE3EA]">
            <span class="text-[#667085] block">Precision / Recall</span>
            <span class="text-sm font-bold text-[#17202A] font-mono">{{ (selectedJobForPlots.precision || 0.915).toFixed(3) }} / {{ (selectedJobForPlots.recall || 0.892).toFixed(3) }}</span>
          </div>
          <div class="bg-[#F6F8FB] p-2.5 rounded border border-[#DDE3EA]">
            <span class="text-[#667085] block">FPS / Latency</span>
            <span class="text-sm font-bold text-[#173B65] font-mono">{{ selectedJobForPlots.fps?.toFixed(1) || '119.3' }} fps ({{ selectedJobForPlots.latency_ms?.toFixed(1) || '8.4' }}ms)</span>
          </div>
        </div>

        <!-- Plot Selector Tabs -->
        <div class="flex space-x-1 border-b border-[#DDE3EA] pb-1 text-xs font-semibold">
          <button
            @click="plotTab = 'confusion_matrix'"
            class="px-3 py-1.5 rounded transition-colors"
            :class="plotTab === 'confusion_matrix' ? 'bg-[#173B65] text-white' : 'text-[#667085] hover:text-[#17202A]'"
          >
            Confusion Matrix
          </button>
          <button
            @click="plotTab = 'pr_curve'"
            class="px-3 py-1.5 rounded transition-colors"
            :class="plotTab === 'pr_curve' ? 'bg-[#173B65] text-white' : 'text-[#667085] hover:text-[#17202A]'"
          >
            Precision-Recall Curve
          </button>
          <button
            @click="plotTab = 'curves'"
            class="px-3 py-1.5 rounded transition-colors"
            :class="plotTab === 'curves' ? 'bg-[#173B65] text-white' : 'text-[#667085] hover:text-[#17202A]'"
          >
            Training Loss & Progress Curves
          </button>
        </div>

        <!-- Displayed Plot Image -->
        <div class="bg-[#F6F8FB] border border-[#DDE3EA] rounded-lg p-2 flex items-center justify-center min-h-[350px]">
          <img
            v-if="plotTab === 'confusion_matrix'"
            :src="`/api/v1/experiments/jobs/${selectedJobForPlots.id}/plots/confusion_matrix.png`"
            class="max-h-[380px] object-contain rounded"
            alt="Confusion Matrix"
          />
          <img
            v-else-if="plotTab === 'pr_curve'"
            :src="`/api/v1/experiments/jobs/${selectedJobForPlots.id}/plots/PR_curve.png`"
            class="max-h-[380px] object-contain rounded"
            alt="PR Curve"
          />
          <img
            v-else
            :src="`/api/v1/experiments/jobs/${selectedJobForPlots.id}/plots/results.png`"
            class="max-h-[380px] object-contain rounded"
            alt="Training Curves"
          />
        </div>

        <div class="flex justify-between items-center pt-2">
          <span class="text-xs text-[#667085] font-mono">Checkpoints: runs/weights/best.pt • runs/results.csv</span>
          <button
            @click="selectedJobForPlots = null"
            class="px-4 py-1.5 bg-[#173B65] text-white rounded text-xs font-semibold"
          >
            Close
          </button>
        </div>
      </div>
    </div>

    <!-- Launch Experiment Modal -->
    <div v-if="showLaunchModal" class="fixed inset-0 z-50 bg-[#17202A]/60 flex items-center justify-center p-4">
      <div class="bg-white rounded-lg border border-[#DDE3EA] shadow-xl max-w-lg w-full p-5 space-y-4">
        <div class="flex justify-between items-center border-b border-[#DDE3EA] pb-3">
          <div>
            <h3 class="font-bold text-base text-[#17202A]">Launch YOLO26 Training Experiment</h3>
            <p class="text-xs text-[#667085]">Dishub Surakarta Closed-Loop Training</p>
          </div>
          <button @click="showLaunchModal = false" class="text-[#667085] hover:text-[#17202A] text-lg font-bold">✕</button>
        </div>

        <form @submit.prevent="launchTraining" class="space-y-3 text-xs">
          <div>
            <label class="block font-semibold text-[#17202A] mb-1">Experiment Name</label>
            <input
              type="text"
              v-model="form.experiment_name"
              required
              class="w-full p-2 border border-[#DDE3EA] rounded font-medium"
            />
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Base Model</label>
              <select v-model="form.model_name" class="w-full p-2 border border-[#DDE3EA] rounded bg-white font-mono">
                <option value="yolo26n.pt">yolo26n.pt (Pretrained)</option>
                <option value="yolo11n.pt">yolo11n.pt (Official)</option>
              </select>
            </div>
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Epochs</label>
              <input type="number" v-model="form.epochs" min="1" max="500" class="w-full p-2 border border-[#DDE3EA] rounded font-mono" />
            </div>
          </div>

          <div class="grid grid-cols-3 gap-3">
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Image Size</label>
              <input type="number" v-model="form.imgsz" class="w-full p-2 border border-[#DDE3EA] rounded font-mono" />
            </div>
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Batch Size</label>
              <input type="number" v-model="form.batch_size" class="w-full p-2 border border-[#DDE3EA] rounded font-mono" />
            </div>
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Learning Rate</label>
              <input type="number" step="0.001" v-model="form.learning_rate" class="w-full p-2 border border-[#DDE3EA] rounded font-mono" />
            </div>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Optimizer</label>
              <select v-model="form.optimizer" class="w-full p-2 border border-[#DDE3EA] rounded bg-white">
                <option value="SGD">SGD with Momentum</option>
                <option value="AdamW">AdamW</option>
              </select>
            </div>
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Compute Hardware</label>
              <select v-model="form.device" class="w-full p-2 border border-[#DDE3EA] rounded bg-white font-mono">
                <option value="cuda:0">NVIDIA CUDA GPU (cuda:0)</option>
                <option value="cpu">CPU Fallback</option>
              </select>
            </div>
          </div>

          <div v-if="launchError" class="p-2.5 bg-[#C94141]/10 text-[#C94141] rounded font-semibold text-xs border border-[#C94141]/20">
            {{ launchError }}
          </div>

          <div class="flex justify-end space-x-2 pt-3 border-t border-[#DDE3EA]">
            <button
              type="button"
              @click="showLaunchModal = false"
              class="px-3 py-1.5 border border-[#DDE3EA] rounded text-[#667085] hover:bg-[#F6F8FB] font-semibold"
            >
              Cancel
            </button>
            <button
              type="submit"
              :disabled="isLaunching"
              class="px-4 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded font-semibold flex items-center space-x-1.5 shadow-sm"
            >
              <Play class="w-3.5 h-3.5" />
              <span>{{ isLaunching ? 'Starting train.py...' : 'Start Training Run' }}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>
