<script setup lang="ts">
import { ref, onMounted } from 'vue'
import apiClient from '@/api/client'
import type { ModelArtifact } from '@/types'
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCcw,
  ShieldAlert,
  HardDrive,
  Check,
  AlertCircle,
  Plus,
  Upload,
  ArrowDownCircle,
  Layers,
  Sparkles
} from 'lucide-vue-next'

const models = ref<ModelArtifact[]>([])
const loading = ref(true)
const testResult = ref<any>(null)
const testingModelId = ref<string | null>(null)
const showDeployModal = ref(false)
const isDeploying = ref(false)
const deployError = ref('')

const deployForm = ref({
  name: 'YOLO26n Dishub Surakarta Fine-Tuned',
  version: 'v1.1.0',
  task: 'object_detection',
  source: 'local_trained',
  source_url: '',
  license: 'AGPL-3.0',
  file_path: '/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/runs/test_run_verify/weights/best.pt',
  input_size: 640,
  classes: ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'],
  format: 'pt',
  notes: 'Trained on Dishub Surakarta CCTV dataset with 96 ATCS intersections.'
})

async function fetchModels() {
  loading.value = true
  try {
    const res = await apiClient.get<ModelArtifact[]>('/models')
    models.value = res.data
  } catch (err) {
    console.error('Failed to fetch models', err)
  } finally {
    loading.value = false
  }
}

async function runSmokeTest(modelId: string, device: string = 'cpu') {
  testingModelId.value = modelId
  testResult.value = null
  try {
    const res = await apiClient.post(`/models/${modelId}/test?device=${device}`)
    testResult.value = res.data
  } catch (err: any) {
    testResult.value = {
      passed: false,
      message: err.response?.data?.detail || 'Smoke test failed'
    }
  } finally {
    testingModelId.value = null
  }
}

async function activateModel(modelId: string) {
  try {
    await apiClient.post(`/models/${modelId}/activate`)
    await fetchModels()
  } catch (err) {
    console.error('Failed to activate model', err)
  }
}

async function rollbackModel(modelId: string) {
  try {
    await apiClient.post(`/models/${modelId}/rollback`)
    await fetchModels()
  } catch (err) {
    console.error('Failed to rollback model', err)
  }
}

async function submitDeployModel() {
  isDeploying.value = true
  deployError.value = ''
  try {
    await apiClient.post('/models/import', deployForm.value)
    showDeployModal.value = false
    await fetchModels()
  } catch (err: any) {
    deployError.value = err.response?.data?.detail || 'Failed to deploy model artifact'
  } finally {
    isDeploying.value = false
  }
}

onMounted(fetchModels)
</script>

<template>
  <div class="space-y-6">
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <h2 class="text-xl font-bold text-[#17202A] tracking-tight">Model Registry & Edge Deployment</h2>
        <p class="text-xs text-[#667085]">YOLO26 model checkpoints (.pt, .onnx, TensorRT), validation smoke tests, and rollback</p>
      </div>

      <div class="flex items-center space-x-2">
        <button
          @click="showDeployModal = true"
          class="px-3.5 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded text-xs font-semibold flex items-center space-x-1.5 shadow-sm transition-colors"
        >
          <Upload class="w-3.5 h-3.5" />
          <span>Deploy / Register Model</span>
        </button>
      </div>
    </div>

    <!-- Smoke test result banner if any -->
    <div
      v-if="testResult"
      class="p-4 rounded border text-xs flex items-center justify-between"
      :class="testResult.passed ? 'bg-[#238A57]/10 border-[#238A57]/30 text-[#238A57]' : 'bg-[#C94141]/10 border-[#C94141]/30 text-[#C94141]'"
    >
      <div class="flex items-center space-x-2">
        <CheckCircle2 v-if="testResult.passed" class="w-4 h-4 shrink-0" />
        <AlertCircle v-else class="w-4 h-4 shrink-0" />
        <span class="font-semibold">{{ testResult.message }}</span>
      </div>
      <button @click="testResult = null" class="font-bold underline ml-4">Dismiss</button>
    </div>

    <!-- Models Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div
        v-for="model in models"
        :key="model.id"
        class="bg-white border rounded-lg p-5 shadow-thin space-y-4"
        :class="model.is_active ? 'border-[#238A57] ring-1 ring-[#238A57]' : 'border-[#DDE3EA]'"
      >
        <div class="flex justify-between items-start">
          <div class="flex items-center space-x-3">
            <div
              class="w-10 h-10 rounded flex items-center justify-center font-bold"
              :class="model.is_active ? 'bg-[#238A57]/10 text-[#238A57]' : 'bg-[#173B65]/10 text-[#173B65]'"
            >
              <Cpu class="w-5 h-5" />
            </div>
            <div>
              <div class="flex items-center space-x-2">
                <h3 class="font-bold text-sm text-[#17202A]">{{ model.name }}</h3>
                <span class="text-[10px] font-mono px-1.5 py-0.2 bg-[#F6F8FB] border border-[#DDE3EA] rounded">{{ model.version }}</span>
              </div>
              <p class="text-xs text-[#667085]">{{ model.task }} • {{ model.framework }} • {{ model.file_format.toUpperCase() }}</p>
            </div>
          </div>

          <span
            class="px-2.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide"
            :class="model.is_active ? 'bg-[#238A57] text-white' : 'bg-[#667085]/10 text-[#667085]'"
          >
            {{ model.is_active ? 'ACTIVE MODEL' : 'STANDBY' }}
          </span>
        </div>

        <!-- Trust Badge & Validation -->
        <div class="flex items-center space-x-2 text-xs">
          <span class="text-[#667085]">Status:</span>
          <span
            class="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
            :class="{
              'bg-[#238A57]/10 text-[#238A57]': model.trust_level === 'official' || model.trust_level === 'local_validated',
              'bg-[#D58A16]/10 text-[#D58A16]': model.trust_level === 'community_unverified' || model.trust_level === 'local_trained'
            }"
          >
            {{ model.trust_level }}
          </span>
          <span class="text-[#667085]">• Validation:</span>
          <span class="font-semibold text-[#17202A] capitalize">{{ model.validation_status }}</span>
        </div>

        <!-- Metadata Details -->
        <div class="p-3 bg-[#F6F8FB] rounded space-y-1.5 text-xs text-[#667085]">
          <div class="flex justify-between">
            <span>File Path:</span>
            <span class="font-mono text-[11px] text-[#17202A] truncate max-w-[240px]" :title="model.file_path">
              {{ model.file_path }}
            </span>
          </div>
          <div class="flex justify-between">
            <span>SHA-256:</span>
            <span class="font-mono text-[11px] text-[#17202A] truncate max-w-[240px]" :title="model.checksum_sha256">
              {{ model.checksum_sha256 }}
            </span>
          </div>
          <div class="flex justify-between">
            <span>Size:</span>
            <span class="font-mono text-[#17202A]">{{ (model.model_size_bytes / (1024 * 1024)).toFixed(2) }} MB</span>
          </div>
          <div class="flex justify-between">
            <span>Input Resolution:</span>
            <span class="font-mono text-[#17202A]">{{ model.input_size }} × {{ model.input_size }} px</span>
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="flex items-center space-x-2 pt-2 border-t border-[#DDE3EA]">
          <button
            @click="runSmokeTest(model.id, 'cpu')"
            :disabled="testingModelId === model.id"
            class="px-2.5 py-1.5 bg-[#F6F8FB] hover:bg-[#DDE3EA] text-[#17202A] border border-[#DDE3EA] rounded text-xs font-semibold flex items-center space-x-1 transition-colors disabled:opacity-50"
          >
            <Play class="w-3.5 h-3.5 text-[#173B65]" />
            <span>Smoke Test (CPU)</span>
          </button>

          <button
            @click="runSmokeTest(model.id, 'cuda:0')"
            :disabled="testingModelId === model.id"
            class="px-2.5 py-1.5 bg-[#F6F8FB] hover:bg-[#DDE3EA] text-[#17202A] border border-[#DDE3EA] rounded text-xs font-semibold flex items-center space-x-1 transition-colors disabled:opacity-50"
          >
            <Play class="w-3.5 h-3.5 text-[#238A57]" />
            <span>Smoke Test (CUDA)</span>
          </button>

          <button
            v-if="!model.is_active"
            @click="activateModel(model.id)"
            class="px-3 py-1.5 bg-[#238A57] hover:bg-[#1C6E46] text-white rounded text-xs font-semibold flex items-center space-x-1 ml-auto transition-colors shadow-sm"
          >
            <Check class="w-3.5 h-3.5" />
            <span>Activate</span>
          </button>

          <button
            v-else
            @click="rollbackModel(model.id)"
            class="px-3 py-1.5 bg-[#C94141]/10 hover:bg-[#C94141]/20 text-[#C94141] rounded text-xs font-semibold flex items-center space-x-1 ml-auto transition-colors"
          >
            <RotateCcw class="w-3.5 h-3.5" />
            <span>Rollback</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Deploy Model Modal -->
    <div v-if="showDeployModal" class="fixed inset-0 z-50 bg-[#17202A]/60 flex items-center justify-center p-4">
      <div class="bg-white rounded-lg border border-[#DDE3EA] shadow-xl max-w-lg w-full p-5 space-y-4">
        <div class="flex justify-between items-center border-b border-[#DDE3EA] pb-3">
          <div>
            <h3 class="font-bold text-base text-[#17202A]">Deploy / Register YOLO26 Model</h3>
            <p class="text-xs text-[#667085]">Upload or register trained weights for live CCTV inference</p>
          </div>
          <button @click="showDeployModal = false" class="text-[#667085] hover:text-[#17202A] text-lg font-bold">✕</button>
        </div>

        <form @submit.prevent="submitDeployModel" class="space-y-3 text-xs">
          <div>
            <label class="block font-semibold text-[#17202A] mb-1">Model Name</label>
            <input type="text" v-model="deployForm.name" required class="w-full p-2 border border-[#DDE3EA] rounded font-medium" />
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Version Tag</label>
              <input type="text" v-model="deployForm.version" required class="w-full p-2 border border-[#DDE3EA] rounded font-mono" />
            </div>
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Format</label>
              <select v-model="deployForm.format" class="w-full p-2 border border-[#DDE3EA] rounded bg-white font-mono">
                <option value="pt">PyTorch (.pt)</option>
                <option value="onnx">ONNX (.onnx)</option>
                <option value="engine">TensorRT (.engine)</option>
              </select>
            </div>
          </div>

          <div>
            <label class="block font-semibold text-[#17202A] mb-1">Model Weights Path (Absolute or Storage)</label>
            <input type="text" v-model="deployForm.file_path" required class="w-full p-2 border border-[#DDE3EA] rounded font-mono text-xs" />
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Task</label>
              <select v-model="deployForm.task" class="w-full p-2 border border-[#DDE3EA] rounded bg-white">
                <option value="object_detection">Object Detection (Vehicles)</option>
                <option value="license_plate_detection">License Plate Detection</option>
                <option value="ocr">License Plate OCR</option>
              </select>
            </div>
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Input Size</label>
              <input type="number" v-model="deployForm.input_size" class="w-full p-2 border border-[#DDE3EA] rounded font-mono" />
            </div>
          </div>

          <div>
            <label class="block font-semibold text-[#17202A] mb-1">Notes & Training Context</label>
            <textarea v-model="deployForm.notes" rows="2" class="w-full p-2 border border-[#DDE3EA] rounded"></textarea>
          </div>

          <div v-if="deployError" class="p-2.5 bg-[#C94141]/10 text-[#C94141] rounded font-semibold text-xs border border-[#C94141]/20">
            {{ deployError }}
          </div>

          <div class="flex justify-end space-x-2 pt-3 border-t border-[#DDE3EA]">
            <button
              type="button"
              @click="showDeployModal = false"
              class="px-3 py-1.5 border border-[#DDE3EA] rounded text-[#667085] hover:bg-[#F6F8FB] font-semibold"
            >
              Cancel
            </button>
            <button
              type="submit"
              :disabled="isDeploying"
              class="px-4 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded font-semibold flex items-center space-x-1.5 shadow-sm"
            >
              <Upload class="w-3.5 h-3.5" />
              <span>{{ isDeploying ? 'Validating...' : 'Validate & Deploy' }}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>
