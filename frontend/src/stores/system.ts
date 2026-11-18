import { defineStore } from 'pinia'
import { ref } from 'vue'
import apiClient from '@/api/client'
import type { SystemMetrics, HealthStatus, PipelinePreset } from '@/types'

export const useSystemStore = defineStore('system', () => {
  const metrics = ref<SystemMetrics | null>(null)
  const health = ref<HealthStatus | null>(null)
  const pipelines = ref<PipelinePreset[]>([])
  const activePipeline = ref<PipelinePreset | null>(null)

  async function fetchMetrics() {
    try {
      const res = await apiClient.get<SystemMetrics>('/system/metrics')
      metrics.value = res.data
    } catch (err) {
      console.error('Failed to fetch system metrics', err)
    }
  }

  async function fetchHealth() {
    try {
      const res = await apiClient.get<HealthStatus>('/health')
      health.value = res.data
    } catch (err) {
      console.error('Failed to fetch system health', err)
    }
  }

  async function fetchPipelines() {
    try {
      const res = await apiClient.get<PipelinePreset[]>('/pipelines')
      pipelines.value = res.data
      const active = res.data.find(p => p.is_active)
      if (active) activePipeline.value = active
    } catch (err) {
      console.error('Failed to fetch pipelines', err)
    }
  }

  async function activatePipeline(presetId: string) {
    try {
      const res = await apiClient.post<PipelinePreset>(`/pipelines/${presetId}/activate`)
      activePipeline.value = res.data
      await fetchPipelines()
    } catch (err) {
      console.error('Failed to activate pipeline', err)
      throw err
    }
  }

  return {
    metrics,
    health,
    pipelines,
    activePipeline,
    fetchMetrics,
    fetchHealth,
    fetchPipelines,
    activatePipeline
  }
})
