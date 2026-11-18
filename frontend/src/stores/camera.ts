import { defineStore } from 'pinia'
import { ref } from 'vue'
import apiClient from '@/api/client'
import type { Camera, CameraHealth } from '@/types'

export const useCameraStore = defineStore('camera', () => {
  const cameras = ref<Camera[]>([])
  const activeCamera = ref<Camera | null>(null)
  const healthMap = ref<Record<string, CameraHealth>>({})
  const loading = ref(false)

  async function fetchCameras() {
    loading.value = true
    try {
      const res = await apiClient.get<Camera[]>('/cameras')
      cameras.value = res.data
      if (res.data.length > 0 && !activeCamera.value) {
        activeCamera.value = res.data[0]
      }
    } finally {
      loading.value = false
    }
  }

  async function fetchCameraHealth(cameraId: string) {
    try {
      const res = await apiClient.get<CameraHealth>(`/cameras/${cameraId}/health`)
      healthMap.value[cameraId] = res.data
      return res.data
    } catch (err) {
      console.error('Failed to probe camera health', err)
      return null
    }
  }

  function setActiveCamera(camera: Camera) {
    activeCamera.value = camera
  }

  return {
    cameras,
    activeCamera,
    healthMap,
    loading,
    fetchCameras,
    fetchCameraHealth,
    setActiveCamera
  }
})
