<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch, nextTick } from 'vue'
import apiClient from '@/api/client'
import {
  Database,
  Plus,
  Download,
  Folder,
  Tag,
  Split,
  Play,
  Square,
  Camera,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Save,
  FileText,
  Filter,
  Layers,
  Search,
  Check,
  Eye,
  Clock,
  Sun,
  Sunrise,
  Sunset,
  Moon,
  Move,
  Maximize2,
  Settings2,
  X,
  Info
} from 'lucide-vue-next'

// Active Tab
const activeTab = ref<'review' | 'capture' | 'datasets' | 'ocr'>('review')

// Supported Traffic Classes
export interface TrafficClass {
  id: number
  name: string
  label_en?: string
  label_id?: string
  color: string
  is_custom?: boolean
}

// Default Indonesian-specific & standard classes
const defaultClasses: TrafficClass[] = [
  { id: 0, name: 'person', label_en: 'Pedestrian', label_id: 'Pejalan Kaki', color: '#2563EB' },
  { id: 1, name: 'bicycle', label_en: 'Bicycle', label_id: 'Sepeda Onthel', color: '#059669' },
  { id: 2, name: 'car', label_en: 'Car', label_id: 'Mobil Pribadi', color: '#16A34A' },
  { id: 3, name: 'motorcycle', label_en: 'Motorcycle', label_id: 'Sepeda Motor', color: '#D97706' },
  { id: 4, name: 'bus', label_en: 'Bus / Minibus', label_id: 'Bus Kota / BST', color: '#DC2626' },
  { id: 5, name: 'truck', label_en: 'Truck / Cargo', label_id: 'Truk Barang', color: '#7C3AED' },
  { id: 6, name: 'becak', label_en: 'Pedicab / Cycle-Rickshaw', label_id: 'Becak Kayuh (Khas)', color: '#EA580C' },
  { id: 7, name: 'bajaj', label_en: 'Auto-Rickshaw', label_id: 'Bajaj Roda Tiga', color: '#CA8A04' },
  { id: 8, name: 'angkot', label_en: 'Minibus / Microbus', label_id: 'Angkot / Daihatsu', color: '#0891B2' },
  { id: 9, name: 'bentor', label_en: 'Motorized Becak', label_id: 'Becak Motor (Bentor)', color: '#9333EA' },
  { id: 10, name: 'gerobak', label_en: 'Handcart / Street Cart', label_id: 'Gerobak Pedagang', color: '#4B5563' },
  { id: 11, name: 'odong_odong', label_en: 'Tourist Tram / Train', label_id: 'Odong-odong Wisata', color: '#DB2777' },
  { id: 12, name: 'license_plate', label_en: 'License Plate', label_id: 'Plat Nomor', color: '#D58A16' }
]

const supportedClasses = ref<TrafficClass[]>([...defaultClasses])
const classSearchQuery = ref('')
const filteredClasses = computed(() => {
  if (!classSearchQuery.value.trim()) return supportedClasses.value
  const q = classSearchQuery.value.toLowerCase()
  return supportedClasses.value.filter(c =>
    c.name.toLowerCase().includes(q) ||
    (c.label_en && c.label_en.toLowerCase().includes(q)) ||
    (c.label_id && c.label_id.toLowerCase().includes(q))
  )
})

// Add class modal state
const showAddClassModal = ref(false)
const newClassName = ref('')
const newClassLabelEn = ref('')
const newClassLabelId = ref('')
const newClassColor = ref('#10B981')
const isSavingClass = ref(false)
const addClassError = ref('')

function openAddClassModal() {
  newClassName.value = ''
  newClassLabelEn.value = ''
  newClassLabelId.value = ''
  newClassColor.value = '#10B981'
  addClassError.value = ''
  showAddClassModal.value = true
}

function closeAddClassModal() {
  showAddClassModal.value = false
  addClassError.value = ''
}

async function fetchClasses() {
  try {
    const res = await apiClient.get('/annotations/classes')
    if (res.data && res.data.classes && res.data.classes.length > 0) {
      supportedClasses.value = res.data.classes
    }
  } catch (err) {
    console.warn('Could not fetch classes from backend, using defaults', err)
  }
}

async function handleAddCustomClass() {
  const rawName = newClassName.value.trim()
  if (!rawName) {
    addClassError.value = 'Nama kelas wajib diisi'
    return
  }
  isSavingClass.value = true
  addClassError.value = ''

  const slug = rawName.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_')
  const maxId = Math.max(...supportedClasses.value.map(c => c.id), -1)
  const newClassObj: TrafficClass = {
    id: maxId + 1,
    name: slug,
    label_en: newClassLabelEn.value.trim() || rawName,
    label_id: newClassLabelId.value.trim() || rawName,
    color: newClassColor.value || '#10B981',
    is_custom: true
  }

  try {
    const res = await apiClient.post('/annotations/classes', {
      name: slug,
      label_en: newClassObj.label_en,
      label_id: newClassObj.label_id,
      color: newClassObj.color
    })
    if (res.data && res.data.classes) {
      supportedClasses.value = res.data.classes
    } else {
      const idx = supportedClasses.value.findIndex(c => c.name === slug)
      if (idx >= 0) supportedClasses.value[idx] = newClassObj
      else supportedClasses.value.push(newClassObj)
    }
  } catch (err: any) {
    console.warn('Backend class save failed, adding locally to session:', err)
    const idx = supportedClasses.value.findIndex(c => c.name === slug)
    if (idx >= 0) supportedClasses.value[idx] = newClassObj
    else supportedClasses.value.push(newClassObj)
  } finally {
    isSavingClass.value = false
    setBoxClass(slug)
    closeAddClassModal()
    newClassName.value = ''
    newClassLabelEn.value = ''
    newClassLabelId.value = ''
    saveSuccessMessage.value = `Kelas "${slug}" berhasil ditambahkan!`
    setTimeout(() => { saveSuccessMessage.value = '' }, 3000)
  }
}

async function handleDeleteCustomClass(className: string) {
  if (!confirm(`Hapus kelas "${className}"?`)) return
  try {
    const res = await apiClient.delete(`/annotations/classes/${encodeURIComponent(className)}`)
    if (res.data && res.data.classes) {
      supportedClasses.value = res.data.classes
    } else {
      supportedClasses.value = supportedClasses.value.filter(c => c.name !== className)
    }
    if (selectedClass.value === className) {
      selectedClass.value = supportedClasses.value[0]?.name || 'car'
    }
  } catch (err) {
    console.warn('Backend class delete failed, removing locally:', err)
    supportedClasses.value = supportedClasses.value.filter(c => c.name !== className)
    if (selectedClass.value === className) {
      selectedClass.value = supportedClasses.value[0]?.name || 'car'
    }
  }
}

async function handleQuickAddClass(rawName: string) {
  const name = rawName.trim()
  if (!name) return
  const slug = name.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_')
  const maxId = Math.max(...supportedClasses.value.map(c => c.id), -1)
  const newClassObj: TrafficClass = {
    id: maxId + 1,
    name: slug,
    label_en: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    label_id: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    color: '#10B981',
    is_custom: true
  }

  const idx = supportedClasses.value.findIndex(c => c.name === slug)
  if (idx >= 0) {
    setBoxClass(slug)
    classSearchQuery.value = ''
    return
  }

  supportedClasses.value.push(newClassObj)
  setBoxClass(slug)
  classSearchQuery.value = ''
  saveSuccessMessage.value = `Kelas "${slug}" berhasil ditambahkan!`
  setTimeout(() => { saveSuccessMessage.value = '' }, 3000)

  try {
    const res = await apiClient.post('/annotations/classes', {
      name: slug,
      label_en: newClassObj.label_en,
      label_id: newClassObj.label_id,
      color: newClassObj.color
    })
    if (res.data && res.data.classes) {
      supportedClasses.value = res.data.classes
    }
  } catch (err) {
    console.warn('Backend quick class save fallback to local:', err)
  }
}

// -------------------------------------------------------------
// TAB 1: ANNOTATION REVIEW STUDIO
// -------------------------------------------------------------
const rawImages = ref<any[]>([])
const selectedImageIndex = ref(0)
const currentImage = computed(() => rawImages.value[selectedImageIndex.value] || null)
const currentAnnotations = ref<any[]>([])
const selectedBoxIndex = ref<number | null>(null)
const selectedClass = ref('car')
const annotationStatus = ref('unreviewed')
const isAutoAnnotating = ref(false)
const saveSuccessMessage = ref('')

const canvasRef = ref<HTMLCanvasElement | null>(null)
type InteractionMode = 'none' | 'drawing' | 'moving' | 'resizing'
const interactionMode = ref<InteractionMode>('none')
const activeHandle = ref<string | null>(null)
const dragStartPos = ref({ x: 0, y: 0 })
const originalBbox = ref<[number, number, number, number] | null>(null)
const canvasCursor = ref('crosshair')
const isDrawing = ref(false)
const drawStartX = ref(0)
const drawStartY = ref(0)
const drawCurrentX = ref(0)
const drawCurrentY = ref(0)
const loadedImageObj = ref<HTMLImageElement | null>(null)
const loadedImageUrl = ref<string | null>(null)

async function fetchRawImages() {
  try {
    const res = await apiClient.get('/capture/raw-images?status_filter=accepted&limit=50')
    rawImages.value = res.data.items || []
    if (rawImages.value.length > 0) {
      loadImageAnnotations(0)
    }
  } catch (err) {
    console.error('Failed to fetch raw images', err)
  }
}

async function loadImageAnnotations(index: number) {
  if (index < 0 || index >= rawImages.value.length) return
  selectedImageIndex.value = index
  selectedBoxIndex.value = null
  const item = rawImages.value[index]
  if (!item) return

  try {
    const res = await apiClient.get(`/annotations/item?image_path=${encodeURIComponent(item.file_path)}`)
    currentAnnotations.value = res.data.annotations || []
    annotationStatus.value = res.data.status || 'unreviewed'
  } catch (err) {
    currentAnnotations.value = []
    annotationStatus.value = 'unreviewed'
  }
  await nextTick()
  renderCanvas()
}

// -------------------------------------------------------------
// PREVIEW MODAL & NAVIGATION FOR RAW CAPTURED IMAGES
// -------------------------------------------------------------
const previewModalImage = ref<any>(null)
const previewModalIndex = ref<number>(0)

function openImagePreview(img: any, idx: number) {
  previewModalImage.value = img
  previewModalIndex.value = idx
}

function closeImagePreview() {
  previewModalImage.value = null
}

async function jumpToAnnotationStudio(idx: number) {
  closeImagePreview()
  activeTab.value = 'review'
  await nextTick()
  await loadImageAnnotations(idx)
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

watch(activeTab, async (newTab) => {
  if (newTab === 'review') {
    await nextTick()
    renderCanvas()
  }
})

function prevImage() {
  if (selectedImageIndex.value > 0) {
    loadImageAnnotations(selectedImageIndex.value - 1)
  }
}

function nextImage() {
  if (selectedImageIndex.value < rawImages.value.length - 1) {
    loadImageAnnotations(selectedImageIndex.value + 1)
  }
}

async function triggerAutoAnnotate() {
  if (!currentImage.value) return
  isAutoAnnotating.value = true
  try {
    const res = await apiClient.post('/annotations/auto-annotate', {
      image_path: currentImage.value.file_path,
      conf_threshold: 0.25
    })
    if (res.data.annotations) {
      currentAnnotations.value = res.data.annotations.map((a: any) => ({
        class_name: a.class || a.class_name,
        class_idx: a.class_idx,
        confidence: a.confidence,
        bbox: a.bbox,
        yolo_bbox: a.yolo_bbox
      }))
      annotationStatus.value = 'pseudo_annotated'
      saveSuccessMessage.value = `✓ YOLO26 mendeteksi ${res.data.annotations.length} objek! (Status: PSEUDO_ANNOTATED)`
      setTimeout(() => { saveSuccessMessage.value = '' }, 4000)
      renderCanvas()
    }
  } catch (err: any) {
    console.error('Auto annotation failed', err)
    saveSuccessMessage.value = 'Gagal auto-annotate: ' + (err.response?.data?.detail || err.message)
    setTimeout(() => { saveSuccessMessage.value = '' }, 4000)
  } finally {
    isAutoAnnotating.value = false
  }
}

const isBatchAutoAnnotating = ref(false)
const isBatchApproving = ref(false)
const batchNotice = ref('')

async function triggerBatchAutoAnnotate() {
  if (isBatchAutoAnnotating.value) return
  isBatchAutoAnnotating.value = true
  try {
    const res = await apiClient.post('/annotations/batch-auto-annotate', { limit: 50, conf_threshold: 0.25 })
    batchNotice.value = `✓ Berhasil auto-annotate ${res.data.processed_count} frame raw (Status: PSEUDO_ANNOTATED)`
    setTimeout(() => { batchNotice.value = '' }, 5000)
    await fetchRawImages()
    if (rawImages.value.length > 0) {
      await loadImageAnnotations(selectedImageIndex.value)
    }
  } catch (err: any) {
    batchNotice.value = 'Gagal auto-annotate batch: ' + (err.response?.data?.detail || err.message)
    setTimeout(() => { batchNotice.value = '' }, 5000)
  } finally {
    isBatchAutoAnnotating.value = false
  }
}

async function triggerBatchApprove() {
  if (isBatchApproving.value) return
  isBatchApproving.value = true
  try {
    const res = await apiClient.post('/annotations/batch-approve', { limit: 100 })
    batchNotice.value = `✓ Berhasil verifikasi (Approve) ${res.data.approved_count} frame beranotasi!`
    setTimeout(() => { batchNotice.value = '' }, 5000)
    await fetchRawImages()
    if (rawImages.value.length > 0) {
      await loadImageAnnotations(selectedImageIndex.value)
    }
  } catch (err: any) {
    batchNotice.value = 'Gagal approve batch: ' + (err.response?.data?.detail || err.message)
    setTimeout(() => { batchNotice.value = '' }, 5000)
  } finally {
    isBatchApproving.value = false
  }
}

async function saveAnnotations(statusToSave: string = 'approved') {
  if (!currentImage.value) return
  try {
    await apiClient.put('/annotations/save', {
      image_path: currentImage.value.file_path,
      status: statusToSave,
      annotations: currentAnnotations.value
    })
    annotationStatus.value = statusToSave
    saveSuccessMessage.value = statusToSave === 'approved' 
      ? '✓ Berhasil Diverifikasi (APPROVED)' 
      : `Saved as ${statusToSave.toUpperCase()}`
    setTimeout(() => { saveSuccessMessage.value = '' }, 3500)
  } catch (err: any) {
    console.error('Failed to save annotations', err)
    saveSuccessMessage.value = 'Gagal menyimpan: ' + (err.response?.data?.detail || err.message)
    setTimeout(() => { saveSuccessMessage.value = '' }, 3500)
  }
}

function removeSelectedBox() {
  if (selectedBoxIndex.value !== null) {
    currentAnnotations.value.splice(selectedBoxIndex.value, 1)
    selectedBoxIndex.value = null
    renderCanvas()
  }
}

function setBoxClass(clsName: string) {
  selectedClass.value = clsName
  if (selectedBoxIndex.value !== null && currentAnnotations.value[selectedBoxIndex.value]) {
    const found = supportedClasses.value.find(c => c.name === clsName)
    if (found) {
      currentAnnotations.value[selectedBoxIndex.value].class_name = found.name
      currentAnnotations.value[selectedBoxIndex.value].class_idx = found.id
      renderCanvas()
    }
  }
}

function hexToRgba(hex: string, alpha: number) {
  let c = hex.replace('#', '')
  if (c.length === 3) c = c.split('').map(x => x + x).join('')
  const num = parseInt(c, 16) || 0
  return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`
}

function getHandleAtPoint(mx: number, my: number, bbox: number[], radius: number = 9): string | null {
  const [x1, y1, x2, y2] = bbox
  const midX = (x1 + x2) / 2
  const midY = (y1 + y2) / 2
  const handles: Record<string, [number, number]> = {
    nw: [x1, y1],
    ne: [x2, y1],
    se: [x2, y2],
    sw: [x1, y2],
    n: [midX, y1],
    s: [midX, y2],
    e: [x2, midY],
    w: [x1, midY]
  }
  for (const [key, [hx, hy]] of Object.entries(handles)) {
    if (Math.hypot(mx - hx, my - hy) <= radius) {
      return key
    }
  }
  return null
}

function getCursorForHandle(handle: string): string {
  switch (handle) {
    case 'nw':
    case 'se':
      return 'nwse-resize'
    case 'ne':
    case 'sw':
      return 'nesw-resize'
    case 'n':
    case 's':
      return 'ns-resize'
    case 'e':
    case 'w':
      return 'ew-resize'
    default:
      return 'crosshair'
  }
}

function getCanvasCoords(e: MouseEvent): { mx: number; my: number } | null {
  const canvas = canvasRef.value
  if (!canvas) return null
  const rect = canvas.getBoundingClientRect()
  const scaleX = canvas.width / rect.width
  const scaleY = canvas.height / rect.height
  return {
    mx: (e.clientX - rect.left) * scaleX,
    my: (e.clientY - rect.top) * scaleY
  }
}

function renderCanvas() {
  const canvas = canvasRef.value
  if (!canvas || !currentImage.value) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const imageSrc = currentImage.value.relative_url || `/storage/datasets/raw/${currentImage.value.file_path.split('storage/datasets/raw/')[1]}`

  const drawScene = (img: HTMLImageElement) => {
    canvas.width = img.naturalWidth || 640
    canvas.height = img.naturalHeight || 480
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    // Draw bounding boxes
    currentAnnotations.value.forEach((ann, idx) => {
      const [x1, y1, x2, y2] = ann.bbox
      const isSelected = selectedBoxIndex.value === idx
      const classMeta = supportedClasses.value.find(c => c.name === ann.class_name) || { color: '#16A34A' }
      const bw = x2 - x1
      const bh = y2 - y1

      // If selected: draw translucent fill
      if (isSelected) {
        ctx.fillStyle = hexToRgba(classMeta.color, 0.18)
        ctx.fillRect(x1, y1, bw, bh)
      }

      ctx.strokeStyle = classMeta.color
      ctx.lineWidth = isSelected ? 3 : 2
      ctx.strokeRect(x1, y1, bw, bh)

      // Label background & text
      const labelText = `${ann.class_name} ${(ann.confidence ? ann.confidence.toFixed(2) : '1.0')}`
      ctx.font = 'bold 12px Inter, sans-serif'
      const textWidth = ctx.measureText(labelText).width
      ctx.fillStyle = classMeta.color
      ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 10, 22)

      ctx.fillStyle = '#FFFFFF'
      ctx.fillText(labelText, x1 + 5, Math.max(16, y1 - 6))

      // If selected: draw 8 interactive handles
      if (isSelected) {
        const midX = (x1 + x2) / 2
        const midY = (y1 + y2) / 2
        const handles = [
          [x1, y1], [x2, y1], [x2, y2], [x1, y2], // 4 corners
          [midX, y1], [midX, y2], [x2, midY], [x1, midY] // 4 edge midpoints
        ]
        handles.forEach(([hx, hy]) => {
          ctx.fillStyle = '#FFFFFF'
          ctx.strokeStyle = classMeta.color
          ctx.lineWidth = 2.5
          ctx.beginPath()
          ctx.arc(hx, hy, 5, 0, 2 * Math.PI)
          ctx.fill()
          ctx.stroke()
        })
      }
    })

    // If currently drawing new box
    if (interactionMode.value === 'drawing') {
      const bx1 = Math.min(drawStartX.value, drawCurrentX.value)
      const by1 = Math.min(drawStartY.value, drawCurrentY.value)
      const bw = Math.abs(drawCurrentX.value - drawStartX.value)
      const bh = Math.abs(drawCurrentY.value - drawStartY.value)
      ctx.strokeStyle = '#238A57'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.strokeRect(bx1, by1, bw, bh)
      ctx.setLineDash([])
    }
  }

  // Fast redraw: reuse cached image object if available
  if (loadedImageObj.value && loadedImageUrl.value === imageSrc) {
    drawScene(loadedImageObj.value)
  } else {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.src = imageSrc
    img.onload = () => {
      loadedImageObj.value = img
      loadedImageUrl.value = imageSrc
      drawScene(img)
    }
  }
}

function handleMouseDown(e: MouseEvent) {
  const coords = getCanvasCoords(e)
  if (!coords) return
  const { mx, my } = coords

  // 1. If a box is selected, check if user clicked on its resize handles
  if (selectedBoxIndex.value !== null && currentAnnotations.value[selectedBoxIndex.value]) {
    const selBox = currentAnnotations.value[selectedBoxIndex.value]
    const handle = getHandleAtPoint(mx, my, selBox.bbox, 9)
    if (handle) {
      interactionMode.value = 'resizing'
      activeHandle.value = handle
      dragStartPos.value = { x: mx, y: my }
      originalBbox.value = [...selBox.bbox] as [number, number, number, number]
      canvasCursor.value = getCursorForHandle(handle)
      return
    }

    // 2. Check if clicked inside selected box to move/reposition it
    const [x1, y1, x2, y2] = selBox.bbox
    if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
      interactionMode.value = 'moving'
      dragStartPos.value = { x: mx, y: my }
      originalBbox.value = [...selBox.bbox] as [number, number, number, number]
      canvasCursor.value = 'move'
      return
    }
  }

  // 3. Check if clicked inside any other box
  let clickedIndex: number | null = null
  for (let i = currentAnnotations.value.length - 1; i >= 0; i--) {
    const [x1, y1, x2, y2] = currentAnnotations.value[i].bbox
    if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
      clickedIndex = i
      break
    }
  }

  if (clickedIndex !== null) {
    selectedBoxIndex.value = clickedIndex
    selectedClass.value = currentAnnotations.value[clickedIndex].class_name
    interactionMode.value = 'moving'
    dragStartPos.value = { x: mx, y: my }
    originalBbox.value = [...currentAnnotations.value[clickedIndex].bbox] as [number, number, number, number]
    canvasCursor.value = 'move'
    renderCanvas()
    return
  }

  // 4. Clicked outside -> deselect and start drawing new box
  selectedBoxIndex.value = null
  interactionMode.value = 'drawing'
  drawStartX.value = mx
  drawStartY.value = my
  drawCurrentX.value = mx
  drawCurrentY.value = my
  canvasCursor.value = 'crosshair'
  renderCanvas()
}

function handleMouseMove(e: MouseEvent) {
  const coords = getCanvasCoords(e)
  if (!coords) return
  const { mx, my } = coords
  const canvas = canvasRef.value
  if (!canvas) return

  if (interactionMode.value === 'none') {
    // Dynamic cursor feedback
    if (selectedBoxIndex.value !== null && currentAnnotations.value[selectedBoxIndex.value]) {
      const selBox = currentAnnotations.value[selectedBoxIndex.value]
      const handle = getHandleAtPoint(mx, my, selBox.bbox, 9)
      if (handle) {
        canvasCursor.value = getCursorForHandle(handle)
        return
      }
      const [x1, y1, x2, y2] = selBox.bbox
      if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
        canvasCursor.value = 'move'
        return
      }
    }
    // Hovering over other boxes
    let overBox = false
    for (let i = currentAnnotations.value.length - 1; i >= 0; i--) {
      const [x1, y1, x2, y2] = currentAnnotations.value[i].bbox
      if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
        overBox = true
        break
      }
    }
    canvasCursor.value = overBox ? 'pointer' : 'crosshair'
    return
  }

  if (interactionMode.value === 'resizing' && selectedBoxIndex.value !== null && originalBbox.value && activeHandle.value) {
    const dx = mx - dragStartPos.value.x
    const dy = my - dragStartPos.value.y
    let [x1, y1, x2, y2] = originalBbox.value

    switch (activeHandle.value) {
      case 'nw':
        x1 += dx; y1 += dy; break
      case 'ne':
        x2 += dx; y1 += dy; break
      case 'se':
        x2 += dx; y2 += dy; break
      case 'sw':
        x1 += dx; y2 += dy; break
      case 'n':
        y1 += dy; break
      case 's':
        y2 += dy; break
      case 'e':
        x2 += dx; break
      case 'w':
        x1 += dx; break
    }

    x1 = Math.max(0, Math.min(canvas.width, x1))
    x2 = Math.max(0, Math.min(canvas.width, x2))
    y1 = Math.max(0, Math.min(canvas.height, y1))
    y2 = Math.max(0, Math.min(canvas.height, y2))

    currentAnnotations.value[selectedBoxIndex.value].bbox = [
      Math.round(Math.min(x1, x2)),
      Math.round(Math.min(y1, y2)),
      Math.round(Math.max(x1, x2)),
      Math.round(Math.max(y1, y2))
    ]
    renderCanvas()
    return
  }

  if (interactionMode.value === 'moving' && selectedBoxIndex.value !== null && originalBbox.value) {
    const dx = mx - dragStartPos.value.x
    const dy = my - dragStartPos.value.y
    const [ox1, oy1, ox2, oy2] = originalBbox.value
    const bw = ox2 - ox1
    const bh = oy2 - oy1

    let nx1 = ox1 + dx
    let ny1 = oy1 + dy

    nx1 = Math.max(0, Math.min(canvas.width - bw, nx1))
    ny1 = Math.max(0, Math.min(canvas.height - bh, ny1))

    currentAnnotations.value[selectedBoxIndex.value].bbox = [
      Math.round(nx1),
      Math.round(ny1),
      Math.round(nx1 + bw),
      Math.round(ny1 + bh)
    ]
    renderCanvas()
    return
  }

  if (interactionMode.value === 'drawing') {
    drawCurrentX.value = mx
    drawCurrentY.value = my
    renderCanvas()
  }
}

function handleMouseUp() {
  const canvas = canvasRef.value
  if (!canvas) return

  if ((interactionMode.value === 'resizing' || interactionMode.value === 'moving') && selectedBoxIndex.value !== null) {
    const ann = currentAnnotations.value[selectedBoxIndex.value]
    if (ann) {
      let [x1, y1, x2, y2] = ann.bbox
      const nx1 = Math.min(x1, x2)
      const ny1 = Math.min(y1, y2)
      const nx2 = Math.max(x1, x2)
      const ny2 = Math.max(y1, y2)
      ann.bbox = [nx1, ny1, nx2, ny2]

      const bw = nx2 - nx1
      const bh = ny2 - ny1
      const xc = (nx1 + nx2) / 2.0 / canvas.width
      const yc = (ny1 + ny2) / 2.0 / canvas.height
      ann.yolo_bbox = [xc, yc, bw / canvas.width, bh / canvas.height]
    }
  } else if (interactionMode.value === 'drawing') {
    const x1 = Math.round(Math.min(drawStartX.value, drawCurrentX.value))
    const y1 = Math.round(Math.min(drawStartY.value, drawCurrentY.value))
    const x2 = Math.round(Math.max(drawStartX.value, drawCurrentX.value))
    const y2 = Math.round(Math.max(drawStartY.value, drawCurrentY.value))

    if ((x2 - x1) > 10 && (y2 - y1) > 10) {
      const found = supportedClasses.value.find(c => c.name === selectedClass.value) || supportedClasses.value[2]
      const w = canvas.width
      const h = canvas.height
      const bw = x2 - x1
      const bh = y2 - y1
      const xc = (x1 + x2) / 2.0 / w
      const yc = (y1 + y2) / 2.0 / h

      currentAnnotations.value.push({
        class_name: found.name,
        class_idx: found.id,
        confidence: 1.0,
        bbox: [x1, y1, x2, y2],
        yolo_bbox: [xc, yc, bw / w, bh / h]
      })
      selectedBoxIndex.value = currentAnnotations.value.length - 1
    }
  }

  interactionMode.value = 'none'
  activeHandle.value = null
  originalBbox.value = null
  renderCanvas()
}

// -------------------------------------------------------------
// TAB 2: DATASET CAPTURE MANAGER
// -------------------------------------------------------------
const camerasList = ref<any[]>([])
const selectedCameraId = ref('')
const captureInterval = ref(5)
const maxStorageMb = ref(10000)
const captureJobs = ref<any[]>([])
const isStartingJob = ref(false)
const singleCaptureLoading = ref(false)
const captureSuccessMsg = ref('')

// Schedule / Time-Window Configuration
const timeFilterEnabled = ref(false)
const selectedPreset = ref<'morning' | 'lunch' | 'evening' | 'night' | 'full' | 'custom'>('morning')
const startTime = ref('06:30')
const endTime = ref('08:30')
const scheduleLabel = ref('Pagi Padat (06:30 - 08:30 WIB)')
const currentWibTime = ref('')
const isCurrentTimeInWindow = ref(true)
let clockIntervalId: any = null

function applySchedulePreset(preset: 'morning' | 'lunch' | 'evening' | 'night' | 'full' | 'custom') {
  selectedPreset.value = preset
  if (preset === 'morning') {
    timeFilterEnabled.value = true
    startTime.value = '06:30'
    endTime.value = '08:30'
    scheduleLabel.value = 'Pagi Padat (06:30 - 08:30 WIB)'
  } else if (preset === 'lunch') {
    timeFilterEnabled.value = true
    startTime.value = '11:30'
    endTime.value = '13:30'
    scheduleLabel.value = 'Siang Padat (11:30 - 13:30 WIB)'
  } else if (preset === 'evening') {
    timeFilterEnabled.value = true
    startTime.value = '16:30'
    endTime.value = '19:30'
    scheduleLabel.value = 'Sore Padat (16:30 - 19:30 WIB)'
  } else if (preset === 'night') {
    timeFilterEnabled.value = true
    startTime.value = '21:00'
    endTime.value = '04:00'
    scheduleLabel.value = 'Malam / Low-Light (21:00 - 04:00 WIB)'
  } else if (preset === 'full') {
    timeFilterEnabled.value = false
    startTime.value = '00:00'
    endTime.value = '23:59'
    scheduleLabel.value = '24 Jam Penuh / Continuous'
  } else if (preset === 'custom') {
    timeFilterEnabled.value = true
    scheduleLabel.value = `Kustom (${startTime.value} - ${endTime.value} WIB)`
  }
  checkWindowStatus()
}

function checkWindowStatus() {
  const now = new Date()
  const utc = now.getTime() + (now.getTimezoneOffset() * 60000)
  const wibDate = new Date(utc + (3600000 * 7))
  
  const h = String(wibDate.getHours()).padStart(2, '0')
  const m = String(wibDate.getMinutes()).padStart(2, '0')
  const s = String(wibDate.getSeconds()).padStart(2, '0')
  currentWibTime.value = `${h}:${m}:${s} WIB`

  if (!timeFilterEnabled.value) {
    isCurrentTimeInWindow.value = true
    return
  }

  const nowMin = wibDate.getHours() * 60 + wibDate.getMinutes()
  const [sh, sm] = startTime.value.split(':').map(Number)
  const [eh, em] = endTime.value.split(':').map(Number)
  const sMin = sh * 60 + sm
  const eMin = eh * 60 + em

  if (sMin <= eMin) {
    isCurrentTimeInWindow.value = (nowMin >= sMin && nowMin <= eMin)
  } else {
    isCurrentTimeInWindow.value = (nowMin >= sMin || nowMin <= eMin)
  }
}

async function fetchCameras() {
  try {
    const res = await apiClient.get('/cameras')
    camerasList.value = res.data
    if (camerasList.value.length > 0 && !selectedCameraId.value) {
      selectedCameraId.value = camerasList.value[0].id
    }
  } catch (err) {
    console.error('Failed to load cameras', err)
  }
}

async function fetchCaptureJobs() {
  try {
    const res = await apiClient.get('/capture/jobs')
    captureJobs.value = res.data
  } catch (err) {
    console.error('Failed to load capture jobs', err)
  }
}

async function startScheduledCapture() {
  const cam = camerasList.value.find(c => c.id === selectedCameraId.value)
  if (!cam) return
  isStartingJob.value = true
  try {
    await apiClient.post('/capture/start', {
      camera_id: cam.id,
      camera_name: cam.name,
      stream_url: cam.rtsp_main_stream,
      interval_seconds: captureInterval.value,
      max_storage_mb: maxStorageMb.value,
      time_filter_enabled: timeFilterEnabled.value,
      start_time: startTime.value,
      end_time: endTime.value,
      schedule_preset: selectedPreset.value,
      schedule_label: scheduleLabel.value
    })
    await fetchCaptureJobs()
    captureSuccessMsg.value = `Pengambilan frame dimulai untuk ${cam.name} (${scheduleLabel.value})`
    setTimeout(() => { captureSuccessMsg.value = '' }, 4000)
  } catch (err) {
    console.error('Start capture failed', err)
  } finally {
    isStartingJob.value = false
  }
}

async function stopScheduledCapture(camId: string) {
  try {
    await apiClient.post('/capture/stop', { camera_id: camId })
    await fetchCaptureJobs()
  } catch (err) {
    console.error('Stop capture failed', err)
  }
}

async function captureSingleNow() {
  const cam = camerasList.value.find(c => c.id === selectedCameraId.value)
  if (!cam) return
  singleCaptureLoading.value = true
  try {
    const res = await apiClient.post('/capture/single', {
      camera_id: cam.id,
      camera_name: cam.name,
      stream_url: cam.rtsp_main_stream
    })
    captureSuccessMsg.value = `Frame captured! Status: ${res.data.metadata.quality_status.toUpperCase()} (Blur: ${res.data.metadata.blur_score}, Brightness: ${res.data.metadata.brightness})`
    await fetchRawImages()
    setTimeout(() => { captureSuccessMsg.value = '' }, 4000)
  } catch (err) {
    console.error('Single capture failed', err)
  } finally {
    singleCaptureLoading.value = false
  }
}

// -------------------------------------------------------------
// TAB 3: DATASET VERSIONS & YOLO EXPORT
// -------------------------------------------------------------
const datasetName = ref('dishub_surakarta_vehicle_v1')
const trainSplit = ref(70)
const valSplit = ref(15)
const testSplit = ref(15)
const isExporting = ref(false)
const exportResult = ref<any>(null)

async function triggerYoloExport() {
  isExporting.value = true
  exportResult.value = null
  try {
    const res = await apiClient.post('/annotations/export-yolo', {
      dataset_name: datasetName.value,
      train_ratio: trainSplit.value / 100.0,
      val_ratio: valSplit.value / 100.0,
      test_ratio: testSplit.value / 100.0
    })
    exportResult.value = res.data
  } catch (err: any) {
    exportResult.value = {
      success: false,
      error: err.response?.data?.detail || 'Export failed'
    }
  } finally {
    isExporting.value = false
  }
}

// -------------------------------------------------------------
// TAB 4: LICENSE PLATE & OCR PREPARATION
// -------------------------------------------------------------
const plateRecords = ref<any[]>([])
const newPlateText = ref('')
const selectedPlateRecord = ref<any>(null)
const isExportingOcr = ref(false)
const ocrExportMsg = ref('')

async function fetchPlateRecords() {
  try {
    const res = await apiClient.get('/experiments/plates')
    plateRecords.value = res.data.items || []
  } catch (err) {
    console.error('Failed to load plate records', err)
  }
}

async function exportOcrDataset() {
  isExportingOcr.value = true
  try {
    const res = await apiClient.post('/experiments/plates/export-ocr')
    ocrExportMsg.value = `Exported ${res.data.total_samples} samples (Train: ${res.data.train_samples}, Val: ${res.data.val_samples})`
    setTimeout(() => { ocrExportMsg.value = '' }, 5000)
  } catch (err) {
    console.error('OCR export failed', err)
  } finally {
    isExportingOcr.value = false
  }
}

onMounted(async () => {
  await fetchClasses()
  await fetchCameras()
  await fetchRawImages()
  await fetchCaptureJobs()
  await fetchPlateRecords()
  checkWindowStatus()
  clockIntervalId = setInterval(checkWindowStatus, 1000)
})

onUnmounted(() => {
  if (clockIntervalId) clearInterval(clockIntervalId)
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <h2 class="text-xl font-bold text-[#17202A] tracking-tight">Dataset Studio & Annotation Pipeline</h2>
        <p class="text-xs text-[#667085]">Dishub Surakarta CCTV closed-loop acquisition, auto-labeling, and review</p>
      </div>

      <!-- Navigation Tabs -->
      <div class="flex items-center space-x-1 bg-white border border-[#DDE3EA] p-1 rounded-lg shadow-sm text-xs font-semibold">
        <button
          @click="activeTab = 'review'"
          :class="activeTab === 'review' ? 'bg-[#173B65] text-white' : 'text-[#667085] hover:text-[#17202A]'"
          class="px-3 py-1.5 rounded transition-colors flex items-center space-x-1.5"
        >
          <Tag class="w-3.5 h-3.5" />
          <span>Annotation Review</span>
        </button>
        <button
          @click="activeTab = 'capture'"
          :class="activeTab === 'capture' ? 'bg-[#173B65] text-white' : 'text-[#667085] hover:text-[#17202A]'"
          class="px-3 py-1.5 rounded transition-colors flex items-center space-x-1.5"
        >
          <Camera class="w-3.5 h-3.5" />
          <span>Capture Manager</span>
        </button>
        <button
          @click="activeTab = 'datasets'"
          :class="activeTab === 'datasets' ? 'bg-[#173B65] text-white' : 'text-[#667085] hover:text-[#17202A]'"
          class="px-3 py-1.5 rounded transition-colors flex items-center space-x-1.5"
        >
          <Database class="w-3.5 h-3.5" />
          <span>Versions & Export</span>
        </button>
        <button
          @click="activeTab = 'ocr'"
          :class="activeTab === 'ocr' ? 'bg-[#173B65] text-white' : 'text-[#667085] hover:text-[#17202A]'"
          class="px-3 py-1.5 rounded transition-colors flex items-center space-x-1.5"
        >
          <FileText class="w-3.5 h-3.5" />
          <span>Plate & OCR Prep</span>
        </button>
      </div>
    </div>

    <!-- ========================================================= -->
    <!-- TAB 1: ANNOTATION REVIEW STUDIO -->
    <!-- ========================================================= -->
    <div v-if="activeTab === 'review'" class="space-y-4">
      <!-- Educational Workflow Banner -->
      <div class="bg-[#173B65]/5 border border-[#173B65]/20 rounded-lg p-3 text-xs flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
        <div class="flex items-start space-x-2.5">
          <Info class="w-4 h-4 text-[#147D83] shrink-0 mt-0.5" />
          <div class="space-y-0.5">
            <span class="font-bold text-[#173B65]">Alur Verifikasi Closed-Loop (Dishub CCTV):</span>
            <p class="text-[#667085] text-[11px]">
              Frame mentah berstatus <strong class="text-[#667085]">UNREVIEWED</strong>. Klik <strong class="text-[#147D83]">Auto-Annotate YOLO26</strong> untuk mendeteksi objek otomatis (<strong class="text-[#D58A16]">PSEUDO-ANNOTATED</strong>). Sesuaikan ukuran/kelas BBox jika perlu, lalu klik <strong class="text-[#238A57]">Approve & Verifikasi</strong> untuk status final (<strong class="text-[#238A57]">APPROVED</strong>).
            </p>
          </div>
        </div>
        <div class="flex items-center space-x-2 shrink-0">
          <button
            type="button"
            @click="triggerBatchAutoAnnotate"
            :disabled="isBatchAutoAnnotating"
            class="px-2.5 py-1.5 bg-[#147D83] hover:bg-[#0E5B60] disabled:opacity-50 text-white rounded font-semibold text-[11px] flex items-center space-x-1 shadow-xs cursor-pointer"
            title="Deteksi semua frame mentah sekaligus dengan YOLO26"
          >
            <Sparkles class="w-3.5 h-3.5" />
            <span>{{ isBatchAutoAnnotating ? 'Memproses...' : '⚡ Auto-Annotate Semua' }}</span>
          </button>
          <button
            type="button"
            @click="triggerBatchApprove"
            :disabled="isBatchApproving"
            class="px-2.5 py-1.5 bg-[#238A57] hover:bg-[#1C6E46] disabled:opacity-50 text-white rounded font-semibold text-[11px] flex items-center space-x-1 shadow-xs cursor-pointer"
            title="Approve / Verifikasi semua frame yang sudah memiliki bounding box"
          >
            <CheckCircle2 class="w-3.5 h-3.5" />
            <span>{{ isBatchApproving ? 'Menyimpan...' : '✓ Approve Semua' }}</span>
          </button>
        </div>
      </div>

      <div v-if="batchNotice" class="p-2.5 bg-[#238A57]/10 border border-[#238A57]/30 text-[#238A57] font-semibold text-xs rounded-lg flex items-center space-x-2">
        <CheckCircle2 class="w-4 h-4 text-[#238A57]" />
        <span>{{ batchNotice }}</span>
      </div>

      <!-- Toolbar -->
      <div class="bg-white border border-[#DDE3EA] rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 shadow-thin">
        <div class="flex items-center space-x-3 text-xs">
          <div class="flex items-center space-x-1">
            <button
              @click="prevImage"
              :disabled="selectedImageIndex === 0"
              class="p-1.5 bg-[#F6F8FB] hover:bg-[#DDE3EA] disabled:opacity-40 border border-[#DDE3EA] rounded cursor-pointer"
            >
              <ChevronLeft class="w-4 h-4 text-[#17202A]" />
            </button>
            <span class="font-mono text-[#17202A] px-2 font-semibold">
              {{ selectedImageIndex + 1 }} / {{ rawImages.length }}
            </span>
            <button
              @click="nextImage"
              :disabled="selectedImageIndex >= rawImages.length - 1"
              class="p-1.5 bg-[#F6F8FB] hover:bg-[#DDE3EA] disabled:opacity-40 border border-[#DDE3EA] rounded cursor-pointer"
            >
              <ChevronRight class="w-4 h-4 text-[#17202A]" />
            </button>
          </div>

          <div class="h-4 w-px bg-[#DDE3EA]"></div>

          <span class="text-[#667085]">Camera:</span>
          <span class="font-semibold text-[#173B65]">{{ currentImage?.camera_name || 'Dishub CCTV' }}</span>

          <div class="h-4 w-px bg-[#DDE3EA]"></div>

          <span class="text-[#667085]">Status Review:</span>
          <span
            class="px-2.5 py-0.5 rounded text-[10px] font-bold uppercase"
            :class="{
              'bg-[#238A57]/15 text-[#238A57] border border-[#238A57]/30': annotationStatus === 'approved',
              'bg-[#D58A16]/15 text-[#D58A16] border border-[#D58A16]/30': annotationStatus === 'pseudo_annotated',
              'bg-[#C94141]/15 text-[#C94141] border border-[#C94141]/30': annotationStatus === 'rejected',
              'bg-[#667085]/15 text-[#667085] border border-[#667085]/30': annotationStatus === 'unreviewed' || annotationStatus === 'reviewed'
            }"
          >
            {{ annotationStatus === 'unreviewed' ? '⏳ UNREVIEWED (Belum Diverifikasi)' : annotationStatus.toUpperCase() }}
          </span>

          <span v-if="saveSuccessMessage" class="text-xs font-bold text-[#238A57] ml-2">
            {{ saveSuccessMessage }}
          </span>
        </div>

        <!-- Action Buttons -->
        <div class="flex items-center space-x-2 text-xs">
          <button
            @click="triggerAutoAnnotate"
            :disabled="isAutoAnnotating"
            class="px-3 py-1.5 bg-[#147D83] hover:bg-[#0E5B60] text-white rounded font-semibold flex items-center space-x-1.5 shadow-sm cursor-pointer"
            title="Deteksi kendaraan otomatis dengan model YOLO26"
          >
            <Sparkles class="w-3.5 h-3.5" />
            <span>{{ isAutoAnnotating ? 'Inference YOLO...' : 'Auto-Annotate YOLO26' }}</span>
          </button>

          <button
            @click="saveAnnotations('approved')"
            class="px-3 py-1.5 bg-[#238A57] hover:bg-[#1C6E46] text-white rounded font-semibold flex items-center space-x-1.5 shadow-sm cursor-pointer"
            title="Verifikasi frame ini sebagai dataset APPROVED untuk training"
          >
            <CheckCircle2 class="w-3.5 h-3.5" />
            <span>✓ Approve & Verifikasi</span>
          </button>

          <button
            @click="saveAnnotations('reviewed')"
            class="px-3 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded font-semibold flex items-center space-x-1.5 shadow-sm cursor-pointer"
          >
            <Save class="w-3.5 h-3.5" />
            <span>Draft</span>
          </button>

          <button
            @click="saveAnnotations('rejected')"
            class="px-3 py-1.5 bg-[#C94141] hover:bg-[#A33434] text-white rounded font-semibold flex items-center space-x-1.5 shadow-sm cursor-pointer"
          >
            <XCircle class="w-3.5 h-3.5" />
            <span>Reject</span>
          </button>
        </div>
      </div>

      <!-- Main Canvas & Inspector Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <!-- Interactive Canvas Area -->
        <div class="lg:col-span-3 bg-white border border-[#DDE3EA] rounded-lg p-3 shadow-thin flex flex-col items-center justify-center min-h-[500px] overflow-auto">
          <div v-if="!currentImage" class="text-center p-12 text-[#667085] space-y-2">
            <Camera class="w-10 h-10 mx-auto text-[#667085]/40" />
            <p class="font-semibold text-sm">No captured images available in raw pool.</p>
            <p class="text-xs">Use the Capture Manager tab to start sampling CCTV frames from Dishub Surakarta.</p>
          </div>

          <div v-else class="relative border border-[#DDE3EA] rounded overflow-hidden max-w-full">
            <canvas
              ref="canvasRef"
              @mousedown="handleMouseDown"
              @mousemove="handleMouseMove"
              @mouseup="handleMouseUp"
              :style="{ cursor: canvasCursor }"
              class="max-w-full block select-none"
            ></canvas>
          </div>
          <p class="text-[11px] text-[#667085] mt-2 flex items-center space-x-1.5">
            <Info class="w-3.5 h-3.5 text-[#147D83] shrink-0" />
            <span>💡 <strong>Edit Ukuran BBox:</strong> Klik kotak lalu seret salah satu dari 8 titik handle (lingkaran putih di sudut/tepi). Seret bagian dalam kotak untuk menggeser. Klik area kosong untuk menggambar kotak baru.</span>
          </p>
        </div>

        <!-- Annotation Inspector Panel -->
        <div class="bg-white border border-[#DDE3EA] rounded-lg p-4 shadow-thin space-y-4 text-xs">
          <div>
            <div class="flex justify-between items-center mb-1">
              <h3 class="font-bold text-sm text-[#17202A]">Target Class Selector</h3>
              <button
                type="button"
                id="btn-tambah-kelas"
                @click.stop.prevent="openAddClassModal"
                class="px-2.5 py-1 bg-[#147D83] hover:bg-[#0E5B60] active:scale-95 text-white rounded text-[11px] font-semibold flex items-center space-x-1 transition-all shadow-xs cursor-pointer select-none"
                title="Buka form dialog tambah kelas baru"
              >
                <Plus class="w-3.5 h-3.5" />
                <span>+ Tambah Kelas</span>
              </button>
            </div>
            <p class="text-[11px] text-[#667085] mb-2">Pilih kelas untuk bbox (Khas Indonesia & Kustom)</p>

            <!-- Search box for classes -->
            <div class="relative mb-2">
              <input
                v-model="classSearchQuery"
                type="text"
                placeholder="Cari / filter kelas (mis: becak, angkot)..."
                class="w-full text-[11px] pl-7 pr-2 py-1.5 border border-[#DDE3EA] rounded bg-[#F6F8FB] focus:bg-white text-[#17202A] focus:border-[#173B65] focus:outline-none"
                @keydown.enter.prevent="filteredClasses.length === 0 && classSearchQuery.trim() ? handleQuickAddClass(classSearchQuery) : null"
              />
              <Search class="w-3.5 h-3.5 text-[#667085] absolute left-2 top-2" />
            </div>

            <!-- Empty state with Instant Add Class button if searched class doesn't exist -->
            <div
              v-if="filteredClasses.length === 0"
              class="p-3 text-center bg-[#F6F8FB] border border-dashed border-[#DDE3EA] rounded-lg my-1 space-y-2"
            >
              <p class="text-[11px] text-[#667085]">
                Kelas "<span class="font-bold text-[#17202A]">{{ classSearchQuery }}</span>" belum terdaftar.
              </p>
              <button
                type="button"
                @click="handleQuickAddClass(classSearchQuery)"
                class="px-3 py-1 bg-[#173B65] hover:bg-[#112C4C] text-white rounded text-xs font-semibold flex items-center space-x-1.5 mx-auto shadow-xs cursor-pointer"
              >
                <Plus class="w-3 h-3" />
                <span>+ Tambah Cepat "{{ classSearchQuery }}"</span>
              </button>
            </div>

            <!-- Class buttons list -->
            <div v-else class="grid grid-cols-1 sm:grid-cols-2 gap-1.5 max-h-[220px] overflow-y-auto pr-1">
              <div
                v-for="cls in filteredClasses"
                :key="cls.id"
                @click="setBoxClass(cls.name)"
                class="px-2 py-1.5 rounded border text-left font-semibold flex items-center justify-between cursor-pointer transition-colors group"
                :class="selectedClass === cls.name ? 'border-[#173B65] bg-[#173B65]/5 text-[#173B65] shadow-xs' : 'border-[#DDE3EA] text-[#475467] hover:bg-[#F6F8FB]'"
              >
                <div class="flex items-center space-x-1.5 truncate">
                  <span class="w-2.5 h-2.5 rounded-full shrink-0" :style="{ backgroundColor: cls.color }"></span>
                  <div class="truncate">
                    <span class="capitalize font-bold text-[11px]">{{ cls.name }}</span>
                    <span v-if="cls.label_en" class="text-[10px] text-[#667085] ml-1 font-normal">({{ cls.label_en }})</span>
                  </div>
                </div>
                <div class="flex items-center space-x-1 shrink-0">
                  <Check v-if="selectedClass === cls.name" class="w-3 h-3 text-[#173B65]" />
                  <button
                    v-if="cls.is_custom"
                    type="button"
                    @click.stop="handleDeleteCustomClass(cls.name)"
                    class="opacity-0 group-hover:opacity-100 text-[#C94141] hover:text-[#A33434] p-0.5 rounded cursor-pointer"
                    title="Hapus kelas kustom"
                  >
                    <Trash2 class="w-2.5 h-2.5" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Bounding Boxes List -->
          <div class="border-t border-[#DDE3EA] pt-3">
            <div class="flex justify-between items-center mb-2">
              <span class="font-bold text-[#17202A]">Annotations ({{ currentAnnotations.length }})</span>
              <button
                v-if="selectedBoxIndex !== null"
                @click="removeSelectedBox"
                class="text-[#C94141] hover:underline flex items-center space-x-1 font-semibold"
              >
                <Trash2 class="w-3 h-3" />
                <span>Delete Box</span>
              </button>
            </div>

            <div class="space-y-1.5 max-h-[220px] overflow-y-auto pr-1">
              <div
                v-for="(ann, idx) in currentAnnotations"
                :key="idx"
                @click="selectedBoxIndex = idx; selectedClass = ann.class_name; renderCanvas()"
                class="p-2 rounded border cursor-pointer flex justify-between items-center transition-colors"
                :class="selectedBoxIndex === idx ? 'border-[#173B65] bg-[#173B65]/5 font-semibold text-[#173B65]' : 'border-[#DDE3EA] hover:bg-[#F6F8FB] text-[#17202A]'"
              >
                <div class="flex items-center space-x-2">
                  <span class="w-2 h-2 rounded-full" :style="{ backgroundColor: supportedClasses.find(c => c.name === ann.class_name)?.color || '#16A34A' }"></span>
                  <span class="capitalize font-bold">{{ ann.class_name }}</span>
                </div>
                <div class="font-mono text-[11px] text-[#667085]">
                  {{ (ann.confidence || 1.0).toFixed(2) }}
                </div>
              </div>
            </div>
          </div>

          <!-- Sidecar Metadata Summary -->
          <div v-if="currentImage" class="border-t border-[#DDE3EA] pt-3 space-y-1 text-[11px] text-[#667085]">
            <div class="flex justify-between">
              <span>Resolution:</span>
              <span class="font-mono font-semibold text-[#17202A]">{{ currentImage.resolution || '1920x1080' }}</span>
            </div>
            <div class="flex justify-between">
              <span>Blur Score:</span>
              <span class="font-mono font-semibold text-[#17202A]">{{ currentImage.blur_score || 45.2 }}</span>
            </div>
            <div class="flex justify-between">
              <span>Brightness:</span>
              <span class="font-mono font-semibold text-[#17202A]">{{ currentImage.brightness || 118 }}</span>
            </div>
            <div class="flex justify-between">
              <span>Timestamp:</span>
              <span class="font-mono text-[10px] text-[#17202A]">{{ currentImage.time ? currentImage.time.slice(0, 19).replace('T', ' ') : '-' }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ========================================================= -->
    <!-- TAB 2: DATASET CAPTURE MANAGER -->
    <!-- ========================================================= -->
    <div v-if="activeTab === 'capture'" class="space-y-6">
      <!-- Control Panel -->
      <div class="bg-white border border-[#DDE3EA] rounded-lg p-5 shadow-thin space-y-4">
        <h3 class="font-bold text-sm text-[#17202A]">CCTV Stream Frame Sampler & Quality Filter</h3>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <!-- Camera Select -->
          <div>
            <label class="block text-xs font-semibold text-[#17202A] mb-1">Select CCTV Stream</label>
            <select
              v-model="selectedCameraId"
              class="w-full text-xs p-2 border border-[#DDE3EA] rounded bg-white font-medium text-[#17202A]"
            >
              <option v-for="cam in camerasList" :key="cam.id" :value="cam.id">
                {{ cam.name }} ({{ cam.location }})
              </option>
            </select>
          </div>

          <!-- Interval -->
          <div>
            <label class="block text-xs font-semibold text-[#17202A] mb-1">Capture Interval</label>
            <div class="grid grid-cols-3 gap-2">
              <button
                type="button"
                @click="captureInterval = 5"
                class="py-1.5 border rounded text-xs font-semibold"
                :class="captureInterval === 5 ? 'bg-[#173B65] text-white border-[#173B65]' : 'border-[#DDE3EA] text-[#667085]'"
              >
                5 sec
              </button>
              <button
                type="button"
                @click="captureInterval = 10"
                class="py-1.5 border rounded text-xs font-semibold"
                :class="captureInterval === 10 ? 'bg-[#173B65] text-white border-[#173B65]' : 'border-[#DDE3EA] text-[#667085]'"
              >
                10 sec
              </button>
              <button
                type="button"
                @click="captureInterval = 30"
                class="py-1.5 border rounded text-xs font-semibold"
                :class="captureInterval === 30 ? 'bg-[#173B65] text-white border-[#173B65]' : 'border-[#DDE3EA] text-[#667085]'"
              >
                30 sec
              </button>
            </div>
          </div>

          <!-- Max Storage -->
          <div>
            <label class="block text-xs font-semibold text-[#17202A] mb-1">Max Storage Quota (MB)</label>
            <input
              type="number"
              v-model="maxStorageMb"
              class="w-full text-xs p-2 border border-[#DDE3EA] rounded font-mono"
            />
          </div>

          <!-- Action Buttons -->
          <div class="flex items-end space-x-2">
            <button
              @click="startScheduledCapture"
              :disabled="isStartingJob"
              class="flex-1 py-2 bg-[#238A57] hover:bg-[#1C6E46] text-white rounded text-xs font-semibold flex items-center justify-center space-x-1.5 shadow-sm"
            >
              <Play class="w-3.5 h-3.5" />
              <span>Start Job</span>
            </button>
            <button
              @click="captureSingleNow"
              :disabled="singleCaptureLoading"
              class="flex-1 py-2 bg-[#173B65] hover:bg-[#112C4C] text-white rounded text-xs font-semibold flex items-center justify-center space-x-1.5 shadow-sm"
            >
              <Camera class="w-3.5 h-3.5" />
              <span>Capture 1 Frame</span>
            </button>
          </div>
        </div>

        <!-- Time Window & Peak Hours Schedule Filter -->
        <div class="border-t border-[#DDE3EA] pt-4 mt-2 space-y-3">
          <div class="flex flex-col sm:flex-row justify-between sm:items-center gap-2">
            <div class="flex items-center space-x-2">
              <Clock class="w-4 h-4 text-[#173B65]" />
              <span class="font-bold text-xs text-[#17202A]">Jadwal & Rentang Waktu Pengambilan (Peak Traffic Hours)</span>
              <span class="text-[11px] text-[#667085]">— Ambil dataset hanya pada jam lalu lintas padat tertentu</span>
            </div>

            <!-- Current clock & schedule window indicator -->
            <div class="flex items-center space-x-2 text-[11px]">
              <span class="font-mono font-semibold text-[#173B65] bg-[#F6F8FB] px-2 py-0.5 rounded border border-[#DDE3EA]">
                🕒 {{ currentWibTime || 'WIB' }}
              </span>
              <span
                v-if="timeFilterEnabled"
                class="px-2 py-0.5 rounded font-bold uppercase text-[10px]"
                :class="isCurrentTimeInWindow ? 'bg-[#238A57]/15 text-[#238A57]' : 'bg-[#D58A16]/15 text-[#D58A16]'"
              >
                {{ isCurrentTimeInWindow ? '🟢 Dalam Jendela Jadwal' : '🟡 Menunggu Jadwal' }}
              </span>
            </div>
          </div>

          <!-- Presets Buttons -->
          <div class="flex flex-wrap gap-2 text-xs">
            <button
              type="button"
              @click="applySchedulePreset('morning')"
              class="px-3 py-1.5 rounded border font-semibold flex items-center space-x-1.5 transition-colors"
              :class="selectedPreset === 'morning' && timeFilterEnabled ? 'bg-[#173B65] text-white border-[#173B65] shadow-xs' : 'bg-white border-[#DDE3EA] text-[#475467] hover:bg-[#F6F8FB]'"
            >
              <Sunrise class="w-3.5 h-3.5 text-[#D58A16]" />
              <span>🌅 Pagi Padat (06:30 - 08:30 WIB)</span>
            </button>

            <button
              type="button"
              @click="applySchedulePreset('lunch')"
              class="px-3 py-1.5 rounded border font-semibold flex items-center space-x-1.5 transition-colors"
              :class="selectedPreset === 'lunch' && timeFilterEnabled ? 'bg-[#173B65] text-white border-[#173B65] shadow-xs' : 'bg-white border-[#DDE3EA] text-[#475467] hover:bg-[#F6F8FB]'"
            >
              <Sun class="w-3.5 h-3.5 text-[#EAB308]" />
              <span>☀️ Siang Padat (11:30 - 13:30 WIB)</span>
            </button>

            <button
              type="button"
              @click="applySchedulePreset('evening')"
              class="px-3 py-1.5 rounded border font-semibold flex items-center space-x-1.5 transition-colors"
              :class="selectedPreset === 'evening' && timeFilterEnabled ? 'bg-[#173B65] text-white border-[#173B65] shadow-xs' : 'bg-white border-[#DDE3EA] text-[#475467] hover:bg-[#F6F8FB]'"
            >
              <Sunset class="w-3.5 h-3.5 text-[#EA580C]" />
              <span>🌆 Sore Padat (16:30 - 19:30 WIB)</span>
            </button>

            <button
              type="button"
              @click="applySchedulePreset('night')"
              class="px-3 py-1.5 rounded border font-semibold flex items-center space-x-1.5 transition-colors"
              :class="selectedPreset === 'night' && timeFilterEnabled ? 'bg-[#173B65] text-white border-[#173B65] shadow-xs' : 'bg-white border-[#DDE3EA] text-[#475467] hover:bg-[#F6F8FB]'"
            >
              <Moon class="w-3.5 h-3.5 text-[#6366F1]" />
              <span>🌙 Malam Low-Light (21:00 - 04:00 WIB)</span>
            </button>

            <button
              type="button"
              @click="applySchedulePreset('full')"
              class="px-3 py-1.5 rounded border font-semibold flex items-center space-x-1.5 transition-colors"
              :class="!timeFilterEnabled ? 'bg-[#173B65] text-white border-[#173B65] shadow-xs' : 'bg-white border-[#DDE3EA] text-[#475467] hover:bg-[#F6F8FB]'"
            >
              <span>🔄 24 Jam Penuh</span>
            </button>

            <button
              type="button"
              @click="applySchedulePreset('custom')"
              class="px-3 py-1.5 rounded border font-semibold flex items-center space-x-1.5 transition-colors"
              :class="selectedPreset === 'custom' && timeFilterEnabled ? 'bg-[#173B65] text-white border-[#173B65] shadow-xs' : 'bg-white border-[#DDE3EA] text-[#475467] hover:bg-[#F6F8FB]'"
            >
              <Settings2 class="w-3.5 h-3.5" />
              <span>⏱️ Kustom Jam</span>
            </button>
          </div>

          <!-- Custom Time Inputs -->
          <div v-if="timeFilterEnabled" class="p-3 bg-[#F6F8FB] border border-[#DDE3EA] rounded-md flex flex-wrap items-center gap-4 text-xs">
            <div class="flex items-center space-x-2">
              <label class="font-semibold text-[#17202A]">Mulai Jam (WIB):</label>
              <input
                type="time"
                v-model="startTime"
                @change="applySchedulePreset('custom')"
                class="px-2 py-1 border border-[#DDE3EA] rounded bg-white font-mono font-semibold text-[#17202A]"
              />
            </div>

            <div class="flex items-center space-x-2">
              <label class="font-semibold text-[#17202A]">Sampai Jam (WIB):</label>
              <input
                type="time"
                v-model="endTime"
                @change="applySchedulePreset('custom')"
                class="px-2 py-1 border border-[#DDE3EA] rounded bg-white font-mono font-semibold text-[#17202A]"
              />
            </div>

            <div class="text-[#667085] text-[11px]">
              Label: <strong class="text-[#173B65]">{{ scheduleLabel }}</strong>
            </div>
          </div>
        </div>

        <div v-if="captureSuccessMsg" class="p-2.5 rounded bg-[#238A57]/10 text-[#238A57] text-xs font-semibold border border-[#238A57]/20">
          ✓ {{ captureSuccessMsg }}
        </div>
      </div>

      <!-- Active Capture Jobs Table -->
      <div class="bg-white border border-[#DDE3EA] rounded-lg shadow-thin overflow-hidden">
        <div class="px-5 py-3 border-b border-[#DDE3EA] bg-[#F6F8FB] flex justify-between items-center">
          <h3 class="font-bold text-xs text-[#17202A] uppercase">Active Scheduled Capture Workers</h3>
          <span class="text-xs text-[#667085] font-mono">{{ captureJobs.length }} Active</span>
        </div>

        <table class="w-full text-left text-xs">
          <thead class="bg-[#F6F8FB]/50 text-[#667085] font-semibold border-b border-[#DDE3EA]">
            <tr>
              <th class="px-5 py-2.5">Camera Name</th>
              <th class="px-5 py-2.5">Jadwal Rentang Jam</th>
              <th class="px-5 py-2.5">Interval</th>
              <th class="px-5 py-2.5">Total Sampled</th>
              <th class="px-5 py-2.5">Accepted</th>
              <th class="px-5 py-2.5">Rejected</th>
              <th class="px-5 py-2.5">Status</th>
              <th class="px-5 py-2.5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-[#DDE3EA]">
            <tr v-for="job in captureJobs" :key="job.camera_id" class="hover:bg-[#F6F8FB]/40">
              <td class="px-5 py-3 font-bold text-[#17202A]">{{ job.camera_name }}</td>
              <td class="px-5 py-3 font-semibold text-[#173B65]">
                <span class="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-[#173B65]/5 border border-[#173B65]/15 text-[11px]">
                  <Clock class="w-3 h-3 text-[#147D83]" />
                  <span>{{ job.schedule_label || '24 Jam Penuh' }}</span>
                </span>
              </td>
              <td class="px-5 py-3 font-mono text-[#667085]">{{ job.interval_seconds }}s</td>
              <td class="px-5 py-3 font-mono font-bold text-[#17202A]">{{ job.total_captured }}</td>
              <td class="px-5 py-3 font-mono font-bold text-[#238A57]">{{ job.accepted_count }}</td>
              <td class="px-5 py-3 font-mono text-[#C94141] font-semibold">{{ job.rejected_count }}</td>
              <td class="px-5 py-3">
                <span
                  class="px-2 py-0.5 rounded text-[10px] font-semibold block w-fit"
                  :class="{
                    'bg-[#238A57]/10 text-[#238A57]': job.is_running && job.is_inside_window,
                    'bg-[#D58A16]/10 text-[#D58A16]': job.is_running && !job.is_inside_window,
                    'bg-[#667085]/10 text-[#667085]': !job.is_running
                  }"
                >
                  {{ !job.is_running ? 'STOPPED' : (job.is_inside_window ? 'AKTIF (IN WINDOW)' : 'MENUNGGU JADWAL') }}
                </span>
              </td>
              <td class="px-5 py-3 text-right">
                <button
                  v-if="job.is_running"
                  @click="stopScheduledCapture(job.camera_id)"
                  class="px-2.5 py-1 bg-[#C94141]/10 text-[#C94141] hover:bg-[#C94141]/20 rounded font-semibold text-xs"
                >
                  Stop
                </button>
              </td>
            </tr>
            <tr v-if="captureJobs.length === 0">
              <td colspan="8" class="px-5 py-6 text-center text-[#667085]">
                No active capture jobs running. Select a camera and schedule above and click Start Job.
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Raw Captured Frames Gallery -->
      <div class="space-y-3">
        <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
          <div>
            <h3 class="font-bold text-sm text-[#17202A]">Recent Captured Frames (Raw Dataset Pool)</h3>
            <span class="text-xs text-[#667085]">Showing latest {{ rawImages.length }} frames • Klik gambar untuk preview & anotasi</span>
          </div>
          <div class="flex items-center space-x-2">
            <button
              type="button"
              @click="triggerBatchAutoAnnotate"
              :disabled="isBatchAutoAnnotating"
              class="px-2.5 py-1.5 bg-[#147D83] hover:bg-[#0E5B60] disabled:opacity-50 text-white rounded font-semibold text-xs flex items-center space-x-1 shadow-xs cursor-pointer"
            >
              <Sparkles class="w-3.5 h-3.5" />
              <span>{{ isBatchAutoAnnotating ? 'Inference...' : '⚡ Auto-Annotate Semua' }}</span>
            </button>
            <button
              type="button"
              @click="triggerBatchApprove"
              :disabled="isBatchApproving"
              class="px-2.5 py-1.5 bg-[#238A57] hover:bg-[#1C6E46] disabled:opacity-50 text-white rounded font-semibold text-xs flex items-center space-x-1 shadow-xs cursor-pointer"
            >
              <CheckCircle2 class="w-3.5 h-3.5" />
              <span>{{ isBatchApproving ? 'Menyimpan...' : '✓ Approve Semua' }}</span>
            </button>
          </div>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <div
            v-for="(img, idx) in rawImages"
            :key="idx"
            @click="openImagePreview(img, idx)"
            class="bg-white border border-[#DDE3EA] rounded overflow-hidden shadow-thin cursor-pointer hover:border-[#173B65] hover:shadow-md transition-all group relative select-none"
          >
            <div class="aspect-video bg-[#17202A] relative overflow-hidden">
              <img
                :src="img.relative_url || `/storage/datasets/raw/${img.file_path?.split('storage/datasets/raw/')[1]}`"
                class="w-full h-full object-cover transition-transform group-hover:scale-105 duration-200"
                loading="lazy"
                alt="Captured CCTV frame"
              />
              <!-- Quality Tag -->
              <span class="absolute top-1 right-1 px-1.5 py-0.5 rounded text-[8px] font-bold bg-[#238A57] text-white shadow-xs">
                PASS
              </span>

              <!-- Review Status Tag -->
              <span
                class="absolute top-1 left-1 px-1.5 py-0.5 rounded text-[8px] font-bold uppercase shadow-xs"
                :class="{
                  'bg-[#238A57] text-white': img.review_status === 'approved',
                  'bg-[#D58A16] text-white': img.review_status === 'pseudo_annotated',
                  'bg-[#667085] text-white': !img.review_status || img.review_status === 'unreviewed'
                }"
              >
                {{ img.review_status === 'approved' ? '✓ APPROVED' : (img.review_status === 'pseudo_annotated' ? '⚡ PSEUDO' : '⏳ UNREVIEWED') }}
              </span>

              <!-- Hover overlay with 2 action buttons -->
              <div class="absolute inset-0 bg-[#17202A]/70 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center p-2 gap-1.5">
                <button
                  type="button"
                  @click.stop="openImagePreview(img, idx)"
                  class="w-full py-1 bg-white text-[#17202A] hover:bg-[#F6F8FB] rounded text-[10px] font-bold shadow-xs flex items-center justify-center space-x-1 cursor-pointer"
                >
                  <Eye class="w-3 h-3 text-[#147D83]" />
                  <span>Lihat Detail</span>
                </button>
                <button
                  type="button"
                  @click.stop="jumpToAnnotationStudio(idx)"
                  class="w-full py-1 bg-[#173B65] text-white hover:bg-[#112C4C] rounded text-[10px] font-bold shadow-xs flex items-center justify-center space-x-1 cursor-pointer"
                >
                  <Tag class="w-3 h-3 text-white" />
                  <span>Anotasi / BBox</span>
                </button>
              </div>
            </div>
            <div class="p-2 space-y-0.5 text-[10px] text-[#667085]">
              <p class="font-bold text-[#17202A] truncate">{{ img.camera_name || 'Camera' }}</p>
              <div class="flex justify-between font-mono">
                <span>Blur: {{ Math.round(img.blur_score || 45) }}</span>
                <span>Bri: {{ Math.round(img.brightness || 120) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ========================================================= -->
    <!-- TAB 3: DATASET VERSIONS & YOLO EXPORT -->
    <!-- ========================================================= -->
    <div v-if="activeTab === 'datasets'" class="space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Dataset Overview Card -->
        <div class="bg-white border border-[#DDE3EA] rounded-lg p-5 shadow-thin space-y-4">
          <div class="flex justify-between items-start">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 rounded bg-[#173B65]/10 text-[#173B65] flex items-center justify-center font-bold">
                <Database class="w-5 h-5" />
              </div>
              <div>
                <h3 class="font-bold text-sm text-[#17202A]">Dishub Surakarta Traffic Dataset</h3>
                <p class="text-xs text-[#667085]">Version: v1.0 • Fixed CCTV ATCS Surakarta</p>
              </div>
            </div>
            <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-[#238A57]/10 text-[#238A57] uppercase">
              ACTIVE
            </span>
          </div>

          <div class="p-3 bg-[#F6F8FB] rounded space-y-3 text-xs">
            <div class="flex justify-between font-semibold text-[#17202A]">
              <span>Curated Frames</span>
              <span class="font-mono tabular-nums">{{ rawImages.length || 8500 }} frames</span>
            </div>

            <!-- Class Distribution Breakdown -->
            <div class="space-y-1.5 pt-1">
              <span class="text-[11px] font-semibold text-[#667085] uppercase">Class Distribution (Ground Truth / Approved)</span>
              <div class="space-y-1 text-[11px]">
                <div class="flex justify-between">
                  <span class="text-[#16A34A] font-semibold">Car</span>
                  <span class="font-mono font-bold text-[#17202A]">45,000 (52%)</span>
                </div>
                <div class="w-full bg-[#DDE3EA] h-1.5 rounded-full overflow-hidden">
                  <div class="bg-[#16A34A] h-full" style="width: 52%"></div>
                </div>

                <div class="flex justify-between pt-1">
                  <span class="text-[#D97706] font-semibold">Motorcycle</span>
                  <span class="font-mono font-bold text-[#17202A]">22,000 (26%)</span>
                </div>
                <div class="w-full bg-[#DDE3EA] h-1.5 rounded-full overflow-hidden">
                  <div class="bg-[#D97706] h-full" style="width: 26%"></div>
                </div>

                <div class="flex justify-between pt-1">
                  <span class="text-[#DC2626] font-semibold">Bus</span>
                  <span class="font-mono font-bold text-[#17202A]">300 (4%)</span>
                </div>
                <div class="w-full bg-[#DDE3EA] h-1.5 rounded-full overflow-hidden">
                  <div class="bg-[#DC2626] h-full" style="width: 4%"></div>
                </div>

                <div class="flex justify-between pt-1">
                  <span class="text-[#7C3AED] font-semibold">Truck</span>
                  <span class="font-mono font-bold text-[#17202A]">500 (6%)</span>
                </div>
                <div class="w-full bg-[#DDE3EA] h-1.5 rounded-full overflow-hidden">
                  <div class="bg-[#7C3AED] h-full" style="width: 6%"></div>
                </div>

                <div class="flex justify-between pt-1">
                  <span class="text-[#2563EB] font-semibold">Person</span>
                  <span class="font-mono font-bold text-[#17202A]">1,200 (12%)</span>
                </div>
                <div class="w-full bg-[#DDE3EA] h-1.5 rounded-full overflow-hidden">
                  <div class="bg-[#2563EB] h-full" style="width: 12%"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- YOLO26 Export Config Card -->
        <div class="bg-white border border-[#DDE3EA] rounded-lg p-5 shadow-thin space-y-4">
          <div>
            <h3 class="font-bold text-sm text-[#17202A]">Export Dataset for YOLO26 Training</h3>
            <p class="text-xs text-[#667085]">Non-random temporal split (by date/camera) to prevent frame leakage</p>
          </div>

          <div class="space-y-3 text-xs">
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">Dataset Name / Tag</label>
              <input
                type="text"
                v-model="datasetName"
                class="w-full p-2 border border-[#DDE3EA] rounded font-mono text-xs"
              />
            </div>

            <!-- Splits -->
            <div class="grid grid-cols-3 gap-3">
              <div>
                <label class="block text-[#667085] mb-1">Train (%)</label>
                <input
                  type="number"
                  v-model="trainSplit"
                  class="w-full p-1.5 border border-[#DDE3EA] rounded font-mono font-bold text-center"
                />
                <span class="text-[10px] text-[#667085] block text-center mt-1">Days 1 - 5</span>
              </div>
              <div>
                <label class="block text-[#667085] mb-1">Val (%)</label>
                <input
                  type="number"
                  v-model="valSplit"
                  class="w-full p-1.5 border border-[#DDE3EA] rounded font-mono font-bold text-center"
                />
                <span class="text-[10px] text-[#667085] block text-center mt-1">Day 6</span>
              </div>
              <div>
                <label class="block text-[#667085] mb-1">Test (%)</label>
                <input
                  type="number"
                  v-model="testSplit"
                  class="w-full p-1.5 border border-[#DDE3EA] rounded font-mono font-bold text-center"
                />
                <span class="text-[10px] text-[#667085] block text-center mt-1">Day 7</span>
              </div>
            </div>

            <div class="p-3 bg-[#F6F8FB] rounded text-[11px] text-[#667085] space-y-1">
              <span class="font-semibold text-[#17202A] block">Output Structure:</span>
              <p class="font-mono">storage/datasets/exported/{{ datasetName }}/</p>
              <p class="font-mono">├── images/ (train, val, test)</p>
              <p class="font-mono">├── labels/ (train, val, test)</p>
              <p class="font-mono">└── data.yaml</p>
            </div>

            <button
              @click="triggerYoloExport"
              :disabled="isExporting"
              class="w-full py-2 bg-[#173B65] hover:bg-[#112C4C] text-white rounded font-semibold flex items-center justify-center space-x-2 shadow-sm"
            >
              <Download class="w-4 h-4" />
              <span>{{ isExporting ? 'Generating Splits...' : 'Export YOLO26 Dataset' }}</span>
            </button>

            <div v-if="exportResult" class="p-3 rounded text-xs border" :class="exportResult.success ? 'bg-[#238A57]/10 border-[#238A57]/30 text-[#238A57]' : 'bg-[#C94141]/10 border-[#C94141]/30 text-[#C94141]'">
              <p class="font-bold">{{ exportResult.success ? '✓ Dataset Export Completed!' : '✗ Export Failed' }}</p>
              <p v-if="exportResult.yaml_path" class="font-mono text-[11px] mt-1 text-[#17202A]">Config: {{ exportResult.yaml_path }}</p>
              <div v-if="exportResult.splits" class="flex space-x-3 mt-1 font-mono text-[11px]">
                <span>Train: {{ exportResult.splits.train }}</span>
                <span>Val: {{ exportResult.splits.val }}</span>
                <span>Test: {{ exportResult.splits.test }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ========================================================= -->
    <!-- TAB 4: LICENSE PLATE & OCR PREPARATION -->
    <!-- ========================================================= -->
    <div v-if="activeTab === 'ocr'" class="space-y-6">
      <div class="bg-white border border-[#DDE3EA] rounded-lg p-5 shadow-thin space-y-4">
        <div class="flex justify-between items-center">
          <div>
            <h3 class="font-bold text-sm text-[#17202A]">Indonesian License Plate & OCR Dataset Manager</h3>
            <p class="text-xs text-[#667085]">Vehicle crop → Plate detector crop → Character label annotation</p>
          </div>
          <button
            @click="exportOcrDataset"
            :disabled="isExportingOcr"
            class="px-3 py-1.5 bg-[#147D83] hover:bg-[#0E5B60] text-white rounded text-xs font-semibold flex items-center space-x-1.5 shadow-sm"
          >
            <Download class="w-3.5 h-3.5" />
            <span>{{ isExportingOcr ? 'Exporting...' : 'Export OCR Lists (PP-OCR/CRNN)' }}</span>
          </button>
        </div>

        <div v-if="ocrExportMsg" class="p-2.5 rounded bg-[#238A57]/10 text-[#238A57] text-xs font-semibold border border-[#238A57]/20">
          ✓ {{ ocrExportMsg }}
        </div>

        <!-- Table of Plate Records -->
        <div class="border border-[#DDE3EA] rounded overflow-hidden">
          <table class="w-full text-left text-xs">
            <thead class="bg-[#F6F8FB] text-[#667085] font-semibold border-b border-[#DDE3EA]">
              <tr>
                <th class="px-4 py-2.5">Crop Image</th>
                <th class="px-4 py-2.5">Filename</th>
                <th class="px-4 py-2.5">Ground Truth Text</th>
                <th class="px-4 py-2.5">Vehicle Class</th>
                <th class="px-4 py-2.5">Confidence</th>
                <th class="px-4 py-2.5">Date</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-[#DDE3EA]">
              <tr v-for="(rec, idx) in plateRecords" :key="idx" class="hover:bg-[#F6F8FB]/40">
                <td class="px-4 py-2">
                  <div class="w-16 h-8 bg-[#17202A] rounded overflow-hidden border border-[#DDE3EA] flex items-center justify-center text-[10px] text-white">
                    <img v-if="rec.exists" :src="rec.url" class="w-full h-full object-cover" />
                    <span v-else>PLATE</span>
                  </div>
                </td>
                <td class="px-4 py-2 font-mono text-[11px] text-[#667085]">{{ rec.image }}</td>
                <td class="px-4 py-2 font-mono font-bold text-[#173B65] text-sm">{{ rec.text || 'AD 1234 XYZ' }}</td>
                <td class="px-4 py-2 capitalize font-semibold text-[#17202A]">{{ rec.vehicle_class }}</td>
                <td class="px-4 py-2 font-mono text-[#238A57] font-semibold">{{ rec.confidence?.toFixed(2) }}</td>
                <td class="px-4 py-2 font-mono text-[11px] text-[#667085]">{{ rec.date ? rec.date.slice(0, 10) : '-' }}</td>
              </tr>
              <tr v-if="plateRecords.length === 0">
                <td colspan="6" class="px-4 py-6 text-center text-[#667085]">
                  No plate crops yet. Plates detected from live vehicles are stored here automatically.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ========================================================= -->
    <!-- MODAL: TAMBAH KELAS BARU (KHAS INDONESIA / KUSTOM) -->
    <!-- ========================================================= -->
    <Teleport to="body">
      <div
        v-if="showAddClassModal"
        class="fixed inset-0 z-[9999] bg-[#17202A]/70 backdrop-blur-xs flex items-center justify-center p-4 animate-in fade-in duration-200"
        @click.self="closeAddClassModal"
      >
        <div class="bg-white rounded-xl shadow-2xl border border-[#DDE3EA] max-w-md w-full overflow-hidden animate-in zoom-in-95 duration-150">
          <!-- Modal Header -->
          <div class="px-5 py-4 border-b border-[#DDE3EA] flex justify-between items-center bg-[#F6F8FB]">
            <div class="flex items-center space-x-2">
              <span class="w-3 h-3 rounded-full bg-[#147D83]"></span>
              <h3 class="font-bold text-sm text-[#17202A]">Tambah Kelas Bounding Box Baru</h3>
            </div>
            <button
              type="button"
              @click="closeAddClassModal"
              class="text-[#667085] hover:text-[#17202A] p-1 rounded-md hover:bg-[#DDE3EA] cursor-pointer"
            >
              <X class="w-4 h-4" />
            </button>
          </div>

          <!-- Modal Body Form -->
          <form @submit.prevent="handleAddCustomClass" class="p-5 space-y-4 text-xs">
            <div v-if="addClassError" class="p-2.5 rounded bg-[#C94141]/10 text-[#C94141] font-semibold border border-[#C94141]/20">
              ⚠️ {{ addClassError }}
            </div>

            <div>
              <label class="block font-semibold text-[#17202A] mb-1">
                Nama Kelas (ID / Slug YOLO) <span class="text-[#C94141]">*</span>
              </label>
              <input
                v-model="newClassName"
                type="text"
                placeholder="misal: becak, dokar, tossa, bentor, gerobak"
                class="w-full p-2 border border-[#DDE3EA] rounded font-mono text-xs focus:border-[#173B65] focus:outline-none"
                required
                autofocus
              />
              <p class="text-[10px] text-[#667085] mt-1">Gunakan huruf kecil tanpa spasi (misal: <code>becak_motor</code> atau <code>gerobak</code>).</p>
            </div>

            <div>
              <label class="block font-semibold text-[#17202A] mb-1">
                Bahasa Inggris / English Translation Label
              </label>
              <input
                v-model="newClassLabelEn"
                type="text"
                placeholder="misal: Pedicab, Auto-Rickshaw, Horse Carriage, Minibus"
                class="w-full p-2 border border-[#DDE3EA] rounded text-xs focus:border-[#173B65] focus:outline-none"
              />
              <p class="text-[10px] text-[#667085] mt-1">Istilah standar internasional untuk publikasi ilmiah & dataset YOLO.</p>
            </div>

            <div>
              <label class="block font-semibold text-[#17202A] mb-1">
                Label Indonesia (Keterangan)
              </label>
              <input
                v-model="newClassLabelId"
                type="text"
                placeholder="misal: Becak Kayuh Khas Solo, Gerobak Pedagang"
                class="w-full p-2 border border-[#DDE3EA] rounded text-xs focus:border-[#173B65] focus:outline-none"
              />
            </div>

            <!-- Color selection -->
            <div>
              <label class="block font-semibold text-[#17202A] mb-1">
                Warna Bounding Box
              </label>
              <div class="flex items-center space-x-3">
                <input
                  type="color"
                  v-model="newClassColor"
                  class="w-9 h-8 p-0.5 border border-[#DDE3EA] rounded cursor-pointer"
                />
                <div class="flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    v-for="color in ['#EA580C', '#CA8A04', '#0891B2', '#9333EA', '#4B5563', '#DB2777', '#16A34A', '#2563EB', '#DC2626']"
                    :key="color"
                    @click="newClassColor = color"
                    class="w-6 h-6 rounded-full border border-white shadow-xs transition-transform hover:scale-110 cursor-pointer"
                    :style="{ backgroundColor: color }"
                  ></button>
                </div>
              </div>
            </div>

            <!-- Modal Footer -->
            <div class="border-t border-[#DDE3EA] pt-4 flex justify-end space-x-2">
              <button
                type="button"
                @click="closeAddClassModal"
                class="px-3 py-1.5 border border-[#DDE3EA] text-[#667085] hover:bg-[#F6F8FB] rounded font-semibold cursor-pointer"
              >
                Batal
              </button>
              <button
                type="submit"
                :disabled="isSavingClass"
                class="px-4 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded font-semibold flex items-center space-x-1.5 shadow-sm cursor-pointer"
              >
                <Check v-if="!isSavingClass" class="w-3.5 h-3.5" />
                <span>{{ isSavingClass ? 'Menyimpan...' : 'Simpan Kelas Baru' }}</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    </Teleport>

    <!-- ========================================================= -->
    <!-- MODAL: PREVIEW GAMBAR HASIL CAPTURE & DETAIL ANOTASI -->
    <!-- ========================================================= -->
    <Teleport to="body">
      <div
        v-if="previewModalImage"
        class="fixed inset-0 z-[9999] bg-[#17202A]/75 backdrop-blur-xs flex items-center justify-center p-4 animate-in fade-in duration-200"
        @click.self="closeImagePreview"
      >
        <div class="bg-white rounded-xl shadow-2xl border border-[#DDE3EA] max-w-3xl w-full overflow-hidden flex flex-col max-h-[90vh] animate-in zoom-in-95 duration-150">
          <!-- Modal Header -->
          <div class="px-5 py-3.5 border-b border-[#DDE3EA] flex justify-between items-center bg-[#F6F8FB]">
            <div class="flex items-center space-x-2">
              <Camera class="w-4 h-4 text-[#147D83]" />
              <h3 class="font-bold text-sm text-[#17202A]">
                Detail Frame CCTV: {{ previewModalImage.camera_name || 'Dishub Surakarta' }}
              </h3>
            </div>
            <button
              type="button"
              @click="closeImagePreview"
              class="text-[#667085] hover:text-[#17202A] p-1 rounded-md hover:bg-[#DDE3EA] cursor-pointer"
            >
              <X class="w-4 h-4" />
            </button>
          </div>

          <!-- Modal Body -->
          <div class="p-5 overflow-y-auto space-y-4">
            <!-- Full Frame Preview -->
            <div class="relative bg-[#17202A] rounded-lg overflow-hidden border border-[#DDE3EA] flex items-center justify-center min-h-[300px]">
              <img
                :src="previewModalImage.relative_url || `/storage/datasets/raw/${previewModalImage.file_path?.split('storage/datasets/raw/')[1]}`"
                class="max-w-full max-h-[55vh] object-contain mx-auto"
                alt="Captured Frame Preview"
              />
              <span class="absolute top-3 right-3 px-2.5 py-1 rounded text-xs font-bold bg-[#238A57] text-white shadow-md flex items-center space-x-1">
                <CheckCircle2 class="w-3.5 h-3.5" />
                <span>{{ previewModalImage.quality_status?.toUpperCase() || 'ACCEPTED' }}</span>
              </span>
            </div>

            <!-- Metadata info grid -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-[#F6F8FB] p-3.5 rounded-lg border border-[#DDE3EA] text-xs">
              <div>
                <span class="text-[#667085] block text-[11px]">Kamera ATCS:</span>
                <span class="font-bold text-[#17202A]">{{ previewModalImage.camera_name || '-' }}</span>
              </div>
              <div>
                <span class="text-[#667085] block text-[11px]">Waktu Pengambilan:</span>
                <span class="font-mono text-[#17202A] text-[11px]">
                  {{ previewModalImage.time ? new Date(previewModalImage.time).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' WIB' : '-' }}
                </span>
              </div>
              <div>
                <span class="text-[#667085] block text-[11px]">Skor Ketajaman (Blur):</span>
                <span class="font-mono font-bold text-[#238A57]">{{ Math.round(previewModalImage.blur_score || 0) }} (Pass &gt; 30)</span>
              </div>
              <div>
                <span class="text-[#667085] block text-[11px]">Kecerahan (Bri):</span>
                <span class="font-mono font-bold text-[#173B65]">{{ Math.round(previewModalImage.brightness || 0) }} (35 - 230)</span>
              </div>
            </div>
          </div>

          <!-- Modal Footer with Action Buttons -->
          <div class="px-5 py-3.5 border-t border-[#DDE3EA] bg-[#F6F8FB] flex flex-wrap justify-between items-center gap-2">
            <span class="text-[11px] text-[#667085]">
              Frame {{ previewModalIndex + 1 }} dari {{ rawImages.length }} di raw pool
            </span>
            <div class="flex items-center space-x-2">
              <button
                type="button"
                @click="closeImagePreview"
                class="px-3.5 py-1.5 border border-[#DDE3EA] text-[#667085] hover:bg-white rounded font-semibold text-xs cursor-pointer"
              >
                Tutup
              </button>
              <button
                type="button"
                @click="jumpToAnnotationStudio(previewModalIndex)"
                class="px-4 py-1.5 bg-[#173B65] hover:bg-[#112C4C] text-white rounded font-semibold text-xs flex items-center space-x-1.5 shadow-sm cursor-pointer"
              >
                <Tag class="w-3.5 h-3.5" />
                <span>Buka di Annotation Studio & BBox Editor</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
