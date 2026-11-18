export interface Role {
  id: string
  name: string
  description: string
}

export interface User {
  id: string
  username: string
  email: string
  full_name: string
  is_active: boolean
  is_superuser: boolean
  roles: Role[]
}

export interface Camera {
  id: string
  name: string
  model: string
  ip_address: string
  rtsp_port: number
  http_port: number
  channel: string
  rtsp_main_stream: string
  rtsp_substream?: string
  isapi_snapshot_url: string
  username: string
  location: string
  installation_height_m: number
  orientation: string
  expected_lane: string
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface CameraHealth {
  id: string
  camera_id: string
  is_online: boolean
  resolution: string
  codec: string
  reported_fps: number
  measured_fps: number
  latest_frame_age_ms: number
  dropped_frames: number
  reconnect_count: number
  snapshot_health: string
  rtsp_health: string
  last_error?: string
  updated_at: string
}

export interface ModelArtifact {
  id: string
  name: string
  version: string
  task: string
  source: string
  source_url: string
  license: string
  checksum_sha256: string
  file_path?: string
  file_format: string
  model_size_bytes: number
  input_size: number
  classes: string[]
  framework: string
  runtime: string
  precision: string
  validation_status: string
  trust_level: 'official' | 'community_unverified' | 'community_validated' | 'local_trained' | 'local_validated'
  is_active: boolean
  installation_date: string
  activation_date?: string
  notes?: string
}

export interface PipelinePreset {
  id: string
  preset_id: string
  name: string
  description: string
  detector_model_id?: string
  plate_model_id?: string
  ocr_model_id?: string
  enabled_classes: number[]
  confidence_threshold: number
  iou_threshold: number
  use_tracking: boolean
  use_rectification: boolean
  use_temporal_voting: boolean
  is_active: boolean
}

export interface RecognitionEvent {
  id: string
  camera_id: string
  pipeline_id: string
  track_id?: string
  timestamp: string
  vehicle_class: string
  vehicle_confidence: number
  vehicle_bbox: number[]
  vehicle_crop_path?: string
  plate_text?: string
  plate_confidence: number
  plate_detector_confidence: number
  plate_bbox?: number[]
  plate_crop_path?: string
  annotated_frame_path?: string
  raw_frame_path?: string
  authorization_status: 'authorized' | 'rejected' | 'unrecognized' | 'manual_review'
  gate_action: 'none' | 'requested_open' | 'blocked'
  review_status: 'unreviewed' | 'approved' | 'rejected' | 'corrected'
  corrected_plate_text?: string
  reviewed_by?: string
  detection_latency_ms: number
  recognition_latency_ms: number
  total_latency_ms: number
  model_versions: Record<string, string>
  trust_level: string
  temporal_votes?: any[]
}

export interface Device {
  id: string
  device_id: string
  device_type: string
  name: string
  ip_address?: string
  is_simulation: boolean
  status: string
  last_heartbeat?: string
  capabilities: Record<string, any>
}

export interface Gate {
  id: string
  gate_name: string
  lane_name: string
  device_id?: string
  current_state: string
  mode: string
  trigger_sensor: boolean
  safety_beam: boolean
  open_limit_switch: boolean
  close_limit_switch: boolean
  relay_state: boolean
  last_command?: string
  last_fault?: string
  updated_at: string
}

export interface SystemMetrics {
  cpu_usage_pct: number
  ram_usage_pct: number
  ram_used_mb: number
  gpu_usage_pct?: number
  gpu_memory_used_mb?: number
  gpu_temperature_c?: number
  disk_usage_pct: number
  inference_fps: number
  inference_latency_ms: number
  queue_depth: number
  dropped_frames: number
  timestamp: string
}

export interface HealthStatus {
  status: string
  version: string
  checks: Record<string, {
    name: string
    status: string
    latency_ms?: number
    message?: string
  }>
  timestamp: string
}

export interface AuditLog {
  id: string
  username: string
  action: string
  target_type: string
  target_id?: string
  ip_address?: string
  details: Record<string, any>
  timestamp: string
}
