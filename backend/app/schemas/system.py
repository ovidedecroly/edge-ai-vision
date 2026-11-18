from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

class SystemMetricsResponse(BaseModel):
    cpu_usage_pct: float
    ram_usage_pct: float
    ram_used_mb: float
    ram_total_mb: float
    gpu_usage_pct: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    disk_usage_pct: float
    disk_used_gb: float
    disk_total_gb: float
    inference_fps: float = 0.0
    inference_latency_ms: float = 0.0
    queue_depth: int = 0
    dropped_frames: int = 0
    uptime_seconds: float = 0.0
    timestamp: datetime

class HealthCheckItem(BaseModel):
    name: str
    status: str # ok, warning, error
    message: Optional[str] = None
    latency_ms: Optional[float] = None

class HealthStatusResponse(BaseModel):
    status: str # ok, degraded, unhealthy
    version: str = "1.0.0"
    checks: Dict[str, HealthCheckItem]
    timestamp: datetime

class HardwareInfoResponse(BaseModel):
    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    gpu_name: Optional[str] = None
    cuda_available: bool = False
    driver_version: Optional[str] = None
    os_info: str
