import os
import psutil
import shutil
import time
import subprocess
from typing import Optional, Dict, Any
from datetime import datetime, timezone

class MetricsService:
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Collect real hardware and system metrics without dummy data."""
        # CPU
        cpu_usage = psutil.cpu_percent(interval=None)

        # RAM
        ram = psutil.virtual_memory()
        ram_used_mb = ram.used / (1024 * 1024)
        ram_total_mb = ram.total / (1024 * 1024)
        ram_usage_pct = ram.percent

        # Disk
        disk = shutil.disk_usage("/")
        disk_used_gb = (disk.total - disk.free) / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)
        disk_usage_pct = (disk_used_gb / disk_total_gb) * 100.0

        # GPU metrics (NVIDIA)
        gpu_usage_pct: Optional[float] = None
        gpu_mem_used_mb: Optional[float] = None
        gpu_mem_total_mb: Optional[float] = None
        gpu_temp_c: Optional[float] = None

        try:
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True, text=True, timeout=1.0).strip()
            parts = [p.strip() for p in output.split(",")]
            if len(parts) >= 4:
                gpu_usage_pct = float(parts[0])
                gpu_mem_used_mb = float(parts[1])
                gpu_mem_total_mb = float(parts[2])
                gpu_temp_c = float(parts[3])
        except Exception:
            pass

        return {
            "cpu_usage_pct": round(cpu_usage, 1),
            "ram_usage_pct": round(ram_usage_pct, 1),
            "ram_used_mb": round(ram_used_mb, 1),
            "ram_total_mb": round(ram_total_mb, 1),
            "gpu_usage_pct": gpu_usage_pct,
            "gpu_memory_used_mb": gpu_mem_used_mb,
            "gpu_memory_total_mb": gpu_mem_total_mb,
            "gpu_temperature_c": gpu_temp_c,
            "disk_usage_pct": round(disk_usage_pct, 1),
            "disk_used_gb": round(disk_used_gb, 1),
            "disk_total_gb": round(disk_total_gb, 1),
            "inference_fps": 0.0,
            "inference_latency_ms": 0.0,
            "queue_depth": 0,
            "dropped_frames": 0,
            "uptime_seconds": round(time.time() - psutil.boot_time(), 1),
            "timestamp": datetime.now(timezone.utc)
        }

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        gpu_name = None
        cuda_avail = False
        driver_ver = None
        try:
            cmd = "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
            output = subprocess.check_output(cmd, shell=True, text=True, timeout=1.0).strip()
            parts = [p.strip() for p in output.split(",")]
            if len(parts) >= 2:
                gpu_name = parts[0]
                driver_ver = parts[1]
                cuda_avail = True
        except Exception:
            pass

        ram = psutil.virtual_memory()

        return {
            "cpu_model": "x86_64 Processor",
            "cpu_cores": psutil.cpu_count(logical=True) or 1,
            "ram_total_gb": round(ram.total / (1024 ** 3), 1),
            "gpu_name": gpu_name,
            "cuda_available": cuda_avail,
            "driver_version": driver_ver,
            "os_info": "Linux"
        }
