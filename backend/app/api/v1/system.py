import os
import time
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from app.config import settings
from app.database.session import get_db
from app.database import models
from app.schemas.system import SystemMetricsResponse, HealthStatusResponse, HardwareInfoResponse, HealthCheckItem
from app.services.metrics_service import MetricsService
from app.services.camera_service import CameraService
from app.auth.dependencies import get_current_user

router = APIRouter()

@router.get("/health", response_model=HealthStatusResponse)
async def get_health(db: AsyncSession = Depends(get_db)):
    """Comprehensive system health endpoint."""
    checks = {}
    is_all_ok = True

    # 1. Database check
    t0 = time.time()
    try:
        await db.execute(text("SELECT 1"))
        db_ms = (time.time() - t0) * 1000
        checks["database"] = HealthCheckItem(name="Database", status="ok", latency_ms=round(db_ms, 2))
    except Exception as e:
        is_all_ok = False
        checks["database"] = HealthCheckItem(name="Database", status="error", message=str(e))

    # 2. Storage check
    try:
        storage_dir = Path(settings.STORAGE_DIR)
        storage_dir.mkdir(parents=True, exist_ok=True)
        checks["storage"] = HealthCheckItem(name="Storage", status="ok", message=str(storage_dir))
    except Exception as e:
        is_all_ok = False
        checks["storage"] = HealthCheckItem(name="Storage", status="error", message=str(e))

    # 3. Model readiness
    stmt = select(models.ModelArtifact).where(models.ModelArtifact.is_active == True)
    active_models = (await db.execute(stmt)).scalars().all()
    if active_models:
        checks["models"] = HealthCheckItem(name="Model Registry", status="ok", message=f"{len(active_models)} active model(s)")
    else:
        checks["models"] = HealthCheckItem(name="Model Registry", status="warning", message="No active model registered yet")

    # 4. Camera check
    stmt_cam = select(models.Camera).where(models.Camera.is_active == True)
    cameras = (await db.execute(stmt_cam)).scalars().all()
    if cameras:
        cam = cameras[0]
        rtsp_ok = CameraService.check_rtsp_port(cam.ip_address, cam.rtsp_port)
        checks["camera"] = HealthCheckItem(
            name=f"Camera ({cam.ip_address})",
            status="ok" if rtsp_ok else "warning",
            message=f"RTSP Port 554: {'Open' if rtsp_ok else 'Unreachable'}"
        )
    else:
        checks["camera"] = HealthCheckItem(name="Camera", status="warning", message="No camera configured")

    overall_status = "ok" if is_all_ok else "degraded"
    return HealthStatusResponse(
        status=overall_status,
        version="1.0.0",
        checks=checks,
        timestamp=datetime.now(timezone.utc)
    )

@router.get("/health/liveness")
async def liveness():
    return {"status": "alive", "timestamp": time.time()}

@router.get("/health/readiness")
async def readiness(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        return {"status": "unready", "database": str(e)}

@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    metrics = MetricsService.get_system_metrics()
    return metrics

@router.get("/system/hardware", response_model=HardwareInfoResponse)
async def get_hardware_info():
    info = MetricsService.get_hardware_info()
    return info
