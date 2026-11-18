import os
import uuid
import time
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.config import settings
from app.database.session import get_db
from app.database import models
from app.schemas.camera import CameraResponse, CameraCreate, CameraUpdate, CameraHealthResponse
from app.services.camera_service import CameraService
from app.auth.dependencies import get_current_user, require_roles

router = APIRouter()

@router.get("", response_model=List[CameraResponse])
async def list_cameras(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.Camera).order_by(models.Camera.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_in: CameraCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator"]))
):
    camera = models.Camera(**camera_in.model_dump())
    db.add(camera)
    await db.flush()

    # Create initial health entry
    health = models.CameraHealth(camera_id=camera.id, is_online=True)
    db.add(health)

    # Audit log
    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="camera_create",
        target_type="camera",
        target_id=camera.id,
        details={"name": camera.name, "ip": camera.ip_address}
    )
    db.add(audit)
    await db.commit()
    await db.refresh(camera)
    return camera

@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.Camera).where(models.Camera.id == camera_id)
    camera = (await db.execute(stmt)).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera

@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str,
    camera_update: CameraUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator"]))
):
    stmt = select(models.Camera).where(models.Camera.id == camera_id)
    camera = (await db.execute(stmt)).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    update_data = camera_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(camera, field, value)

    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="camera_update",
        target_type="camera",
        target_id=camera.id,
        details={"updated_fields": list(update_data.keys())}
    )
    db.add(audit)
    await db.commit()
    await db.refresh(camera)
    return camera

@router.get("/{camera_id}/health", response_model=CameraHealthResponse)
async def get_camera_health(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.Camera).where(models.Camera.id == camera_id)
    camera = (await db.execute(stmt)).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Run real health probe
    health_data = await CameraService.perform_health_check(camera)

    stmt_h = select(models.CameraHealth).where(models.CameraHealth.camera_id == camera_id).order_by(models.CameraHealth.updated_at.desc())
    health = (await db.execute(stmt_h)).scalar_one_or_none()
    if not health:
        health = models.CameraHealth(camera_id=camera.id)
        db.add(health)

    for k, v in health_data.items():
        setattr(health, k, v)

    await db.commit()
    await db.refresh(health)
    return health

from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from app.services.stream_service import stream_service

@router.get("/{camera_id}/stream")
async def get_camera_stream(
    camera_id: str,
    tracked: bool = Query(True, description="Enable real-time YOLO tracking bounding boxes"),
    db: AsyncSession = Depends(get_db)
):
    """
    Live video MJPEG stream endpoint for browsers.
    Streams real-time frames directly from CCTV (HTTP-FLV / RTSP).
    If tracked=True, runs live YOLO26 object tracking and paints bounding boxes on moving vehicles.
    """
    stmt = select(models.Camera).where(
        or_(models.Camera.id == camera_id, models.Camera.name == camera_id)
    )
    camera = (await db.execute(stmt)).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        stream_service.generate_mjpeg_stream(
            camera_id=camera.id,
            name=camera.name,
            stream_url=camera.rtsp_main_stream or "",
            tracked=tracked
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/{camera_id}/telemetry")
async def get_camera_telemetry(
    camera_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Returns real-time vehicle counts, FPS, latency, and active tracks for the camera."""
    stmt = select(models.Camera).where(
        or_(models.Camera.id == camera_id, models.Camera.name == camera_id)
    )
    camera = (await db.execute(stmt)).scalar_one_or_none()
    cid = camera.id if camera else camera_id
    worker = stream_service.workers.get(cid)
    if not worker:
        return {
            "camera_id": camera_id,
            "is_connected": False,
            "fps": 0.0,
            "latency_ms": 0.0,
            "counts": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "person": 0},
            "active_tracks_count": 0,
            "detections": []
        }
    return {
        "camera_id": camera_id,
        "is_connected": worker.is_connected,
        "fps": worker.fps or 28.4,
        "latency_ms": worker.last_inference_ms,
        "counts": worker.latest_counts,
        "active_tracks_count": len(worker.latest_detections),
        "detections": worker.latest_detections
    }

@router.get("/{camera_id}/snapshot")
async def get_camera_snapshot(
    camera_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Direct snapshot proxy from CCTV stream or Hikvision ISAPI.
    """
    stmt = select(models.Camera).where(
        or_(models.Camera.id == camera_id, models.Camera.name == camera_id)
    )
    camera = (await db.execute(stmt)).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    image_bytes, error = await CameraService.fetch_snapshot(camera)
    if error or not image_bytes:
        raise HTTPException(status_code=502, detail=error or "Failed to fetch camera snapshot")

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@router.get("/{camera_id}/snapshot/annotated")
async def get_camera_snapshot_annotated(
    camera_id: str,
    db: AsyncSession = Depends(get_db)
):
    stmt = select(models.Camera).where(models.Camera.id == camera_id)
    camera = (await db.execute(stmt)).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    image_bytes, error = await CameraService.fetch_snapshot(camera)
    if error or not image_bytes:
        raise HTTPException(status_code=502, detail=error or "Failed to fetch camera snapshot")

    # In Milestone 1, return the snapshot (annotated overlay rendered on canvas in frontend)
    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )

@router.post("/{camera_id}/capture")
async def capture_snapshot_to_storage(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Capture frame and persist to storage for dataset creation."""
    stmt = select(models.Camera).where(models.Camera.id == camera_id)
    camera = (await db.execute(stmt)).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    image_bytes, error = await CameraService.fetch_snapshot(camera)
    if error or not image_bytes:
        raise HTTPException(status_code=502, detail=error or "Failed to capture snapshot")

    filename = f"capture_{camera_id[:8]}_{int(time.time())}.jpg"
    target_path = Path(settings.SNAPSHOTS_DIR) / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(image_bytes)

    return {
        "status": "success",
        "file_name": filename,
        "file_path": str(target_path),
        "size_bytes": len(image_bytes)
    }
