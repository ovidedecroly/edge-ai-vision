from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user, require_permission
from app.services.capture_service import capture_manager
from app.config import settings

router = APIRouter(prefix="/capture", tags=["Dataset Capture Manager"])

class StartCaptureRequest(BaseModel):
    camera_id: str
    camera_name: str
    stream_url: str
    interval_seconds: int = Field(default=5, ge=1, le=3600)
    dataset_id: Optional[str] = None
    max_storage_mb: int = Field(default=10000, ge=100)
    time_filter_enabled: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    schedule_preset: Optional[str] = None
    schedule_label: Optional[str] = None

class StopCaptureRequest(BaseModel):
    camera_id: str

class SingleCaptureRequest(BaseModel):
    camera_id: str
    camera_name: str
    stream_url: str
    dataset_id: Optional[str] = None

@router.get("/jobs", response_model=List[Dict[str, Any]])
async def list_capture_jobs(
    current_user = Depends(get_current_user)
):
    """Lists all automated dataset capture jobs and live statistics."""
    return capture_manager.list_all_jobs()

@router.post("/start", status_code=status.HTTP_200_OK)
async def start_capture_job(
    req: StartCaptureRequest,
    current_user = Depends(require_permission("datasets:write"))
):
    """Starts background scheduled frame capture from live CCTV stream with optional time window filter."""
    res = capture_manager.start_job(
        camera_id=req.camera_id,
        camera_name=req.camera_name,
        stream_url=req.stream_url,
        interval_seconds=req.interval_seconds,
        dataset_id=req.dataset_id,
        time_filter_enabled=req.time_filter_enabled,
        start_time=req.start_time,
        end_time=req.end_time,
        schedule_preset=req.schedule_preset,
        schedule_label=req.schedule_label,
    )
    return {
        "status": "success",
        "message": f"Capture job started for {req.camera_name} ({req.schedule_label or '24h continuous'})",
        "job": res
    }

@router.post("/stop", status_code=status.HTTP_200_OK)
async def stop_capture_job(
    req: StopCaptureRequest,
    current_user = Depends(require_permission("datasets:write"))
):
    """Stops an active background capture job."""
    res = capture_manager.stop_job(camera_id=req.camera_id)
    return {
        "status": "success",
        "message": f"Capture job stopped for camera {req.camera_id}",
        "job": res
    }

@router.post("/single", status_code=status.HTTP_200_OK)
async def capture_single_frame(
    req: SingleCaptureRequest,
    current_user = Depends(require_permission("datasets:write"))
):
    """Immediately captures 1 frame from CCTV stream, assesses quality, and stores."""
    res = capture_manager.capture_single_frame(
        camera_id=req.camera_id,
        camera_name=req.camera_name,
        stream_url=req.stream_url,
        dataset_id=req.dataset_id,
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Capture failed"))
    return res

@router.get("/raw-images", status_code=status.HTTP_200_OK)
async def list_raw_images(
    camera_id: Optional[str] = Query(None),
    status_filter: str = Query("accepted", pattern="^(accepted|rejected|all)$"),
    limit: int = Query(50, ge=1, le=500),
    current_user = Depends(get_current_user)
):
    """Retrieves list of captured frames with quality metrics and sidecar metadata."""
    raw_dir = Path(settings.STORAGE_DIR) / "datasets" / ("raw" if status_filter == "accepted" else "rejected")
    results = []
    
    if raw_dir.exists():
        # Find all JSON metadata files
        json_files = sorted(raw_dir.glob("*/*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        import json
        for jf in json_files[:limit]:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
                    if camera_id and data.get("camera_id") != camera_id:
                        continue
                    img_file = jf.with_suffix(".jpg")
                    if img_file.exists():
                        data["file_path"] = str(img_file)
                    results.append(data)
            except Exception:
                continue

    return {
        "total": len(results),
        "items": results
    }
