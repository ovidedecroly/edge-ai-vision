from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Response
from pydantic import BaseModel, Field

from app.services.unsgate_service import unsgate_manager
from app.auth.dependencies import get_current_user
from app.database import models

router = APIRouter()


class ModelSelectRequest(BaseModel):
    model_id: str = Field(..., description="yolo26_ocr or yolo26_lprnet")


class RelayActionRequest(BaseModel):
    direction: str = Field(..., description="masuk or keluar")
    action: str = Field(..., description="open, close, toggle")


class AutoDetectToggleRequest(BaseModel):
    direction: str = Field(..., description="masuk or keluar")
    enabled: bool = Field(..., description="True or False")


class WhitelistCreateRequest(BaseModel):
    plate_number: str
    owner_name: str
    category: str = "Sivitas UNS"
    unit: str = "Fakultas Teknik"
    rfid_uid: Optional[str] = "-"
    vehicle_type: str = "Mobil"


@router.get("/model")
async def get_gate_models():
    """Returns currently selected model and available AI pipelines."""
    return unsgate_manager.get_models_info()


@router.post("/model")
async def set_gate_model(req: ModelSelectRequest):
    """Switches active AI pipeline between YOLO26+OCR and YOLO26+LPRNet."""
    ok = unsgate_manager.set_active_model(req.model_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid model_id. Choose 'yolo26_ocr' or 'yolo26_lprnet'.")
    return {
        "success": True,
        "message": f"Active AI model switched to {req.model_id}",
        "active_model": unsgate_manager.active_model_id
    }


@router.get("/status")
async def get_gate_status():
    """Returns live status of Gate Masuk (Lane 1) and Gate Keluar (Lane 2)."""
    return unsgate_manager.get_status()


@router.get("/stream/{direction}")
async def get_gate_stream(direction: str):
    """Returns real-time JPEG stream frame from Hikvision Entrance Gate DS-2CD1043G2-LIU."""
    jpeg_bytes = unsgate_manager.get_live_stream_jpeg(direction)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.post("/relay")
async def control_gate_relay(req: RelayActionRequest):
    """Controls barrier gate relay (buka manual, tutup manual, selalu buka)."""
    res = unsgate_manager.control_relay(req.direction, req.action)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Failed to control relay"))
    return res


@router.post("/capture/{direction}")
async def trigger_gate_capture(direction: str):
    """
    Triggers ANPR capture on Gate Masuk or Gate Keluar using currently active model.
    Runs YOLO26 detector + OCR or LPRNet, generates clean bounding box,
    verifies UNS whitelist, and operates barrier gate.
    """
    result = unsgate_manager.execute_capture(direction)
    return result


@router.post("/auto-detect")
async def toggle_auto_detect(req: AutoDetectToggleRequest):
    """Enables or disables automatic capture when vehicle passes."""
    return unsgate_manager.set_auto_detect(req.direction, req.enabled)


@router.post("/check-auto/{direction}")
async def check_vehicle_auto_detect(direction: str):
    """
    Checks if vehicle/plate is present in camera frame and auto-triggers ANPR if detected.
    Used by frontend interval loop or external sensor/camera alert.
    """
    return unsgate_manager.check_and_auto_detect(direction)


@router.get("/plates")
async def get_recent_plates(
    limit: int = Query(50, ge=1, le=100),
    direction: Optional[str] = None,
    search: Optional[str] = None
):
    """Returns recent vehicles recognized at UNSGate."""
    items = unsgate_manager.recent_history
    if direction:
        items = [i for i in items if i["direction"] == direction]
    if search:
        q = search.upper().strip()
        items = [i for i in items if q in i["plate_number"] or q in i.get("owner_name", "").upper()]

    return {
        "total": len(items),
        "items": items[:limit]
    }


@router.get("/summary")
async def get_gate_summary():
    """Dashboard KPIs matching AutoGateUNS."""
    today = sum(g["dailyCount"] for g in unsgate_manager.gate_states.values())
    masuk_count = unsgate_manager.gate_states["masuk"]["dailyCount"]
    keluar_count = unsgate_manager.gate_states["keluar"]["dailyCount"]
    inside = max(0, masuk_count - keluar_count + 44)

    return {
        "total_events": 15420 + today,
        "today_events": today,
        "vehicles_inside": inside,
        "total_nodes": 2,
        "online_nodes": 2,
        "offline_nodes": 0,
        "active_model": unsgate_manager.active_model_id
    }


@router.get("/whitelist")
async def get_whitelist():
    """Lists registered Sivitas Akademika UNS vehicles."""
    return {
        "total": len(unsgate_manager.whitelist),
        "items": unsgate_manager.whitelist
    }


@router.post("/whitelist")
async def add_whitelist(req: WhitelistCreateRequest):
    """Registers a new vehicle for automatic barrier access."""
    new_entry = {
        "plate_number": req.plate_number.upper().strip(),
        "owner_name": req.owner_name,
        "category": req.category,
        "unit": req.unit,
        "rfid_uid": req.rfid_uid or "-",
        "vehicle_type": req.vehicle_type,
        "status": "Aktif (Auto-Open)"
    }
    unsgate_manager.whitelist.append(new_entry)
    return {"success": True, "entry": new_entry}


class DeviceCreateRequest(BaseModel):
    name: str = "Kamera Gerbang Baru"
    type: str = "camera_in"  # camera_in, camera_out, relay
    direction: str = "masuk"  # masuk, keluar, both
    model: str = "Hikvision DS-2CD1043G2-LIU"
    ip_address: str = "192.168.1.64"
    http_port: int = 80
    rtsp_port: int = 554
    channel: int = 1
    username: str = "admin"
    password: str = "pitUNS@1103"
    isapi_url: Optional[str] = None
    rtsp_url: Optional[str] = None
    is_active: bool = True


@router.get("/devices")
async def list_devices():
    """Returns list of configured gate cameras and relay devices."""
    return {"items": unsgate_manager.get_devices()}


@router.post("/devices")
async def create_device(req: DeviceCreateRequest):
    """Registers a new camera or barrier device directly from Web UI."""
    return unsgate_manager.add_device(req.dict())


@router.put("/devices/{device_id}")
async def update_device(device_id: str, data: Dict[str, Any] = Body(...)):
    """Updates device configurations."""
    res = unsgate_manager.update_device(device_id, data)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail="Perangkat tidak ditemukan")
    return res


@router.delete("/devices/{device_id}")
async def remove_device(device_id: str):
    """Deletes device configuration."""
    res = unsgate_manager.delete_device(device_id)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail="Perangkat tidak ditemukan")
    return res


@router.post("/devices/{device_id}/test")
async def test_device(device_id: str):
    """Probes device IP and port connection from backend."""
    return unsgate_manager.test_device_connection(device_id)
