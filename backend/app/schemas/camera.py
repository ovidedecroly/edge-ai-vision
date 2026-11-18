from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class CameraBase(BaseModel):
    name: str = "Entrance Gate DS-2CD1043G2-LIU"
    model: str = "DS-2CD1043G2-LIU"
    ip_address: str = "192.168.1.64"
    rtsp_port: int = 554
    http_port: int = 80
    channel: str = "1"
    rtsp_main_stream: str
    rtsp_substream: Optional[str] = None
    isapi_snapshot_url: str
    location: str = "Main Entrance"
    installation_height_m: float = 2.5
    orientation: str = "North"
    expected_lane: str = "Lane 1"
    roi_config: Optional[Dict[str, Any]] = None
    reconnect_policy: Optional[Dict[str, Any]] = None
    capture_policy: Optional[Dict[str, Any]] = None
    is_active: bool = True

class CameraCreate(CameraBase):
    username: str = "admin"
    password: str

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    ip_address: Optional[str] = None
    rtsp_port: Optional[int] = None
    http_port: Optional[int] = None
    channel: Optional[str] = None
    rtsp_main_stream: Optional[str] = None
    rtsp_substream: Optional[str] = None
    isapi_snapshot_url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    location: Optional[str] = None
    installation_height_m: Optional[float] = None
    orientation: Optional[str] = None
    expected_lane: Optional[str] = None
    roi_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class CameraHealthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    camera_id: str
    is_online: bool
    resolution: str
    codec: str
    reported_fps: float
    measured_fps: float
    latest_frame_age_ms: float
    dropped_frames: int
    reconnect_count: int
    snapshot_health: str
    rtsp_health: str
    last_error: Optional[str] = None
    updated_at: datetime

class CameraResponse(CameraBase):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime
    updated_at: datetime
    username: str
    # Password and encrypted_credentials are intentionally omitted for security!
