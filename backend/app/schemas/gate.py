from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class GateCommandRequest(BaseModel):
    command: str # request_open, close, emergency_stop, reset
    reason: Optional[str] = "Operator manual command"

class GateCommandResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    gate_id: str
    command: str
    source: str
    status: str
    reason: str
    created_at: datetime

class DeviceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    device_id: str
    device_type: str
    name: str
    ip_address: Optional[str] = None
    is_simulation: bool
    status: str
    last_heartbeat: Optional[datetime] = None
    capabilities: Dict[str, Any]

class GateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    gate_name: str
    lane_name: str
    device_id: Optional[str] = None
    current_state: str # offline, idle, recognizing, authorized, rejected, opening, open, vehicle_passing, closing, blocked, fault, emergency_stop
    mode: str
    trigger_sensor: bool
    safety_beam: bool
    open_limit_switch: bool
    close_limit_switch: bool
    relay_state: bool
    last_command: Optional[str] = None
    last_fault: Optional[str] = None
    updated_at: datetime
