from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class VehicleDetectionData(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] # [x1, y1, x2, y2]

class PlateRecognitionData(BaseModel):
    text: Optional[str] = None
    detector_confidence: float = 0.0
    recognizer_confidence: float = 0.0
    bbox: Optional[List[float]] = None

class LatencyBreakdown(BaseModel):
    detection_ms: float = 0.0
    recognition_ms: float = 0.0
    total_ms: float = 0.0

class TypedRecognitionEvent(BaseModel):
    type: str = "recognition.completed"
    timestamp: str
    camera_id: str
    track_id: Optional[str] = None
    pipeline_id: str
    vehicle: VehicleDetectionData
    plate: Optional[PlateRecognitionData] = None
    latency: LatencyBreakdown
    model_versions: Dict[str, str] = {}
    trust_level: str = "official"

class RecognitionEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    camera_id: str
    pipeline_id: str
    track_id: Optional[str] = None
    timestamp: datetime
    vehicle_class: str
    vehicle_confidence: float
    vehicle_bbox: List[float]
    vehicle_crop_path: Optional[str] = None
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    plate_detector_confidence: float = 0.0
    plate_bbox: Optional[List[float]] = None
    plate_crop_path: Optional[str] = None
    annotated_frame_path: Optional[str] = None
    raw_frame_path: Optional[str] = None
    authorization_status: str
    gate_action: str
    review_status: str
    corrected_plate_text: Optional[str] = None
    reviewed_by: Optional[str] = None
    detection_latency_ms: float
    recognition_latency_ms: float
    total_latency_ms: float
    model_versions: Dict[str, str] = {}
    trust_level: str
    temporal_votes: Optional[List[Dict[str, Any]]] = []

class EventReviewRequest(BaseModel):
    review_status: str # approved, rejected, corrected
    corrected_plate_text: Optional[str] = None
    authorization_status: Optional[str] = None # authorized, rejected
    notes: Optional[str] = None
