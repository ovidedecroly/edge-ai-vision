from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class ModelArtifactBase(BaseModel):
    name: str
    version: str = "v1.0.0"
    task: str # object_detection, license_plate_detection, ocr
    source: str # ultralytics_official, community, local_trained
    source_url: str = ""
    license: str = "AGPL-3.0"
    file_path: str
    file_format: str = "pt"
    model_size_bytes: int = 0
    input_size: int = 640
    classes: List[str] = []
    framework: str = "ultralytics"
    runtime: str = "pytorch"
    precision: str = "fp32"
    validation_status: str = "unverified"
    validation_dataset: Optional[str] = None
    validation_metrics: Dict[str, Any] = {}
    trust_level: str = "official" # official, community_unverified, community_validated, local_trained, local_validated
    notes: str = ""

class ModelImportRequest(BaseModel):
    name: str
    version: str = "v1.0.0"
    task: str = "license_plate_detection"
    source: str = "community"
    source_url: str
    license: str = "MIT"
    file_path: str
    input_size: int = 640
    classes: List[str] = ["license_plate"]
    notes: str = ""

class ModelArtifactResponse(ModelArtifactBase):
    model_config = ConfigDict(from_attributes=True)
    id: str
    checksum_sha256: str
    is_active: bool
    installation_date: datetime
    activation_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class ModelSmokeTestResponse(BaseModel):
    model_id: str
    passed: bool
    device: str
    inference_time_ms: float
    output_shape: List[int]
    message: str
