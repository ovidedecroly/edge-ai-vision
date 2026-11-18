from typing import Optional, List
from pydantic import BaseModel, ConfigDict

class PipelinePresetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    preset_id: str
    name: str
    description: str
    detector_model_id: Optional[str] = None
    plate_model_id: Optional[str] = None
    ocr_model_id: Optional[str] = None
    enabled_classes: List[int] = []
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    use_tracking: bool = True
    use_rectification: bool = False
    use_temporal_voting: bool = False
    is_active: bool = False

class PipelinePresetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled_classes: Optional[List[int]] = None
    confidence_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    use_tracking: Optional[bool] = None
    use_rectification: Optional[bool] = None
    use_temporal_voting: Optional[bool] = None
