import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    String, Text, Boolean, Integer, Float, ForeignKey, JSON, DateTime, Table, Column
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import BaseModel, Base, utc_now

# Association table for User <-> Role
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", String(36), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)

# Association table for Role <-> Permission
role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", String(36), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("permission_id", String(36), ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True),
)

class User(BaseModel):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), default="")
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    roles: Mapped[List["Role"]] = relationship("Role", secondary=user_roles, back_populates="users", lazy="selectin")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="user")

class Role(BaseModel):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(String(64), unique=True, index=True) # Administrator, Researcher, Operator, Reviewer, Viewer
    description: Mapped[str] = mapped_column(String(255), default="")

    users: Mapped[List["User"]] = relationship("User", secondary=user_roles, back_populates="roles")
    permissions: Mapped[List["Permission"]] = relationship("Permission", secondary=role_permissions, back_populates="roles", lazy="selectin")

class Permission(BaseModel):
    __tablename__ = "permissions"

    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str] = mapped_column(String(255), default="")

    roles: Mapped[List["Role"]] = relationship("Role", secondary=role_permissions, back_populates="permissions")

class Camera(BaseModel):
    __tablename__ = "cameras"

    name: Mapped[str] = mapped_column(String(128), index=True)
    model: Mapped[str] = mapped_column(String(128), default="DS-2CD1043G2-LIU")
    ip_address: Mapped[str] = mapped_column(String(64), default="192.168.1.64")
    rtsp_port: Mapped[int] = mapped_column(Integer, default=554)
    http_port: Mapped[int] = mapped_column(Integer, default=80)
    channel: Mapped[str] = mapped_column(String(16), default="1")
    rtsp_main_stream: Mapped[str] = mapped_column(Text)
    rtsp_substream: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    isapi_snapshot_url: Mapped[str] = mapped_column(Text)
    encrypted_credentials: Mapped[str] = mapped_column(Text, default="")
    username: Mapped[str] = mapped_column(String(64), default="admin")
    password: Mapped[str] = mapped_column(String(128), default="pitUNS@1103")
    location: Mapped[str] = mapped_column(String(128), default="Main Entrance")
    installation_height_m: Mapped[float] = mapped_column(Float, default=2.5)
    orientation: Mapped[str] = mapped_column(String(64), default="North")
    expected_lane: Mapped[str] = mapped_column(String(64), default="Lane 1")
    roi_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)
    reconnect_policy: Mapped[Dict[str, Any]] = mapped_column(JSON, default=lambda: {"max_retries": 10, "backoff_sec": 2})
    capture_policy: Mapped[Dict[str, Any]] = mapped_column(JSON, default=lambda: {"fps": 15, "skip_frames": 0})
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    health_records: Mapped[List["CameraHealth"]] = relationship("CameraHealth", back_populates="camera", cascade="all, delete-orphan")
    events: Mapped[List["RecognitionEvent"]] = relationship("RecognitionEvent", back_populates="camera")

class CameraHealth(BaseModel):
    __tablename__ = "camera_health"

    camera_id: Mapped[str] = mapped_column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    is_online: Mapped[bool] = mapped_column(Boolean, default=True)
    resolution: Mapped[str] = mapped_column(String(32), default="2560x1440")
    codec: Mapped[str] = mapped_column(String(16), default="H264")
    reported_fps: Mapped[float] = mapped_column(Float, default=25.0)
    measured_fps: Mapped[float] = mapped_column(Float, default=0.0)
    latest_frame_age_ms: Mapped[float] = mapped_column(Float, default=0.0)
    dropped_frames: Mapped[int] = mapped_column(Integer, default=0)
    reconnect_count: Mapped[int] = mapped_column(Integer, default=0)
    snapshot_health: Mapped[str] = mapped_column(String(32), default="ok")
    rtsp_health: Mapped[str] = mapped_column(String(32), default="ok")
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="health_records")

class ModelArtifact(BaseModel):
    __tablename__ = "model_artifacts"

    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(32), default="v1.0.0")
    task: Mapped[str] = mapped_column(String(64), index=True) # object_detection, license_plate_detection, ocr
    source: Mapped[str] = mapped_column(String(64)) # ultralytics_official, community, local_trained
    source_url: Mapped[str] = mapped_column(Text, default="")
    license: Mapped[str] = mapped_column(String(64), default="AGPL-3.0")
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True)
    file_path: Mapped[str] = mapped_column(Text)
    file_format: Mapped[str] = mapped_column(String(16), default="pt") # pt, onnx, engine
    model_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    input_size: Mapped[int] = mapped_column(Integer, default=640)
    classes: Mapped[List[str]] = mapped_column(JSON, default=list)
    framework: Mapped[str] = mapped_column(String(32), default="ultralytics")
    runtime: Mapped[str] = mapped_column(String(32), default="pytorch") # pytorch, onnxruntime, tensorrt
    precision: Mapped[str] = mapped_column(String(16), default="fp32") # fp32, fp16, int8
    validation_status: Mapped[str] = mapped_column(String(32), default="unverified") # unverified, verified, validated
    validation_dataset: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    validation_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    trust_level: Mapped[str] = mapped_column(String(32), default="official") # official, community_unverified, community_validated, local_trained, local_validated
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    installation_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    activation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")

    deployments: Mapped[List["ModelDeployment"]] = relationship("ModelDeployment", back_populates="model_artifact")

class ModelDeployment(BaseModel):
    __tablename__ = "model_deployments"

    model_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_artifacts.id", ondelete="CASCADE"))
    device: Mapped[str] = mapped_column(String(32), default="cuda:0") # cpu, cuda:0
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    undeployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active") # active, standby, terminated
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    model_artifact: Mapped["ModelArtifact"] = relationship("ModelArtifact", back_populates="deployments")

class PipelinePreset(BaseModel):
    __tablename__ = "pipeline_presets"

    preset_id: Mapped[str] = mapped_column(String(64), unique=True, index=True) # general-objects, person-monitoring, vehicle-monitoring, anpr-bootstrap, anpr-research
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    detector_model_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    plate_model_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    ocr_model_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    enabled_classes: Mapped[List[int]] = mapped_column(JSON, default=list)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.35)
    iou_threshold: Mapped[float] = mapped_column(Float, default=0.45)
    use_tracking: Mapped[bool] = mapped_column(Boolean, default=True)
    use_rectification: Mapped[bool] = mapped_column(Boolean, default=False)
    use_temporal_voting: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

class RecognitionEvent(BaseModel):
    __tablename__ = "recognition_events"

    camera_id: Mapped[str] = mapped_column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    pipeline_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    vehicle_class: Mapped[str] = mapped_column(String(32), default="unknown") # car, motorcycle, bus, truck, person, bicycle
    vehicle_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    vehicle_bbox: Mapped[List[float]] = mapped_column(JSON, default=list) # [x1, y1, x2, y2]
    vehicle_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    plate_text: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    plate_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    plate_detector_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    plate_bbox: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    plate_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    annotated_frame_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_frame_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    authorization_status: Mapped[str] = mapped_column(String(32), default="unrecognized", index=True) # authorized, rejected, unrecognized, manual_review
    gate_action: Mapped[str] = mapped_column(String(32), default="none") # none, requested_open, blocked
    review_status: Mapped[str] = mapped_column(String(32), default="unreviewed") # unreviewed, approved, rejected, corrected
    corrected_plate_text: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    detection_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    recognition_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    model_versions: Mapped[Dict[str, str]] = mapped_column(JSON, default=dict)
    trust_level: Mapped[str] = mapped_column(String(32), default="official")
    temporal_votes: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, default=list)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="events")
    candidates: Mapped[List["RecognitionCandidate"]] = relationship("RecognitionCandidate", back_populates="event", cascade="all, delete-orphan")

class RecognitionCandidate(BaseModel):
    __tablename__ = "recognition_candidates"

    event_id: Mapped[str] = mapped_column(String(36), ForeignKey("recognition_events.id", ondelete="CASCADE"), index=True)
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    plate_text: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    character_confidences: Mapped[List[float]] = mapped_column(JSON, default=list)
    plate_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)

    event: Mapped["RecognitionEvent"] = relationship("RecognitionEvent", back_populates="candidates")

class ObjectDetection(BaseModel):
    __tablename__ = "object_detections"

    camera_id: Mapped[str] = mapped_column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[List[float]] = mapped_column(JSON) # [x1, y1, x2, y2]
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

class VehicleTrack(BaseModel):
    __tablename__ = "vehicle_tracks"

    track_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    camera_id: Mapped[str] = mapped_column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"))
    vehicle_class: Mapped[str] = mapped_column(String(32))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    frame_count: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="active") # active, completed, lost
    best_plate_text: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

class Dataset(BaseModel):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    task: Mapped[str] = mapped_column(String(64), default="plate_detection") # plate_detection, ocr, vehicle_detection
    storage_path: Mapped[str] = mapped_column(Text)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    splits: Mapped[Dict[str, int]] = mapped_column(JSON, default=lambda: {"train": 0, "validation": 0, "test": 0})
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False)

    items: Mapped[List["DatasetItem"]] = relationship("DatasetItem", back_populates="dataset", cascade="all, delete-orphan")
    versions: Mapped[List["DatasetVersion"]] = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")

class DatasetItem(BaseModel):
    __tablename__ = "dataset_items"

    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    file_path: Mapped[str] = mapped_column(Text)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    annotations: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list) # bboxes, text labels
    plate_text: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    vehicle_class: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    day_night: Mapped[str] = mapped_column(String(16), default="day") # day, night
    weather: Mapped[str] = mapped_column(String(32), default="clear")
    angle_category: Mapped[str] = mapped_column(String(32), default="front")
    distance_category: Mapped[str] = mapped_column(String(32), default="medium")
    is_blur: Mapped[bool] = mapped_column(Boolean, default=False)
    is_glare: Mapped[bool] = mapped_column(Boolean, default=False)
    is_occluded: Mapped[bool] = mapped_column(Boolean, default=False)
    group_identifier: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True) # group by vehicle or session
    split: Mapped[str] = mapped_column(String(32), default="unassigned") # unassigned, train, validation, test
    status: Mapped[str] = mapped_column(String(32), default="unreviewed") # unreviewed, annotated, verified, rejected

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="items")

class DatasetVersion(BaseModel):
    __tablename__ = "dataset_versions"

    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"))
    version_tag: Mapped[str] = mapped_column(String(32)) # v1.0, v1.1
    manifest_path: Mapped[str] = mapped_column(Text)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    train_count: Mapped[int] = mapped_column(Integer, default=0)
    val_count: Mapped[int] = mapped_column(Integer, default=0)
    test_count: Mapped[int] = mapped_column(Integer, default=0)
    checksum_sha256: Mapped[str] = mapped_column(String(64))

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="versions")

class Experiment(BaseModel):
    __tablename__ = "experiments"

    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    objective: Mapped[str] = mapped_column(Text, default="")
    dataset_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    detector_model: Mapped[str] = mapped_column(String(64))
    recognizer_model: Mapped[str] = mapped_column(String(64))
    tracker_type: Mapped[str] = mapped_column(String(32), default="ByteTrack")
    input_resolution: Mapped[int] = mapped_column(Integer, default=640)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.35)
    iou_threshold: Mapped[float] = mapped_column(Float, default=0.45)
    use_rectification: Mapped[bool] = mapped_column(Boolean, default=False)
    use_quality_selection: Mapped[bool] = mapped_column(Boolean, default=False)
    use_temporal_voting: Mapped[bool] = mapped_column(Boolean, default=False)
    precision: Mapped[str] = mapped_column(String(16), default="fp32")
    runtime: Mapped[str] = mapped_column(String(32), default="pytorch")
    hardware: Mapped[str] = mapped_column(String(64), default="NVIDIA RTX 3070 Ti")
    status: Mapped[str] = mapped_column(String(32), default="created") # created, running, completed, failed
    notes: Mapped[str] = mapped_column(Text, default="")

    runs: Mapped[List["ExperimentRun"]] = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")
    training_jobs: Mapped[List["TrainingJob"]] = relationship("TrainingJob", back_populates="experiment", cascade="all, delete-orphan")

class DatasetImage(BaseModel):
    __tablename__ = "dataset_images"

    camera_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    camera_name: Mapped[str] = mapped_column(String(128), default="dishub_camera")
    dataset_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True)
    file_path: Mapped[str] = mapped_column(Text)
    file_name: Mapped[str] = mapped_column(String(128), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    resolution: Mapped[str] = mapped_column(String(32), default="1920x1080")
    brightness: Mapped[float] = mapped_column(Float, default=120.0)
    blur_score: Mapped[float] = mapped_column(Float, default=50.0)
    glare_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    dhash: Mapped[str] = mapped_column(String(64), default="")
    is_accepted: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    rejection_reasons: Mapped[List[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="raw", index=True) # raw, pseudo_annotated, reviewed, approved, rejected
    split: Mapped[str] = mapped_column(String(32), default="unassigned", index=True) # unassigned, train, val, test
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    annotations: Mapped[List["Annotation"]] = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")

class Annotation(BaseModel):
    __tablename__ = "annotations"

    image_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_images.id", ondelete="CASCADE"), index=True)
    class_name: Mapped[str] = mapped_column(String(64), index=True) # person, car, motorcycle, bus, truck, bicycle, license_plate
    class_idx: Mapped[int] = mapped_column(Integer, default=2)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    x_min: Mapped[int] = mapped_column(Integer, default=0)
    y_min: Mapped[int] = mapped_column(Integer, default=0)
    x_max: Mapped[int] = mapped_column(Integer, default=0)
    y_max: Mapped[int] = mapped_column(Integer, default=0)
    x_center: Mapped[float] = mapped_column(Float, default=0.0)
    y_center: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[float] = mapped_column(Float, default=0.0)
    height: Mapped[float] = mapped_column(Float, default=0.0)
    is_crowd: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(32), default="draft") # draft, approved, rejected
    annotator_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    plate_text: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    image: Mapped["DatasetImage"] = relationship("DatasetImage", back_populates="annotations")

class TrainingJob(BaseModel):
    __tablename__ = "training_jobs"

    experiment_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    model_name: Mapped[str] = mapped_column(String(128), default="yolo26n.pt")
    dataset_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    dataset_path: Mapped[str] = mapped_column(Text, default="")
    epochs: Mapped[int] = mapped_column(Integer, default=100)
    batch_size: Mapped[int] = mapped_column(Integer, default=16)
    imgsz: Mapped[int] = mapped_column(Integer, default=640)
    optimizer: Mapped[str] = mapped_column(String(32), default="SGD")
    learning_rate: Mapped[float] = mapped_column(Float, default=0.01)
    augmentation: Mapped[str] = mapped_column(String(64), default="default")
    hardware: Mapped[str] = mapped_column(String(64), default="NVIDIA RTX / CUDA")
    device: Mapped[str] = mapped_column(String(32), default="cuda:0")
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True) # queued, running, completed, failed, cancelled
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=100)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    best_map50: Mapped[float] = mapped_column(Float, default=0.0)
    best_map50_95: Mapped[float] = mapped_column(Float, default=0.0)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)
    f1_score: Mapped[float] = mapped_column(Float, default=0.0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    gpu_usage_pct: Mapped[float] = mapped_column(Float, default=0.0)
    weights_dir: Mapped[str] = mapped_column(Text, default="")
    best_pt_path: Mapped[str] = mapped_column(Text, default="")
    last_pt_path: Mapped[str] = mapped_column(Text, default="")
    results_csv_path: Mapped[str] = mapped_column(Text, default="")
    plots_dir: Mapped[str] = mapped_column(Text, default="")
    log_output: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    experiment: Mapped[Optional["Experiment"]] = relationship("Experiment", back_populates="training_jobs")

class ExperimentRun(BaseModel):
    __tablename__ = "experiment_runs"

    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"))
    run_number: Mapped[int] = mapped_column(Integer, default=1)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="runs")
    metric_records: Mapped[List["ExperimentMetric"]] = relationship("ExperimentMetric", back_populates="run", cascade="all, delete-orphan")

class ExperimentMetric(BaseModel):
    __tablename__ = "experiment_metrics"

    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiment_runs.id", ondelete="CASCADE"), index=True)
    metric_name: Mapped[str] = mapped_column(String(64), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    step: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped["ExperimentRun"] = relationship("ExperimentRun", back_populates="metric_records")

class WhitelistEntry(BaseModel):
    __tablename__ = "whitelist_entries"

    plate_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    owner_name: Mapped[str] = mapped_column(String(128))
    vehicle_type: Mapped[str] = mapped_column(String(32), default="car")
    department: Mapped[str] = mapped_column(String(128), default="")
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str] = mapped_column(Text, default="")

class Device(BaseModel):
    __tablename__ = "devices"

    device_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    device_type: Mapped[str] = mapped_column(String(32), default="ESP32") # ESP32, STM32, BarrierGateRelay
    name: Mapped[str] = mapped_column(String(128))
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_simulation: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(32), default="offline")
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    capabilities: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    gates: Mapped[List["Gate"]] = relationship("Gate", back_populates="device")

class Gate(BaseModel):
    __tablename__ = "gates"

    gate_name: Mapped[str] = mapped_column(String(128))
    lane_name: Mapped[str] = mapped_column(String(64), default="Lane 1")
    device_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("devices.id", ondelete="SET NULL"), nullable=True)
    current_state: Mapped[str] = mapped_column(String(32), default="idle") # offline, idle, recognizing, authorized, rejected, opening, open, vehicle_passing, closing, blocked, fault, emergency_stop
    mode: Mapped[str] = mapped_column(String(16), default="automatic") # automatic, manual
    trigger_sensor: Mapped[bool] = mapped_column(Boolean, default=False)
    safety_beam: Mapped[bool] = mapped_column(Boolean, default=False)
    open_limit_switch: Mapped[bool] = mapped_column(Boolean, default=False)
    close_limit_switch: Mapped[bool] = mapped_column(Boolean, default=True)
    relay_state: Mapped[bool] = mapped_column(Boolean, default=False)
    last_command: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_fault: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    device: Mapped[Optional["Device"]] = relationship("Device", back_populates="gates")
    commands: Mapped[List["GateCommand"]] = relationship("GateCommand", back_populates="gate", cascade="all, delete-orphan")

class GateCommand(BaseModel):
    __tablename__ = "gate_commands"

    gate_id: Mapped[str] = mapped_column(String(36), ForeignKey("gates.id", ondelete="CASCADE"))
    command: Mapped[str] = mapped_column(String(32)) # request_open, close, emergency_stop, reset
    source: Mapped[str] = mapped_column(String(32), default="ai_decision") # ai_decision, manual_operator, automated_safety
    issued_by_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="sent") # sent, acknowledged, executed, rejected
    reason: Mapped[str] = mapped_column(String(255), default="")

    gate: Mapped["Gate"] = relationship("Gate", back_populates="commands")

class SystemMetric(BaseModel):
    __tablename__ = "system_metrics"

    cpu_usage_pct: Mapped[float] = mapped_column(Float, default=0.0)
    ram_usage_pct: Mapped[float] = mapped_column(Float, default=0.0)
    ram_used_mb: Mapped[float] = mapped_column(Float, default=0.0)
    gpu_usage_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_memory_used_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_temperature_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    disk_usage_pct: Mapped[float] = mapped_column(Float, default=0.0)
    inference_fps: Mapped[float] = mapped_column(Float, default=0.0)
    inference_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    queue_depth: Mapped[int] = mapped_column(Integer, default=0)
    dropped_frames: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

class Alert(BaseModel):
    __tablename__ = "alerts"

    severity: Mapped[str] = mapped_column(String(16), default="info") # info, warning, error, critical
    source: Mapped[str] = mapped_column(String(64), default="system") # camera, worker, gate, system, auth
    title: Mapped[str] = mapped_column(String(128))
    message: Mapped[str] = mapped_column(Text)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

class AuditLog(BaseModel):
    __tablename__ = "audit_logs"

    user_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    username: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(64), index=True) # login, logout, model_activation, model_rollback, camera_update, ocr_correction, manual_gate, export, delete
    target_type: Mapped[str] = mapped_column(String(64)) # model, camera, gate, event, dataset, user
    target_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")
