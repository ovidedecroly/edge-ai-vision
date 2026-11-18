import os
from pathlib import Path
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.environ.get("ENV_FILE", str(BASE_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "Modular Edge-AI Vision Monitoring Platform"
    APP_ENV: str = "production"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "default_secret_key_please_change_in_production_environment"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            import json
            if isinstance(v, str):
                return json.loads(v)
            return v
        return []

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/storage/edge_vision.db"

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def resolve_database_url(cls, v: str) -> str:
        if "sqlite" in v and not v.startswith("sqlite+aiosqlite:////") and not v.startswith("sqlite:////"):
            # Handle relative sqlite path e.g. sqlite+aiosqlite:///../storage/edge_vision.db
            prefix = "sqlite+aiosqlite:///" if "sqlite+aiosqlite" in v else "sqlite:///"
            rel_part = v.replace(prefix, "")
            resolved = (BASE_DIR / rel_part).resolve()
            return f"{prefix}{resolved}"
        return v

    # Storage paths
    STORAGE_DIR: str = str(BASE_DIR / "storage")
    MODELS_DIR: str = str(BASE_DIR / "storage" / "models")
    SNAPSHOTS_DIR: str = str(BASE_DIR / "storage" / "snapshots")
    DATASETS_DIR: str = str(BASE_DIR / "storage" / "datasets")
    LOGS_DIR: str = str(BASE_DIR / "storage" / "logs")

    @field_validator("STORAGE_DIR", "MODELS_DIR", "SNAPSHOTS_DIR", "DATASETS_DIR", "LOGS_DIR", mode="after")
    @classmethod
    def resolve_storage_paths(cls, v: str) -> str:
        path = Path(v)
        if not path.is_absolute():
            return str((BASE_DIR / path).resolve())
        return str(path.resolve())

    # Camera Defaults
    DEFAULT_CAMERA_NAME: str = "Hikvision Entrance Gate DS-2CD1043G2-LIU"
    DEFAULT_CAMERA_MODEL: str = "DS-2CD1043G2-LIU"
    DEFAULT_CAMERA_IP: str = "192.168.1.64"
    DEFAULT_CAMERA_RTSP_PORT: int = 554
    DEFAULT_CAMERA_HTTP_PORT: int = 80
    DEFAULT_CAMERA_CHANNEL: str = "1"
    DEFAULT_CAMERA_USERNAME: str = "admin"
    DEFAULT_CAMERA_PASSWORD: str = "pitUNS@1103"
    DEFAULT_CAMERA_RTSP_STREAM: str = "rtsp://admin:pitUNS%401103@192.168.1.64:554/Streaming/Channels/101"
    DEFAULT_CAMERA_ISAPI_SNAPSHOT: str = "http://192.168.1.64/ISAPI/Streaming/channels/1/picture"

    # Go2RTC
    GO2RTC_HOST: str = "http://localhost:1984"
    GO2RTC_STREAM_NAME: str = "kamera_hikvision"
    GO2RTC_WEBRTC_PORT: int = 8555

    # Active Pipeline
    ACTIVE_PIPELINE: str = "general-objects"
    DEFAULT_DETECTOR_MODEL: str = "coco-yolo11n-v1"
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45

    # Admin seed
    ADMIN_USERNAME: str = "admin"
    ADMIN_EMAIL: str = "admin@edgeai.io"
    ADMIN_PASSWORD: str = "Admin@123456"
    ADMIN_FULL_NAME: str = "System Administrator"

settings = Settings()
