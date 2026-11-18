import asyncio
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select
from app.config import settings
from app.database.base import Base
from app.database import models

logger = logging.getLogger(__name__)

# Engine configuration
connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args = {"check_same_thread": False, "timeout": 30.0}

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    connect_args=connect_args,
    pool_pre_ping=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """Create tables and initialize baseline data (Admin, Roles, Default Camera, Pipelines)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        # 1. Initialize Roles
        roles_data = [
            ("Administrator", "Full access to all system configurations, models, devices, and users"),
            ("Researcher", "Access to datasets, models, experiments, and annotations"),
            ("Annotator", "Access to annotation studio, bounding box editing, and review"),
            ("Operator", "Access to live monitoring, recognition events, and gate control"),
            ("Reviewer", "Access to annotation validation and OCR review"),
            ("Viewer", "Read-only access to dashboard and live views"),
        ]
        created_roles = {}
        for role_name, desc in roles_data:
            stmt = select(models.Role).where(models.Role.name == role_name)
            result = await session.execute(stmt)
            role = result.scalar_one_or_none()
            if not role:
                role = models.Role(name=role_name, description=desc)
                session.add(role)
                await session.flush()
            created_roles[role_name] = role

        # 2. Initialize Super Admin User
        stmt = select(models.User).where(models.User.username == settings.ADMIN_USERNAME)
        admin_user = (await session.execute(stmt)).scalar_one_or_none()
        if not admin_user:
            from app.auth.security import get_password_hash
            admin_user = models.User(
                username=settings.ADMIN_USERNAME,
                email=settings.ADMIN_EMAIL,
                full_name=settings.ADMIN_FULL_NAME,
                hashed_password=get_password_hash(settings.ADMIN_PASSWORD),
                is_active=True,
                is_superuser=True
            )
            admin_user.roles.append(created_roles["Administrator"])
            session.add(admin_user)
            logger.info("Initialized default administrator user: %s", settings.ADMIN_USERNAME)

        # 2b. Initialize Operator (Pos Satpam) User
        stmt_op = select(models.User).where(models.User.username == "operator")
        op_user = (await session.execute(stmt_op)).scalar_one_or_none()
        if not op_user:
            from app.auth.security import get_password_hash
            op_user = models.User(
                username="operator",
                email="operator@uns.ac.id",
                full_name="Petugas Pos Satpam UNS",
                hashed_password=get_password_hash("Operator@123"),
                is_active=True,
                is_superuser=False
            )
            op_user.roles.append(created_roles["Operator"])
            session.add(op_user)
            logger.info("Initialized default operator user: operator")

        # 3. Initialize Default Camera (Hikvision DS-2CD1043G2-LIU)
        stmt = select(models.Camera).where(models.Camera.ip_address == settings.DEFAULT_CAMERA_IP)
        default_cam = (await session.execute(stmt)).scalar_one_or_none()
        if not default_cam:
            default_cam = models.Camera(
                name=settings.DEFAULT_CAMERA_NAME,
                model=settings.DEFAULT_CAMERA_MODEL,
                ip_address=settings.DEFAULT_CAMERA_IP,
                rtsp_port=settings.DEFAULT_CAMERA_RTSP_PORT,
                http_port=settings.DEFAULT_CAMERA_HTTP_PORT,
                channel=settings.DEFAULT_CAMERA_CHANNEL,
                rtsp_main_stream=settings.DEFAULT_CAMERA_RTSP_STREAM,
                isapi_snapshot_url=settings.DEFAULT_CAMERA_ISAPI_SNAPSHOT,
                username=settings.DEFAULT_CAMERA_USERNAME,
                password=settings.DEFAULT_CAMERA_PASSWORD,
                location="Entrance Gate 1",
                is_active=True
            )
            session.add(default_cam)
            await session.flush()

            # Add initial health record
            health = models.CameraHealth(
                camera_id=default_cam.id,
                is_online=True,
                resolution="2560x1440",
                codec="H264",
                reported_fps=25.0,
                measured_fps=25.0,
                snapshot_health="ok",
                rtsp_health="ok"
            )
            session.add(health)
            logger.info("Initialized default camera: %s (%s)", default_cam.name, default_cam.ip_address)

        # 3b. Seed Dishub Surakarta CCTV Cameras (96 ATCS Intersections)
        import json
        from pathlib import Path
        dishub_path = Path(__file__).parent.parent / "data" / "dishub_cameras.json"
        if dishub_path.exists():
            try:
                with open(dishub_path) as f:
                    dishub_cams = json.load(f)
                
                # Check existing dishub cameras count
                stmt_count = select(models.Camera).where(models.Camera.model == "ATCS Dishub Surakarta")
                existing_count = len((await session.execute(stmt_count)).scalars().all())
                
                if existing_count == 0:
                    for d_cam in dishub_cams:
                        cam_ip = f"dishub-{d_cam['id']}"
                        dishub_entry = models.Camera(
                            name=f"{d_cam['name']}",
                            model="ATCS Dishub Surakarta",
                            ip_address=cam_ip,
                            rtsp_port=8086,
                            http_port=8086,
                            channel=str(d_cam['id']),
                            rtsp_main_stream=d_cam['stream_url'],
                            isapi_snapshot_url="",
                            username="dishub",
                            password="public",
                            location=f"{d_cam.get('district', 'Surakarta')}",
                            is_active=True
                        )
                        session.add(dishub_entry)
                    logger.info("Successfully seeded %d Dishub Surakarta ATCS CCTV cameras.", len(dishub_cams))
            except Exception as e:
                logger.warning("Failed to seed Dishub cameras: %s", e)

        # 4. Initialize Pipeline Presets (Section 9)
        presets = [
            {
                "preset_id": "general-objects",
                "name": "General Objects",
                "description": "YOLO11n COCO -> selected COCO classes -> tracking -> object events",
                "enabled_classes": [0, 1, 2, 3, 5, 7], # person, bicycle, car, motorcycle, bus, truck
                "use_tracking": True,
                "is_active": True
            },
            {
                "preset_id": "person-monitoring",
                "name": "Person Monitoring",
                "description": "YOLO11n COCO -> person -> tracking -> person count and event",
                "enabled_classes": [0], # person
                "use_tracking": True,
                "is_active": False
            },
            {
                "preset_id": "vehicle-monitoring",
                "name": "Vehicle Monitoring",
                "description": "YOLO11n COCO -> bicycle, car, motorcycle, bus, truck -> tracking -> vehicle event",
                "enabled_classes": [1, 2, 3, 5, 7],
                "use_tracking": True,
                "is_active": False
            },
            {
                "preset_id": "anpr-bootstrap",
                "name": "ANPR Bootstrap",
                "description": "YOLO11n COCO vehicle detector -> vehicle ROI -> community plate detector -> PP-OCR -> experimental recognition event",
                "enabled_classes": [2, 3, 5, 7],
                "use_tracking": True,
                "use_rectification": False,
                "use_temporal_voting": False,
                "is_active": False
            },
            {
                "preset_id": "anpr-research",
                "name": "ANPR Research",
                "description": "Local vehicle/plate model -> plate rectification -> local recognizer -> frame-quality selection -> temporal voting -> validated recognition event",
                "enabled_classes": [2, 3, 5, 7],
                "use_tracking": True,
                "use_rectification": True,
                "use_temporal_voting": True,
                "is_active": False
            },
        ]
        for p in presets:
            stmt = select(models.PipelinePreset).where(models.PipelinePreset.preset_id == p["preset_id"])
            preset = (await session.execute(stmt)).scalar_one_or_none()
            if not preset:
                preset = models.PipelinePreset(
                    preset_id=p["preset_id"],
                    name=p["name"],
                    description=p["description"],
                    enabled_classes=p["enabled_classes"],
                    use_tracking=p["use_tracking"],
                    use_rectification=p.get("use_rectification", False),
                    use_temporal_voting=p.get("use_temporal_voting", False),
                    is_active=p["is_active"]
                )
                session.add(preset)

        # 5. Initialize Gate Simulation Device (Section 22)
        stmt = select(models.Device).where(models.Device.device_id == "sim-esp32-gate-1")
        gate_device = (await session.execute(stmt)).scalar_one_or_none()
        if not gate_device:
            gate_device = models.Device(
                device_id="sim-esp32-gate-1",
                device_type="ESP32",
                name="Simulated Gate Controller ESP32",
                is_simulation=True,
                status="idle",
                capabilities={"relay": True, "safety_beam": True, "loop_detector": True}
            )
            session.add(gate_device)
            await session.flush()

            gate = models.Gate(
                gate_name="Main Entrance Barrier Gate",
                lane_name="Lane 1",
                device_id=gate_device.id,
                current_state="idle",
                mode="automatic",
                trigger_sensor=False,
                safety_beam=True,
                close_limit_switch=True,
                open_limit_switch=False,
                relay_state=False
            )
            session.add(gate)

        await session.commit()
