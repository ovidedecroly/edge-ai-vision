import os
import sys
import json
import hashlib
import asyncio
import logging
from pathlib import Path
import urllib.request
import time

# Ensure app package is importable
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.database.session import AsyncSessionLocal, init_db
from app.database import models
from sqlalchemy import select

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model-bootstrap")

OFFICIAL_YOLO11N_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def smoke_test_model(model_path: Path, device: str = "cpu") -> dict:
    """Run smoke test and verify CPU fallback."""
    from ultralytics import YOLO
    import numpy as np

    logger.info("Running smoke test on device: %s...", device)
    start_time = time.time()
    
    # Load model
    model = YOLO(str(model_path))
    
    # Create synthetic test frame (640x640x3)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Inference
    results = model.predict(dummy_frame, device=device, verbose=False)
    elapsed_ms = (time.time() - start_time) * 1000

    if not results or len(results) == 0:
        raise RuntimeError("Inference yielded no results object")

    logger.info("Smoke test on %s passed in %.2f ms", device, elapsed_ms)
    return {
        "device": device,
        "latency_ms": elapsed_ms,
        "classes_count": len(model.names)
    }

async def register_model_in_db(
    name: str,
    version: str,
    task: str,
    source: str,
    source_url: str,
    file_path: str,
    checksum: str,
    size_bytes: int,
    classes: list,
    license_name: str,
    trust_level: str
):
    await init_db()
    async with AsyncSessionLocal() as session:
        stmt = select(models.ModelArtifact).where(models.ModelArtifact.checksum_sha256 == checksum)
        existing = (await session.execute(stmt)).scalar_one_or_none()
        
        if existing:
            logger.info("Model '%s' with checksum %s already registered in database.", name, checksum[:8])
            existing.is_active = True
            await session.commit()
            return existing

        artifact = models.ModelArtifact(
            name=name,
            version=version,
            task=task,
            source=source,
            source_url=source_url,
            license=license_name,
            checksum_sha256=checksum,
            file_path=file_path,
            file_format="pt",
            model_size_bytes=size_bytes,
            input_size=640,
            classes=classes,
            framework="ultralytics",
            runtime="pytorch",
            precision="fp32",
            validation_status="verified",
            trust_level=trust_level,
            is_active=True,
            notes="Official Ultralytics YOLO11n pretrained on COCO dataset."
        )
        session.add(artifact)
        await session.commit()
        logger.info("Registered model artifact '%s' (ID: %s) into database.", name, artifact.id)
        return artifact

def run_bootstrap():
    logger.info("==================================================================")
    logger.info("  Starting Official YOLO11 Model Bootstrap")
    logger.info("==================================================================")

    target_dir = Path(settings.MODELS_DIR) / "official" / "yolo11n-coco"
    target_dir.mkdir(parents=True, exist_ok=True)
    model_file = target_dir / "model.pt"
    metadata_file = target_dir / "metadata.json"
    checksum_file = target_dir / "checksum.sha256"

    # Step 1 & 2: Check & Download
    if not model_file.exists() or model_file.stat().st_size < 1000000:
        logger.info("Downloading official yolo11n.pt from %s...", OFFICIAL_YOLO11N_URL)
        urllib.request.urlretrieve(OFFICIAL_YOLO11N_URL, model_file)
        logger.info("Download completed (%d bytes)", model_file.stat().st_size)
    else:
        logger.info("Existing model file found at %s (%d bytes)", model_file, model_file.stat().st_size)

    # Step 4: Compute SHA-256
    sha256 = calculate_sha256(model_file)
    checksum_file.write_text(f"{sha256}  model.pt\n")
    logger.info("SHA-256 checksum: %s", sha256)

    # Step 5 & 6: Source & License Metadata
    metadata = {
        "id": "coco-yolo11n-v1",
        "name": "YOLO11n COCO",
        "task": "object_detection",
        "source": "ultralytics_official",
        "source_url": OFFICIAL_YOLO11N_URL,
        "license": "AGPL-3.0",
        "file": "model.pt",
        "checksum_sha256": sha256,
        "file_size_bytes": model_file.stat().st_size,
        "dataset": "COCO",
        "input_size": 640,
        "default_confidence": 0.35,
        "default_iou": 0.45,
        "enabled_classes": [0, 1, 2, 3, 5, 7], # person, bicycle, car, motorcycle, bus, truck
        "class_names": COCO_CLASSES,
        "trust_level": "official",
        "active": True,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    metadata_file.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model metadata to %s", metadata_file)

    # Step 7 & 8: Smoke test & CPU fallback
    try:
        # Test CPU
        smoke_test_model(model_file, device="cpu")
        # Test CUDA if available
        import torch
        if torch.cuda.is_available():
            smoke_test_model(model_file, device="cuda:0")
    except Exception as e:
        logger.error("Load test or smoke test FAILED: %s. Rejecting model activation!", str(e))
        sys.exit(1)

    # Step 9: Register in Database
    asyncio.run(register_model_in_db(
        name="YOLO11n COCO",
        version="v1.0.0",
        task="object_detection",
        source="ultralytics_official",
        source_url=OFFICIAL_YOLO11N_URL,
        file_path=str(model_file),
        checksum=sha256,
        size_bytes=model_file.stat().st_size,
        classes=COCO_CLASSES,
        license_name="AGPL-3.0",
        trust_level="official"
    ))

    logger.info("==================================================================")
    logger.info("  YOLO11 Model Bootstrap Completed Successfully!")
    logger.info("==================================================================")

if __name__ == "__main__":
    run_bootstrap()
