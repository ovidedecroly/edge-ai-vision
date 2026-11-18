import os
import uuid
import numpy as np
import cv2
import pytest
from pathlib import Path
from httpx import AsyncClient

from app.config import settings
from app.services.quality_filter import quality_filter
from app.services.dataset_exporter import dataset_exporter

def test_quality_filter_logic():
    """Validates blur, brightness, glare, and duplicate hash algorithms."""
    # 1. Clear sharp image with gradients and proper daylight brightness (mean ~120)
    sharp = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.rectangle(sharp, (50, 50), (200, 200), (40, 40, 40), -1)
    cv2.putText(sharp, "TEST SHARP", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    res_sharp = quality_filter.evaluate(sharp)
    assert res_sharp["accepted"] is True
    assert res_sharp["blur_score"] > 30.0

    # 2. Blurry image
    blurry = cv2.GaussianBlur(sharp, (45, 45), 0)
    res_blur = quality_filter.evaluate(blurry)
    assert res_blur["accepted"] is False
    assert "too_blurry" in res_blur["rejection_reasons"]

    # 3. Too dark image
    dark = np.ones((480, 640, 3), dtype=np.uint8) * 10
    res_dark = quality_filter.evaluate(dark)
    assert res_dark["accepted"] is False
    assert "too_dark" in res_dark["rejection_reasons"]

    # 4. Duplicate dHash check
    hash1 = res_sharp["dhash"]
    res_dup = quality_filter.evaluate(sharp, previous_dhash=hash1)
    assert res_dup["is_duplicate"] is True
    assert "duplicate_frame" in res_dup["rejection_reasons"]

@pytest.mark.asyncio
async def test_capture_and_annotation_api(async_client: AsyncClient, admin_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Test Capture Jobs listing
    jobs_res = await async_client.get(f"{settings.API_V1_STR}/capture/jobs", headers=headers)
    assert jobs_res.status_code == 200
    assert isinstance(jobs_res.json(), list)

    # 2. Create a test image to simulate captured frame
    test_cam = "cam_test_01"
    test_date = "2026-09-25"
    dest_dir = Path(settings.STORAGE_DIR) / "datasets" / "raw" / test_cam / test_date
    dest_dir.mkdir(parents=True, exist_ok=True)
    test_img_path = dest_dir / "img_test_frame.jpg"

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (80, 80), (300, 250), (120, 180, 200), -1)
    cv2.imwrite(str(test_img_path), dummy_frame)

    # 3. Test Auto-Annotation API
    auto_res = await async_client.post(
        f"{settings.API_V1_STR}/annotations/auto-annotate",
        json={"image_path": str(test_img_path)},
        headers=headers
    )
    assert auto_res.status_code == 200
    ann_data = auto_res.json()
    assert ann_data["success"] is True

    # 4. Test Get Annotation Item
    item_res = await async_client.get(
        f"{settings.API_V1_STR}/annotations/item?image_path={str(test_img_path)}",
        headers=headers
    )
    assert item_res.status_code == 200
    item_data = item_res.json()
    assert item_data["width"] == 640
    assert item_data["height"] == 480

    # 5. Test Save Human Annotations (Review)
    save_res = await async_client.put(
        f"{settings.API_V1_STR}/annotations/save",
        json={
            "image_path": str(test_img_path),
            "status": "approved",
            "annotations": [
                {
                    "class_name": "car",
                    "class_idx": 2,
                    "confidence": 0.95,
                    "bbox": [80, 80, 300, 250],
                    "yolo_bbox": [0.296875, 0.343750, 0.343750, 0.354167]
                }
            ]
        },
        headers=headers
    )
    assert save_res.status_code == 200
    assert save_res.json()["status"] == "success"

    # 6. Test Non-Random YOLO Dataset Export
    export_res = await async_client.post(
        f"{settings.API_V1_STR}/annotations/export-yolo",
        json={"dataset_name": f"dishub_test_export_{uuid.uuid4().hex[:6]}"},
        headers=headers
    )
    assert export_res.status_code == 200
    exp_json = export_res.json()
    assert exp_json["success"] is True
    assert Path(exp_json["data_yaml"]).exists()
