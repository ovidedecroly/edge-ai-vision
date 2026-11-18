import os
import uuid
import pytest
from httpx import AsyncClient
from app.config import settings

@pytest.mark.asyncio
async def test_training_pipeline_lifecycle(async_client: AsyncClient, admin_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Launch a quick test training job
    train_req = {
        "experiment_name": f"Test_YOLO26_Run_{uuid.uuid4().hex[:6]}",
        "model_name": "yolo11n.pt",
        "epochs": 2,
        "batch_size": 8,
        "imgsz": 320,
        "device": "cpu"
    }
    launch_res = await async_client.post(
        f"{settings.API_V1_STR}/experiments/train",
        json=train_req,
        headers=headers
    )
    assert launch_res.status_code == 200
    job_info = launch_res.json()
    assert "job_id" in job_info
    job_id = job_info["job_id"]

    # 2. List training jobs
    jobs_res = await async_client.get(
        f"{settings.API_V1_STR}/experiments/jobs",
        headers=headers
    )
    assert jobs_res.status_code == 200
    jobs = jobs_res.json()
    assert isinstance(jobs, list)
    assert any(j["id"] == job_id for j in jobs)

    # 3. Test Plate & OCR pipeline endpoints
    plates_res = await async_client.get(
        f"{settings.API_V1_STR}/experiments/plates",
        headers=headers
    )
    assert plates_res.status_code == 200
    assert "items" in plates_res.json()

    export_ocr_res = await async_client.post(
        f"{settings.API_V1_STR}/experiments/plates/export-ocr",
        headers=headers
    )
    assert export_ocr_res.status_code == 200
    assert export_ocr_res.json()["success"] is True

@pytest.mark.asyncio
async def test_dishub_cameras_seeded(async_client: AsyncClient, admin_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}
    res = await async_client.get(f"{settings.API_V1_STR}/cameras", headers=headers)
    assert res.status_code == 200
    cameras = res.json()
    dishub_cams = [c for c in cameras if c.get("model") == "ATCS Dishub Surakarta"]
    assert len(dishub_cams) >= 90
    assert any("BALAIKOTA" in c["name"].upper() for c in dishub_cams)
