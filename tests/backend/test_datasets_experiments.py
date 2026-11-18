import uuid
import pytest
from httpx import AsyncClient
from app.config import settings

@pytest.mark.asyncio
async def test_dataset_lifecycle(async_client: AsyncClient, admin_token: str):
    unique_ds_name = f"Test Plate Dataset {uuid.uuid4().hex[:6]}"
    # 1. Create dataset
    create_res = await async_client.post(
        f"{settings.API_V1_STR}/datasets",
        json={"name": unique_ds_name, "description": "Test dataset description", "task": "plate_detection"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert create_res.status_code == 201
    ds = create_res.json()
    assert ds["name"] == unique_ds_name
    ds_id = ds["id"]

    # 2. List datasets
    list_res = await async_client.get(
        f"{settings.API_V1_STR}/datasets",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert list_res.status_code == 200
    assert any(d["id"] == ds_id for d in list_res.json())

    # 3. Export dataset
    export_res = await async_client.post(
        f"{settings.API_V1_STR}/datasets/{ds_id}/export?export_format=yolo",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert export_res.status_code == 200
    assert export_res.json()["status"] == "success"

@pytest.mark.asyncio
async def test_experiment_lifecycle(async_client: AsyncClient, admin_token: str):
    unique_exp_name = f"Ablation Study {uuid.uuid4().hex[:6]}"
    # 1. Create experiment
    exp_res = await async_client.post(
        f"{settings.API_V1_STR}/experiments",
        json={
            "name": unique_exp_name,
            "objective": "Vehicle ROI + Plate Detection",
            "detector_model": "yolo11n.pt",
            "recognizer_model": "PP-OCR Baseline",
            "precision": "fp32"
        },
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert exp_res.status_code == 201
    exp = exp_res.json()
    assert exp["name"] == unique_exp_name
    exp_id = exp["id"]

    # 2. Get metrics
    metrics_res = await async_client.get(
        f"{settings.API_V1_STR}/experiments/{exp_id}/metrics",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert metrics_res.status_code == 200
    runs = metrics_res.json()
    assert len(runs) >= 1
    assert "mAP@0.5" in runs[0]["metrics"]
