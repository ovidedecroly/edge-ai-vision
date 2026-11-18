import pytest
from httpx import AsyncClient
from app.config import settings

@pytest.mark.asyncio
async def test_list_models(async_client: AsyncClient, admin_token: str):
    response = await async_client.get(
        f"{settings.API_V1_STR}/models",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    models_list = response.json()
    assert isinstance(models_list, list)
    assert len(models_list) >= 1
    coco_model = models_list[0]
    assert "YOLO11n COCO" in coco_model["name"]
    assert coco_model["task"] == "object_detection"
    assert coco_model["trust_level"] == "official"
    assert coco_model["checksum_sha256"] == "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"

@pytest.mark.asyncio
async def test_model_smoke_test(async_client: AsyncClient, admin_token: str):
    list_res = await async_client.get(
        f"{settings.API_V1_STR}/models",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    model_id = list_res.json()[0]["id"]

    response = await async_client.post(
        f"{settings.API_V1_STR}/models/{model_id}/test?device=cpu",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    test_result = response.json()
    assert test_result["passed"] is True
    assert test_result["device"] == "cpu"
    assert test_result["inference_time_ms"] > 0
