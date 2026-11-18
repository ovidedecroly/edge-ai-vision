import pytest
from httpx import AsyncClient
from app.config import settings

@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    response = await async_client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "database" in data["checks"]
    assert data["checks"]["database"]["status"] == "ok"
    assert "storage" in data["checks"]
    assert data["checks"]["storage"]["status"] == "ok"

@pytest.mark.asyncio
async def test_health_liveness_readiness(async_client: AsyncClient):
    liveness_res = await async_client.get(f"{settings.API_V1_STR}/health/liveness")
    assert liveness_res.status_code == 200
    assert liveness_res.json()["status"] == "alive"

    readiness_res = await async_client.get(f"{settings.API_V1_STR}/health/readiness")
    assert readiness_res.status_code == 200
    assert readiness_res.json()["status"] == "ready"

@pytest.mark.asyncio
async def test_system_metrics(async_client: AsyncClient):
    response = await async_client.get(f"{settings.API_V1_STR}/system/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "cpu_usage_pct" in metrics
    assert "ram_usage_pct" in metrics
    assert "disk_usage_pct" in metrics

@pytest.mark.asyncio
async def test_pipelines(async_client: AsyncClient, admin_token: str):
    response = await async_client.get(
        f"{settings.API_V1_STR}/pipelines",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    pipelines = response.json()
    assert len(pipelines) >= 5
    preset_ids = [p["preset_id"] for p in pipelines]
    assert "general-objects" in preset_ids
    assert "person-monitoring" in preset_ids
    assert "vehicle-monitoring" in preset_ids
    assert "anpr-bootstrap" in preset_ids
    assert "anpr-research" in preset_ids
