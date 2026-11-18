import pytest
from httpx import AsyncClient
from app.config import settings

@pytest.mark.asyncio
async def test_list_cameras(async_client: AsyncClient, admin_token: str):
    response = await async_client.get(
        f"{settings.API_V1_STR}/cameras",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    default_cam = next((c for c in data if c["ip_address"] == settings.DEFAULT_CAMERA_IP), None)
    assert default_cam is not None
    assert default_cam["ip_address"] == settings.DEFAULT_CAMERA_IP
    assert default_cam["model"] == settings.DEFAULT_CAMERA_MODEL

@pytest.mark.asyncio
async def test_get_camera_by_id(async_client: AsyncClient, admin_token: str):
    # First get cameras list
    list_res = await async_client.get(
        f"{settings.API_V1_STR}/cameras",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    cam_id = list_res.json()[0]["id"]

    response = await async_client.get(
        f"{settings.API_V1_STR}/cameras/{cam_id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == cam_id
    assert "password" not in data or data.get("password") == ""  # Sensitive credentials not exposed in response schema

@pytest.mark.asyncio
async def test_camera_health_check(async_client: AsyncClient, admin_token: str):
    list_res = await async_client.get(
        f"{settings.API_V1_STR}/cameras",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    cam_id = list_res.json()[0]["id"]

    response = await async_client.get(
        f"{settings.API_V1_STR}/cameras/{cam_id}/health",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    health = response.json()
    assert health["camera_id"] == cam_id
    assert "is_online" in health
    assert "rtsp_health" in health
    assert "snapshot_health" in health

@pytest.mark.asyncio
async def test_camera_snapshot(async_client: AsyncClient, admin_token: str):
    list_res = await async_client.get(
        f"{settings.API_V1_STR}/cameras",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    cam_id = list_res.json()[0]["id"]

    response = await async_client.get(
        f"{settings.API_V1_STR}/cameras/{cam_id}/snapshot"
    )
    # The physical camera is on 192.168.1.64.
    # It returns 200 JPEG or 502 if unreachable.
    assert response.status_code in [200, 502]
    if response.status_code == 200:
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 1000
