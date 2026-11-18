import pytest
from httpx import AsyncClient
from app.config import settings
from app.auth.security import verify_password, get_password_hash, create_access_token, decode_token

@pytest.mark.asyncio
async def test_password_hashing():
    raw = "MySecurePassword123"
    hashed = get_password_hash(raw)
    assert hashed != raw
    assert verify_password(raw, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

@pytest.mark.asyncio
async def test_jwt_token_creation_and_decoding():
    token = create_access_token(subject="user-123", username="testuser", roles=["Administrator"])
    payload = decode_token(token)
    assert payload["sub"] == "user-123"
    assert payload["username"] == "testuser"
    assert "Administrator" in payload["roles"]
    assert payload["type"] == "access"

@pytest.mark.asyncio
async def test_login_success(async_client: AsyncClient):
    response = await async_client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={"username": settings.ADMIN_USERNAME, "password": settings.ADMIN_PASSWORD}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_invalid_password(async_client: AsyncClient):
    response = await async_client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={"username": settings.ADMIN_USERNAME, "password": "WrongPassword!@#"}
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_login_json(async_client: AsyncClient):
    response = await async_client.post(
        f"{settings.API_V1_STR}/auth/login/json",
        json={"username": settings.ADMIN_USERNAME, "password": settings.ADMIN_PASSWORD}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data

@pytest.mark.asyncio
async def test_get_me(async_client: AsyncClient, admin_token: str):
    response = await async_client.get(
        f"{settings.API_V1_STR}/auth/me",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == settings.ADMIN_USERNAME
    assert data["email"] == settings.ADMIN_EMAIL
