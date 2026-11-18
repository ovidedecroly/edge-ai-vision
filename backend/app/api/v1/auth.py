from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.database import models
from app.schemas.auth import Token, UserResponse, LoginRequest
from app.auth.security import verify_password, create_access_token, create_refresh_token, decode_token
from app.auth.dependencies import get_current_user

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    stmt = select(models.User).where(models.User.username == form_data.username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user account")

    user.last_login = datetime.now(timezone.utc)
    # Log audit
    audit = models.AuditLog(
        user_id=user.id,
        username=user.username,
        action="login",
        target_type="auth",
        target_id=user.id,
        details={"status": "success"}
    )
    db.add(audit)
    await db.commit()

    roles = [r.name for r in user.roles]
    access_token = create_access_token(subject=user.id, username=user.username, roles=roles)
    refresh_token = create_refresh_token(subject=user.id, username=user.username)

    return Token(
        access_token=access_token,
        token_type="bearer",
        refresh_token=refresh_token,
        expires_in=7200
    )

@router.post("/login/json", response_model=Token)
async def login_json(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    stmt = select(models.User).where(models.User.username == req.username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user account")

    user.last_login = datetime.now(timezone.utc)
    roles = [r.name for r in user.roles]
    access_token = create_access_token(subject=user.id, username=user.username, roles=roles)
    refresh_token = create_refresh_token(subject=user.id, username=user.username)

    return Token(
        access_token=access_token,
        token_type="bearer",
        refresh_token=refresh_token,
        expires_in=7200
    )

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Invalid token type")
        user_id = payload.get("sub")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    stmt = select(models.User).where(models.User.id == user_id)
    user = (await db.execute(stmt)).scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive or not found")

    roles = [r.name for r in user.roles]
    new_access_token = create_access_token(subject=user.id, username=user.username, roles=roles)
    new_refresh_token = create_refresh_token(subject=user.id, username=user.username)

    return Token(
        access_token=new_access_token,
        token_type="bearer",
        refresh_token=new_refresh_token,
        expires_in=7200
    )

@router.post("/logout")
async def logout(current_user: models.User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="logout",
        target_type="auth",
        target_id=current_user.id
    )
    db.add(audit)
    await db.commit()
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: models.User = Depends(get_current_user)):
    return current_user
