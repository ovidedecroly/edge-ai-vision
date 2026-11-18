from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.database import models
from app.schemas.auth import UserResponse
from app.auth.dependencies import get_current_user, require_roles

router = APIRouter()

from sqlalchemy.orm import selectinload
from fastapi import HTTPException
from app.auth.security import get_password_hash
from pydantic import BaseModel

class CreateUserReq(BaseModel):
    username: str
    email: str
    full_name: str = ""
    password: str
    role: str = "Operator" # Administrator or Operator

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator"]))
):
    stmt = select(models.User).options(selectinload(models.User.roles)).order_by(models.User.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("/users")
async def create_new_user(
    req: CreateUserReq,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator"]))
):
    stmt = select(models.User).where(models.User.username == req.username)
    if (await db.execute(stmt)).scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username sudah digunakan")

    stmt_role = select(models.Role).where(models.Role.name == req.role)
    role_obj = (await db.execute(stmt_role)).scalar_one_or_none()
    if not role_obj:
        role_obj = models.Role(name=req.role, description="Auto-created role")
        db.add(role_obj)
        await db.flush()

    new_user = models.User(
        username=req.username,
        email=req.email,
        full_name=req.full_name,
        hashed_password=get_password_hash(req.password),
        is_active=True,
        is_superuser=(req.role == "Administrator")
    )
    new_user.roles.append(role_obj)
    db.add(new_user)
    await db.commit()
    return {"success": True, "message": f"Pengguna {req.username} berhasil dibuat"}

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator"]))
):
    if user_id == str(current_user.id):
        raise HTTPException(status_code=400, detail="Tidak dapat menghapus akun yang sedang aktif")
    stmt = select(models.User).where(models.User.id == user_id)
    user_obj = (await db.execute(stmt)).scalar_one_or_none()
    if not user_obj:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan")
    await db.delete(user_obj)
    await db.commit()
    return {"success": True, "message": "Pengguna berhasil dihapus"}

@router.get("/audit-logs")
async def list_audit_logs(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Researcher"]))
):
    stmt = select(models.AuditLog).order_by(models.AuditLog.timestamp.desc()).limit(limit)
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return [
        {
            "id": log.id,
            "username": log.username,
            "action": log.action,
            "target_type": log.target_type,
            "target_id": log.target_id,
            "details": log.details,
            "timestamp": log.timestamp
        }
        for log in logs
    ]
