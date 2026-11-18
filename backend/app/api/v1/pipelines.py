from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.database import models
from app.schemas.pipeline import PipelinePresetResponse, PipelinePresetUpdate
from app.auth.dependencies import get_current_user, require_roles

router = APIRouter()

@router.get("", response_model=List[PipelinePresetResponse])
async def list_pipelines(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.PipelinePreset).order_by(models.PipelinePreset.created_at)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("/{preset_id}/activate", response_model=PipelinePresetResponse)
async def activate_pipeline(
    preset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Operator", "Researcher"]))
):
    stmt = select(models.PipelinePreset).where(models.PipelinePreset.preset_id == preset_id)
    preset = (await db.execute(stmt)).scalar_one_or_none()
    if not preset:
        raise HTTPException(status_code=404, detail="Pipeline preset not found")

    # Set others inactive
    stmt_all = select(models.PipelinePreset)
    all_presets = (await db.execute(stmt_all)).scalars().all()
    for p in all_presets:
        p.is_active = (p.preset_id == preset_id)

    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="pipeline_activation",
        target_type="pipeline",
        target_id=preset.preset_id,
        details={"name": preset.name}
    )
    db.add(audit)
    await db.commit()
    await db.refresh(preset)
    return preset
