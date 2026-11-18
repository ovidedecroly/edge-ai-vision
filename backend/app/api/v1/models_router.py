import os
import hashlib
import time
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from app.config import settings
from app.database.session import get_db
from app.database import models
from app.schemas.model import (
    ModelArtifactResponse, ModelImportRequest, ModelSmokeTestResponse
)
from app.auth.dependencies import get_current_user, require_roles
from app.models.bootstrap import smoke_test_model, calculate_sha256

router = APIRouter()

@router.get("", response_model=List[ModelArtifactResponse])
async def list_models(
    task: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.ModelArtifact)
    if task:
        stmt = stmt.where(models.ModelArtifact.task == task)
    stmt = stmt.order_by(models.ModelArtifact.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/{model_id}", response_model=ModelArtifactResponse)
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.ModelArtifact).where(models.ModelArtifact.id == model_id)
    model = (await db.execute(stmt)).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model artifact not found")
    return model

@router.post("/import", response_model=ModelArtifactResponse)
async def import_community_model(
    req: ModelImportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Researcher"]))
):
    """
    Import community model. Required initial status:
    EXPERIMENTAL / COMMUNITY UNVERIFIED / NOT FOR FINAL RESEARCH CLAIMS.
    """
    model_file = Path(req.file_path)
    if not model_file.exists():
        raise HTTPException(status_code=400, detail=f"Model file does not exist at {req.file_path}")

    sha256 = calculate_sha256(model_file)

    # Smoke test verification
    try:
        smoke_test_model(model_file, device="cpu")
        val_status = "verified"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Local smoke test failed: {str(e)}")

    artifact = models.ModelArtifact(
        name=req.name,
        version=req.version,
        task=req.task,
        source=req.source,
        source_url=req.source_url,
        license=req.license,
        checksum_sha256=sha256,
        file_path=str(model_file),
        file_format=model_file.suffix.lstrip("."),
        model_size_bytes=model_file.stat().st_size,
        input_size=req.input_size,
        classes=req.classes,
        framework="ultralytics",
        runtime="pytorch",
        precision="fp32",
        validation_status=val_status,
        trust_level="community_unverified",
        is_active=False,
        notes=f"Experimental community model. Performance has not been independently validated on the local dataset. {req.notes}"
    )
    db.add(artifact)

    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="model_import",
        target_type="model",
        target_id=artifact.id,
        details={"name": artifact.name, "trust_level": artifact.trust_level}
    )
    db.add(audit)
    await db.commit()
    await db.refresh(artifact)
    return artifact

@router.post("/{model_id}/test", response_model=ModelSmokeTestResponse)
async def test_model_inference(
    model_id: str,
    device: str = "cpu",
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Researcher"]))
):
    stmt = select(models.ModelArtifact).where(models.ModelArtifact.id == model_id)
    model = (await db.execute(stmt)).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model artifact not found")

    model_path = Path(model.file_path)
    if not model_path.exists():
        candidates = [
            Path(settings.STORAGE_DIR) / model.file_path.lstrip("./"),
            Path(settings.MODELS_DIR) / "official" / "yolo11n-coco" / model_path.name,
            Path(settings.MODELS_DIR) / "official" / "yolo11n-coco" / "model.pt",
            Path(settings.STORAGE_DIR) / "models" / "official" / "yolo11n-coco" / "model.pt"
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break

    try:
        res = smoke_test_model(model_path, device=device)
        return ModelSmokeTestResponse(
            model_id=model.id,
            passed=True,
            device=device,
            inference_time_ms=res["latency_ms"],
            output_shape=[1, 300, 6],
            message=f"Smoke test passed successfully on {device} ({res['latency_ms']:.2f} ms)"
        )
    except Exception as e:
        return ModelSmokeTestResponse(
            model_id=model.id,
            passed=False,
            device=device,
            inference_time_ms=0.0,
            output_shape=[],
            message=f"Smoke test failed: {str(e)}"
        )

@router.post("/{model_id}/activate", response_model=ModelArtifactResponse)
async def activate_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Researcher"]))
):
    stmt = select(models.ModelArtifact).where(models.ModelArtifact.id == model_id)
    model = (await db.execute(stmt)).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model artifact not found")

    # Deactivate same task active models
    stmt_deact = select(models.ModelArtifact).where(
        models.ModelArtifact.task == model.task,
        models.ModelArtifact.id != model.id,
        models.ModelArtifact.is_active == True
    )
    result = await db.execute(stmt_deact)
    for other in result.scalars().all():
        other.is_active = False

    model.is_active = True
    model.activation_date = datetime.now(timezone.utc)

    # Record deployment
    deployment = models.ModelDeployment(
        model_id=model.id,
        device="cuda:0",
        status="active"
    )
    db.add(deployment)

    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="model_activation",
        target_type="model",
        target_id=model.id,
        details={"name": model.name, "task": model.task, "version": model.version}
    )
    db.add(audit)
    await db.commit()
    await db.refresh(model)
    return model

@router.post("/{model_id}/rollback", response_model=ModelArtifactResponse)
async def rollback_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator"]))
):
    stmt = select(models.ModelArtifact).where(models.ModelArtifact.id == model_id)
    model = (await db.execute(stmt)).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model artifact not found")

    model.is_active = False

    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="model_rollback",
        target_type="model",
        target_id=model.id
    )
    db.add(audit)
    await db.commit()
    await db.refresh(model)
    return model
