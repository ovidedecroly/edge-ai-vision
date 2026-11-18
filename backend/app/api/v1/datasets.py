import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, ConfigDict
from app.config import settings
from app.database.session import get_db
from app.database import models
from app.auth.dependencies import get_current_user, require_roles

router = APIRouter()

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    task: str = "plate_detection" # plate_detection, ocr, vehicle_detection

class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: str
    task: str
    storage_path: str
    total_items: int
    splits: Dict[str, int]
    is_locked: bool

class DatasetSplitRequest(BaseModel):
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

@router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.Dataset).order_by(models.Dataset.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    req: DatasetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Researcher"]))
):
    storage_dir = Path(settings.DATASETS_DIR) / req.name.lower().replace(" ", "_")
    storage_dir.mkdir(parents=True, exist_ok=True)

    dataset = models.Dataset(
        name=req.name,
        description=req.description or "",
        task=req.task,
        storage_path=str(storage_dir),
        total_items=0,
        splits={"train": 0, "validation": 0, "test": 0}
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.Dataset).where(models.Dataset.id == dataset_id)
    dataset = (await db.execute(stmt)).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@router.post("/{dataset_id}/split")
async def split_dataset(
    dataset_id: str,
    req: DatasetSplitRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Researcher"]))
):
    """
    Split dataset based on group_identifier (vehicle / session)
    to prevent data leakage between train, val, and test sets.
    """
    stmt = select(models.Dataset).where(models.Dataset.id == dataset_id)
    dataset = (await db.execute(stmt)).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    stmt_items = select(models.DatasetItem).where(models.DatasetItem.dataset_id == dataset_id)
    items = (await db.execute(stmt_items)).scalars().all()

    # Group items by group_identifier
    groups: Dict[str, List[models.DatasetItem]] = {}
    for item in items:
        gid = item.group_identifier or item.id
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(item)

    group_keys = list(groups.keys())
    random.seed(42)
    random.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(n * req.train_ratio)
    n_val = int(n * req.val_ratio)

    counts = {"train": 0, "validation": 0, "test": 0}
    for idx, gid in enumerate(group_keys):
        if idx < n_train:
            target_split = "train"
        elif idx < n_train + n_val:
            target_split = "validation"
        else:
            target_split = "test"

        for it in groups[gid]:
            it.split = target_split
            counts[target_split] += 1

    dataset.splits = counts
    await db.commit()
    return {"status": "success", "splits": counts, "total_items": len(items)}

@router.post("/{dataset_id}/export")
async def export_dataset(
    dataset_id: str,
    export_format: str = "yolo", # yolo, ocr_csv, json
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Researcher"]))
):
    stmt = select(models.Dataset).where(models.Dataset.id == dataset_id)
    dataset = (await db.execute(stmt)).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    export_dir = Path(settings.STORAGE_DIR) / "exports" / f"{dataset.name.lower().replace(' ', '_')}_{export_format}"
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = export_dir / "manifest.json"
    manifest_data = {
        "dataset_name": dataset.name,
        "format": export_format,
        "splits": dataset.splits,
        "exported_by": current_user.username
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    return {
        "status": "success",
        "format": export_format,
        "export_path": str(export_dir),
        "manifest": str(manifest_file)
    }
