import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.database import models
from app.schemas.event import RecognitionEventResponse, EventReviewRequest
from app.auth.dependencies import get_current_user, require_roles

router = APIRouter()

@router.get("", response_model=List[RecognitionEventResponse])
async def list_events(
    camera_id: Optional[str] = None,
    plate_text: Optional[str] = None,
    vehicle_class: Optional[str] = None,
    authorization_status: Optional[str] = None,
    review_status: Optional[str] = None,
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.RecognitionEvent)
    if camera_id:
        stmt = stmt.where(models.RecognitionEvent.camera_id == camera_id)
    if plate_text:
        stmt = stmt.where(models.RecognitionEvent.plate_text.ilike(f"%{plate_text}%"))
    if vehicle_class:
        stmt = stmt.where(models.RecognitionEvent.vehicle_class == vehicle_class)
    if authorization_status:
        stmt = stmt.where(models.RecognitionEvent.authorization_status == authorization_status)
    if review_status:
        stmt = stmt.where(models.RecognitionEvent.review_status == review_status)

    stmt = stmt.order_by(models.RecognitionEvent.timestamp.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/{event_id}", response_model=RecognitionEventResponse)
async def get_event(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.RecognitionEvent).where(models.RecognitionEvent.id == event_id)
    event = (await db.execute(stmt)).scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@router.post("/{event_id}/review", response_model=RecognitionEventResponse)
async def review_event(
    event_id: str,
    review: EventReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Reviewer", "Operator"]))
):
    stmt = select(models.RecognitionEvent).where(models.RecognitionEvent.id == event_id)
    event = (await db.execute(stmt)).scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    event.review_status = review.review_status
    event.reviewed_by = current_user.username
    if review.corrected_plate_text:
        event.corrected_plate_text = review.corrected_plate_text
    if review.authorization_status:
        event.authorization_status = review.authorization_status

    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="event_review",
        target_type="event",
        target_id=event.id,
        details={"status": review.review_status, "corrected_text": review.corrected_plate_text}
    )
    db.add(audit)
    await db.commit()
    await db.refresh(event)
    return event

@router.post("/{event_id}/add-to-dataset")
async def add_event_to_dataset(
    event_id: str,
    dataset_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Researcher"]))
):
    stmt = select(models.RecognitionEvent).where(models.RecognitionEvent.id == event_id)
    event = (await db.execute(stmt)).scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Get or create default dataset
    if not dataset_id:
        stmt_d = select(models.Dataset).where(models.Dataset.name == "Default Research Dataset")
        dataset = (await db.execute(stmt_d)).scalar_one_or_none()
        if not dataset:
            dataset = models.Dataset(
                name="Default Research Dataset",
                description="Dataset collected from live recognition events",
                storage_path="storage/datasets/default"
            )
            db.add(dataset)
            await db.flush()
    else:
        stmt_d = select(models.Dataset).where(models.Dataset.id == dataset_id)
        dataset = (await db.execute(stmt_d)).scalar_one_or_none()
        if not dataset:
            raise HTTPException(status_code=404, detail="Target dataset not found")

    item = models.DatasetItem(
        dataset_id=dataset.id,
        file_path=event.raw_frame_path or event.annotated_frame_path or "storage/snapshots/placeholder.jpg",
        thumbnail_path=event.plate_crop_path,
        plate_text=event.corrected_plate_text or event.plate_text,
        vehicle_class=event.vehicle_class,
        group_identifier=event.track_id or event.id,
        status="annotated"
    )
    db.add(item)
    dataset.total_items += 1
    await db.commit()
    return {"message": "Event successfully added to dataset", "item_id": item.id}
