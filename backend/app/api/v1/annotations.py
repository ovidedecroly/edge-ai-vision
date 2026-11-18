import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user, require_permission
from app.services.auto_annotator import auto_annotator, RESEARCH_CLASSES, CLASS_NAME_TO_IDX
from app.services.dataset_exporter import dataset_exporter
from app.config import settings

logger = logging.getLogger("edge_vision.annotations_router")
router = APIRouter(prefix="/annotations", tags=["Annotation Studio & Review"])

class BoundingBox(BaseModel):
    class_name: str
    class_idx: int = Field(default=2)
    confidence: float = Field(default=1.0)
    bbox: List[int] # [x1, y1, x2, y2]
    yolo_bbox: Optional[List[float]] = None # [xc, yc, w, h] normalized

class SaveAnnotationRequest(BaseModel):
    image_path: str
    annotations: List[BoundingBox]
    status: str = Field(default="reviewed", pattern="^(reviewed|approved|rejected)$")
    notes: Optional[str] = None

class AutoAnnotateRequest(BaseModel):
    image_path: Optional[str] = None
    directory_path: Optional[str] = None
    conf_threshold: float = Field(default=0.25, ge=0.05, le=1.0)

class ExportDatasetRequest(BaseModel):
    dataset_name: str = Field(default="dishub_traffic_yolo_v1")
    camera_id: Optional[str] = None
    train_ratio: float = Field(default=0.70, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.5)
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.5)
    classes: Optional[List[str]] = None

class AddClassRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    label_en: Optional[str] = None
    label_id: Optional[str] = None
    color: Optional[str] = "#10B981"

DEFAULT_RESEARCH_CLASSES = [
    {"id": 0, "name": "person", "label_en": "Pedestrian", "label_id": "Pejalan Kaki", "color": "#2563EB"},
    {"id": 1, "name": "bicycle", "label_en": "Bicycle", "label_id": "Sepeda Onthel", "color": "#059669"},
    {"id": 2, "name": "car", "label_en": "Car", "label_id": "Mobil Pribadi", "color": "#16A34A"},
    {"id": 3, "name": "motorcycle", "label_en": "Motorcycle", "label_id": "Sepeda Motor", "color": "#D97706"},
    {"id": 4, "name": "bus", "label_en": "Bus / Minibus", "label_id": "Bus Kota / BST", "color": "#DC2626"},
    {"id": 5, "name": "truck", "label_en": "Truck / Cargo", "label_id": "Truk Barang", "color": "#7C3AED"},
    {"id": 6, "name": "becak", "label_en": "Pedicab / Cycle-Rickshaw", "label_id": "Becak Kayuh (Khas Indonesia)", "color": "#EA580C"},
    {"id": 7, "name": "bajaj", "label_en": "Auto-Rickshaw", "label_id": "Bajaj Roda Tiga", "color": "#CA8A04"},
    {"id": 8, "name": "angkot", "label_en": "Minibus / Microbus", "label_id": "Angkot / Daihatsu", "color": "#0891B2"},
    {"id": 9, "name": "bentor", "label_en": "Motorized Becak", "label_id": "Becak Motor (Bentor)", "color": "#9333EA"},
    {"id": 10, "name": "gerobak", "label_en": "Handcart / Street Cart", "label_id": "Gerobak Pedagang", "color": "#4B5563"},
    {"id": 11, "name": "odong_odong", "label_en": "Tourist Tram / Train", "label_id": "Odong-odong Wisata", "color": "#DB2777"},
    {"id": 12, "name": "license_plate", "label_en": "License Plate", "label_id": "Plat Nomor Kendaraan", "color": "#D58A16"},
]

def get_custom_classes_path() -> Path:
    storage_dir = Path(settings.STORAGE_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir / "custom_classes.json"

def load_all_classes() -> List[Dict[str, Any]]:
    classes_map = {c["name"]: dict(c) for c in DEFAULT_RESEARCH_CLASSES}
    p = get_custom_classes_path()
    if p.exists():
        try:
            with open(p, "r") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    for item in saved:
                        if isinstance(item, dict) and "name" in item:
                            classes_map[item["name"]] = item
        except Exception as e:
            logger.warning("Failed to load custom classes: %s", e)
    res = list(classes_map.values())
    res.sort(key=lambda x: x.get("id", 999))
    return res

def save_custom_class(new_cls: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = get_custom_classes_path()
    current = []
    if p.exists():
        try:
            with open(p, "r") as f:
                current = json.load(f)
        except Exception:
            current = []
    existing_idx = next((i for i, c in enumerate(current) if c.get("name") == new_cls["name"]), None)
    if existing_idx is not None:
        current[existing_idx] = new_cls
    else:
        current.append(new_cls)
    with open(p, "w") as f:
        json.dump(current, f, indent=2)
    return load_all_classes()

def delete_custom_class(class_name: str) -> List[Dict[str, Any]]:
    p = get_custom_classes_path()
    if p.exists():
        try:
            with open(p, "r") as f:
                current = json.load(f)
            current = [c for c in current if c.get("name") != class_name]
            with open(p, "w") as f:
                json.dump(current, f, indent=2)
        except Exception as e:
            logger.warning("Failed to delete custom class: %s", e)
    return load_all_classes()

@router.get("/classes")
async def get_supported_classes():
    """Returns target research vehicle & traffic classes including Indonesian specialties and custom classes."""
    return {
        "classes": load_all_classes()
    }

@router.post("/classes")
async def add_custom_class(
    req: AddClassRequest,
    current_user = Depends(require_permission("datasets:write"))
):
    """Adds a new custom vehicle or traffic class."""
    slug_name = req.name.strip().lower().replace(" ", "_").replace("-", "_")
    all_classes = load_all_classes()
    
    # Determine new class ID
    max_id = max([c["id"] for c in all_classes], default=-1)
    existing = next((c for c in all_classes if c["name"] == slug_name), None)
    class_id = existing["id"] if existing else max_id + 1

    new_class_obj = {
        "id": class_id,
        "name": slug_name,
        "label_en": req.label_en or req.name.title(),
        "label_id": req.label_id or req.name.title(),
        "color": req.color or "#10B981",
        "is_custom": True,
    }
    updated = save_custom_class(new_class_obj)
    return {
        "status": "success",
        "message": f"Class '{slug_name}' registered successfully",
        "class": new_class_obj,
        "classes": updated
    }

@router.delete("/classes/{class_name}")
async def remove_custom_class(
    class_name: str,
    current_user = Depends(require_permission("datasets:write"))
):
    """Deletes a user-added custom class."""
    updated = delete_custom_class(class_name.strip().lower())
    return {
        "status": "success",
        "message": f"Class '{class_name}' removed",
        "classes": updated
    }

def resolve_image_path(raw_path: Any) -> Path:
    if not raw_path:
        return Path("")
    p = Path(raw_path)
    if p.exists():
        return p
    raw_str = str(raw_path).replace("\\", "/")
    for marker in ["/datasets/", "datasets/", "/storage/", "storage/"]:
        if marker in raw_str:
            sub = raw_str.split(marker, 1)[1]
            if "dataset" in marker:
                cand = Path(settings.STORAGE_DIR) / "datasets" / sub
            else:
                cand = Path(settings.STORAGE_DIR) / sub
            if cand.exists():
                return cand
    cand = settings.BASE_DIR / raw_str.lstrip("/")
    if cand.exists():
        return cand
    cand = Path(settings.STORAGE_DIR) / raw_str.lstrip("/")
    if cand.exists():
        return cand
    file_name = p.name
    matches = list((Path(settings.STORAGE_DIR) / "datasets").glob(f"**/{file_name}"))
    if matches:
        return matches[0]
    return p

@router.post("/auto-annotate")
async def trigger_auto_annotation(
    req: AutoAnnotateRequest,
    current_user = Depends(require_permission("datasets:write"))
):
    """Generates pseudo-labels using YOLO26n on a single image or folder."""
    if req.image_path:
        p = resolve_image_path(req.image_path)
        res = auto_annotator.annotate_file(p, save_yolo_txt=True)
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("error", "Annotation failed"))
        return res
    elif req.directory_path:
        d = Path(req.directory_path)
        if not d.is_absolute():
            d = settings.BASE_DIR / d
        res = auto_annotator.batch_annotate_directory(d, save_coco_json=True)
        return {"status": "success", "result": res}
    else:
        raise HTTPException(status_code=400, detail="Must provide image_path or directory_path")

class BatchAutoAnnotateRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)
    conf_threshold: float = Field(default=0.25, ge=0.05, le=1.0)
    status_filter: str = Field(default="unreviewed", pattern="^(unreviewed|all)$")

@router.post("/batch-auto-annotate")
async def batch_auto_annotate_unreviewed(
    req: BatchAutoAnnotateRequest = BatchAutoAnnotateRequest(),
    current_user = Depends(require_permission("datasets:write"))
):
    """Auto-annotates all unreviewed or raw frames using YOLO."""
    raw_dir = Path(settings.STORAGE_DIR) / "datasets" / "raw"
    if not raw_dir.exists():
        return {"status": "success", "processed_count": 0, "items": []}
    
    json_files = sorted(raw_dir.glob("*/*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    processed = []
    for jf in json_files:
        if len(processed) >= req.limit:
            break
        try:
            with open(jf, "r") as f:
                meta = json.load(f)
            curr_status = meta.get("review_status", "unreviewed")
            if req.status_filter == "unreviewed" and curr_status not in ["unreviewed", None]:
                continue
            
            img_path = jf.with_suffix(".jpg")
            if not img_path.exists():
                continue
            
            res = auto_annotator.annotate_file(img_path, save_yolo_txt=True)
            if res.get("success"):
                meta["review_status"] = "pseudo_annotated"
                meta["total_annotations"] = len(res.get("annotations", []))
                meta["file_path"] = str(img_path)
                with open(jf, "w") as f:
                    json.dump(meta, f, indent=2)
                processed.append({
                    "id": meta.get("id", jf.stem),
                    "file_path": str(img_path),
                    "annotations_count": len(res.get("annotations", []))
                })
        except Exception as e:
            logger.warning("Failed to auto-annotate %s: %s", jf, e)
    
    return {
        "status": "success",
        "processed_count": len(processed),
        "items": processed
    }

class BatchApproveRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)

@router.post("/batch-approve")
async def batch_approve_annotations(
    req: BatchApproveRequest = BatchApproveRequest(),
    current_user = Depends(require_permission("datasets:write"))
):
    """Batch approves frames that have annotations."""
    raw_dir = Path(settings.STORAGE_DIR) / "datasets" / "raw"
    if not raw_dir.exists():
        return {"status": "success", "approved_count": 0}
    
    json_files = sorted(raw_dir.glob("*/*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    approved = []
    for jf in json_files:
        if len(approved) >= req.limit:
            break
        try:
            with open(jf, "r") as f:
                meta = json.load(f)
            
            txt_path = jf.with_suffix(".txt")
            if txt_path.exists() and txt_path.stat().st_size > 0:
                meta["review_status"] = "approved"
                meta["reviewed_by"] = getattr(current_user, "username", "admin")
                meta["file_path"] = str(jf.with_suffix(".jpg"))
                with open(jf, "w") as f:
                    json.dump(meta, f, indent=2)
                approved.append(meta.get("id", jf.stem))
        except Exception:
            continue
            
    return {
        "status": "success",
        "approved_count": len(approved),
        "approved_ids": approved
    }

@router.get("/item")
async def get_annotation_item(
    image_path: str = Query(..., description="Absolute or relative path to image"),
    current_user = Depends(get_current_user)
):
    """Retrieves image dimensions and all active bounding boxes for review."""
    p = resolve_image_path(image_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    import cv2
    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image")
    h, w = img.shape[:2]

    # Check sidecar .txt
    txt_path = p.with_suffix(".txt")
    annotations = []
    if txt_path.exists():
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c_idx = int(parts[0])
                    xc, yc, nw, nh = map(float, parts[1:5])
                    bw = int(nw * w)
                    bh = int(nh * h)
                    x1 = int((xc * w) - (bw / 2))
                    y1 = int((yc * h) - (bh / 2))
                    x2 = x1 + bw
                    y2 = y1 + bh

                    class_name = "car"
                    for item in load_all_classes():
                        if item.get("id") == c_idx:
                            class_name = item.get("name", "car")
                            break

                    annotations.append({
                        "class_name": class_name,
                        "class_idx": c_idx,
                        "confidence": float(parts[5]) if len(parts) > 5 else 1.0,
                        "bbox": [x1, y1, x2, y2],
                        "yolo_bbox": [xc, yc, nw, nh]
                    })

    # Check metadata status
    json_path = p.with_suffix(".json")
    item_status = "unreviewed"
    metadata = {}
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
                item_status = metadata.get("review_status", "unreviewed")
        except Exception:
            pass

    return {
        "image_path": str(p),
        "width": w,
        "height": h,
        "annotations": annotations,
        "status": item_status,
        "metadata": metadata
    }

@router.put("/save")
async def save_human_annotations(
    req: SaveAnnotationRequest,
    current_user = Depends(require_permission("datasets:write"))
):
    """Saves human-corrected bounding boxes to YOLO sidecar .txt and updates review status."""
    p = resolve_image_path(req.image_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    import cv2
    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image")
    h, w = img.shape[:2]

    # Write normalized YOLO TXT
    lines = []
    for ann in req.annotations:
        c_idx = ann.class_idx
        if ann.yolo_bbox and len(ann.yolo_bbox) == 4:
            xc, yc, nw, nh = ann.yolo_bbox
        else:
            x1, y1, x2, y2 = ann.bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            nw = bw / float(w)
            nh = bh / float(h)
        lines.append(f"{c_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} {ann.confidence:.2f}")

    txt_path = p.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    # Update metadata JSON
    json_path = p.with_suffix(".json")
    meta = {}
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
        except Exception:
            pass
    meta["review_status"] = req.status
    meta["reviewed_by"] = current_user.username
    meta["total_annotations"] = len(req.annotations)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Sync to DB annotations table
    try:
        import sqlite3, uuid
        from datetime import datetime, timezone
        db_path = Path(settings.STORAGE_DIR) / "edge_vision.db"
        if db_path.exists():
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT id FROM dataset_images WHERE file_path = ? OR file_name = ?", (str(p), p.name))
                row = cur.fetchone()
                img_id = row[0] if row else p.stem
                now_str = datetime.now(timezone.utc).isoformat()
                # Update image status
                cur.execute(
                    "UPDATE dataset_images SET status = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
                    (req.status, current_user.username, now_str, img_id)
                )
                # Replace annotations for this image
                cur.execute("DELETE FROM annotations WHERE image_id = ?", (img_id,))
                for ann in req.annotations:
                    ann_id = str(uuid.uuid4())
                    y_xc = ann.yolo_bbox[0] if ann.yolo_bbox and len(ann.yolo_bbox) == 4 else 0.0
                    y_yc = ann.yolo_bbox[1] if ann.yolo_bbox and len(ann.yolo_bbox) == 4 else 0.0
                    y_w = ann.yolo_bbox[2] if ann.yolo_bbox and len(ann.yolo_bbox) == 4 else 0.0
                    y_h = ann.yolo_bbox[3] if ann.yolo_bbox and len(ann.yolo_bbox) == 4 else 0.0
                    cur.execute(
                        """
                        INSERT INTO annotations (
                            id, image_id, class_name, class_idx, confidence,
                            x_min, y_min, x_max, y_max, x_center, y_center, width, height,
                            is_crowd, status, annotator_id, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ann_id, img_id, ann.class_name, ann.class_idx, ann.confidence,
                            ann.bbox[0], ann.bbox[1], ann.bbox[2], ann.bbox[3],
                            y_xc, y_yc, y_w, y_h,
                            0, req.status, current_user.username,
                            now_str, now_str
                        )
                    )
                conn.commit()
    except Exception as e:
        logger.debug("DB sync for annotation: %s", e)

    return {
        "status": "success",
        "message": f"Saved {len(req.annotations)} annotations with status '{req.status}'",
        "annotations_count": len(req.annotations)
    }

@router.post("/export-yolo")
async def export_reviewed_dataset(
    req: ExportDatasetRequest,
    current_user = Depends(require_permission("datasets:write"))
):
    """Exports all reviewed/approved raw images into standard Ultralytics YOLO26 directory structure."""
    raw_dir = Path(settings.STORAGE_DIR) / "datasets" / "raw"
    items = []

    if raw_dir.exists():
        # Scan raw images that have sidecar txt files
        for txt_file in raw_dir.glob("*/*/*.txt"):
            img_file = txt_file.with_suffix(".jpg")
            if not img_file.exists():
                img_file = txt_file.with_suffix(".png")
            if img_file.exists():
                json_file = txt_file.with_suffix(".json")
                time_val = ""
                cam_val = txt_file.parent.parent.name
                if json_file.exists():
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                            time_val = data.get("time", "")
                    except Exception:
                        pass

                items.append({
                    "file_path": str(img_file),
                    "camera_id": cam_val,
                    "timestamp": time_val
                })

    if not items:
        raise HTTPException(status_code=400, detail="No annotated images found to export")

    export_dir = Path(settings.STORAGE_DIR) / "datasets" / "exported"
    class_names = req.classes or [c["name"] for c in load_all_classes()]
    res = dataset_exporter.export_yolo_dataset(
        raw_items=items,
        output_dir=export_dir,
        dataset_name=req.dataset_name,
        train_ratio=req.train_ratio,
        val_ratio=req.val_ratio,
        test_ratio=req.test_ratio,
        classes=class_names,
    )
    return res
