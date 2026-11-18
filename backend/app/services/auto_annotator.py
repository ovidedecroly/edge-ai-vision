import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import torch
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("edge_vision.auto_annotator")

# Target Research Vehicle & Traffic Classes (Mapping COCO IDs to Project Names)
RESEARCH_CLASSES = {
    0: {"name": "person", "yolo_idx": 0},
    1: {"name": "bicycle", "yolo_idx": 1},
    2: {"name": "car", "yolo_idx": 2},
    3: {"name": "motorcycle", "yolo_idx": 3},
    5: {"name": "bus", "yolo_idx": 4},
    7: {"name": "truck", "yolo_idx": 5},
}

CLASS_NAME_TO_IDX = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 4,
    "truck": 5,
}

class AutoAnnotator:
    """
    YOLO26 / YOLO11 Automated Pseudo-labeling Engine.
    Generates draft annotations in YOLO TXT format and COCO JSON format.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "cuda:0",
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.conf = conf
        self.iou = iou
        self.enabled_coco_classes = list(RESEARCH_CLASSES.keys())

        logger.info("Initializing AutoAnnotator with model %s on %s...", model_path, self.device)
        try:
            self.model = YOLO(model_path)
            # Warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            logger.info("AutoAnnotator warmed up successfully.")
        except Exception as e:
            logger.warning("CUDA initialization failed (%s). Falling back to CPU.", e)
            self.device = "cpu"
            self.model = YOLO(model_path)

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        conf_override: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Runs model inference and produces normalized YOLO and pixel bounding boxes."""
        h, w = frame_bgr.shape[:2]
        confidence = conf_override if conf_override is not None else self.conf

        results = self.model.predict(
            frame_bgr,
            conf=confidence,
            iou=self.iou,
            classes=self.enabled_coco_classes,
            device=self.device,
            verbose=False,
        )

        annotations = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for b in boxes:
                coco_cls = int(b.cls[0].item())
                score = float(b.conf[0].item())
                xyxy = [int(v) for v in b.xyxy[0].tolist()] # [x1, y1, x2, y2]

                class_meta = RESEARCH_CLASSES.get(coco_cls, {"name": "car", "yolo_idx": 2})
                class_name = class_meta["name"]
                yolo_idx = class_meta["yolo_idx"]

                # Convert to normalized YOLO coordinates (x_center, y_center, width, height)
                x1, y1, x2, y2 = xyxy
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                xc = (x1 + x2) / 2.0 / w
                yc = (y1 + y2) / 2.0 / h
                nw = bw / float(w)
                nh = bh / float(h)

                annotations.append({
                    "class": class_name,
                    "class_idx": yolo_idx,
                    "confidence": round(score, 3),
                    "bbox": xyxy,
                    "yolo_bbox": [round(xc, 5), round(yc, 5), round(nw, 5), round(nh, 5)],
                    "status": "auto_generated",
                })

        return annotations

    def annotate_file(
        self,
        image_path: Path,
        save_yolo_txt: bool = True
    ) -> Dict[str, Any]:
        """Loads an image file, runs auto-annotation, and optionally saves sidecar .txt."""
        image_path = Path(image_path)
        if not image_path.exists():
            from app.config import settings
            file_name = image_path.name
            matches = list((Path(settings.STORAGE_DIR) / "datasets").glob(f"**/{file_name}"))
            if matches:
                image_path = matches[0]
        if not image_path.exists():
            return {"success": False, "error": f"Image file not found: {image_path}"}

        frame = cv2.imread(str(image_path))
        if frame is None:
            return {"success": False, "error": "Failed to decode image"}

        h, w = frame.shape[:2]
        annotations = self.annotate_frame(frame)

        # Save sidecar YOLO TXT format
        if save_yolo_txt:
            txt_path = image_path.with_suffix(".txt")
            lines = []
            for ann in annotations:
                idx = ann["class_idx"]
                xc, yc, nw, nh = ann["yolo_bbox"]
                lines.append(f"{idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} {ann['confidence']:.2f}")
            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

        # Sync to DB annotations table
        try:
            import sqlite3, uuid
            from datetime import datetime, timezone
            from app.config import settings
            db_path = Path(settings.STORAGE_DIR) / "edge_vision.db"
            if db_path.exists():
                with sqlite3.connect(str(db_path)) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT id FROM dataset_images WHERE file_path = ? OR file_name = ?", (str(image_path), image_path.name))
                    row = cur.fetchone()
                    img_id = row[0] if row else image_path.stem
                    now_str = datetime.now(timezone.utc).isoformat()
                    cur.execute("UPDATE dataset_images SET status = 'pseudo_annotated', updated_at = ? WHERE id = ?", (now_str, img_id))
                    cur.execute("DELETE FROM annotations WHERE image_id = ?", (img_id,))
                    for ann in annotations:
                        ann_id = str(uuid.uuid4())
                        bb = ann["bbox"]
                        ybb = ann["yolo_bbox"]
                        cur.execute(
                            """
                            INSERT INTO annotations (
                                id, image_id, class_name, class_idx, confidence,
                                x_min, y_min, x_max, y_max, x_center, y_center, width, height,
                                is_crowd, status, annotator_id, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ann_id, img_id, ann["class"], ann["class_idx"], ann["confidence"],
                                bb[0], bb[1], bb[2], bb[3],
                                ybb[0], ybb[1], ybb[2], ybb[3],
                                0, "draft", "yolo26_auto",
                                now_str, now_str
                            )
                        )
                    conn.commit()
        except Exception as e:
            logger.debug("DB sync for auto annotations: %s", e)

        return {
            "success": True,
            "image_path": str(image_path),
            "width": w,
            "height": h,
            "annotations": annotations,
            "total_objects": len(annotations),
        }

    def batch_annotate_directory(
        self,
        directory_path: Path,
        save_coco_json: bool = True
    ) -> Dict[str, Any]:
        """Batch generates pseudo-labels for all .jpg/.png files in a folder."""
        dir_path = Path(directory_path)
        image_files = sorted(list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png")))

        total = len(image_files)
        processed = 0
        total_objects = 0
        coco_images = []
        coco_annotations = []
        ann_id = 1

        for idx, img_p in enumerate(image_files, start=1):
            res = self.annotate_file(img_p, save_yolo_txt=True)
            if res.get("success"):
                processed += 1
                w, h = res["width"], res["height"]
                anns = res["annotations"]
                total_objects += len(anns)

                if save_coco_json:
                    coco_images.append({
                        "id": idx,
                        "file_name": img_p.name,
                        "width": w,
                        "height": h,
                    })
                    for a in anns:
                        x1, y1, x2, y2 = a["bbox"]
                        bw, bh = x2 - x1, y2 - y1
                        coco_annotations.append({
                            "id": ann_id,
                            "image_id": idx,
                            "category_id": a["class_idx"],
                            "bbox": [x1, y1, bw, bh],
                            "area": bw * bh,
                            "score": a["confidence"],
                            "iscrowd": 0,
                        })
                        ann_id += 1

        # Write COCO JSON if requested
        if save_coco_json and coco_images:
            categories = [{"id": v["yolo_idx"], "name": v["name"]} for v in RESEARCH_CLASSES.values()]
            coco_dataset = {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": categories,
            }
            coco_path = dir_path / "annotations_coco.json"
            with open(coco_path, "w") as f:
                json.dump(coco_dataset, f, indent=2)

        return {
            "total_images": total,
            "processed_images": processed,
            "total_objects_detected": total_objects,
            "directory": str(dir_path),
        }

auto_annotator = AutoAnnotator()
