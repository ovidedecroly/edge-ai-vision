import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("worker.yolo_detector")

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light"
}

class YOLODetector:
    """
    Ultralytics YOLO11 Official Object Detector.
    Supports COCO class filtering, CUDA/CPU selection, and batch/single inference.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        confidence: float = 0.35,
        iou: float = 0.45,
        enabled_classes: Optional[List[int]] = None
    ):
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.enabled_classes = enabled_classes if enabled_classes is not None else [0, 1, 2, 3, 5, 7]

        logger.info("Loading YOLO11 detector from %s (device: %s)...", model_path, device)
        try:
            self.model = YOLO(model_path)
            # Run one warm-up inference on dummy image
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            logger.info("YOLO11 loaded and warmed up successfully.")
        except Exception as e:
            if "cuda" in self.device:
                logger.warning("CUDA failed for YOLO11 (%s). Falling back to CPU...", str(e))
                self.device = "cpu"
                self.model = YOLO(model_path)
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model.predict(dummy, device="cpu", verbose=False)
                logger.info("YOLO11 running on CPU fallback.")
            else:
                raise

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs object detection on frame_bgr.
        Returns: (detections, latency_ms)
        """
        start = time.time()
        results = self.model.predict(
            frame_bgr,
            conf=self.confidence,
            iou=self.iou,
            classes=self.enabled_classes,
            device=self.device,
            verbose=False
        )
        latency_ms = (time.time() - start) * 1000

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for b in boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].tolist() # [x1, y1, x2, y2]
                cls_name = self.model.names.get(cls_id, str(cls_id))

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(c, 1) for c in xyxy]
                })

        return detections, round(latency_ms, 2)
