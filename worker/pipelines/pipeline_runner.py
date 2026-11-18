import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from worker.detectors.yolo_detector import YOLODetector
from worker.tracking.byte_tracker import SimpleByteTracker
from worker.recognizers.ocr_recognizer import BaselinePPOCRRecognizer
from worker.rectification.plate_rectifier import PlateRectifier
from worker.quality.frame_quality import FrameQualityAssessor
from worker.voting.temporal_voter import TemporalVoter

logger = logging.getLogger("worker.pipeline_runner")

class PipelineRunner:
    """
    Modular Vision Pipeline Runner.
    Executes preset configurations: general-objects, person-monitoring,
    vehicle-monitoring, anpr-bootstrap, anpr-research.
    """
    def __init__(self, preset_id: str, yolo_model_path: str, device: str = "cuda:0"):
        self.preset_id = preset_id
        self.device = device

        # Define class filtering per preset
        if preset_id == "person-monitoring":
            classes = [0] # person
        elif preset_id in ["vehicle-monitoring", "anpr-bootstrap", "anpr-research"]:
            classes = [1, 2, 3, 5, 7] # bicycle, car, motorcycle, bus, truck
        else: # general-objects
            classes = [0, 1, 2, 3, 5, 7]

        self.detector = YOLODetector(
            model_path=yolo_model_path,
            device=device,
            confidence=0.35,
            iou=0.45,
            enabled_classes=classes
        )
        self.tracker = SimpleByteTracker()
        self.recognizer = BaselinePPOCRRecognizer()
        self.voter = TemporalVoter()

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Processes single frame through active pipeline.
        Returns typed event dictionary.
        """
        start_total = time.time()

        # Step 1: Object Detection
        detections, det_latency = self.detector.detect(frame_bgr)

        # Step 2: Multi-object Tracking
        tracked_objects = self.tracker.update(detections)

        # Step 3: ANPR pipeline processing if vehicle detected
        events = []
        recog_latency = 0.0

        if self.preset_id in ["anpr-bootstrap", "anpr-research"]:
            vehicles = [t for t in tracked_objects if t["class_name"] in ["car", "motorcycle", "bus", "truck"]]
            for v in vehicles:
                veh_crop = PlateRectifier.crop_box(frame_bgr, v["bbox"])
                if veh_crop is not None:
                    # In bootstrap mode, evaluate plate region and OCR
                    ocr_res = self.recognizer.recognize(veh_crop)
                    recog_latency += ocr_res["processing_time_ms"]

                    # Assess quality
                    quality = FrameQualityAssessor.assess(veh_crop)

                    # Temporal voting
                    self.voter.add_candidate(
                        track_id=v["track_id"],
                        text=ocr_res["text"],
                        confidence=ocr_res["confidence"],
                        quality_score=quality["score"]
                    )
                    best_text, best_conf, votes = self.voter.get_consensus(v["track_id"])

                    events.append({
                        "track_id": v["track_id"],
                        "vehicle_class": v["class_name"],
                        "vehicle_confidence": v["confidence"],
                        "vehicle_bbox": v["bbox"],
                        "plate_text": best_text,
                        "plate_confidence": best_conf,
                        "vote_count": votes,
                        "trust_level": "experimental" if self.preset_id == "anpr-bootstrap" else "local_validated"
                    })

        total_latency = (time.time() - start_total) * 1000

        return {
            "preset_id": self.preset_id,
            "tracked_objects": tracked_objects,
            "recognition_events": events,
            "latency": {
                "detection_ms": det_latency,
                "recognition_ms": round(recog_latency, 2),
                "total_ms": round(total_latency, 2)
            }
        }
