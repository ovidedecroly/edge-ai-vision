import sys
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
backend_dir = project_dir / "backend"
for p in [str(project_dir), str(backend_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.config import settings
from worker.detectors.yolo_detector import YOLODetector
from worker.tracking.byte_tracker import SimpleByteTracker
from worker.recognizers.ocr_recognizer import BaselinePPOCRRecognizer, normalize_indonesian_plate
from worker.quality.frame_quality import FrameQualityAssessor
from worker.voting.temporal_voter import TemporalVoter
from worker.pipelines.pipeline_runner import PipelineRunner

def test_yolo_detector_loading_and_fallback():
    model_path = Path(settings.MODELS_DIR) / "official" / "yolo11n-coco" / "model.pt"
    assert model_path.exists(), "Official model must exist"

    detector = YOLODetector(
        model_path=str(model_path),
        device="cpu",
        enabled_classes=[0, 2] # person, car
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, latency = detector.detect(dummy_frame)
    assert isinstance(detections, list)
    assert latency > 0

def test_simple_byte_tracker():
    tracker = SimpleByteTracker(iou_threshold=0.3)
    # Frame 1: Car at [100, 100, 200, 200]
    dets1 = [{"class_id": 2, "class_name": "car", "confidence": 0.9, "bbox": [100.0, 100.0, 200.0, 200.0]}]
    tracks1 = tracker.update(dets1)
    assert len(tracks1) == 1
    track_id1 = tracks1[0]["track_id"]

    # Frame 2: Car moved slightly to [105, 105, 205, 205]
    dets2 = [{"class_id": 2, "class_name": "car", "confidence": 0.92, "bbox": [105.0, 105.0, 205.0, 205.0]}]
    tracks2 = tracker.update(dets2)
    assert len(tracks2) == 1
    assert tracks2[0]["track_id"] == track_id1, "Track ID must remain persistent across frames"

def test_ocr_recognizer_standard_format():
    recognizer = BaselinePPOCRRecognizer()
    crop = np.zeros((50, 150, 3), dtype=np.uint8)
    res = recognizer.recognize(crop)

    assert "text" in res
    assert "confidence" in res
    assert "character_confidences" in res
    assert "processing_time_ms" in res
    assert "model_id" in res
    assert res["confidence"] > 0
    assert normalize_indonesian_plate("ad-1234-xyz") == "AD1234XYZ"

def test_frame_quality_assessor():
    # Sharp gradient image
    sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp_img[:, :50] = 255
    q_sharp = FrameQualityAssessor.assess(sharp_img)
    assert "score" in q_sharp
    assert "laplacian_var" in q_sharp

    # Uniform black image (blur/flat)
    flat_img = np.zeros((100, 100, 3), dtype=np.uint8)
    q_flat = FrameQualityAssessor.assess(flat_img)
    assert q_flat["is_blur"] is True

def test_temporal_voter():
    voter = TemporalVoter(min_votes=2)
    track_id = "track-200"

    voter.add_candidate(track_id, "AD1234XYZ", 0.95, quality_score=0.9)
    voter.add_candidate(track_id, "AD1234XYZ", 0.92, quality_score=0.85)
    voter.add_candidate(track_id, "AD1234XY", 0.70, quality_score=0.5)

    consensus, conf, votes = voter.get_consensus(track_id)
    assert consensus == "AD1234XYZ", "Majority temporal vote must choose highest consensus"
    assert votes == 2
    assert conf > 0.8
