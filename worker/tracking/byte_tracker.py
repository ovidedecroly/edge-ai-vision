import math
from typing import List, Dict, Any, Tuple
import numpy as np

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

class TrackedObject:
    def __init__(self, track_id: int, bbox: List[float], class_id: int, class_name: str, confidence: float):
        self.track_id = f"track-{track_id}"
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history: List[List[float]] = [bbox]

    def update(self, bbox: List[float], confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        if len(self.history) > 30:
            self.history.pop(0)

class SimpleByteTracker:
    """
    IoU & Centroid based multi-object tracker compatible with ByteTrack association principles.
    Maintains persistent track IDs across frames and handles track termination.
    """
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[TrackedObject] = []
        self._next_id = 100

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes raw detections and returns tracked detections with persistent track_id.
        """
        for t in self.tracks:
            t.time_since_update += 1
            t.age += 1

        matched_tracks = set()
        matched_detections = set()

        # Step 1: Match high confidence detections with active tracks using IoU
        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    if t.class_id == d["class_id"]:
                        iou_matrix[i, j] = calculate_iou(t.bbox, d["bbox"])

            while True:
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if i not in matched_tracks and j not in matched_detections:
                    self.tracks[i].update(detections[j]["bbox"], detections[j]["confidence"])
                    matched_tracks.add(i)
                    matched_detections.add(j)
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Step 2: Initialize new tracks for unmatched detections
        for j, d in enumerate(detections):
            if j not in matched_detections:
                new_track = TrackedObject(
                    track_id=self._next_id,
                    bbox=d["bbox"],
                    class_id=d["class_id"],
                    class_name=d["class_name"],
                    confidence=d["confidence"]
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # Step 3: Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Step 4: Return confirmed tracks
        results = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                results.append({
                    "track_id": t.track_id,
                    "class_id": t.class_id,
                    "class_name": t.class_name,
                    "confidence": t.confidence,
                    "bbox": t.bbox,
                    "hits": t.hits
                })

        return results
