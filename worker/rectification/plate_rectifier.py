import cv2
import numpy as np
from typing import Optional, List

class PlateRectifier:
    """Extracts bounding box crop and applies affine/perspective rectification."""
    @staticmethod
    def crop_box(image: np.ndarray, bbox: List[float], pad_pct: float = 0.05) -> Optional[np.ndarray]:
        if image is None or len(bbox) < 4:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        pw = (x2 - x1) * pad_pct
        ph = (y2 - y1) * pad_pct

        x1 = max(0, int(x1 - pw))
        y1 = max(0, int(y1 - ph))
        x2 = min(w, int(x2 + pw))
        y2 = min(h, int(y2 + ph))

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def rectify_perspective(crop: np.ndarray) -> np.ndarray:
        """Standardizes plate aspect ratio to 3:1 (e.g. 150x50)."""
        if crop is None or crop.size == 0:
            return crop
        return cv2.resize(crop, (150, 50), interpolation=cv2.INTER_LINEAR)
