import cv2
import numpy as np
from typing import Dict, Any

class FrameQualityAssessor:
    """Evaluates frame quality: sharpness/blur score, glare, and contrast."""
    @staticmethod
    def assess(crop_bgr: np.ndarray) -> Dict[str, Any]:
        if crop_bgr is None or crop_bgr.size == 0:
            return {"score": 0.0, "is_blur": True, "is_glare": False}

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Blur evaluation using Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blur = lap_var < 100.0 # threshold for blurred crop

        # 2. Glare evaluation using top percentile brightness
        high_pixel_ratio = np.sum(gray > 240) / gray.size
        is_glare = high_pixel_ratio > 0.15

        # Quality composite score (0.0 to 1.0)
        quality_score = min(1.0, lap_var / 500.0) * (0.5 if is_glare else 1.0)

        return {
            "score": round(float(quality_score), 2),
            "laplacian_var": round(float(lap_var), 1),
            "is_blur": bool(is_blur),
            "is_glare": bool(is_glare)
        }
