import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class ImageQualityFilter:
    """
    Evaluates frame quality for automated dataset curation:
    - Blur score (Laplacian variance)
    - Brightness (average grayscale intensity)
    - Glare / Overexposure ratio
    - Duplicate detection via perceptual difference hash (dHash)
    """

    def __init__(
        self,
        min_blur_score: float = 30.0,
        min_brightness: float = 35.0,
        max_brightness: float = 230.0,
        max_glare_ratio: float = 0.20,
        duplicate_hamming_threshold: int = 4
    ):
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_glare_ratio = max_glare_ratio
        self.duplicate_hamming_threshold = duplicate_hamming_threshold

    @staticmethod
    def compute_dhash(gray: np.ndarray, hash_size: int = 8) -> str:
        """Computes a 64-bit difference hash (dHash) for fast duplicate detection."""
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return "".join(["1" if b else "0" for b in diff.flatten()])

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Computes Hamming distance between two binary hashes."""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 999
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def evaluate(
        self,
        frame_bgr: np.ndarray,
        previous_dhash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a frame and returns comprehensive quality metrics and acceptance decision.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "accepted": False,
                "rejection_reasons": ["empty_frame"],
                "blur_score": 0.0,
                "brightness": 0,
                "glare_ratio": 0.0,
                "dhash": "",
                "is_duplicate": False,
                "hamming_distance": 999
            }

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Blur evaluation using Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Brightness evaluation (0-255)
        brightness = float(np.mean(gray))

        # 3. Glare evaluation
        glare_ratio = float(np.sum(gray > 245) / gray.size)

        # 4. dHash for duplicate check
        current_hash = self.compute_dhash(gray)
        hamming_dist = self.hamming_distance(current_hash, previous_dhash) if previous_dhash else 999
        is_duplicate = hamming_dist <= self.duplicate_hamming_threshold

        # Decision logic
        rejection_reasons: List[str] = []
        if blur_score < self.min_blur_score:
            rejection_reasons.append("too_blurry")
        if brightness < self.min_brightness:
            rejection_reasons.append("too_dark")
        if brightness > self.max_brightness or glare_ratio > self.max_glare_ratio:
            rejection_reasons.append("too_bright_or_glare")
        if is_duplicate:
            rejection_reasons.append("duplicate_frame")

        accepted = len(rejection_reasons) == 0

        return {
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "blur_score": round(blur_score, 1),
            "brightness": int(round(brightness)),
            "glare_ratio": round(glare_ratio, 3),
            "dhash": current_hash,
            "is_duplicate": is_duplicate,
            "hamming_distance": hamming_dist,
            "resolution": f"{w}x{h}"
        }

quality_filter = ImageQualityFilter()
