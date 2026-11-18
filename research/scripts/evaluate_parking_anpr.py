#!/usr/bin/env python3
"""
Comprehensive Indonesian Parking ANPR Research Pipeline & Ablation Benchmark.
Combines:
  1. YOLO26 High-Resolution Plate Detector
  2. Spatial Layout Bisection (70:30 upper registration / lower tax expiry)
  3. Adaptive Enhancement (CLAHE + Bilateral)
  4. Sequence Character Recognition (EasyOCR)
  5. Korlantas Syntactic Grammar & Disambiguation Parser
"""

import sys
import re
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parking_anpr")

# Whitelist of Official Indonesian Samsat Region Codes
SAMSAT_REGIONS = {
    "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T",
    "AA", "AB", "AD", "AE", "AG", "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL",
    "BM", "BN", "BP", "DA", "DB", "DC", "DD", "DE", "DG", "DH", "DK", "DM", "DN",
    "DR", "DT", "EA", "EB", "ED", "KB", "KH", "KT", "KU", "PA", "PB"
}

# Visual confusion dictionaries for position-aware correction
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B", "@": "A"
}
LETTER_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "E": "3", "A": "4",
    "S": "5", "G": "6", "B": "8", "T": "7"
}


class KorlantasSyntacticParser:
    """
    Applies Indonesian Motor Vehicle Registration (Korlantas Polri) formal grammar rules:
    Format: [1-2 Region Letters] + [1-4 Registration Digits] + [1-3 Suffix Letters]
    """

    @staticmethod
    def clean_raw_ocr(text_list: List[str]) -> str:
        raw = "".join(text_list).upper()
        # Remove noisy symbols common from plate frames/bolts
        cleaned = re.sub(r'[^A-Z0-9]', '', raw)
        return cleaned

    @classmethod
    def disambiguate_plate(cls, raw_str: str) -> Dict[str, Any]:
        """
        Parses and disambiguates candidate Indonesian plate string.
        """
        raw_str = raw_str.upper().strip()
        if not raw_str:
            return {"valid": False, "formatted": "", "region": "", "number": "", "suffix": ""}

        # Strip common leading noise characters from border screws: e.g. (0D -> AD, @D -> AD
        if len(raw_str) > 2 and raw_str[0] in ["@", "(", "|", "!", "1", "0"] and raw_str[1] == "D":
            raw_str = "A" + raw_str[1:]
        elif len(raw_str) > 2 and raw_str[0] == "8" and raw_str[1].isdigit():
            # '8' followed by digit is almost always Jakarta plate 'B'
            raw_str = "B" + raw_str[1:]

        # Regex search for candidate pattern: 1-2 letters, 1-4 digits, 1-3 letters
        pattern = r'^([A-Z0-9]{1,2})([0-9A-Z]{1,4})([A-Z0-9]{1,3})'
        match = re.search(pattern, raw_str)
        if not match:
            # Fallback simple letter/digit segmentation
            parts = re.findall(r'[A-Za-z]+|\d+', raw_str)
            if len(parts) >= 2:
                reg_cand = parts[0]
                num_cand = parts[1]
                suf_cand = parts[2] if len(parts) > 2 else ""
            else:
                return {"valid": False, "formatted": raw_str, "region": "", "number": "", "suffix": ""}
        else:
            reg_cand, num_cand, suf_cand = match.groups()

        # Disambiguate Region (Letters only)
        fixed_reg = "".join(DIGIT_TO_LETTER.get(c, c) for c in reg_cand)
        if fixed_reg not in SAMSAT_REGIONS:
            # Check single letter fallback
            if fixed_reg[:1] in SAMSAT_REGIONS:
                # Extra character might belong to number
                num_cand = fixed_reg[1:] + num_cand
                fixed_reg = fixed_reg[:1]

        # Disambiguate Number (Digits only)
        fixed_num = "".join(LETTER_TO_DIGIT.get(c, c) for c in num_cand if c.isalnum())
        # Truncate to maximum 4 digits per Indonesian law
        fixed_num = fixed_num[:4]

        # Disambiguate Suffix (Letters only)
        fixed_suf = "".join(DIGIT_TO_LETTER.get(c, c) for c in suf_cand if c.isalnum())
        fixed_suf = fixed_suf[:3]

        formatted = f"{fixed_reg} {fixed_num} {fixed_suf}".strip()
        is_valid = bool(fixed_reg in SAMSAT_REGIONS and fixed_num and len(fixed_num) <= 4)

        return {
            "valid": is_valid,
            "formatted": formatted,
            "region": fixed_reg,
            "number": fixed_num,
            "suffix": fixed_suf,
            "raw": raw_str,
        }

    @staticmethod
    def parse_tax_expiry(raw_lower: List[str]) -> str:
        """Parses MM.YY tax expiration date from lower sub-ROI."""
        cleaned = "".join(raw_lower).replace(" ", "")
        digits = re.sub(r'\D', '', cleaned)
        if len(digits) >= 4:
            mm, yy = digits[:2], digits[2:4]
            try:
                if 1 <= int(mm) <= 12:
                    return f"{mm}.{yy}"
            except ValueError:
                pass
        return ""


def preprocess_plate_image(crop: np.ndarray, method: str = "proposed_full") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Applies image enhancement and spatial layout decomposition.
    """
    ch, cw = crop.shape[:2]
    if method == "baseline":
        return crop, None

    if method == "clahe_only":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray), None

    if method == "bisection_only":
        upper = crop[:int(ch * 0.70), :]
        lower = crop[int(ch * 0.60):, :]
        return upper, lower

    # Proposed full pipeline: 3% margin crop (screw/frame removal) + 70:30 Bisection + CLAHE + Bilateral
    m_y = max(1, int(ch * 0.03))
    m_x = max(1, int(cw * 0.03))
    inner_crop = crop[m_y:ch - m_y, m_x:cw - m_x]
    ih, iw = inner_crop.shape[:2]

    upper_raw = inner_crop[:int(ih * 0.70), :]
    lower_raw = inner_crop[int(ih * 0.58):, :]

    # Upper enhancement (Registration Number)
    upper_gray = cv2.cvtColor(upper_raw, cv2.COLOR_BGR2GRAY)
    upper_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(upper_gray)
    upper_filtered = cv2.bilateralFilter(upper_clahe, d=5, sigmaColor=50, sigmaSpace=50)

    # Lower enhancement (Tax Expiry)
    lower_gray = cv2.cvtColor(lower_raw, cv2.COLOR_BGR2GRAY)
    scale = max(1.5, 40.0 / max(1, lower_gray.shape[0]))
    lower_resized = cv2.resize(lower_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, lower_otsu = cv2.threshold(lower_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return upper_filtered, lower_otsu


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev[j + 1] + 1
            delete = curr[j] + 1
            replace = prev[j] + (c1 != c2)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def character_error_rate(pred: str, gt: str) -> float:
    """CER = edit_distance / len(gt)."""
    p = re.sub(r'[^A-Z0-9]', '', pred.upper())
    g = re.sub(r'[^A-Z0-9]', '', gt.upper())
    if not g:
        return 0.0 if not p else 1.0
    dist = levenshtein_distance(p, g)
    return min(1.0, dist / len(g))


class ParkingANPREvaluator:
    def __init__(self, weights_path: Path, use_gpu: bool = False):
        self.plate_model = YOLO(str(weights_path))
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
        self.parser = KorlantasSyntacticParser()

    def process_image(self, img_bgr: np.ndarray, method: str = "proposed_full") -> Dict[str, Any]:
        h, w = img_bgr.shape[:2]
        t0 = time.perf_counter()

        # Step 1: Plate Detection
        det_res = self.plate_model(img_bgr, conf=0.25, verbose=False)[0]
        t_det = (time.perf_counter() - t0) * 1000

        boxes = det_res.boxes.xyxy.cpu().numpy()
        confs = det_res.boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            return {
                "detected": False,
                "plate_raw": "",
                "plate_formatted": "",
                "tax_expiry": "",
                "is_valid": False,
                "latency_total_ms": t_det,
                "latency_det_ms": t_det,
                "latency_ocr_ms": 0.0,
            }

        # Select highest confidence plate detection
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = map(int, boxes[best_idx])
        plate_crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        # Step 2: Preprocessing & Bisection
        t1 = time.perf_counter()
        upper_roi, lower_roi = preprocess_plate_image(plate_crop, method=method)
        t_prep = (time.perf_counter() - t1) * 1000

        # Step 3: OCR Recognition
        t2 = time.perf_counter()
        upper_ocr = self.reader.readtext(upper_roi, detail=0)
        lower_ocr = self.reader.readtext(lower_roi, detail=0) if lower_roi is not None else []
        t_ocr = (time.perf_counter() - t2) * 1000

        raw_upper_str = self.parser.clean_raw_ocr(upper_ocr)

        # Step 4: Syntactic Disambiguation & Grammar Correction
        if method == "proposed_full":
            parsed = self.parser.disambiguate_plate(raw_upper_str)
            expiry_str = self.parser.parse_tax_expiry(lower_ocr)
        else:
            # Baseline just returns whatever OCR produced
            parsed = {
                "valid": bool(raw_upper_str),
                "formatted": raw_upper_str,
                "region": "", "number": "", "suffix": "",
                "raw": raw_upper_str
            }
            expiry_str = ""

        t_total = (time.perf_counter() - t0) * 1000

        return {
            "detected": True,
            "conf": float(confs[best_idx]),
            "bbox": [x1, y1, x2, y2],
            "plate_raw": raw_upper_str,
            "plate_formatted": parsed["formatted"],
            "region": parsed.get("region", ""),
            "number": parsed.get("number", ""),
            "suffix": parsed.get("suffix", ""),
            "tax_expiry": expiry_str,
            "is_valid": parsed.get("valid", False),
            "latency_total_ms": round(t_total, 2),
            "latency_det_ms": round(t_det, 2),
            "latency_prep_ms": round(t_prep, 2),
            "latency_ocr_ms": round(t_ocr, 2),
        }


def run_ablation_benchmark(samples: List[Tuple[Path, str]], weights_path: Path, output_json: Path):
    logger.info("Initializing Parking ANPR Evaluator...")
    evaluator = ParkingANPREvaluator(weights_path=weights_path, use_gpu=False)

    methods = ["baseline", "clahe_only", "bisection_only", "proposed_full"]
    results = {}

    for method in methods:
        logger.info(f"Running Ablation Method: [{method}] on {len(samples)} samples...")
        exact_matches = 0
        cer_scores = []
        latencies = []
        valid_grammars = 0
        details = []

        for img_path, gt_plate in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            res = evaluator.process_image(img, method=method)
            pred_clean = re.sub(r'[^A-Z0-9]', '', res["plate_formatted"].upper())
            gt_clean = re.sub(r'[^A-Z0-9]', '', gt_plate.upper())

            is_exact = (pred_clean == gt_clean)
            cer = character_error_rate(pred_clean, gt_clean)

            if is_exact:
                exact_matches += 1
            if res.get("is_valid", False):
                valid_grammars += 1

            cer_scores.append(cer)
            latencies.append(res["latency_total_ms"])

            details.append({
                "image": img_path.name,
                "gt": gt_plate,
                "pred_formatted": res["plate_formatted"],
                "tax_expiry": res["tax_expiry"],
                "exact_match": is_exact,
                "cer": round(cer, 4),
                "latency_ms": res["latency_total_ms"],
            })

        n = max(1, len(samples))
        results[method] = {
            "total_samples": len(samples),
            "exact_match_count": exact_matches,
            "exact_match_accuracy": round((exact_matches / n) * 100.0, 2),
            "mean_cer": round(float(np.mean(cer_scores)) * 100.0, 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "valid_syntax_rate": round((valid_grammars / n) * 100.0, 2),
            "details": details[:10]  # sample previews
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved evaluation results to: {output_json}")

    print("\n" + "=" * 80)
    print("INDONESIAN PARKING ANPR ABLATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Method / Configuration':<25} | {'Exact Match Acc':<16} | {'Mean CER':<10} | {'Syntax Validity':<16} | {'Latency':<10}")
    print("-" * 80)
    for m, r in results.items():
        print(f"{m:<25} | {r['exact_match_accuracy']:>14.1f}% | {r['mean_cer']:>8.1f}% | {r['valid_syntax_rate']:>14.1f}% | {r['mean_latency_ms']:>7.1f} ms")
    print("=" * 80 + "\n")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Indonesian Parking ANPR with OCR and Syntax Rules")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum evaluation samples")
    args = parser.parse_args()

    weights_path = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/models/anpr/best.pt")
    output_json = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/runs/parking_anpr_experiment_results.json")

    samples = []
    
    # 1. Local img88 parking gate ground truth images (large frontal view)
    local_test_dir = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/raw_datasets/repo_img88_plates/test images")
    if local_test_dir.exists():
        for p in sorted(local_test_dir.glob("*.jpg")):
            samples.append((p, p.stem))
    logger.info(f"Loaded {len(samples)} local parking gate test images.")

    # 2. Kaggle Indonesian Plate Multi-Source Text Dataset
    kaggle_label_csv = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/raw_datasets/kaggle_linkgish_plates/plate_text_dataset/plate_text_dataset/label.csv")
    kaggle_img_dir = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/raw_datasets/kaggle_linkgish_plates/plate_text_dataset/plate_text_dataset/dataset")

    if kaggle_label_csv.exists() and kaggle_img_dir.exists():
        import pandas as pd
        df = pd.read_csv(kaggle_label_csv)
        added = 0
        for _, row in df.iterrows():
            fname = str(row["filename"]).strip()
            label = str(row["label"]).strip()
            img_p = kaggle_img_dir / fname
            if img_p.exists() and label and len(label) >= 4:
                samples.append((img_p, label))
                added += 1
                if added >= (args.max_samples - 5):
                    break
        logger.info(f"Loaded {added} ground truth samples from Kaggle plate_text_dataset.")

    if not samples:
        logger.error("No test samples found.")
        return

    logger.info(f"Total benchmark evaluation samples: {len(samples)}")
    run_ablation_benchmark(samples, weights_path=weights_path, output_json=output_json)


if __name__ == "__main__":
    main()

