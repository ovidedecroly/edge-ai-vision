"""
Master Pipeline & Benchmark Suite for PlantSeg (115 Plant Disease Classes)
Evaluates:
1. Standard Mask R-CNN (ResNet-50-FPN)
2. YOLO26-Backbone Mask R-CNN
3. Fast YOLO26-Mask R-CNN (ProtoNet + Light-Head)
4. Pure YOLO26-seg (One-Stage)
"""

import os
import sys
import time
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# Add scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from yolo26_mask_rcnn_model import create_yolo26_mask_rcnn
from benchmark_suite import FastYOLO26MaskRCNN, benchmark_latency_and_fps

DATA_DIR = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/data/plantseg")
ZIP_FILE = DATA_DIR / "plantsegv2.zip"
COCO_DIR = DATA_DIR / "PlantSeg" / "coco_annotations"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
OUT_DIR = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/runs/plantseg_experiments")


def organize_plantseg_images():
    """
    Unpacks plantsegv2.zip if not unpacked, and organizes images into train/val/test splits
    matching the COCO annotations.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    train_img_dir = IMAGES_DIR / "train"
    val_img_dir = IMAGES_DIR / "val"
    test_img_dir = IMAGES_DIR / "test"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)

    # Check if already organized
    if len(list(train_img_dir.glob("*.jpg"))) > 1000 and len(list(val_img_dir.glob("*.jpg"))) > 500:
        print("[INFO] PlantSeg images already organized into train/val/test splits.")
        return

    # Check if zip exists
    if not ZIP_FILE.exists():
        print(f"[WAIT] {ZIP_FILE} is still downloading...")
        return

    print(f"[INFO] Extracting {ZIP_FILE.name}...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
        zf.extractall(DATA_DIR)
    print(f"[SUCCESS] Extracted archive.")

    # Find where images are extracted
    found_images = {}
    for p in DATA_DIR.rglob("*.jpg"):
        if "images/" not in str(p):
            found_images[p.name] = p

    print(f"[INFO] Discovered {len(found_images)} raw images.")

    # Map to splits using json files
    splits = {
        "train": (COCO_DIR / "annotation_train.json", train_img_dir),
        "val": (COCO_DIR / "annotation_val.json", val_img_dir),
        "test": (COCO_DIR / "annotation_test.json", test_img_dir)
    }

    for s_name, (json_p, dest_dir) in splits.items():
        with open(json_p, "r") as f:
            c_data = json.load(f)
        count = 0
        for item in c_data["images"]:
            fname = item["file_name"]
            if fname in found_images:
                src = found_images[fname]
                dst = dest_dir / fname
                if not dst.exists():
                    shutil.copy2(src, dst)
                count += 1
        print(f"[INFO] Organized {count} images into {dest_dir}")


def benchmark_plantseg_models(num_classes: int = 116):
    """
    Benchmarks all 4 models on PlantSeg dimensions (115 disease classes + 1 background).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    print(f"\n{'='*70}")
    print(f" BENCHMARKING MODELS ON PLANTSEG (115 DISEASE CLASSES, {num_classes} DETECTOR OUTPUTS)")
    print(f"{'='*70}\n")

    # Model 1: Standard Mask R-CNN (ResNet-50-FPN)
    print("[1/4] Instantiating Standard Mask R-CNN (ResNet-50-FPN)...")
    m_std = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    p_std = sum(p.numel() for p in m_std.parameters()) / 1e6

    # Model 2: YOLO26-Backbone Mask R-CNN
    print("[2/4] Instantiating YOLO26-Backboned Mask R-CNN...")
    m_yolo26_fpn = create_yolo26_mask_rcnn(
        num_classes=num_classes,
        yolo_weights="/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt"
    )
    p_yolo26_fpn = sum(p.numel() for p in m_yolo26_fpn.parameters()) / 1e6

    # Model 3: Fast YOLO26-Mask R-CNN (ProtoNet + Light-Head)
    print("[3/4] Instantiating Fast YOLO26-Mask R-CNN (ProtoNet + Light-Head)...")
    m_fast = FastYOLO26MaskRCNN(
        num_classes=num_classes,
        yolo_weights="/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt",
        num_prototypes=32
    )
    p_fast = sum(p.numel() for p in m_fast.parameters()) / 1e6

    # Model 4: Pure YOLO26-seg
    from ultralytics import YOLO
    print("[4/4] Loading Pure YOLO26-seg...")
    yolo_model = YOLO("/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt")
    p_yolo = sum(p.numel() for p in yolo_model.model.parameters()) / 1e6

    params_summary = {
        "Standard Mask R-CNN (ResNet-50)": round(p_std, 2),
        "YOLO26-Backboned Mask R-CNN": round(p_yolo26_fpn, 2),
        "Fast YOLO26-Mask R-CNN (Ours)": round(p_fast, 2),
        "Pure YOLO26-seg": round(p_yolo, 2)
    }
    print("\nParameters Summary (M):", params_summary)

    # Latency GPU (640x640)
    print("\n---> Running GPU Benchmarks (RTX 3070 Ti, 640x640)...")
    lat_gpu_std, fps_gpu_std = benchmark_latency_and_fps(m_std, device_gpu, input_size=(1, 3, 640, 640), runs=30)
    lat_gpu_yolo26, fps_gpu_yolo26 = benchmark_latency_and_fps(m_yolo26_fpn, device_gpu, input_size=(1, 3, 640, 640), runs=30)
    lat_gpu_fast, fps_gpu_fast = benchmark_latency_and_fps(m_fast, device_gpu, input_size=(1, 3, 640, 640), runs=30)

    # Measure YOLO26-seg GPU
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        yolo_model.predict(dummy_img, device="cuda:0", verbose=False)
    t0 = time.time()
    for _ in range(30):
        yolo_model.predict(dummy_img, device="cuda:0", verbose=False)
    lat_gpu_yolo = ((time.time() - t0) / 30) * 1000
    fps_gpu_yolo = 1000.0 / lat_gpu_yolo

    print(f"Standard Mask R-CNN GPU: {lat_gpu_std:.2f} ms ({fps_gpu_std:.1f} FPS)")
    print(f"YOLO26-Backbone GPU    : {lat_gpu_yolo26:.2f} ms ({fps_gpu_yolo26:.1f} FPS)")
    print(f"Fast YOLO26-Mask GPU   : {lat_gpu_fast:.2f} ms ({fps_gpu_fast:.1f} FPS)")
    print(f"Pure YOLO26-seg GPU    : {lat_gpu_yolo:.2f} ms ({fps_gpu_yolo:.1f} FPS)")

    results = {
        "params": params_summary,
        "gpu_benchmarks": {
            "Standard Mask R-CNN": {"latency_ms": round(lat_gpu_std, 2), "fps": round(fps_gpu_std, 1)},
            "YOLO26-Backboned Mask R-CNN": {"latency_ms": round(lat_gpu_yolo26, 2), "fps": round(fps_gpu_yolo26, 1)},
            "Fast YOLO26-Mask R-CNN": {"latency_ms": round(lat_gpu_fast, 2), "fps": round(fps_gpu_fast, 1)},
            "Pure YOLO26-seg": {"latency_ms": round(lat_gpu_yolo, 2), "fps": round(fps_gpu_yolo, 1)}
        }
    }

    with open(OUT_DIR / "plantseg_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] PlantSeg benchmarks saved to: {OUT_DIR / 'plantseg_benchmark_results.json'}")


if __name__ == "__main__":
    benchmark_plantseg_models()
