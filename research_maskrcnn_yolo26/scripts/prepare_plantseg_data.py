"""
Data Preparation & Annotation Converter for PlantSeg Dataset
Converts COCO instance segmentation annotations from PlantSeg into:
1. YOLO-seg polygon format for YOLO26-seg training (plantseg.yaml)
2. PyTorch Dataset format for Mask R-CNN and YOLO26-Mask R-CNN
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/data/plantseg")
COCO_DIR = DATA_DIR / "PlantSeg" / "coco_annotations"
ZIP_FILE = DATA_DIR / "plantsegv2.zip"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
YAML_FILE = DATA_DIR / "plantseg.yaml"

# Load the 115 class names
CLASS_NAMES = [
    'apple_black_rot', 'apple_mosaic_virus', 'apple_rust', 'apple_scab', 'banana_anthracnose',
    'banana_black_leaf_streak', 'banana_bunchy_top', 'banana_cigar_end_rot', 'banana_cordana_leaf_spot',
    'banana_panama_disease', 'basil_downy_mildew', 'bean_halo_blight', 'bean_mosaic_virus', 'bean_rust',
    'bell_pepper_bacterial_spot', 'bell_pepper_blossom_end_rot', 'bell_pepper_frogeye_leaf_spot',
    'bell_pepper_powdery_mildew', 'blueberry_anthracnose', 'blueberry_botrytis_blight',
    'blueberry_mummy_berry', 'blueberry_rust', 'blueberry_scorch', 'broccoli_alternaria_leaf_spot',
    'broccoli_downy_mildew', 'broccoli_ring_spot', 'cabbage_alternaria_leaf_spot', 'cabbage_black_rot',
    'cabbage_downy_mildew', 'carrot_alternaria_leaf_blight', 'carrot_cavity_spot',
    'carrot_cercospora_leaf_blight', 'cauliflower_alternaria_leaf_spot', 'cauliflower_bacterial_soft_rot',
    'celery_anthracnose', 'celery_early_blight', 'cherry_leaf_spot', 'cherry_powdery_mildew',
    'citrus_canker', 'citrus_greening_disease', 'coffee_berry_blotch', 'coffee_black_rot',
    'coffee_brown_eye_spot', 'coffee_leaf_rust', 'corn_gray_leaf_spot', 'corn_northern_leaf_blight',
    'corn_rust', 'corn_smut', 'cucumber_angular_leaf_spot', 'cucumber_bacterial_wilt',
    'cucumber_powdery_mildew', 'eggplant_cercospora_leaf_spot', 'eggplant_phomopsis_fruit_rot',
    'eggplant_phytophthora_blight', 'garlic_leaf_blight', 'garlic_rust', 'ginger_leaf_spot',
    'ginger_sheath_blight', 'grape_black_rot', 'grape_downy_mildew', 'grape_leaf_spot',
    'grapevine_leafroll_disease', 'lettuce_downy_mildew', 'lettuce_mosaic_virus', 'maple_tar_spot',
    'peach_anthracnose', 'peach_brown_rot', 'peach_leaf_curl', 'peach_rust', 'peach_scab',
    'plum_bacterial_spot', 'plum_brown_rot', 'plum_pocket_disease', 'plum_pox_virus', 'plum_rust',
    'potato_early_blight', 'potato_late_blight', 'raspberry_fire_blight', 'raspberry_gray_mold',
    'raspberry_leaf_spot', 'raspberry_yellow_rust', 'rice_blast', 'rice_sheath_blight',
    'soybean_bacterial_blight', 'soybean_brown_spot', 'soybean_downy_mildew', 'soybean_frog_eye_leaf_spot',
    'soybean_mosaic', 'soybean_rust', 'squash_powdery_mildew', 'strawberry_anthracnose',
    'strawberry_leaf_scorch', 'tobacco_blue_mold', 'tobacco_brown_spot', 'tobacco_frogeye_leaf_spot',
    'tobacco_mosaic_virus', 'tomato_bacterial_leaf_spot', 'tomato_early_blight', 'tomato_late_blight',
    'tomato_leaf_mold', 'tomato_mosaic_virus', 'tomato_septoria_leaf_spot',
    'tomato_yellow_leaf_curl_virus', 'wheat_bacterial_leaf_streak', 'wheat_head_scab',
    'wheat_leaf_rust', 'wheat_loose_smut', 'wheat_powdery_mildew', 'wheat_septoria_blotch',
    'wheat_stem_rust', 'wheat_stripe_rust', 'zucchini_bacterial_wilt', 'zucchini_downy_mildew',
    'zucchini_powdery_mildew', 'zucchini_yellow_mosaic_virus'
]


def extract_plantseg_zip():
    if not ZIP_FILE.exists():
        print(f"[ERROR] Zip file {ZIP_FILE} not found!")
        return False
    print(f"[INFO] Extracting {ZIP_FILE.name}...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
        zf.extractall(DATA_DIR)
    print(f"[SUCCESS] Extracted PlantSeg archive to {DATA_DIR}")
    return True


def convert_coco_to_yolo():
    splits = {
        "train": COCO_DIR / "annotation_train.json",
        "val": COCO_DIR / "annotation_val.json",
        "test": COCO_DIR / "annotation_test.json"
    }

    for split_name, json_path in splits.items():
        if not json_path.exists():
            print(f"[WARN] {json_path} does not exist, skipping.")
            continue

        print(f"[INFO] Converting {split_name} split from {json_path.name}...")
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        split_label_dir = LABELS_DIR / split_name
        split_label_dir.mkdir(parents=True, exist_ok=True)

        images_dict = {img["id"]: img for img in coco_data["images"]}
        img_annotations: Dict[int, List] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)

        count = 0
        for img_id, img_info in images_dict.items():
            img_w = img_info["width"]
            img_h = img_info["height"]
            fname = Path(img_info["file_name"]).stem
            label_file = split_label_dir / f"{fname}.txt"

            anns = img_annotations.get(img_id, [])
            lines = []
            for ann in anns:
                cat_id = ann["category_id"]
                segm = ann.get("segmentation", [])
                if not segm:
                    continue

                for poly in segm:
                    if len(poly) < 6:
                        continue
                    coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] = np.clip(coords[:, 0] / img_w, 0.0, 1.0)
                    coords[:, 1] = np.clip(coords[:, 1] / img_h, 0.0, 1.0)
                    coord_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in coords])
                    lines.append(f"{cat_id} {coord_str}")

            if lines:
                with open(label_file, "w") as lf:
                    lf.write("\n".join(lines) + "\n")
                count += 1

        print(f"[INFO] Created {count} YOLO label files in {split_label_dir}")

    # Generate plantseg.yaml
    yaml_content = f"""# Ultralytics YOLO26-seg Dataset Configuration for PlantSeg (115 Disease Categories)
path: {DATA_DIR}
train: images/train
val: images/val
test: images/test

names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    with open(YAML_FILE, "w") as yf:
        yf.write(yaml_content)
    print(f"[SUCCESS] plantseg.yaml generated at {YAML_FILE}")


if __name__ == "__main__":
    convert_coco_to_yolo()
