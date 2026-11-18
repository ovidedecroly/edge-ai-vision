"""
Dataset Curator & Multi-Source Harmonization Pipeline for Indonesian License Plate Detection.
Combines multiple open datasets (Roboflow, Kaggle, GitHub, Local CCTV) into a standardized YOLO dataset.
"""

import os
import shutil
import hashlib
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np


class DatasetCurator:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.train_img_dir = self.output_root / "train" / "images"
        self.train_lbl_dir = self.output_root / "train" / "labels"
        self.val_img_dir = self.output_root / "valid" / "images"
        self.val_lbl_dir = self.output_root / "valid" / "labels"
        self.valid_img_dir = self.val_img_dir
        self.valid_lbl_dir = self.val_lbl_dir
        self.test_img_dir = self.output_root / "test" / "images"
        self.test_lbl_dir = self.output_root / "test" / "labels"

        self.seen_hashes = set()
        self.stats = {
            "sources": {},
            "total_images": 0,
            "total_boxes": 0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
        }

    def initialize_dirs(self):
        for d in [
            self.train_img_dir,
            self.train_lbl_dir,
            self.val_img_dir,
            self.val_lbl_dir,
            self.test_img_dir,
            self.test_lbl_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def process_source_folder(
        self,
        source_name: str,
        images_dir: Path,
        labels_dir: Path,
        target_split: str = "auto",
        split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    ):
        """
        Reads images and labels from a source folder, standardizes class ID to 0,
        filters corrupted items, deduplicates, and places them into train/val/test.
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        if not images_dir.exists():
            print(f"[WARN] Images directory does not exist: {images_dir}")
            return

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in valid_exts]
        print(f"[{source_name}] Found {len(image_files)} images to process...")

        src_stats = {"images": 0, "boxes": 0, "skipped_duplicate": 0, "skipped_no_box": 0}
        rng = np.random.RandomState(42)

        for img_p in image_files:
            # Check duplicate
            f_hash = self.compute_file_hash(img_p)
            if f_hash in self.seen_hashes:
                src_stats["skipped_duplicate"] += 1
                continue

            lbl_p = labels_dir / f"{img_p.stem}.txt"
            standardized_boxes = []

            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                # Re-map whatever source class ID to 0 (license_plate)
                                xc = float(parts[1])
                                yc = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])

                                # Validate bounding box range
                                if 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                                    standardized_boxes.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                            except ValueError:
                                continue

            if not standardized_boxes:
                src_stats["skipped_no_box"] += 1
                continue

            # Assign split
            if target_split in ("train", "valid", "val", "test"):
                split = "val" if target_split in ("valid", "val") else target_split
            else:
                prob = rng.rand()
                if prob < split_ratio[0]:
                    split = "train"
                elif prob < split_ratio[0] + split_ratio[1]:
                    split = "val"
                else:
                    split = "test"

            # Destination
            prefix = f"{source_name}_{f_hash[:8]}"
            out_img = getattr(self, f"{split}_img_dir") / f"{prefix}{img_p.suffix.lower()}"
            out_lbl = getattr(self, f"{split}_lbl_dir") / f"{prefix}.txt"

            shutil.copy2(img_p, out_img)
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.writelines(standardized_boxes)

            self.seen_hashes.add(f_hash)
            src_stats["images"] += 1
            src_stats["boxes"] += len(standardized_boxes)
            self.stats[f"{split}_count"] += 1

        self.stats["sources"][source_name] = src_stats
        self.stats["total_images"] += src_stats["images"]
        self.stats["total_boxes"] += src_stats["boxes"]
        print(f"[{source_name}] Completed: {src_stats['images']} images accepted, {src_stats['boxes']} boxes.")

    def generate_yaml(self) -> Path:
        """Generates standard Ultralytics YOLO data.yaml configuration."""
        yaml_content = {
            "path": str(self.output_root.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 1,
            "names": {0: "license_plate"},
        }
        yaml_path = self.output_root / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"[INFO] data.yaml generated at: {yaml_path}")
        return yaml_path

    def print_summary(self):
        print("\n" + "=" * 55)
        print("--- INDONESIAN LICENSE PLATE DATASET SUMMARY ---")
        print("=" * 55)
        for s_name, s_info in self.stats["sources"].items():
            print(f" * Source [{s_name}]: {s_info['images']} images, {s_info['boxes']} boxes, {s_info['skipped_duplicate']} dups dropped")
        print("-" * 55)
        print(f"Total Unique Images : {self.stats['total_images']}")
        print(f"Total Plate Labels  : {self.stats['total_boxes']}")
        print(f"Train Set Split     : {self.stats['train_count']} ({self.stats['train_count']/max(1, self.stats['total_images'])*100:.1f}%)")
        print(f"Validation Set Split: {self.stats['val_count']} ({self.stats['val_count']/max(1, self.stats['total_images'])*100:.1f}%)")
        print(f"Test Set Split      : {self.stats['test_count']} ({self.stats['test_count']/max(1, self.stats['total_images'])*100:.1f}%)")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    out_dir = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/indonesian_plates_combined")
    raw_root = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/raw_datasets")
    
    curator = DatasetCurator(output_root=out_dir)
    curator.initialize_dirs()

    # 1. Salsabila Lighting Dataset (Multi-lighting conditions)
    salsabila_dir = raw_root / "repo_salsabila_lighting"
    if salsabila_dir.exists():
        for sp in ["train", "valid", "test"]:
            img_d = salsabila_dir / sp / "images"
            lbl_d = salsabila_dir / sp / "labels"
            if img_d.exists() and lbl_d.exists():
                curator.process_source_folder(
                    source_name=f"salsabila_lighting_{sp}",
                    images_dir=img_d,
                    labels_dir=lbl_d,
                    target_split=sp
                )

    # 2. FZNekyoo Indonesian Plates Dataset (1000 real vehicle plates)
    fznekyoo_dir = raw_root / "repo_fznekyoo_plates"
    if fznekyoo_dir.exists():
        img_d = fznekyoo_dir / "gamabr"
        lbl_d = fznekyoo_dir / "label"
        if img_d.exists() and lbl_d.exists():
            curator.process_source_folder(
                source_name="fznekyoo_plates_1000",
                images_dir=img_d,
                labels_dir=lbl_d,
                target_split="auto"
            )

    # 3. Surakarta Local CCTV Captured Frames
    cctv_dir = raw_root / "surakarta_cctv_local"
    if cctv_dir.exists():
        curator.process_source_folder(
            source_name="surakarta_dishub_cctv",
            images_dir=cctv_dir,
            labels_dir=cctv_dir,
            target_split="auto"
        )

    # 4. Storage datasets raw (if any additional captures exist)
    local_raw = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/datasets/raw")
    if local_raw.exists():
        for sub in local_raw.glob("*/*"):
            if sub.is_dir() and any(sub.glob("*.txt")):
                curator.process_source_folder(
                    source_name=f"edge_storage_{sub.name}",
                    images_dir=sub,
                    labels_dir=sub,
                    target_split="auto"
                )

    curator.generate_yaml()
    curator.print_summary()
