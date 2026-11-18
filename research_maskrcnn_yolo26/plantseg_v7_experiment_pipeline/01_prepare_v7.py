import json
import shutil
from pathlib import Path
from collections import Counter
import numpy as np

from config import (
    resolve_dataset_dir, PREPARED_ROOT, EXPECTED_IMAGES, EXPECTED_VALID_ANN,
    NUM_DISEASE_CLASSES
)
from utils import load_json, save_json, sanitize_coco, ensure_dirs

def coco_to_yolo(coco, image_dir: Path, label_dir: Path):
    label_dir.mkdir(parents=True, exist_ok=True)
    img_lookup = {img["id"]: img for img in coco["images"]}
    by_img = {}
    for ann in coco["annotations"]:
        by_img.setdefault(ann["image_id"], []).append(ann)

    n_labels = 0
    n_polygons = 0
    for image_id, info in img_lookup.items():
        w, h = info["width"], info["height"]
        out = label_dir / (Path(info["file_name"]).stem + ".txt")
        lines = []
        for ann in by_img.get(image_id, []):
            cid = int(ann["category_id"])
            seg = ann.get("segmentation", [])
            if not isinstance(seg, list):
                continue
            for poly in seg:
                if len(poly) < 6:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0] / float(w), 0, 1)
                pts[:, 1] = np.clip(pts[:, 1] / float(h), 0, 1)
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
                lines.append(f"{cid} {coords}")
                n_polygons += 1
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_labels += 1
    return {"label_files": n_labels, "polygons": n_polygons}

def main():
    ds = resolve_dataset_dir()
    clean_dir = PREPARED_ROOT / "coco"
    yolo_root = PREPARED_ROOT / "yolo"
    ensure_dirs(clean_dir, yolo_root)

    report = {"dataset_dir": str(ds), "splits": {}}

    # Load official class names from Metadata.csv (since raw JSON categories list is empty)
    class_names = []
    meta_file = ds / "Metadata.csv"
    if meta_file.exists():
        import csv
        mapping = {}
        with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                mapping[int(row["Index"])] = row["Disease"]
        if len(mapping) == NUM_DISEASE_CLASSES:
            class_names = [mapping[i] for i in range(NUM_DISEASE_CLASSES)]

    for split in ["train", "val", "test"]:
        src_json = ds / f"annotation_{split}.json"
        coco = load_json(src_json)
        clean, stats = sanitize_coco(coco)

        if class_names and not clean.get("categories"):
            clean["categories"] = [{"id": i, "name": class_names[i]} for i in range(NUM_DISEASE_CLASSES)]
            stats["categories"] = len(clean["categories"])

        out_json = clean_dir / f"annotation_{split}.clean.json"
        save_json(clean, out_json)

        image_dir = ds / "images" / split
        existing = {p.name for p in image_dir.iterdir() if p.is_file()}
        json_files = {img["file_name"] for img in clean["images"]}
        missing = sorted(json_files - existing)
        extra = sorted(existing - json_files)

        yolo_stats = coco_to_yolo(clean, image_dir, yolo_root / "labels" / split)
        report["splits"][split] = {
            **stats,
            "image_files_on_disk": len(existing),
            "missing_json_images_on_disk": len(missing),
            "extra_disk_images_not_in_json": len(extra),
            "yolo": yolo_stats,
        }

        print(f"[{split}] images={stats['images']} valid_ann={stats['annotations_valid']} "
              f"orphan={stats['annotations_orphan']} missing_images={len(missing)}")

        if split in EXPECTED_IMAGES and stats["images"] != EXPECTED_IMAGES[split]:
            print(f"[WARN] Expected {EXPECTED_IMAGES[split]} images but found {stats['images']}")
        if split in EXPECTED_VALID_ANN and stats["annotations_valid"] != EXPECTED_VALID_ANN[split]:
            print(f"[WARN] Expected {EXPECTED_VALID_ANN[split]} valid anns but found {stats['annotations_valid']}")

    if not class_names or len(class_names) != NUM_DISEASE_CLASSES:
        raise RuntimeError(f"Expected {NUM_DISEASE_CLASSES} categories, found {len(class_names)}")

    # Build a non-destructive Ultralytics view under PREPARED_ROOT/yolo.
    # Images are symlinked; labels are generated from the sanitized COCO polygons.
    yolo_images = yolo_root / "images"
    yolo_images.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        link = yolo_images / split
        target = ds / "images" / split
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == target.resolve():
                pass
            elif link.is_dir() and not link.is_symlink():
                pass
            else:
                link.unlink()
                link.symlink_to(target, target_is_directory=True)
        else:
            link.symlink_to(target, target_is_directory=True)

    # Ensure dataset directory has labels link so Ultralytics path resolution works
    ds_labels = ds / "labels"
    if not ds_labels.exists() and not ds_labels.is_symlink():
        try:
            ds_labels.symlink_to(yolo_root / "labels", target_is_directory=True)
        except Exception as e:
            print(f"[WARN] Could not symlink ds_labels: {e}")

    yaml_path = yolo_root / "plantseg_v7.yaml"
    yaml_lines = [
        f"path: {yolo_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:"
    ]
    for i, n in enumerate(class_names):
        yaml_lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(yaml_lines) + "\n")

    save_json(report, PREPARED_ROOT / "audit_report.json")
    print(f"\nPrepared PlantSeg v7 at {PREPARED_ROOT}")
    print(f"Ultralytics YAML: {yaml_path}")

if __name__ == "__main__":
    main()
