import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

logger = logging.getLogger("edge_vision.dataset_exporter")

DEFAULT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "becak",
    "bajaj",
    "angkot",
    "bentor",
    "gerobak",
    "odong_odong",
    "license_plate"
]

class DatasetExporter:
    """
    Exports reviewed annotations into standard Ultralytics YOLO26 training format.
    Enforces NON-RANDOM temporal/camera splitting to prevent continuous video frame leakage.
    """

    @staticmethod
    def export_yolo_dataset(
        raw_items: List[Dict[str, Any]],
        output_dir: Path,
        dataset_name: str = "dishub_traffic_yolo",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Splits dataset by timestamp/camera grouping and copies images + labels.
        """
        output_dir = Path(output_dir)
        dataset_root = output_dir / dataset_name
        
        # Prepare Ultralytics folder structure
        for split in ["train", "val", "test"]:
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        # 1. Non-random temporal sorting: sort all items chronologically
        sorted_items = sorted(
            raw_items,
            key=lambda x: (x.get("camera_id", ""), x.get("timestamp", x.get("time", "")))
        )

        total = len(sorted_items)
        if total == 0:
            return {"success": False, "error": "No items to export"}

        # 2. Assign splits chronologically (e.g. days 1-5 train, day 6 val, day 7 test)
        n_train = max(1, int(total * train_ratio))
        n_val = max(1, int(total * val_ratio)) if total > 2 else 0

        counts = {"train": 0, "val": 0, "test": 0}

        for idx, item in enumerate(sorted_items):
            if idx < n_train:
                split = "train"
            elif idx < (n_train + n_val):
                split = "val"
            else:
                split = "test"

            img_src = Path(item["file_path"])
            if not img_src.exists():
                continue

            # Target paths
            img_dest = dataset_root / "images" / split / img_src.name
            label_dest = dataset_root / "labels" / split / img_src.with_suffix(".txt").name

            # Copy image
            shutil.copy2(img_src, img_dest)

            # Copy or generate label .txt
            label_src = img_src.with_suffix(".txt")
            if label_src.exists():
                shutil.copy2(label_src, label_dest)
            elif "annotations" in item:
                # Generate from annotations list
                lines = []
                for ann in item["annotations"]:
                    c_idx = ann.get("class_idx", 2)
                    xc, yc, w, h = ann["yolo_bbox"]
                    lines.append(f"{c_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                with open(label_dest, "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
            else:
                # Empty file (no detections)
                label_dest.touch()

            counts[split] += 1

        # 3. Create data.yaml
        classes_to_export = classes if classes is not None else DEFAULT_CLASSES
        yaml_content = {
            "path": str(dataset_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(classes_to_export)},
            "nc": len(classes_to_export)
        }

        yaml_path = dataset_root / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        return {
            "success": True,
            "dataset_root": str(dataset_root),
            "data_yaml": str(yaml_path),
            "counts": counts,
            "total_exported": sum(counts.values()),
            "split_strategy": "chronological_camera_block"
        }

dataset_exporter = DatasetExporter()
