"""
Training & Ablation Study Engine for Indonesian License Plate Detection.
Executes systematic training experiments using Ultralytics YOLO26, logging metrics for academic paper tables.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
from ultralytics import YOLO


class ANPRTrainer:
    def __init__(self, runs_dir: Path, data_yaml: Path):
        self.runs_dir = Path(runs_dir)
        self.data_yaml = Path(data_yaml)
        self.results_log = self.runs_dir / "experiment_registry.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_training_experiment(
        self,
        experiment_id: str,
        base_model: str = "yolo26n.pt",
        epochs: int = 50,
        imgsz: int = 640,
        batch_size: int = 16,
        device: str = "0",  # "cpu" or "0"
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Runs one YOLO training job and captures validation metrics (Precision, Recall, mAP50, mAP50-95).
        """
        print(f"\n=======================================================")
        print(f"STARTING EXPERIMENT: {experiment_id}")
        print(f"Model: {base_model} | Epochs: {epochs} | ImgSz: {imgsz} | Device: {device}")
        print(f"Desc: {description}")
        print(f"=======================================================\n")

        t_start = time.time()
        model = YOLO(base_model)

        # Train model
        train_args = {
            "data": str(self.data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch_size,
            "device": device,
            "project": str(self.runs_dir),
            "name": experiment_id,
            "exist_ok": True,
            "verbose": True,
            "plots": True,
            "save": True,
            # Augmentation tuning for CCTV robustness
            "mosaic": 1.0,
            "degrees": 10.0,
            "perspective": 0.0005,
            "flipud": 0.0,
            "fliplr": 0.5,
        }

        results = model.train(**train_args)
        total_time_sec = round(time.time() - t_start, 2)

        # Run validation on Test split
        print("\n[INFO] Evaluating on Test Split...")
        val_results = model.val(
            data=str(self.data_yaml),
            split="test",
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        # Robust metric extraction
        try:
            p = float(val_results.box.mp) if hasattr(val_results, "box") and hasattr(val_results.box, "mp") else float(val_results.results_dict.get("metrics/precision(B)", 0.0))
            r = float(val_results.box.mr) if hasattr(val_results, "box") and hasattr(val_results.box, "mr") else float(val_results.results_dict.get("metrics/recall(B)", 0.0))
            m50 = float(val_results.box.map50) if hasattr(val_results, "box") and hasattr(val_results.box, "map50") else float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
            m95 = float(val_results.box.map) if hasattr(val_results, "box") and hasattr(val_results.box, "map") else float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))
        except Exception:
            p = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
            r = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
            m50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
            m95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))

        metrics = {
            "experiment_id": experiment_id,
            "description": description,
            "base_model": base_model,
            "epochs": epochs,
            "imgsz": imgsz,
            "total_time_min": round(total_time_sec / 60, 2),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(m50, 4),
            "map50_95": round(m95, 4),
            "fitness": round(float(val_results.fitness), 4) if hasattr(val_results, "fitness") else 0.0,
            "weights_path": str(self.runs_dir / experiment_id / "weights" / "best.pt"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._save_to_registry(metrics)
        print(f"\n[EXPERIMENT COMPLETED] {experiment_id}")
        print(f" * Precision : {metrics['precision']}")
        print(f" * Recall    : {metrics['recall']}")
        print(f" * mAP@0.5   : {metrics['map50']}")
        print(f" * mAP@0.5:95: {metrics['map50_95']}")
        print(f" * Best Path : {metrics['weights_path']}\n")
        return metrics

    def _save_to_registry(self, metrics: Dict[str, Any]):
        registry = []
        if self.results_log.exists():
            try:
                with open(self.results_log, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except Exception:
                registry = []

        registry = [r for r in registry if r.get("experiment_id") != metrics["experiment_id"]]
        registry.append(metrics)

        with open(self.results_log, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        print(f"[INFO] Metrics appended to: {self.results_log}")


if __name__ == "__main__":
    runs_path = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/runs")
    yaml_path = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/indonesian_plates_combined/data.yaml")

    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found at {yaml_path}. Run 01_dataset_curator.py first.")
        sys.exit(1)

    trainer = ANPRTrainer(runs_dir=runs_path, data_yaml=yaml_path)

    # Detect device
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using compute device: {device}")

    # Standard experiment execution
    trainer.run_training_experiment(
        experiment_id="exp1_yolo26n_multisource_plate",
        base_model="/media/ovide/DATA/PIT-UNS/edge-ai-vision/yolo26n.pt",
        epochs=30,
        imgsz=640,
        batch_size=16 if device != "cpu" else 4,
        device=device,
        description="YOLO26 Nano baseline on aggregated Indonesian multi-source license plate dataset."
    )
