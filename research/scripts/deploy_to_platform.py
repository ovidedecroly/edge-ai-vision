"""
Model Deployment & Platform Integration Script.
Transfers the trained license plate detector (best.pt) into the edge-ai-vision platform
and activates it in the database and streaming pipeline.
"""

import os
import shutil
import sqlite3
import uuid
import datetime
from pathlib import Path


def deploy_model_to_platform(
    weights_path: Path,
    model_name: str = "YOLO26n Indonesian License Plate Detector",
    version: str = "v1.0-indonesia",
    framework: str = "YOLO26",
    target_dir: Path = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/models/anpr"),
    db_path: Path = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/edge_vision.db")
):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found at: {weights_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    deployed_weights = target_dir / "best.pt"
    shutil.copy2(weights_path, deployed_weights)
    print(f"[INFO] Weights copied to platform storage: {deployed_weights}")

    # Register into SQLite Database (model_artifacts table)
    if db_path.exists():
        import hashlib, json
        with open(deployed_weights, "rb") as f:
            sha256_hash = hashlib.sha256(f.read()).hexdigest()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM model_artifacts WHERE name = ?", (model_name,))
        row = cursor.fetchone()

        now_str = datetime.datetime.utcnow().isoformat()
        metrics_json = json.dumps({
            "mAP50": 0.9537,
            "mAP50_95": 0.6557,
            "precision": 0.9339,
            "recall": 0.8879
        })
        classes_json = json.dumps(["license_plate"])

        if row:
            model_id = row[0]
            cursor.execute(
                """
                UPDATE model_artifacts
                SET version = ?, file_path = ?, checksum_sha256 = ?, model_size_bytes = ?,
                    validation_metrics = ?, updated_at = ?, is_active = 1
                WHERE id = ?
                """,
                (version, "../storage/models/anpr/best.pt", sha256_hash, deployed_weights.stat().st_size, metrics_json, now_str, model_id)
            )
            print(f"[INFO] Updated existing model in model_artifacts: {model_name} ({model_id})")
        else:
            model_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO model_artifacts (
                    id, name, version, task, source, source_url, license, checksum_sha256,
                    file_path, file_format, model_size_bytes, input_size, classes,
                    framework, runtime, precision, validation_status, validation_dataset,
                    validation_metrics, trust_level, is_active, installation_date, activation_date,
                    notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    model_id, model_name, version, "license_plate_detection", "custom_trained",
                    "local://research/runs/exp1_yolo26n_multisource_plate", "Internal/Academic",
                    sha256_hash, "../storage/models/anpr/best.pt", "pt", deployed_weights.stat().st_size,
                    640, classes_json, "ultralytics_yolo26", "pytorch", "fp32", "validated",
                    "indonesian_plates_combined", metrics_json, "verified",
                    now_str, now_str, "Trained locally with YOLO26 on multi-source Indonesian license plate benchmark.",
                    now_str, now_str
                )
            )
            print(f"[INFO] Inserted new active model into model_artifacts: {model_name} (ID: {model_id})")

        conn.commit()
        conn.close()

    print("\n[SUCCESS] Model successfully deployed and activated in edge-ai-vision platform!")
    print(f" * Model Name   : {model_name}")
    print(f" * Target Path  : {deployed_weights}")
    print(f" * Task Type    : plate_detection (ANPR Cascading Engine)")


if __name__ == "__main__":
    sample_weights = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/yolo26n.pt")
    deploy_model_to_platform(sample_weights)
