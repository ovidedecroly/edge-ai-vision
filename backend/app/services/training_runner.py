import os
import sys
import uuid
import subprocess
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import sqlite3

from app.config import settings

logger = logging.getLogger("edge_vision.training_runner")

class TrainingJobRunner:
    """
    Manages asynchronous background training jobs for YOLO26.
    Executes train.py in a dedicated subprocess and tracks live status & logs.
    """

    def __init__(self):
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.db_path = str(Path(settings.STORAGE_DIR) / "edge_vision.db")

    def start_training_job(
        self,
        experiment_id: Optional[str],
        name: str,
        model_name: str = "yolo26n.pt",
        dataset_path: str = "",
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        optimizer: str = "SGD",
        learning_rate: float = 0.01,
        augmentation: str = "default",
        hardware: str = "NVIDIA CUDA / RTX",
        device: str = "cuda:0"
    ) -> Dict[str, Any]:
        """Creates a training job record in the database and launches train.py."""
        job_id = str(uuid.uuid4())
        now_str = datetime.now(timezone.utc).isoformat()
        runs_dir = Path(settings.STORAGE_DIR) / "runs"
        exp_run_dir = runs_dir / name.lower().replace(" ", "_")
        weights_dir = exp_run_dir / "weights"
        plots_dir = exp_run_dir / "plots"

        # 1. Insert DB record
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO training_jobs (
                        id, experiment_id, name, model_name, dataset_path,
                        epochs, batch_size, imgsz, optimizer, learning_rate,
                        augmentation, hardware, device, status, current_epoch,
                        total_epochs, progress_pct, best_map50, best_map50_95,
                        precision, recall, f1_score, fps, latency_ms,
                        gpu_usage_pct, weights_dir, best_pt_path, last_pt_path,
                        results_csv_path, plots_dir, log_output, started_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id, experiment_id, name, model_name, dataset_path,
                        epochs, batch_size, imgsz, optimizer, learning_rate,
                        augmentation, hardware, device, "running", 0,
                        epochs, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, str(weights_dir), str(weights_dir / "best.pt"), str(weights_dir / "last.pt"),
                        str(exp_run_dir / "results.csv"), str(plots_dir), f"Launched training job {name}",
                        now_str, now_str, now_str
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to insert training job to DB: %s", e)

        # 2. Launch train.py in background thread
        thread = threading.Thread(
            target=self._run_training_subprocess,
            args=(job_id, name, model_name, dataset_path, epochs, imgsz, batch_size, optimizer, learning_rate, device, str(runs_dir)),
            daemon=True,
            name=f"TrainingRunner-{job_id}"
        )
        thread.start()

        return {
            "job_id": job_id,
            "status": "running",
            "name": name,
            "model_name": model_name,
            "epochs": epochs,
            "output_dir": str(exp_run_dir)
        }

    def _run_training_subprocess(
        self,
        job_id: str,
        name: str,
        model_name: str,
        dataset_path: str,
        epochs: int,
        imgsz: int,
        batch_size: int,
        optimizer: str,
        learning_rate: float,
        device: str,
        runs_dir: str
    ):
        """Runs train.py and captures output."""
        python_exec = sys.executable
        train_script = Path(__file__).resolve().parent.parent.parent / "train.py"
        if not train_script.exists():
            train_script = Path(__file__).resolve().parent.parent / "train.py"

        clean_name = name.lower().replace(" ", "_")
        cmd = [
            python_exec,
            str(train_script),
            "--model", model_name,
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch_size),
            "--optimizer", optimizer,
            "--lr0", str(learning_rate),
            "--device", device,
            "--project", runs_dir,
            "--name", clean_name,
            "--job-id", job_id,
            "--db-path", self.db_path
        ]
        if dataset_path:
            cmd.extend(["--dataset", dataset_path])

        logger.info("Executing training command: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.active_processes[job_id] = proc
            logs = []
            for line in proc.stdout:
                logs.append(line)
                if len(logs) > 500:
                    logs.pop(0)

            proc.wait()
            return_code = proc.returncode
            final_status = "completed" if return_code == 0 else "failed"

            # Update final log output in DB
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                now_str = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    "UPDATE training_jobs SET status = ?, log_output = ?, completed_at = ?, updated_at = ? WHERE id = ?",
                    (final_status, "".join(logs[-100:]), now_str, now_str, job_id)
                )
                conn.commit()

        except Exception as e:
            logger.error("Error executing training process: %s", e)
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                now_str = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    "UPDATE training_jobs SET status = 'failed', log_output = ?, completed_at = ?, updated_at = ? WHERE id = ?",
                    (str(e), now_str, now_str, job_id)
                )
                conn.commit()
        finally:
            self.active_processes.pop(job_id, None)

    def cancel_job(self, job_id: str) -> bool:
        """Kills active training job process."""
        proc = self.active_processes.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                now_str = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    "UPDATE training_jobs SET status = 'cancelled', completed_at = ?, updated_at = ? WHERE id = ?",
                    (now_str, now_str, job_id)
                )
                conn.commit()
            return True
        return False

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Queries training jobs from database."""
        jobs = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ?
                    """,
                    (limit,)
                )
                for row in cur.fetchall():
                    jobs.append(dict(row))
        except Exception as e:
            logger.error("Failed to list training jobs: %s", e)
        return jobs

training_runner = TrainingJobRunner()
