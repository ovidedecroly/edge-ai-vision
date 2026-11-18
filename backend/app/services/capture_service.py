import os
import json
import time
import asyncio
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2

from app.config import settings
from app.services.quality_filter import quality_filter

logger = logging.getLogger("edge_vision.capture_service")

class CaptureJob:
    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        stream_url: str,
        interval_seconds: int = 5,
        dataset_id: Optional[str] = None,
        max_storage_mb: int = 10000,
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.stream_url = stream_url
        self.interval_seconds = max(1, interval_seconds)
        self.dataset_id = dataset_id
        self.max_storage_mb = max_storage_mb

        self.time_filter_enabled: bool = False
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.schedule_preset: Optional[str] = None
        self.schedule_label: Optional[str] = None

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.last_dhash: Optional[str] = None
        self.total_captured = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.last_capture_time: Optional[str] = None
        self.last_status: str = "idle"
        self.last_error: Optional[str] = None

class DatasetCaptureManager:
    """
    Manages automated & scheduled frame captures from CCTV streams,
    quality evaluation, and storage hierarchy organization.
    """

    def __init__(self):
        self.jobs: Dict[str, CaptureJob] = {}
        storage_base = Path(settings.STORAGE_DIR)
        self.raw_dir = storage_base / "datasets" / "raw"
        self.rejected_dir = storage_base / "datasets" / "rejected"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

    def start_job(
        self,
        camera_id: str,
        camera_name: str,
        stream_url: str,
        interval_seconds: int = 5,
        dataset_id: Optional[str] = None,
        time_filter_enabled: bool = False,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        schedule_preset: Optional[str] = None,
        schedule_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Starts a periodic capture worker for a camera with optional time window filter."""
        if camera_id in self.jobs and self.jobs[camera_id].is_running:
            return self.get_job_status(camera_id)

        job = CaptureJob(
            camera_id=camera_id,
            camera_name=camera_name,
            stream_url=stream_url,
            interval_seconds=interval_seconds,
            dataset_id=dataset_id,
        )
        job.time_filter_enabled = time_filter_enabled
        job.start_time = start_time
        job.end_time = end_time
        job.schedule_preset = schedule_preset
        job.schedule_label = schedule_label

        job.is_running = True
        job._stop_event.clear()
        job._thread = threading.Thread(
            target=self._capture_worker,
            args=(job,),
            daemon=True,
            name=f"CaptureWorker-{camera_id}"
        )
        job._thread.start()
        self.jobs[camera_id] = job
        logger.info(
            "Started CaptureJob for camera %s (interval=%ds, schedule=%s, window=%s-%s)",
            camera_id, interval_seconds, schedule_label or "Continuous", start_time, end_time
        )
        return self.get_job_status(camera_id)

    def stop_job(self, camera_id: str) -> Dict[str, Any]:
        """Stops an active capture worker."""
        if camera_id in self.jobs:
            job = self.jobs[camera_id]
            job.is_running = False
            job._stop_event.set()
            if job._thread and job._thread.is_alive():
                job._thread.join(timeout=2.0)
            logger.info("Stopped CaptureJob for camera %s", camera_id)
            return self.get_job_status(camera_id)
        return {"camera_id": camera_id, "is_running": False, "status": "not_found"}

    def get_job_status(self, camera_id: str) -> Dict[str, Any]:
        job = self.jobs.get(camera_id)
        if not job:
            return {"camera_id": camera_id, "is_running": False, "status": "stopped"}
        
        # Calculate whether currently inside scheduled window
        is_inside_window = True
        current_time_str = ""
        try:
            from datetime import timezone, timedelta
            wib = timezone(timedelta(hours=7))
            now_wib = datetime.now(wib)
            current_time_str = now_wib.strftime("%H:%M")
            if job.time_filter_enabled and job.start_time and job.end_time:
                s_h, s_m = map(int, job.start_time.split(":"))
                e_h, e_m = map(int, job.end_time.split(":"))
                now_min = now_wib.hour * 60 + now_wib.minute
                s_min = s_h * 60 + s_m
                e_min = e_h * 60 + e_m
                if s_min <= e_min:
                    is_inside_window = (s_min <= now_min <= e_min)
                else:
                    is_inside_window = (now_min >= s_min or now_min <= e_min)
        except Exception:
            pass

        return {
            "camera_id": job.camera_id,
            "camera_name": job.camera_name,
            "is_running": job.is_running,
            "interval_seconds": job.interval_seconds,
            "dataset_id": job.dataset_id,
            "time_filter_enabled": job.time_filter_enabled,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "schedule_preset": job.schedule_preset,
            "schedule_label": job.schedule_label or ("24 Jam / Continuous" if not job.time_filter_enabled else f"{job.start_time} - {job.end_time}"),
            "is_inside_window": is_inside_window,
            "current_time_wib": current_time_str,
            "total_captured": job.total_captured,
            "accepted_count": job.accepted_count,
            "rejected_count": job.rejected_count,
            "last_capture_time": job.last_capture_time,
            "last_status": job.last_status,
            "last_error": job.last_error,
        }

    def list_all_jobs(self) -> List[Dict[str, Any]]:
        return [self.get_job_status(cid) for cid in self.jobs]

    def capture_single_frame(
        self,
        camera_id: str,
        camera_name: str,
        stream_url: str,
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synchronously captures one frame, filters quality, and saves with metadata."""
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return {"success": False, "error": f"Failed to open stream {stream_url}"}

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"success": False, "error": "Failed to read frame from stream"}

        eval_res = quality_filter.evaluate(frame, previous_dhash=None)
        metadata = self._save_captured_frame(
            camera_id=camera_id,
            camera_name=camera_name,
            frame=frame,
            quality_info=eval_res,
            dataset_id=dataset_id
        )
        return {"success": True, "metadata": metadata}

    def _capture_worker(self, job: CaptureJob):
        """Background thread executing interval frame captures with time window schedule check."""
        while not job._stop_event.is_set():
            try:
                # 1. Check time window schedule if enabled
                if job.time_filter_enabled and job.start_time and job.end_time:
                    from datetime import timezone, timedelta
                    wib = timezone(timedelta(hours=7))
                    now_wib = datetime.now(wib)
                    now_min = now_wib.hour * 60 + now_wib.minute
                    now_str = now_wib.strftime("%H:%M")

                    try:
                        s_h, s_m = map(int, job.start_time.split(":"))
                        e_h, e_m = map(int, job.end_time.split(":"))
                        s_min = s_h * 60 + s_m
                        e_min = e_h * 60 + e_m

                        if s_min <= e_min:
                            in_window = (s_min <= now_min <= e_min)
                        else:
                            in_window = (now_min >= s_min or now_min <= e_min)
                    except Exception:
                        in_window = True

                    if not in_window:
                        job.last_status = f"Menunggu Jadwal ({job.start_time} - {job.end_time} WIB, sekarang {now_str})"
                        # Sleep briefly then re-check
                        job._stop_event.wait(timeout=min(job.interval_seconds, 15))
                        continue

                # 2. Capture frame from CCTV
                cap = cv2.VideoCapture(job.stream_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    job.last_status = "connection_failed"
                    job.last_error = f"Cannot connect to {job.stream_url}"
                    time.sleep(min(job.interval_seconds, 5))
                    continue

                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    job.total_captured += 1
                    eval_res = quality_filter.evaluate(frame, previous_dhash=job.last_dhash)
                    
                    if eval_res["accepted"]:
                        job.accepted_count += 1
                        job.last_dhash = eval_res["dhash"]
                        job.last_status = "accepted"
                    else:
                        job.rejected_count += 1
                        job.last_status = f"rejected: {','.join(eval_res['rejection_reasons'])}"

                    self._save_captured_frame(
                        camera_id=job.camera_id,
                        camera_name=job.camera_name,
                        frame=frame,
                        quality_info=eval_res,
                        dataset_id=job.dataset_id
                    )

                    job.last_capture_time = datetime.now(timezone.utc).isoformat()
                    job.last_error = None
                else:
                    job.last_status = "frame_drop"

            except Exception as e:
                logger.error("Capture worker error: %s", e)
                job.last_error = str(e)
                job.last_status = "error"

            # Sleep for interval
            job._stop_event.wait(timeout=job.interval_seconds)

    def _save_captured_frame(
        self,
        camera_id: str,
        camera_name: str,
        frame,
        quality_info: Dict[str, Any],
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Saves image to hierarchical path and writes metadata JSON."""
        now = datetime.now(timezone.utc)
        date_folder = now.strftime("%Y-%m-%d")
        timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        image_id = f"img_{camera_id}_{timestamp_str}"
        filename = f"{image_id}.jpg"

        is_accepted = quality_info.get("accepted", False)
        base_dir = self.raw_dir if is_accepted else self.rejected_dir
        dest_folder = base_dir / camera_id / date_folder
        dest_folder.mkdir(parents=True, exist_ok=True)

        image_path = dest_folder / filename
        cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Metadata format requested in spec
        metadata = {
            "id": image_id,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "time": now.isoformat(),
            "file_path": str(image_path),
            "relative_url": f"/datasets/{'raw' if is_accepted else 'rejected'}/{camera_id}/{date_folder}/{filename}",
            "resolution": quality_info.get("resolution", "unknown"),
            "brightness": quality_info.get("brightness", 0),
            "blur_score": quality_info.get("blur_score", 0.0),
            "quality_status": "accepted" if is_accepted else "rejected",
            "rejection_reasons": quality_info.get("rejection_reasons", []),
            "dataset_id": dataset_id,
        }

        # Write sidecar JSON metadata
        meta_path = dest_folder / f"{image_id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Sync to DB dataset_images table
        try:
            import sqlite3
            db_path = Path(settings.STORAGE_DIR) / "edge_vision.db"
            if db_path.exists():
                with sqlite3.connect(str(db_path)) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO dataset_images (
                            id, camera_id, camera_name, dataset_id, file_path, file_name,
                            captured_at, resolution, brightness, blur_score, glare_ratio,
                            dhash, is_accepted, rejection_reasons, status, split, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            image_id,
                            camera_id,
                            camera_name,
                            dataset_id,
                            str(image_path),
                            filename,
                            now.isoformat(),
                            metadata.get("resolution", "1920x1080"),
                            float(metadata.get("brightness", 0)),
                            float(metadata.get("blur_score", 0.0)),
                            float(quality_info.get("glare_ratio", 0.0)),
                            str(quality_info.get("dhash", "")),
                            1 if is_accepted else 0,
                            json.dumps(quality_info.get("rejection_reasons", [])),
                            "raw",
                            "unassigned",
                            now.isoformat(),
                            now.isoformat()
                        )
                    )
                    conn.commit()
        except Exception as e:
            logger.debug("DB sync for captured image: %s", e)

        return metadata

capture_manager = DatasetCaptureManager()
