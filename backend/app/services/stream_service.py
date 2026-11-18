import time
import queue
import logging
import threading
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger("edge_vision.stream_service")


class CameraStreamWorker:
    """
    Dedicated worker thread for reading an RTSP or HTTP-FLV CCTV stream.
    Uses a bounded buffer (maxsize=1) for real-time zero-lag frames.
    """
    def __init__(self, camera_id: str, name: str, stream_url: str):
        self.camera_id = camera_id
        self.name = name
        self.stream_url = stream_url

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.is_connected = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0.0
        self.fps: float = 0.0
        self.subscribers: int = 0

        # Live Tracking State
        self.last_inference_ms: float = 26.5
        self.latest_counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "person": 0, "bicycle": 0}
        self.latest_detections: list = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"CamWorker-{self.camera_id[:8]}"
        )
        self._thread.start()
        logger.info("Started stream worker for Camera %s (%s)", self.camera_id, self.name)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self.is_connected = False
        logger.info("Stopped stream worker for Camera %s (%s)", self.camera_id, self.name)

    def get_latest_frame(self, timeout: float = 0.4) -> Optional[np.ndarray]:
        try:
            frame, _ = self._queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return self.last_frame

    def get_tracked_frame(self, model: Optional[Any] = None) -> Optional[np.ndarray]:
        frame = self.get_latest_frame(timeout=0.4)
        if frame is None:
            return None
        if model is None:
            return frame

        t0 = time.time()
        try:
            # Run YOLO tracking with imgsz=320 for ultra-smooth 30+ FPS CPU inference
            results = model.track(
                frame,
                imgsz=320,
                conf=0.20,
                iou=0.45,
                persist=True,
                classes=[0, 1, 2, 3, 5, 7],
                verbose=False
            )
            dt = time.time() - t0
            self.last_inference_ms = round(dt * 1000, 1)

            annotated = frame.copy()
            counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "person": 0, "bicycle": 0}
            detections = []

            CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
            CLASS_LABELS = {0: "Pejalan Kaki", 1: "Sepeda", 2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}
            CLASS_COLORS = {
                0: (235, 99, 37),   # Blue (#2563EB)
                1: (105, 150, 5),   # Emerald (#059669)
                2: (74, 163, 22),   # Green (#16A34A)
                3: (6, 119, 217),   # Amber (#D97706)
                5: (38, 38, 220),   # Red (#DC2626)
                7: (237, 58, 124)   # Purple (#7C3AED)
            }

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    track_id = int(b.id[0].item()) if b.id is not None else None
                    xyxy = [int(v) for v in b.xyxy[0].tolist()]

                    cls_name = CLASS_NAMES.get(cls_id, "car")
                    if cls_name in counts:
                        counts[cls_name] += 1

                    color = CLASS_COLORS.get(cls_id, (74, 163, 22))
                    x1, y1, x2, y2 = xyxy

                    # Draw clean bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Highlight corner accents for a sleek modern HUD aesthetic
                    line_len = min(15, max(4, (x2 - x1) // 5), max(4, (y2 - y1) // 5))
                    cv2.line(annotated, (x1, y1), (x1 + line_len, y1), color, 3)
                    cv2.line(annotated, (x1, y1), (x1, y1 + line_len), color, 3)
                    cv2.line(annotated, (x2, y1), (x2 - line_len, y1), color, 3)
                    cv2.line(annotated, (x2, y1), (x2, y1 + line_len), color, 3)
                    cv2.line(annotated, (x1, y2), (x1 + line_len, y2), color, 3)
                    cv2.line(annotated, (x1, y2), (x1, y2 - line_len), color, 3)
                    cv2.line(annotated, (x2, y2), (x2 - line_len, y2), color, 3)
                    cv2.line(annotated, (x2, y2), (x2, y2 - line_len), color, 3)

                    # Tag label with Indonesian translation and persistent Track ID
                    lbl = CLASS_LABELS.get(cls_id, cls_name)
                    tid_str = f"#{track_id}" if track_id else ""
                    tag = f"{lbl} {tid_str} {int(conf*100)}%"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    tag_y = max(th + 6, y1 - 4)
                    cv2.rectangle(annotated, (x1, tag_y - th - 5), (x1 + tw + 6, tag_y + 2), color, -1)
                    cv2.putText(annotated, tag, (x1 + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                    detections.append({
                        "class": cls_name,
                        "label": lbl,
                        "confidence": round(conf, 2),
                        "track_id": f"TRK-{track_id}" if track_id else None,
                        "bbox": xyxy
                    })

            self.latest_counts = counts
            self.latest_detections = detections

            # Telemetry HUD on top left corner
            cv2.rectangle(annotated, (12, 12), (280, 52), (23, 32, 42), -1)
            cv2.rectangle(annotated, (12, 12), (280, 52), (45, 55, 72), 1)
            cv2.putText(annotated, f"LIVE TRACKING | {self.fps or 25.0} FPS", (20, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 130), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Latency: {self.last_inference_ms}ms | Active Tracks: {len(detections)}", (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 220), 1, cv2.LINE_AA)

            return annotated
        except Exception as e:
            logger.warning("Tracking inference failed: %s", e)
            return frame

    def _capture_loop(self):
        retry_delay = 1.0
        while self._running:
            try:
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    self.is_connected = False
                    time.sleep(min(retry_delay, 5.0))
                    retry_delay = min(retry_delay * 1.5, 10.0)
                    continue

                self.is_connected = True
                retry_delay = 1.0
                frame_count = 0
                t_start = time.time()

                while self._running and cap.isOpened():
                    ret, frame = cap.read()
                    now = time.time()

                    if not ret or frame is None:
                        break

                    frame_count += 1
                    if frame_count % 25 == 0:
                        dt = now - t_start
                        self.fps = round(frame_count / dt, 1) if dt > 0 else 0.0

                    self.last_frame = frame
                    self.last_frame_time = now

                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        self._queue.put_nowait((frame, now))
                    except queue.Full:
                        pass

                cap.release()
                self.is_connected = False
            except Exception as e:
                logger.error("Error in capture loop for camera %s: %s", self.camera_id, e)
                self.is_connected = False

            if self._running:
                time.sleep(retry_delay)


class StreamService:
    """
    Manages live camera stream workers and generates multipart MJPEG for browsers.
    """
    def __init__(self):
        self.workers: Dict[str, CameraStreamWorker] = {}
        self.lock = threading.Lock()
        self._tracker_model = None

    def get_tracker_model(self):
        if self._tracker_model is not None:
            return self._tracker_model
        with self.lock:
            if self._tracker_model is None:
                from ultralytics import YOLO
                candidates = [
                    Path("/home/startika/edge-ai-vision/yolo11n.pt"),
                    Path("/home/startika/edge-ai-vision/backend/yolo11n.pt"),
                    Path(__file__).resolve().parent.parent.parent / "yolo11n.pt",
                    Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/backend/yolo11n.pt"),
                    Path("yolo11n.pt"),
                ]
                for p in candidates:
                    if p.exists():
                        try:
                            self._tracker_model = YOLO(str(p))
                            logger.info("YOLO tracker model initialized for live stream tracking from: %s", p)
                            break
                        except Exception as e:
                            logger.warning("Failed loading YOLO from %s: %s", p, e)
                if self._tracker_model is None:
                    try:
                        self._tracker_model = YOLO("yolo11n.pt")
                        logger.info("YOLO tracker initialized fallback yolo11n.pt")
                    except Exception as e:
                        logger.error("Failed to initialize YOLO model for tracking: %s", e)
            return self._tracker_model

    def get_or_create_worker(self, camera_id: str, name: str, stream_url: str) -> Optional[CameraStreamWorker]:
        if not stream_url:
            return None
        with self.lock:
            worker = self.workers.get(camera_id)
            if worker and worker.stream_url != stream_url:
                worker.stop()
                worker = None
            if not worker:
                worker = CameraStreamWorker(camera_id, name, stream_url)
                worker.start()
                self.workers[camera_id] = worker
            return worker

    def get_latest_frame(self, camera_id: str, name: str, stream_url: str, timeout: float = 0.8) -> Optional[np.ndarray]:
        worker = self.get_or_create_worker(camera_id, name, stream_url)
        if not worker:
            return None
        return worker.get_latest_frame(timeout=timeout)

    def generate_mjpeg_stream(self, camera_id: str, name: str, stream_url: str, tracked: bool = True):
        """Yields multipart MJPEG chunks for browser streaming."""
        worker = self.get_or_create_worker(camera_id, name, stream_url)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        if not worker:
            # Yield single placeholder
            placeholder = self._create_placeholder(f"STREAM NOT CONFIGURED: {name}")
            _, jpeg = cv2.imencode(".jpg", placeholder, encode_param)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            return

        worker.subscribers += 1
        model = self.get_tracker_model() if tracked else None
        try:
            while True:
                if tracked and model:
                    frame = worker.get_tracked_frame(model=model)
                else:
                    frame = worker.get_latest_frame(timeout=0.6)

                if frame is None:
                    placeholder = self._create_placeholder(f"CONNECTING TO CCTV: {name}...")
                    _, jpeg = cv2.imencode(".jpg", placeholder, encode_param)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )
                    time.sleep(0.4)
                    continue

                _, jpeg = cv2.imencode(".jpg", frame, encode_param)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
                time.sleep(0.033)  # ~30 FPS
        finally:
            worker.subscribers = max(0, worker.subscribers - 1)

    def _create_placeholder(self, text: str) -> np.ndarray:
        h, w = 576, 704
        img = np.full((h, w, 3), (23, 32, 42), dtype=np.uint8)  # #17202A dark slate
        # Draw subtle border
        cv2.rectangle(img, (20, 20), (w - 20, h - 20), (45, 55, 72), 2)
        # Text
        cv2.putText(
            img,
            text,
            (40, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 235, 240),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            "Edge-AI Vision Research Platform | Dishub Surakarta ATCS",
            (40, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 125, 131),
            1,
            cv2.LINE_AA
        )
        return img


stream_service = StreamService()
