import time
import threading
import queue
import logging
import cv2
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger("worker.rtsp_reader")

class RTSPFrameReader:
    """
    Threaded RTSP Stream Reader with a bounded latest-frame buffer (size 1).
    Guarantees zero-lag inference by always discarding stale buffered frames.
    Implements automatic reconnection with exponential backoff.
    """
    def __init__(self, rtsp_url: str, max_reconnect_attempts: int = 20, backoff_base: float = 1.5):
        self.rtsp_url = rtsp_url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.backoff_base = backoff_base

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.last_frame_time = 0.0
        self.fps = 0.0
        self.dropped_frames = 0
        self.reconnect_count = 0
        self.is_connected = False
        self.last_error: Optional[str] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="RTSPReaderThread")
        self._thread.start()
        logger.info("RTSP frame reader started for %s", self.rtsp_url.split("@")[-1])

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.is_connected = False
        logger.info("RTSP frame reader stopped.")

    def get_latest_frame(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """Returns (frame_bgr, frame_timestamp) or (None, 0.0) if timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None, 0.0

    def _capture_loop(self):
        retry_delay = 1.0
        while self._running:
            logger.info("Opening RTSP connection: %s", self.rtsp_url.split("@")[-1])
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.is_connected = False
                self.reconnect_count += 1
                self.last_error = f"Failed to open RTSP stream (attempt {self.reconnect_count})"
                logger.warning("%s. Retrying in %.1f seconds...", self.last_error, retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * self.backoff_base, 30.0)
                continue

            self.is_connected = True
            retry_delay = 1.0
            frame_count = 0
            start_time = time.time()

            logger.info("RTSP connection established successfully.")

            while self._running and cap.isOpened():
                ret, frame = cap.read()
                now = time.time()

                if not ret or frame is None:
                    logger.warning("RTSP read returned empty frame. Stream may have disconnected.")
                    break

                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = now - start_time
                    self.fps = round(frame_count / elapsed, 1) if elapsed > 0 else 0.0

                self.last_frame_time = now

                # Keep only latest frame in bounded buffer
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass

                try:
                    self._queue.put_nowait((frame, now))
                except queue.Full:
                    self.dropped_frames += 1

            cap.release()
            self.is_connected = False
            self.reconnect_count += 1
            if self._running:
                logger.warning("RTSP connection lost. Reconnecting in %.1f seconds...", retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * self.backoff_base, 30.0)
