import os
import sys
import time
import logging
from pathlib import Path

# Ensure paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
backend_dir = project_dir / "backend"
for p in [str(project_dir), str(backend_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.config import settings
from worker.camera.rtsp_reader import RTSPFrameReader
from worker.pipelines.pipeline_runner import PipelineRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("worker.main")

def run_worker():
    logger.info("==================================================================")
    logger.info("  Starting Modular Edge-AI Vision Worker Process")
    logger.info("==================================================================")

    model_path = Path(settings.MODELS_DIR) / "official" / "yolo11n-coco" / "model.pt"
    if not model_path.exists():
        logger.error("Official YOLO11 model not found at %s! Please run bootstrap first.", model_path)
        sys.exit(1)

    device = "cuda:0"
    logger.info("Initializing Pipeline: %s on %s...", settings.ACTIVE_PIPELINE, device)
    pipeline = PipelineRunner(
        preset_id=settings.ACTIVE_PIPELINE,
        yolo_model_path=str(model_path),
        device=device
    )

    logger.info("Connecting to RTSP Stream: %s...", settings.DEFAULT_CAMERA_RTSP_STREAM.split("@")[-1])
    reader = RTSPFrameReader(rtsp_url=settings.DEFAULT_CAMERA_RTSP_STREAM)
    reader.start()

    logger.info("AI Worker active and processing live stream frames...")
    processed_count = 0
    start_time = time.time()

    try:
        while True:
            frame, frame_time = reader.get_latest_frame(timeout=0.5)
            if frame is None:
                continue

            results = pipeline.process_frame(frame)
            processed_count += 1

            if processed_count % 50 == 0:
                fps = processed_count / (time.time() - start_time)
                logger.info(
                    "Worker Loop - Processed: %d frames | Speed: %.1f FPS | Latency: %.1f ms | Tracked: %d objects",
                    processed_count, fps, results["latency"]["total_ms"], len(results["tracked_objects"])
                )

            # Throttle slightly if needed to avoid maxing out GPU unnecessarily
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Shutting down AI Worker...")
    finally:
        reader.stop()
        logger.info("AI Worker shutdown completed.")

if __name__ == "__main__":
    run_worker()
