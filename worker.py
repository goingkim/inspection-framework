from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:
    from .config import STREAM_WAIT_TIMEOUT, YOLO_CONF, YOLO_IMGSZ
    from .camera import MjpegCamera, create_error_frame
    from .det_logger import logger
except ImportError:
    from config import STREAM_WAIT_TIMEOUT, YOLO_CONF, YOLO_IMGSZ
    from camera import MjpegCamera, create_error_frame
    from det_logger import logger


class DetectionWorker:
    """Runs YOLO inference on frames from one camera stream."""

    def __init__(
        self,
        model,
        camera: MjpegCamera,
        conf: float = YOLO_CONF,
        imgsz: int = YOLO_IMGSZ,
    ):
        self.model = model
        self.camera = camera
        self.conf = conf
        self.imgsz = imgsz
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="DetectionWorker", daemon=True)
        self._latest_frame: Optional[np.ndarray] = None
        self._last_error: Optional[str] = None
        self._last_inference_time: Optional[float] = None
        self._started_at = time.time()
        self.frame_count = 0
        self.detection_count = 0
        self.last_latency_ms = 0.0

    def start(self) -> None:
        self.camera.start()
        if not self._thread.is_alive():
            self._thread.start()
        logger.info("detection worker started")

    def stop(self) -> None:
        self._stop.set()
        self.camera.stop()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=3)
        logger.info("detection worker stopped")

    def get_frame_wait(self, timeout: float = STREAM_WAIT_TIMEOUT) -> Optional[np.ndarray]:
        with self._condition:
            if self._latest_frame is None and not self._stop.is_set():
                self._condition.wait(timeout=timeout)
            return None if self._latest_frame is None else self._latest_frame.copy()

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._condition:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def status(self) -> dict:
        elapsed = max(time.time() - self._started_at, 0.001)
        last_age = None if self._last_inference_time is None else round(time.time() - self._last_inference_time, 2)
        return {
            "running": self._thread.is_alive() and not self._stop.is_set(),
            "frames": self.frame_count,
            "detections": self.detection_count,
            "fps": round(self.frame_count / elapsed, 2),
            "last_latency_ms": round(self.last_latency_ms, 1),
            "last_inference_age": last_age,
            "last_error": self._last_error,
            "camera": self.camera.status(),
        }

    def _loop(self) -> None:
        last_seq = -1
        while not self._stop.is_set():
            seq, frame = self.camera.get_frame_wait(last_seq, timeout=1.0)
            if frame is None or seq == last_seq:
                continue
            last_seq = seq

            started = time.time()
            try:
                results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
                result = results[0]
                annotated = result.plot()
                detections = self._count_boxes(result)
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error("inference failed", exc)
                annotated = create_error_frame("Inference failed\nCheck logs and model setup.", frame.shape[1], frame.shape[0])
                detections = 0

            self.last_latency_ms = (time.time() - started) * 1000
            self.frame_count += 1
            self.detection_count += detections
            self._last_inference_time = time.time()

            with self._condition:
                self._latest_frame = annotated
                self._condition.notify_all()

    @staticmethod
    def _count_boxes(result) -> int:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return 0
        try:
            return len(boxes)
        except TypeError:
            return 0
