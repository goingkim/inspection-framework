from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from flask import Flask

try:
    from .camera import Cam2IPServer, MjpegCamera
    from .config import (
        BASE_DIR,
        CAMERA_DELAY_MS,
        CAMERA_HEIGHT,
        CAMERA_INDEX,
        CAMERA_WIDTH,
        CKPT_DIR,
        YOLO_CONF,
        YOLO_IMGSZ,
        YOLO_MODEL,
    )
    from .det_logger import logger
    from .worker import DetectionWorker
except ImportError:
    from camera import Cam2IPServer, MjpegCamera
    from config import (
        BASE_DIR,
        CAMERA_DELAY_MS,
        CAMERA_HEIGHT,
        CAMERA_INDEX,
        CAMERA_WIDTH,
        CKPT_DIR,
        YOLO_CONF,
        YOLO_IMGSZ,
        YOLO_MODEL,
    )
    from det_logger import logger
    from worker import DetectionWorker


def resolve_yolo_model(model_name: str | Path) -> str:
    model_path = Path(model_name)
    if not model_path.is_absolute():
        if model_path.parent == Path("."):
            ckpt_model = CKPT_DIR / model_path
            if ckpt_model.exists():
                return str(ckpt_model)
        project_model = BASE_DIR / model_path
        if project_model.exists():
            return str(project_model)
    return str(model_path)


class AppState:
    """Shared runtime state for the demo server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cam2ip = Cam2IPServer()
        self.worker: Optional[DetectionWorker] = None
        self.model = None
        self.model_name: Optional[str] = None

    def load_model(self, model_name: str = YOLO_MODEL):
        model_name = resolve_yolo_model(model_name)
        with self.lock:
            if self.model is not None and self.model_name == model_name:
                return self.model

        logger.info(f"loading YOLO model: {model_name}")
        from ultralytics import YOLO

        model = YOLO(model_name)

        with self.lock:
            self.model = model
            self.model_name = model_name
            return self.model

    def start(
        self,
        index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        delay_ms: int = CAMERA_DELAY_MS,
        conf: float = YOLO_CONF,
        imgsz: int = YOLO_IMGSZ,
        model_name: str = YOLO_MODEL,
    ) -> dict:
        self.cam2ip.start(index=index, width=width, height=height, delay_ms=delay_ms)
        model = self.load_model(model_name)
        camera = MjpegCamera(width=width, height=height)
        worker = DetectionWorker(model=model, camera=camera, conf=conf, imgsz=imgsz)

        with self.lock:
            old_worker = self.worker
            self.worker = worker

        if old_worker is not None:
            old_worker.stop()

        worker.start()
        return self.status()

    def stop(self, stop_cam2ip: bool = True) -> dict:
        with self.lock:
            worker = self.worker
            self.worker = None

        if worker is not None:
            worker.stop()
        if stop_cam2ip:
            self.cam2ip.stop()
        return self.status()

    def latest_frame(self):
        with self.lock:
            worker = self.worker
        return None if worker is None else worker.latest_frame()

    def status(self) -> dict:
        with self.lock:
            worker = self.worker
            model_name = self.model_name
            model_loaded = self.model is not None

        return {
            "model": model_name or YOLO_MODEL,
            "model_loaded": model_loaded,
            "cam2ip": self.cam2ip.status(),
            "worker": None if worker is None else worker.status(),
        }


state = AppState()


def create_app() -> Flask:
    app = Flask(__name__)
    try:
        from .api import api
    except ImportError:
        from api import api
    app.register_blueprint(api)
    return app
