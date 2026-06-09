from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

try:
    from .config import (
        CAM2IP_EXE,
        CAM2IP_HOST,
        CAM2IP_PORT,
    )
    from .det_logger import logger
except ImportError:
    from config import (
        CAM2IP_EXE,
        CAM2IP_HOST,
        CAM2IP_PORT,
    )
    from det_logger import logger


_CAM2IP_ADDR = f"{CAM2IP_HOST}:{CAM2IP_PORT}"
_CAM2IP_URL = f"http://{_CAM2IP_ADDR}"
_STARTUP_TIMEOUT = 10.0
_READY_INTERVAL = 0.25
_CHUNK_SIZE = 4096
_READ_TIMEOUT = (5, 10)
_RECONNECT_DELAY = 0.5
_MAX_BUFFER_BYTES = 2 * 1024 * 1024


class Cam2IPServer:
    """Starts and stops the bundled cam2ip process when needed."""

    def __init__(
        self,
        exe_path: Path = CAM2IP_EXE,
    ):
        self.exe_path = Path(exe_path)
        self.bind_addr = _CAM2IP_ADDR
        self.jpeg_url = f"{_CAM2IP_URL}/jpeg"
        self.mjpeg_url = f"{_CAM2IP_URL}/mjpeg"
        self._process: Optional[subprocess.Popen] = None
        self._external = False
        self._lock = threading.Lock()

    def build_command(self, index: int, width: int, height: int, delay_ms: int) -> list[str]:
        return [
            str(self.exe_path),
            "-bind-addr",
            self.bind_addr,
            "-index",
            str(index),
            "-width",
            str(width),
            "-height",
            str(height),
            "-delay",
            str(delay_ms),
        ]

    def start(self, index: int, width: int, height: int, delay_ms: int) -> None:
        with self._lock:
            if self.is_ready():
                self._external = self._process is None or self._process.poll() is not None
                logger.info(f"cam2ip already available at {self.jpeg_url}")
                return

            if not self.exe_path.exists():
                raise FileNotFoundError(f"cam2ip executable not found: {self.exe_path}")

            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._process = subprocess.Popen(
                self.build_command(index, width, height, delay_ms),
                cwd=str(self.exe_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._external = False
            logger.info(f"cam2ip started at {self.mjpeg_url}")

        deadline = time.time() + _STARTUP_TIMEOUT
        while time.time() < deadline:
            if self.is_ready():
                return
            if self._process and self._process.poll() is not None:
                raise RuntimeError("cam2ip exited before it became ready")
            time.sleep(_READY_INTERVAL)

        self.stop()
        raise TimeoutError(f"cam2ip did not become ready within {_STARTUP_TIMEOUT}s")

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            should_stop = process is not None and not self._external
            self._external = False

        if not should_stop or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        logger.info("cam2ip stopped")

    def is_ready(self) -> bool:
        try:
            response = requests.get(self.jpeg_url, timeout=(1, 1))
            return response.status_code == 200 and bool(response.content)
        except requests.RequestException:
            return False

    def status(self) -> dict:
        process_running = self._process is not None and self._process.poll() is None
        return {
            "ready": self.is_ready(),
            "managed_process": process_running,
            "external_process": self._external,
            "exe": str(self.exe_path),
            "mjpeg_url": self.mjpeg_url,
            "jpeg_url": self.jpeg_url,
        }


class MjpegCamera:
    """Reads JPEG frames from a cam2ip MJPEG endpoint on a background thread."""

    def __init__(self, url: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None):
        self.url = url or f"{_CAM2IP_URL}/mjpeg"
        self.width = width
        self.height = height
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="MjpegCamera", daemon=True)
        self._latest_frame: Optional[np.ndarray] = None
        self._seq = 0
        self._frames = 0
        self._last_error: Optional[str] = None
        self._last_frame_time: Optional[float] = None

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def get_frame_wait(self, last_seq: int, timeout: float) -> tuple[int, Optional[np.ndarray]]:
        with self._condition:
            if self._seq == last_seq and not self._stop.is_set():
                self._condition.wait(timeout=timeout)
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            return self._seq, frame

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._condition:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def status(self) -> dict:
        last_age = None if self._last_frame_time is None else round(time.time() - self._last_frame_time, 2)
        return {
            "url": self.url,
            "frames": self._frames,
            "sequence": self._seq,
            "last_frame_age": last_age,
            "last_error": self._last_error,
            "running": self._thread.is_alive() and not self._stop.is_set(),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._read_stream_once()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error("camera stream error", exc)
                time.sleep(_RECONNECT_DELAY)

    def _read_stream_once(self) -> None:
        with requests.get(self.url, stream=True, timeout=_READ_TIMEOUT) as response:
            response.raise_for_status()
            buffer = bytearray()
            for chunk in response.iter_content(_CHUNK_SIZE):
                if self._stop.is_set():
                    return
                if not chunk:
                    continue
                buffer.extend(chunk)
                if len(buffer) > _MAX_BUFFER_BYTES:
                    buffer.clear()
                    self._last_error = "MJPEG buffer overflow; buffer reset"
                    continue

                while True:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2) if start != -1 else -1
                    if start == -1 or end == -1:
                        break
                    jpg = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._store_frame(frame)

    def _store_frame(self, frame: np.ndarray) -> None:
        if self.width and self.height:
            current_h, current_w = frame.shape[:2]
            if (current_w, current_h) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height))

        with self._condition:
            self._latest_frame = frame
            self._seq += 1
            self._frames += 1
            self._last_frame_time = time.time()
            self._last_error = None
            self._condition.notify_all()


def create_error_frame(message: str, width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    accent_green = (0, 200, 80)
    cv2.rectangle(frame, (8, 8), (width - 8, height - 8), accent_green, 2)
    cv2.putText(frame, "DEMO BACKEND", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, accent_green, 2)
    y = 92
    for line in message.splitlines() or [message]:
        cv2.putText(frame, line[:70], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
        y += 28
        if y > height - 32:
            break
    return frame
