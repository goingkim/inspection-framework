from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from .config import LOG_DIR
except ImportError:
    from config import LOG_DIR


class DetLogger:
    """Small file logger for the educational demo."""

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def info(self, message: str) -> None:
        self._write("app", "INFO", message)
        print(f"[INFO] {message}", flush=True)

    def debug(self, message: str) -> None:
        self._write("app", "DEBUG", message)

    def error(self, message: str, exc: Optional[BaseException] = None) -> None:
        if exc is not None:
            message = f"{message} ({type(exc).__name__}: {exc})"
        self._write("error", "ERROR", message)
        print(f"[ERROR] {message}", file=sys.stderr, flush=True)

    def _write(self, stem: str, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"{stem}_{datetime.now().strftime('%Y%m%d')}.log"
        line = f"[{timestamp}] {level}: {message}"
        try:
            with self._lock:
                with (self.log_dir / filename).open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
        except OSError:
            pass


logger = DetLogger()
