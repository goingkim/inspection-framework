from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, request

try:
    from .app import state
    from .camera import create_error_frame
    from .config import (
        CAMERA_DELAY_MS,
        CAMERA_HEIGHT,
        CAMERA_INDEX,
        CAMERA_WIDTH,
        JPEG_QUALITY,
        STREAM_WAIT_TIMEOUT,
        YOLO_CONF,
        YOLO_IMGSZ,
        YOLO_MODEL,
    )
except ImportError:
    from app import state
    from camera import create_error_frame
    from config import (
        CAMERA_DELAY_MS,
        CAMERA_HEIGHT,
        CAMERA_INDEX,
        CAMERA_WIDTH,
        JPEG_QUALITY,
        STREAM_WAIT_TIMEOUT,
        YOLO_CONF,
        YOLO_IMGSZ,
        YOLO_MODEL,
    )

api = Blueprint("api", __name__)


def _as_int(name: str, default: int) -> int:
    return int(request.args.get(name, default))


def _as_float(name: str, default: float) -> float:
    return float(request.args.get(name, default))


def _encode_jpeg(frame: np.ndarray) -> Optional[bytes]:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def _jpeg_response(message: str, status_code: int = 200) -> Response:
    frame = create_error_frame(message, CAMERA_WIDTH, CAMERA_HEIGHT)
    payload = _encode_jpeg(frame) or b""
    return Response(payload, status=status_code, mimetype="image/jpeg")


@api.route("/")
def index():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>inspection-framework</title>
  <style>
    body { margin: 0; font-family: Georgia, serif; background: #f5efe2; color: #1f2723; }
    main { max-width: 960px; margin: 40px auto; padding: 0 24px; }
    h1 { font-size: 42px; margin-bottom: 8px; }
    p { line-height: 1.55; }
    .bar { display: flex; gap: 12px; margin: 20px 0; flex-wrap: wrap; }
    a { color: #1f2723; font-weight: 700; }
    button { padding: 10px 16px; border: 1px solid #1f2723; background: #d98745; cursor: pointer; }
    img { width: 100%; border: 6px solid #1f2723; background: #111; }
  </style>
</head>
<body>
  <main>
    <h1>inspection-framework</h1>
    <p>Start the demo, then watch the annotated MJPEG stream below.</p>
    <div class="bar">
      <button onclick="fetch('/start').then(() => location.reload())">Start</button>
      <button onclick="fetch('/stop').then(() => location.reload())">Stop</button>
      <a href="/status">Status JSON</a>
    </div>
    <img src="/stream" alt="YOLO stream">
  </main>
</body>
</html>
"""


@api.route("/ping")
def ping():
    return "pong", 200


@api.route("/health")
def health():
    status = state.status()
    return jsonify({"status": "ok", **status})


@api.route("/status")
def status():
    return jsonify(state.status())


@api.route("/start", methods=["GET", "POST"])
def start():
    try:
        result = state.start(
            index=_as_int("index", CAMERA_INDEX),
            width=_as_int("width", CAMERA_WIDTH),
            height=_as_int("height", CAMERA_HEIGHT),
            delay_ms=_as_int("delay", CAMERA_DELAY_MS),
            conf=_as_float("conf", YOLO_CONF),
            imgsz=_as_int("imgsz", YOLO_IMGSZ),
            model_name=request.args.get("model", YOLO_MODEL),
        )
        return jsonify({"started": True, **result})
    except Exception as exc:
        return jsonify({"started": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@api.route("/stop", methods=["GET", "POST"])
def stop():
    return jsonify({"stopped": True, **state.stop()})


@api.route("/snapshot")
def snapshot():
    frame = state.latest_frame()
    if frame is None:
        return _jpeg_response("No frame yet.\nCall /start first.", 404)
    payload = _encode_jpeg(frame)
    if payload is None:
        return _jpeg_response("JPEG encoding failed.", 500)
    return Response(payload, mimetype="image/jpeg")


@api.route("/stream")
def stream():
    worker = state.worker
    if worker is None:
        return _jpeg_response("No active demo.\nOpen /start first.", 404)

    def generate():
        last_error_frame = 0.0
        while worker.status()["running"]:
            frame = worker.get_frame_wait(timeout=STREAM_WAIT_TIMEOUT)
            if frame is None:
                now = time.time()
                if now - last_error_frame < 1.0:
                    continue
                frame = create_error_frame("Waiting for webcam frames...", CAMERA_WIDTH, CAMERA_HEIGHT)
                last_error_frame = now
            payload = _encode_jpeg(frame)
            if payload is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
