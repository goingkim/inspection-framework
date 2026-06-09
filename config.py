from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
CKPT_DIR = Path(os.getenv("CKPT_DIR", BASE_DIR / "ckpt"))

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5000"))

CAM2IP_EXE = Path(os.getenv("CAM2IP_EXE", BASE_DIR / "cam2ip-1.4" / "cam2ip-1.4" / "cam2ip.exe"))
CAM2IP_HOST = "127.0.0.1"
CAM2IP_PORT = 56000

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_DELAY_MS = int(os.getenv("CAMERA_DELAY_MS", "30"))

DEFAULT_YOLO_MODEL = CKPT_DIR / "yolo11n.pt"
YOLO_MODEL = os.getenv("YOLO_MODEL", str(DEFAULT_YOLO_MODEL))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))

JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
STREAM_WAIT_TIMEOUT = float(os.getenv("STREAM_WAIT_TIMEOUT", "0.1"))

AUTO_START_DEMO = os.getenv("AUTO_START_DEMO", "0").lower() in {"1", "true", "yes", "on"}
