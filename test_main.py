from __future__ import annotations

import time
from pathlib import Path

import numpy as np


PUBLIC_FILES = [
    "api.py",
    "app.py",
    "camera.py",
    "config.py",
    "det_logger.py",
    "main.py",
    "worker.py",
    "README.md",
]


class TestPublicSanitization:
    def test_old_project_identifiers_are_removed(self):
        forbidden = [
            "".join(["hidis", "_", "lagging"]),
            "Share" + "_Rev",
            "AI" + "3-",
            "BN" + "7I",
            "SV" + "-20",
            "car" + "model",
            "V" + "IN=",
            "D:" + "/Detection",
            "Soft" + "wares",
        ]
        root = Path(__file__).resolve().parent
        for filename in PUBLIC_FILES:
            text = (root / filename).read_text(encoding="utf-8")
            for needle in forbidden:
                assert needle not in text, f"{needle!r} remains in {filename}"


class TestConfig:
    def test_public_defaults(self):
        import config

        assert config.CKPT_DIR == config.BASE_DIR / "ckpt"
        assert config.YOLO_MODEL == str(config.CKPT_DIR / "yolo11n.pt")
        assert config.CAMERA_WIDTH > 0
        assert config.CAMERA_HEIGHT > 0
        assert config.CAM2IP_EXE.name == "cam2ip.exe"
        assert config.CAM2IP_HOST == "127.0.0.1"
        assert config.CAM2IP_PORT == 56000

    def test_model_resolver_prefers_ckpt_for_bare_filenames(self, monkeypatch, tmp_path):
        import app

        model_file = tmp_path / "local.pt"
        model_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(app, "CKPT_DIR", tmp_path)

        assert app.resolve_yolo_model("local.pt") == str(model_file)
        assert app.resolve_yolo_model("missing.pt") == "missing.pt"

    def test_model_resolver_accepts_project_relative_paths(self, monkeypatch, tmp_path):
        import app

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_file = model_dir / "local.pt"
        model_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(app, "BASE_DIR", tmp_path)

        assert app.resolve_yolo_model("models/local.pt") == str(model_file)


class TestCamera:
    def test_cam2ip_command_contains_expected_args(self, tmp_path):
        from camera import Cam2IPServer

        exe = tmp_path / "cam2ip.exe"
        exe.write_text("", encoding="utf-8")
        server = Cam2IPServer(exe_path=exe)
        command = server.build_command(index=1, width=320, height=240, delay_ms=50)

        assert command[0] == str(exe)
        assert "-bind-addr" in command and "127.0.0.1:56000" in command
        assert "-index" in command and "1" in command
        assert "-width" in command and "320" in command
        assert "-height" in command and "240" in command
        assert "-delay" in command and "50" in command

    def test_error_frame_shape(self):
        from camera import create_error_frame

        frame = create_error_frame("test", width=320, height=240)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)

    def test_mjpeg_camera_store_frame_resizes_and_increments_sequence(self):
        from camera import MjpegCamera

        camera = MjpegCamera(width=320, height=240)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera._store_frame(frame)
        seq, stored = camera.get_frame_wait(last_seq=0, timeout=0.01)

        assert seq == 1
        assert stored is not None
        assert stored.shape == (240, 320, 3)
        assert camera.status()["frames"] == 1


class FakeResult:
    boxes = [object()]

    def plot(self):
        return np.full((120, 160, 3), 255, dtype=np.uint8)


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, frame, conf, imgsz, verbose):
        self.calls += 1
        return [FakeResult()]


class FakeCamera:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.seq = 0
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_frame_wait(self, last_seq, timeout):
        if self.seq == 0:
            self.seq = 1
            return self.seq, self.frame.copy()
        time.sleep(min(timeout, 0.01))
        return self.seq, None

    def status(self):
        return {"frames": self.seq, "running": self.started and not self.stopped}


class TestWorker:
    def test_detection_worker_processes_one_frame(self):
        from worker import DetectionWorker

        model = FakeModel()
        camera = FakeCamera()
        worker = DetectionWorker(model=model, camera=camera, conf=0.25, imgsz=320)
        worker.start()
        try:
            frame = worker.get_frame_wait(timeout=2.0)
            assert frame is not None
            assert model.calls >= 1
            assert worker.status()["frames"] >= 1
            assert worker.status()["detections"] >= 1
        finally:
            worker.stop()
        assert camera.stopped is True


class TestAPI:
    def test_basic_endpoints(self):
        from app import create_app

        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        assert client.get("/ping").data == b"pong"
        assert client.get("/health").status_code == 200
        assert client.get("/status").status_code == 200

    def test_start_uses_state_and_returns_json(self, monkeypatch):
        import api as api_module
        from app import create_app

        class DummyState:
            worker = None

            def start(self, **kwargs):
                self.kwargs = kwargs
                return {"model": kwargs["model_name"], "cam2ip": {"ready": True}, "worker": None}

            def stop(self):
                return {"cam2ip": {"ready": False}, "worker": None}

            def status(self):
                return {"model": "dummy.pt", "cam2ip": {"ready": True}, "worker": None}

            def latest_frame(self):
                return None

        dummy = DummyState()
        monkeypatch.setattr(api_module, "state", dummy)

        app = create_app()
        app.config["TESTING"] = True
        response = app.test_client().get("/start?index=2&width=320&height=240&conf=0.5&model=dummy.pt")
        data = response.get_json()

        assert response.status_code == 200
        assert data["started"] is True
        assert dummy.kwargs["index"] == 2
        assert dummy.kwargs["width"] == 320
        assert dummy.kwargs["height"] == 240
        assert dummy.kwargs["conf"] == 0.5
        assert dummy.kwargs["model_name"] == "dummy.pt"
