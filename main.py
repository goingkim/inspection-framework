from __future__ import annotations

import atexit

try:
    from .app import create_app, state
    from .config import AUTO_START_DEMO, BACKEND_HOST, BACKEND_PORT
    from .det_logger import logger
except ImportError:
    from app import create_app, state
    from config import AUTO_START_DEMO, BACKEND_HOST, BACKEND_PORT
    from det_logger import logger


app = create_app()


def _shutdown() -> None:
    try:
        state.stop(stop_cam2ip=True)
    except Exception:
        pass


atexit.register(_shutdown)


if __name__ == "__main__":
    if AUTO_START_DEMO:
        try:
            state.start()
        except Exception as exc:
            logger.error("auto start failed", exc)

    logger.info(f"server listening on http://127.0.0.1:{BACKEND_PORT}")
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False, threaded=True)
