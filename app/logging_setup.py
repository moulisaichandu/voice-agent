"""logging_setup.py — Configure logging once, before any other app module imports."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Idempotent on the handler (never attaches a second one on re-import
    under pytest), but the level is applied every call — app.main calls this
    twice: once before app.config is imported (so anything config itself logs
    during import has somewhere to go), then again with the real LOG_LEVEL
    once config is loaded."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s %(name)-20s [%(threadName)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
