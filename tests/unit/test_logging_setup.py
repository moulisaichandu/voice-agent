"""Regression test for LOG_LEVEL: it was read in app/config.py but never
actually applied anywhere (app.main called setup_logging() with no argument),
so changing it in .env silently did nothing.
"""

import logging

from app.logging_setup import setup_logging


def test_setup_logging_applies_the_level_even_on_repeated_calls():
    """app.main calls setup_logging() twice (once before app.config is
    importable, once with the real LOG_LEVEL after) — the second call must
    not be a no-op just because a handler already exists."""
    root = logging.getLogger()
    original_level = root.level
    try:
        setup_logging("DEBUG")
        assert root.level == logging.DEBUG
        setup_logging("WARNING")
        assert root.level == logging.WARNING
    finally:
        root.setLevel(original_level)


def test_setup_logging_never_attaches_a_second_handler():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    try:
        setup_logging("INFO")
        setup_logging("DEBUG")
        assert len(root.handlers) == len(original_handlers)
    finally:
        root.setLevel(logging.INFO)
