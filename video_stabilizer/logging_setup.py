# ログ

from __future__ import annotations

import logging
import os


def ensure_stdio_streams() -> None:
    import sys

    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def configure_logging() -> None:
    ensure_stdio_streams()
    level_name = os.environ.get("VS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
