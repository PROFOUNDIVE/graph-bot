from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE = _LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log"
_LOG_FORMAT = "[%(levelname)s: %(module)s > %(funcName)s] %(message)s"

logger = logging.getLogger("graph_bot")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(stream_handler)
