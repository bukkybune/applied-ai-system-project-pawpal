"""One place to switch on file logging for the app, the demo and the evaluator.

Library code only calls logging.getLogger("pawpal_ai..."); nothing is written
to disk unless an entry point calls setup_logging(). Tests never do.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging(level: int = logging.INFO, log_dir: Path = LOG_DIR) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "pawpal_ai.log"
    logger = logging.getLogger("pawpal_ai")
    logger.setLevel(level)
    # Streamlit reruns the script on every click; add the handler only once.
    if not any(getattr(handler, "_pawpal", False) for handler in logger.handlers):
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        handler._pawpal = True
        logger.addHandler(handler)
    return path
