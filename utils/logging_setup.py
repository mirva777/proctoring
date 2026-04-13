"""
Structured logging configuration.

Configures a root logger with a JSON-like structured format for easy ingestion
into log aggregation tools (ELK, CloudWatch, etc.) as well as a readable
human format for console output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_JSON_FORMAT = (
    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
)
_CONSOLE_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    json: bool = False,
) -> None:
    """
    Configure root logging.

    Args:
        level:    Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to a file.
        json:     If True, emit JSON lines to the log file.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    # Console handler (always human-readable)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        fmt = _JSON_FORMAT if json else _CONSOLE_FORMAT
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)
