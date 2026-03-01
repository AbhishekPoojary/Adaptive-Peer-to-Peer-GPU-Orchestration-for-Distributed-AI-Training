"""
utils/logger.py
----------------
Structured logging setup.

LOG_FORMAT=text  → human-readable coloured output (development default)
LOG_FORMAT=json  → JSON lines output (production / log aggregators)

Call `setup_logging()` once at application startup.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        # Propagate any extra fields set via `extra={}` in log calls
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                if not key.startswith("_"):
                    log_obj[key] = val
        return json.dumps(log_obj, default=str)


class _TextFormatter(logging.Formatter):
    _FMT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """
    Configure the root logger.

    Args:
        level: Log level string (DEBUG / INFO / WARNING / ERROR).
        fmt:   "text" for human-readable output, "json" for structured JSON lines.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter: logging.Formatter = (
        _JsonFormatter() if fmt.lower() == "json" else _TextFormatter()
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured: level=%s format=%s", level.upper(), fmt
    )
