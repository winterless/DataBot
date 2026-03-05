"""Structured logger utilities for unattended pipeline runs."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_LOG_FORMAT = "%(message)s"


class JsonFormatter(logging.Formatter):
    """Format log records as one-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", None)
        if trace_id:
            payload["trace_id"] = trace_id
        run_id = getattr(record, "run_id", None)
        if run_id:
            payload["run_id"] = run_id
        step_id = getattr(record, "step_id", None)
        if step_id:
            payload["step_id"] = step_id
        return json.dumps(payload, ensure_ascii=False)


def _resolve_level(default: str = "INFO") -> int:
    level = os.getenv("DATABOT_LOG_LEVEL", default).upper()
    return getattr(logging, level, logging.INFO)


def _build_stream_handler() -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter(_DEFAULT_LOG_FORMAT))
    return handler


def _build_file_handler(log_file: Path) -> logging.Handler:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(JsonFormatter(_DEFAULT_LOG_FORMAT))
    return handler


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a process-safe logger with JSON stdout/file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(_resolve_level())
    logger.propagate = False
    logger.addHandler(_build_stream_handler())

    target = log_file or os.getenv("DATABOT_LOG_FILE", "logs/databot.log")
    if target:
        logger.addHandler(_build_file_handler(Path(target)))
    return logger


def with_trace(logger: logging.Logger, *, trace_id: str = "", run_id: str = "", step_id: str = "") -> logging.LoggerAdapter:
    """Bind trace metadata to every log event."""
    extra = {"trace_id": trace_id, "run_id": run_id, "step_id": step_id}
    return logging.LoggerAdapter(logger, extra)
