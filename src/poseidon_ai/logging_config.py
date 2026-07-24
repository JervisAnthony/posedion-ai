"""Centralized logging configuration for Poseidon AI."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_BACKUP_COUNT = 3


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = Path("logs/poseidon.log"),
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """Configure console and rotating file logging.

    Args:
        level: Minimum logging level to record.
        log_file: Destination log file. Pass None to disable file logging.
        max_bytes: Maximum size of one log file before rotation.
        backup_count: Number of rotated log files to retain.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        handlers.append(
            RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )