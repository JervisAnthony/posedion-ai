"""Centralized logging configuration for Poseidon AI."""

from __future__ import annotations

import logging
from pathlib import Path


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = Path("logs/poseidon.log"),
) -> None:
    """Configure console and optional file logging.

    Args:
        level: Minimum logging level to record.
        log_file: Destination log file. Pass None to disable file logging.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        handlers.append(
            logging.FileHandler(
                log_file,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )