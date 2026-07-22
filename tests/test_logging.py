"""Tests for Poseidon AI logging configuration."""

import logging
from pathlib import Path

from poseidon_ai.logging_config import configure_logging


def test_configure_logging_creates_log_file(tmp_path: Path) -> None:
    """Logging configuration should create and write to a log file."""
    log_file = tmp_path / "poseidon.log"

    configure_logging(log_file=log_file)

    logger = logging.getLogger("poseidon_ai.test")
    logger.info("Test log message")

    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file.exists()
    assert "Test log message" in log_file.read_text(encoding="utf-8")


def test_configure_logging_without_file() -> None:
    """Logging configuration should support console-only logging."""
    configure_logging(log_file=None)

    root_logger = logging.getLogger()

    assert root_logger.level == logging.INFO
    assert any(
        isinstance(handler, logging.StreamHandler)
        for handler in root_logger.handlers
    )