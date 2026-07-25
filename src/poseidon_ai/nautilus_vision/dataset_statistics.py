"""Dataset statistics models for Nautilus Vision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetStatistics:
    """Represents summary statistics for an image dataset."""

    dataset_path: Path

    total_images: int = 0
    valid_images: int = 0
    invalid_images: int = 0

    total_size_bytes: int = 0

    min_width: int = 0
    max_width: int = 0

    min_height: int = 0
    max_height: int = 0

    average_width: float = 0.0
    average_height: float = 0.0