"""Dataset statistics models for Nautilus Vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InvalidImageDiagnostic:
    """Describes why an image failed dataset validation."""

    image_path: Path
    errors: tuple[str, ...]


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

    min_pixel_count: int = 0
    max_pixel_count: int = 0
    average_pixel_count: float = 0.0

    extension_counts: dict[str, int] = field(default_factory=dict)
    invalid_image_diagnostics: list[InvalidImageDiagnostic] = field(
        default_factory=list
    )
    channel_counts: dict[int, int] = field(default_factory=dict)
