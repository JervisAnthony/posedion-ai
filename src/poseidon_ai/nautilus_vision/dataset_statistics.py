"""Dataset statistics models for Nautilus Vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DuplicateImageGroup:
    """Represents valid images with identical file bytes."""

    sha256: str
    image_paths: tuple[Path, ...]


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

    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 0.0
    average_aspect_ratio: float = 0.0

    min_file_size_bytes: int = 0
    max_file_size_bytes: int = 0
    average_file_size_bytes: float = 0.0

    extension_counts: dict[str, int] = field(default_factory=dict)
    invalid_image_diagnostics: list[InvalidImageDiagnostic] = field(
        default_factory=list
    )
    channel_counts: dict[int, int] = field(default_factory=dict)
    orientation_counts: dict[str, int] = field(default_factory=dict)
    duplicate_image_groups: list[DuplicateImageGroup] = field(
        default_factory=list
    )

    @property
    def duplicate_group_count(self) -> int:
        """Return the number of exact duplicate groups."""

        return len(self.duplicate_image_groups)

    @property
    def duplicate_file_count(self) -> int:
        """Return the number of files in exact duplicate groups."""

        return sum(
            len(group.image_paths)
            for group in self.duplicate_image_groups
        )

    @property
    def redundant_copy_count(self) -> int:
        """Return removable copies while retaining one file per group."""

        return sum(
            len(group.image_paths) - 1
            for group in self.duplicate_image_groups
        )
