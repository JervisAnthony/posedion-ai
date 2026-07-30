from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
    DuplicateImageGroup,
    ImageFormatStatistics,
    InvalidImageDiagnostic,
)


def test_dataset_statistics_defaults() -> None:
    """DatasetStatistics should initialize with default values."""

    stats = DatasetStatistics(
        dataset_path=Path("dataset"),
    )

    assert stats.total_images == 0
    assert stats.valid_images == 0
    assert stats.invalid_images == 0
    assert stats.total_size_bytes == 0

    assert stats.average_width == 0.0
    assert stats.average_height == 0.0
    assert stats.min_pixel_count == 0
    assert stats.max_pixel_count == 0
    assert stats.average_pixel_count == 0.0
    assert stats.min_aspect_ratio == 0.0
    assert stats.max_aspect_ratio == 0.0
    assert stats.average_aspect_ratio == 0.0
    assert stats.min_file_size_bytes == 0
    assert stats.max_file_size_bytes == 0
    assert stats.average_file_size_bytes == 0.0
    assert stats.extension_counts == {}
    assert stats.format_statistics == {}
    assert stats.channel_counts == {}
    assert stats.orientation_counts == {}
    assert stats.duplicate_image_groups == []
    assert stats.duplicate_group_count == 0
    assert stats.duplicate_file_count == 0
    assert stats.redundant_copy_count == 0
    assert stats.invalid_image_diagnostics == []


def test_extension_counts_default_is_not_shared() -> None:
    """Each DatasetStatistics instance should own its extension counts."""

    first = DatasetStatistics(dataset_path=Path("first"))
    second = DatasetStatistics(dataset_path=Path("second"))

    first.extension_counts["jpeg"] = 1

    assert second.extension_counts == {}


def test_format_statistics_default_is_not_shared() -> None:
    """Each DatasetStatistics instance should own its format statistics."""

    first = DatasetStatistics(dataset_path=Path("first"))
    second = DatasetStatistics(dataset_path=Path("second"))

    first.format_statistics["jpeg"] = ImageFormatStatistics(
        total_images=1,
        valid_images=1,
        invalid_images=0,
        total_valid_size_bytes=1_024,
        average_valid_size_bytes=1_024.0,
    )

    assert second.format_statistics == {}


def test_image_format_statistics_is_immutable() -> None:
    """Keep completed per-format aggregates immutable."""

    statistics = ImageFormatStatistics(
        total_images=1,
        valid_images=1,
        invalid_images=0,
        total_valid_size_bytes=1_024,
        average_valid_size_bytes=1_024.0,
    )

    with pytest.raises(FrozenInstanceError):
        statistics.valid_images = 2


def test_channel_counts_default_is_not_shared() -> None:
    """Each DatasetStatistics instance should own its channel counts."""

    first = DatasetStatistics(dataset_path=Path("first"))
    second = DatasetStatistics(dataset_path=Path("second"))

    first.channel_counts[3] = 1

    assert second.channel_counts == {}


def test_orientation_counts_default_is_not_shared() -> None:
    """Each DatasetStatistics instance should own orientation counts."""

    first = DatasetStatistics(dataset_path=Path("first"))
    second = DatasetStatistics(dataset_path=Path("second"))

    first.orientation_counts["landscape"] = 1

    assert second.orientation_counts == {}


def test_duplicate_image_groups_default_is_not_shared() -> None:
    """Each DatasetStatistics instance should own its duplicate groups."""

    first = DatasetStatistics(dataset_path=Path("first"))
    second = DatasetStatistics(dataset_path=Path("second"))

    first.duplicate_image_groups.append(
        DuplicateImageGroup(
            sha256="a" * 64,
            image_paths=(
                Path("first/a.jpg"),
                Path("first/b.jpg"),
            ),
        )
    )

    assert second.duplicate_image_groups == []


def test_duplicate_counts_are_derived_from_groups() -> None:
    """Derive group, file, and redundant-copy counts."""

    stats = DatasetStatistics(
        dataset_path=Path("dataset"),
        duplicate_image_groups=[
            DuplicateImageGroup(
                sha256="a" * 64,
                image_paths=(
                    Path("dataset/a.jpg"),
                    Path("dataset/b.jpg"),
                ),
            ),
            DuplicateImageGroup(
                sha256="b" * 64,
                image_paths=(
                    Path("dataset/c.png"),
                    Path("dataset/d.png"),
                    Path("dataset/e.png"),
                ),
            ),
        ],
    )

    assert stats.duplicate_group_count == 2
    assert stats.duplicate_file_count == 5
    assert stats.redundant_copy_count == 3
    assert (
        stats.redundant_copy_count
        == stats.duplicate_file_count - stats.duplicate_group_count
    )


def test_invalid_image_diagnostics_default_is_not_shared() -> None:
    """Each DatasetStatistics instance should own its diagnostics list."""

    first = DatasetStatistics(
        dataset_path=Path("first"),
    )
    second = DatasetStatistics(
        dataset_path=Path("second"),
    )

    first.invalid_image_diagnostics.append(
        InvalidImageDiagnostic(
            image_path=Path("first/corrupt.jpg"),
            errors=("Image could not be decoded.",),
        )
    )

    assert len(first.invalid_image_diagnostics) == 1
    assert second.invalid_image_diagnostics == []
