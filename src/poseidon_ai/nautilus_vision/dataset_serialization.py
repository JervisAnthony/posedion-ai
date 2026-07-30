"""Shared serialization helpers for Nautilus Vision dataset reports."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_statistics import (
    DuplicateImageGroup,
    InvalidImageDiagnostic,
)


def serialize_channel_counts(
    channel_counts: Mapping[int, int],
) -> dict[str, int]:
    """Return channel counts with deterministic textual keys."""

    return {
        str(channels): count
        for channels, count in sorted(channel_counts.items())
    }


def serialize_resolution_statistics(
    min_pixel_count: int,
    max_pixel_count: int,
    average_pixel_count: float,
) -> dict[str, int | float]:
    """Return raw pixel counts and decimal megapixel values."""

    return {
        "minimum_pixels": min_pixel_count,
        "maximum_pixels": max_pixel_count,
        "average_pixels": average_pixel_count,
        "minimum_megapixels": round(
            min_pixel_count / 1_000_000,
            6,
        ),
        "maximum_megapixels": round(
            max_pixel_count / 1_000_000,
            6,
        ),
        "average_megapixels": round(
            average_pixel_count / 1_000_000,
            6,
        ),
    }


def serialize_duplicate_images(
    duplicate_groups: Sequence[DuplicateImageGroup],
) -> dict[str, object]:
    """Return exact duplicate groups in a deterministic structure."""

    sorted_groups: list[tuple[str, tuple[Path, ...]]] = []
    for group in duplicate_groups:
        image_paths = tuple(
            sorted(
                group.image_paths,
                key=lambda path: (
                    path.as_posix().casefold(),
                    path.as_posix(),
                ),
            )
        )
        sorted_groups.append(
            (group.sha256, image_paths)
        )

    sorted_groups.sort(
        key=lambda group: (
            group[1][0].as_posix().casefold(),
            group[1][0].as_posix(),
            group[0],
        )
    )
    file_count = sum(
        len(image_paths)
        for _, image_paths in sorted_groups
    )
    group_count = len(sorted_groups)

    return {
        "group_count": group_count,
        "file_count": file_count,
        "redundant_copy_count": file_count - group_count,
        "groups": [
            {
                "sha256": digest,
                "image_paths": [
                    path.as_posix()
                    for path in image_paths
                ],
            }
            for digest, image_paths in sorted_groups
        ],
    }


def serialize_invalid_image_diagnostics(
    diagnostics: Sequence[InvalidImageDiagnostic],
) -> list[dict[str, object]]:
    """Return diagnostics in a portable, deterministic structure."""

    return [
        {
            "image_path": diagnostic.image_path.as_posix(),
            "errors": list(diagnostic.errors),
        }
        for diagnostic in sorted(
            diagnostics,
            key=lambda diagnostic: (
                diagnostic.image_path.as_posix().casefold(),
                diagnostic.image_path.as_posix(),
            ),
        )
    ]
