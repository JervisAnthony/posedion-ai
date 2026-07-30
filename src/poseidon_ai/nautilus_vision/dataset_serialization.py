"""Shared serialization helpers for Nautilus Vision dataset reports."""

from collections.abc import Mapping, Sequence

from poseidon_ai.nautilus_vision.dataset_statistics import (
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
