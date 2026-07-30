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
