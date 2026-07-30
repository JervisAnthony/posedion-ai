"""Dataset manifest models and JSON Lines serialization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_statistics import DatasetStatistics


@dataclass(frozen=True, slots=True)
class DatasetManifestEntry:
    """Immutable inventory information for one supported image candidate."""

    path: Path
    extension: str
    is_valid: bool
    validation_errors: tuple[str, ...]
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    size_bytes: int | None = None
    pixel_count: int | None = None
    megapixels: float | None = None
    duplicate_group_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetAnalysisResult:
    """Aggregate statistics and optional per-candidate manifest entries."""

    statistics: DatasetStatistics
    manifest_entries: tuple[DatasetManifestEntry, ...]


def format_dataset_manifest_jsonl(
    entries: Sequence[DatasetManifestEntry],
) -> str:
    """Serialize manifest entries as deterministic JSON Lines."""

    lines: list[str] = []

    for entry in entries:
        payload = {
            "path": entry.path.as_posix(),
            "extension": entry.extension,
            "is_valid": entry.is_valid,
            "validation_errors": list(entry.validation_errors),
            "width": entry.width,
            "height": entry.height,
            "channels": entry.channels,
            "size_bytes": entry.size_bytes,
            "pixel_count": entry.pixel_count,
            "megapixels": entry.megapixels,
            "duplicate_group_sha256": entry.duplicate_group_sha256,
        }
        lines.append(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    if not lines:
        return ""

    return "\n".join(lines) + "\n"
