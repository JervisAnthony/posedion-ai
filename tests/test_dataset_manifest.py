"""Tests for dataset manifest models and JSONL serialization."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from poseidon_ai.nautilus_vision.dataset_manifest import (
    DatasetManifestEntry,
    format_dataset_manifest_jsonl,
)


def test_manifest_entry_preserves_immutable_values() -> None:
    """Keep paths and validation errors in immutable model fields."""

    entry = DatasetManifestEntry(
        path=Path("nested") / "corrupt.png",
        extension="png",
        is_valid=False,
        validation_errors=(
            "Width is invalid.",
            "Height is invalid.",
        ),
    )

    assert entry.path == Path("nested") / "corrupt.png"
    assert entry.validation_errors == (
        "Width is invalid.",
        "Height is invalid.",
    )

    with pytest.raises(FrozenInstanceError):
        entry.validation_errors = ()


def test_format_valid_manifest_entry_uses_exact_schema() -> None:
    """Serialize every valid-image field with stable key order and types."""

    digest = "a" * 64
    entry = DatasetManifestEntry(
        path=Path("nested") / "coral.jpg",
        extension="jpeg",
        is_valid=True,
        validation_errors=(),
        width=640,
        height=480,
        channels=3,
        size_bytes=24_576,
        pixel_count=307_200,
        megapixels=0.3072,
        duplicate_group_sha256=digest,
    )

    result = format_dataset_manifest_jsonl([entry])
    record = json.loads(result)

    assert list(record) == [
        "path",
        "extension",
        "is_valid",
        "validation_errors",
        "width",
        "height",
        "channels",
        "size_bytes",
        "pixel_count",
        "megapixels",
        "duplicate_group_sha256",
    ]
    assert "aspect_ratio" not in record
    assert "orientation" not in record
    assert "orientation_category" not in record
    assert "min_file_size_bytes" not in record
    assert "max_file_size_bytes" not in record
    assert "average_file_size_bytes" not in record
    assert "file_size_statistics" not in record
    assert "formatted_file_size" not in record
    assert "format_statistics" not in record
    assert "format_total_images" not in record
    assert "format_valid_images" not in record
    assert "format_invalid_images" not in record
    assert "format_total_valid_size_bytes" not in record
    assert "format_average_valid_size_bytes" not in record
    assert record == {
        "path": "nested/coral.jpg",
        "extension": "jpeg",
        "is_valid": True,
        "validation_errors": [],
        "width": 640,
        "height": 480,
        "channels": 3,
        "size_bytes": 24_576,
        "pixel_count": 307_200,
        "megapixels": 0.3072,
        "duplicate_group_sha256": digest,
    }
    assert isinstance(record["megapixels"], float)
    assert result.endswith("\n")
    assert not result.startswith("[")


def test_format_invalid_manifest_entry_uses_null_metadata() -> None:
    """Keep invalid metadata null and serialize portable Unicode paths."""

    entry = DatasetManifestEntry(
        path=Path("récif") / "corrupt.png",
        extension="png",
        is_valid=False,
        validation_errors=("Image could not be decoded.",),
    )

    result = format_dataset_manifest_jsonl([entry])
    record = json.loads(result)

    assert record["path"] == "récif/corrupt.png"
    assert "\\u00e9" not in result
    assert record["validation_errors"] == [
        "Image could not be decoded."
    ]
    assert all(
        record[key] is None
        for key in (
            "width",
            "height",
            "channels",
            "size_bytes",
            "pixel_count",
            "megapixels",
            "duplicate_group_sha256",
        )
    )


def test_format_multiple_manifest_entries_as_independent_lines() -> None:
    """Emit one independently parseable compact object per line."""

    entries = [
        DatasetManifestEntry(
            path=Path("first.jpg"),
            extension="jpeg",
            is_valid=True,
            validation_errors=(),
        ),
        DatasetManifestEntry(
            path=Path("second.png"),
            extension="png",
            is_valid=False,
            validation_errors=("Invalid.",),
        ),
    ]

    result = format_dataset_manifest_jsonl(entries)
    lines = result.splitlines()

    assert len(lines) == 2
    assert [json.loads(line)["path"] for line in lines] == [
        "first.jpg",
        "second.png",
    ]
    assert all("\n" not in line for line in lines)
    assert result.count("\n") == 2


def test_format_empty_manifest_is_empty_string() -> None:
    """Represent an empty candidate set as a zero-length manifest."""

    assert format_dataset_manifest_jsonl(()) == ""
