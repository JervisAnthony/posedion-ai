from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_markdown import (
    format_dataset_markdown,
)
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
)


def create_statistics() -> DatasetStatistics:
    """Create representative dataset statistics."""

    return DatasetStatistics(
        dataset_path=Path("data/sample_dataset"),
        total_images=4,
        valid_images=3,
        invalid_images=1,
        min_width=640,
        max_width=1280,
        average_width=906.67,
        min_height=480,
        max_height=720,
        average_height=600.0,
        total_size_bytes=2048,
        extension_counts={
            "webp": 1,
            "jpeg": 2,
            "png": 1,
        },
    )


def test_format_dataset_markdown() -> None:
    """Format dataset statistics as Markdown."""

    stats = create_statistics()

    result = format_dataset_markdown(
        Path("data/sample_dataset"),
        stats,
    )

    assert "# Dataset Summary" in result
    assert "## Overview" in result
    assert "## Image Formats" in result
    assert "## Width" in result
    assert "## Height" in result
    assert "## Dataset Size" in result

    assert "| Dataset Path | data/sample_dataset |" in result
    assert "| Total Images | 4 |" in result
    assert "| Valid Images | 3 |" in result
    assert "| Invalid Images | 1 |" in result
    assert "| JPEG | 2 |" in result
    assert "| PNG | 1 |" in result
    assert "| WEBP | 1 |" in result
    assert result.index("| JPEG |") < result.index("| PNG |")
    assert result.index("| PNG |") < result.index("| WEBP |")

    assert "| Minimum | 640 |" in result
    assert "| Maximum | 1280 |" in result
    assert "| Average | 906.67 |" in result

    assert "| Minimum | 480 |" in result
    assert "| Maximum | 720 |" in result
    assert "| Average | 600.00 |" in result

    assert "| Size | 2.00 KB |" in result


def test_format_dataset_markdown_with_no_images() -> None:
    """Explain when no supported image formats were found."""

    stats = create_statistics()
    stats.extension_counts = {}

    result = format_dataset_markdown(
        Path("data/sample_dataset"),
        stats,
    )

    assert "## Image Formats" in result
    assert "No supported image files found." in result
