from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_markdown import (
    format_dataset_markdown,
)
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
    DuplicateImageGroup,
    ImageFormatStatistics,
    InvalidImageDiagnostic,
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
        min_pixel_count=307_200,
        max_pixel_count=2_073_600,
        average_pixel_count=1_190_400.0,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0,
        average_aspect_ratio=7 / 6,
        orientation_counts={
            "landscape": 1,
            "portrait": 1,
            "square": 1,
        },
        min_file_size_bytes=256,
        max_file_size_bytes=1024,
        average_file_size_bytes=2048 / 3,
        total_size_bytes=2048,
        extension_counts={
            "webp": 1,
            "jpeg": 2,
            "png": 1,
        },
        format_statistics={
            "webp": ImageFormatStatistics(1, 0, 1, 0, 0.0),
            "jpeg": ImageFormatStatistics(2, 2, 0, 1_365, 682.5),
            "png": ImageFormatStatistics(1, 1, 0, 683, 683.0),
        },
        channel_counts={
            10: 1,
            1: 1,
            2: 1,
        },
        duplicate_image_groups=[
            DuplicateImageGroup(
                sha256="a" * 64,
                image_paths=(
                    Path("data/copy-c.jpg"),
                    Path("data/copy-a.jpg"),
                    Path("data/copy`b.jpg"),
                ),
            )
        ],
        invalid_image_diagnostics=[
            InvalidImageDiagnostic(
                image_path=Path("data/a-corrupt.jpg"),
                errors=("Image could not be decoded.",),
            )
        ],
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
    assert "## Image Format Statistics" in result
    assert "## Image Channels" in result
    assert "## Image Resolution" in result
    assert "## Image Aspect Ratios" in result
    assert "## Image File Sizes" in result
    assert "## Exact Duplicate Images" in result
    assert "## Invalid Image Diagnostics" in result
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
    assert (
        "| Format | Total | Valid | Invalid | "
        "Total Valid Bytes | Average Valid Bytes |"
    ) in result
    assert "| JPEG | 2 | 2 | 0 | 1,365 | 682.50 |" in result
    assert "| PNG | 1 | 1 | 0 | 683 | 683.00 |" in result
    assert "| WEBP | 1 | 0 | 1 | 0 | 0.00 |" in result
    assert result.index(
        "## Image Formats"
    ) < result.index("## Image Format Statistics")
    assert result.index(
        "## Image Format Statistics"
    ) < result.index("## Image Channels")
    assert "| Channels | Images |" in result
    assert "| 1 | 1 |" in result
    assert "| 2 | 1 |" in result
    assert "| 10 | 1 |" in result
    assert result.index("| 1 | 1 |") < result.index("| 2 | 1 |")
    assert result.index("| 2 | 1 |") < result.index("| 10 | 1 |")
    assert result.index("## Image Channels") < result.index(
        "## Image Resolution"
    )
    assert result.index("## Image Resolution") < result.index(
        "## Image Aspect Ratios"
    )
    assert result.index("## Image Aspect Ratios") < result.index(
        "## Image File Sizes"
    )
    assert result.index("## Image File Sizes") < result.index(
        "## Exact Duplicate Images"
    )
    assert result.index("## Exact Duplicate Images") < result.index(
        "## Invalid Image Diagnostics"
    )
    assert "| Metric | Pixels | Megapixels |" in result
    assert "| Minimum | 307,200 | 0.31 |" in result
    assert "| Maximum | 2,073,600 | 2.07 |" in result
    assert "| Average | 1,190,400.00 | 1.19 |" in result
    assert "| Minimum Ratio | 0.50 |" in result
    assert "| Maximum Ratio | 2.00 |" in result
    assert "| Average Ratio | 1.17 |" in result
    assert "| Landscape Images | 1 |" in result
    assert "| Portrait Images | 1 |" in result
    assert "| Square Images | 1 |" in result
    assert "| Minimum | 256 |" in result
    assert "| Maximum | 1,024 |" in result
    assert "| Average | 682.67 |" in result
    assert "| Duplicate Groups | 1 |" in result
    assert "| Files in Groups | 3 |" in result
    assert "| Redundant Copies | 2 |" in result
    assert "### Duplicate Group 1" in result
    assert f"**SHA-256:** `{'a' * 64}`" in result
    assert "- `data/copy-a.jpg`" in result
    assert "- ``data/copy`b.jpg``" in result
    assert "- `data/copy-c.jpg`" in result
    assert result.index("data/copy-a.jpg") < result.index(
        "data/copy-c.jpg"
    )
    assert result.index("data/copy-c.jpg") < result.index(
        "data/copy`b.jpg"
    )
    assert "### `data/a-corrupt.jpg`" in result
    assert "- Image could not be decoded." in result

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
    stats.format_statistics = {}
    stats.channel_counts = {}
    stats.duplicate_image_groups = []
    stats.total_images = 0
    stats.valid_images = 0
    stats.invalid_images = 0
    stats.min_width = 0
    stats.max_width = 0
    stats.average_width = 0.0
    stats.min_height = 0
    stats.max_height = 0
    stats.average_height = 0.0
    stats.min_pixel_count = 0
    stats.max_pixel_count = 0
    stats.average_pixel_count = 0.0
    stats.min_aspect_ratio = 0.0
    stats.max_aspect_ratio = 0.0
    stats.average_aspect_ratio = 0.0
    stats.orientation_counts = {}
    stats.min_file_size_bytes = 0
    stats.max_file_size_bytes = 0
    stats.average_file_size_bytes = 0.0
    stats.total_size_bytes = 0
    stats.invalid_image_diagnostics = []

    result = format_dataset_markdown(
        Path("data/sample_dataset"),
        stats,
    )

    assert "## Image Formats" in result
    assert "No supported image files found." in result
    assert "No image format statistics found." in result
    assert "No valid image channel data found." in result
    assert "No valid image resolution data found." in result
    assert "No valid image aspect ratio data found." in result
    assert "No valid image file size data found." in result
    assert "No exact duplicate images found." in result
    assert "No invalid images found." in result


def test_format_dataset_markdown_sorts_and_escapes_diagnostics() -> None:
    """Render all diagnostics safely in portable path order."""

    stats = create_statistics()
    stats.total_images = 5
    stats.invalid_images = 2
    stats.extension_counts["png"] = 2
    stats.invalid_image_diagnostics = [
        InvalidImageDiagnostic(
            image_path=Path("data/z`small.png"),
            errors=(
                "Width *10px* is below minimum 32px.",
                "Height [10px] is below minimum 32px.",
            ),
        ),
        *stats.invalid_image_diagnostics,
    ]

    result = format_dataset_markdown(
        Path("data/sample_dataset"),
        stats,
    )

    assert result.index("data/a-corrupt.jpg") < result.index(
        "data/z`small.png"
    )
    assert "### ``data/z`small.png``" in result
    assert "- Width \\*10px\\* is below minimum 32px." in result
    assert "- Height \\[10px\\] is below minimum 32px." in result
