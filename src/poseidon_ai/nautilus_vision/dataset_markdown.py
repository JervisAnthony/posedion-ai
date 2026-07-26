from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
)
from poseidon_ai.utils.filesize import format_file_size


def format_dataset_markdown(
    dataset_path: Path,
    stats: DatasetStatistics,
) -> str:
    """Return dataset statistics as Markdown."""

    markdown_lines = [
        "# Dataset Summary",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Dataset Path | {dataset_path.as_posix()} |",
        f"| Total Images | {stats.total_images} |",
        f"| Valid Images | {stats.valid_images} |",
        f"| Invalid Images | {stats.invalid_images} |",
        "",
        "## Width",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Minimum | {stats.min_width} |",
        f"| Maximum | {stats.max_width} |",
        f"| Average | {stats.average_width:.2f} |",
        "",
        "## Height",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Minimum | {stats.min_height} |",
        f"| Maximum | {stats.max_height} |",
        f"| Average | {stats.average_height:.2f} |",
        "",
        "## Dataset Size",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Size | {format_file_size(stats.total_size_bytes)} |",
    ]

    return "\n".join(markdown_lines)