"""Command-line dataset summary for Nautilus Vision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_analyzer import analyze_dataset
from poseidon_ai.nautilus_vision.dataset_csv import format_dataset_csv
from poseidon_ai.nautilus_vision.dataset_statistics import DatasetStatistics
from poseidon_ai.utils.filesize import format_file_size
from poseidon_ai.nautilus_vision.dataset_markdown import (
    format_dataset_markdown,
)

def format_dataset_summary(
    dataset_path: Path,
    stats: DatasetStatistics,
) -> str:
    """Format dataset statistics for terminal output."""

    return (
        "Dataset Summary\n"
        "========================\n"
        f"Dataset Path      : {dataset_path}\n"
        f"Total Images      : {stats.total_images}\n"
        f"Valid Images      : {stats.valid_images}\n"
        f"Invalid Images    : {stats.invalid_images}\n"
        "\n"
        "Width\n"
        "-----\n"
        f"Minimum           : {stats.min_width}\n"
        f"Maximum           : {stats.max_width}\n"
        f"Average           : {stats.average_width:.2f}\n"
        "\n"
        "Height\n"
        "------\n"
        f"Minimum           : {stats.min_height}\n"
        f"Maximum           : {stats.max_height}\n"
        f"Average           : {stats.average_height:.2f}\n"
        "\n"
        "Dataset Size\n"
        "------------\n"
        f"Dataset Size      : {format_file_size(stats.total_size_bytes)}\n"
    )

def format_dataset_summary_json(
    dataset_path: Path,
    stats: DatasetStatistics,
) -> str:
    """Format dataset statistics as JSON."""

    payload = {
        "dataset_path": str(dataset_path),
        "total_images": stats.total_images,
        "valid_images": stats.valid_images,
        "invalid_images": stats.invalid_images,
        "width": {
            "minimum": stats.min_width,
            "maximum": stats.max_width,
            "average": stats.average_width,
        },
        "height": {
            "minimum": stats.min_height,
            "maximum": stats.max_height,
            "average": stats.average_height,
        },
        "total_size_bytes": stats.total_size_bytes,
        "formatted_size": format_file_size(
            stats.total_size_bytes
        ),
    }

    return json.dumps(
        payload,
        indent=2,
    )

def main() -> None:
    """Run the dataset summary command."""

    parser = argparse.ArgumentParser(
        description="Display statistics for an image dataset."
    )

    parser.add_argument(
        "dataset_path",
        help="Path to the image dataset.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output dataset statistics as JSON.",
    )

    parser.add_argument(
        "--format",
        choices=("text", "json", "csv", "markdown"),
        default="text",
        help="Output format: text, JSON, CSV, or Markdown.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to a file.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    stats = analyze_dataset(dataset_path)

    output_format = "json" if args.json else args.format

    if output_format == "json":
        summary = format_dataset_summary_json(
            dataset_path,
            stats,
        )
    elif output_format == "csv":
        summary = format_dataset_csv(
            dataset_path,
            stats,
        )
    elif output_format == "markdown":
        summary = format_dataset_markdown(
            dataset_path,
            stats,
        )
    else:
        summary = format_dataset_summary(
            dataset_path,
            stats,
        )

    if args.output:
        args.output.write_text(
            summary,
            encoding="utf-8",
        )
    else:
        print(summary)

if __name__ == "__main__":
    main()