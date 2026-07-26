"""Command-line dataset summary for Nautilus Vision."""

from __future__ import annotations

import argparse
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_analyzer import analyze_dataset
from poseidon_ai.nautilus_vision.dataset_statistics import DatasetStatistics

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
        f"Total Bytes       : {stats.total_size_bytes}\n"
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

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    stats = analyze_dataset(dataset_path)

    summary = format_dataset_summary(dataset_path, stats)
    print(summary)

if __name__ == "__main__":
    main()