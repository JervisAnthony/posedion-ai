from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
)


def format_dataset_csv(
    dataset_path: Path,
    stats: DatasetStatistics,
) -> str:
    """Return dataset statistics as CSV."""
    
    csv_lines = [
        "dataset_path,total_images,valid_images,invalid_images,"
        "min_width,max_width,average_width,"
        "min_height,max_height,average_height,"
        "total_size_bytes",
        f"{dataset_path},{stats.total_images},{stats.valid_images},"
        f"{stats.invalid_images},{stats.min_width},{stats.max_width},"
        f"{stats.average_width:.2f},{stats.min_height},{stats.max_height},"
        f"{stats.average_height:.2f},{stats.total_size_bytes}",
    ]

    return "\n".join(csv_lines)
