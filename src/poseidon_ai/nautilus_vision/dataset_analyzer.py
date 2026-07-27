"""Dataset analysis utilities for Nautilus Vision."""

from __future__ import annotations

from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_loader import load_image_dataset
from poseidon_ai.nautilus_vision.dataset_statistics import DatasetStatistics
from poseidon_ai.nautilus_vision.image_metadata import get_image_metadata
from poseidon_ai.nautilus_vision.image_validator import validate_image


def analyze_dataset(
    dataset_path: str | Path,
    *,
    recursive: bool = False,
) -> DatasetStatistics:
    """Analyze an image dataset and compute summary statistics.

    Parameters
    ----------
    dataset_path:
        Directory containing the image dataset.
    recursive:
        Search nested directories when True.

    Returns
    -------
    DatasetStatistics
        Summary statistics for valid and invalid images.
    """
    dataset_directory = Path(dataset_path)

    image_paths = load_image_dataset(
        dataset_directory,
        recursive=recursive,
        validate=False,
    )

    stats = DatasetStatistics(dataset_path=dataset_directory)

    widths: list[int] = []
    heights: list[int] = []

    for image_path in image_paths:
        stats.total_images += 1

        extension = image_path.suffix.lower().removeprefix(".")
        extension = {
            "jpg": "jpeg",
            "tif": "tiff",
        }.get(extension, extension)
        stats.extension_counts[extension] = (
            stats.extension_counts.get(extension, 0) + 1
        )

        validation_result = validate_image(image_path)

        if not validation_result.is_valid:
            stats.invalid_images += 1
            continue

        metadata = get_image_metadata(image_path)

        stats.valid_images += 1
        stats.total_size_bytes += metadata["size_bytes"]

        widths.append(metadata["width"])
        heights.append(metadata["height"])

    if widths:
        stats.min_width = min(widths)
        stats.max_width = max(widths)
        stats.average_width = sum(widths) / len(widths)

    if heights:
        stats.min_height = min(heights)
        stats.max_height = max(heights)
        stats.average_height = sum(heights) / len(heights)

    stats.extension_counts = dict(sorted(stats.extension_counts.items()))

    return stats
