from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_statistics import DatasetStatistics


def test_dataset_statistics_defaults() -> None:
    """DatasetStatistics should initialize with default values."""

    stats = DatasetStatistics(
        dataset_path=Path("dataset"),
    )

    assert stats.total_images == 0
    assert stats.valid_images == 0
    assert stats.invalid_images == 0
    assert stats.total_size_bytes == 0

    assert stats.average_width == 0.0
    assert stats.average_height == 0.0