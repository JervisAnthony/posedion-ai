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
    assert stats.extension_counts == {}


def test_extension_counts_default_is_not_shared() -> None:
    """Each DatasetStatistics instance should own its extension counts."""

    first = DatasetStatistics(dataset_path=Path("first"))
    second = DatasetStatistics(dataset_path=Path("second"))

    first.extension_counts["jpeg"] = 1

    assert second.extension_counts == {}
