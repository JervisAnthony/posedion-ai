from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_csv import (
    format_dataset_csv,
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
    )

def test_format_dataset_csv() -> None:
    """Format dataset statistics as CSV."""

    stats = create_statistics()

    result = format_dataset_csv(
        Path("data/sample_dataset"),
        stats,
    )

    lines = result.splitlines()

    assert lines[0] == (
        "dataset_path,total_images,valid_images,invalid_images,"
        "min_width,max_width,average_width,"
        "min_height,max_height,average_height,"
        "total_size_bytes"
    )

    values = lines[1].split(",")

    assert values[0] == str(Path("data/sample_dataset"))
    assert values[1] == "4"
    assert values[2] == "3"
    assert values[3] == "1"
    assert values[4] == "640"
    assert values[5] == "1280"
    assert values[6] == "906.67"
    assert values[7] == "480"
    assert values[8] == "720"
    assert values[9] == "600.00"
    assert values[10] == "2048"