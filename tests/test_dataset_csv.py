import csv
import io
import json
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
        extension_counts={
            "webp": 1,
            "jpeg": 2,
            "png": 1,
        },
    )

def test_format_dataset_csv() -> None:
    """Format dataset statistics as CSV."""

    stats = create_statistics()

    result = format_dataset_csv(
        Path("data/sample_dataset"),
        stats,
    )

    rows = list(csv.reader(io.StringIO(result)))
    header, values = rows

    assert header == [
        "dataset_path",
        "total_images",
        "valid_images",
        "invalid_images",
        "extension_counts",
        "min_width",
        "max_width",
        "average_width",
        "min_height",
        "max_height",
        "average_height",
        "total_size_bytes",
    ]
    assert values[0] == str(Path("data/sample_dataset"))
    assert values[1] == "4"
    assert values[2] == "3"
    assert values[3] == "1"
    assert json.loads(values[4]) == {
        "jpeg": 2,
        "png": 1,
        "webp": 1,
    }
    assert values[5] == "640"
    assert values[6] == "1280"
    assert values[7] == "906.67"
    assert values[8] == "480"
    assert values[9] == "720"
    assert values[10] == "600.00"
    assert values[11] == "2048"


def test_format_dataset_csv_escapes_extension_json() -> None:
    """Produce valid CSV around JSON containing commas and quotes."""

    result = format_dataset_csv(
        Path("data/sample_dataset"),
        create_statistics(),
    )

    rows = list(csv.reader(io.StringIO(result)))

    assert len(rows) == 2
    assert json.loads(rows[1][4]) == {
        "jpeg": 2,
        "png": 1,
        "webp": 1,
    }


def test_format_dataset_csv_with_no_images() -> None:
    """Represent empty extension statistics as a JSON object."""

    stats = create_statistics()
    stats.extension_counts = {}

    result = format_dataset_csv(
        Path("data/sample_dataset"),
        stats,
    )

    rows = list(csv.reader(io.StringIO(result)))

    assert json.loads(rows[1][4]) == {}
