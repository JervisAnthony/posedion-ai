"""Tests for dataset summary formatting."""

import json
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
)
from poseidon_ai.nautilus_vision.dataset_summary import (
    format_dataset_summary_json,
)


def test_format_dataset_summary_json() -> None:
    """Format dataset statistics as valid JSON."""

    stats = DatasetStatistics(
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

    result = format_dataset_summary_json(
        Path("data/sample_dataset"),
        stats,
    )

    payload = json.loads(result)

    assert payload["dataset_path"] == "data\\sample_dataset"
    assert payload["total_images"] == 4
    assert payload["valid_images"] == 3
    assert payload["invalid_images"] == 1
    assert payload["width"]["minimum"] == 640
    assert payload["height"]["average"] == 600.0
    assert payload["total_size_bytes"] == 2048
    assert payload["formatted_size"] == "2.00 KB"