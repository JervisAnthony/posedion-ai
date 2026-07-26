"""Tests for dataset summary formatting."""

import json
import sys
from pathlib import Path

import pytest

import poseidon_ai.nautilus_vision.dataset_summary as dataset_summary
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
)
from poseidon_ai.nautilus_vision.dataset_summary import (
    format_dataset_summary_json,
)


def create_statistics() -> DatasetStatistics:
    """Create representative dataset statistics for CLI tests."""

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


def test_format_dataset_summary_json() -> None:
    """Format dataset statistics as valid JSON."""

    stats = create_statistics()

    result = format_dataset_summary_json(
        Path("data/sample_dataset"),
        stats,
    )

    payload = json.loads(result)

    assert payload["dataset_path"] == str(
        Path("data/sample_dataset")
    )
    assert payload["total_images"] == 4
    assert payload["valid_images"] == 3
    assert payload["invalid_images"] == 1
    assert payload["width"]["minimum"] == 640
    assert payload["height"]["average"] == 600.0
    assert payload["total_size_bytes"] == 2048
    assert payload["formatted_size"] == "2.00 KB"


def test_main_prints_text_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print a human-readable summary without output flags."""

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        lambda dataset_path: create_statistics(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
        ],
    )

    dataset_summary.main()

    captured = capsys.readouterr()

    assert "Dataset Summary" in captured.out
    assert "Total Images      : 4" in captured.out
    assert "Valid Images      : 3" in captured.out
    assert "Invalid Images    : 1" in captured.out


def test_main_prints_json_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print JSON when the --json flag is provided."""

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        lambda dataset_path: create_statistics(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
            "--json",
        ],
    )

    dataset_summary.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["total_images"] == 4
    assert payload["formatted_size"] == "2.00 KB"

def test_main_writes_summary_to_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write the summary to a file when --output is provided."""

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        lambda dataset_path: create_statistics(),
    )

    output_path = tmp_path / "report.txt"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
            "--output",
            str(output_path),
        ],
    )

    dataset_summary.main()

    captured = capsys.readouterr()
    report = output_path.read_text(encoding="utf-8")

    assert captured.out == ""
    assert "Dataset Summary" in report
    assert "Total Images      : 4" in report

def test_main_prints_markdown_summary(
    monkeypatch,
    capsys,
) -> None:
    """Print the dataset summary as Markdown."""

    stats = create_statistics()

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        lambda dataset_path: stats,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
            "--format",
            "markdown",
        ],
    )

    dataset_summary.main()

    captured = capsys.readouterr()

    assert "# Dataset Summary" in captured.out
    assert "## Overview" in captured.out
    assert "| Total Images | 4 |" in captured.out
    assert "| Average | 906.67 |" in captured.out
    assert "| Size | 2.00 KB |" in captured.out
