"""Tests for dataset summary formatting."""

import csv
import io
import json
import sys
from pathlib import Path

import pytest

import poseidon_ai.nautilus_vision.dataset_summary as dataset_summary
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
    InvalidImageDiagnostic,
)
from poseidon_ai.nautilus_vision.dataset_summary import (
    format_dataset_summary,
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
        extension_counts={
            "webp": 1,
            "jpeg": 2,
            "png": 1,
        },
        invalid_image_diagnostics=[
            InvalidImageDiagnostic(
                image_path=Path("data/a-corrupt.jpg"),
                errors=("Image could not be decoded.",),
            )
        ],
    )


def create_multiple_diagnostic_statistics() -> DatasetStatistics:
    """Create statistics with unsorted, multi-error diagnostics."""

    stats = create_statistics()
    stats.total_images = 5
    stats.invalid_images = 2
    stats.extension_counts["png"] = 2
    stats.invalid_image_diagnostics = [
        InvalidImageDiagnostic(
            image_path=Path("data/z-small.png"),
            errors=(
                "Width 10px is below minimum 32px.",
                "Height 10px is below minimum 32px.",
            ),
        ),
        *stats.invalid_image_diagnostics,
    ]
    return stats


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
    assert payload["extension_counts"] == {
        "jpeg": 2,
        "png": 1,
        "webp": 1,
    }
    assert list(payload["extension_counts"]) == [
        "jpeg",
        "png",
        "webp",
    ]
    assert payload["invalid_image_diagnostics"] == [
        {
            "image_path": "data/a-corrupt.jpg",
            "errors": ["Image could not be decoded."],
        }
    ]
    assert payload["width"]["minimum"] == 640
    assert payload["height"]["average"] == 600.0
    assert payload["total_size_bytes"] == 2048
    assert payload["formatted_size"] == "2.00 KB"


def test_format_dataset_summary_json_with_no_images() -> None:
    """Represent empty extension statistics as a JSON object."""

    stats = create_statistics()
    stats.extension_counts = {}
    stats.total_images = 0
    stats.valid_images = 0
    stats.invalid_images = 0
    stats.min_width = 0
    stats.max_width = 0
    stats.average_width = 0.0
    stats.min_height = 0
    stats.max_height = 0
    stats.average_height = 0.0
    stats.total_size_bytes = 0
    stats.invalid_image_diagnostics = []

    result = format_dataset_summary_json(
        Path("data/sample_dataset"),
        stats,
    )

    assert json.loads(result)["extension_counts"] == {}
    assert json.loads(result)["invalid_image_diagnostics"] == []


def test_format_dataset_summary_json_sorts_all_diagnostics() -> None:
    """Serialize every diagnostic and error in portable path order."""

    result = format_dataset_summary_json(
        Path("data/sample_dataset"),
        create_multiple_diagnostic_statistics(),
    )

    diagnostics = json.loads(result)["invalid_image_diagnostics"]

    assert diagnostics == [
        {
            "image_path": "data/a-corrupt.jpg",
            "errors": ["Image could not be decoded."],
        },
        {
            "image_path": "data/z-small.png",
            "errors": [
                "Width 10px is below minimum 32px.",
                "Height 10px is below minimum 32px.",
            ],
        },
    ]


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
    assert "Image Formats" in captured.out
    assert "JPEG              : 2" in captured.out
    assert "PNG               : 1" in captured.out
    assert "WEBP              : 1" in captured.out
    assert captured.out.index("JPEG") < captured.out.index("PNG")
    assert captured.out.index("PNG") < captured.out.index("WEBP")
    assert "Invalid Image Diagnostics" in captured.out
    assert "data/a-corrupt.jpg" in captured.out
    assert "  - Image could not be decoded." in captured.out


def test_text_summary_sorts_all_diagnostics() -> None:
    """Render every diagnostic and error in portable path order."""

    result = format_dataset_summary(
        Path("data/sample_dataset"),
        create_multiple_diagnostic_statistics(),
    )

    assert result.index("data/a-corrupt.jpg") < result.index(
        "data/z-small.png"
    )
    assert "  - Image could not be decoded." in result
    assert "  - Width 10px is below minimum 32px." in result
    assert "  - Height 10px is below minimum 32px." in result


def test_text_summary_with_no_invalid_images() -> None:
    """Render the empty diagnostic state."""

    stats = create_statistics()
    stats.total_images = stats.valid_images
    stats.invalid_images = 0
    stats.extension_counts = {
        "jpeg": 1,
        "png": 1,
        "webp": 1,
    }
    stats.invalid_image_diagnostics = []

    result = format_dataset_summary(
        Path("data/sample_dataset"),
        stats,
    )

    assert "No invalid images found." in result


def test_main_prints_empty_image_formats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explain when no supported image formats were found."""

    stats = create_statistics()
    stats.extension_counts = {}
    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        lambda dataset_path: stats,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", "data/sample_dataset"],
    )

    dataset_summary.main()

    assert (
        "No supported image files found."
        in capsys.readouterr().out
    )


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
    assert payload["invalid_image_diagnostics"][0][
        "image_path"
    ] == "data/a-corrupt.jpg"


def test_main_reports_analyzer_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pass analyzer-produced diagnostics to the selected formatter."""

    invalid_image = tmp_path / "broken.jpg"
    invalid_image.write_bytes(b"not a decodable image")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--format",
            "json",
        ],
    )

    dataset_summary.main()

    payload = json.loads(capsys.readouterr().out)
    diagnostics = payload["invalid_image_diagnostics"]

    assert payload["invalid_images"] == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["image_path"].endswith("/broken.jpg")
    assert diagnostics[0]["errors"] == [
        "Image could not be decoded."
    ]


def test_main_prints_csv_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print CSV when selected through --format."""

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
            "--format",
            "csv",
        ],
    )

    dataset_summary.main()

    captured = capsys.readouterr()

    rows = list(
        csv.reader(
            io.StringIO(captured.out)
        )
    )

    assert rows[0] == [
        "dataset_path",
        "total_images",
        "valid_images",
        "invalid_images",
        "extension_counts",
        "invalid_image_diagnostics",
        "min_width",
        "max_width",
        "average_width",
        "min_height",
        "max_height",
        "average_height",
        "total_size_bytes",
    ]

    assert rows[1] == [
        str(Path("data/sample_dataset")),
        "4",
        "3",
        "1",
        '{"jpeg": 2, "png": 1, "webp": 1}',
        (
            '[{"image_path": "data/a-corrupt.jpg", '
            '"errors": ["Image could not be decoded."]}]'
        ),
        "640",
        "1280",
        "906.67",
        "480",
        "720",
        "600.00",
        "2048",
    ]


def test_json_shortcut_takes_precedence_over_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Prefer the backward-compatible --json shortcut."""

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
            "--format",
            "csv",
        ],
    )

    dataset_summary.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["total_images"] == 4
    assert payload["formatted_size"] == "2.00 KB"
    assert payload["extension_counts"] == {
        "jpeg": 2,
        "png": 1,
        "webp": 1,
    }
    assert payload["invalid_image_diagnostics"][0][
        "errors"
    ] == ["Image could not be decoded."]


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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    assert "| JPEG | 2 |" in captured.out
    assert "## Invalid Image Diagnostics" in captured.out
    assert "### `data/a-corrupt.jpg`" in captured.out
    assert "| Average | 906.67 |" in captured.out
    assert "| Size | 2.00 KB |" in captured.out
