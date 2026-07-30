"""Tests for dataset summary formatting."""

import csv
import io
import json
import sys
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
import pytest

import poseidon_ai.nautilus_vision.dataset_summary as dataset_summary
from poseidon_ai.nautilus_vision.dataset_manifest import (
    DatasetAnalysisResult,
    DatasetManifestEntry,
)
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
    DuplicateImageGroup,
    ImageFormatStatistics,
    InvalidImageDiagnostic,
)
from poseidon_ai.nautilus_vision.dataset_summary import (
    format_dataset_summary,
    format_dataset_summary_json,
)
from poseidon_ai.nautilus_vision.image_hash import calculate_sha256


def create_test_image(
    path: Path,
    *,
    width: int = 100,
    height: int = 100,
) -> None:
    """Create a valid image for CLI integration tests."""

    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image) is True


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
        min_pixel_count=307_200,
        max_pixel_count=2_073_600,
        average_pixel_count=1_190_400.0,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0,
        average_aspect_ratio=7 / 6,
        orientation_counts={
            "landscape": 1,
            "portrait": 1,
            "square": 1,
        },
        min_file_size_bytes=256,
        max_file_size_bytes=1024,
        average_file_size_bytes=2048 / 3,
        total_size_bytes=2048,
        extension_counts={
            "webp": 1,
            "jpeg": 2,
            "png": 1,
        },
        format_statistics={
            "webp": ImageFormatStatistics(1, 0, 1, 0, 0.0),
            "jpeg": ImageFormatStatistics(2, 2, 0, 1_365, 682.5),
            "png": ImageFormatStatistics(1, 1, 0, 683, 683.0),
        },
        channel_counts={
            10: 1,
            1: 1,
            2: 1,
        },
        duplicate_image_groups=[
            DuplicateImageGroup(
                sha256="a" * 64,
                image_paths=(
                    Path("data/copy-c.jpg"),
                    Path("data/copy-a.jpg"),
                    Path("data/copy-b.jpg"),
                ),
            )
        ],
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
    assert payload["format_statistics"] == {
        "jpeg": {
            "total_images": 2,
            "valid_images": 2,
            "invalid_images": 0,
            "total_valid_size_bytes": 1_365,
            "average_valid_size_bytes": 682.5,
        },
        "png": {
            "total_images": 1,
            "valid_images": 1,
            "invalid_images": 0,
            "total_valid_size_bytes": 683,
            "average_valid_size_bytes": 683.0,
        },
        "webp": {
            "total_images": 1,
            "valid_images": 0,
            "invalid_images": 1,
            "total_valid_size_bytes": 0,
            "average_valid_size_bytes": 0.0,
        },
    }
    assert list(payload["format_statistics"]) == [
        "jpeg",
        "png",
        "webp",
    ]
    assert list(payload["format_statistics"]["jpeg"]) == [
        "total_images",
        "valid_images",
        "invalid_images",
        "total_valid_size_bytes",
        "average_valid_size_bytes",
    ]
    assert all(
        isinstance(
            payload["format_statistics"]["jpeg"][key],
            int,
        )
        for key in (
            "total_images",
            "valid_images",
            "invalid_images",
            "total_valid_size_bytes",
        )
    )
    assert isinstance(
        payload["format_statistics"]["jpeg"][
            "average_valid_size_bytes"
        ],
        float,
    )
    assert payload["channel_counts"] == {
        "1": 1,
        "2": 1,
        "10": 1,
    }
    assert list(payload["channel_counts"]) == ["1", "2", "10"]
    assert payload["resolution_statistics"] == {
        "minimum_pixels": 307_200,
        "maximum_pixels": 2_073_600,
        "average_pixels": 1_190_400.0,
        "minimum_megapixels": 0.3072,
        "maximum_megapixels": 2.0736,
        "average_megapixels": 1.1904,
    }
    assert payload["aspect_ratio_statistics"] == {
        "minimum": 0.5,
        "maximum": 2.0,
        "average": 1.166667,
        "orientation_counts": {
            "landscape": 1,
            "portrait": 1,
            "square": 1,
        },
    }
    assert payload["file_size_statistics"] == {
        "minimum_bytes": 256,
        "maximum_bytes": 1024,
        "average_bytes": 682.67,
    }
    assert list(payload["file_size_statistics"]) == [
        "minimum_bytes",
        "maximum_bytes",
        "average_bytes",
    ]
    assert isinstance(
        payload["file_size_statistics"]["minimum_bytes"],
        int,
    )
    assert isinstance(
        payload["file_size_statistics"]["maximum_bytes"],
        int,
    )
    assert isinstance(
        payload["file_size_statistics"]["average_bytes"],
        float,
    )
    assert list(
        payload["aspect_ratio_statistics"]["orientation_counts"]
    ) == ["landscape", "portrait", "square"]
    assert all(
        isinstance(
            payload["aspect_ratio_statistics"][key],
            float,
        )
        for key in ("minimum", "maximum", "average")
    )
    assert all(
        isinstance(count, int)
        for count in payload["aspect_ratio_statistics"][
            "orientation_counts"
        ].values()
    )
    assert payload["duplicate_images"] == {
        "group_count": 1,
        "file_count": 3,
        "redundant_copy_count": 2,
        "groups": [
            {
                "sha256": "a" * 64,
                "image_paths": [
                    "data/copy-a.jpg",
                    "data/copy-b.jpg",
                    "data/copy-c.jpg",
                ],
            }
        ],
    }
    assert all(
        isinstance(
            payload["duplicate_images"][key],
            int,
        )
        for key in (
            "group_count",
            "file_count",
            "redundant_copy_count",
        )
    )
    assert all(
        isinstance(value, int | float)
        for value in payload["resolution_statistics"].values()
    )
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
    assert set(payload) == {
        "dataset_path",
        "total_images",
        "valid_images",
        "invalid_images",
        "extension_counts",
        "format_statistics",
        "channel_counts",
        "resolution_statistics",
        "aspect_ratio_statistics",
        "file_size_statistics",
        "duplicate_images",
        "invalid_image_diagnostics",
        "width",
        "height",
        "total_size_bytes",
        "formatted_size",
    }
    payload_keys = list(payload)
    assert payload_keys.index(
        "extension_counts"
    ) < payload_keys.index("format_statistics")
    assert payload_keys.index(
        "format_statistics"
    ) < payload_keys.index("channel_counts")
    assert payload_keys.index(
        "resolution_statistics"
    ) < payload_keys.index("aspect_ratio_statistics")
    assert payload_keys.index(
        "aspect_ratio_statistics"
    ) < payload_keys.index("file_size_statistics")
    assert payload_keys.index(
        "file_size_statistics"
    ) < payload_keys.index("duplicate_images")


def test_format_dataset_summary_json_with_no_images() -> None:
    """Represent empty extension statistics as a JSON object."""

    stats = create_statistics()
    stats.extension_counts = {}
    stats.format_statistics = {}
    stats.channel_counts = {}
    stats.duplicate_image_groups = []
    stats.total_images = 0
    stats.valid_images = 0
    stats.invalid_images = 0
    stats.min_width = 0
    stats.max_width = 0
    stats.average_width = 0.0
    stats.min_height = 0
    stats.max_height = 0
    stats.average_height = 0.0
    stats.min_pixel_count = 0
    stats.max_pixel_count = 0
    stats.average_pixel_count = 0.0
    stats.min_aspect_ratio = 0.0
    stats.max_aspect_ratio = 0.0
    stats.average_aspect_ratio = 0.0
    stats.orientation_counts = {}
    stats.min_file_size_bytes = 0
    stats.max_file_size_bytes = 0
    stats.average_file_size_bytes = 0.0
    stats.total_size_bytes = 0
    stats.invalid_image_diagnostics = []

    result = format_dataset_summary_json(
        Path("data/sample_dataset"),
        stats,
    )

    assert json.loads(result)["extension_counts"] == {}
    assert json.loads(result)["format_statistics"] == {}
    assert json.loads(result)["channel_counts"] == {}
    assert json.loads(result)["resolution_statistics"] == {
        "minimum_pixels": 0,
        "maximum_pixels": 0,
        "average_pixels": 0.0,
        "minimum_megapixels": 0.0,
        "maximum_megapixels": 0.0,
        "average_megapixels": 0.0,
    }
    assert json.loads(result)["aspect_ratio_statistics"] == {
        "minimum": 0.0,
        "maximum": 0.0,
        "average": 0.0,
        "orientation_counts": {
            "landscape": 0,
            "portrait": 0,
            "square": 0,
        },
    }
    assert json.loads(result)["file_size_statistics"] == {
        "minimum_bytes": 0,
        "maximum_bytes": 0,
        "average_bytes": 0.0,
    }
    assert json.loads(result)["duplicate_images"] == {
        "group_count": 0,
        "file_count": 0,
        "redundant_copy_count": 0,
        "groups": [],
    }
    assert json.loads(result)["invalid_image_diagnostics"] == []


def test_format_dataset_summary_json_rounds_format_average() -> None:
    """Round per-format averages without changing numeric output types."""

    stats = DatasetStatistics(
        dataset_path=Path("data/sample_dataset"),
        format_statistics={
            "jpeg": ImageFormatStatistics(
                total_images=3,
                valid_images=3,
                invalid_images=0,
                total_valid_size_bytes=1,
                average_valid_size_bytes=1 / 3,
            )
        },
    )

    payload = json.loads(
        format_dataset_summary_json(
            Path("data/sample_dataset"),
            stats,
        )
    )

    assert payload["format_statistics"]["jpeg"][
        "average_valid_size_bytes"
    ] == 0.33
    assert isinstance(
        payload["format_statistics"]["jpeg"][
            "average_valid_size_bytes"
        ],
        float,
    )


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
        lambda dataset_path, *, recursive=False, min_width=32,
        min_height=32: create_statistics(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
        ],
    )

    assert dataset_summary.main() == 0

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
    assert "Image Format Statistics" in captured.out
    assert "  Total Images        : 2" in captured.out
    assert "  Valid Images        : 2" in captured.out
    assert "  Invalid Images      : 1" in captured.out
    assert "  Total Valid Bytes   : 1,365" in captured.out
    assert "  Average Valid Bytes : 682.50" in captured.out
    assert captured.out.index(
        "Image Formats"
    ) < captured.out.index("Image Format Statistics")
    assert captured.out.index(
        "Image Format Statistics"
    ) < captured.out.index("Image Channels")
    assert "Image Channels" in captured.out
    assert "1 channel          : 1" in captured.out
    assert "2 channels         : 1" in captured.out
    assert "10 channels        : 1" in captured.out
    assert captured.out.index("1 channel") < captured.out.index(
        "2 channels"
    )
    assert captured.out.index("2 channels") < captured.out.index(
        "10 channels"
    )
    assert "Image Resolution" in captured.out
    assert "Minimum Pixels    : 307,200" in captured.out
    assert "Maximum Pixels    : 2,073,600" in captured.out
    assert "Average Pixels    : 1,190,400.00" in captured.out
    assert "Minimum MP        : 0.31" in captured.out
    assert "Maximum MP        : 2.07" in captured.out
    assert "Average MP        : 1.19" in captured.out
    assert "Image Aspect Ratios" in captured.out
    assert "Minimum Ratio      : 0.50" in captured.out
    assert "Maximum Ratio      : 2.00" in captured.out
    assert "Average Ratio      : 1.17" in captured.out
    assert "Landscape Images   : 1" in captured.out
    assert "Portrait Images    : 1" in captured.out
    assert "Square Images      : 1" in captured.out
    assert "Image File Sizes" in captured.out
    assert "Minimum Bytes      : 256" in captured.out
    assert "Maximum Bytes      : 1,024" in captured.out
    assert "Average Bytes      : 682.67" in captured.out
    assert captured.out.index("Image Channels") < captured.out.index(
        "Image Resolution"
    )
    assert captured.out.index("Image Resolution") < captured.out.index(
        "Image Aspect Ratios"
    )
    assert captured.out.index("Image Aspect Ratios") < captured.out.index(
        "Image File Sizes"
    )
    assert captured.out.index("Image File Sizes") < captured.out.index(
        "Exact Duplicate Images"
    )
    assert "Exact Duplicate Images" in captured.out
    assert "Duplicate Groups   : 1" in captured.out
    assert "Files in Groups    : 3" in captured.out
    assert "Redundant Copies   : 2" in captured.out
    assert f"SHA-256            : {'a' * 64}" in captured.out
    assert "- data/copy-a.jpg" in captured.out
    assert "- data/copy-b.jpg" in captured.out
    assert "- data/copy-c.jpg" in captured.out
    assert captured.out.index("data/copy-a.jpg") < captured.out.index(
        "data/copy-b.jpg"
    )
    assert captured.out.index("data/copy-b.jpg") < captured.out.index(
        "data/copy-c.jpg"
    )
    assert captured.out.index("Exact Duplicate Images") < captured.out.index(
        "Invalid Image Diagnostics"
    )
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


def test_text_summary_with_no_valid_images() -> None:
    """Render the explicit empty resolution state."""

    stats = create_statistics()
    stats.valid_images = 0
    stats.channel_counts = {}
    stats.duplicate_image_groups = []
    stats.min_pixel_count = 0
    stats.max_pixel_count = 0
    stats.average_pixel_count = 0.0
    stats.min_aspect_ratio = 0.0
    stats.max_aspect_ratio = 0.0
    stats.average_aspect_ratio = 0.0
    stats.orientation_counts = {
        "landscape": 0,
        "portrait": 0,
        "square": 0,
    }
    stats.min_file_size_bytes = 0
    stats.max_file_size_bytes = 0
    stats.average_file_size_bytes = 0.0

    result = format_dataset_summary(
        Path("data/sample_dataset"),
        stats,
    )

    assert "No valid image channel data found." in result
    assert "No valid image resolution data found." in result
    assert "No valid image aspect ratio data found." in result
    assert "No valid image file size data found." in result
    assert "No exact duplicate images found." in result
    assert "Invalid Image Diagnostics" in result


def test_main_prints_empty_image_formats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explain when no supported image formats were found."""

    stats = create_statistics()
    stats.extension_counts = {}
    stats.format_statistics = {}
    stats.channel_counts = {}
    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        lambda dataset_path, *, recursive=False, min_width=32,
        min_height=32: stats,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", "data/sample_dataset"],
    )

    assert dataset_summary.main() == 0

    output = capsys.readouterr().out
    assert "No supported image files found." in output
    assert "No image format statistics found." in output
    assert "No valid image channel data found." in output


def test_main_prints_json_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print JSON when the --json flag is provided."""

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        lambda dataset_path, *, recursive=False, min_width=32,
        min_height=32: create_statistics(),
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

    assert dataset_summary.main() == 0

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

    assert dataset_summary.main() == 0

    payload = json.loads(capsys.readouterr().out)
    diagnostics = payload["invalid_image_diagnostics"]

    assert payload["invalid_images"] == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["image_path"].endswith("/broken.jpg")
    assert diagnostics[0]["errors"] == [
        "Image could not be decoded."
    ]


def test_main_writes_recursive_exact_duplicates_to_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose analyzer duplicate groups through recursive file output."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    source_path = tmp_path / "source.png"
    nested_copy = nested_directory / "copy.png"
    create_test_image(source_path)
    copyfile(source_path, nested_copy)
    invalid_content = b"identical but not decodable"
    first_invalid = tmp_path / "broken-a.jpg"
    second_invalid = tmp_path / "broken-b.jpg"
    first_invalid.write_bytes(invalid_content)
    second_invalid.write_bytes(invalid_content)
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--recursive",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    duplicate_images = payload["duplicate_images"]

    assert payload["valid_images"] == 2
    assert payload["invalid_images"] == 2
    assert duplicate_images == {
        "group_count": 1,
        "file_count": 2,
        "redundant_copy_count": 1,
        "groups": [
            {
                "sha256": calculate_sha256(source_path),
                "image_paths": [
                    nested_copy.as_posix(),
                    source_path.as_posix(),
                ],
            }
        ],
    }
    serialized_paths = duplicate_images["groups"][0]["image_paths"]
    assert first_invalid.as_posix() not in serialized_paths
    assert second_invalid.as_posix() not in serialized_paths
    assert captured.out == ""
    assert captured.err == ""


def test_main_prints_csv_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print CSV when selected through --format."""

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        lambda dataset_path, *, recursive=False, min_width=32,
        min_height=32: create_statistics(),
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

    assert dataset_summary.main() == 0

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
        "format_statistics",
        "channel_counts",
        "resolution_statistics",
        "aspect_ratio_statistics",
        "file_size_statistics",
        "duplicate_images",
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
            '{"jpeg": {"total_images": 2, "valid_images": 2, '
            '"invalid_images": 0, "total_valid_size_bytes": 1365, '
            '"average_valid_size_bytes": 682.5}, "png": '
            '{"total_images": 1, "valid_images": 1, '
            '"invalid_images": 0, "total_valid_size_bytes": 683, '
            '"average_valid_size_bytes": 683.0}, "webp": '
            '{"total_images": 1, "valid_images": 0, '
            '"invalid_images": 1, "total_valid_size_bytes": 0, '
            '"average_valid_size_bytes": 0.0}}'
        ),
        '{"1": 1, "2": 1, "10": 1}',
        (
            '{"minimum_pixels": 307200, '
            '"maximum_pixels": 2073600, '
            '"average_pixels": 1190400.0, '
            '"minimum_megapixels": 0.3072, '
            '"maximum_megapixels": 2.0736, '
            '"average_megapixels": 1.1904}'
        ),
        (
            '{"minimum": 0.5, "maximum": 2.0, '
            '"average": 1.166667, "orientation_counts": '
            '{"landscape": 1, "portrait": 1, "square": 1}}'
        ),
        (
            '{"minimum_bytes": 256, "maximum_bytes": 1024, '
            '"average_bytes": 682.67}'
        ),
        (
            '{"group_count": 1, "file_count": 3, '
            '"redundant_copy_count": 2, "groups": '
            '[{"sha256": "'
            + "a" * 64
            + '", "image_paths": ["data/copy-a.jpg", '
            '"data/copy-b.jpg", "data/copy-c.jpg"]}]}'
        ),
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
        lambda dataset_path, *, recursive=False, min_width=32,
        min_height=32: create_statistics(),
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

    assert dataset_summary.main() == 0

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
        lambda dataset_path, *, recursive=False, min_width=32,
        min_height=32: create_statistics(),
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

    assert dataset_summary.main() == 0

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
        lambda dataset_path, *, recursive=False, min_width=32,
        min_height=32: stats,
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

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()

    assert "# Dataset Summary" in captured.out
    assert "## Overview" in captured.out
    assert "| Total Images | 4 |" in captured.out
    assert "| JPEG | 2 |" in captured.out
    assert "## Image Channels" in captured.out
    assert "## Image Resolution" in captured.out
    assert "## Image Aspect Ratios" in captured.out
    assert "## Image File Sizes" in captured.out
    assert "## Exact Duplicate Images" in captured.out
    assert "| Minimum | 307,200 | 0.31 |" in captured.out
    assert "| Maximum | 2,073,600 | 2.07 |" in captured.out
    assert "| Average | 1,190,400.00 | 1.19 |" in captured.out
    assert "| Minimum Ratio | 0.50 |" in captured.out
    assert "| Maximum Ratio | 2.00 |" in captured.out
    assert "| Average Ratio | 1.17 |" in captured.out
    assert "| Landscape Images | 1 |" in captured.out
    assert "| Portrait Images | 1 |" in captured.out
    assert "| Square Images | 1 |" in captured.out
    assert "| Minimum | 256 |" in captured.out
    assert "| Maximum | 1,024 |" in captured.out
    assert "| Average | 682.67 |" in captured.out
    assert "| Duplicate Groups | 1 |" in captured.out
    assert f"**SHA-256:** `{'a' * 64}`" in captured.out
    assert "## Invalid Image Diagnostics" in captured.out
    assert "### `data/a-corrupt.jpg`" in captured.out
    assert "| Average | 906.67 |" in captured.out
    assert "| Size | 2.00 KB |" in captured.out


def test_main_reports_missing_dataset_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a missing dataset path without a traceback."""

    dataset_path = tmp_path / "missing"
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", str(dataset_path)],
    )

    assert dataset_summary.main() == 1

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == (
        f"Error: dataset path does not exist: {dataset_path}\n"
    )
    assert "Traceback" not in captured.err


def test_main_reports_dataset_path_that_is_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report when the dataset path is not a directory."""

    dataset_path = tmp_path / "image.jpg"
    dataset_path.write_bytes(b"not an image")
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", str(dataset_path)],
    )

    assert dataset_summary.main() == 1

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == (
        f"Error: dataset path is not a directory: {dataset_path}\n"
    )
    assert "Traceback" not in captured.err


def test_main_reports_missing_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a missing output parent without creating it."""

    output_directory = tmp_path / "reports"
    output_path = output_directory / "summary.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )

    assert dataset_summary.main() == 1

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == (
        "Error: output directory does not exist: "
        f"{output_directory}\n"
    )
    assert "Traceback" not in captured.err
    assert not output_directory.exists()
    assert not output_path.exists()


def test_main_reports_output_path_that_is_a_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report when the output path is an existing directory."""

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    output_path = tmp_path / "reports"
    output_path.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(dataset_path),
            "--output",
            str(output_path),
        ],
    )

    assert dataset_summary.main() == 1

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == (
        f"Error: output path is not a file: {output_path}\n"
    )
    assert "Traceback" not in captured.err


def test_main_reports_output_permission_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report an expected output permission error."""

    output_path = tmp_path / "summary.txt"

    def deny_write(
        path: Path,
        data: str,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "write_text", deny_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )

    assert dataset_summary.main() == 1

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == (
        "Error: could not write output file "
        f"{output_path}: permission denied\n"
    )
    assert "Traceback" not in captured.err
    assert not output_path.exists()


def test_main_reports_dataset_filesystem_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report an expected dataset filesystem error."""

    def deny_dataset_access(
        dataset_path: Path,
        *,
        recursive: bool = False,
        min_width: int = 32,
        min_height: int = 32,
    ) -> DatasetStatistics:
        raise PermissionError("permission denied")

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        deny_dataset_access,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", str(tmp_path)],
    )

    assert dataset_summary.main() == 1

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == (
        "Error: could not read dataset path "
        f"{tmp_path}: permission denied\n"
    )
    assert "Traceback" not in captured.err


def test_main_empty_dataset_remains_successful(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Treat a valid empty dataset as a successful report."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", str(tmp_path)],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()

    assert "No supported image files found." in captured.out
    assert "No invalid images found." in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["dataset-summary"],
        [
            "dataset-summary",
            "data/sample_dataset",
            "--format",
            "xml",
        ],
    ],
)
def test_main_preserves_argparse_usage_errors(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve argparse status 2 for command-line usage errors."""

    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as error:
        dataset_summary.main()

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert captured.out == ""
    assert "usage:" in captured.err
    assert "Traceback" not in captured.err


def test_main_does_not_hide_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Allow unexpected programming errors to surface."""

    def raise_unexpected_error(
        dataset_path: Path,
        *,
        recursive: bool = False,
        min_width: int = 32,
        min_height: int = 32,
    ) -> DatasetStatistics:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        raise_unexpected_error,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", str(tmp_path)],
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        dataset_summary.main()


def test_main_default_scanning_excludes_nested_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep top-level-only scanning as the CLI default."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    create_test_image(tmp_path / "root.jpg")
    create_test_image(nested_directory / "nested.png")
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

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["total_images"] == 1
    assert payload["valid_images"] == 1
    assert payload["extension_counts"] == {"jpeg": 1}
    assert payload["channel_counts"] == {"3": 1}
    assert payload["resolution_statistics"] == {
        "minimum_pixels": 10_000,
        "maximum_pixels": 10_000,
        "average_pixels": 10_000.0,
        "minimum_megapixels": 0.01,
        "maximum_megapixels": 0.01,
        "average_megapixels": 0.01,
    }
    assert captured.err == ""


def test_main_recursive_scanning_includes_nested_valid_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Include supported images across nested directory levels."""

    first_level = tmp_path / "first"
    second_level = first_level / "second"
    empty_directory = tmp_path / "empty"
    second_level.mkdir(parents=True)
    empty_directory.mkdir()
    create_test_image(tmp_path / "root.jpg")
    create_test_image(
        first_level / "nested.png",
        width=120,
        height=80,
    )
    create_test_image(
        second_level / "deep.webp",
        width=140,
        height=60,
    )
    (second_level / "notes.txt").write_text(
        "unsupported",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--recursive",
            "--format",
            "json",
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["total_images"] == 3
    assert payload["valid_images"] == 3
    assert payload["invalid_images"] == 0
    assert payload["extension_counts"] == {
        "jpeg": 1,
        "png": 1,
        "webp": 1,
    }
    assert payload["channel_counts"] == {"3": 3}
    assert payload["width"] == {
        "minimum": 100,
        "maximum": 140,
        "average": 120.0,
    }
    assert payload["height"] == {
        "minimum": 60,
        "maximum": 100,
        "average": 80.0,
    }
    assert payload["resolution_statistics"] == {
        "minimum_pixels": 8_400,
        "maximum_pixels": 10_000,
        "average_pixels": 28_000 / 3,
        "minimum_megapixels": 0.0084,
        "maximum_megapixels": 0.01,
        "average_megapixels": 0.009333,
    }
    assert payload["total_size_bytes"] > 0
    assert captured.err == ""


def test_main_recursive_scanning_reports_nested_invalid_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Include nested corrupt images in successful diagnostics."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    invalid_image = nested_directory / "broken.jpg"
    invalid_image.write_bytes(b"not a decodable image")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--recursive",
            "--format",
            "json",
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["total_images"] == 1
    assert payload["valid_images"] == 0
    assert payload["invalid_images"] == 1
    assert payload["extension_counts"] == {"jpeg": 1}
    assert payload["channel_counts"] == {}
    assert payload["resolution_statistics"] == {
        "minimum_pixels": 0,
        "maximum_pixels": 0,
        "average_pixels": 0.0,
        "minimum_megapixels": 0.0,
        "maximum_megapixels": 0.0,
        "average_megapixels": 0.0,
    }
    assert payload["invalid_image_diagnostics"] == [
        {
            "image_path": invalid_image.as_posix(),
            "errors": ["Image could not be decoded."],
        }
    ]
    assert captured.err == ""


@pytest.mark.parametrize(
    ("arguments", "expected_recursive"),
    [
        ([], False),
        (["--recursive"], True),
    ],
)
def test_main_forwards_recursive_flag_to_analyzer(
    arguments: list[str],
    expected_recursive: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forward the public recursive option to the existing analyzer."""

    received_recursive: list[bool] = []

    def record_recursive(
        dataset_path: Path,
        *,
        recursive: bool = False,
        min_width: int = 32,
        min_height: int = 32,
    ) -> DatasetStatistics:
        received_recursive.append(recursive)
        return create_statistics()

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        record_recursive,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
            *arguments,
        ],
    )

    assert dataset_summary.main() == 0
    assert received_recursive == [expected_recursive]
    assert capsys.readouterr().err == ""


def test_main_recursive_scanning_writes_output_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write recursive JSON results through the existing output path."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    create_test_image(tmp_path / "root.jpg")
    create_test_image(nested_directory / "nested.png")
    output_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--recursive",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["total_images"] == 2
    assert payload["extension_counts"] == {
        "jpeg": 1,
        "png": 1,
    }
    assert payload["resolution_statistics"] == {
        "minimum_pixels": 10_000,
        "maximum_pixels": 10_000,
        "average_pixels": 10_000.0,
        "minimum_megapixels": 0.01,
        "maximum_megapixels": 0.01,
        "average_megapixels": 0.01,
    }
    assert captured.out == ""
    assert captured.err == ""


def test_main_help_lists_recursive_option(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Advertise recursive scanning in argparse help."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", "--help"],
    )

    with pytest.raises(SystemExit) as error:
        dataset_summary.main()

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "--recursive" in captured.out
    assert "--min-width PIXELS" in captured.out
    assert "--min-height PIXELS" in captured.out
    assert "--manifest-output PATH" in captured.out
    assert captured.err == ""


def test_main_rejects_recursive_option_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject values supplied to the boolean recursive flag."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
            "--recursive",
            "true",
        ],
    )

    with pytest.raises(SystemExit) as error:
        dataset_summary.main()

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert captured.out == ""
    assert "unrecognized arguments: true" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("arguments", "expected_thresholds"),
    [
        ([], (32, 32)),
        (["--min-width", "64"], (64, 32)),
        (["--min-height", "48"], (32, 48)),
        (
            [
                "--min-width",
                "64",
                "--min-height",
                "48",
            ],
            (64, 48),
        ),
    ],
)
def test_main_forwards_validation_thresholds(
    arguments: list[str],
    expected_thresholds: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forward independent threshold values to the analyzer."""

    received_thresholds: list[tuple[int, int]] = []

    def record_thresholds(
        dataset_path: Path,
        *,
        recursive: bool = False,
        min_width: int = 32,
        min_height: int = 32,
    ) -> DatasetStatistics:
        received_thresholds.append((min_width, min_height))
        return create_statistics()

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        record_thresholds,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
            *arguments,
        ],
    )

    assert dataset_summary.main() == 0
    assert received_thresholds == [expected_thresholds]
    assert capsys.readouterr().err == ""


def test_main_default_thresholds_reject_small_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep a 20-by-20 image invalid under CLI defaults."""

    create_test_image(
        tmp_path / "small.png",
        width=20,
        height=20,
    )
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

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["valid_images"] == 0
    assert payload["invalid_images"] == 1
    assert payload["format_statistics"]["png"] == {
        "total_images": 1,
        "valid_images": 0,
        "invalid_images": 1,
        "total_valid_size_bytes": 0,
        "average_valid_size_bytes": 0.0,
    }
    assert payload["channel_counts"] == {}
    assert payload["resolution_statistics"]["minimum_pixels"] == 0
    assert payload["resolution_statistics"]["maximum_pixels"] == 0
    assert payload["resolution_statistics"]["average_pixels"] == 0.0
    assert payload["invalid_image_diagnostics"][0]["errors"] == [
        "Width 20px is below minimum 32px.",
        "Height 20px is below minimum 32px.",
    ]
    assert captured.err == ""


def test_main_lower_thresholds_accept_small_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Allow a small image under deliberate lower CLI thresholds."""

    create_test_image(
        tmp_path / "small.png",
        width=20,
        height=20,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--min-width",
            "10",
            "--min-height",
            "10",
            "--format",
            "json",
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["valid_images"] == 1
    assert payload["invalid_images"] == 0
    image_size = (tmp_path / "small.png").stat().st_size
    assert payload["format_statistics"]["png"] == {
        "total_images": 1,
        "valid_images": 1,
        "invalid_images": 0,
        "total_valid_size_bytes": image_size,
        "average_valid_size_bytes": float(image_size),
    }
    assert payload["channel_counts"] == {"3": 1}
    assert payload["resolution_statistics"] == {
        "minimum_pixels": 400,
        "maximum_pixels": 400,
        "average_pixels": 400.0,
        "minimum_megapixels": 0.0004,
        "maximum_megapixels": 0.0004,
        "average_megapixels": 0.0004,
    }
    assert payload["invalid_image_diagnostics"] == []
    assert captured.err == ""


def test_main_higher_width_produces_invalid_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Treat a custom validation failure as a successful result."""

    create_test_image(
        tmp_path / "image.png",
        width=50,
        height=60,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--min-width",
            "100",
            "--min-height",
            "40",
            "--format",
            "json",
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["valid_images"] == 0
    assert payload["invalid_images"] == 1
    assert payload["channel_counts"] == {}
    assert payload["resolution_statistics"]["minimum_pixels"] == 0
    assert payload["resolution_statistics"]["maximum_pixels"] == 0
    assert payload["resolution_statistics"]["average_pixels"] == 0.0
    assert payload["invalid_image_diagnostics"][0]["errors"] == [
        "Width 50px is below minimum 100px.",
    ]
    assert captured.err == ""


def test_main_custom_thresholds_apply_recursively(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Apply custom thresholds to nested images."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    create_test_image(
        nested_directory / "small.png",
        width=20,
        height=20,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--recursive",
            "--min-width",
            "10",
            "--min-height",
            "10",
            "--format",
            "json",
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["total_images"] == 1
    assert payload["valid_images"] == 1
    assert payload["invalid_images"] == 0
    assert payload["channel_counts"] == {"3": 1}
    assert payload["resolution_statistics"]["minimum_pixels"] == 400
    assert payload["resolution_statistics"]["maximum_pixels"] == 400
    assert payload["resolution_statistics"]["average_pixels"] == 400.0
    assert captured.err == ""


def test_main_custom_thresholds_write_output_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write custom-threshold results through the existing output path."""

    create_test_image(
        tmp_path / "small.png",
        width=20,
        height=20,
    )
    output_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--min-width",
            "10",
            "--min-height",
            "10",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["valid_images"] == 1
    assert payload["invalid_images"] == 0
    assert payload["resolution_statistics"]["minimum_pixels"] == 400
    assert payload["resolution_statistics"]["maximum_pixels"] == 400
    assert payload["resolution_statistics"]["average_pixels"] == 400.0
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["--min-width", "0"],
        ["--min-height", "0"],
        ["--min-width", "-1"],
        ["--min-height", "-1"],
        ["--min-width", "abc"],
        ["--min-height", "12.5"],
    ],
)
def test_main_rejects_invalid_threshold_before_analysis(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep invalid thresholds as argparse status-2 failures."""

    def fail_if_called(
        dataset_path: Path,
        *,
        recursive: bool = False,
        min_width: int = 32,
        min_height: int = 32,
    ) -> DatasetStatistics:
        raise AssertionError("analyzer must not be called")

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        fail_if_called,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
            *arguments,
        ],
    )

    with pytest.raises(SystemExit) as error:
        dataset_summary.main()

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert captured.out == ""
    assert "must be a positive integer" in captured.err
    assert "Traceback" not in captured.err


def test_main_without_manifest_uses_existing_analyzer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the aggregate-only CLI on the existing public analyzer API."""

    analyzer_calls: list[Path] = []

    def record_analysis(
        dataset_path: Path,
        *,
        recursive: bool = False,
        min_width: int = 32,
        min_height: int = 32,
    ) -> DatasetStatistics:
        analyzer_calls.append(dataset_path)
        return create_statistics()

    def unexpected_manifest_analysis(*args, **kwargs):
        raise AssertionError("manifest analyzer must not be called")

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        record_analysis,
    )
    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset_with_manifest",
        unexpected_manifest_analysis,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-summary", "data/sample_dataset"],
    )

    assert dataset_summary.main() == 0
    assert analyzer_calls == [Path("data/sample_dataset")]
    assert "Dataset Summary" in capsys.readouterr().out


def test_main_with_manifest_uses_combined_analyzer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use one combined analysis when manifest output is requested."""

    manifest_path = tmp_path / "manifest.jsonl"
    analyzer_calls: list[tuple[bool, int, int]] = []

    def record_manifest_analysis(
        dataset_path: Path,
        *,
        recursive: bool = False,
        min_width: int = 32,
        min_height: int = 32,
    ) -> DatasetAnalysisResult:
        analyzer_calls.append((recursive, min_width, min_height))
        return DatasetAnalysisResult(
            statistics=create_statistics(),
            manifest_entries=(
                DatasetManifestEntry(
                    path=Path("coral.jpg"),
                    extension="jpeg",
                    is_valid=True,
                    validation_errors=(),
                ),
            ),
        )

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError("aggregate-only analyzer must not be called")

    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset",
        unexpected_analysis,
    )
    monkeypatch.setattr(
        dataset_summary,
        "analyze_dataset_with_manifest",
        record_manifest_analysis,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            "data/sample_dataset",
            "--recursive",
            "--min-width",
            "64",
            "--min-height",
            "48",
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    record = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert analyzer_calls == [(True, 64, 48)]
    assert record["path"] == "coral.jpg"
    assert "Dataset Summary" in captured.out
    assert captured.err == ""


def test_main_manifest_includes_invalid_and_omits_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write supported candidates and keep printing the aggregate report."""

    create_test_image(tmp_path / "coral.jpg", width=80, height=60)
    (tmp_path / "broken.png").write_bytes(b"not an image")
    (tmp_path / "notes.txt").write_text("unsupported", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["path"] for record in records] == [
        "broken.png",
        "coral.jpg",
    ]
    assert records[0]["is_valid"] is False
    assert records[0]["validation_errors"] == [
        "Image could not be decoded."
    ]
    assert records[0]["width"] is None
    assert records[1]["is_valid"] is True
    assert records[1]["width"] == 80
    assert records[1]["height"] == 60
    assert "Dataset Summary" in captured.out
    assert captured.err == ""


def test_main_writes_recursive_manifest_and_json_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write both outputs and expose recursive duplicate membership."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    source_path = tmp_path / "source.png"
    nested_copy = nested_directory / "copy.png"
    create_test_image(source_path)
    copyfile(source_path, nested_copy)
    (nested_directory / "broken.jpg").write_bytes(b"not an image")
    report_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--recursive",
            "--format",
            "json",
            "--output",
            str(report_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    records_by_path = {
        record["path"]: record
        for record in records
    }
    digest = calculate_sha256(source_path)

    assert report["total_images"] == 3
    assert report["valid_images"] == 2
    assert report["invalid_images"] == 1
    source_size = source_path.stat().st_size
    assert report["format_statistics"] == {
        "jpeg": {
            "total_images": 1,
            "valid_images": 0,
            "invalid_images": 1,
            "total_valid_size_bytes": 0,
            "average_valid_size_bytes": 0.0,
        },
        "png": {
            "total_images": 2,
            "valid_images": 2,
            "invalid_images": 0,
            "total_valid_size_bytes": source_size * 2,
            "average_valid_size_bytes": float(source_size),
        },
    }
    assert report["duplicate_images"]["group_count"] == 1
    assert report["aspect_ratio_statistics"] == {
        "minimum": 1.0,
        "maximum": 1.0,
        "average": 1.0,
        "orientation_counts": {
            "landscape": 0,
            "portrait": 0,
            "square": 2,
        },
    }
    assert report["file_size_statistics"] == {
        "minimum_bytes": source_size,
        "maximum_bytes": source_size,
        "average_bytes": float(source_size),
    }
    assert report["total_size_bytes"] == source_size * 2
    assert records_by_path["source.png"][
        "duplicate_group_sha256"
    ] == digest
    assert records_by_path["nested/copy.png"][
        "duplicate_group_sha256"
    ] == digest
    assert records_by_path["nested/broken.jpg"]["is_valid"] is False
    assert all(len(record) == 11 for record in records)
    assert all("format_statistics" not in record for record in records)
    assert captured.out == ""
    assert captured.err == ""


def test_main_manifest_respects_custom_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Apply CLI thresholds before manifest metadata is populated."""

    create_test_image(tmp_path / "image.png", width=50, height=60)
    manifest_path = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--min-width",
            "100",
            "--min-height",
            "40",
            "--format",
            "json",
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert report["valid_images"] == 0
    assert report["invalid_images"] == 1
    assert report["aspect_ratio_statistics"] == {
        "minimum": 0.0,
        "maximum": 0.0,
        "average": 0.0,
        "orientation_counts": {
            "landscape": 0,
            "portrait": 0,
            "square": 0,
        },
    }
    assert report["file_size_statistics"] == {
        "minimum_bytes": 0,
        "maximum_bytes": 0,
        "average_bytes": 0.0,
    }
    assert report["total_size_bytes"] == 0
    assert record["is_valid"] is False
    assert record["validation_errors"] == [
        "Width 50px is below minimum 100px."
    ]
    assert record["width"] is None
    assert captured.err == ""


def test_main_empty_dataset_writes_empty_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep empty datasets successful with a zero-byte manifest."""

    manifest_path = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert dataset_summary.main() == 0

    captured = capsys.readouterr()
    assert manifest_path.read_bytes() == b""
    assert "No supported image files found." in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("failure_kind", ["missing_parent", "directory"])
def test_main_reports_manifest_path_failures(
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report expected manifest path failures without aggregate output."""

    if failure_kind == "missing_parent":
        manifest_directory = tmp_path / "missing"
        manifest_path = manifest_directory / "manifest.jsonl"
        expected_error = (
            "Error: manifest directory does not exist: "
            f"{manifest_directory}\n"
        )
    else:
        manifest_path = tmp_path / "manifest"
        manifest_path.mkdir()
        expected_error = (
            f"Error: manifest path is not a file: {manifest_path}\n"
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert dataset_summary.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == expected_error
    assert "Traceback" not in captured.err


def test_main_reports_manifest_permission_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Translate manifest permission errors and suppress aggregate output."""

    manifest_path = tmp_path / "manifest.jsonl"

    def deny_write(
        path: Path,
        data: str,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "write_text", deny_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset-summary",
            str(tmp_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert dataset_summary.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Error: could not write manifest file "
        f"{manifest_path}: permission denied\n"
    )
    assert "Traceback" not in captured.err
