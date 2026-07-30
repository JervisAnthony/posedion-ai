import csv
import io
import json
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_csv import (
    format_dataset_csv,
)
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
    DuplicateImageGroup,
    ImageFormatStatistics,
    InvalidImageDiagnostic,
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
    assert values[0] == str(Path("data/sample_dataset"))
    assert values[1] == "4"
    assert values[2] == "3"
    assert values[3] == "1"
    assert json.loads(values[4]) == {
        "jpeg": 2,
        "png": 1,
        "webp": 1,
    }
    assert json.loads(values[5]) == {
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
    assert list(json.loads(values[5])) == ["jpeg", "png", "webp"]
    assert list(json.loads(values[5])["jpeg"]) == [
        "total_images",
        "valid_images",
        "invalid_images",
        "total_valid_size_bytes",
        "average_valid_size_bytes",
    ]
    assert all(
        isinstance(json.loads(values[5])["jpeg"][key], int)
        for key in (
            "total_images",
            "valid_images",
            "invalid_images",
            "total_valid_size_bytes",
        )
    )
    assert isinstance(
        json.loads(values[5])["jpeg"]["average_valid_size_bytes"],
        float,
    )
    assert json.loads(values[6]) == {
        "1": 1,
        "2": 1,
        "10": 1,
    }
    assert list(json.loads(values[6])) == ["1", "2", "10"]
    assert json.loads(values[7]) == {
        "minimum_pixels": 307_200,
        "maximum_pixels": 2_073_600,
        "average_pixels": 1_190_400.0,
        "minimum_megapixels": 0.3072,
        "maximum_megapixels": 2.0736,
        "average_megapixels": 1.1904,
    }
    assert json.loads(values[8]) == {
        "minimum": 0.5,
        "maximum": 2.0,
        "average": 1.166667,
        "orientation_counts": {
            "landscape": 1,
            "portrait": 1,
            "square": 1,
        },
    }
    assert list(
        json.loads(values[8])["orientation_counts"]
    ) == ["landscape", "portrait", "square"]
    assert all(
        isinstance(json.loads(values[8])[key], float)
        for key in ("minimum", "maximum", "average")
    )
    assert json.loads(values[9]) == {
        "minimum_bytes": 256,
        "maximum_bytes": 1024,
        "average_bytes": 682.67,
    }
    assert list(json.loads(values[9])) == [
        "minimum_bytes",
        "maximum_bytes",
        "average_bytes",
    ]
    assert isinstance(json.loads(values[9])["minimum_bytes"], int)
    assert isinstance(json.loads(values[9])["maximum_bytes"], int)
    assert isinstance(json.loads(values[9])["average_bytes"], float)
    assert json.loads(values[10]) == {
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
    assert json.loads(values[11]) == [
        {
            "image_path": "data/a-corrupt.jpg",
            "errors": ["Image could not be decoded."],
        }
    ]
    assert values[12] == "640"
    assert values[13] == "1280"
    assert values[14] == "906.67"
    assert values[15] == "480"
    assert values[16] == "720"
    assert values[17] == "600.00"
    assert values[18] == "2048"


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


def test_format_dataset_csv_rounds_format_average() -> None:
    """Round the numeric per-format average inside one JSON cell."""

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

    rows = list(
        csv.reader(
            io.StringIO(
                format_dataset_csv(
                    Path("data/sample_dataset"),
                    stats,
                )
            )
        )
    )
    average = json.loads(rows[1][5])["jpeg"][
        "average_valid_size_bytes"
    ]

    assert average == 0.33
    assert isinstance(average, float)


def test_format_dataset_csv_escapes_and_sorts_diagnostics() -> None:
    """Preserve quoted, comma-separated, and multiple diagnostic errors."""

    stats = create_statistics()
    stats.total_images = 5
    stats.invalid_images = 2
    stats.extension_counts["png"] = 2
    stats.invalid_image_diagnostics = [
        InvalidImageDiagnostic(
            image_path=Path("data/z-small.png"),
            errors=(
                'Width is "10px", below minimum.',
                "Height 10px is below minimum 32px.",
            ),
        ),
        *stats.invalid_image_diagnostics,
    ]

    result = format_dataset_csv(
        Path("data/sample_dataset"),
        stats,
    )
    rows = list(csv.reader(io.StringIO(result)))

    assert len(rows) == 2
    assert json.loads(rows[1][11]) == [
        {
            "image_path": "data/a-corrupt.jpg",
            "errors": ["Image could not be decoded."],
        },
        {
            "image_path": "data/z-small.png",
            "errors": [
                'Width is "10px", below minimum.',
                "Height 10px is below minimum 32px.",
            ],
        },
    ]


def test_format_dataset_csv_with_no_images() -> None:
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

    result = format_dataset_csv(
        Path("data/sample_dataset"),
        stats,
    )

    rows = list(csv.reader(io.StringIO(result)))

    assert json.loads(rows[1][4]) == {}
    assert json.loads(rows[1][5]) == {}
    assert json.loads(rows[1][6]) == {}
    assert json.loads(rows[1][7]) == {
        "minimum_pixels": 0,
        "maximum_pixels": 0,
        "average_pixels": 0.0,
        "minimum_megapixels": 0.0,
        "maximum_megapixels": 0.0,
        "average_megapixels": 0.0,
    }
    assert json.loads(rows[1][8]) == {
        "minimum": 0.0,
        "maximum": 0.0,
        "average": 0.0,
        "orientation_counts": {
            "landscape": 0,
            "portrait": 0,
            "square": 0,
        },
    }
    assert json.loads(rows[1][9]) == {
        "minimum_bytes": 0,
        "maximum_bytes": 0,
        "average_bytes": 0.0,
    }
    assert json.loads(rows[1][10]) == {
        "group_count": 0,
        "file_count": 0,
        "redundant_copy_count": 0,
        "groups": [],
    }
    assert json.loads(rows[1][11]) == []


def test_format_dataset_csv_fills_missing_orientation_categories() -> None:
    """Serialize every supported orientation in stable order."""

    stats = create_statistics()
    stats.orientation_counts = {"portrait": 3}

    rows = list(
        csv.reader(
            io.StringIO(
                format_dataset_csv(
                    Path("data/sample_dataset"),
                    stats,
                )
            )
        )
    )
    orientation_counts = json.loads(rows[1][8])[
        "orientation_counts"
    ]

    assert orientation_counts == {
        "landscape": 0,
        "portrait": 3,
        "square": 0,
    }
    assert list(orientation_counts) == [
        "landscape",
        "portrait",
        "square",
    ]
