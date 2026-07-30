import cv2
import numpy as np
import pytest

from pathlib import Path

import poseidon_ai.nautilus_vision.dataset_analyzer as dataset_analyzer
from poseidon_ai.nautilus_vision.dataset_analyzer import analyze_dataset
from poseidon_ai.nautilus_vision.image_metadata import get_image_metadata

def create_test_image(
    path: Path,
    *,
    width: int = 100,
    height: int = 100,
) -> None:
    """Create a simple white test image."""

    image = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    success = cv2.imwrite(str(path), image)

    assert success


def test_analyze_empty_dataset(tmp_path: Path) -> None:
    """An empty dataset should contain zero statistics."""

    stats = analyze_dataset(tmp_path)

    assert stats.total_images == 0
    assert stats.valid_images == 0
    assert stats.invalid_images == 0
    assert stats.extension_counts == {}
    assert stats.channel_counts == {}
    assert stats.min_pixel_count == 0
    assert stats.max_pixel_count == 0
    assert stats.average_pixel_count == 0.0

def test_analyze_single_valid_image(
    tmp_path: Path,
) -> None:
    """Analyze a dataset containing one valid image."""

    image_path = tmp_path / "fish.jpg"

    create_test_image(image_path)

    stats = analyze_dataset(tmp_path)

    assert stats.total_images == 1
    assert stats.valid_images == 1
    assert stats.invalid_images == 0
    assert stats.extension_counts == {"jpeg": 1}
    assert stats.channel_counts == {3: 1}
    assert sum(stats.channel_counts.values()) == stats.valid_images
    assert sum(stats.extension_counts.values()) == stats.total_images

    assert stats.min_width == 100
    assert stats.max_width == 100
    assert stats.average_width == 100

    assert stats.min_height == 100
    assert stats.max_height == 100
    assert stats.average_height == 100

    assert stats.min_pixel_count == 10_000
    assert stats.max_pixel_count == 10_000
    assert stats.average_pixel_count == 10_000

    assert stats.total_size_bytes > 0

def test_analyze_mixed_valid_and_invalid_images(
    tmp_path: Path,
) -> None:
    """Analyze a dataset containing valid and invalid images."""

    create_test_image(tmp_path / "fish.jpg")

    (tmp_path / "broken.jpg").write_bytes(
        b"This is not a valid image."
    )

    stats = analyze_dataset(tmp_path)

    assert stats.total_images == 2
    assert stats.valid_images == 1
    assert stats.invalid_images == 1
    assert stats.extension_counts == {"jpeg": 2}
    assert stats.channel_counts == {3: 1}
    assert sum(stats.channel_counts.values()) == stats.valid_images
    assert sum(stats.extension_counts.values()) == stats.total_images

    assert stats.min_pixel_count == 10_000
    assert stats.max_pixel_count == 10_000
    assert stats.average_pixel_count == 10_000
    assert stats.total_size_bytes > 0

def test_analyze_ignores_unsupported_files(
    tmp_path: Path,
) -> None:
    """Unsupported files should be ignored."""

    create_test_image(tmp_path / "fish.jpg")

    (tmp_path / "notes.txt").write_text("Hello")

    (tmp_path / "readme.md").write_text("# Dataset")

    stats = analyze_dataset(tmp_path)

    assert stats.total_images == 1
    assert stats.valid_images == 1
    assert stats.invalid_images == 0
    assert stats.extension_counts == {"jpeg": 1}


def test_analyze_aggregates_matching_channel_counts(
    tmp_path: Path,
) -> None:
    """Aggregate multiple valid images with the decoded channel count."""

    create_test_image(tmp_path / "first.jpg")
    create_test_image(tmp_path / "second.png")

    stats = analyze_dataset(tmp_path)

    assert stats.channel_counts == {3: 2}
    assert sum(stats.channel_counts.values()) == stats.valid_images


def test_analyze_sorts_distinct_channel_counts_numerically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sort metadata-provided channel values as integers."""

    first_path = tmp_path / "two.png"
    second_path = tmp_path / "ten.png"
    create_test_image(first_path)
    create_test_image(second_path)

    def metadata_with_distinct_channels(image_path: Path) -> dict:
        metadata = get_image_metadata(image_path)
        metadata["channels"] = {
            "two.png": 2,
            "ten.png": 10,
        }[image_path.name]
        return metadata

    monkeypatch.setattr(
        dataset_analyzer,
        "get_image_metadata",
        metadata_with_distinct_channels,
    )

    stats = analyze_dataset(tmp_path)

    assert stats.channel_counts == {2: 1, 10: 1}
    assert list(stats.channel_counts) == [2, 10]
    assert sum(stats.channel_counts.values()) == stats.valid_images


def test_analyze_recursive_images_contribute_channel_counts(
    tmp_path: Path,
) -> None:
    """Include metadata channels from nested valid images when enabled."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    create_test_image(nested_directory / "fish.png")

    stats = analyze_dataset(tmp_path, recursive=True)

    assert stats.channel_counts == {3: 1}
    assert sum(stats.channel_counts.values()) == stats.valid_images
    assert stats.min_pixel_count == 10_000
    assert stats.max_pixel_count == 10_000
    assert stats.average_pixel_count == 10_000


def test_analyze_uses_actual_per_image_pixel_areas(
    tmp_path: Path,
) -> None:
    """Calculate extrema from each image area, not dimension extrema."""

    create_test_image(
        tmp_path / "wide.png",
        width=100,
        height=40,
    )
    create_test_image(
        tmp_path / "tall.png",
        width=50,
        height=200,
    )

    stats = analyze_dataset(tmp_path)

    assert stats.min_width == 50
    assert stats.max_width == 100
    assert stats.min_height == 40
    assert stats.max_height == 200
    assert stats.min_pixel_count == 4_000
    assert stats.max_pixel_count == 10_000
    assert stats.average_pixel_count == 7_000
    assert (
        stats.min_pixel_count
        <= stats.average_pixel_count
        <= stats.max_pixel_count
    )
    assert stats.channel_counts == {3: 2}
    assert stats.valid_images == 2


def test_analyze_requests_metadata_once_per_valid_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not request metadata again for resolution statistics."""

    image_paths = [
        tmp_path / "first.png",
        tmp_path / "second.png",
    ]
    for image_path in image_paths:
        create_test_image(image_path)

    metadata_calls: list[Path] = []

    def tracked_metadata(image_path: Path) -> dict:
        metadata_calls.append(image_path)
        return get_image_metadata(image_path)

    monkeypatch.setattr(
        dataset_analyzer,
        "get_image_metadata",
        tracked_metadata,
    )

    stats = analyze_dataset(tmp_path)

    assert metadata_calls == image_paths
    assert stats.min_pixel_count == 10_000
    assert stats.max_pixel_count == 10_000
    assert stats.average_pixel_count == 10_000


def test_analyze_counts_mixed_supported_extensions(
    tmp_path: Path,
) -> None:
    """Group discovered supported files by normalized extension."""

    for filename in (
        "coral.jpg",
        "turtle.jpeg",
        "reef.PNG",
        "diver.webp",
    ):
        (tmp_path / filename).write_bytes(b"invalid image")

    stats = analyze_dataset(tmp_path)

    assert stats.extension_counts == {
        "jpeg": 2,
        "png": 1,
        "webp": 1,
    }
    assert stats.invalid_images == 4
    assert sum(stats.extension_counts.values()) == stats.total_images


def test_analyze_normalizes_extension_case_and_aliases(
    tmp_path: Path,
) -> None:
    """Normalize extension case and supported JPEG and TIFF aliases."""

    for filename in (
        "one.JPG",
        "two.JpEg",
        "three.png",
        "four.PNG",
        "five.tif",
        "six.TIFF",
    ):
        (tmp_path / filename).write_bytes(b"invalid image")

    stats = analyze_dataset(tmp_path)

    assert stats.extension_counts == {
        "jpeg": 2,
        "png": 2,
        "tiff": 2,
    }
    assert list(stats.extension_counts) == ["jpeg", "png", "tiff"]
    assert sum(stats.extension_counts.values()) == stats.total_images

def test_analyze_dataset_captures_invalid_image_diagnostics(
    tmp_path: Path,
) -> None:
    """Capture the path and validation errors for invalid images."""

    invalid_path = tmp_path / "corrupt.jpg"
    invalid_path.write_bytes(b"not a valid image")

    stats = analyze_dataset(tmp_path)

    assert stats.total_images == 1
    assert stats.valid_images == 0
    assert stats.invalid_images == 1
    assert stats.min_pixel_count == 0
    assert stats.max_pixel_count == 0
    assert stats.average_pixel_count == 0.0

    assert len(stats.invalid_image_diagnostics) == 1

    diagnostic = stats.invalid_image_diagnostics[0]

    assert diagnostic.image_path == invalid_path
    assert diagnostic.errors == (
        "Image could not be decoded.",
    )

def test_analyze_dataset_preserves_multiple_validation_errors(
    tmp_path: Path,
) -> None:
    """Preserve every validation error associated with an image."""

    image_path = tmp_path / "small.png"

    image = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(image_path),
        image,
    )

    stats = analyze_dataset(tmp_path)

    assert stats.invalid_images == 1
    assert stats.channel_counts == {}
    assert stats.min_pixel_count == 0
    assert stats.max_pixel_count == 0
    assert stats.average_pixel_count == 0.0
    assert len(stats.invalid_image_diagnostics) == 1

    diagnostic = stats.invalid_image_diagnostics[0]

    assert diagnostic.image_path == image_path
    assert diagnostic.errors == (
        "Width 10px is below minimum 32px.",
        "Height 10px is below minimum 32px.",
    )
    assert (
        len(stats.invalid_image_diagnostics)
        == stats.invalid_images
    )


def test_analyze_dataset_keeps_default_validation_thresholds(
    tmp_path: Path,
) -> None:
    """Keep 32-by-32 validation when thresholds are omitted."""

    image_path = tmp_path / "small.png"
    create_test_image(image_path, width=20, height=20)

    stats = analyze_dataset(tmp_path)

    assert stats.valid_images == 0
    assert stats.invalid_images == 1
    assert stats.channel_counts == {}
    assert stats.min_pixel_count == 0
    assert stats.max_pixel_count == 0
    assert stats.average_pixel_count == 0.0
    assert stats.invalid_image_diagnostics[0].errors == (
        "Width 20px is below minimum 32px.",
        "Height 20px is below minimum 32px.",
    )


def test_analyze_dataset_accepts_lower_validation_thresholds(
    tmp_path: Path,
) -> None:
    """Allow a small image under explicit lower thresholds."""

    create_test_image(
        tmp_path / "small.png",
        width=20,
        height=20,
    )

    stats = analyze_dataset(
        tmp_path,
        min_width=10,
        min_height=10,
    )

    assert stats.total_images == 1
    assert stats.valid_images == 1
    assert stats.invalid_images == 0
    assert stats.channel_counts == {3: 1}
    assert sum(stats.channel_counts.values()) == stats.valid_images
    assert stats.min_pixel_count == 400
    assert stats.max_pixel_count == 400
    assert stats.average_pixel_count == 400
    assert stats.invalid_image_diagnostics == []


def test_analyze_dataset_applies_custom_width_independently(
    tmp_path: Path,
) -> None:
    """Preserve only the failing custom width diagnostic."""

    image_path = tmp_path / "image.png"
    create_test_image(image_path, width=50, height=60)

    stats = analyze_dataset(
        tmp_path,
        min_width=100,
        min_height=40,
    )

    assert stats.valid_images == 0
    assert stats.invalid_images == 1
    assert stats.channel_counts == {}
    assert stats.min_pixel_count == 0
    assert stats.max_pixel_count == 0
    assert stats.average_pixel_count == 0.0
    assert stats.invalid_image_diagnostics[0].image_path == image_path
    assert stats.invalid_image_diagnostics[0].errors == (
        "Width 50px is below minimum 100px.",
    )


def test_analyze_dataset_preserves_custom_error_order(
    tmp_path: Path,
) -> None:
    """Keep width before height for multiple custom failures."""

    create_test_image(
        tmp_path / "image.png",
        width=50,
        height=60,
    )

    stats = analyze_dataset(
        tmp_path,
        min_width=100,
        min_height=100,
    )

    assert stats.invalid_image_diagnostics[0].errors == (
        "Width 50px is below minimum 100px.",
        "Height 60px is below minimum 100px.",
    )
