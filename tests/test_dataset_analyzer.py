import cv2
import numpy as np
import pytest

from pathlib import Path
from shutil import copyfile

import poseidon_ai.nautilus_vision.dataset_analyzer as dataset_analyzer
from poseidon_ai.nautilus_vision.dataset_analyzer import analyze_dataset
from poseidon_ai.nautilus_vision.image_hash import calculate_sha256
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
    assert stats.duplicate_image_groups == []

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


def test_analyze_groups_three_byte_identical_valid_images(
    tmp_path: Path,
) -> None:
    """Create one deterministic group for three exact copies."""

    source_path = tmp_path / "z-source.png"
    first_copy = tmp_path / "A-copy.png"
    second_copy = tmp_path / "m-copy.png"
    create_test_image(source_path)
    copyfile(source_path, first_copy)
    copyfile(source_path, second_copy)

    stats = analyze_dataset(tmp_path)

    group = stats.duplicate_image_groups[0]
    assert stats.duplicate_group_count == 1
    assert stats.duplicate_file_count == 3
    assert stats.redundant_copy_count == 2
    assert group.sha256 == calculate_sha256(source_path)
    assert group.image_paths == (
        first_copy,
        second_copy,
        source_path,
    )
    assert all(
        len(duplicate_group.image_paths) >= 2
        for duplicate_group in stats.duplicate_image_groups
    )
    assert (
        stats.redundant_copy_count
        == stats.duplicate_file_count - stats.duplicate_group_count
    )
    assert stats.channel_counts == {3: 3}
    assert stats.min_pixel_count == 10_000
    assert stats.total_size_bytes == source_path.stat().st_size * 3


def test_analyze_does_not_group_matching_sizes_with_different_hashes(
    tmp_path: Path,
) -> None:
    """Treat file size as a candidate filter, not duplicate proof."""

    first_path = tmp_path / "first.bmp"
    second_path = tmp_path / "second.bmp"
    first_image = np.zeros((100, 100, 3), dtype=np.uint8)
    second_image = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(first_path), first_image)
    assert cv2.imwrite(str(second_path), second_image)
    assert first_path.stat().st_size == second_path.stat().st_size

    stats = analyze_dataset(tmp_path)

    assert stats.valid_images == 2
    assert stats.duplicate_image_groups == []
    assert stats.duplicate_group_count == 0


def test_analyze_never_hashes_invalid_identical_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude invalid supported files before candidate hashing."""

    content = b"identical but not decodable"
    (tmp_path / "first.jpg").write_bytes(content)
    (tmp_path / "second.jpg").write_bytes(content)

    def unexpected_hash(image_path: Path) -> str:
        raise AssertionError(f"invalid image was hashed: {image_path}")

    monkeypatch.setattr(
        dataset_analyzer,
        "calculate_sha256",
        unexpected_hash,
    )

    stats = analyze_dataset(tmp_path)

    assert stats.valid_images == 0
    assert stats.invalid_images == 2
    assert stats.duplicate_image_groups == []
    assert len(stats.invalid_image_diagnostics) == 2


def test_analyze_does_not_hash_unique_size_valid_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip hashing when no other valid file has the same size."""

    first_path = tmp_path / "first.bmp"
    second_path = tmp_path / "second.bmp"
    create_test_image(first_path, width=100, height=100)
    create_test_image(second_path, width=120, height=100)
    assert first_path.stat().st_size != second_path.stat().st_size
    hash_calls: list[Path] = []

    def tracked_hash(image_path: Path) -> str:
        hash_calls.append(image_path)
        return calculate_sha256(image_path)

    monkeypatch.setattr(
        dataset_analyzer,
        "calculate_sha256",
        tracked_hash,
    )

    stats = analyze_dataset(tmp_path)

    assert hash_calls == []
    assert stats.duplicate_image_groups == []


def test_analyze_hashes_each_candidate_at_most_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hash every same-size candidate exactly once."""

    source_path = tmp_path / "source.bmp"
    first_copy = tmp_path / "copy-a.bmp"
    second_copy = tmp_path / "copy-b.bmp"
    different_path = tmp_path / "different.bmp"
    create_test_image(source_path)
    copyfile(source_path, first_copy)
    copyfile(source_path, second_copy)
    different_image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(different_path), different_image)
    assert len(
        {
            path.stat().st_size
            for path in (
                source_path,
                first_copy,
                second_copy,
                different_path,
            )
        }
    ) == 1
    hash_calls: list[Path] = []

    def tracked_hash(image_path: Path) -> str:
        hash_calls.append(image_path)
        return calculate_sha256(image_path)

    monkeypatch.setattr(
        dataset_analyzer,
        "calculate_sha256",
        tracked_hash,
    )

    stats = analyze_dataset(tmp_path)

    assert sorted(hash_calls) == sorted(
        [
            source_path,
            first_copy,
            second_copy,
            different_path,
        ]
    )
    assert len(hash_calls) == len(set(hash_calls)) == 4
    assert stats.duplicate_group_count == 1
    assert stats.duplicate_file_count == 3


def test_analyze_sorts_multiple_duplicate_groups(
    tmp_path: Path,
) -> None:
    """Sort paths within groups and groups by their first path."""

    a_first = tmp_path / "a-first.bmp"
    a_second = tmp_path / "A-second.bmp"
    z_first = tmp_path / "z-first.bmp"
    z_second = tmp_path / "Z-second.bmp"
    create_test_image(a_first)
    copyfile(a_first, a_second)
    different_image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(z_first), different_image)
    copyfile(z_first, z_second)

    stats = analyze_dataset(tmp_path)

    assert stats.duplicate_group_count == 2
    assert stats.duplicate_image_groups[0].image_paths == (
        a_first,
        a_second,
    )
    assert stats.duplicate_image_groups[1].image_paths == (
        z_first,
        z_second,
    )
    assert (
        stats.duplicate_image_groups[0].sha256
        != stats.duplicate_image_groups[1].sha256
    )


def test_analyze_recursive_duplicate_requires_recursive_scanning(
    tmp_path: Path,
) -> None:
    """Include a nested exact copy only under recursive scanning."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    source_path = tmp_path / "source.png"
    nested_copy = nested_directory / "copy.png"
    create_test_image(source_path)
    copyfile(source_path, nested_copy)

    top_level_stats = analyze_dataset(tmp_path)
    recursive_stats = analyze_dataset(tmp_path, recursive=True)

    assert top_level_stats.duplicate_image_groups == []
    assert recursive_stats.duplicate_group_count == 1
    assert recursive_stats.duplicate_image_groups[0].image_paths == (
        nested_copy,
        source_path,
    )


def test_analyze_thresholds_control_duplicate_eligibility(
    tmp_path: Path,
) -> None:
    """Hash exact copies only when both pass configured thresholds."""

    source_path = tmp_path / "source.png"
    copy_path = tmp_path / "copy.png"
    create_test_image(source_path, width=20, height=20)
    copyfile(source_path, copy_path)

    default_stats = analyze_dataset(tmp_path)
    lowered_stats = analyze_dataset(
        tmp_path,
        min_width=10,
        min_height=10,
    )

    assert default_stats.valid_images == 0
    assert default_stats.duplicate_image_groups == []
    assert lowered_stats.valid_images == 2
    assert lowered_stats.duplicate_group_count == 1
    assert lowered_stats.duplicate_file_count == 2


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
