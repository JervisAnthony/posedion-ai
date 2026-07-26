import cv2
import numpy as np

from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_analyzer import analyze_dataset

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

    assert stats.min_width == 100
    assert stats.max_width == 100
    assert stats.average_width == 100

    assert stats.min_height == 100
    assert stats.max_height == 100
    assert stats.average_height == 100

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