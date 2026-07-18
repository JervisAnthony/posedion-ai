from pathlib import Path

import cv2
import numpy as np
import pytest

from poseidon_ai.nautilus_vision.dataset_loader import load_image_dataset


def create_test_image(path: Path) -> None:
    """Create a valid test image at the supplied path."""

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image) is True


def test_missing_directory_raises_file_not_found(tmp_path: Path):
    """A missing dataset directory should raise FileNotFoundError."""

    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_image_dataset(missing_directory)


def test_file_path_raises_not_a_directory(tmp_path: Path):
    """A file supplied as the dataset path should be rejected."""

    file_path = tmp_path / "image.jpg"
    create_test_image(file_path)

    with pytest.raises(NotADirectoryError):
        load_image_dataset(file_path)


def test_empty_directory_returns_empty_list(tmp_path: Path):
    """An empty dataset directory should return no images."""

    assert load_image_dataset(tmp_path) == []


def test_loader_returns_supported_images_in_sorted_order(tmp_path: Path):
    """Supported images should be returned alphabetically."""

    create_test_image(tmp_path / "zebra.jpg")
    create_test_image(tmp_path / "alpha.png")
    create_test_image(tmp_path / "middle.jpeg")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    images = load_image_dataset(tmp_path)

    assert [path.name for path in images] == [
        "alpha.png",
        "middle.jpeg",
        "zebra.jpg",
    ]


def test_non_recursive_loader_ignores_nested_images(tmp_path: Path):
    """Nested images should be ignored by default."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()

    create_test_image(tmp_path / "root.jpg")
    create_test_image(nested_directory / "nested.jpg")

    images = load_image_dataset(tmp_path)

    assert [path.name for path in images] == ["root.jpg"]


def test_recursive_loader_includes_nested_images(tmp_path: Path):
    """Nested images should be included when recursive mode is enabled."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()

    create_test_image(tmp_path / "root.jpg")
    create_test_image(nested_directory / "nested.jpg")

    images = load_image_dataset(tmp_path, recursive=True)

    assert [path.name for path in images] == [
        "nested.jpg",
        "root.jpg",
    ]