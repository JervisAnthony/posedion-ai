from pathlib import Path

import cv2
import numpy as np
import pytest

from poseidon_ai.nautilus_vision.image_metadata import get_image_metadata


def test_missing_image_raises_file_not_found():
    """A missing image should raise FileNotFoundError."""

    missing_file = Path("this_image_does_not_exist.jpg")

    with pytest.raises(FileNotFoundError):
        get_image_metadata(missing_file)


def test_get_image_metadata_returns_expected_values(tmp_path: Path):
    """Metadata should match the generated test image."""

    image_path = tmp_path / "sample_image.jpg"

    image = np.zeros((100, 200, 3), dtype=np.uint8)

    created = cv2.imwrite(str(image_path), image)

    assert created is True

    metadata = get_image_metadata(image_path)

    assert metadata["filename"] == "sample_image.jpg"
    assert metadata["width"] == 200
    assert metadata["height"] == 100
    assert metadata["channels"] == 3
    assert metadata["size_bytes"] > 0