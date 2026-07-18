from pathlib import Path

import cv2
import numpy as np

from poseidon_ai.nautilus_vision.image_validator import validate_image


def test_missing_image_returns_invalid(tmp_path: Path):

    image = tmp_path / "missing.jpg"

    result = validate_image(image)

    assert result.is_valid is False


def test_valid_image_returns_valid(tmp_path: Path):

    image_path = tmp_path / "sample.jpg"

    image = np.zeros((100, 200, 3), dtype=np.uint8)

    cv2.imwrite(str(image_path), image)

    result = validate_image(image_path)

    assert result.is_valid

    assert result.width == 200

    assert result.height == 100

    assert result.channels == 3