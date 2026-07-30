from pathlib import Path

import cv2
import numpy as np
import pytest

from poseidon_ai.nautilus_vision.image_validator import validate_image


def create_test_image(
    path: Path,
    *,
    width: int,
    height: int,
) -> None:
    """Create a decodable image with explicit dimensions."""

    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image) is True


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


def test_default_minimum_dimensions_remain_32_pixels(
    tmp_path: Path,
) -> None:
    """Keep the existing 32-pixel defaults."""

    image_path = tmp_path / "small.png"
    create_test_image(image_path, width=31, height=32)

    result = validate_image(image_path)

    assert result.is_valid is False
    assert result.errors == (
        "Width 31px is below minimum 32px.",
    )


def test_lower_custom_thresholds_allow_small_image(
    tmp_path: Path,
) -> None:
    """Allow a small image when both thresholds are lowered."""

    image_path = tmp_path / "small.png"
    create_test_image(image_path, width=20, height=20)

    result = validate_image(
        image_path,
        min_width=10,
        min_height=10,
    )

    assert result.is_valid is True
    assert result.errors == ()


@pytest.mark.parametrize(
    ("min_width", "min_height", "expected_errors"),
    [
        (
            100,
            40,
            ("Width 50px is below minimum 100px.",),
        ),
        (
            40,
            100,
            ("Height 60px is below minimum 100px.",),
        ),
        (
            100,
            100,
            (
                "Width 50px is below minimum 100px.",
                "Height 60px is below minimum 100px.",
            ),
        ),
    ],
)
def test_custom_dimension_failures_are_independent(
    min_width: int,
    min_height: int,
    expected_errors: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Report only failing axes and preserve width-height order."""

    image_path = tmp_path / "image.png"
    create_test_image(image_path, width=50, height=60)

    result = validate_image(
        image_path,
        min_width=min_width,
        min_height=min_height,
    )

    assert result.is_valid is False
    assert result.errors == expected_errors
