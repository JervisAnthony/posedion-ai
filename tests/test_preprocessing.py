import numpy as np
import pytest

from poseidon_ai.nautilus_vision.preprocessing import letterbox_image


def test_landscape_image_is_letterboxed_to_target_size():
    """A landscape image should be resized and vertically padded."""

    image = np.zeros((100, 200, 3), dtype=np.uint8)

    result = letterbox_image(
        image,
        target_width=640,
        target_height=640,
    )

    assert result.image.shape == (640, 640, 3)
    assert result.scale == pytest.approx(3.2)
    assert result.padding_left == 0
    assert result.padding_top == 160
    assert result.original_width == 200
    assert result.original_height == 100


def test_portrait_image_is_letterboxed_to_target_size():
    """A portrait image should be resized and horizontally padded."""

    image = np.zeros((200, 100, 3), dtype=np.uint8)

    result = letterbox_image(
        image,
        target_width=640,
        target_height=640,
    )

    assert result.image.shape == (640, 640, 3)
    assert result.scale == pytest.approx(3.2)
    assert result.padding_left == 160
    assert result.padding_top == 0


def test_square_image_requires_no_padding():
    """A square image should fill a square target without padding."""

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = letterbox_image(
        image,
        target_width=640,
        target_height=640,
    )

    assert result.image.shape == (640, 640, 3)
    assert result.padding_left == 0
    assert result.padding_top == 0


def test_custom_target_dimensions_are_supported():
    """The output should match non-square target dimensions."""

    image = np.zeros((100, 200, 3), dtype=np.uint8)

    result = letterbox_image(
        image,
        target_width=320,
        target_height=192,
    )

    assert result.image.shape == (192, 320, 3)


def test_empty_image_raises_value_error():
    """An empty NumPy image should be rejected."""

    image = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError, match="must not be empty"):
        letterbox_image(image)


def test_invalid_target_dimensions_raise_value_error():
    """Target dimensions must be positive."""

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive"):
        letterbox_image(image, target_width=0)