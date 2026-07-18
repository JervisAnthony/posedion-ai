"""Image preprocessing utilities for Nautilus Vision."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxResult:
    """Result of resizing and padding an image."""

    image: np.ndarray
    scale: float
    padding_left: int
    padding_top: int
    original_width: int
    original_height: int


def letterbox_image(
    image: np.ndarray,
    target_width: int = 640,
    target_height: int = 640,
    padding_value: int = 114,
) -> LetterboxResult:
    """Resize an image while preserving aspect ratio and add padding.

    Parameters
    ----------
    image:
        Source image represented as a NumPy array.
    target_width:
        Width of the output image.
    target_height:
        Height of the output image.
    padding_value:
        Pixel value used for the padded area.

    Returns
    -------
    LetterboxResult
        Resized image and transformation information.

    Raises
    ------
    ValueError
        If the image or target dimensions are invalid.
    """

    if image is None or image.size == 0:
        raise ValueError("Image must not be empty.")

    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive integers.")

    original_height, original_width = image.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height

    padding_left = horizontal_padding // 2
    padding_right = horizontal_padding - padding_left
    padding_top = vertical_padding // 2
    padding_bottom = vertical_padding - padding_top

    padded_image = cv2.copyMakeBorder(
        resized_image,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(padding_value, padding_value, padding_value),
    )

    return LetterboxResult(
        image=padded_image,
        scale=scale,
        padding_left=padding_left,
        padding_top=padding_top,
        original_width=original_width,
        original_height=original_height,
    )