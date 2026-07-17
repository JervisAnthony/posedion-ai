"""
Image loading utilities for Nautilus Vision.

This module provides helper functions for loading and validating
images before they enter the computer vision pipeline.
"""

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    image_path : str | Path
        Path to the image.

    Returns
    -------
    np.ndarray
        Image represented as a NumPy array.

    Raises
    ------
    FileNotFoundError
        If the image cannot be loaded.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    return image