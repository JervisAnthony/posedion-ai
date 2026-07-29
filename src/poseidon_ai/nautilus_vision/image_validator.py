"""Image validation utilities for Nautilus Vision."""

from dataclasses import dataclass
from pathlib import Path

import cv2


DEFAULT_MIN_WIDTH = 32
DEFAULT_MIN_HEIGHT = 32

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


@dataclass(frozen=True)
class ImageValidationResult:
    """Result of validating an image."""

    is_valid: bool
    errors: tuple[str, ...]

    width: int | None = None
    height: int | None = None
    channels: int | None = None


def validate_image(
    image_path: str | Path,
    *,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> ImageValidationResult:
    """Validate whether an image can enter the vision pipeline."""

    path = Path(image_path)
    errors = []

    if not path.exists():
        return ImageValidationResult(
            False,
            (f"Image does not exist: {path}",),
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        errors.append(
            f"Unsupported extension: {path.suffix}"
        )

    image = cv2.imread(str(path))

    if image is None:
        errors.append(
            "Image could not be decoded."
        )

        return ImageValidationResult(
            False,
            tuple(errors),
        )

    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    if width < min_width:
        errors.append(
            f"Width {width}px is below minimum {min_width}px."
        )

    if height < min_height:
        errors.append(
            f"Height {height}px is below minimum {min_height}px."
        )

    return ImageValidationResult(
        len(errors) == 0,
        tuple(errors),
        width,
        height,
        channels,
    )
