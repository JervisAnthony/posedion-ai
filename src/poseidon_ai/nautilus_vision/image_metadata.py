from pathlib import Path

import cv2


def get_image_metadata(image_path: str | Path) -> dict:
    """
    Extract metadata from an image.
    """

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    height, width = image.shape[:2]

    channels = 1 if len(image.shape) == 2 else image.shape[2]

    return {
        "filename": image_path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "size_bytes": image_path.stat().st_size,
    }