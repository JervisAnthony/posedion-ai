"""Create a small image dataset for local development."""

from pathlib import Path

import cv2
import numpy as np


def create_image(
    path: Path,
    *,
    width: int,
    height: int,
) -> None:
    """Create and save a simple test image."""

    image = np.full(
        (height, width, 3),
        fill_value=128,
        dtype=np.uint8,
    )

    saved = cv2.imwrite(str(path), image)

    if not saved:
        raise RuntimeError(f"Failed to create image: {path}")


def main() -> None:
    """Create a representative sample dataset."""

    dataset_path = Path("data/sample_dataset")
    dataset_path.mkdir(parents=True, exist_ok=True)

    create_image(
        dataset_path / "fish_01.jpg",
        width=640,
        height=480,
    )

    create_image(
        dataset_path / "fish_02.png",
        width=1280,
        height=720,
    )

    create_image(
        dataset_path / "coral_01.jpg",
        width=800,
        height=600,
    )

    (dataset_path / "broken.jpg").write_bytes(
        b"This is not a valid image."
    )

    (dataset_path / "notes.txt").write_text(
        "Sample Nautilus Vision dataset.",
        encoding="utf-8",
    )

    print(f"Sample dataset created at: {dataset_path.resolve()}")


if __name__ == "__main__":
    main()