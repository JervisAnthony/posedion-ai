"""Dataset loading utilities for Nautilus Vision."""

from pathlib import Path

from poseidon_ai.nautilus_vision.image_validator import (
    SUPPORTED_EXTENSIONS,
    validate_image,
)


def load_image_dataset(
    directory: str | Path,
    *,
    recursive: bool = False,
    validate: bool = True,
) -> list[Path]:
    """Return supported image files from a directory.

    Parameters
    ----------
    directory:
        Directory containing image files.
    recursive:
        Search nested directories when True.
    validate:
        Exclude unreadable or invalid images when True.

    Returns
    -------
    list[Path]
        Sorted paths to supported image files.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    NotADirectoryError
        If the supplied path is not a directory.
    """

    dataset_directory = Path(directory)

    if not dataset_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_directory}"
        )

    if not dataset_directory.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {dataset_directory}"
        )

    candidates = (
        dataset_directory.rglob("*")
        if recursive
        else dataset_directory.iterdir()
    )

    image_paths: list[Path] = []

    for candidate in candidates:
        if not candidate.is_file():
            continue

        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if validate and not validate_image(candidate).is_valid:
            continue

        image_paths.append(candidate)

    return sorted(
        image_paths,
        key=lambda path: str(path.relative_to(dataset_directory)).lower(),
    )