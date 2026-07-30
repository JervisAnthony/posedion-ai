"""Content hashing utilities for Nautilus Vision images."""

import hashlib
from pathlib import Path


def calculate_sha256(
    image_path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return the SHA-256 digest of a file read incrementally."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    path = Path(image_path)

    with path.open("rb") as image_file:
        while chunk := image_file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()
