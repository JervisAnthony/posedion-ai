"""Tests for incremental image content hashing."""

import hashlib
from pathlib import Path

import pytest

from poseidon_ai.nautilus_vision.image_hash import calculate_sha256


def test_identical_bytes_produce_matching_sha256(
    tmp_path: Path,
) -> None:
    """Hash identical file content consistently."""

    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    content = b"poseidon-exact-duplicate"
    first_path.write_bytes(content)
    second_path.write_bytes(content)

    assert calculate_sha256(first_path) == calculate_sha256(second_path)


def test_different_bytes_produce_different_sha256(
    tmp_path: Path,
) -> None:
    """Distinguish files with different byte content."""

    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")

    assert calculate_sha256(first_path) != calculate_sha256(second_path)


def test_sha256_matches_hashlib_with_small_chunks(
    tmp_path: Path,
) -> None:
    """Match hashlib while reading content across small chunks."""

    image_path = tmp_path / "image.bin"
    content = b"known-content-across-several-chunks"
    image_path.write_bytes(content)

    assert calculate_sha256(
        image_path,
        chunk_size=3,
    ) == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_sha256_rejects_non_positive_chunk_size(
    tmp_path: Path,
    chunk_size: int,
) -> None:
    """Require a positive incremental read size."""

    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"content")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        calculate_sha256(image_path, chunk_size=chunk_size)


def test_sha256_preserves_missing_file_error(
    tmp_path: Path,
) -> None:
    """Allow normal filesystem errors to propagate."""

    with pytest.raises(FileNotFoundError):
        calculate_sha256(tmp_path / "missing.bin")
