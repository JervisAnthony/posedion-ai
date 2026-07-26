"""Tests for file size formatting utilities."""

import pytest

from poseidon_ai.utils.filesize import format_file_size


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (0, "0 Bytes"),
        (500, "500 Bytes"),
        (1024, "1.00 KB"),
        (1536, "1.50 KB"),
        (1024**2, "1.00 MB"),
        (1024**3, "1.00 GB"),
        (1024**4, "1.00 TB"),
        (1024**5, "1.00 PB"),
    ],
)
def test_format_file_size(
    size_bytes: int,
    expected: str,
) -> None:
    """Format byte values using the appropriate unit."""

    assert format_file_size(size_bytes) == expected