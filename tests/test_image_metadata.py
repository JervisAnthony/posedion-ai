from pathlib import Path

import pytest

from poseidon_ai.nautilus_vision.image_metadata import get_image_metadata


def test_missing_image_raises_file_not_found():
    """A missing image should raise FileNotFoundError."""

    missing_file = Path("this_image_does_not_exist.jpg")

    with pytest.raises(FileNotFoundError):
        get_image_metadata(missing_file)