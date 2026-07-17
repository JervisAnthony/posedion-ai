from pathlib import Path

import pytest

from poseidon_ai.nautilus_vision.image_loader import load_image


def test_missing_image_raises_file_not_found():
    """Loading a non-existent image should raise FileNotFoundError."""

    missing_file = Path("this_image_does_not_exist.jpg")

    with pytest.raises(FileNotFoundError):
        load_image(missing_file)