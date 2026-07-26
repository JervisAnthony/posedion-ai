from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_analyzer import analyze_dataset


def test_analyze_empty_dataset(tmp_path: Path) -> None:
    """An empty dataset should contain zero statistics."""

    stats = analyze_dataset(tmp_path)

    assert stats.total_images == 0
    assert stats.valid_images == 0
    assert stats.invalid_images == 0