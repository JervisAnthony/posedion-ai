"""Tests for installed project script metadata."""

import tomllib
from pathlib import Path


def load_project_scripts() -> dict[str, str]:
    """Load the project script mappings from pyproject.toml."""

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    return pyproject["project"]["scripts"]


def test_dataset_summary_script_mapping() -> None:
    """Expose the existing dataset-summary main function."""

    scripts = load_project_scripts()

    assert scripts["poseidon-dataset-summary"] == (
        "poseidon_ai.nautilus_vision.dataset_summary:main"
    )


def test_poseidon_inspect_script_mapping_is_unchanged() -> None:
    """Preserve the existing single-image inspection command."""

    scripts = load_project_scripts()

    assert scripts["poseidon-inspect"] == (
        "poseidon_ai.nautilus_vision.inspect_image:main"
    )
