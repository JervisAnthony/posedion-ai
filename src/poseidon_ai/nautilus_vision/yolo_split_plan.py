"""In-memory planning for configured YOLO dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poseidon_ai.nautilus_vision.yolo_config import (
    YoloDatasetConfiguration,
)


@dataclass(frozen=True, slots=True)
class YoloDatasetSplit:
    """One configured image directory and its derived label directory."""

    name: str
    image_directory: Path
    label_directory: Path


@dataclass(frozen=True, slots=True)
class YoloDatasetSplitPlan:
    """The immutable ordered split plan for one configuration."""

    config_path: Path
    splits: tuple[YoloDatasetSplit, ...]

    @property
    def training_split(self) -> YoloDatasetSplit:
        """Return the existing training split."""

        return self.splits[0]

    @property
    def validation_split(self) -> YoloDatasetSplit:
        """Return the existing validation split."""

        return self.splits[1]

    @property
    def test_split(self) -> YoloDatasetSplit | None:
        """Return the existing optional test split."""

        return self.splits[2] if len(self.splits) == 3 else None


@dataclass(frozen=True, slots=True)
class YoloDatasetSplitPlanValidationResult:
    """The immutable result of configured split planning."""

    is_valid: bool
    plan: YoloDatasetSplitPlan | None
    errors: tuple[str, ...]


def _directory_name(
    value: object,
    *,
    option_name: str,
) -> tuple[str | None, str | None]:
    """Return one stripped ordinary path component or its exact error."""

    error = (
        f"{option_name} must be a non-empty single path component."
    )
    if not isinstance(value, str):
        return None, error

    stripped = value.strip()
    if (
        not stripped
        or stripped in {".", ".."}
        or "/" in stripped
        or "\\" in stripped
    ):
        return None, error

    component = Path(stripped)
    if (
        component.is_absolute()
        or component.anchor
        or len(component.parts) != 1
    ):
        return None, error

    return stripped, None


def _label_directory(
    image_directory: Path,
    *,
    images_directory_name: str,
    labels_directory_name: str,
) -> Path | None:
    """Replace the final matching image-directory path component."""

    parts = list(image_directory.parts)
    matching_indexes = [
        index
        for index, part in enumerate(parts)
        if part == images_directory_name
    ]
    if not matching_indexes:
        return None

    parts[matching_indexes[-1]] = labels_directory_name
    return Path(*parts)


def build_yolo_dataset_split_plan(
    configuration: YoloDatasetConfiguration,
    *,
    images_directory_name: str = "images",
    labels_directory_name: str = "labels",
) -> YoloDatasetSplitPlanValidationResult:
    """Build an ordered split plan without inspecting the filesystem."""

    option_errors: list[str] = []
    image_name, image_error = _directory_name(
        images_directory_name,
        option_name="images_directory_name",
    )
    if image_error is not None:
        option_errors.append(image_error)

    label_name, label_error = _directory_name(
        labels_directory_name,
        option_name="labels_directory_name",
    )
    if label_error is not None:
        option_errors.append(label_error)

    if (
        image_name is not None
        and label_name is not None
        and image_name == label_name
    ):
        option_errors.append(
            "Image and label directory names must be different."
        )

    if option_errors:
        return YoloDatasetSplitPlanValidationResult(
            is_valid=False,
            plan=None,
            errors=tuple(option_errors),
        )

    assert image_name is not None
    assert label_name is not None
    configured_splits: list[tuple[str, Path]] = [
        ("train", configuration.train_path),
        ("validation", configuration.validation_path),
    ]
    if configuration.test_path is not None:
        configured_splits.append(("test", configuration.test_path))

    splits: list[YoloDatasetSplit] = []
    split_errors: list[str] = []
    for split_name, image_directory in configured_splits:
        label_directory = _label_directory(
            image_directory,
            images_directory_name=image_name,
            labels_directory_name=label_name,
        )
        if label_directory is None:
            split_errors.append(
                f"Split '{split_name}' image path does not contain "
                f"directory component '{image_name}': {image_directory}"
            )
            continue

        splits.append(
            YoloDatasetSplit(
                name=split_name,
                image_directory=image_directory,
                label_directory=label_directory,
            )
        )

    if split_errors:
        return YoloDatasetSplitPlanValidationResult(
            is_valid=False,
            plan=None,
            errors=tuple(split_errors),
        )

    return YoloDatasetSplitPlanValidationResult(
        is_valid=True,
        plan=YoloDatasetSplitPlan(
            config_path=configuration.config_path,
            splits=tuple(splits),
        ),
        errors=(),
    )
