"""Strict validation for YOLO detection dataset YAML configurations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True, slots=True)
class YoloClassDefinition:
    """One normalized numeric YOLO class definition."""

    class_id: int
    name: str


@dataclass(frozen=True, slots=True)
class YoloDatasetConfiguration:
    """Validated paths and class definitions from one YOLO configuration."""

    config_path: Path
    dataset_root: Path
    train_path: Path
    validation_path: Path
    test_path: Path | None
    classes: tuple[YoloClassDefinition, ...]

    @property
    def number_of_classes(self) -> int:
        """Return the number of configured classes."""

        return len(self.classes)


@dataclass(frozen=True, slots=True)
class YoloDatasetConfigValidationResult:
    """The immutable result of validating one YOLO dataset configuration."""

    is_valid: bool
    configuration: YoloDatasetConfiguration | None
    errors: tuple[str, ...]


def _invalid_result(
    *errors: str,
) -> YoloDatasetConfigValidationResult:
    """Return an invalid result without a partial public configuration."""

    return YoloDatasetConfigValidationResult(
        is_valid=False,
        configuration=None,
        errors=errors,
    )


def _non_empty_string(value: object) -> str | None:
    """Return one stripped non-empty string, or None when invalid."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _construct_path(root: Path, value: str) -> Path:
    """Construct an absolute or root-relative path without normalization."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_yolo_dataset_config(
    config_path: str | Path,
) -> YoloDatasetConfigValidationResult:
    """Parse and validate one UTF-8 YOLO dataset YAML configuration."""

    path = Path(config_path)

    if not path.exists():
        return _invalid_result(
            f"Dataset configuration does not exist: {path}"
        )
    if not path.is_file():
        return _invalid_result(
            f"Dataset configuration path is not a file: {path}"
        )
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return _invalid_result(
            f"Unsupported dataset configuration extension: {path.suffix}"
        )

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _invalid_result(
            f"Dataset configuration is not valid UTF-8: {path}"
        )

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return _invalid_result(
            f"Dataset configuration contains invalid YAML: {path}"
        )

    if not isinstance(parsed, Mapping):
        return _invalid_result(
            "Dataset configuration must contain a YAML mapping."
        )

    data: Mapping[Any, Any] = parsed
    errors: list[str] = []

    dataset_root: Path | None
    if "path" not in data:
        dataset_root = path.parent
    else:
        path_value = _non_empty_string(data["path"])
        if path_value is None:
            errors.append(
                "Field 'path' must be a non-empty string when provided."
            )
            dataset_root = None
        else:
            dataset_root = _construct_path(path.parent, path_value)

    train_value = _non_empty_string(data.get("train"))
    if train_value is None:
        errors.append("Field 'train' must be a non-empty string.")

    validation_value = _non_empty_string(data.get("val"))
    if validation_value is None:
        errors.append("Field 'val' must be a non-empty string.")

    test_value: str | None = None
    if "test" in data and data["test"] is not None:
        test_value = _non_empty_string(data["test"])
        if test_value is None:
            errors.append(
                "Field 'test' must be a non-empty string or null "
                "when provided."
            )

    class_definitions: list[YoloClassDefinition] = []
    class_errors = False
    names = data.get("names")

    if not isinstance(names, (list, Mapping)) or not names:
        errors.append(
            "Field 'names' must be a non-empty list or mapping."
        )
        class_errors = True
    elif isinstance(names, list):
        for index, raw_name in enumerate(names):
            name = _non_empty_string(raw_name)
            if name is None:
                errors.append(
                    f"Class name at index {index} must be a "
                    "non-empty string."
                )
                class_errors = True
                continue
            class_definitions.append(
                YoloClassDefinition(class_id=index, name=name)
            )
    else:
        valid_class_ids: list[int] = []
        for raw_class_id, raw_name in names.items():
            if (
                not isinstance(raw_class_id, int)
                or isinstance(raw_class_id, bool)
                or raw_class_id < 0
            ):
                errors.append(
                    "Class ID key must be a non-negative integer: "
                    f"{raw_class_id!r}."
                )
                class_errors = True
                continue

            valid_class_ids.append(raw_class_id)
            name = _non_empty_string(raw_name)
            if name is None:
                errors.append(
                    f"Class name for class ID {raw_class_id} must be "
                    "a non-empty string."
                )
                class_errors = True
                continue
            class_definitions.append(
                YoloClassDefinition(
                    class_id=raw_class_id,
                    name=name,
                )
            )

        sorted_class_ids = sorted(valid_class_ids)
        if sorted_class_ids and sorted_class_ids != list(
            range(len(sorted_class_ids))
        ):
            found = ", ".join(
                str(class_id) for class_id in sorted_class_ids
            )
            errors.append(
                "Class IDs must be contiguous and start at 0; "
                f"found: {found}."
            )
            class_errors = True

        class_definitions.sort(
            key=lambda definition: definition.class_id
        )

    seen_names: set[str] = set()
    for definition in class_definitions:
        if definition.name in seen_names:
            errors.append(
                "Class names must be unique; duplicate: "
                f"{definition.name!r}."
            )
            class_errors = True
        else:
            seen_names.add(definition.name)

    if "nc" in data:
        declared_count = data["nc"]
        if (
            not isinstance(declared_count, int)
            or isinstance(declared_count, bool)
            or declared_count <= 0
        ):
            errors.append(
                "Field 'nc' must be a positive integer when provided."
            )
        elif not class_errors and declared_count != len(
            class_definitions
        ):
            errors.append(
                "Field 'nc' must equal the number of configured classes: "
                f"expected {len(class_definitions)}, "
                f"found {declared_count}."
            )

    if errors:
        return YoloDatasetConfigValidationResult(
            is_valid=False,
            configuration=None,
            errors=tuple(errors),
        )

    completed_root = cast(Path, dataset_root)
    configuration = YoloDatasetConfiguration(
        config_path=path,
        dataset_root=completed_root,
        train_path=_construct_path(
            completed_root,
            cast(str, train_value),
        ),
        validation_path=_construct_path(
            completed_root,
            cast(str, validation_value),
        ),
        test_path=(
            _construct_path(completed_root, test_value)
            if test_value is not None
            else None
        ),
        classes=tuple(class_definitions),
    )
    return YoloDatasetConfigValidationResult(
        is_valid=True,
        configuration=configuration,
        errors=(),
    )
