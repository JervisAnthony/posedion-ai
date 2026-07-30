"""Strict parsing and validation for YOLO detection label files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class YoloDetectionAnnotation:
    """One normalized object-detection annotation from a YOLO label."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line_number: int


@dataclass(frozen=True, slots=True)
class YoloLabelValidationResult:
    """The annotations and errors produced while validating one label."""

    is_valid: bool
    annotations: tuple[YoloDetectionAnnotation, ...]
    errors: tuple[str, ...]


def _parse_class_id(
    value: str,
    line_number: int,
) -> tuple[int | None, str | None]:
    """Parse a non-negative base-10 class identifier."""

    try:
        class_id = int(value, 10)
    except ValueError:
        class_id = -1

    if class_id < 0:
        return (
            None,
            (
                f"Line {line_number}: class_id must be a "
                "non-negative integer."
            ),
        )

    return class_id, None


def _parse_coordinate(
    value: str,
    *,
    field_name: str,
    line_number: int,
    allow_zero: bool,
) -> tuple[float | None, str | None]:
    """Parse one finite normalized coordinate."""

    try:
        coordinate = float(value)
    except ValueError:
        coordinate = math.nan

    if not math.isfinite(coordinate):
        return (
            None,
            (
                f"Line {line_number}: {field_name} must be a "
                "finite number."
            ),
        )

    if allow_zero:
        if not 0.0 <= coordinate <= 1.0:
            return (
                None,
                (
                    f"Line {line_number}: {field_name} must be "
                    "between 0.0 and 1.0 inclusive."
                ),
            )
    elif not 0.0 < coordinate <= 1.0:
        return (
            None,
            (
                f"Line {line_number}: {field_name} must be "
                "greater than 0.0 and at most 1.0."
            ),
        )

    return coordinate, None


def validate_yolo_label(
    label_path: str | Path,
) -> YoloLabelValidationResult:
    """Parse and validate one UTF-8 YOLO detection label file."""

    path = Path(label_path)

    if not path.exists():
        return YoloLabelValidationResult(
            is_valid=False,
            annotations=(),
            errors=(f"Label file does not exist: {path}",),
        )

    if not path.is_file():
        return YoloLabelValidationResult(
            is_valid=False,
            annotations=(),
            errors=(f"Label path is not a file: {path}",),
        )

    if path.suffix.lower() != ".txt":
        return YoloLabelValidationResult(
            is_valid=False,
            annotations=(),
            errors=(
                f"Unsupported label extension: {path.suffix}",
            ),
        )

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return YoloLabelValidationResult(
            is_valid=False,
            annotations=(),
            errors=(f"Label file is not valid UTF-8: {path}",),
        )

    annotations: list[YoloDetectionAnnotation] = []
    errors: list[str] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            continue

        fields = line.split()
        if len(fields) != 5:
            errors.append(
                f"Line {line_number}: expected 5 fields, "
                f"found {len(fields)}."
            )
            continue

        line_errors: list[str] = []

        class_id, class_error = _parse_class_id(
            fields[0],
            line_number,
        )
        if class_error:
            line_errors.append(class_error)

        coordinates: list[float | None] = []
        for field_name, value, allow_zero in (
            ("x_center", fields[1], True),
            ("y_center", fields[2], True),
            ("width", fields[3], False),
            ("height", fields[4], False),
        ):
            coordinate, coordinate_error = _parse_coordinate(
                value,
                field_name=field_name,
                line_number=line_number,
                allow_zero=allow_zero,
            )
            coordinates.append(coordinate)
            if coordinate_error:
                line_errors.append(coordinate_error)

        if line_errors:
            errors.extend(line_errors)
            continue

        x_center, y_center, width, height = coordinates
        assert class_id is not None
        assert x_center is not None
        assert y_center is not None
        assert width is not None
        assert height is not None
        annotations.append(
            YoloDetectionAnnotation(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                line_number=line_number,
            )
        )

    return YoloLabelValidationResult(
        is_valid=not errors,
        annotations=tuple(annotations),
        errors=tuple(errors),
    )
