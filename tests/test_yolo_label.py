"""Tests for strict YOLO detection-label validation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from poseidon_ai.nautilus_vision.yolo_label import (
    YoloDetectionAnnotation,
    YoloLabelValidationResult,
    validate_yolo_label,
)


def write_label(path: Path, content: str) -> Path:
    """Write one UTF-8 label file and return its path."""

    path.write_text(content, encoding="utf-8")
    return path


def test_missing_label_returns_exact_error(tmp_path: Path) -> None:
    """Reject a missing path before content parsing."""

    label_path = tmp_path / "missing.txt"

    result = validate_yolo_label(label_path)

    assert result == YoloLabelValidationResult(
        is_valid=False,
        annotations=(),
        errors=(f"Label file does not exist: {label_path}",),
    )


def test_directory_returns_exact_not_a_file_error(
    tmp_path: Path,
) -> None:
    """Reject an existing directory path."""

    result = validate_yolo_label(tmp_path)

    assert result.errors == (
        f"Label path is not a file: {tmp_path}",
    )
    assert result.annotations == ()
    assert result.is_valid is False


@pytest.mark.parametrize("filename", ["labels.csv", "labels"])
def test_unsupported_extension_returns_exact_error(
    filename: str,
    tmp_path: Path,
) -> None:
    """Reject non-TXT and extensionless regular files."""

    label_path = write_label(tmp_path / filename, "")

    result = validate_yolo_label(label_path)

    assert result.errors == (
        f"Unsupported label extension: {label_path.suffix}",
    )
    assert result.annotations == ()


def test_uppercase_txt_extension_is_accepted(tmp_path: Path) -> None:
    """Match the supported extension case-insensitively."""

    label_path = write_label(
        tmp_path / "labels.TXT",
        "0 0.5 0.5 0.2 0.3\n",
    )

    result = validate_yolo_label(label_path)

    assert result.is_valid is True
    assert len(result.annotations) == 1


def test_invalid_utf8_returns_exact_error(tmp_path: Path) -> None:
    """Translate only UTF-8 decoding failures into validation errors."""

    label_path = tmp_path / "labels.txt"
    label_path.write_bytes(b"\xff\xfe")

    result = validate_yolo_label(label_path)

    assert result == YoloLabelValidationResult(
        is_valid=False,
        annotations=(),
        errors=(f"Label file is not valid UTF-8: {label_path}",),
    )


def test_unexpected_filesystem_error_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve normal library semantics for unexpected read failures."""

    label_path = write_label(tmp_path / "labels.txt", "")

    def deny_read(
        path: Path,
        *,
        encoding: str,
    ) -> str:
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "read_text", deny_read)

    with pytest.raises(PermissionError, match="permission denied"):
        validate_yolo_label(label_path)


@pytest.mark.parametrize("content", ["", " \n\t\n  \r\n"])
def test_empty_or_whitespace_only_label_is_valid(
    content: str,
    tmp_path: Path,
) -> None:
    """Treat zero annotations as a valid label."""

    result = validate_yolo_label(
        write_label(tmp_path / "labels.txt", content)
    )

    assert result == YoloLabelValidationResult(
        is_valid=True,
        annotations=(),
        errors=(),
    )


def test_blank_lines_preserve_physical_line_numbers(
    tmp_path: Path,
) -> None:
    """Ignore blank lines without renumbering later annotations."""

    label_path = write_label(
        tmp_path / "labels.txt",
        "\n0 0.5 0.5 0.2 0.3\n \t \n3 0.1 0.2 0.3 0.4\n",
    )

    result = validate_yolo_label(label_path)

    assert [item.line_number for item in result.annotations] == [2, 4]


def test_one_valid_annotation_preserves_exact_values(
    tmp_path: Path,
) -> None:
    """Parse the public annotation fields without rounding."""

    label_path = write_label(
        tmp_path / "labels.txt",
        "25 0.123456789012345 0.75 0.1 0.2\n",
    )

    result = validate_yolo_label(str(label_path))

    assert result.is_valid is True
    assert result.errors == ()
    assert result.annotations == (
        YoloDetectionAnnotation(
            class_id=25,
            x_center=0.123456789012345,
            y_center=0.75,
            width=0.1,
            height=0.2,
            line_number=1,
        ),
    )


def test_multiple_annotations_preserve_source_order(
    tmp_path: Path,
) -> None:
    """Do not sort annotations by class identifier."""

    label_path = write_label(
        tmp_path / "labels.txt",
        "3 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8\n",
    )

    result = validate_yolo_label(label_path)

    assert [item.class_id for item in result.annotations] == [3, 1]
    assert [item.line_number for item in result.annotations] == [1, 2]


def test_whitespace_tokenization_and_leading_zero_class(
    tmp_path: Path,
) -> None:
    """Accept mixed whitespace and base-10 identifiers with zeros."""

    label_path = write_label(
        tmp_path / "labels.txt",
        " \t007   0.5\t0.5  0.2\t0.3   \n",
    )

    result = validate_yolo_label(label_path)

    assert result.is_valid is True
    assert result.annotations[0].class_id == 7


def test_coordinate_boundaries_and_scientific_notation(
    tmp_path: Path,
) -> None:
    """Accept direct range boundaries and finite scientific notation."""

    label_path = write_label(
        tmp_path / "labels.txt",
        "0 0.0 1.0 1.0 1e-1\n1 5e-1 2.5e-1 1e-2 1.0\n",
    )

    result = validate_yolo_label(label_path)

    assert result.is_valid is True
    assert result.annotations[0].x_center == 0.0
    assert result.annotations[0].y_center == 1.0
    assert result.annotations[0].width == 1.0
    assert result.annotations[1].x_center == 0.5
    assert result.annotations[1].height == 1.0


@pytest.mark.parametrize(
    ("line", "field_count"),
    [
        ("0 0.5 0.5 0.2", 4),
        ("0 0.5 0.5 0.2 0.3 extra", 6),
        ("0 0.5 0.5 0.2 0.3 # comment", 7),
    ],
)
def test_wrong_field_count_returns_exact_error(
    line: str,
    field_count: int,
    tmp_path: Path,
) -> None:
    """Reject malformed field counts without stripping comments."""

    result = validate_yolo_label(
        write_label(tmp_path / "labels.txt", line)
    )

    assert result.errors == (
        f"Line 1: expected 5 fields, found {field_count}.",
    )
    assert result.annotations == ()


@pytest.mark.parametrize("class_id", ["-1", "1.0", "fish", "2e1"])
def test_invalid_class_identifier_returns_exact_error(
    class_id: str,
    tmp_path: Path,
) -> None:
    """Reject negative, floating-point, and nonnumeric class identifiers."""

    result = validate_yolo_label(
        write_label(
            tmp_path / "labels.txt",
            f"{class_id} 0.5 0.5 0.2 0.3",
        )
    )

    assert result.errors == (
        "Line 1: class_id must be a non-negative integer.",
    )


def test_valid_line_after_invalid_class_is_retained(
    tmp_path: Path,
) -> None:
    """Continue parsing after an invalid class identifier."""

    label_path = write_label(
        tmp_path / "labels.txt",
        "-1 0.5 0.5 0.2 0.3\n4 0.25 0.75 0.1 0.2\n",
    )

    result = validate_yolo_label(label_path)

    assert result.is_valid is False
    assert [item.class_id for item in result.annotations] == [4]
    assert result.errors == (
        "Line 1: class_id must be a non-negative integer.",
    )


@pytest.mark.parametrize(
    ("field_name", "values"),
    [
        ("x_center", ("fish", "0.5", "0.2", "0.3")),
        ("y_center", ("0.5", "nan", "0.2", "0.3")),
        ("width", ("0.5", "0.5", "inf", "0.3")),
        ("height", ("0.5", "0.5", "0.2", "-inf")),
    ],
)
def test_nonnumeric_and_nonfinite_coordinates_are_rejected(
    field_name: str,
    values: tuple[str, str, str, str],
    tmp_path: Path,
) -> None:
    """Require every coordinate token to convert to a finite float."""

    result = validate_yolo_label(
        write_label(
            tmp_path / "labels.txt",
            f"0 {' '.join(values)}",
        )
    )

    assert result.errors == (
        f"Line 1: {field_name} must be a finite number.",
    )


def test_parsing_continues_after_nonfinite_coordinate(
    tmp_path: Path,
) -> None:
    """Retain a later annotation after a nonfinite coordinate."""

    label_path = write_label(
        tmp_path / "labels.txt",
        "0 nan 0.5 0.2 0.3\n2 0.1 0.2 0.3 0.4\n",
    )

    result = validate_yolo_label(label_path)

    assert [item.class_id for item in result.annotations] == [2]
    assert result.errors == (
        "Line 1: x_center must be a finite number.",
    )


@pytest.mark.parametrize(
    ("field_name", "values", "expected"),
    [
        (
            "x_center",
            ("-0.1", "0.5", "0.2", "0.3"),
            "must be between 0.0 and 1.0 inclusive.",
        ),
        (
            "x_center",
            ("1.1", "0.5", "0.2", "0.3"),
            "must be between 0.0 and 1.0 inclusive.",
        ),
        (
            "y_center",
            ("0.5", "-0.1", "0.2", "0.3"),
            "must be between 0.0 and 1.0 inclusive.",
        ),
        (
            "y_center",
            ("0.5", "1.1", "0.2", "0.3"),
            "must be between 0.0 and 1.0 inclusive.",
        ),
        (
            "width",
            ("0.5", "0.5", "0", "0.3"),
            "must be greater than 0.0 and at most 1.0.",
        ),
        (
            "width",
            ("0.5", "0.5", "-0.1", "0.3"),
            "must be greater than 0.0 and at most 1.0.",
        ),
        (
            "width",
            ("0.5", "0.5", "1.1", "0.3"),
            "must be greater than 0.0 and at most 1.0.",
        ),
        (
            "height",
            ("0.5", "0.5", "0.2", "0"),
            "must be greater than 0.0 and at most 1.0.",
        ),
        (
            "height",
            ("0.5", "0.5", "0.2", "-0.1"),
            "must be greater than 0.0 and at most 1.0.",
        ),
        (
            "height",
            ("0.5", "0.5", "0.2", "1.1"),
            "must be greater than 0.0 and at most 1.0.",
        ),
        (
            "width",
            ("0.5", "0.5", "-0.0", "0.3"),
            "must be greater than 0.0 and at most 1.0.",
        ),
        (
            "height",
            ("0.5", "0.5", "0.2", "-0.0"),
            "must be greater than 0.0 and at most 1.0.",
        ),
    ],
)
def test_coordinate_range_errors_are_exact(
    field_name: str,
    values: tuple[str, str, str, str],
    expected: str,
    tmp_path: Path,
) -> None:
    """Apply the distinct center and dimension range rules."""

    result = validate_yolo_label(
        write_label(
            tmp_path / "labels.txt",
            f"0 {' '.join(values)}",
        )
    )

    assert result.errors == (
        f"Line 1: {field_name} {expected}",
    )


def test_one_line_collects_all_field_errors_in_order(
    tmp_path: Path,
) -> None:
    """Validate every eligible field in YOLO field order."""

    result = validate_yolo_label(
        write_label(
            tmp_path / "labels.txt",
            "-1 nan 2.0 0 -0.5",
        )
    )

    assert result.errors == (
        "Line 1: class_id must be a non-negative integer.",
        "Line 1: x_center must be a finite number.",
        (
            "Line 1: y_center must be between 0.0 and "
            "1.0 inclusive."
        ),
        (
            "Line 1: width must be greater than 0.0 and "
            "at most 1.0."
        ),
        (
            "Line 1: height must be greater than 0.0 and "
            "at most 1.0."
        ),
    )


def test_errors_follow_source_lines_and_valid_annotations_remain(
    tmp_path: Path,
) -> None:
    """Preserve global error order and partial parsing results."""

    label_path = write_label(
        tmp_path / "labels.txt",
        "0 0.5 0.5 0.2 0.3\n"
        "fish 0.5 2.0 0.2 0.3\n"
        "4 0.25 0.75 0.1 0.2\n"
        "1 0.5 0.5 0 -0.0\n",
    )

    result = validate_yolo_label(label_path)

    assert result.is_valid is False
    assert [item.class_id for item in result.annotations] == [0, 4]
    assert [item.line_number for item in result.annotations] == [1, 3]
    assert result.errors == (
        "Line 2: class_id must be a non-negative integer.",
        (
            "Line 2: y_center must be between 0.0 and "
            "1.0 inclusive."
        ),
        (
            "Line 4: width must be greater than 0.0 and "
            "at most 1.0."
        ),
        (
            "Line 4: height must be greater than 0.0 and "
            "at most 1.0."
        ),
    )


def test_models_are_frozen_and_slotted(tmp_path: Path) -> None:
    """Expose immutable models without normal instance dictionaries."""

    result = validate_yolo_label(
        write_label(
            tmp_path / "labels.txt",
            "0 0.5 0.5 0.2 0.3",
        )
    )
    annotation = result.annotations[0]

    with pytest.raises(FrozenInstanceError):
        annotation.class_id = 2
    with pytest.raises(FrozenInstanceError):
        result.is_valid = False
    assert not hasattr(annotation, "__dict__")
    assert not hasattr(result, "__dict__")


def test_repeated_validation_returns_independent_tuples(
    tmp_path: Path,
) -> None:
    """Create fresh immutable result collections on each call."""

    label_path = write_label(
        tmp_path / "labels.txt",
        "0 0.5 0.5 0.2 0.3\n-1 nan 2.0 0 -0.5\n",
    )

    first = validate_yolo_label(label_path)
    second = validate_yolo_label(label_path)

    assert first == second
    assert first.annotations is not second.annotations
    assert first.errors is not second.errors


def test_boundary_crossing_box_is_not_rejected(tmp_path: Path) -> None:
    """Validate direct fields without calculating image-boundary edges."""

    result = validate_yolo_label(
        write_label(
            tmp_path / "labels.txt",
            "0 0.05 0.50 0.20 0.25",
        )
    )

    assert result.is_valid is True
    assert len(result.annotations) == 1
    assert result.errors == ()
