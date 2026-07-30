"""Tests for library-level YOLO dataset analysis."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from poseidon_ai.nautilus_vision import dataset_loader, yolo_dataset
from poseidon_ai.nautilus_vision.yolo_dataset import (
    YoloClassCount,
    YoloDatasetAnalysisResult,
    YoloImageLabelPair,
    YoloPairingConflict,
    analyze_yolo_dataset,
)
from poseidon_ai.nautilus_vision.yolo_label import (
    YoloDetectionAnnotation,
    YoloLabelValidationResult,
    validate_yolo_label,
)


def create_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Create and return separate image and label roots."""

    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    return image_root, label_root


def write_image(path: Path) -> Path:
    """Write arbitrary bytes to one supported image candidate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not decoded image content")
    return path


def write_label(path: Path, content: str = "") -> Path:
    """Write one UTF-8 YOLO label candidate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_image_directory_raises_exact_error(
    tmp_path: Path,
) -> None:
    """Reject a missing image root using normal exception semantics."""

    image_root = tmp_path / "missing-images"
    label_root = tmp_path / "missing-labels"

    with pytest.raises(FileNotFoundError) as error:
        analyze_yolo_dataset(image_root, label_root)

    assert str(error.value) == (
        f"Image directory does not exist: {image_root}"
    )


def test_image_file_path_raises_exact_error(tmp_path: Path) -> None:
    """Reject a regular file supplied as the image root."""

    image_root = write_image(tmp_path / "image.jpg")
    label_root = tmp_path / "labels"
    label_root.mkdir()

    with pytest.raises(NotADirectoryError) as error:
        analyze_yolo_dataset(image_root, label_root)

    assert str(error.value) == (
        f"Image path is not a directory: {image_root}"
    )


def test_missing_label_directory_raises_exact_error(
    tmp_path: Path,
) -> None:
    """Validate an existing image root before rejecting a missing label root."""

    image_root = tmp_path / "images"
    image_root.mkdir()
    label_root = tmp_path / "missing-labels"

    with pytest.raises(FileNotFoundError) as error:
        analyze_yolo_dataset(image_root, label_root)

    assert str(error.value) == (
        f"Label directory does not exist: {label_root}"
    )


def test_label_file_path_raises_exact_error(tmp_path: Path) -> None:
    """Reject a regular file supplied as the label root."""

    image_root = tmp_path / "images"
    image_root.mkdir()
    label_root = write_label(tmp_path / "labels.txt")

    with pytest.raises(NotADirectoryError) as error:
        analyze_yolo_dataset(image_root, label_root)

    assert str(error.value) == (
        f"Label path is not a directory: {label_root}"
    )


def test_image_root_validation_precedes_label_root_validation(
    tmp_path: Path,
) -> None:
    """Report the image-root failure when both roots are missing."""

    image_root = tmp_path / "missing-images"
    label_root = tmp_path / "missing-labels"

    with pytest.raises(FileNotFoundError) as error:
        analyze_yolo_dataset(image_root, label_root)

    assert str(error.value) == (
        f"Image directory does not exist: {image_root}"
    )


def test_unexpected_discovery_error_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow unexpected filesystem failures to propagate."""

    image_root, label_root = create_roots(tmp_path)

    def deny_discovery(
        directory: str | Path,
        *,
        recursive: bool,
        validate: bool,
    ) -> list[Path]:
        raise PermissionError("discovery denied")

    monkeypatch.setattr(
        yolo_dataset,
        "load_image_dataset",
        deny_discovery,
    )

    with pytest.raises(PermissionError, match="discovery denied"):
        analyze_yolo_dataset(image_root, label_root)


def test_direct_discovery_accepts_supported_case_insensitive_files(
    tmp_path: Path,
) -> None:
    """Discover direct images and TXT labels while ignoring other files."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "alpha.JPG")
    write_image(image_root / "beta.png")
    write_image(image_root / "ignored.gif")
    write_label(label_root / "alpha.TXT")
    write_label(label_root / "beta.txt")
    write_label(label_root / "ignored.csv")

    result = analyze_yolo_dataset(image_root, label_root)

    assert [pair.pairing_key for pair in result.pairs] == [
        "alpha",
        "beta",
    ]
    assert result.total_images == 2
    assert result.total_label_files == 2


def test_supported_looking_directories_are_ignored(
    tmp_path: Path,
) -> None:
    """Do not classify directories as image or label files."""

    image_root, label_root = create_roots(tmp_path)
    (image_root / "fake.jpg").mkdir()
    (label_root / "fake.txt").mkdir()

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.total_images == 0
    assert result.total_label_files == 0
    assert result.pairs == ()


def test_image_discovery_never_validates_or_decodes_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse discovery with validation disabled for arbitrary image bytes."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "fish.jpg")
    write_label(label_root / "fish.txt")

    def fail_validation(*args: object, **kwargs: object) -> object:
        raise AssertionError("image validation was requested")

    monkeypatch.setattr(
        dataset_loader,
        "validate_image",
        fail_validation,
    )

    result = analyze_yolo_dataset(image_root, label_root)

    assert len(result.pairs) == 1


def test_nonrecursive_analysis_ignores_nested_files(
    tmp_path: Path,
) -> None:
    """Apply top-level-only discovery to both roots by default."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "root.jpg")
    write_label(label_root / "root.txt")
    write_image(image_root / "nested" / "fish.jpg")
    write_label(label_root / "nested" / "fish.txt")

    result = analyze_yolo_dataset(image_root, label_root)

    assert [pair.pairing_key for pair in result.pairs] == ["root"]
    assert result.recursive is False


def test_recursive_analysis_pairs_nested_relative_paths(
    tmp_path: Path,
) -> None:
    """Discover nested candidates under both explicitly supplied roots."""

    image_root, label_root = create_roots(tmp_path)
    image = write_image(image_root / "train" / "fish.jpg")
    label = write_label(
        label_root / "train" / "fish.TXT",
        "0 0.5 0.5 0.2 0.3\n",
    )

    result = analyze_yolo_dataset(
        str(image_root),
        str(label_root),
        recursive=True,
    )

    assert result.recursive is True
    assert result.image_directory == image_root
    assert result.label_directory == label_root
    assert result.pairs[0].pairing_key == "train/fish"
    assert result.pairs[0].image_path == image
    assert result.pairs[0].label_path == label


def test_identical_nested_stems_remain_independent(
    tmp_path: Path,
) -> None:
    """Preserve relative directories instead of pairing by filename alone."""

    image_root, label_root = create_roots(tmp_path)
    for directory in ("train", "validation"):
        write_image(image_root / directory / "fish.jpg")
        write_label(label_root / directory / "fish.txt")

    result = analyze_yolo_dataset(
        image_root,
        label_root,
        recursive=True,
    )

    assert [pair.pairing_key for pair in result.pairs] == [
        "train/fish",
        "validation/fish",
    ]


def test_pairing_key_removes_only_final_suffix_and_uses_posix_paths(
    tmp_path: Path,
) -> None:
    """Retain suffix-like stem segments and portable separators."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "archive" / "deep" / "fish.v1.jpeg")
    write_label(label_root / "archive" / "deep" / "fish.v1.txt")

    result = analyze_yolo_dataset(
        image_root,
        label_root,
        recursive=True,
    )

    assert result.pairs[0].pairing_key == "archive/deep/fish.v1"
    assert "\\" not in result.pairs[0].pairing_key


def test_pairing_keys_are_case_sensitive(tmp_path: Path) -> None:
    """Do not lowercase relative stems across separate roots."""

    image_root, label_root = create_roots(tmp_path)
    image = write_image(image_root / "Fish.jpg")
    label = write_label(label_root / "fish.txt")

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.pairs == ()
    assert result.missing_label_images == (image,)
    assert result.orphan_label_files == (label,)


def test_different_image_extensions_create_pairing_conflict(
    tmp_path: Path,
) -> None:
    """Never choose one of multiple supported images sharing a stem."""

    image_root, label_root = create_roots(tmp_path)
    jpg = write_image(image_root / "fish.jpg")
    png = write_image(image_root / "fish.png")
    label = write_label(label_root / "fish.txt")

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.pairs == ()
    assert result.missing_label_images == ()
    assert result.orphan_label_files == ()
    assert result.pairing_conflicts == (
        YoloPairingConflict(
            pairing_key="fish",
            image_paths=(jpg, png),
            label_paths=(label,),
        ),
    )


def test_missing_and_orphan_paths_are_deterministically_ordered(
    tmp_path: Path,
) -> None:
    """Sort each diagnostic collection by its root-relative POSIX path."""

    image_root, label_root = create_roots(tmp_path)
    image_a = write_image(image_root / "a.jpg")
    image_a_b = write_image(image_root / "a.b.jpg")
    label_z = write_label(label_root / "z.txt")
    label_z_a = write_label(label_root / "z.a.txt")
    write_image(image_root / "ignored.gif")
    write_label(label_root / "ignored.csv")

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.missing_label_images == (image_a_b, image_a)
    assert result.orphan_label_files == (label_z_a, label_z)


def test_multiple_images_without_label_are_one_conflict(
    tmp_path: Path,
) -> None:
    """Give conflict classification precedence over missing diagnostics."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "fish.png")
    write_image(image_root / "fish.jpg")

    result = analyze_yolo_dataset(image_root, label_root)

    assert len(result.pairing_conflicts) == 1
    assert result.pairing_conflicts[0].pairing_key == "fish"
    assert [path.suffix for path in result.pairing_conflicts[0].image_paths] == [
        ".jpg",
        ".png",
    ]
    assert result.missing_label_images == ()


def test_multiple_label_discovery_results_create_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Represent case-only duplicate label candidates without discarding one."""

    image_root, label_root = create_roots(tmp_path)
    image = write_image(image_root / "fish.jpg")
    lower = write_label(label_root / "fish.txt")
    upper = label_root / "fish.TXT"

    monkeypatch.setattr(
        yolo_dataset,
        "_discover_label_files",
        lambda root, recursive: [upper, lower],
    )

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.pairs == ()
    assert result.pairing_conflicts == (
        YoloPairingConflict(
            pairing_key="fish",
            image_paths=(image,),
            label_paths=(upper, lower),
        ),
    )
    assert result.orphan_label_files == ()


def test_multiple_labels_without_image_are_one_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give label conflict classification precedence over orphan diagnostics."""

    image_root, label_root = create_roots(tmp_path)
    lower = write_label(label_root / "fish.txt")
    upper = label_root / "fish.TXT"

    monkeypatch.setattr(
        yolo_dataset,
        "_discover_label_files",
        lambda root, recursive: [lower, upper],
    )

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.pairs == ()
    assert result.orphan_label_files == ()
    assert result.pairing_conflicts == (
        YoloPairingConflict(
            pairing_key="fish",
            image_paths=(),
            label_paths=(upper, lower),
        ),
    )


def test_conflicts_are_ordered_and_never_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sort conflicts and skip every label under ambiguous keys."""

    image_root, label_root = create_roots(tmp_path)
    for stem in ("zebra", "alpha"):
        write_image(image_root / f"{stem}.png")
        write_image(image_root / f"{stem}.jpg")
        write_label(label_root / f"{stem}.txt", "invalid")

    def fail_validation(path: str | Path) -> YoloLabelValidationResult:
        raise AssertionError(f"conflicting label was validated: {path}")

    monkeypatch.setattr(
        yolo_dataset,
        "validate_yolo_label",
        fail_validation,
    )

    result = analyze_yolo_dataset(image_root, label_root)

    assert [conflict.pairing_key for conflict in result.pairing_conflicts] == [
        "alpha",
        "zebra",
    ]
    assert result.pairs == ()
    assert result.valid_label_files == 0
    assert result.invalid_label_files == 0
    assert result.total_annotations == 0


def test_orphan_label_is_not_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover orphan labels without parsing their contents."""

    image_root, label_root = create_roots(tmp_path)
    orphan = write_label(label_root / "orphan.txt", "invalid")

    def fail_validation(path: str | Path) -> YoloLabelValidationResult:
        raise AssertionError(f"orphan label was validated: {path}")

    monkeypatch.setattr(
        yolo_dataset,
        "validate_yolo_label",
        fail_validation,
    )

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.orphan_label_files == (orphan,)
    assert result.total_label_files == 1
    assert result.valid_label_files == 0
    assert result.invalid_label_files == 0


def test_each_ordinary_pair_is_validated_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the exact validator result and avoid repeated parsing."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "fish.jpg")
    label = write_label(label_root / "fish.txt")
    validation = YoloLabelValidationResult(
        is_valid=True,
        annotations=(),
        errors=(),
    )
    calls: list[Path] = []

    def track_validation(path: str | Path) -> YoloLabelValidationResult:
        calls.append(Path(path))
        return validation

    monkeypatch.setattr(
        yolo_dataset,
        "validate_yolo_label",
        track_validation,
    )

    result = analyze_yolo_dataset(image_root, label_root)

    assert calls == [label]
    assert result.pairs[0].label_validation is validation


def test_valid_invalid_empty_and_blank_pairs_preserve_validation(
    tmp_path: Path,
) -> None:
    """Continue across invalid labels and classify paired files correctly."""

    image_root, label_root = create_roots(tmp_path)
    for stem in ("valid", "invalid", "empty", "blank"):
        write_image(image_root / f"{stem}.jpg")
    write_label(
        label_root / "valid.txt",
        "3 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8\n",
    )
    write_label(
        label_root / "invalid.txt",
        "5 0.5 0.5 0.2 0.3\n\n-1 nan 2.0 0 -0.5\n",
    )
    write_label(label_root / "empty.txt")
    write_label(label_root / "blank.txt", " \n\t\n")

    result = analyze_yolo_dataset(image_root, label_root)
    pairs = {pair.pairing_key: pair for pair in result.pairs}

    assert [pair.pairing_key for pair in result.pairs] == [
        "blank",
        "empty",
        "invalid",
        "valid",
    ]
    assert pairs["valid"].label_validation.is_valid is True
    assert [
        annotation.class_id
        for annotation in pairs["valid"].label_validation.annotations
    ] == [3, 1]
    assert [
        annotation.line_number
        for annotation in pairs["invalid"].label_validation.annotations
    ] == [1]
    assert len(pairs["invalid"].label_validation.errors) == 5
    assert pairs["empty"].label_validation.annotations == ()
    assert pairs["blank"].label_validation.annotations == ()
    assert result.valid_label_files == 3
    assert result.invalid_label_files == 1
    assert result.empty_label_files == 2


def test_annotation_statistics_include_only_fully_valid_labels(
    tmp_path: Path,
) -> None:
    """Aggregate duplicate lines and classes while excluding partial parses."""

    image_root, label_root = create_roots(tmp_path)
    for stem in ("first", "second", "invalid", "empty"):
        write_image(image_root / f"{stem}.jpg")
    write_label(
        label_root / "first.txt",
        "10 0.5 0.5 0.2 0.3\n"
        "2 0.4 0.4 0.1 0.1\n"
        "2 0.4 0.4 0.1 0.1\n",
    )
    write_label(
        label_root / "second.txt",
        "10 0.25 0.75 0.1 0.2\n",
    )
    write_label(
        label_root / "invalid.txt",
        "99 0.5 0.5 0.2 0.3\n-1 nan 2.0 0 -0.5\n",
    )
    write_label(label_root / "empty.txt")

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.total_annotations == 4
    assert result.class_counts == (
        YoloClassCount(class_id=2, annotation_count=2),
        YoloClassCount(class_id=10, annotation_count=2),
    )
    assert all(
        isinstance(class_count.class_id, int)
        and isinstance(class_count.annotation_count, int)
        for class_count in result.class_counts
    )
    assert sum(
        class_count.annotation_count
        for class_count in result.class_counts
    ) == result.total_annotations


def test_orphans_conflicts_and_missing_images_add_no_annotations(
    tmp_path: Path,
) -> None:
    """Exclude every non-paired classification from aggregate statistics."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "missing.jpg")
    write_label(
        label_root / "orphan.txt",
        "1 0.5 0.5 0.2 0.3\n",
    )
    write_image(image_root / "conflict.jpg")
    write_image(image_root / "conflict.png")
    write_label(
        label_root / "conflict.txt",
        "2 0.5 0.5 0.2 0.3\n",
    )

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.total_annotations == 0
    assert result.class_counts == ()
    assert result.valid_label_files == 0
    assert result.invalid_label_files == 0
    assert result.empty_label_files == 0


def test_result_classification_relationships_hold(
    tmp_path: Path,
) -> None:
    """Account for every discovered image and label exactly once."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "paired.jpg")
    write_label(
        label_root / "paired.txt",
        "0 0.5 0.5 0.2 0.3\n",
    )
    write_image(image_root / "missing.jpg")
    write_label(label_root / "orphan.txt")
    write_image(image_root / "conflict.jpg")
    write_image(image_root / "conflict.png")
    write_label(label_root / "conflict.txt")

    result = analyze_yolo_dataset(image_root, label_root)

    assert result.total_images == (
        len(result.pairs)
        + len(result.missing_label_images)
        + sum(
            len(conflict.image_paths)
            for conflict in result.pairing_conflicts
        )
    )
    assert result.total_label_files == (
        len(result.pairs)
        + len(result.orphan_label_files)
        + sum(
            len(conflict.label_paths)
            for conflict in result.pairing_conflicts
        )
    )
    assert (
        result.valid_label_files + result.invalid_label_files
        == len(result.pairs)
    )
    assert result.empty_label_files <= result.valid_label_files


def test_empty_dataset_returns_zero_counts_and_empty_tuples(
    tmp_path: Path,
) -> None:
    """Represent empty roots without mutable collection defaults."""

    image_root, label_root = create_roots(tmp_path)

    result = analyze_yolo_dataset(image_root, label_root)

    assert result == YoloDatasetAnalysisResult(
        image_directory=image_root,
        label_directory=label_root,
        recursive=False,
        pairs=(),
        missing_label_images=(),
        orphan_label_files=(),
        pairing_conflicts=(),
        total_images=0,
        total_label_files=0,
        valid_label_files=0,
        invalid_label_files=0,
        empty_label_files=0,
        total_annotations=0,
        class_counts=(),
    )


def test_dataset_models_are_frozen_slotted_and_tuple_backed(
    tmp_path: Path,
) -> None:
    """Expose immutable public models without normal instance dictionaries."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "fish.jpg")
    write_label(
        label_root / "fish.txt",
        "0 0.5 0.5 0.2 0.3\n",
    )
    write_image(image_root / "conflict.jpg")
    write_image(image_root / "conflict.png")
    write_label(label_root / "conflict.txt")

    result = analyze_yolo_dataset(image_root, label_root)
    models = (
        result,
        result.pairs[0],
        result.pairing_conflicts[0],
        result.class_counts[0],
    )

    with pytest.raises(FrozenInstanceError):
        result.total_images = 99
    with pytest.raises(TypeError):
        result.pairs[0] = result.pairs[0]
    assert all(not hasattr(model, "__dict__") for model in models)
    assert isinstance(result.pairs, tuple)
    assert isinstance(result.missing_label_images, tuple)
    assert isinstance(result.orphan_label_files, tuple)
    assert isinstance(result.pairing_conflicts, tuple)
    assert isinstance(result.class_counts, tuple)


def test_repeated_analysis_returns_independent_results(
    tmp_path: Path,
) -> None:
    """Create fresh result, pair, validation, and public tuples per call."""

    image_root, label_root = create_roots(tmp_path)
    write_image(image_root / "fish.jpg")
    write_label(
        label_root / "fish.txt",
        "0 0.5 0.5 0.2 0.3\n",
    )

    first = analyze_yolo_dataset(image_root, label_root)
    second = analyze_yolo_dataset(image_root, label_root)

    assert first == second
    assert first is not second
    assert first.pairs is not second.pairs
    assert first.pairs[0] is not second.pairs[0]
    assert (
        first.pairs[0].label_validation
        is not second.pairs[0].label_validation
    )
    assert first.class_counts is not second.class_counts


def test_image_and_label_roots_may_be_the_same_directory(
    tmp_path: Path,
) -> None:
    """Require no class map, data YAML, or inferred directory layout."""

    write_image(tmp_path / "fish.jpg")
    write_label(
        tmp_path / "fish.txt",
        "8 0.5 0.5 0.2 0.3\n",
    )

    result = analyze_yolo_dataset(tmp_path, tmp_path)

    assert result.total_annotations == 1
    assert result.class_counts == (
        YoloClassCount(class_id=8, annotation_count=1),
    )


def test_single_file_validator_remains_independently_usable(
    tmp_path: Path,
) -> None:
    """Protect the existing single-file validation boundary."""

    label = write_label(
        tmp_path / "single.txt",
        "0 0.05 0.50 0.20 0.25\n",
    )

    result = validate_yolo_label(label)

    assert result.is_valid is True
    assert result.annotations == (
        YoloDetectionAnnotation(
            class_id=0,
            x_center=0.05,
            y_center=0.5,
            width=0.2,
            height=0.25,
            line_number=1,
        ),
    )


def test_public_model_field_order_matches_contract() -> None:
    """Preserve the stable field ordering of every public value model."""

    assert tuple(YoloImageLabelPair.__dataclass_fields__) == (
        "pairing_key",
        "image_path",
        "label_path",
        "label_validation",
    )
    assert tuple(YoloPairingConflict.__dataclass_fields__) == (
        "pairing_key",
        "image_paths",
        "label_paths",
    )
    assert tuple(YoloClassCount.__dataclass_fields__) == (
        "class_id",
        "annotation_count",
    )
    assert tuple(YoloDatasetAnalysisResult.__dataclass_fields__) == (
        "image_directory",
        "label_directory",
        "recursive",
        "pairs",
        "missing_label_images",
        "orphan_label_files",
        "pairing_conflicts",
        "total_images",
        "total_label_files",
        "valid_label_files",
        "invalid_label_files",
        "empty_label_files",
        "total_annotations",
        "class_counts",
    )
