"""Tests for in-memory YOLO configured-class validation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from poseidon_ai.nautilus_vision import (
    yolo_class_validation,
    yolo_config,
    yolo_dataset,
    yolo_label,
)
from poseidon_ai.nautilus_vision.yolo_class_validation import (
    YoloConfiguredClassUsage,
    YoloDatasetClassValidationResult,
    YoloUnknownClassOccurrence,
    validate_yolo_dataset_classes,
)
from poseidon_ai.nautilus_vision.yolo_config import (
    YoloClassDefinition,
    YoloDatasetConfiguration,
    validate_yolo_dataset_config,
)
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
)


def make_configuration(
    *definitions: YoloClassDefinition,
) -> YoloDatasetConfiguration:
    """Construct an immutable configuration without filesystem access."""

    classes = definitions or (
        YoloClassDefinition(class_id=0, name="fish"),
        YoloClassDefinition(class_id=1, name="Turtle Ray"),
        YoloClassDefinition(class_id=2, name="shark"),
    )
    return YoloDatasetConfiguration(
        config_path=Path("dataset/data.yaml"),
        dataset_root=Path("dataset"),
        train_path=Path("dataset/images/train"),
        validation_path=Path("dataset/images/val"),
        test_path=None,
        classes=tuple(classes),
    )


def make_annotation(
    class_id: int,
    line_number: int,
) -> YoloDetectionAnnotation:
    """Construct one parsed annotation with stable coordinates."""

    return YoloDetectionAnnotation(
        class_id=class_id,
        x_center=0.5,
        y_center=0.5,
        width=0.2,
        height=0.2,
        line_number=line_number,
    )


def make_pair(
    pairing_key: str,
    *annotations: YoloDetectionAnnotation,
    is_valid: bool = True,
    label_path: Path | None = None,
) -> YoloImageLabelPair:
    """Construct one ordinary pair and its retained validation result."""

    path = label_path or Path("labels") / f"{pairing_key}.txt"
    return YoloImageLabelPair(
        pairing_key=pairing_key,
        image_path=Path("images") / f"{pairing_key}.jpg",
        label_path=path,
        label_validation=YoloLabelValidationResult(
            is_valid=is_valid,
            annotations=tuple(annotations),
            errors=() if is_valid else ("retained label error",),
        ),
    )


def make_analysis(
    *pairs: YoloImageLabelPair,
    missing_label_images: tuple[Path, ...] = (),
    orphan_label_files: tuple[Path, ...] = (),
    pairing_conflicts: tuple[YoloPairingConflict, ...] = (),
) -> YoloDatasetAnalysisResult:
    """Construct analysis aggregates with Commit 42 counting semantics."""

    counts: dict[int, int] = {}
    valid_label_files = 0
    invalid_label_files = 0
    empty_label_files = 0
    for pair in pairs:
        validation = pair.label_validation
        if validation.is_valid:
            valid_label_files += 1
            if not validation.annotations:
                empty_label_files += 1
            for annotation in validation.annotations:
                counts[annotation.class_id] = (
                    counts.get(annotation.class_id, 0) + 1
                )
        else:
            invalid_label_files += 1

    class_counts = tuple(
        YoloClassCount(class_id=class_id, annotation_count=count)
        for class_id, count in sorted(counts.items())
    )
    return YoloDatasetAnalysisResult(
        image_directory=Path("images"),
        label_directory=Path("labels"),
        recursive=True,
        pairs=tuple(pairs),
        missing_label_images=missing_label_images,
        orphan_label_files=orphan_label_files,
        pairing_conflicts=pairing_conflicts,
        total_images=(
            len(pairs)
            + len(missing_label_images)
            + sum(
                len(conflict.image_paths)
                for conflict in pairing_conflicts
            )
        ),
        total_label_files=(
            len(pairs)
            + len(orphan_label_files)
            + sum(
                len(conflict.label_paths)
                for conflict in pairing_conflicts
            )
        ),
        valid_label_files=valid_label_files,
        invalid_label_files=invalid_label_files,
        empty_label_files=empty_label_files,
        total_annotations=sum(counts.values()),
        class_counts=class_counts,
    )


def test_public_models_are_frozen_slotted_and_tuple_backed() -> None:
    """Expose only immutable value models without instance dictionaries."""

    configuration = make_configuration()
    result = validate_yolo_dataset_classes(
        configuration,
        make_analysis(
            make_pair("unknown", make_annotation(7, 1))
        ),
    )
    usage = result.class_usage[0]
    occurrence = result.unknown_class_occurrences[0]

    with pytest.raises(FrozenInstanceError):
        usage.annotation_count = 99
    with pytest.raises(FrozenInstanceError):
        occurrence.class_id = 0
    with pytest.raises(FrozenInstanceError):
        result.is_valid = True
    with pytest.raises(TypeError):
        result.class_usage[0] = usage

    assert all(
        not hasattr(model, "__dict__")
        for model in (usage, occurrence, result)
    )
    assert isinstance(result.class_usage, tuple)
    assert isinstance(result.unknown_class_occurrences, tuple)
    assert isinstance(result.unobserved_classes, tuple)
    assert isinstance(result.errors, tuple)


def test_public_model_field_order_matches_contract() -> None:
    """Preserve the documented public field order."""

    assert tuple(YoloConfiguredClassUsage.__dataclass_fields__) == (
        "class_id",
        "name",
        "annotation_count",
    )
    assert tuple(YoloUnknownClassOccurrence.__dataclass_fields__) == (
        "class_id",
        "pairing_key",
        "label_path",
        "line_number",
        "label_is_valid",
    )
    assert tuple(YoloDatasetClassValidationResult.__dataclass_fields__) == (
        "is_valid",
        "class_usage",
        "unknown_class_occurrences",
        "unobserved_classes",
        "errors",
    )


def test_every_configured_class_has_ordered_exact_usage() -> None:
    """Return all classes in ID order without renormalizing names."""

    shark = YoloClassDefinition(class_id=2, name="Reef Shark")
    fish = YoloClassDefinition(class_id=0, name="Fish")
    turtle = YoloClassDefinition(class_id=1, name="green turtle")
    configuration = make_configuration(shark, fish, turtle)

    result = validate_yolo_dataset_classes(
        configuration,
        make_analysis(
            make_pair(
                "first",
                make_annotation(2, 1),
                make_annotation(0, 2),
                make_annotation(0, 3),
            ),
            make_pair("second", make_annotation(2, 1)),
            make_pair("empty"),
        ),
    )

    assert result.class_usage == (
        YoloConfiguredClassUsage(0, "Fish", 2),
        YoloConfiguredClassUsage(1, "green turtle", 0),
        YoloConfiguredClassUsage(2, "Reef Shark", 2),
    )


def test_invalid_labels_and_unpaired_diagnostics_add_no_usage() -> None:
    """Count no known annotations outside fully valid ordinary pairs."""

    conflict = YoloPairingConflict(
        pairing_key="conflict",
        image_paths=(Path("images/conflict.jpg"),),
        label_paths=(Path("labels/conflict.txt"), Path("labels/C.txt")),
    )
    analysis = make_analysis(
        make_pair(
            "invalid",
            make_annotation(0, 1),
            is_valid=False,
        ),
        missing_label_images=(Path("images/missing.jpg"),),
        orphan_label_files=(Path("labels/orphan.txt"),),
        pairing_conflicts=(conflict,),
    )

    result = validate_yolo_dataset_classes(
        make_configuration(),
        analysis,
    )

    assert tuple(item.annotation_count for item in result.class_usage) == (
        0,
        0,
        0,
    )
    assert result.unknown_class_occurrences == ()
    assert result.is_valid is True


def test_one_observed_class_leaves_other_classes_unobserved() -> None:
    """Treat zero usage as informational while preserving exact definitions."""

    configuration = make_configuration()
    result = validate_yolo_dataset_classes(
        configuration,
        make_analysis(make_pair("fish", make_annotation(0, 1))),
    )

    assert result.unobserved_classes == configuration.classes[1:]
    assert result.unobserved_classes[0] is configuration.classes[1]
    assert result.unobserved_classes[1] is configuration.classes[2]
    assert result.is_valid is True
    assert result.errors == ()


def test_usage_matches_filtered_dataset_class_counts() -> None:
    """Match Commit 42 aggregates for configured valid-label IDs only."""

    configuration = make_configuration()
    analysis = make_analysis(
        make_pair(
            "first",
            make_annotation(0, 1),
            make_annotation(2, 2),
            make_annotation(9, 3),
        ),
        make_pair("second", make_annotation(2, 1)),
        make_pair(
            "invalid",
            make_annotation(1, 1),
            make_annotation(8, 2),
            is_valid=False,
        ),
    )

    result = validate_yolo_dataset_classes(configuration, analysis)
    configured_ids = {
        definition.class_id for definition in configuration.classes
    }

    assert sum(item.annotation_count for item in result.class_usage) == sum(
        count.annotation_count
        for count in analysis.class_counts
        if count.class_id in configured_ids
    )
    assert analysis.total_annotations == 4
    assert sum(item.annotation_count for item in result.class_usage) == 3


def test_one_valid_unknown_produces_exact_complete_diagnostic() -> None:
    """Preserve pair, path, line, validity, ID, and exact error text."""

    label_path = Path("labels/train/fish.txt")
    result = validate_yolo_dataset_classes(
        make_configuration(),
        make_analysis(
            make_pair(
                "train/Fish",
                make_annotation(7, 3),
                label_path=label_path,
            )
        ),
    )

    assert result.unknown_class_occurrences == (
        YoloUnknownClassOccurrence(
            class_id=7,
            pairing_key="train/Fish",
            label_path=label_path,
            line_number=3,
            label_is_valid=True,
        ),
    )
    assert result.errors == (
        "Pair 'train/Fish', line 3: class_id 7 is not defined in the "
        "dataset configuration.",
    )
    assert result.is_valid is False


def test_invalid_label_unknown_is_reported_but_never_counted() -> None:
    """Inspect retained partial annotations without adding configured usage."""

    result = validate_yolo_dataset_classes(
        make_configuration(),
        make_analysis(
            make_pair(
                "partial",
                make_annotation(0, 1),
                make_annotation(8, 4),
                is_valid=False,
            )
        ),
    )

    assert result.unknown_class_occurrences[0].class_id == 8
    assert result.unknown_class_occurrences[0].line_number == 4
    assert result.unknown_class_occurrences[0].label_is_valid is False
    assert all(item.annotation_count == 0 for item in result.class_usage)
    assert result.unobserved_classes == make_configuration().classes


def test_repeated_unknown_annotations_are_not_deduplicated() -> None:
    """Retain each unknown annotation even when IDs and lines repeat."""

    result = validate_yolo_dataset_classes(
        make_configuration(),
        make_analysis(
            make_pair(
                "repeat",
                make_annotation(6, 2),
                make_annotation(6, 2),
                make_annotation(6, 3),
            )
        ),
    )

    assert [
        (occurrence.class_id, occurrence.line_number)
        for occurrence in result.unknown_class_occurrences
    ] == [(6, 2), (6, 2), (6, 3)]
    assert len(result.errors) == 3


def test_known_annotations_create_no_unknown_occurrences() -> None:
    """Return a valid result with no errors when every ID is configured."""

    result = validate_yolo_dataset_classes(
        make_configuration(),
        make_analysis(
            make_pair("known", make_annotation(0, 1), make_annotation(2, 2))
        ),
    )

    assert result.is_valid is True
    assert result.unknown_class_occurrences == ()
    assert result.errors == ()


def test_unknown_occurrence_order_is_fully_deterministic() -> None:
    """Sort by key, line, class ID, and finally POSIX label-path text."""

    pairs = (
        make_pair("zeta", make_annotation(5, 1)),
        make_pair(
            "alpha",
            make_annotation(9, 4),
            make_annotation(8, 4),
            make_annotation(7, 2),
            label_path=Path("labels/z.txt"),
        ),
        make_pair(
            "alpha",
            make_annotation(8, 4),
            label_path=Path("labels/a.txt"),
        ),
    )

    result = validate_yolo_dataset_classes(
        make_configuration(),
        make_analysis(*reversed(pairs)),
    )

    assert [
        (
            occurrence.pairing_key,
            occurrence.line_number,
            occurrence.class_id,
            occurrence.label_path.as_posix(),
        )
        for occurrence in result.unknown_class_occurrences
    ] == [
        ("alpha", 2, 7, "labels/z.txt"),
        ("alpha", 4, 8, "labels/a.txt"),
        ("alpha", 4, 8, "labels/z.txt"),
        ("alpha", 4, 9, "labels/z.txt"),
        ("zeta", 1, 5, "labels/zeta.txt"),
    ]
    assert result.errors == tuple(
        f"Pair '{item.pairing_key}', line {item.line_number}: class_id "
        f"{item.class_id} is not defined in the dataset configuration."
        for item in result.unknown_class_occurrences
    )


@pytest.mark.parametrize(
    "analysis",
    [
        make_analysis(),
        make_analysis(
            make_pair("invalid", make_annotation(0, 1), is_valid=False)
        ),
        make_analysis(make_pair("unknown", make_annotation(9, 1))),
    ],
    ids=("empty", "invalid-only", "unknown-only"),
)
def test_all_classes_are_unobserved_without_known_valid_usage(
    analysis: YoloDatasetAnalysisResult,
) -> None:
    """Return exact configured objects without invalidating zero usage."""

    configuration = make_configuration()
    result = validate_yolo_dataset_classes(configuration, analysis)

    assert result.unobserved_classes == configuration.classes
    assert all(
        actual is expected
        for actual, expected in zip(
            result.unobserved_classes,
            configuration.classes,
            strict=True,
        )
    )
    assert result.is_valid is (not result.unknown_class_occurrences)


def test_invalid_result_remains_complete() -> None:
    """Return usage, unknowns, unobserved classes, and errors on failure."""

    result = validate_yolo_dataset_classes(
        make_configuration(),
        make_analysis(
            make_pair(
                "mixed",
                make_annotation(0, 1),
                make_annotation(7, 2),
                make_annotation(1, 3),
                make_annotation(8, 4),
            )
        ),
    )

    assert result.is_valid is False
    assert tuple(item.annotation_count for item in result.class_usage) == (
        1,
        1,
        0,
    )
    assert [item.class_id for item in result.unknown_class_occurrences] == [
        7,
        8,
    ]
    assert result.unobserved_classes == (make_configuration().classes[2],)
    assert len(result.errors) == 2


def test_inputs_are_not_mutated() -> None:
    """Leave both supplied immutable object graphs unchanged."""

    configuration = make_configuration()
    analysis = make_analysis(
        make_pair("mixed", make_annotation(0, 1), make_annotation(7, 2))
    )
    original_configuration = configuration
    original_analysis = analysis

    validate_yolo_dataset_classes(configuration, analysis)

    assert configuration == original_configuration
    assert configuration is original_configuration
    assert analysis == original_analysis
    assert analysis is original_analysis


def test_composition_calls_no_parser_analyzer_or_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use only supplied models and lexical Path operations."""

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "unexpected parsing, analysis, or filesystem access"
        )

    monkeypatch.setattr(yolo_label, "validate_yolo_label", forbidden)
    monkeypatch.setattr(yolo_dataset, "analyze_yolo_dataset", forbidden)
    monkeypatch.setattr(yolo_config, "validate_yolo_dataset_config", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)

    result = validate_yolo_dataset_classes(
        make_configuration(),
        make_analysis(make_pair("fish", make_annotation(7, 1))),
    )

    assert result.unknown_class_occurrences[0].class_id == 7
    assert not hasattr(yolo_class_validation, "analyze_yolo_dataset")
    assert not hasattr(yolo_class_validation, "validate_yolo_label")


@pytest.mark.parametrize(
    "names_yaml",
    [
        "names:\n  - fish\n  - turtle\n  - shark\n",
        "names:\n  0: fish\n  1: turtle\n  2: shark\n",
    ],
    ids=("list", "mapping"),
)
def test_real_configuration_and_analysis_validate_known_ids(
    names_yaml: str,
    tmp_path: Path,
) -> None:
    """Compose real Commit 42 and 43 results for both names forms."""

    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    (image_root / "marine.jpg").write_bytes(b"arbitrary image bytes")
    (label_root / "marine.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "train: images\nval: images\n" + names_yaml,
        encoding="utf-8",
    )

    config_result = validate_yolo_dataset_config(config_path)
    analysis = analyze_yolo_dataset(image_root, label_root)
    assert config_result.configuration is not None
    configuration = config_result.configuration
    original_classes = configuration.classes
    original_counts = analysis.class_counts
    original_total = analysis.total_annotations

    result = validate_yolo_dataset_classes(configuration, analysis)

    assert result.is_valid is True
    assert tuple(item.annotation_count for item in result.class_usage) == (
        1,
        1,
        0,
    )
    assert result.unobserved_classes == (configuration.classes[2],)
    assert analysis.total_annotations == original_total == 2
    assert analysis.class_counts is original_counts
    assert configuration.classes is original_classes


def test_real_unknown_ids_from_valid_and_invalid_labels_are_reported(
    tmp_path: Path,
) -> None:
    """Retain real physical lines and distinguish containing-label validity."""

    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    for stem in ("valid", "invalid"):
        (image_root / f"{stem}.jpg").write_bytes(b"not decoded")
    (label_root / "valid.txt").write_text(
        "\n7 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (label_root / "invalid.txt").write_text(
        "8 0.5 0.5 0.2 0.2\n-1 nan 2.0 0 -0.5\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "train: images\nval: images\nnames: [fish, turtle, shark]\n",
        encoding="utf-8",
    )
    config_result = validate_yolo_dataset_config(config_path)
    analysis = analyze_yolo_dataset(image_root, label_root)
    assert config_result.configuration is not None

    result = validate_yolo_dataset_classes(
        config_result.configuration,
        analysis,
    )

    assert [
        (item.class_id, item.line_number, item.label_is_valid)
        for item in result.unknown_class_occurrences
    ] == [(8, 1, False), (7, 2, True)]
    assert all(item.annotation_count == 0 for item in result.class_usage)
    assert analysis.total_annotations == 1
    assert analysis.class_counts == (
        YoloClassCount(class_id=7, annotation_count=1),
    )


def test_unknown_classes_raise_no_exception() -> None:
    """Represent every incompatibility in the result rather than raising."""

    result = validate_yolo_dataset_classes(
        make_configuration(),
        make_analysis(
            make_pair(
                "many",
                *(
                    make_annotation(100 + i, i + 1)
                    for i in range(5)
                ),
            )
        ),
    )

    assert result.is_valid is False
    assert len(result.unknown_class_occurrences) == 5
    assert len(result.errors) == 5
