"""Tests for pure configured YOLO cross-split aggregation."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast
import builtins
import tomllib

import pytest

from poseidon_ai.nautilus_vision import yolo_split_summary
from poseidon_ai.nautilus_vision.yolo_class_validation import (
    YoloConfiguredClassUsage,
    YoloDatasetClassValidationResult,
    YoloUnknownClassOccurrence,
)
from poseidon_ai.nautilus_vision.yolo_config import (
    YoloClassDefinition,
    YoloDatasetConfiguration,
    validate_yolo_dataset_config,
)
from poseidon_ai.nautilus_vision.yolo_dataset import (
    YoloDatasetAnalysisResult,
    YoloImageLabelPair,
    YoloPairingConflict,
)
from poseidon_ai.nautilus_vision.yolo_label import (
    YoloLabelValidationResult,
)
from poseidon_ai.nautilus_vision.yolo_split_analysis import (
    YoloConfiguredSplitAnalysisResult,
    YoloDatasetSplitAnalysis,
    YoloDatasetSplitAnalysisFailure,
    analyze_yolo_dataset_splits,
)
from poseidon_ai.nautilus_vision.yolo_split_plan import (
    YoloDatasetSplit,
    YoloDatasetSplitPlan,
    build_yolo_dataset_split_plan,
)
from poseidon_ai.nautilus_vision.yolo_split_summary import (
    YoloCrossSplitClassUsage,
    YoloCrossSplitSummary,
    YoloSplitSummary,
    summarize_yolo_dataset_splits,
)


def make_configuration(
    *,
    config_path: Path = Path("dataset/data.yaml"),
    classes: tuple[YoloClassDefinition, ...] | None = None,
) -> YoloDatasetConfiguration:
    """Construct one validated-shape immutable configuration."""

    definitions = classes or (
        YoloClassDefinition(0, "fish"),
        YoloClassDefinition(1, "turtle"),
        YoloClassDefinition(2, "shark"),
    )
    return YoloDatasetConfiguration(
        config_path=config_path,
        dataset_root=Path("dataset"),
        train_path=Path("dataset/train/images"),
        validation_path=Path("dataset/validation/images"),
        test_path=Path("dataset/test/images"),
        classes=definitions,
    )


def make_split(name: str) -> YoloDatasetSplit:
    """Construct one explicit planned split."""

    return YoloDatasetSplit(
        name=name,
        image_directory=Path(name) / "images",
        label_directory=Path(name) / "labels",
    )


def make_pair(index: int) -> YoloImageLabelPair:
    """Construct one pair whose nested data is never reopened."""

    return YoloImageLabelPair(
        pairing_key=f"pair-{index}",
        image_path=Path("images") / f"pair-{index}.jpg",
        label_path=Path("labels") / f"pair-{index}.txt",
        label_validation=YoloLabelValidationResult(
            is_valid=True,
            annotations=(),
            errors=(),
        ),
    )


def make_analysis(
    *,
    total_images: int = 0,
    total_label_files: int = 0,
    paired_images: int = 0,
    missing_label_images: int = 0,
    orphan_label_files: int = 0,
    pairing_conflicts: int = 0,
    valid_label_files: int = 0,
    invalid_label_files: int = 0,
    empty_label_files: int = 0,
    total_annotations: int = 0,
) -> YoloDatasetAnalysisResult:
    """Construct counts and diagnostic tuple lengths independently."""

    return YoloDatasetAnalysisResult(
        image_directory=Path("images"),
        label_directory=Path("labels"),
        recursive=False,
        pairs=tuple(make_pair(index) for index in range(paired_images)),
        missing_label_images=tuple(
            Path("images") / f"missing-{index}.jpg"
            for index in range(missing_label_images)
        ),
        orphan_label_files=tuple(
            Path("labels") / f"orphan-{index}.txt"
            for index in range(orphan_label_files)
        ),
        pairing_conflicts=tuple(
            YoloPairingConflict(
                pairing_key=f"conflict-{index}",
                image_paths=(
                    Path("images") / f"conflict-{index}.jpg",
                    Path("images") / f"conflict-{index}.png",
                ),
                label_paths=(
                    Path("labels") / f"conflict-{index}.txt",
                ),
            )
            for index in range(pairing_conflicts)
        ),
        total_images=total_images,
        total_label_files=total_label_files,
        valid_label_files=valid_label_files,
        invalid_label_files=invalid_label_files,
        empty_label_files=empty_label_files,
        total_annotations=total_annotations,
        class_counts=(),
    )


def make_occurrence(
    class_id: int,
    *,
    label_is_valid: bool,
    index: int,
) -> YoloUnknownClassOccurrence:
    """Construct one retained unknown-class diagnostic."""

    return YoloUnknownClassOccurrence(
        class_id=class_id,
        pairing_key=f"unknown-{index}",
        label_path=Path("labels") / f"unknown-{index}.txt",
        line_number=index + 1,
        label_is_valid=label_is_valid,
    )


def make_validation(
    configuration: YoloDatasetConfiguration,
    *,
    counts: tuple[int, ...] = (0, 0, 0),
    unknown_valid: int = 0,
    unknown_invalid: int = 0,
) -> YoloDatasetClassValidationResult:
    """Construct class usage and unknowns with existing public models."""

    usage = tuple(
        YoloConfiguredClassUsage(
            class_id=definition.class_id,
            name=definition.name,
            annotation_count=count,
        )
        for definition, count in zip(
            configuration.classes,
            counts,
            strict=True,
        )
    )
    occurrences = tuple(
        make_occurrence(7, label_is_valid=True, index=index)
        for index in range(unknown_valid)
    ) + tuple(
        make_occurrence(
            8,
            label_is_valid=False,
            index=unknown_valid + index,
        )
        for index in range(unknown_invalid)
    )
    return YoloDatasetClassValidationResult(
        is_valid=not occurrences,
        class_usage=usage,
        unknown_class_occurrences=occurrences,
        unobserved_classes=tuple(
            definition
            for definition, count in zip(
                configuration.classes,
                counts,
                strict=True,
            )
            if count == 0
        ),
        errors=tuple("unknown" for _ in occurrences),
    )


def make_success(
    configuration: YoloDatasetConfiguration,
    name: str,
    *,
    analysis: YoloDatasetAnalysisResult | None = None,
    validation: YoloDatasetClassValidationResult | None = None,
) -> YoloDatasetSplitAnalysis:
    """Construct one successful configured split-analysis outcome."""

    return YoloDatasetSplitAnalysis(
        split=make_split(name),
        dataset_analysis=analysis or make_analysis(),
        class_validation=(
            validation or make_validation(configuration)
        ),
    )


def make_failure(name: str) -> YoloDatasetSplitAnalysisFailure:
    """Construct one expected root-failure outcome."""

    return YoloDatasetSplitAnalysisFailure(
        split=make_split(name),
        error_type="FileNotFoundError",
        message=f"missing {name}",
    )


def make_split_analysis(
    configuration: YoloDatasetConfiguration,
    *outcomes: YoloDatasetSplitAnalysis | YoloDatasetSplitAnalysisFailure,
    recursive: bool = False,
) -> YoloConfiguredSplitAnalysisResult:
    """Construct one ordered completed split-analysis result."""

    return YoloConfiguredSplitAnalysisResult(
        config_path=configuration.config_path,
        recursive=recursive,
        outcomes=tuple(outcomes),
    )


def test_public_models_match_immutable_field_contract() -> None:
    """Expose frozen slotted models with exact field ordering."""

    configuration = make_configuration()
    result = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(
            configuration,
            make_success(configuration, "train"),
        ),
    )
    split_summary = result.successful_summaries[0]
    usage = result.class_usage[0]

    with pytest.raises(FrozenInstanceError):
        split_summary.total_images = 1
    with pytest.raises(FrozenInstanceError):
        usage.annotation_count = 1
    with pytest.raises(FrozenInstanceError):
        result.total_images = 1
    with pytest.raises(TypeError):
        result.outcomes[0] = make_failure("test")

    assert all(
        not hasattr(model, "__dict__")
        for model in (split_summary, usage, result)
    )
    assert tuple(YoloSplitSummary.__dataclass_fields__) == (
        "split",
        "total_images",
        "total_label_files",
        "paired_images",
        "missing_label_images",
        "orphan_label_files",
        "pairing_conflicts",
        "valid_label_files",
        "invalid_label_files",
        "empty_label_files",
        "total_annotations",
        "configured_annotation_count",
        "unknown_class_occurrences",
        "unknown_class_occurrences_in_valid_labels",
        "unknown_class_occurrences_in_invalid_labels",
        "class_validation_is_valid",
        "class_usage",
        "unobserved_classes",
    )
    assert tuple(YoloCrossSplitClassUsage.__dataclass_fields__) == (
        "class_id",
        "name",
        "annotation_count",
        "observed_split_names",
    )
    assert tuple(YoloCrossSplitSummary.__dataclass_fields__) == (
        "config_path",
        "recursive",
        "outcomes",
        "total_images",
        "total_label_files",
        "total_paired_images",
        "total_missing_label_images",
        "total_orphan_label_files",
        "total_pairing_conflicts",
        "total_valid_label_files",
        "total_invalid_label_files",
        "total_empty_label_files",
        "total_annotations",
        "configured_annotation_count",
        "unknown_class_occurrences",
        "unknown_class_occurrences_in_valid_labels",
        "unknown_class_occurrences_in_invalid_labels",
        "class_usage",
        "unobserved_classes",
    )


def test_repeated_calls_return_independent_immutable_outer_models() -> None:
    """Build fresh summaries while retaining required nested identity."""

    configuration = make_configuration()
    success = make_success(configuration, "train")
    split_analysis = make_split_analysis(configuration, success)

    first = summarize_yolo_dataset_splits(configuration, split_analysis)
    second = summarize_yolo_dataset_splits(configuration, split_analysis)

    assert first == second
    assert first is not second
    assert first.outcomes is not second.outcomes
    assert first.outcomes[0] is not second.outcomes[0]
    assert first.class_usage is not second.class_usage
    assert first.successful_summaries[0].class_usage is (
        success.class_validation.class_usage
    )


def test_matching_config_paths_preserve_exact_path_and_recursive() -> None:
    """Accept exact lexical path equality without normalizing inputs."""

    config_path = Path("dataset/../config/data.yaml")
    configuration = make_configuration(config_path=config_path)
    split_analysis = YoloConfiguredSplitAnalysisResult(
        config_path=config_path,
        recursive=True,
        outcomes=(),
    )

    result = summarize_yolo_dataset_splits(configuration, split_analysis)

    assert result.config_path is split_analysis.config_path
    assert result.recursive is True


def test_mismatch_raises_exact_error_before_outcome_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject incompatible sources before reading outcomes or doing work."""

    configuration = make_configuration(
        config_path=Path("first/data.yaml")
    )

    class MismatchedAnalysis:
        config_path = Path("second/data.yaml")

        @property
        def outcomes(self) -> object:
            raise AssertionError("outcomes were inspected")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected filesystem or analysis call")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(
        yolo_split_summary,
        "analyze_yolo_dataset_splits",
        forbidden,
        raising=False,
    )

    with pytest.raises(ValueError) as error:
        summarize_yolo_dataset_splits(
            configuration,
            cast(YoloConfiguredSplitAnalysisResult, MismatchedAnalysis()),
        )

    assert str(error.value) == (
        "Split analysis config_path does not match the dataset "
        "configuration config_path."
    )


def test_successes_are_flattened_and_failures_preserve_identity_order(
) -> None:
    """Replace only successful outcomes while retaining failure objects."""

    configuration = make_configuration()
    first_failure = make_failure("test")
    success = make_success(configuration, "train")
    second_failure = YoloDatasetSplitAnalysisFailure(
        split=make_split("validation"),
        error_type="NotADirectoryError",
        message="validation is a file",
    )
    result = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(
            configuration,
            first_failure,
            success,
            second_failure,
        ),
    )

    assert isinstance(result.outcomes[0], YoloDatasetSplitAnalysisFailure)
    assert isinstance(result.outcomes[1], YoloSplitSummary)
    assert isinstance(result.outcomes[2], YoloDatasetSplitAnalysisFailure)
    assert result.outcomes[0] is first_failure
    assert result.outcomes[2] is second_failure
    assert result.outcomes[1].split is success.split
    assert result.outcomes[1].class_usage is (
        success.class_validation.class_usage
    )
    assert result.outcomes[1].unobserved_classes is (
        success.class_validation.unobserved_classes
    )
    assert first_failure.error_type == "FileNotFoundError"
    assert first_failure.message == "missing test"


def test_per_split_counts_copy_and_derive_exact_semantics() -> None:
    """Copy aggregates and count diagnostic groups and occurrences."""

    configuration = make_configuration()
    analysis = make_analysis(
        total_images=14,
        total_label_files=13,
        paired_images=5,
        missing_label_images=2,
        orphan_label_files=3,
        pairing_conflicts=2,
        valid_label_files=3,
        invalid_label_files=2,
        empty_label_files=1,
        total_annotations=9,
    )
    validation = make_validation(
        configuration,
        counts=(2, 3, 1),
        unknown_valid=3,
        unknown_invalid=2,
    )
    summary = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(
            configuration,
            make_success(
                configuration,
                "train",
                analysis=analysis,
                validation=validation,
            ),
        ),
    ).successful_summaries[0]

    assert (
        summary.total_images,
        summary.total_label_files,
        summary.paired_images,
        summary.missing_label_images,
        summary.orphan_label_files,
        summary.pairing_conflicts,
    ) == (14, 13, 5, 2, 3, 2)
    assert (
        summary.valid_label_files,
        summary.invalid_label_files,
        summary.empty_label_files,
        summary.total_annotations,
    ) == (3, 2, 1, 9)
    assert summary.configured_annotation_count == 6
    assert summary.unknown_class_occurrences == 5
    assert summary.unknown_class_occurrences_in_valid_labels == 3
    assert summary.unknown_class_occurrences_in_invalid_labels == 2
    assert summary.class_validation_is_valid is False
    assert (
        summary.configured_annotation_count
        + summary.unknown_class_occurrences_in_valid_labels
        == summary.total_annotations
    )


def test_multiple_successes_sum_every_global_count() -> None:
    """Sum each independent count across successful summaries only."""

    configuration = make_configuration()
    first = make_success(
        configuration,
        "validation",
        analysis=make_analysis(
            total_images=8,
            total_label_files=7,
            paired_images=4,
            missing_label_images=1,
            orphan_label_files=2,
            pairing_conflicts=1,
            valid_label_files=3,
            invalid_label_files=1,
            empty_label_files=1,
            total_annotations=4,
        ),
        validation=make_validation(
            configuration,
            counts=(2, 1, 0),
            unknown_valid=1,
            unknown_invalid=2,
        ),
    )
    second = make_success(
        configuration,
        "train",
        analysis=make_analysis(
            total_images=3,
            total_label_files=4,
            paired_images=2,
            missing_label_images=1,
            orphan_label_files=1,
            pairing_conflicts=2,
            valid_label_files=1,
            invalid_label_files=1,
            empty_label_files=0,
            total_annotations=3,
        ),
        validation=make_validation(
            configuration,
            counts=(0, 2, 0),
            unknown_valid=1,
            unknown_invalid=1,
        ),
    )
    result = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(
            configuration,
            first,
            make_failure("test"),
            second,
        ),
    )

    assert (
        result.total_images,
        result.total_label_files,
        result.total_paired_images,
        result.total_missing_label_images,
        result.total_orphan_label_files,
        result.total_pairing_conflicts,
    ) == (11, 11, 6, 2, 3, 3)
    assert (
        result.total_valid_label_files,
        result.total_invalid_label_files,
        result.total_empty_label_files,
        result.total_annotations,
    ) == (4, 2, 1, 7)
    assert result.configured_annotation_count == 5
    assert result.unknown_class_occurrences == 5
    assert result.unknown_class_occurrences_in_valid_labels == 2
    assert result.unknown_class_occurrences_in_invalid_labels == 3
    assert result.total_annotations == sum(
        summary.total_annotations
        for summary in result.successful_summaries
    )
    assert (
        result.configured_annotation_count
        + result.unknown_class_occurrences_in_valid_labels
        == result.total_annotations
    )


def test_all_failed_result_has_zero_totals_and_all_classes_unobserved(
) -> None:
    """Exclude every failed split from totals and configured usage."""

    configuration = make_configuration()
    failures = (make_failure("validation"), make_failure("train"))
    result = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(configuration, *failures),
    )

    numeric_fields = (
        result.total_images,
        result.total_label_files,
        result.total_paired_images,
        result.total_missing_label_images,
        result.total_orphan_label_files,
        result.total_pairing_conflicts,
        result.total_valid_label_files,
        result.total_invalid_label_files,
        result.total_empty_label_files,
        result.total_annotations,
        result.configured_annotation_count,
        result.unknown_class_occurrences,
        result.unknown_class_occurrences_in_valid_labels,
        result.unknown_class_occurrences_in_invalid_labels,
    )
    assert all(value == 0 for value in numeric_fields)
    assert result.unobserved_classes == configuration.classes
    assert all(
        actual is expected
        for actual, expected in zip(
            result.unobserved_classes,
            configuration.classes,
            strict=True,
        )
    )
    assert result.failed_splits == failures
    assert result.is_complete is False


def test_empty_success_is_complete_but_every_class_is_unobserved() -> None:
    """Keep operational completeness independent of class coverage."""

    configuration = make_configuration()
    result = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(
            configuration,
            make_success(configuration, "train"),
        ),
    )

    assert result.is_complete is True
    assert result.total_images == 0
    assert result.configured_annotation_count == 0
    assert result.unobserved_classes == configuration.classes
    assert all(not usage.observed_split_names for usage in result.class_usage)


def test_configured_class_aggregation_uses_configuration_order_and_names(
) -> None:
    """Aggregate usage canonically while preserving outcome-name order."""

    definitions = (
        YoloClassDefinition(2, "Shark From Config"),
        YoloClassDefinition(0, "Fish From Config"),
        YoloClassDefinition(1, "Turtle From Config"),
    )
    configuration = make_configuration(classes=definitions)

    def validation(
        counts: tuple[int, ...],
    ) -> YoloDatasetClassValidationResult:
        result = make_validation(configuration, counts=counts)
        return YoloDatasetClassValidationResult(
            is_valid=True,
            class_usage=tuple(
                YoloConfiguredClassUsage(
                    usage.class_id,
                    f"ignored-{usage.name}",
                    usage.annotation_count,
                )
                for usage in result.class_usage
            ),
            unknown_class_occurrences=(),
            unobserved_classes=result.unobserved_classes,
            errors=(),
        )

    outcomes = (
        make_success(
            configuration,
            "test",
            validation=validation((1, 0, 4)),
        ),
        make_failure("failed"),
        make_success(
            configuration,
            "train",
            validation=validation((2, 3, 0)),
        ),
        make_success(
            configuration,
            "train",
            validation=validation((1, 1, 0)),
        ),
    )
    result = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(configuration, *outcomes),
    )

    assert result.class_usage == (
        YoloCrossSplitClassUsage(
            2,
            "Shark From Config",
            4,
            ("test", "train"),
        ),
        YoloCrossSplitClassUsage(0, "Fish From Config", 4, ("train",)),
        YoloCrossSplitClassUsage(1, "Turtle From Config", 4, ("test",)),
    )
    assert all(
        "failed" not in item.observed_split_names
        for item in result.class_usage
    )
    assert result.unobserved_classes == ()


@pytest.mark.parametrize(
    ("counts", "unknown_valid", "unknown_invalid"),
    [
        ((0, 0, 0), 1, 0),
        ((0, 0, 0), 0, 1),
        ((0, 0, 0), 1, 1),
    ],
    ids=("unknown-valid", "known-invalid-only", "both-unknown-kinds"),
)
def test_unknown_or_invalid_only_annotations_leave_classes_unobserved(
    counts: tuple[int, ...],
    unknown_valid: int,
    unknown_invalid: int,
) -> None:
    """Never translate unknown or invalid-label diagnostics into usage."""

    configuration = make_configuration()
    validation = make_validation(
        configuration,
        counts=counts,
        unknown_valid=unknown_valid,
        unknown_invalid=unknown_invalid,
    )
    total_annotations = unknown_valid
    result = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(
            configuration,
            make_success(
                configuration,
                "train",
                analysis=make_analysis(
                    total_annotations=total_annotations
                ),
                validation=validation,
            ),
        ),
    )

    assert result.configured_annotation_count == 0
    assert result.total_annotations == unknown_valid
    assert result.unobserved_classes == configuration.classes
    assert result.is_complete is True


def test_derived_properties_preserve_order_identity_without_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filter exact nested outcomes and derive counts without any I/O."""

    configuration = make_configuration()
    source = make_split_analysis(
        configuration,
        make_failure("test"),
        make_success(configuration, "validation"),
        make_failure("train"),
    )
    result = summarize_yolo_dataset_splits(configuration, source)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("derived property accessed filesystem")

    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "is_dir", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)

    assert result.successful_split_count == 1
    assert result.failed_split_count == 2
    assert result.successful_summaries[0] is result.outcomes[1]
    assert result.failed_splits[0] is result.outcomes[0]
    assert result.failed_splits[1] is result.outcomes[2]
    assert result.is_complete is False


def test_summary_layer_invokes_no_io_or_domain_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read only supplied immutable values at the composition boundary."""

    configuration = make_configuration()
    source = make_split_analysis(
        configuration,
        make_success(configuration, "train"),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected I/O or domain execution")

    with monkeypatch.context() as patcher:
        for attribute in (
            "validate_yolo_dataset_config",
            "build_yolo_dataset_split_plan",
            "analyze_yolo_dataset_splits",
            "analyze_yolo_dataset",
            "validate_yolo_dataset_classes",
            "validate_yolo_label",
        ):
            patcher.setattr(
                yolo_split_summary,
                attribute,
                forbidden,
                raising=False,
            )
        patcher.setattr(builtins, "open", forbidden)
        for attribute in (
            "open",
            "read_text",
            "exists",
            "is_file",
            "is_dir",
            "iterdir",
            "glob",
            "rglob",
        ):
            patcher.setattr(Path, attribute, forbidden)

        result = summarize_yolo_dataset_splits(configuration, source)

    assert result.successful_split_count == 1
    assert not hasattr(yolo_split_summary, "validate_yolo_dataset_config")
    assert not hasattr(yolo_split_summary, "build_yolo_dataset_split_plan")
    assert not hasattr(yolo_split_summary, "analyze_yolo_dataset_splits")
    assert not hasattr(yolo_split_summary, "analyze_yolo_dataset")
    assert not hasattr(yolo_split_summary, "validate_yolo_dataset_classes")
    assert not hasattr(yolo_split_summary, "validate_yolo_label")


def test_inputs_nested_results_and_failures_are_not_mutated() -> None:
    """Leave the complete supplied object graph unchanged and unretained."""

    configuration = make_configuration()
    success = make_success(configuration, "train")
    failure = make_failure("test")
    source = make_split_analysis(configuration, success, failure)
    original_outcomes = source.outcomes
    original_classes = configuration.classes

    result = summarize_yolo_dataset_splits(configuration, source)

    assert source.outcomes is original_outcomes
    assert source.outcomes == (success, failure)
    assert configuration.classes is original_classes
    assert result.failed_splits[0] is failure
    assert result.successful_summaries[0].split is success.split
    assert not hasattr(result.successful_summaries[0], "dataset_analysis")
    assert not hasattr(result.successful_summaries[0], "class_validation")
    assert not hasattr(result, "configuration")
    assert not hasattr(result, "split_analysis")
    assert not hasattr(result, "is_training_ready")
    assert not hasattr(result, "leakage")


def write_image(path: Path) -> None:
    """Write arbitrary supported-image bytes without decoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"arbitrary image bytes")


def write_label(path: Path, text: str) -> None:
    """Write one explicit UTF-8 label fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_real_config_plan_and_analysis_produce_expected_summary(
    tmp_path: Path,
) -> None:
    """Compose real prior layers once and aggregate their outcomes."""

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "path: .\n"
        "train: train/images\n"
        "val: validation/images\n"
        "test: test/images\n"
        "names: [fish, turtle, shark]\n",
        encoding="utf-8",
    )
    train_images = tmp_path / "train" / "images"
    train_labels = tmp_path / "train" / "labels"
    write_image(train_images / "known.jpg")
    write_label(
        train_labels / "known.txt",
        "0 0.5 0.5 0.2 0.2\n0 0.4 0.4 0.1 0.1\n",
    )
    write_image(train_images / "empty.jpg")
    write_label(train_labels / "empty.txt", "")

    val_images = tmp_path / "validation" / "images"
    val_labels = tmp_path / "validation" / "labels"
    write_image(val_images / "known.jpg")
    write_label(val_labels / "known.txt", "1 0.5 0.5 0.2 0.2\n")
    write_image(val_images / "unknown.jpg")
    write_label(val_labels / "unknown.txt", "7 0.5 0.5 0.2 0.2\n")
    write_image(val_images / "invalid.jpg")
    write_label(
        val_labels / "invalid.txt",
        "8 0.5 0.5 0.2 0.2\ninvalid line\n",
    )
    write_image(val_images / "missing.jpg")
    write_label(val_labels / "orphan.txt", "0 0.5 0.5 0.2 0.2\n")
    write_image(val_images / "conflict.jpg")
    write_image(val_images / "conflict.png")
    write_label(val_labels / "conflict.txt", "0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "test" / "images").mkdir(parents=True)

    config_result = validate_yolo_dataset_config(config_path)
    assert config_result.configuration is not None
    configuration = config_result.configuration
    plan_result = build_yolo_dataset_split_plan(configuration)
    assert plan_result.plan is not None
    split_analysis = analyze_yolo_dataset_splits(
        configuration,
        plan_result.plan,
    )
    original_outcomes = split_analysis.outcomes
    original_failure = split_analysis.failed_splits[0]

    result = summarize_yolo_dataset_splits(
        configuration,
        split_analysis,
    )

    assert [outcome.split.name for outcome in result.outcomes] == [
        "train",
        "validation",
        "test",
    ]
    assert [item.split.name for item in result.successful_summaries] == [
        "train",
        "validation",
    ]
    assert result.failed_splits[0] is original_failure
    assert split_analysis.outcomes is original_outcomes
    assert (
        result.total_images,
        result.total_label_files,
        result.total_paired_images,
        result.total_missing_label_images,
        result.total_orphan_label_files,
        result.total_pairing_conflicts,
    ) == (8, 7, 5, 1, 1, 1)
    assert (
        result.total_valid_label_files,
        result.total_invalid_label_files,
        result.total_empty_label_files,
        result.total_annotations,
    ) == (4, 1, 1, 4)
    assert result.configured_annotation_count == 3
    assert result.unknown_class_occurrences == 2
    assert result.unknown_class_occurrences_in_valid_labels == 1
    assert result.unknown_class_occurrences_in_invalid_labels == 1
    assert result.class_usage == (
        YoloCrossSplitClassUsage(0, "fish", 2, ("train",)),
        YoloCrossSplitClassUsage(1, "turtle", 1, ("validation",)),
        YoloCrossSplitClassUsage(2, "shark", 0, ()),
    )
    assert result.unobserved_classes == (configuration.classes[2],)
    assert result.unobserved_classes[0] is configuration.classes[2]
    assert result.is_complete is False


def test_real_optional_test_is_included_when_successful(
    tmp_path: Path,
) -> None:
    """Retain a successful optional test split in aggregate ordering."""

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "path: .\n"
        "train: train/images\n"
        "val: validation/images\n"
        "test: test/images\n"
        "names: [fish]\n",
        encoding="utf-8",
    )
    for name in ("train", "validation", "test"):
        (tmp_path / name / "images").mkdir(parents=True)
        (tmp_path / name / "labels").mkdir(parents=True)

    config_result = validate_yolo_dataset_config(config_path)
    assert config_result.configuration is not None
    configuration = config_result.configuration
    plan_result = build_yolo_dataset_split_plan(configuration)
    assert plan_result.plan is not None
    analysis = analyze_yolo_dataset_splits(
        configuration,
        plan_result.plan,
    )
    result = summarize_yolo_dataset_splits(configuration, analysis)

    assert [item.split.name for item in result.successful_summaries] == [
        "train",
        "validation",
        "test",
    ]
    assert result.successful_split_count == 3
    assert result.failed_split_count == 0
    assert result.is_complete is True
    assert result.total_images == 0
    assert result.unobserved_classes == configuration.classes


def test_custom_outcome_order_is_preserved_in_observed_names() -> None:
    """Never sort outcomes or observed split names alphabetically."""

    configuration = make_configuration()
    outcomes = tuple(
        make_success(
            configuration,
            name,
            validation=make_validation(
                configuration,
                counts=(1, 0, 0),
            ),
        )
        for name in ("test", "train", "validation")
    )

    result = summarize_yolo_dataset_splits(
        configuration,
        make_split_analysis(configuration, *outcomes),
    )

    assert [item.split.name for item in result.successful_summaries] == [
        "test",
        "train",
        "validation",
    ]
    assert result.class_usage[0].observed_split_names == (
        "test",
        "train",
        "validation",
    )


def test_project_commands_dependencies_and_existing_schemas_are_unchanged(
) -> None:
    """Add no CLI, dependency, report, or manifest integration."""

    with Path("pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["scripts"] == {
        "poseidon-inspect": "poseidon_ai.nautilus_vision.inspect_image:main",
        "poseidon-dataset-summary": (
            "poseidon_ai.nautilus_vision.dataset_summary:main"
        ),
    }
    assert project["project"]["dependencies"] == [
        "numpy>=2.1,<3.0",
        "opencv-python>=4.10,<5.0",
        "PyYAML>=6.0.2,<7.0",
    ]
    assert not hasattr(yolo_split_summary, "main")
    assert not hasattr(yolo_split_summary, "format_summary")
    assert not hasattr(yolo_split_summary, "write_manifest")
