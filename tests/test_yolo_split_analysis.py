"""Tests for ordered configured YOLO split analysis."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import tomllib

import pytest

from poseidon_ai.nautilus_vision import (
    dataset_loader,
    yolo_class_validation,
    yolo_config,
    yolo_label,
    yolo_split_analysis,
    yolo_split_plan,
)
from poseidon_ai.nautilus_vision.yolo_class_validation import (
    YoloConfiguredClassUsage,
    YoloDatasetClassValidationResult,
)
from poseidon_ai.nautilus_vision.yolo_config import (
    YoloClassDefinition,
    YoloDatasetConfiguration,
)
from poseidon_ai.nautilus_vision.yolo_dataset import (
    YoloClassCount,
    YoloDatasetAnalysisResult,
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
)


def make_configuration(
    *,
    config_path: Path = Path("dataset/data.yaml"),
) -> YoloDatasetConfiguration:
    """Construct one validated-shape configuration for focused tests."""

    return YoloDatasetConfiguration(
        config_path=config_path,
        dataset_root=Path("dataset"),
        train_path=Path("dataset/images/train"),
        validation_path=Path("dataset/images/validation"),
        test_path=Path("dataset/images/test"),
        classes=(
            YoloClassDefinition(class_id=0, name="fish"),
            YoloClassDefinition(class_id=1, name="turtle"),
        ),
    )


def make_split(
    name: str,
    *,
    image_directory: Path | None = None,
    label_directory: Path | None = None,
) -> YoloDatasetSplit:
    """Construct one explicit planned split."""

    return YoloDatasetSplit(
        name=name,
        image_directory=(
            image_directory or Path("images") / name
        ),
        label_directory=(
            label_directory or Path("labels") / name
        ),
    )


def make_plan(
    *splits: YoloDatasetSplit,
    config_path: Path = Path("dataset/data.yaml"),
) -> YoloDatasetSplitPlan:
    """Construct one immutable plan with caller-controlled ordering."""

    return YoloDatasetSplitPlan(
        config_path=config_path,
        splits=tuple(splits),
    )


def make_dataset_analysis(
    split: YoloDatasetSplit,
) -> YoloDatasetAnalysisResult:
    """Construct one successful empty dataset-analysis result."""

    return YoloDatasetAnalysisResult(
        image_directory=split.image_directory,
        label_directory=split.label_directory,
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


def make_class_validation(
    *,
    is_valid: bool = True,
) -> YoloDatasetClassValidationResult:
    """Construct one immutable configured-class result."""

    return YoloDatasetClassValidationResult(
        is_valid=is_valid,
        class_usage=(
            YoloConfiguredClassUsage(
                class_id=0,
                name="fish",
                annotation_count=0,
            ),
        ),
        unknown_class_occurrences=(),
        unobserved_classes=(),
        errors=() if is_valid else ("unknown class",),
    )


def test_public_models_are_frozen_slotted_and_tuple_backed() -> None:
    """Expose immutable models without normal instance dictionaries."""

    split = make_split("train")
    success = YoloDatasetSplitAnalysis(
        split=split,
        dataset_analysis=make_dataset_analysis(split),
        class_validation=make_class_validation(),
    )
    failure = YoloDatasetSplitAnalysisFailure(
        split=make_split("validation"),
        error_type="FileNotFoundError",
        message="missing",
    )
    result = YoloConfiguredSplitAnalysisResult(
        config_path=Path("dataset/data.yaml"),
        recursive=False,
        outcomes=(success, failure),
    )

    with pytest.raises(FrozenInstanceError):
        success.split = failure.split
    with pytest.raises(FrozenInstanceError):
        failure.message = "changed"
    with pytest.raises(FrozenInstanceError):
        result.recursive = True
    with pytest.raises(TypeError):
        result.outcomes[0] = failure

    assert isinstance(result.outcomes, tuple)
    assert all(
        not hasattr(model, "__dict__")
        for model in (success, failure, result)
    )


def test_public_model_field_order_matches_contract() -> None:
    """Preserve the documented public model field order."""

    assert tuple(YoloDatasetSplitAnalysis.__dataclass_fields__) == (
        "split",
        "dataset_analysis",
        "class_validation",
    )
    assert tuple(YoloDatasetSplitAnalysisFailure.__dataclass_fields__) == (
        "split",
        "error_type",
        "message",
    )
    assert tuple(YoloConfiguredSplitAnalysisResult.__dataclass_fields__) == (
        "config_path",
        "recursive",
        "outcomes",
    )


def test_config_path_mismatch_raises_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unrelated configurations and plans with the exact error."""

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("split execution must not start")

    monkeypatch.setattr(yolo_split_analysis, "analyze_yolo_dataset", forbidden)
    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        forbidden,
    )
    configuration = make_configuration(
        config_path=Path("dataset/first.yaml")
    )
    plan = make_plan(
        make_split("train"),
        config_path=Path("dataset/second.yaml"),
    )

    with pytest.raises(ValueError) as error:
        analyze_yolo_dataset_splits(configuration, plan)

    assert str(error.value) == (
        "Split plan config_path does not match the dataset "
        "configuration config_path."
    )


def test_config_paths_are_compared_without_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept exact lexical equality without resolving either path."""

    config_path = Path("dataset/../config/data.yaml")
    configuration = make_configuration(config_path=config_path)
    plan = make_plan(config_path=Path("dataset/../config/data.yaml"))

    def forbidden(path: Path) -> Path:
        raise AssertionError(f"unexpected resolution: {path}")

    monkeypatch.setattr(Path, "resolve", forbidden)

    result = analyze_yolo_dataset_splits(configuration, plan)

    assert result.config_path is plan.config_path
    assert result.outcomes == ()
    assert result.is_complete is True


@pytest.mark.parametrize("recursive", [False, True])
def test_delegation_preserves_order_paths_results_and_recursive(
    recursive: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call both existing components once per successful planned split."""

    splits = (
        make_split("test"),
        make_split("train"),
        make_split("validation"),
    )
    configuration = make_configuration()
    plan = make_plan(*splits)
    analyses = tuple(make_dataset_analysis(split) for split in splits)
    validations = tuple(make_class_validation() for _ in splits)
    analysis_calls: list[tuple[Path, Path, bool]] = []
    validation_calls: list[
        tuple[YoloDatasetConfiguration, YoloDatasetAnalysisResult]
    ] = []

    def fake_analysis(
        image_directory: str | Path,
        label_directory: str | Path,
        *,
        recursive: bool,
    ) -> YoloDatasetAnalysisResult:
        assert isinstance(image_directory, Path)
        assert isinstance(label_directory, Path)
        analysis_calls.append(
            (image_directory, label_directory, recursive)
        )
        return analyses[len(analysis_calls) - 1]

    def fake_validation(
        supplied_configuration: YoloDatasetConfiguration,
        dataset_analysis: YoloDatasetAnalysisResult,
    ) -> YoloDatasetClassValidationResult:
        validation_calls.append(
            (supplied_configuration, dataset_analysis)
        )
        return validations[len(validation_calls) - 1]

    monkeypatch.setattr(
        yolo_split_analysis,
        "analyze_yolo_dataset",
        fake_analysis,
    )
    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        fake_validation,
    )

    result = analyze_yolo_dataset_splits(
        configuration,
        plan,
        recursive=recursive,
    )

    assert result.config_path is plan.config_path
    assert result.recursive is recursive
    assert analysis_calls == [
        (split.image_directory, split.label_directory, recursive)
        for split in splits
    ]
    assert all(
        actual_image is split.image_directory
        and actual_label is split.label_directory
        for (actual_image, actual_label, _), split in zip(
            analysis_calls,
            splits,
            strict=True,
        )
    )
    assert validation_calls == [
        (configuration, analysis) for analysis in analyses
    ]
    assert len(result.outcomes) == 3
    assert all(
        isinstance(outcome, YoloDatasetSplitAnalysis)
        for outcome in result.outcomes
    )
    for index, outcome in enumerate(result.successful_splits):
        assert outcome.split is splits[index]
        assert outcome.dataset_analysis is analyses[index]
        assert outcome.class_validation is validations[index]


def test_repeated_calls_return_independent_outer_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create fresh result, outcome tuple, and outcome models per call."""

    split = make_split("train")
    plan = make_plan(split)
    analysis = make_dataset_analysis(split)
    validation = make_class_validation()
    monkeypatch.setattr(
        yolo_split_analysis,
        "analyze_yolo_dataset",
        lambda *args, **kwargs: analysis,
    )
    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        lambda *args, **kwargs: validation,
    )

    first = analyze_yolo_dataset_splits(make_configuration(), plan)
    second = analyze_yolo_dataset_splits(make_configuration(), plan)

    assert first == second
    assert first is not second
    assert first.outcomes is not second.outcomes
    assert first.outcomes[0] is not second.outcomes[0]


def test_empty_manual_plan_returns_complete_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow a manually supplied empty plan without runtime assertions."""

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("empty plans execute no components")

    monkeypatch.setattr(yolo_split_analysis, "analyze_yolo_dataset", forbidden)
    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        forbidden,
    )

    result = analyze_yolo_dataset_splits(
        make_configuration(),
        make_plan(),
    )

    assert result.outcomes == ()
    assert result.successful_splits == ()
    assert result.failed_splits == ()
    assert result.is_complete is True


@pytest.mark.parametrize(
    "raised_error",
    [
        FileNotFoundError("root missing"),
        NotADirectoryError("root is a file"),
    ],
    ids=("missing", "not-directory"),
)
def test_expected_root_error_becomes_exact_failure(
    raised_error: OSError,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture only expected root failures without class validation."""

    split = make_split("train")
    validation_calls = 0

    def fail_analysis(*args: object, **kwargs: object) -> object:
        raise raised_error

    def track_validation(*args: object, **kwargs: object) -> object:
        nonlocal validation_calls
        validation_calls += 1
        return make_class_validation()

    monkeypatch.setattr(
        yolo_split_analysis,
        "analyze_yolo_dataset",
        fail_analysis,
    )
    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        track_validation,
    )

    result = analyze_yolo_dataset_splits(
        make_configuration(),
        make_plan(split),
    )

    assert result.outcomes == (
        YoloDatasetSplitAnalysisFailure(
            split=split,
            error_type=type(raised_error).__name__,
            message=str(raised_error),
        ),
    )
    assert result.outcomes[0].split is split
    assert validation_calls == 0
    assert result.is_complete is False


def test_expected_failures_continue_in_plan_order_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain multiple failures while continuing through later splits."""

    splits = tuple(
        make_split(name) for name in ("train", "validation", "test")
    )
    plan = make_plan(*splits)
    validation_analysis = make_dataset_analysis(splits[1])
    analysis_results: list[object] = [
        FileNotFoundError("train missing"),
        validation_analysis,
        NotADirectoryError("test is a file"),
    ]
    analysis_calls: list[str] = []
    validation_calls: list[YoloDatasetAnalysisResult] = []

    def fake_analysis(
        image_directory: str | Path,
        label_directory: str | Path,
        *,
        recursive: bool,
    ) -> YoloDatasetAnalysisResult:
        analysis_calls.append(str(image_directory))
        next_result = analysis_results[len(analysis_calls) - 1]
        if isinstance(next_result, BaseException):
            raise next_result
        assert isinstance(next_result, YoloDatasetAnalysisResult)
        return next_result

    def fake_validation(
        configuration: YoloDatasetConfiguration,
        dataset_analysis: YoloDatasetAnalysisResult,
    ) -> YoloDatasetClassValidationResult:
        validation_calls.append(dataset_analysis)
        return make_class_validation()

    monkeypatch.setattr(
        yolo_split_analysis,
        "analyze_yolo_dataset",
        fake_analysis,
    )
    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        fake_validation,
    )

    result = analyze_yolo_dataset_splits(make_configuration(), plan)

    assert len(analysis_calls) == 3
    assert validation_calls == [validation_analysis]
    assert [outcome.split for outcome in result.outcomes] == list(splits)
    assert [failure.split for failure in result.failed_splits] == [
        splits[0],
        splits[2],
    ]
    assert result.successful_splits[0].split is splits[1]
    assert result.is_complete is False


@pytest.mark.parametrize(
    "raised_error",
    [
        PermissionError("permission denied"),
        OSError("unexpected operating-system failure"),
        KeyboardInterrupt(),
    ],
    ids=("permission", "os-error", "keyboard-interrupt"),
)
def test_unexpected_analysis_failures_propagate(
    raised_error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid broad exception swallowing around dataset analysis."""

    def fail_analysis(*args: object, **kwargs: object) -> object:
        raise raised_error

    monkeypatch.setattr(
        yolo_split_analysis,
        "analyze_yolo_dataset",
        fail_analysis,
    )

    with pytest.raises(type(raised_error)) as error:
        analyze_yolo_dataset_splits(
            make_configuration(),
            make_plan(make_split("train")),
        )

    assert error.value is raised_error


def test_unexpected_class_validation_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate configured-class validation failures unchanged."""

    split = make_split("train")
    analysis = make_dataset_analysis(split)
    raised_error = RuntimeError("class validation failed")
    monkeypatch.setattr(
        yolo_split_analysis,
        "analyze_yolo_dataset",
        lambda *args, **kwargs: analysis,
    )

    def fail_validation(*args: object, **kwargs: object) -> object:
        raise raised_error

    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        fail_validation,
    )

    with pytest.raises(RuntimeError) as error:
        analyze_yolo_dataset_splits(
            make_configuration(),
            make_plan(split),
        )

    assert error.value is raised_error


def test_derived_properties_preserve_exact_outcomes_without_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filter immutable outcomes in their existing order and identity."""

    first_success = YoloDatasetSplitAnalysis(
        split=make_split("test"),
        dataset_analysis=make_dataset_analysis(make_split("test")),
        class_validation=make_class_validation(is_valid=False),
    )
    first_failure = YoloDatasetSplitAnalysisFailure(
        split=make_split("train"),
        error_type="FileNotFoundError",
        message="missing",
    )
    second_success = YoloDatasetSplitAnalysis(
        split=make_split("validation"),
        dataset_analysis=make_dataset_analysis(make_split("validation")),
        class_validation=make_class_validation(),
    )
    second_failure = YoloDatasetSplitAnalysisFailure(
        split=make_split("custom"),
        error_type="NotADirectoryError",
        message="not a directory",
    )
    result = YoloConfiguredSplitAnalysisResult(
        config_path=Path("data.yaml"),
        recursive=False,
        outcomes=(
            first_success,
            first_failure,
            second_success,
            second_failure,
        ),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("derived properties accessed the filesystem")

    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "is_dir", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)

    assert result.is_complete is False
    assert result.successful_splits == (first_success, second_success)
    assert result.failed_splits == (first_failure, second_failure)
    assert result.successful_splits[0] is first_success
    assert result.successful_splits[1] is second_success
    assert result.failed_splits[0] is first_failure
    assert result.failed_splits[1] is second_failure
    assert first_success.class_validation.is_valid is False


def test_all_class_invalid_outcomes_are_operationally_complete() -> None:
    """Keep annotation compatibility distinct from split completion."""

    split = make_split("train")
    outcome = YoloDatasetSplitAnalysis(
        split=split,
        dataset_analysis=make_dataset_analysis(split),
        class_validation=make_class_validation(is_valid=False),
    )
    result = YoloConfiguredSplitAnalysisResult(
        config_path=Path("data.yaml"),
        recursive=False,
        outcomes=(outcome,),
    )

    assert result.is_complete is True
    assert result.successful_splits == (outcome,)
    assert result.failed_splits == ()


def test_orchestrator_delegates_without_direct_lower_level_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid YAML, planning, label parsing, and direct discovery calls."""

    split = make_split("train")
    analysis = make_dataset_analysis(split)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected lower-level call")

    monkeypatch.setattr(yolo_config, "validate_yolo_dataset_config", forbidden)
    monkeypatch.setattr(
        yolo_split_plan,
        "build_yolo_dataset_split_plan",
        forbidden,
    )
    monkeypatch.setattr(yolo_label, "validate_yolo_label", forbidden)
    monkeypatch.setattr(dataset_loader, "load_image_dataset", forbidden)
    monkeypatch.setattr(
        yolo_split_analysis,
        "analyze_yolo_dataset",
        lambda *args, **kwargs: analysis,
    )
    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        lambda *args, **kwargs: make_class_validation(),
    )

    result = analyze_yolo_dataset_splits(
        make_configuration(),
        make_plan(split),
    )

    assert result.is_complete is True
    assert not hasattr(yolo_split_analysis, "validate_yolo_dataset_config")
    assert not hasattr(yolo_split_analysis, "build_yolo_dataset_split_plan")
    assert not hasattr(yolo_split_analysis, "validate_yolo_label")
    assert not hasattr(yolo_split_analysis, "load_image_dataset")


def test_inputs_and_nested_dependency_results_are_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve configuration, plan, analysis, and validation objects."""

    configuration = make_configuration()
    split = make_split("train")
    plan = make_plan(split, config_path=configuration.config_path)
    analysis = make_dataset_analysis(split)
    validation = make_class_validation()
    monkeypatch.setattr(
        yolo_split_analysis,
        "analyze_yolo_dataset",
        lambda *args, **kwargs: analysis,
    )
    monkeypatch.setattr(
        yolo_split_analysis,
        "validate_yolo_dataset_classes",
        lambda *args, **kwargs: validation,
    )

    result = analyze_yolo_dataset_splits(configuration, plan)

    outcome = result.successful_splits[0]
    assert outcome.split is split
    assert outcome.dataset_analysis is analysis
    assert outcome.class_validation is validation
    assert plan.splits == (split,)
    assert plan.config_path is configuration.config_path
    assert analysis.pairs == ()
    assert validation.errors == ()


def create_roots(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Create explicit image and label roots for one integration split."""

    image_root = tmp_path / name / "images"
    label_root = tmp_path / name / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    return image_root, label_root


def write_image(path: Path) -> Path:
    """Write arbitrary supported-image bytes without decoding them."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"arbitrary image bytes")
    return path


def write_label(path: Path, content: str = "") -> Path:
    """Write one explicit UTF-8 YOLO label."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_real_splits_preserve_per_split_diagnostics_and_independence(
    tmp_path: Path,
) -> None:
    """Execute successful, class-invalid, and failed splits in order."""

    train_images, train_labels = create_roots(tmp_path, "train")
    write_image(train_images / "known.jpg")
    write_label(train_labels / "known.txt", "0 0.5 0.5 0.2 0.2\n")

    val_images, val_labels = create_roots(tmp_path, "validation")
    write_image(val_images / "invalid.jpg")
    write_label(
        val_labels / "invalid.txt",
        "7 0.5 0.5 0.2 0.2\n-1 nan 2.0 0 -0.5\n",
    )
    write_image(val_images / "known.jpg")
    write_label(val_labels / "known.txt", "1 0.5 0.5 0.2 0.2\n")
    write_image(val_images / "missing.jpg")
    write_label(val_labels / "orphan.txt", "0 0.5 0.5 0.2 0.2\n")
    write_image(val_images / "conflict.jpg")
    write_image(val_images / "conflict.png")
    write_label(val_labels / "conflict.txt", "0 0.5 0.5 0.2 0.2\n")

    test_images = tmp_path / "test" / "images"
    test_images.mkdir(parents=True)
    test_labels = tmp_path / "test" / "missing-labels"
    splits = (
        make_split(
            "train",
            image_directory=train_images,
            label_directory=train_labels,
        ),
        make_split(
            "validation",
            image_directory=val_images,
            label_directory=val_labels,
        ),
        make_split(
            "test",
            image_directory=test_images,
            label_directory=test_labels,
        ),
    )

    result = analyze_yolo_dataset_splits(
        make_configuration(),
        make_plan(*splits),
    )

    assert [outcome.split.name for outcome in result.outcomes] == [
        "train",
        "validation",
        "test",
    ]
    assert result.is_complete is False
    assert [item.split.name for item in result.successful_splits] == [
        "train",
        "validation",
    ]
    assert [item.split.name for item in result.failed_splits] == ["test"]
    train, validation = result.successful_splits
    assert train.dataset_analysis.total_annotations == 1
    assert train.dataset_analysis.class_counts == (
        YoloClassCount(class_id=0, annotation_count=1),
    )
    assert tuple(
        item.annotation_count for item in train.class_validation.class_usage
    ) == (1, 0)
    assert validation.dataset_analysis.invalid_label_files == 1
    assert validation.dataset_analysis.total_annotations == 1
    assert validation.dataset_analysis.missing_label_images == (
        val_images / "missing.jpg",
    )
    assert validation.dataset_analysis.orphan_label_files == (
        val_labels / "orphan.txt",
    )
    assert len(validation.dataset_analysis.pairing_conflicts) == 1
    assert validation.class_validation.is_valid is False
    assert [
        occurrence.class_id
        for occurrence in validation.class_validation.unknown_class_occurrences
    ] == [7]
    assert tuple(
        item.annotation_count
        for item in validation.class_validation.class_usage
    ) == (0, 1)
    failure = result.failed_splits[0]
    assert failure.error_type == "FileNotFoundError"
    assert failure.message == f"Label directory does not exist: {test_labels}"
    assert not hasattr(result, "total_annotations")
    assert not hasattr(result, "class_counts")


def test_all_successful_empty_splits_are_complete(tmp_path: Path) -> None:
    """Treat existing empty roots as successful zero-count analyses."""

    splits = []
    for name in ("train", "validation", "test"):
        images, labels = create_roots(tmp_path, name)
        splits.append(
            make_split(
                name,
                image_directory=images,
                label_directory=labels,
            )
        )

    result = analyze_yolo_dataset_splits(
        make_configuration(),
        make_plan(*splits),
    )

    assert result.is_complete is True
    assert len(result.successful_splits) == 3
    assert result.failed_splits == ()
    assert all(
        outcome.dataset_analysis.total_images == 0
        and outcome.dataset_analysis.total_annotations == 0
        for outcome in result.successful_splits
    )


@pytest.mark.parametrize(
    ("recursive", "expected_annotations"),
    [(False, 0), (True, 1)],
)
def test_recursive_mode_reuses_existing_nested_discovery_semantics(
    recursive: bool,
    expected_annotations: int,
    tmp_path: Path,
) -> None:
    """Forward recursive mode rather than implementing discovery locally."""

    images, labels = create_roots(tmp_path, "train")
    write_image(images / "nested" / "fish.jpg")
    write_label(
        labels / "nested" / "fish.txt",
        "0 0.5 0.5 0.2 0.2\n",
    )
    split = make_split(
        "train",
        image_directory=images,
        label_directory=labels,
    )

    result = analyze_yolo_dataset_splits(
        make_configuration(),
        make_plan(split),
        recursive=recursive,
    )

    assert result.recursive is recursive
    assert result.successful_splits[
        0
    ].dataset_analysis.total_annotations == expected_annotations


@pytest.mark.parametrize(
    ("root_case", "expected_type", "expected_message_prefix"),
    [
        (
            "missing-image",
            "FileNotFoundError",
            "Image directory does not exist",
        ),
        (
            "missing-label",
            "FileNotFoundError",
            "Label directory does not exist",
        ),
        ("file-image", "NotADirectoryError", "Image path is not a directory"),
        ("file-label", "NotADirectoryError", "Label path is not a directory"),
    ],
)
def test_real_expected_root_failures_are_structured(
    root_case: str,
    expected_type: str,
    expected_message_prefix: str,
    tmp_path: Path,
) -> None:
    """Preserve each existing analyzer root failure without retrying."""

    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    if root_case != "missing-image":
        if root_case == "file-image":
            image_root.write_bytes(b"file")
        else:
            image_root.mkdir()
    if root_case != "missing-label":
        if root_case == "file-label":
            label_root.write_text("file", encoding="utf-8")
        else:
            label_root.mkdir()
    split = make_split(
        "train",
        image_directory=image_root,
        label_directory=label_root,
    )

    result = analyze_yolo_dataset_splits(
        make_configuration(),
        make_plan(split),
    )

    assert result.successful_splits == ()
    assert len(result.failed_splits) == 1
    failure = result.failed_splits[0]
    assert failure.error_type == expected_type
    assert failure.message == f"{expected_message_prefix}: " + str(
        image_root if "Image" in expected_message_prefix else label_root
    )


def test_project_commands_and_dependencies_are_unchanged() -> None:
    """Add no CLI entry point or dependency for split execution."""

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
