"""Tests for configured YOLO dataset split planning."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import tomllib
from typing import cast

import pytest

from poseidon_ai.nautilus_vision import (
    yolo_class_validation,
    yolo_dataset,
    yolo_label,
    yolo_split_plan,
)
from poseidon_ai.nautilus_vision.yolo_config import (
    YoloClassDefinition,
    YoloDatasetConfiguration,
    validate_yolo_dataset_config,
)
from poseidon_ai.nautilus_vision.yolo_split_plan import (
    YoloDatasetSplit,
    YoloDatasetSplitPlan,
    YoloDatasetSplitPlanValidationResult,
    build_yolo_dataset_split_plan,
)


IMAGES_NAME_ERROR = (
    "images_directory_name must be a non-empty single path component."
)
LABELS_NAME_ERROR = (
    "labels_directory_name must be a non-empty single path component."
)


def make_configuration(
    *,
    train_path: Path = Path("dataset/images/train"),
    validation_path: Path = Path("dataset/images/val"),
    test_path: Path | None = Path("dataset/images/test"),
) -> YoloDatasetConfiguration:
    """Construct one immutable configuration for focused planning tests."""

    return YoloDatasetConfiguration(
        config_path=Path("dataset/data.yaml"),
        dataset_root=Path("dataset"),
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        classes=(YoloClassDefinition(class_id=0, name="fish"),),
    )


def require_plan(
    result: YoloDatasetSplitPlanValidationResult,
) -> YoloDatasetSplitPlan:
    """Return the plan from an expected valid result."""

    assert result.is_valid is True
    assert result.errors == ()
    assert result.plan is not None
    return result.plan


def test_public_models_are_frozen_slotted_and_tuple_backed() -> None:
    """Expose immutable public models without instance dictionaries."""

    result = build_yolo_dataset_split_plan(make_configuration())
    plan = require_plan(result)
    split = plan.training_split

    with pytest.raises(FrozenInstanceError):
        split.name = "test"
    with pytest.raises(FrozenInstanceError):
        plan.config_path = Path("other.yaml")
    with pytest.raises(FrozenInstanceError):
        result.is_valid = False
    with pytest.raises(TypeError):
        plan.splits[0] = split

    assert isinstance(plan.splits, tuple)
    assert isinstance(result.errors, tuple)
    assert all(
        not hasattr(model, "__dict__")
        for model in (split, plan, result)
    )


def test_public_model_field_order_matches_contract() -> None:
    """Preserve the documented public model field order."""

    assert tuple(YoloDatasetSplit.__dataclass_fields__) == (
        "name",
        "image_directory",
        "label_directory",
    )
    assert tuple(YoloDatasetSplitPlan.__dataclass_fields__) == (
        "config_path",
        "splits",
    )
    assert tuple(
        YoloDatasetSplitPlanValidationResult.__dataclass_fields__
    ) == (
        "is_valid",
        "plan",
        "errors",
    )


def test_default_plan_preserves_paths_and_orders_all_splits() -> None:
    """Plan train, validation, and test using the default convention."""

    configuration = make_configuration()

    plan = require_plan(build_yolo_dataset_split_plan(configuration))

    assert plan.config_path is configuration.config_path
    assert plan.splits == (
        YoloDatasetSplit(
            name="train",
            image_directory=configuration.train_path,
            label_directory=Path("dataset/labels/train"),
        ),
        YoloDatasetSplit(
            name="validation",
            image_directory=configuration.validation_path,
            label_directory=Path("dataset/labels/val"),
        ),
        YoloDatasetSplit(
            name="test",
            image_directory=configuration.test_path,
            label_directory=Path("dataset/labels/test"),
        ),
    )
    assert plan.splits[0].image_directory is configuration.train_path
    assert plan.splits[1].image_directory is configuration.validation_path
    assert plan.splits[2].image_directory is configuration.test_path


def test_split_properties_return_existing_objects() -> None:
    """Return plan members directly without constructing replacements."""

    plan = require_plan(
        build_yolo_dataset_split_plan(make_configuration())
    )

    assert plan.training_split is plan.splits[0]
    assert plan.validation_split is plan.splits[1]
    assert plan.test_split is plan.splits[2]


def test_absent_test_is_omitted_and_property_returns_none() -> None:
    """Produce exactly train and validation when test is not configured."""

    plan = require_plan(
        build_yolo_dataset_split_plan(
            make_configuration(test_path=None)
        )
    )

    assert [split.name for split in plan.splits] == [
        "train",
        "validation",
    ]
    assert plan.test_split is None


def test_repeated_calls_return_independent_immutable_results() -> None:
    """Create fresh results, plans, split objects, and tuples per call."""

    configuration = make_configuration()

    first = build_yolo_dataset_split_plan(configuration)
    second = build_yolo_dataset_split_plan(configuration)
    first_plan = require_plan(first)
    second_plan = require_plan(second)

    assert first == second
    assert first is not second
    assert first_plan is not second_plan
    assert first_plan.splits is not second_plan.splits
    assert all(
        first_split is not second_split
        for first_split, second_split in zip(
            first_plan.splits,
            second_plan.splits,
            strict=True,
        )
    )


@pytest.mark.parametrize(
    ("image_path", "expected_label_path"),
    [
        (Path("images"), Path("labels")),
        (Path("dataset/images/train"), Path("dataset/labels/train")),
        (
            Path("../marine data/images/Train Set"),
            Path("../marine data/labels/Train Set"),
        ),
        (
            Path("collections/images/archive/images/train"),
            Path("collections/images/archive/labels/train"),
        ),
        (
            Path("~/images/$POSEIDON_DATA"),
            Path("~/labels/$POSEIDON_DATA"),
        ),
    ],
    ids=(
        "component-only",
        "relative",
        "parents-spaces-and-case",
        "final-match-only",
        "no-home-or-environment-expansion",
    ),
)
def test_path_components_are_replaced_without_normalization(
    image_path: Path,
    expected_label_path: Path,
) -> None:
    """Preserve every lexical component except the final exact match."""

    plan = require_plan(
        build_yolo_dataset_split_plan(
            make_configuration(
                train_path=image_path,
                validation_path=Path("images/val"),
                test_path=None,
            )
        )
    )

    assert plan.training_split.image_directory is image_path
    assert plan.training_split.label_directory == expected_label_path
    assert plan.training_split.label_directory.is_absolute() is (
        image_path.is_absolute()
    )


def test_absolute_paths_remain_absolute(tmp_path: Path) -> None:
    """Preserve the platform's absolute anchor during reconstruction."""

    image_path = tmp_path / "marine" / "images" / "train"

    plan = require_plan(
        build_yolo_dataset_split_plan(
            make_configuration(
                train_path=image_path,
                validation_path=tmp_path / "images" / "val",
                test_path=None,
            )
        )
    )

    assert plan.training_split.image_directory == image_path
    assert plan.training_split.label_directory == (
        tmp_path / "marine" / "labels" / "train"
    )
    assert plan.training_split.label_directory.is_absolute()


@pytest.mark.parametrize(
    "image_path",
    [
        Path("dataset/marine-images/train"),
        Path("dataset/images-backup/train"),
        Path("dataset/myimages/train"),
        Path("dataset/Images/train"),
    ],
    ids=("suffix", "prefix", "embedded", "case-sensitive"),
)
def test_nonmatching_components_return_exact_train_error(
    image_path: Path,
) -> None:
    """Match only a complete, case-sensitive directory component."""

    result = build_yolo_dataset_split_plan(
        make_configuration(
            train_path=image_path,
            validation_path=Path("images/val"),
            test_path=None,
        )
    )

    assert result == YoloDatasetSplitPlanValidationResult(
        is_valid=False,
        plan=None,
        errors=(
            "Split 'train' image path does not contain directory "
            f"component 'images': {image_path}",
        ),
    )


def test_path_construction_does_not_resolve_or_make_absolute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use lexical components without resolving or absolutizing paths."""

    def forbidden(path: Path) -> Path:
        raise AssertionError(f"unexpected path normalization: {path}")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "absolute", forbidden)
    monkeypatch.setattr(Path, "expanduser", forbidden)

    plan = require_plan(
        build_yolo_dataset_split_plan(
            make_configuration(
                train_path=Path("../data/images/train"),
                validation_path=Path("~/images/$VAL"),
                test_path=None,
            )
        )
    )

    assert plan.training_split.label_directory == Path(
        "../data/labels/train"
    )
    assert plan.validation_split.label_directory == Path(
        "~/labels/$VAL"
    )


def test_custom_directory_names_are_stripped_and_used() -> None:
    """Derive labels from valid custom complete component names."""

    plan = require_plan(
        build_yolo_dataset_split_plan(
            make_configuration(
                train_path=Path("dataset/pictures/train"),
                validation_path=Path("dataset/pictures/val"),
                test_path=None,
            ),
            images_directory_name="  pictures  ",
            labels_directory_name="  annotations  ",
        )
    )

    assert plan.training_split.label_directory == Path(
        "dataset/annotations/train"
    )
    assert plan.validation_split.label_directory == Path(
        "dataset/annotations/val"
    )


def test_custom_directory_matching_remains_case_sensitive() -> None:
    """Do not lowercase configured paths or custom component names."""

    image_path = Path("dataset/Pictures/train")
    result = build_yolo_dataset_split_plan(
        make_configuration(
            train_path=image_path,
            validation_path=Path("pictures/val"),
            test_path=None,
        ),
        images_directory_name="pictures",
        labels_directory_name="annotations",
    )

    assert result.errors == (
        "Split 'train' image path does not contain directory component "
        f"'pictures': {image_path}",
    )
    assert result.plan is None


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        ".",
        "..",
        "data/images",
        r"data\images",
        "/images",
        Path("images"),
        42,
        None,
    ],
    ids=(
        "empty",
        "blank",
        "dot",
        "dot-dot",
        "forward-separator",
        "back-separator",
        "absolute",
        "path-object",
        "integer",
        "none",
    ),
)
def test_invalid_images_directory_name_is_rejected(value: object) -> None:
    """Reject every non-string or non-component image option."""

    result = build_yolo_dataset_split_plan(
        make_configuration(),
        images_directory_name=value,  # type: ignore[arg-type]
    )

    assert result == YoloDatasetSplitPlanValidationResult(
        is_valid=False,
        plan=None,
        errors=(IMAGES_NAME_ERROR,),
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        ".",
        "..",
        "data/labels",
        r"data\labels",
        "/labels",
        Path("labels"),
        42,
        None,
    ],
    ids=(
        "empty",
        "blank",
        "dot",
        "dot-dot",
        "forward-separator",
        "back-separator",
        "absolute",
        "path-object",
        "integer",
        "none",
    ),
)
def test_invalid_labels_directory_name_is_rejected(value: object) -> None:
    """Apply the same exact validation to the label option."""

    result = build_yolo_dataset_split_plan(
        make_configuration(),
        labels_directory_name=value,  # type: ignore[arg-type]
    )

    assert result == YoloDatasetSplitPlanValidationResult(
        is_valid=False,
        plan=None,
        errors=(LABELS_NAME_ERROR,),
    )


def test_option_errors_follow_images_then_labels_order() -> None:
    """Collect independent name errors in the required order."""

    result = build_yolo_dataset_split_plan(
        cast(YoloDatasetConfiguration, object()),
        images_directory_name="data/images",
        labels_directory_name=None,  # type: ignore[arg-type]
    )

    assert result == YoloDatasetSplitPlanValidationResult(
        is_valid=False,
        plan=None,
        errors=(IMAGES_NAME_ERROR, LABELS_NAME_ERROR),
    )


def test_equal_valid_directory_names_are_rejected_after_validation() -> None:
    """Reject equal stripped names using case-sensitive comparison."""

    result = build_yolo_dataset_split_plan(
        make_configuration(),
        images_directory_name=" data ",
        labels_directory_name="data",
    )

    assert result.errors == (
        "Image and label directory names must be different.",
    )
    assert result.plan is None


def test_case_distinct_directory_names_are_allowed() -> None:
    """Treat otherwise equal names with distinct case as different."""

    plan = require_plan(
        build_yolo_dataset_split_plan(
            make_configuration(
                train_path=Path("dataset/data/train"),
                validation_path=Path("dataset/data/val"),
                test_path=None,
            ),
            images_directory_name="data",
            labels_directory_name="Data",
        )
    )

    assert plan.training_split.label_directory == Path(
        "dataset/Data/train"
    )


def test_all_split_errors_are_exact_and_ordered() -> None:
    """Collect train, validation, and test failures without a partial plan."""

    configuration = make_configuration(
        train_path=Path("dataset/train"),
        validation_path=Path("dataset/Images/val"),
        test_path=Path("dataset/test"),
    )

    result = build_yolo_dataset_split_plan(configuration)

    assert result == YoloDatasetSplitPlanValidationResult(
        is_valid=False,
        plan=None,
        errors=(
            "Split 'train' image path does not contain directory "
            f"component 'images': {configuration.train_path}",
            "Split 'validation' image path does not contain directory "
            f"component 'images': {configuration.validation_path}",
            "Split 'test' image path does not contain directory component "
            f"'images': {configuration.test_path}",
        ),
    )


def test_split_error_preserves_supplied_path_text() -> None:
    """Format the exact lexical Path without absolute platform details."""

    image_path = Path("../marine data/Images/Train Set")

    result = build_yolo_dataset_split_plan(
        make_configuration(
            train_path=image_path,
            validation_path=Path("images/val"),
            test_path=None,
        )
    )

    assert result.errors == (
        "Split 'train' image path does not contain directory component "
        f"'images': {image_path}",
    )


def test_planning_performs_no_filesystem_or_domain_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use only lexical input models and never execute downstream work."""

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected filesystem or domain execution")

    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    monkeypatch.setattr(Path, "is_dir", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(yolo_dataset, "analyze_yolo_dataset", forbidden)
    monkeypatch.setattr(yolo_label, "validate_yolo_label", forbidden)
    monkeypatch.setattr(
        yolo_class_validation,
        "validate_yolo_dataset_classes",
        forbidden,
    )

    plan = require_plan(
        build_yolo_dataset_split_plan(make_configuration())
    )

    assert len(plan.splits) == 3
    assert not hasattr(yolo_split_plan, "analyze_yolo_dataset")
    assert not hasattr(yolo_split_plan, "validate_yolo_label")
    assert not hasattr(yolo_split_plan, "validate_yolo_dataset_classes")


def test_configuration_input_is_not_mutated() -> None:
    """Preserve the exact immutable input object and all configured paths."""

    configuration = make_configuration()
    original_paths = (
        configuration.config_path,
        configuration.train_path,
        configuration.validation_path,
        configuration.test_path,
    )

    build_yolo_dataset_split_plan(configuration)

    assert (
        configuration.config_path,
        configuration.train_path,
        configuration.validation_path,
        configuration.test_path,
    ) == original_paths
    assert configuration.train_path is original_paths[1]
    assert configuration.validation_path is original_paths[2]
    assert configuration.test_path is original_paths[3]


@pytest.mark.parametrize(
    "names_yaml",
    [
        "names:\n  - fish\n  - turtle\n",
        "names:\n  0: fish\n  1: turtle\n",
    ],
    ids=("list", "mapping"),
)
def test_real_yaml_configuration_produces_expected_plan(
    names_yaml: str,
    tmp_path: Path,
) -> None:
    """Compose split planning with both real configuration name forms."""

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "path: ../datasets/marine\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        + names_yaml,
        encoding="utf-8",
    )
    config_result = validate_yolo_dataset_config(config_path)
    assert config_result.configuration is not None
    configuration = config_result.configuration

    plan = require_plan(build_yolo_dataset_split_plan(configuration))

    expected_root = tmp_path / "../datasets/marine"
    assert [split.name for split in plan.splits] == [
        "train",
        "validation",
        "test",
    ]
    assert [split.image_directory for split in plan.splits] == [
        expected_root / "images/train",
        expected_root / "images/val",
        expected_root / "images/test",
    ]
    assert [split.label_directory for split in plan.splits] == [
        expected_root / "labels/train",
        expected_root / "labels/val",
        expected_root / "labels/test",
    ]
    assert ".." in plan.training_split.image_directory.parts


def test_omitted_yaml_dataset_root_uses_config_parent(
    tmp_path: Path,
) -> None:
    """Plan already constructed split paths when YAML path is omitted."""

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "train: images/train\n"
        "val: images/val\n"
        "names: [fish]\n",
        encoding="utf-8",
    )
    config_result = validate_yolo_dataset_config(config_path)
    assert config_result.configuration is not None

    plan = require_plan(
        build_yolo_dataset_split_plan(config_result.configuration)
    )

    assert plan.training_split.image_directory == tmp_path / "images/train"
    assert plan.training_split.label_directory == tmp_path / "labels/train"
    assert plan.test_split is None


def test_project_commands_and_dependencies_are_unchanged() -> None:
    """Add no CLI entry point or dependency for pure path planning."""

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
