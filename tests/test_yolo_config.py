"""Tests for strict YOLO dataset configuration validation."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import tomllib

import pytest
import yaml

from poseidon_ai.nautilus_vision import yolo_config
from poseidon_ai.nautilus_vision.yolo_config import (
    YoloClassDefinition,
    YoloDatasetConfiguration,
    YoloDatasetConfigValidationResult,
    validate_yolo_dataset_config,
)


VALID_FIELDS = """\
train: images/train
val: images/val
names:
  - fish
  - turtle
"""


def write_config(
    path: Path,
    content: str = VALID_FIELDS,
) -> Path:
    """Write one UTF-8 YAML configuration and return its path."""

    path.write_text(content, encoding="utf-8")
    return path


def require_configuration(
    result: YoloDatasetConfigValidationResult,
) -> YoloDatasetConfiguration:
    """Return the configuration from an expected valid result."""

    assert result.is_valid is True
    assert result.errors == ()
    assert result.configuration is not None
    return result.configuration


def test_missing_configuration_returns_exact_error(
    tmp_path: Path,
) -> None:
    """Return a structured error for a missing YAML path."""

    path = tmp_path / "missing.yaml"

    result = validate_yolo_dataset_config(path)

    assert result == YoloDatasetConfigValidationResult(
        is_valid=False,
        configuration=None,
        errors=(f"Dataset configuration does not exist: {path}",),
    )


def test_directory_returns_exact_not_a_file_error(
    tmp_path: Path,
) -> None:
    """Reject an existing directory before extension or YAML parsing."""

    result = validate_yolo_dataset_config(tmp_path)

    assert result.errors == (
        f"Dataset configuration path is not a file: {tmp_path}",
    )
    assert result.configuration is None


@pytest.mark.parametrize("filename", ["data.txt", "data"])
def test_unsupported_extension_returns_exact_error(
    filename: str,
    tmp_path: Path,
) -> None:
    """Reject unsupported and extensionless regular files."""

    path = write_config(tmp_path / filename)

    result = validate_yolo_dataset_config(path)

    assert result.errors == (
        f"Unsupported dataset configuration extension: {path.suffix}",
    )
    assert result.configuration is None


@pytest.mark.parametrize(
    "filename",
    ["data.yaml", "data.yml", "DATA.YAML", "dataset.YmL"],
)
def test_yaml_extensions_are_accepted_case_insensitively(
    filename: str,
    tmp_path: Path,
) -> None:
    """Accept both YAML suffix aliases in any case."""

    result = validate_yolo_dataset_config(
        write_config(tmp_path / filename)
    )

    assert result.is_valid is True


def test_invalid_utf8_returns_exact_error(tmp_path: Path) -> None:
    """Translate only UTF-8 decode failures into the defined error."""

    path = tmp_path / "data.yaml"
    path.write_bytes(b"\xff\xfe")

    result = validate_yolo_dataset_config(path)

    assert result.errors == (
        f"Dataset configuration is not valid UTF-8: {path}",
    )
    assert result.configuration is None


@pytest.mark.parametrize(
    "content",
    [
        "names: [fish\n",
        "!!python/object:builtins.object {}\n",
    ],
)
def test_invalid_or_unsafe_yaml_returns_generic_error(
    content: str,
    tmp_path: Path,
) -> None:
    """Use safe loading and hide parser-specific traces."""

    path = write_config(tmp_path / "data.yaml", content)

    result = validate_yolo_dataset_config(path)

    assert result.errors == (
        f"Dataset configuration contains invalid YAML: {path}",
    )
    assert result.configuration is None


def test_unexpected_filesystem_error_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve normal exception semantics for unexpected read failures."""

    path = write_config(tmp_path / "data.yaml")

    def deny_read(
        source: Path,
        *,
        encoding: str,
    ) -> str:
        raise PermissionError("configuration denied")

    monkeypatch.setattr(Path, "read_text", deny_read)

    with pytest.raises(PermissionError, match="configuration denied"):
        validate_yolo_dataset_config(path)


@pytest.mark.parametrize(
    "content",
    ["", "# comment only\n", "null\n", "42\n", "- item\n"],
)
def test_non_mapping_yaml_roots_are_rejected(
    content: str,
    tmp_path: Path,
) -> None:
    """Reject empty, null, scalar, and sequence YAML roots."""

    result = validate_yolo_dataset_config(
        write_config(tmp_path / "data.yaml", content)
    )

    assert result.errors == (
        "Dataset configuration must contain a YAML mapping.",
    )


def test_mapping_root_proceeds_to_ordered_field_validation(
    tmp_path: Path,
) -> None:
    """Validate schema fields after accepting a mapping root."""

    result = validate_yolo_dataset_config(
        write_config(tmp_path / "data.yaml", "{}\n")
    )

    assert result.errors == (
        "Field 'train' must be a non-empty string.",
        "Field 'val' must be a non-empty string.",
        "Field 'names' must be a non-empty list or mapping.",
    )


def test_omitted_path_uses_configuration_parent(tmp_path: Path) -> None:
    """Default the dataset root to the YAML file's parent."""

    config_path = write_config(tmp_path / "data.yaml")

    configuration = require_configuration(
        validate_yolo_dataset_config(str(config_path))
    )

    assert configuration.config_path == config_path
    assert configuration.dataset_root == config_path.parent


def test_relative_path_is_stripped_joined_and_not_normalized(
    tmp_path: Path,
) -> None:
    """Retain lexical parent components in a relative dataset root."""

    config_path = write_config(
        tmp_path / "data.yaml",
        "path: ' ../datasets/marine-life '\n" + VALID_FIELDS,
    )

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )

    assert configuration.dataset_root == (
        config_path.parent / Path("../datasets/marine-life")
    )
    assert ".." in configuration.dataset_root.parts
    assert not configuration.dataset_root.exists()


def test_absolute_dataset_root_remains_absolute(tmp_path: Path) -> None:
    """Store an absolute configured root without joining the YAML parent."""

    absolute_root = tmp_path / "not-created" / "dataset"
    config_path = write_config(
        tmp_path / "data.yaml",
        f"path: '{absolute_root.as_posix()}'\n" + VALID_FIELDS,
    )

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )

    assert configuration.dataset_root == absolute_root
    assert configuration.dataset_root.is_absolute()
    assert not absolute_root.exists()


@pytest.mark.parametrize(
    "path_line",
    ["path: ''", "path: '   '", "path: 42", "path: null"],
)
def test_invalid_provided_dataset_root_is_rejected(
    path_line: str,
    tmp_path: Path,
) -> None:
    """Require a non-empty string whenever path is present."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            path_line + "\n" + VALID_FIELDS,
        )
    )

    assert result.errors == (
        "Field 'path' must be a non-empty string when provided.",
    )


def test_path_construction_never_calls_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every configured path lexical and unexpanded."""

    config_path = write_config(
        tmp_path / "data.yaml",
        "path: '$DATA_ROOT/~/../marine'\n" + VALID_FIELDS,
    )

    def fail_resolve(*args: object, **kwargs: object) -> Path:
        raise AssertionError("resolve was called")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )

    assert "$DATA_ROOT" in str(configuration.dataset_root)
    assert "~" in configuration.dataset_root.parts
    assert ".." in configuration.dataset_root.parts


def test_split_paths_are_stripped_and_joined_to_dataset_root(
    tmp_path: Path,
) -> None:
    """Construct train, validation, and test paths without traversal."""

    config_path = write_config(
        tmp_path / "data.yaml",
        """\
path: dataset
train: ' images/train '
val: ' images/validation '
test: ' images/test '
names: [fish]
""",
    )

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )
    root = config_path.parent / "dataset"

    assert configuration.train_path == root / "images/train"
    assert configuration.validation_path == root / "images/validation"
    assert configuration.test_path == root / "images/test"
    assert not root.exists()


@pytest.mark.parametrize("test_line", ["", "test: null\n"])
def test_absent_or_null_test_path_produces_none(
    test_line: str,
    tmp_path: Path,
) -> None:
    """Treat the optional test split as absent when omitted or null."""

    config_path = write_config(
        tmp_path / "data.yaml",
        test_line + VALID_FIELDS,
    )

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )

    assert configuration.test_path is None


def test_absolute_split_paths_remain_absolute(tmp_path: Path) -> None:
    """Do not join absolute split paths to the dataset root."""

    train_path = tmp_path / "external-train"
    val_path = tmp_path / "external-val"
    test_path = tmp_path / "external-test"
    config_path = write_config(
        tmp_path / "data.yaml",
        f"""\
train: '{train_path.as_posix()}'
val: '{val_path.as_posix()}'
test: '{test_path.as_posix()}'
names: [fish]
""",
    )

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )

    assert configuration.train_path == train_path
    assert configuration.validation_path == val_path
    assert configuration.test_path == test_path
    assert all(
        path.is_absolute()
        for path in (
            configuration.train_path,
            configuration.validation_path,
            configuration.test_path,
        )
        if path is not None
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("train", None, "Field 'train' must be a non-empty string."),
        ("train", "''", "Field 'train' must be a non-empty string."),
        ("train", "42", "Field 'train' must be a non-empty string."),
        ("train", "[images/train]", "Field 'train' must be a non-empty string."),
        ("val", None, "Field 'val' must be a non-empty string."),
        ("val", "'   '", "Field 'val' must be a non-empty string."),
        ("val", "false", "Field 'val' must be a non-empty string."),
        ("val", "[images/val]", "Field 'val' must be a non-empty string."),
        (
            "test",
            "''",
            "Field 'test' must be a non-empty string or null when provided.",
        ),
        (
            "test",
            "42",
            "Field 'test' must be a non-empty string or null when provided.",
        ),
        (
            "test",
            "[images/test]",
            "Field 'test' must be a non-empty string or null when provided.",
        ),
    ],
)
def test_invalid_split_values_are_rejected(
    field: str,
    value: str | None,
    expected: str,
    tmp_path: Path,
) -> None:
    """Reject missing, empty, non-string, and list split values."""

    fields = {
        "train": "images/train",
        "val": "images/val",
    }
    if value is None:
        fields.pop(field)
    else:
        fields[field] = value
    yaml_text = "\n".join(
        f"{name}: {field_value}"
        for name, field_value in fields.items()
    )
    yaml_text += "\nnames: [fish]\n"

    result = validate_yolo_dataset_config(
        write_config(tmp_path / "data.yaml", yaml_text)
    )

    assert result.errors == (expected,)


def test_split_paths_are_not_opened_traversed_or_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read only the YAML file and leave nonexistent split paths untouched."""

    config_path = write_config(
        tmp_path / "data.yaml",
        """\
train: labels/train.txt
val: images/val
test: images/test
names: [fish]
""",
    )
    original_read_text = Path.read_text
    read_paths: list[Path] = []

    def track_read(source: Path, *, encoding: str) -> str:
        read_paths.append(source)
        return original_read_text(source, encoding=encoding)

    def fail_iteration(*args: object, **kwargs: object) -> object:
        raise AssertionError("a split directory was traversed")

    monkeypatch.setattr(Path, "read_text", track_read)
    monkeypatch.setattr(Path, "iterdir", fail_iteration)
    monkeypatch.setattr(Path, "rglob", fail_iteration)

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )

    assert read_paths == [config_path]
    assert not configuration.train_path.exists()
    assert not configuration.validation_path.exists()
    assert configuration.test_path is not None
    assert not configuration.test_path.exists()


def test_valid_names_list_creates_ordered_normalized_classes(
    tmp_path: Path,
) -> None:
    """Use list indexes as IDs and preserve internal name spacing."""

    config_path = write_config(
        tmp_path / "data.yaml",
        """\
train: images/train
val: images/val
names:
  - ' fish '
  - ' sea turtle '
  - Whale Shark
""",
    )

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )

    assert configuration.classes == (
        YoloClassDefinition(class_id=0, name="fish"),
        YoloClassDefinition(class_id=1, name="sea turtle"),
        YoloClassDefinition(class_id=2, name="Whale Shark"),
    )
    assert configuration.number_of_classes == 3


@pytest.mark.parametrize(
    "names_yaml",
    ["[]", "{}", "fish", "null", "true", "42"],
)
def test_invalid_names_structure_is_rejected(
    names_yaml: str,
    tmp_path: Path,
) -> None:
    """Require a non-empty list or mapping and reject strings as sequences."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            f"train: images/train\nval: images/val\nnames: {names_yaml}\n",
        )
    )

    assert result.errors == (
        "Field 'names' must be a non-empty list or mapping.",
    )


@pytest.mark.parametrize(
    ("entry", "index"),
    [("''", 1), ("'   '", 1), ("42", 1), ("true", 1), ("null", 1)],
)
def test_invalid_names_list_entry_returns_exact_error(
    entry: str,
    index: int,
    tmp_path: Path,
) -> None:
    """Reject blank and non-string list entries by their source index."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            "train: images/train\n"
            "val: images/val\n"
            f"names: [fish, {entry}, shark]\n",
        )
    )

    assert result.errors == (
        f"Class name at index {index} must be a non-empty string.",
    )
    assert result.configuration is None


def test_names_list_validation_continues_and_detects_duplicate(
    tmp_path: Path,
) -> None:
    """Collect a later duplicate after an invalid list entry."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            "train: images/train\n"
            "val: images/val\n"
            "names: [fish, null, ' fish ']\n",
        )
    )

    assert result.errors == (
        "Class name at index 1 must be a non-empty string.",
        "Class names must be unique; duplicate: 'fish'.",
    )


def test_case_distinct_class_names_are_valid(tmp_path: Path) -> None:
    """Compare normalized class names case-sensitively."""

    configuration = require_configuration(
        validate_yolo_dataset_config(
            write_config(
                tmp_path / "data.yaml",
                "train: images/train\n"
                "val: images/val\n"
                "names: [fish, Fish]\n",
            )
        )
    )

    assert [item.name for item in configuration.classes] == [
        "fish",
        "Fish",
    ]


def test_mapping_classes_are_sorted_and_names_are_stripped(
    tmp_path: Path,
) -> None:
    """Ignore mapping source order in the normalized representation."""

    config_path = write_config(
        tmp_path / "data.yaml",
        """\
train: images/train
val: images/val
names:
  2: ' shark '
  0: fish
  1: ' sea turtle '
""",
    )

    configuration = require_configuration(
        validate_yolo_dataset_config(config_path)
    )

    assert configuration.classes == (
        YoloClassDefinition(class_id=0, name="fish"),
        YoloClassDefinition(class_id=1, name="sea turtle"),
        YoloClassDefinition(class_id=2, name="shark"),
    )


@pytest.mark.parametrize(
    ("key_yaml", "expected_key"),
    [
        ("-1", "-1"),
        ("true", "True"),
        ("1.5", "1.5"),
        ("'1'", "'1'"),
        ("fish", "'fish'"),
        ("null", "None"),
    ],
)
def test_invalid_mapping_keys_are_not_coerced(
    key_yaml: str,
    expected_key: str,
    tmp_path: Path,
) -> None:
    """Require actual non-negative integer class keys, excluding booleans."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            "train: images/train\n"
            "val: images/val\n"
            f"names: {{{key_yaml}: fish}}\n",
        )
    )

    assert result.errors == (
        "Class ID key must be a non-negative integer: "
        f"{expected_key}.",
    )


@pytest.mark.parametrize(
    ("mapping", "found"),
    [
        ("1: fish\n  2: turtle", "1, 2"),
        ("0: fish\n  2: shark", "0, 2"),
        ("0: fish\n  2: shark\n  4: ray", "0, 2, 4"),
    ],
)
def test_non_contiguous_mapping_ids_return_one_exact_error(
    mapping: str,
    found: str,
    tmp_path: Path,
) -> None:
    """Require contiguous IDs beginning at zero and report sorted IDs."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            "train: images/train\n"
            "val: images/val\n"
            f"names:\n  {mapping}\n",
        )
    )

    assert result.errors == (
        "Class IDs must be contiguous and start at 0; "
        f"found: {found}.",
    )


def test_invalid_mapping_name_uses_class_id(tmp_path: Path) -> None:
    """Report invalid mapping values against their valid integer ID."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            "train: images/train\n"
            "val: images/val\n"
            "names: {0: fish, 1: null, 2: shark}\n",
        )
    )

    assert result.errors == (
        "Class name for class ID 1 must be a non-empty string.",
    )


def test_mapping_duplicate_stripped_name_is_rejected(
    tmp_path: Path,
) -> None:
    """Detect normalized duplicates in ascending class-ID order."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            "train: images/train\n"
            "val: images/val\n"
            "names: {0: fish, 1: ' fish '}\n",
        )
    )

    assert result.errors == (
        "Class names must be unique; duplicate: 'fish'.",
    )


def test_missing_or_matching_nc_uses_derived_class_count(
    tmp_path: Path,
) -> None:
    """Accept absent or matching declarations without storing nc."""

    without_nc = require_configuration(
        validate_yolo_dataset_config(
            write_config(tmp_path / "without.yaml")
        )
    )
    with_nc = require_configuration(
        validate_yolo_dataset_config(
            write_config(tmp_path / "with.yaml", VALID_FIELDS + "nc: 2\n")
        )
    )

    assert without_nc.number_of_classes == 2
    assert with_nc.number_of_classes == 2
    assert "nc" not in YoloDatasetConfiguration.__dataclass_fields__


@pytest.mark.parametrize(
    "nc_yaml",
    ["0", "-1", "true", "1.5", "'2'", "null"],
)
def test_invalid_nc_type_or_range_is_rejected(
    nc_yaml: str,
    tmp_path: Path,
) -> None:
    """Require a positive non-boolean integer when nc is present."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            VALID_FIELDS + f"nc: {nc_yaml}\n",
        )
    )

    assert result.errors == (
        "Field 'nc' must be a positive integer when provided.",
    )


def test_mismatched_nc_returns_exact_error(tmp_path: Path) -> None:
    """Compare a valid declaration with parsed class definitions."""

    result = validate_yolo_dataset_config(
        write_config(tmp_path / "data.yaml", VALID_FIELDS + "nc: 3\n")
    )

    assert result.errors == (
        "Field 'nc' must equal the number of configured classes: "
        "expected 2, found 3.",
    )


def test_nc_mismatch_is_skipped_when_class_parsing_is_invalid(
    tmp_path: Path,
) -> None:
    """Do not run the dependent count check after a class schema failure."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            "train: images/train\n"
            "val: images/val\n"
            "names: {0: fish, 2: shark}\n"
            "nc: 3\n",
        )
    )

    assert result.errors == (
        "Class IDs must be contiguous and start at 0; found: 0, 2.",
    )


def test_independent_schema_errors_follow_required_order(
    tmp_path: Path,
) -> None:
    """Collect independent field, class, duplicate, and nc errors in order."""

    result = validate_yolo_dataset_config(
        write_config(
            tmp_path / "data.yaml",
            """\
path: ' '
train: []
test: 42
names:
  fish: ignored
  3: null
  0: fish
  2: ' fish '
  4: shark
nc: false
""",
        )
    )

    assert result.is_valid is False
    assert result.configuration is None
    assert result.errors == (
        "Field 'path' must be a non-empty string when provided.",
        "Field 'train' must be a non-empty string.",
        "Field 'val' must be a non-empty string.",
        "Field 'test' must be a non-empty string or null when provided.",
        "Class ID key must be a non-negative integer: 'fish'.",
        "Class name for class ID 3 must be a non-empty string.",
        "Class IDs must be contiguous and start at 0; found: 0, 2, 3, 4.",
        "Class names must be unique; duplicate: 'fish'.",
        "Field 'nc' must be a positive integer when provided.",
    )


def test_valid_result_ignores_unknown_top_level_fields(
    tmp_path: Path,
) -> None:
    """Ignore ordinary additional YOLO metadata outside the public model."""

    config_path = write_config(
        tmp_path / "data.yaml",
        VALID_FIELDS
        + "task: detect\n"
        + "download: https://example.invalid/data.zip\n"
        + "roboflow: {workspace: marine}\n",
    )

    result = validate_yolo_dataset_config(config_path)
    configuration = require_configuration(result)

    assert configuration.number_of_classes == 2
    assert tuple(YoloDatasetConfiguration.__dataclass_fields__) == (
        "config_path",
        "dataset_root",
        "train_path",
        "validation_path",
        "test_path",
        "classes",
    )


def test_public_models_are_frozen_slotted_and_tuple_backed(
    tmp_path: Path,
) -> None:
    """Expose immutable models without normal instance dictionaries."""

    result = validate_yolo_dataset_config(
        write_config(tmp_path / "data.yaml")
    )
    configuration = require_configuration(result)
    class_definition = configuration.classes[0]

    with pytest.raises(FrozenInstanceError):
        class_definition.name = "shark"
    with pytest.raises(FrozenInstanceError):
        configuration.train_path = Path("other")
    with pytest.raises(FrozenInstanceError):
        result.is_valid = False
    with pytest.raises(TypeError):
        configuration.classes[0] = class_definition

    assert isinstance(result.errors, tuple)
    assert isinstance(configuration.classes, tuple)
    assert all(
        not hasattr(model, "__dict__")
        for model in (class_definition, configuration, result)
    )


def test_repeated_validation_returns_independent_results(
    tmp_path: Path,
) -> None:
    """Create fresh immutable results, configurations, and class tuples."""

    config_path = write_config(tmp_path / "data.yaml")

    first = validate_yolo_dataset_config(config_path)
    second = validate_yolo_dataset_config(config_path)

    assert first == second
    assert first is not second
    assert first.configuration is not second.configuration
    assert first.configuration is not None
    assert second.configuration is not None
    assert first.configuration.classes is not second.configuration.classes


def test_configuration_module_has_no_analysis_integration(
    tmp_path: Path,
) -> None:
    """Keep configuration parsing independent of image and label analysis."""

    result = validate_yolo_dataset_config(
        write_config(tmp_path / "data.yaml")
    )

    assert result.is_valid is True
    assert not hasattr(yolo_config, "analyze_yolo_dataset")
    assert not hasattr(yolo_config, "validate_yolo_label")


def test_existing_cli_entry_points_are_unchanged() -> None:
    """Do not add a configuration or label-analysis command."""

    with Path("pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["scripts"] == {
        "poseidon-inspect": (
            "poseidon_ai.nautilus_vision.inspect_image:main"
        ),
        "poseidon-dataset-summary": (
            "poseidon_ai.nautilus_vision.dataset_summary:main"
        ),
    }


def test_pyyaml_is_importable_and_safe_loader_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route YAML parsing through PyYAML safe_load exactly once."""

    config_path = write_config(tmp_path / "data.yaml")
    original_safe_load = yaml.safe_load
    calls: list[str] = []

    def track_safe_load(text: str) -> object:
        calls.append(text)
        return original_safe_load(text)

    monkeypatch.setattr(yaml, "safe_load", track_safe_load)

    result = validate_yolo_dataset_config(config_path)

    assert result.is_valid is True
    assert calls == [VALID_FIELDS]
    assert tuple(int(part) for part in yaml.__version__.split(".")) >= (
        6,
        0,
        2,
    )
