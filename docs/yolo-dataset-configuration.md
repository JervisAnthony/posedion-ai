# YOLO dataset configuration

Nautilus Vision provides strict library-level parsing and validation for one
YOLO detection dataset configuration. The component normalizes declared
paths and class definitions into immutable models without inspecting a
dataset or invoking image-label analysis.

## Public API

```python
from poseidon_ai.nautilus_vision.yolo_config import (
    YoloClassDefinition,
    YoloDatasetConfiguration,
    YoloDatasetConfigValidationResult,
    validate_yolo_dataset_config,
)

result = validate_yolo_dataset_config("dataset/data.yaml")
```

All public models are frozen and slotted. Class definitions and errors are
immutable tuples. Invalid results contain deterministic errors and no partial
configuration.

## File and YAML safety

Configuration files must be UTF-8 regular files with a case-insensitive
`.yaml` or `.yml` suffix. Parsing uses PyYAML's `yaml.safe_load()` exclusively.
Malformed syntax, unsupported tags, and unsafe Python-object tags return one
generic invalid-YAML error and cannot construct Python objects.

The YAML root must be a mapping. Empty, comment-only, scalar, sequence, and
null documents are invalid. Unexpected filesystem failures retain normal
Python exception semantics.

## Supported fields

The initial public model owns:

- `path`: optional dataset root;
- `train`: required training split path;
- `val`: required validation split path;
- `test`: optional test split path or null;
- `names`: required class-name list or integer-keyed mapping;
- `nc`: optional declared class count.

Unknown top-level metadata is ignored rather than retained or rejected.
Fields such as `download`, `roboflow`, `task`, and augmentation metadata are
outside this initial model.

## Path construction

When `path` is omitted, the dataset root is the YAML file's parent directory.
A relative `path` is joined to that parent; an absolute `path` remains
absolute. Relative split paths are joined to the resulting dataset root,
while absolute split paths remain absolute.

Surrounding whitespace is removed from path strings. Paths are not resolved,
made absolute, normalized, expanded, or traversed. Parent components such as
`..`, environment-variable text, and `~` remain lexical. Dataset and split
paths are not required to exist, and the component creates no directories.

Each split currently accepts exactly one non-empty string. Lists, remote
URLs, globs, and inventory text-file semantics are not supported.

## Class definitions

Class names can use a YAML list:

```yaml
train: images/train
val: images/val

names:
  - fish
  - turtle
  - shark
```

List indexes become class IDs `0`, `1`, and `2`.

They can also use a mapping with actual integer keys:

```yaml
names:
  0: fish
  1: turtle
  2: shark
```

Mapping keys must be non-negative integers, begin at zero, and remain
contiguous. Booleans, floating-point keys, quoted numeric strings, arbitrary
strings, and null keys are rejected rather than coerced. Returned definitions
are always ordered by ascending numeric class ID.

Every class name must be a non-empty string. Surrounding whitespace is
stripped, internal spaces and case are preserved, and normalized names must
be unique. Thus `fish` and `Fish` are distinct, while `fish` and `" fish "`
are duplicates.

## Declared class count

`nc` is optional. When present, it must be a positive non-boolean integer and
must equal the number of successfully parsed classes. The validated model
does not store `nc` redundantly; `number_of_classes` derives from the class
tuple.

The mismatch check is skipped when class parsing already failed, avoiding a
dependent or misleading count error.

## Representative valid configuration

```yaml
path: ../datasets/marine-life
train: images/train
val: images/val
test: images/test

names:
  0: fish
  1: turtle
  2: shark

nc: 3
```

The list-based equivalent may omit `path` and `test`:

```yaml
train: images/train
val: images/val

names:
  - fish
  - turtle
  - shark
```

Here the dataset root is the YAML file's parent and `test_path` is `None`.

## Representative invalid configurations

Missing training split:

```yaml
val: images/val
names: [fish]
```

Non-contiguous IDs:

```yaml
train: images/train
val: images/val
names: {0: fish, 2: shark}
```

Duplicate normalized names:

```yaml
train: images/train
val: images/val
names: [fish, " fish "]
```

Mismatched class count:

```yaml
train: images/train
val: images/val
names: [fish, turtle]
nc: 3
```

Unsupported list-valued split:

```yaml
train: [images/train-a, images/train-b]
val: images/val
names: [fish]
```

## Error ordering

After successful YAML parsing, independent errors are ordered by `path`,
`train`, `val`, `test`, names structure, individual key/name source order,
class-ID contiguity, duplicate name by class ID, `nc` type, then `nc`
mismatch. Validation continues across independent fields. File, YAML-syntax,
and non-mapping-root failures return immediately with their single defined
error.

## Configured-class validation relationship

The ordered class definitions in a validated `YoloDatasetConfiguration` can
be supplied to the separate
[configured-class validator](yolo-class-validation.md) alongside an existing
`YoloDatasetAnalysisResult`. Configuration validation remains independently
usable: it does not inspect annotations or invoke dataset analysis. Comparing
observed annotation IDs with configured definitions is a pure, separate
composition step.

## Configured split-planning relationship

The split paths in a validated `YoloDatasetConfiguration` can feed the
separate [configured split planner](yolo-split-planning.md). Planning consumes
the already constructed train, validation, and optional test paths, derives
explicit image and label directory pairs, and does not reparse YAML or inspect
any path.

Configuration validation remains independently usable. It neither invokes
split planning nor requires configured directories to exist.

## Configured split-execution relationship

The [configured split executor](yolo-split-analysis.md) uses the configuration's
class definitions for every successfully analyzed split without reparsing
YAML. Before execution, exact `config_path` equality ensures that an unrelated
configuration and split plan are not combined. Paths are compared lexically
without resolution.

## Current limitations

The component does not inspect or create directories, discover images or
labels, infer label paths, invoke `analyze_yolo_dataset()`, analyze splits,
detect leakage, or assess training readiness. It does not download data,
expand paths, rewrite or emit YAML, or access remote URLs. Observed-versus-
configured comparison is available only when callers explicitly compose this
validated result with an existing dataset analysis.

Dataset-summary reports, the JSONL image manifest, model training, inference,
and CLI commands remain separate and unchanged.
