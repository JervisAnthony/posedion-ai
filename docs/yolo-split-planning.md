# YOLO configured split planning

Nautilus Vision can convert an already validated YOLO dataset configuration
into an immutable plan of explicit image and label directory pairs. Planning
is deterministic and purely lexical: it does not inspect or execute any split.

## Public API

```python
from poseidon_ai.nautilus_vision.yolo_split_plan import (
    YoloDatasetSplit,
    YoloDatasetSplitPlan,
    YoloDatasetSplitPlanValidationResult,
    build_yolo_dataset_split_plan,
)

result = build_yolo_dataset_split_plan(configuration)
```

`configuration` must be an existing validated `YoloDatasetConfiguration`.
The planner does not load or revalidate its YAML source.

All three public models are frozen and slotted dataclasses. Split and error
collections are immutable tuples. A valid result contains a complete plan and
no errors; an invalid result contains no partial plan.

## Split ordering and access

Plans always contain splits in this order:

1. `train`;
2. `validation`;
3. `test`, only when `configuration.test_path` is not `None`.

The plan's `training_split`, `validation_split`, and `test_split` properties
return the existing tuple members. `test_split` is `None` when test is absent.
The exact configuration `config_path` is retained on the plan.

## Default image-to-label convention

By default, the planner finds complete path components named `images` and
replaces the final match with `labels`:

```text
dataset/images/train
    -> dataset/labels/train

collections/images/archive/images/train
    -> collections/images/archive/labels/train
```

Replacement is component-based. Text embedded inside `marine-images`,
`images-backup`, or `myimages` does not match. Matching is case-sensitive, so
`Images` does not match the default `images` component.

## Custom directory names

Callers can supply other conventional component names:

```python
result = build_yolo_dataset_split_plan(
    configuration,
    images_directory_name="pictures",
    labels_directory_name="annotations",
)
```

Surrounding option whitespace is stripped. Each option must otherwise be a
non-empty single path component: it cannot be `.`, `..`, absolute, non-string,
or contain `/` or `\`. Image and label names must differ using case-sensitive
comparison.

Option errors are ordered first for the image name, then the label name, then
valid-name equality. Invalid options stop planning before configuration paths
are inspected.

## Path preservation

The configured image `Path` is stored exactly. Label paths are reconstructed
from `Path.parts`, preserving:

- relative or absolute form;
- drive and anchor information;
- `..` components;
- spaces and case;
- home-marker and environment-variable text;
- every component except the final exact image-directory match.

The planner never calls `resolve()`, `absolute()`, or `expanduser()`, and it
does not expand environment variables or normalize Unicode.

## Representative configuration

Given a configuration at `config/data.yaml` containing:

```yaml
path: ../datasets/marine
train: images/train
val: images/val
test: images/test

names:
  - fish
  - turtle
  - shark
```

the validated configuration constructs paths relative to the YAML parent.
Planning then produces:

```text
train
  image: config/../datasets/marine/images/train
  label: config/../datasets/marine/labels/train

validation
  image: config/../datasets/marine/images/val
  label: config/../datasets/marine/labels/val

test
  image: config/../datasets/marine/images/test
  label: config/../datasets/marine/labels/test
```

The `..` component remains present. None of these directories needs to exist.

## Deterministic validation errors

Configured paths are checked in train, validation, then optional test order.
Every missing image-directory component produces an error such as:

```text
Split 'train' image path does not contain directory component 'images': dataset/train
```

All affected configured splits are reported. The planner does not guess a
label path and returns `plan=None` whenever any split fails.

Invalid directory-name examples include:

```text
""
"   "
"."
".."
"data/images"
"data\images"
"/images"
Path("images")
42
None
```

## Current limitations

This component only plans configured paths. It performs no path existence or
directory checks, creation, discovery, traversal, globbing, image or label
opening, image-label pairing, label parsing, dataset analysis, configured-class
validation, split statistics, leakage detection, or training-readiness
assessment.

Automatic execution across planned splits remains future work. Reports, the
JSONL image manifest, CLIs, training, and inference remain separate and
unchanged.
