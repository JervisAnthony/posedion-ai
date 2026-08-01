# YOLO configured split analysis

Nautilus Vision can execute an existing immutable YOLO split plan in its
supplied order. Each planned image and label directory pair is delegated to
the existing dataset analyzer, then every successful analysis is delegated to
configured-class validation.

## Public API

```python
from poseidon_ai.nautilus_vision.yolo_split_analysis import (
    YoloConfiguredSplitAnalysisResult,
    YoloDatasetSplitAnalysis,
    YoloDatasetSplitAnalysisFailure,
    analyze_yolo_dataset_splits,
)

result = analyze_yolo_dataset_splits(
    configuration,
    split_plan,
    recursive=True,
)
```

Callers must supply an existing `YoloDatasetConfiguration` and
`YoloDatasetSplitPlan`. The executor does not parse YAML, validate a
configuration, or construct a plan.

All three public models are frozen and slotted dataclasses. Public outcome
collections are immutable tuples, and successful outcomes preserve the exact
nested analysis and validation objects returned by their owning components.

## Configuration compatibility

Before any split is executed, `split_plan.config_path` must equal
`configuration.config_path` using exact `Path` equality. Neither path is
resolved. A mismatch raises:

```text
Split plan config_path does not match the dataset configuration config_path.
```

This guard prevents accidentally combining a plan and configuration produced
for different YAML sources. It does not broadly revalidate either input or
compare individual planned directories with configuration split paths.

## Ordered split execution

Every entry in `split_plan.splits` is processed exactly once in its existing
order. For each split, the executor calls:

```python
dataset_analysis = analyze_yolo_dataset(
    split.image_directory,
    split.label_directory,
    recursive=recursive,
)
```

The exact `Path` objects and common recursive value are forwarded. Discovery,
image-label pairing, label parsing, and statistics remain owned by the
single-split analyzer.

After a successful analysis, the executor calls exactly once:

```python
class_validation = validate_yolo_dataset_classes(
    configuration,
    dataset_analysis,
)
```

Configured-class usage and unknown-ID detection remain owned by that existing
validator.

## Successful outcomes

`YoloDatasetSplitAnalysis` contains:

- the exact planned `split`;
- the exact `YoloDatasetAnalysisResult` returned by dataset analysis;
- the exact `YoloDatasetClassValidationResult` returned by configured-class
  validation.

Content diagnostics do not make split execution fail. Invalid label files,
missing-label images, orphan labels, pairing conflicts, unknown configured
class IDs, unobserved classes, empty datasets, and zero annotations remain
inspectable inside successful outcomes.

A class-invalid outcome is therefore still operationally successful.

## Expected root failures

Only `FileNotFoundError` and `NotADirectoryError` raised by
`analyze_yolo_dataset()` are converted into
`YoloDatasetSplitAnalysisFailure`. A failure preserves:

- the exact planned split;
- the exact exception class name in `error_type`;
- `str(exception)` in `message`.

No traceback or exception object is stored. Configured-class validation is not
called for a failed split, and the executor continues with later planned
splits. Multiple failures remain ordered relative to the plan.

## Unexpected failures

Unexpected failures propagate normally. This includes `PermissionError`,
generic `OSError`, unexpected decoding failures, programming errors,
configured-class validation errors, `KeyboardInterrupt`, and `SystemExit`.
They are neither converted into outcomes nor silently swallowed.

## Completion and derived views

`YoloConfiguredSplitAnalysisResult` retains the plan's exact `config_path`,
the supplied recursive flag, and one ordered outcome per planned split.

Its derived properties are:

- `is_complete`: true exactly when no expected root-failure outcome exists;
- `successful_splits`: successful outcomes in original order;
- `failed_splits`: expected root failures in original order.

These views contain the exact outcome objects. They perform no filesystem
access. A manually supplied empty plan produces an empty outcome tuple and is
operationally complete.

Operational completeness does not mean annotation validity, dataset quality,
production readiness, or training readiness. Unknown classes and other
content diagnostics do not make `is_complete` false.

## Representative continuation example

Given train and validation roots that can be analyzed, plus a missing test
label root, the ordered outcomes are:

```text
train       YoloDatasetSplitAnalysis
validation  YoloDatasetSplitAnalysis
test        YoloDatasetSplitAnalysisFailure(FileNotFoundError)
```

The result is incomplete, while the train and validation dataset and class
results remain fully available.

## Recursive behavior

`recursive=False` is the default and preserves top-level-only analyzer
semantics. With `recursive=True`, the same value is forwarded to every split,
and nested discovery remains entirely owned by `analyze_yolo_dataset()`.

## Cross-split summary relationship

Configured split analysis remains the execution layer. The separate
[cross-split summary composer](yolo-split-summary.md) consumes its completed
immutable outcomes without executing or retrying any split. Successful
outcomes provide existing dataset and configured-class results; original
failure outcomes remain visible in order and are excluded from aggregate
totals.

Split analysis itself performs no cross-split aggregation.

## No execution-layer cross-split aggregation

Each split remains independent. The result does not calculate global image,
label, annotation, or class totals; combine diagnostics; inspect overlap;
detect cross-split duplicate content; calculate class balance; or assign
readiness status.

## Current limitations

This component does not read YAML, build plans, derive paths, duplicate
discovery or parsing, aggregate splits, detect leakage, serialize reports,
modify the JSONL manifest, provide a CLI command, train models, or run
inference.

Cross-split readiness policy, leakage diagnostics, and reporting remain future
layers.
