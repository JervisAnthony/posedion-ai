# YOLO cross-split summary

Nautilus Vision can compose an already validated dataset configuration and an
already completed configured split analysis into one deterministic immutable
cross-split summary. The composer performs only in-memory reads of those
existing results.

## Public API

```python
from poseidon_ai.nautilus_vision.yolo_split_summary import (
    YoloCrossSplitClassUsage,
    YoloCrossSplitSummary,
    YoloSplitSummary,
    summarize_yolo_dataset_splits,
)

summary = summarize_yolo_dataset_splits(
    configuration,
    split_analysis,
)
```

The inputs must be an existing `YoloDatasetConfiguration` and an existing
`YoloConfiguredSplitAnalysisResult`. The composer neither creates nor
revalidates either input.

All three result models are frozen and slotted dataclasses. Every public
collection is an immutable tuple. The result stores neither complete input.

## Configuration compatibility

Before inspecting any outcome, the composer requires exact `Path` equality
between `configuration.config_path` and `split_analysis.config_path`. Paths
are compared lexically and are not resolved. A mismatch raises:

```text
Split analysis config_path does not match the dataset configuration config_path.
```

This compatibility check does not reparse YAML or broadly validate manually
constructed public dataclass values.

## Ordered outcome composition

Outcomes retain the split-analysis result's existing order. Every successful
`YoloDatasetSplitAnalysis` becomes a new `YoloSplitSummary`. Every
`YoloDatasetSplitAnalysisFailure` remains the exact original failure object,
including its split, exception type name, and message.

The summary does not execute, retry, or invent empty results for failed
splits. Failed splits contribute zero to every global total.

The derived `successful_summaries` and `failed_splits` tuples preserve order
and contain the exact objects already stored in the cross-split outcome tuple.
The split-count properties derive from those outcomes.

## Successful per-split fields

Each `YoloSplitSummary` retains the exact split and the existing
configured-class usage and unobserved-class tuples. It flattens these counts:

- total images and label files;
- uniquely paired images;
- missing-label images and orphan label files;
- pairing-conflict groups;
- valid, invalid, and empty label files;
- total annotations from fully valid labels;
- configured annotation usage;
- all unknown-class occurrences;
- unknown occurrences in valid labels;
- unknown occurrences retained from invalid labels;
- configured-class validation validity.

Pairing conflicts count groups, not the individual paths within a group.
Missing and orphan diagnostics use their tuple lengths. Counts are copied or
derived from existing immutable results; no path or annotation is reopened.

The per-split `class_usage` and `unobserved_classes` fields preserve the exact
source tuples and nested objects. The complete dataset-analysis and
class-validation results are not stored.

## Global totals

Every global dataset count is the sum of its corresponding successful
per-split count. This includes images, labels, pairs, missing images, orphan
labels, conflict groups, valid, invalid, and empty labels, annotations, and
unknown diagnostics.

Expected failure outcomes remain visible but are excluded from all totals.
An all-failed analysis therefore has zero totals. Existing empty successful
splits also contribute zero, but remain successful outcomes.

## Annotation and unknown-class semantics

`total_annotations` follows existing dataset-analysis semantics: it includes
all annotations from fully valid paired labels. This includes valid-label
annotations whose class IDs are not configured.

`configured_annotation_count` sums existing configured usage, which contains
only configured IDs from fully valid labels. Therefore, for valid public
inputs:

```text
configured_annotation_count
    + unknown_class_occurrences_in_valid_labels
    == total_annotations
```

Unknown occurrences retained from invalid labels remain diagnostics. They do
not contribute to `total_annotations` or configured usage. The total unknown
count is the sum of its valid-label and invalid-label counts.

## Configured class usage

Cross-split `class_usage` contains one `YoloCrossSplitClassUsage` for every
configured class. Entries use configuration order and names. Each entry
contains the sum of existing configured usage across successful splits and
the split names where that usage is positive.

Observed split names follow outcome order, are not sorted alphabetically, and
appear at most once per class. Failed split names never appear. Unknown class
IDs and annotations retained from invalid labels never enter configured usage.

## Globally unobserved classes

A configured class is globally unobserved when its aggregate configured usage
is zero. `unobserved_classes` contains the exact class-definition objects from
the configuration in configuration order.

Classes remain globally unobserved when they appear only in failed splits,
invalid labels, or as unknown IDs. Every class is unobserved for an all-failed
analysis or for successful but empty splits. This is informational and does
not affect operational completeness.

## Completion semantics

`is_complete` is true exactly when no expected failure outcome exists. It
means every planned split was analyzed successfully. It does not mean:

- all labels are valid or paired;
- every configured class is observed;
- no unknown classes or pairing conflicts exist;
- every split contains annotations;
- the dataset is balanced, leakage-free, or training ready.

A complete summary may contain any available content diagnostic.

## Representative outcomes

Given successful train and validation outcomes followed by a missing test
label root, summary outcome order remains:

```text
train       YoloSplitSummary
validation  YoloSplitSummary
test        YoloDatasetSplitAnalysisFailure
```

Only train and validation contribute to totals and observed split names. The
result is incomplete because the original test failure remains present.

With all-successful empty splits, every outcome is a zero-count summary,
global totals are zero, every configured class is unobserved, and the result
is complete. With all failures, totals are also zero and every configured
class is unobserved, but the result is incomplete.

## Explicit limitations

This component does not read configuration files, parse YAML or labels, build
plans, execute splits, discover or traverse files, open images or labels,
revalidate configured classes, inspect coordinates, hash content, detect
cross-split leakage or duplicates, score class balance, apply severity or
minimum-sample policies, or decide training readiness.

It does not serialize reports, alter image-report or JSONL manifest schemas,
add a CLI command, train models, or run inference. Those layers remain
separate.
