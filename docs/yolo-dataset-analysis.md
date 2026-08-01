# YOLO dataset analysis

Nautilus Vision provides deterministic library-level analysis for explicitly
supplied image and YOLO label directories. It discovers candidates, pairs
them by relative paths, validates unique label pairs, and returns immutable
diagnostics and annotation counts without decoding image contents.

## Public API

```python
from poseidon_ai.nautilus_vision.yolo_dataset import (
    YoloClassCount,
    YoloDatasetAnalysisResult,
    YoloImageLabelPair,
    YoloPairingConflict,
    analyze_yolo_dataset,
)

result = analyze_yolo_dataset(
    "dataset/images",
    "dataset/labels",
    recursive=True,
)
```

All four result models are frozen and slotted. Public collections are
immutable tuples, paths remain `Path` objects, and class identifiers remain
integers.

## Directory and discovery rules

Image and label roots are explicit arguments. The image root is checked
first, then the label root; each must exist and be a directory. Configuration
failures raise `FileNotFoundError` or `NotADirectoryError`. Roots are stored
as supplied without resolution, and unexpected filesystem failures retain
normal Python exception semantics.

Image discovery reuses the existing supported-image discovery mechanism and
case-insensitive extension definition. It requests discovery with image
validation disabled, so arbitrary candidate bytes are accepted and images
are not decoded, validated, hashed, or inspected for metadata.

Labels are regular files with a case-insensitive `.txt` suffix. Other files
and directories are ignored. Discovery does not parse labels.

With `recursive=False`, only direct child files are considered. With
`recursive=True`, eligible descendants under both roots are considered.

## Pairing

Each pairing key is calculated by:

1. making the candidate path relative to its corresponding root;
2. removing only the final suffix;
3. converting the remaining relative path to POSIX form.

For example, these candidates both produce `train/fish`:

```text
images/train/fish.jpg
labels/train/fish.txt
```

Directory and stem casing is preserved. Pairing keys are case-sensitive, so
`Fish.jpg` does not pair with `fish.txt`. A name such as
`archive/fish.v1.jpeg` produces `archive/fish.v1`.

For every key:

- one image and one label produce an ordinary pair;
- one image and no label produce a missing-label diagnostic;
- no image and one label produce an orphan-label diagnostic;
- multiple images or multiple labels produce one pairing conflict.

Conflict precedence prevents ambiguous paths from also appearing as missing
or orphan diagnostics. A conflict is never resolved by choosing the first
path, and its labels are not validated.

## Label validation and counts

Every uniquely paired label is passed exactly once to
`validate_yolo_label()`. Its exact immutable `YoloLabelValidationResult` is
retained on the pair, including ordered errors and any valid annotations
parsed from other lines.

The dataset result reports:

- `valid_label_files`;
- `invalid_label_files`;
- `empty_label_files`;
- `total_annotations`;
- numeric per-class `class_counts`.

An empty or whitespace-only valid label represents an image with no object
annotations. It counts as both valid and empty, and contributes zero
annotations.

Aggregate annotation statistics include only fully valid paired labels.
Commit 41 preserves valid-line annotations when another line is invalid;
those partial annotations remain inspectable through the pair but are
excluded from dataset totals. Orphan and conflicting labels are not parsed
and do not contribute to paired-label or annotation counts.

Class counts are `YoloClassCount` values ordered by ascending numeric class
identifier. Duplicate annotation lines are counted independently.

## Deterministic ordering

- discovered candidates use root-relative POSIX paths;
- pairs and conflicts are ordered by pairing key;
- missing images and orphan labels are ordered by their relative paths;
- paths inside conflicts are ordered by their corresponding relative paths;
- class counts are ordered by ascending class identifier.

Filesystem iteration order does not affect the public result.

## Dataset-configuration relationship

`YoloDatasetAnalysisResult` remains configuration-independent, and
`analyze_yolo_dataset()` continues to accept explicit image and label roots.
The separate [configured-class validator](yolo-class-validation.md) consumes
its immutable ordinary-pair results together with an already validated
configuration.

This composition step does not change Commit 42 aggregate class counts or
totals. It inspects every successfully parsed annotation retained in an
ordinary pair for unknown configured IDs, including partial annotations from
invalid files. Those partial annotations remain excluded from aggregate and
configured usage counts. Single-split analysis remains independently callable,
while configured multi-split execution is a separate explicit composition.

## Split-planning relationship

The separate [configured split planner](yolo-split-planning.md) converts an
already validated configuration into explicit image and label directory pairs.
`analyze_yolo_dataset()` still requires callers to supply one image root and
one label root explicitly. Planning does not invoke analysis automatically;
the separate executor consumes a completed plan when callers request it.

## Configured split-execution relationship

`analyze_yolo_dataset()` remains the independently usable single-split
analysis boundary. The configured
[split executor](yolo-split-analysis.md) invokes it exactly once for every
planned split and preserves each returned `YoloDatasetAnalysisResult` for
independent inspection. It introduces no cross-split totals, class counts, or
combined diagnostics.

## Cross-split summary relationship

Existing per-split dataset totals feed the separate immutable
[cross-split summary](yolo-split-summary.md). Missing-label, orphan-label,
pairing-conflict, and label-validity counts remain independently available in
each successful split summary and are summed across successful splits only.
This composition does not alter or rerun single-split analysis.

## Example layout

```text
dataset/
  images/
    train/
      fish.jpg
      turtle.png
      background.jpeg
  labels/
    train/
      fish.txt
      turtle.txt
      background.txt
```

With recursive analysis, all three relative keys pair. `background.txt` may
be empty and valid, representing an image with no annotations.

A missing-label example:

```text
images/train/shark.jpg
```

with no `labels/train/shark.txt` appears in `missing_label_images`.

An orphan-label example:

```text
labels/train/ray.txt
```

with no supported image sharing `train/ray` appears in
`orphan_label_files`.

A conflict example:

```text
images/train/fish.jpg
images/train/fish.png
labels/train/fish.txt
```

Both images share `train/fish`, so the key produces one
`YoloPairingConflict`, no ordinary pair, and no missing or orphan diagnostic.

## Empty datasets

When both roots contain no supported candidates, every collection is empty
and every count is zero.

## Current limitations

This component does not infer an `images` or `labels` directory or discover
training, validation, or test splits. The separate configuration validator
parses YAML but does not invoke dataset analysis. This component does not
decode or validate images, enforce box boundaries, convert coordinates to
pixels, rewrite labels, or detect duplicate or overlapping boxes.

Segmentation polygons, pose keypoints, oriented boxes, model training, and
inference are not supported. Dataset-summary reports and the JSONL image
manifest remain separate and unchanged. No label-analysis CLI is provided.
