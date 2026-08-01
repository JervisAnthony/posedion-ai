# YOLO configured-class validation

Nautilus Vision can compare the annotation class IDs already retained by a
YOLO dataset analysis with the ordered class definitions from an already
validated dataset configuration. This library-only component reports
configured usage, unknown IDs, and unobserved configured classes without any
additional filesystem work.

## Public API

```python
from poseidon_ai.nautilus_vision.yolo_class_validation import (
    YoloConfiguredClassUsage,
    YoloDatasetClassValidationResult,
    YoloUnknownClassOccurrence,
    validate_yolo_dataset_classes,
)

result = validate_yolo_dataset_classes(
    configuration,
    dataset_analysis,
)
```

`configuration` must be an existing `YoloDatasetConfiguration`, and
`dataset_analysis` must be an existing `YoloDatasetAnalysisResult`. Their
independent validators and analyzers are not called by this function.

All three public result models are frozen and slotted dataclasses. Every
collection is an immutable tuple, and the result stores neither complete input
object.

## Pure in-memory composition

The validator reads only the supplied immutable objects. It performs no YAML
loading, label parsing, image access, file discovery, directory traversal, or
path existence checks. It does not invoke `validate_yolo_label()`,
`analyze_yolo_dataset()`, or `validate_yolo_dataset_config()`.

Only annotations retained in ordinary unique pairs are available for
inspection. Missing-label images, orphan labels, and pairing-conflict labels
are not inspected because dataset analysis did not parse them.

## Configured usage

`class_usage` contains one `YoloConfiguredClassUsage` per configured class,
ordered by ascending class ID. Each entry preserves the configuration's exact
normalized class name and provides `annotation_count`.

Usage follows dataset aggregate semantics:

- annotations in fully valid paired labels contribute to known configured
  class counts;
- duplicate valid annotation lines each contribute;
- empty valid labels contribute zero;
- every annotation in an invalid label contributes zero;
- unknown IDs never increment a configured class;
- missing, orphan, and conflicting labels contribute zero.

For example, two valid class-0 annotations and one valid class-1 annotation
against `fish`, `turtle`, and `shark` produce counts `2`, `1`, and `0`.

## Unknown-class diagnostics

Every successfully parsed annotation retained in every ordinary pair is
checked, including valid-line annotations retained inside otherwise invalid
labels. An ID absent from `configuration.classes` creates one
`YoloUnknownClassOccurrence` with:

- the parsed non-negative `class_id`;
- the pair's exact case-sensitive `pairing_key`;
- the pair's exact `label_path`;
- the annotation's physical one-based `line_number`;
- the containing label result's `label_is_valid` value.

Occurrences are never deduplicated. Each occurrence produces one error:

```text
Pair 'train/fish', line 3: class_id 7 is not defined in the dataset configuration.
```

Unknown annotations from invalid labels are reported for complete available
diagnostic coverage, but no annotation from an invalid label contributes to
configured usage.

Occurrences are ordered by pairing key, numeric line number, numeric class ID,
then the label path's POSIX text. Errors follow exactly the same order, making
results independent of input pair and filesystem iteration order.

## Unobserved classes and validity

`unobserved_classes` contains the exact `YoloClassDefinition` objects whose
valid configured usage is zero, ordered by class ID. They are informational,
not errors, and do not invalidate a result.

`is_valid` is true exactly when there are no unknown occurrences. Therefore:

- a compatible dataset is valid even when some configured classes are unused;
- an empty dataset is valid and returns every configured class as unobserved;
- invalid-only labels leave every configured class unobserved;
- unknown-only annotations leave every configured class unobserved and make
  the result invalid.

Validation does not stop after the first unknown ID. An invalid result still
contains complete usage, occurrences, unobserved classes, and ordered errors.

## Representative example

```python
class_result = validate_yolo_dataset_classes(
    configuration,
    dataset_analysis,
)

for usage in class_result.class_usage:
    print(usage.class_id, usage.name, usage.annotation_count)

for occurrence in class_result.unknown_class_occurrences:
    print(
        occurrence.pairing_key,
        occurrence.line_number,
        occurrence.class_id,
        occurrence.label_is_valid,
    )
```

## Explicit limitations

The component does not load configuration files, parse YAML or labels,
discover or traverse paths, decode images, infer dataset layouts, orchestrate
train/validation/test splits, remap class IDs, rewrite labels, calculate
percentages or class-balance policy, or assess full training readiness.

It does not integrate with dataset-summary reports or the JSONL image
manifest, add a CLI command, train a model, or run inference.
