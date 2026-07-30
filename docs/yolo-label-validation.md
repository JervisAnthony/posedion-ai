# YOLO label validation

Nautilus Vision provides strict library-level parsing and validation for one
YOLO object-detection label file at a time. This boundary is independent of
image dataset analysis and is intended for reuse by future annotation
workflows.

## Public API

```python
from poseidon_ai.nautilus_vision.yolo_label import (
    YoloDetectionAnnotation,
    YoloLabelValidationResult,
    validate_yolo_label,
)

result = validate_yolo_label("labels/example.txt")
```

`YoloDetectionAnnotation` and `YoloLabelValidationResult` are frozen, slotted
dataclasses. Annotation and error collections are immutable tuples.

## Detection format

Every nonblank line must contain exactly five whitespace-separated fields:

```text
class_id x_center y_center width height
```

Valid examples:

```text
0 0.500000 0.500000 0.250000 0.300000
3 0.125000 0.750000 0.100000 0.200000
```

Spaces, tabs, and surrounding whitespace are accepted. Blank lines are
ignored but still count toward physical one-based line numbers. Empty and
whitespace-only files are valid and represent zero annotations.

The format has no comment syntax. A `#` token or trailing comment text is
ordinary input and can cause a field-count or numeric error.

## File rules

- The path must exist and be a regular file.
- Only `.txt` is supported, matched case-insensitively, so `.TXT` is valid.
- Content must decode as UTF-8.
- Missing paths, directories, unsupported suffixes, and invalid UTF-8 return
  structured validation errors.
- Other filesystem errors retain normal Python exception semantics.

## Field rules

`class_id` must be a base-10 integer greater than or equal to zero. There is
no configured maximum and no class-name mapping.

Coordinates are parsed with `float()` and must be finite:

- `x_center` and `y_center` use the inclusive range `[0.0, 1.0]`.
- `width` and `height` use the range `(0.0, 1.0]`.
- NaN and positive or negative infinity are invalid.
- Zero, negative zero, and negative values are invalid for width and height.

Parsed coordinate values are not rounded.

## Errors and partial parsing

Errors contain physical source line numbers and remain ordered first by line,
then by field: `class_id`, `x_center`, `y_center`, `width`, and `height`.
Every field on a five-field line is validated, so one line can produce
multiple errors. A malformed field count produces one count error and skips
field parsing for that line.

Parsing continues after invalid lines. Valid annotations remain available in
source order even when `is_valid` is false because another line failed.

Representative invalid input:

```text
-1 nan 2.0 0 -0.5
0 0.5 0.5 0.2
0 0.5 0.5 0.2 0.3 # comment
```

## Explicit boundaries

Validation checks normalized stored fields only. It does not calculate box
edges or reject a box that would cross an image boundary. It does not open an
image, derive pixel coordinates, clip boxes, calculate area or intersection
over union, or detect duplicate or overlapping boxes.

The component does not provide dataset-level image-label pairing,
missing-label or orphan-label detection, class-name configuration,
segmentation polygons, pose/keypoint labels, YAML configuration, model
training, inference, report integration, or a CLI command.
