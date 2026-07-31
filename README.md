# Poseidon AI

[![Tests](https://github.com/JervisAnthony/posedion-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/JervisAnthony/posedion-ai/actions/workflows/tests.yml)

Poseidon AI is an evolving underwater-intelligence platform. Its currently
implemented component, **Nautilus Vision**, provides tested image and dataset
engineering utilities for underwater computer-vision workflows.

## Overview

The repository is useful today for inspecting, validating, preprocessing, and
summarizing image datasets. Broader diver-safety, ocean-intelligence, model
training, inference, and operational-monitoring capabilities remain roadmap
work.

## Status

| Area | Status |
|------|--------|
| Nautilus Vision image utilities | Implemented |
| Nautilus Vision dataset analysis and reporting | Implemented |
| Invalid-image diagnostic reporting | Implemented in all dataset reports |
| Dataset-summary installed command | Implemented |
| Model training and inference | Planned |
| Other Poseidon AI components | Planned or exploratory |

## Current Nautilus Vision capabilities

- Discovers supported image files in deterministic path order.
- Loads images with OpenCV and extracts filename, dimensions, channel count,
  and file size.
- Validates file existence, supported extensions, decodability, and
  independently configurable minimum dimensions, defaulting to 32 pixels.
- Parses and strictly validates individual UTF-8 YOLO detection-label files
  with deterministic, line-numbered errors.
- Analyzes explicitly supplied YOLO image and label directories by relative
  pairing keys, reporting missing, orphan, conflicting, valid, invalid,
  empty, total-annotation, and per-class annotation counts.
- Safely parses and validates library-level YOLO dataset YAML configuration,
  including split paths, ordered class definitions, and optional class counts.
- Resizes and pads images while preserving aspect ratio.
- Analyzes valid and invalid image counts, width and height statistics, valid
  image bytes, normalized extension counts, and decoded valid-image channel
  counts, plus valid-image pixel-area, megapixel, aspect-ratio, and
  orientation statistics, minimum, maximum, and average file sizes, and
  per-format valid, invalid, and valid-byte statistics.
- Detects byte-identical valid image files using SHA-256 exact-content hashes.
- Captures structured paths and every validation error for invalid images,
  then exposes them in each dataset report.
- Renders dataset reports as text, JSON, CSV, or Markdown.
- Writes any selected report to a UTF-8 file.
- Exports a deterministic JSONL manifest with one relative, portable record
  per supported valid or invalid image candidate.
- Reports common dataset and output-path failures without tracebacks.
- Provides centralized console and rotating-file logging utilities.
- Uses `pyproject.toml` for package metadata, dependencies, Python support,
  and the installed `poseidon-inspect` and `poseidon-dataset-summary`
  commands.

## Installation

Poseidon AI requires Python `>=3.13,<3.14`.

```bash
git clone https://github.com/JervisAnthony/posedion-ai.git
cd posedion-ai
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the package in editable mode with its development dependencies:

```bash
python -m pip install -e ".[dev]"
```

This installs the declared runtime dependencies, NumPy, OpenCV, and PyYAML,
plus pytest from the `dev` extra.

## Quick start

Inspect one image after installation:

```bash
poseidon-inspect path/to/image.jpg
```

Summarize the supported images directly inside a directory:

```bash
poseidon-dataset-summary path/to/dataset
```

The equivalent module invocation remains available:

```bash
python -m poseidon_ai.nautilus_vision.dataset_summary path/to/dataset
```

Scanning is top-level only by default. Use `--recursive` to include supported
images in nested directories. Full syntax and behavior are covered in the
[dataset-summary CLI guide](docs/dataset-summary-cli.md).

## YOLO label validation and dataset analysis

Each nonblank line in a YOLO detection label uses:

```text
class_id x_center y_center width height
```

Class identifiers must be non-negative integers. Center coordinates use the
inclusive range `[0, 1]`; widths and heights use `(0, 1]`. Empty label files
are valid and represent no annotations. Parsing reports deterministic errors
with physical source line numbers.

```python
from poseidon_ai.nautilus_vision.yolo_label import (
    validate_yolo_label,
)

result = validate_yolo_label("labels/example.txt")
```

The single-file validator remains independently usable. Dataset analysis
pairs supported images and `.txt` labels by their root-relative POSIX paths
without final extensions:

```python
from poseidon_ai.nautilus_vision.yolo_dataset import (
    analyze_yolo_dataset,
)

result = analyze_yolo_dataset(
    "dataset/images",
    "dataset/labels",
    recursive=True,
)

print(result.total_annotations)
print(result.missing_label_images)
print(result.orphan_label_files)
```

The result includes missing-label images, orphan labels, ambiguous pairing-key
conflicts, valid, invalid, and empty paired-label counts, total annotations,
and numeric per-class counts. Aggregate annotations come only from fully
valid paired labels; an empty valid label represents an image with no
annotations. This component does not decode image contents or configure class
names. It is library-only: no label CLI or dataset-summary integration is
provided.

See [single-file YOLO validation](docs/yolo-label-validation.md) and
[YOLO dataset analysis](docs/yolo-dataset-analysis.md).

### YOLO dataset configuration

The library accepts UTF-8 `.yaml` and `.yml` files with an optional dataset
root, required `train` and `val` split paths, an optional `test` path, class
names as a list or contiguous integer-keyed mapping, and an optional `nc`
consistency declaration.

```python
from poseidon_ai.nautilus_vision.yolo_config import (
    validate_yolo_dataset_config,
)

result = validate_yolo_dataset_config("dataset/data.yaml")

if result.is_valid:
    config = result.configuration
    print(config.train_path)
    print(config.number_of_classes)
```

Configuration validation is library-only. Paths are constructed but not
traversed or required to exist, split paths are not analyzed automatically,
and configuration is not connected to model training. No label-analysis or
configuration CLI is installed. See
[YOLO dataset configuration](docs/yolo-dataset-configuration.md).

## Dataset-summary CLI

```bash
# Default text report
poseidon-dataset-summary path/to/dataset

# Include images in nested directories
poseidon-dataset-summary path/to/dataset --recursive

# Require images to be at least 64 pixels wide and high
poseidon-dataset-summary path/to/dataset --min-width 64 --min-height 64

# JSON using the format selector
poseidon-dataset-summary path/to/dataset --format json

# Backward-compatible JSON shortcut
poseidon-dataset-summary path/to/dataset --json

# CSV
poseidon-dataset-summary path/to/dataset --format csv

# Markdown
poseidon-dataset-summary path/to/dataset --format markdown

# Write the selected report instead of printing it
poseidon-dataset-summary path/to/dataset --format json --output report.json

# Export a per-candidate JSONL manifest alongside the report
poseidon-dataset-summary path/to/dataset \
    --manifest-output dataset-manifest.jsonl
```

Valid manifest entries contain the metadata already collected by analysis.
Invalid entries contain ordered validation errors and null metadata fields;
unsupported files are omitted.

## Output examples

Given two valid images (`JPEG` and `PNG`), representative output is:

### Text

```text
Dataset Summary
========================
Dataset Path      : data/sample_dataset
Total Images      : 2
Valid Images      : 2
Invalid Images    : 0

Image Formats
-------------
JPEG              : 1
PNG               : 1

Image Format Statistics
-----------------------
JPEG
  Total Images        : 1
  Valid Images        : 1
  Invalid Images      : 0
  Total Valid Bytes   : 512
  Average Valid Bytes : 512.00

PNG
  Total Images        : 1
  Valid Images        : 1
  Invalid Images      : 0
  Total Valid Bytes   : 1,536
  Average Valid Bytes : 1,536.00

Image Channels
--------------
3 channels         : 2

Image Resolution
----------------
Minimum Pixels    : 307,200
Maximum Pixels    : 921,600
Average Pixels    : 614,400.00
Minimum MP        : 0.31
Maximum MP        : 0.92
Average MP        : 0.61

Image Aspect Ratios
-------------------
Minimum Ratio      : 1.33
Maximum Ratio      : 1.78
Average Ratio      : 1.56
Landscape Images   : 2
Portrait Images    : 0
Square Images      : 0

Image File Sizes
----------------
Minimum Bytes      : 512
Maximum Bytes      : 1,536
Average Bytes      : 1,024.00

Exact Duplicate Images
----------------------
No exact duplicate images found.

Invalid Image Diagnostics
-------------------------
No invalid images found.
```

The complete report continues with width, height, and formatted dataset-size
sections.

### JSON

```json
{
  "dataset_path": "data/sample_dataset",
  "total_images": 2,
  "valid_images": 2,
  "invalid_images": 0,
  "extension_counts": {
    "jpeg": 1,
    "png": 1
  },
  "format_statistics": {
    "jpeg": {
      "total_images": 1,
      "valid_images": 1,
      "invalid_images": 0,
      "total_valid_size_bytes": 512,
      "average_valid_size_bytes": 512.0
    },
    "png": {
      "total_images": 1,
      "valid_images": 1,
      "invalid_images": 0,
      "total_valid_size_bytes": 1536,
      "average_valid_size_bytes": 1536.0
    }
  },
  "channel_counts": {
    "3": 2
  },
  "resolution_statistics": {
    "minimum_pixels": 307200,
    "maximum_pixels": 921600,
    "average_pixels": 614400.0,
    "minimum_megapixels": 0.3072,
    "maximum_megapixels": 0.9216,
    "average_megapixels": 0.6144
  },
  "aspect_ratio_statistics": {
    "minimum": 1.333333,
    "maximum": 1.777778,
    "average": 1.555556,
    "orientation_counts": {
      "landscape": 2,
      "portrait": 0,
      "square": 0
    }
  },
  "file_size_statistics": {
    "minimum_bytes": 512,
    "maximum_bytes": 1536,
    "average_bytes": 1024.0
  },
  "duplicate_images": {
    "group_count": 0,
    "file_count": 0,
    "redundant_copy_count": 0,
    "groups": []
  },
  "invalid_image_diagnostics": [],
  "width": {
    "minimum": 640,
    "maximum": 1280,
    "average": 960.0
  },
  "height": {
    "minimum": 480,
    "maximum": 720,
    "average": 600.0
  },
  "total_size_bytes": 2048,
  "formatted_size": "2.00 KB"
}
```

### CSV

```csv
dataset_path,total_images,valid_images,invalid_images,extension_counts,format_statistics,channel_counts,resolution_statistics,aspect_ratio_statistics,file_size_statistics,duplicate_images,invalid_image_diagnostics,min_width,max_width,average_width,min_height,max_height,average_height,total_size_bytes
data/sample_dataset,2,2,0,"{""jpeg"": 1, ""png"": 1}","{""jpeg"": {""total_images"": 1, ""valid_images"": 1, ""invalid_images"": 0, ""total_valid_size_bytes"": 512, ""average_valid_size_bytes"": 512.0}, ""png"": {""total_images"": 1, ""valid_images"": 1, ""invalid_images"": 0, ""total_valid_size_bytes"": 1536, ""average_valid_size_bytes"": 1536.0}}","{""3"": 2}","{""minimum_pixels"": 307200, ""maximum_pixels"": 921600, ""average_pixels"": 614400.0, ""minimum_megapixels"": 0.3072, ""maximum_megapixels"": 0.9216, ""average_megapixels"": 0.6144}","{""minimum"": 1.333333, ""maximum"": 1.777778, ""average"": 1.555556, ""orientation_counts"": {""landscape"": 2, ""portrait"": 0, ""square"": 0}}","{""minimum_bytes"": 512, ""maximum_bytes"": 1536, ""average_bytes"": 1024.0}","{""group_count"": 0, ""file_count"": 0, ""redundant_copy_count"": 0, ""groups"": []}",[],640,1280,960.00,480,720,600.00,2048
```

### Markdown

```markdown
# Dataset Summary

## Overview

| Metric | Value |
|--------|------:|
| Dataset Path | data/sample_dataset |
| Total Images | 2 |
| Valid Images | 2 |
| Invalid Images | 0 |

## Image Formats

| Format | Images |
|--------|-------:|
| JPEG | 1 |
| PNG | 1 |

## Image Format Statistics

| Format | Total | Valid | Invalid | Total Valid Bytes | Average Valid Bytes |
|--------|------:|------:|--------:|------------------:|--------------------:|
| JPEG | 1 | 1 | 0 | 512 | 512.00 |
| PNG | 1 | 1 | 0 | 1,536 | 1,536.00 |

## Image Channels

| Channels | Images |
|---------:|-------:|
| 3 | 2 |

## Image Resolution

| Metric | Pixels | Megapixels |
|--------|-------:|-----------:|
| Minimum | 307,200 | 0.31 |
| Maximum | 921,600 | 0.92 |
| Average | 614,400.00 | 0.61 |

## Image Aspect Ratios

| Metric | Value |
|--------|------:|
| Minimum Ratio | 1.33 |
| Maximum Ratio | 1.78 |
| Average Ratio | 1.56 |
| Landscape Images | 2 |
| Portrait Images | 0 |
| Square Images | 0 |

## Image File Sizes

| Metric | Bytes |
|--------|------:|
| Minimum | 512 |
| Maximum | 1,536 |
| Average | 1,024.00 |

## Exact Duplicate Images

No exact duplicate images found.

## Invalid Image Diagnostics

No invalid images found.
```

Channel counts are the numeric decoded channel counts reported by the current
metadata pipeline for valid images; they do not infer colour-mode semantics.
Per-format statistics use the existing normalized aliases (`jpg` to `jpeg`
and `tif` to `tiff`). Invalid candidates contribute to format totals but not
valid bytes; valid sizes reuse collected metadata, and duplicate files
contribute independently.
Resolution statistics use each valid image's actual decoded width × height
pixel area. Megapixels are derived from those areas without inferring DPI,
quality, or semantic resolution categories.
Aspect ratios use decoded `width / height`. Orientation compares those
integer dimensions directly as landscape, portrait, or square; invalid
images do not contribute, and EXIF orientation is not interpreted.
File-size statistics reuse each valid image's collected `size_bytes` metadata.
Invalid images do not contribute; exact duplicate members each contribute
because they remain separate dataset files.
Exact duplicate detection compares SHA-256 hashes of complete valid-file
bytes. Visually similar, resized, or re-encoded images are not duplicates
unless their bytes are identical.

## Supported image formats

Nautilus Vision discovers these case-insensitive suffixes:

| Format | Suffixes | Normalized report key |
|--------|----------|-----------------------|
| BMP | `.bmp` | `bmp` |
| JPEG | `.jpg`, `.jpeg` | `jpeg` |
| PNG | `.png` | `png` |
| TIFF | `.tif`, `.tiff` | `tiff` |
| WebP | `.webp` | `webp` |

## Running tests

```bash
# Complete suite
python -m pytest

# Focused formatter and CLI coverage
python -m pytest tests/test_dataset_summary.py -v
```

GitHub Actions runs the complete test suite on Ubuntu and Windows with
Python 3.13.

The suite contained 333 passing tests when this documentation was verified.

## Project structure

```text
posedion-ai/
├── docs/
│   ├── architecture.md
│   ├── dataset-summary-cli.md
│   ├── roadmap.md
│   ├── yolo-dataset-configuration.md
│   ├── yolo-dataset-analysis.md
│   └── yolo-label-validation.md
├── scripts/
│   └── create_sample_dataset.py
├── src/poseidon_ai/
│   ├── logging_config.py
│   ├── nautilus_vision/
│   │   ├── dataset_analyzer.py
│   │   ├── dataset_loader.py
│   │   ├── dataset_statistics.py
│   │   ├── dataset_summary.py
│   │   ├── image_hash.py
│   │   ├── image_validator.py
│   │   ├── yolo_config.py
│   │   ├── yolo_dataset.py
│   │   └── yolo_label.py
│   └── utils/
└── tests/
```

## Architecture

Nautilus Vision separates discovery, validation, metadata collection,
analysis, and presentation. A formatter registry gives the CLI a consistent
selection mechanism for all four report types. See
[Architecture](docs/architecture.md) for the component and data-flow details.

## Engineering principles

- Analysis and presentation remain separate.
- Structured dataclasses carry statistics and validation diagnostics.
- Formatter output is deterministic, including normalized extension and
  numeric channel order, resolution serialization, and duplicate groups.
- CSV serialization uses the Python standard library.
- CLI behavior and structured output are tested.
- Paths are handled with `pathlib`, with portable report tests.

## Roadmap

The [Poseidon AI roadmap](docs/roadmap.md) distinguishes completed,
in-progress, planned, and exploratory work across Nautilus Vision,
BuddySense, CurrentAI, DecoGuard, DiveAware, and SurfaceOps. Roadmap entries
describe direction, not delivery commitments.

## License

Poseidon AI is available under the [MIT License](LICENSE).
