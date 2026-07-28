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
| Dataset-summary installed command | Planned; use the Python module today |
| Model training and inference | Planned |
| Other Poseidon AI components | Planned or exploratory |

## Current Nautilus Vision capabilities

- Discovers supported image files in deterministic path order.
- Loads images with OpenCV and extracts filename, dimensions, channel count,
  and file size.
- Validates file existence, supported extensions, decodability, and minimum
  dimensions.
- Resizes and pads images while preserving aspect ratio.
- Analyzes valid and invalid image counts, width and height statistics, valid
  image bytes, and normalized extension counts.
- Captures structured paths and every validation error for invalid images,
  then exposes them in each dataset report.
- Renders dataset reports as text, JSON, CSV, or Markdown.
- Writes any selected report to a UTF-8 file.
- Provides centralized console and rotating-file logging utilities.
- Uses `pyproject.toml` for package metadata, dependencies, Python support,
  and the installed `poseidon-inspect` command.

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

This installs the declared runtime dependencies, NumPy and OpenCV, plus
pytest from the `dev` extra.

## Quick start

Inspect one image after installation:

```bash
poseidon-inspect path/to/image.jpg
```

Summarize the supported images directly inside a directory:

```bash
python -m poseidon_ai.nautilus_vision.dataset_summary path/to/dataset
```

The dataset-summary CLI is not recursive. Its current syntax and behavior are
covered in the [dataset-summary CLI guide](docs/dataset-summary-cli.md).

## Dataset-summary CLI

```bash
# Default text report
python -m poseidon_ai.nautilus_vision.dataset_summary path/to/dataset

# JSON using the format selector
python -m poseidon_ai.nautilus_vision.dataset_summary path/to/dataset --format json

# Backward-compatible JSON shortcut
python -m poseidon_ai.nautilus_vision.dataset_summary path/to/dataset --json

# CSV
python -m poseidon_ai.nautilus_vision.dataset_summary path/to/dataset --format csv

# Markdown
python -m poseidon_ai.nautilus_vision.dataset_summary path/to/dataset --format markdown

# Write the selected report instead of printing it
python -m poseidon_ai.nautilus_vision.dataset_summary path/to/dataset --format json --output report.json
```

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
dataset_path,total_images,valid_images,invalid_images,extension_counts,invalid_image_diagnostics,min_width,max_width,average_width,min_height,max_height,average_height,total_size_bytes
data/sample_dataset,2,2,0,"{""jpeg"": 1, ""png"": 1}",[],640,1280,960.00,480,720,600.00,2048
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

## Invalid Image Diagnostics

No invalid images found.
```

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

The suite contained 59 passing tests when this documentation was verified.

## Project structure

```text
posedion-ai/
├── docs/
│   ├── architecture.md
│   ├── dataset-summary-cli.md
│   └── roadmap.md
├── scripts/
│   └── create_sample_dataset.py
├── src/poseidon_ai/
│   ├── logging_config.py
│   ├── nautilus_vision/
│   │   ├── dataset_analyzer.py
│   │   ├── dataset_loader.py
│   │   ├── dataset_statistics.py
│   │   ├── dataset_summary.py
│   │   └── image_validator.py
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
- Formatter output is deterministic, including normalized extension order.
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
