# Poseidon AI roadmap

This roadmap separates current behavior from future direction. Planned and
exploratory entries are not delivery commitments and have no assigned dates.

## Nautilus Vision

### Completed

- Image loading with OpenCV
- Image metadata extraction
- Image validation with structured results
- Strict parsing and validation of individual YOLO detection-label files
- Dataset-level YOLO image-label pairing with missing-label, orphan-label,
  pairing-conflict, and annotation-count diagnostics
- Supported-image dataset discovery
- Optional recursive discovery in the internal loader API
- Optional recursive dataset scanning through the `--recursive` CLI flag
- Configurable minimum width and height through `--min-width` and
  `--min-height`
- Aspect-ratio-preserving letterbox preprocessing
- Dataset totals, valid/invalid counts, dimensions, and valid-image size
- Normalized image-extension statistics
- Per-format total, valid, invalid, total-valid-byte, and average-valid-byte
  statistics across every aggregate report format
- Decoded valid-image channel distribution statistics across all report
  formats
- Valid-image pixel-area and megapixel statistics across every report format
- Valid-image aspect-ratio and landscape, portrait, and square orientation
  statistics across every aggregate report format
- Valid-image minimum, maximum, and average file-size statistics across every
  aggregate report format
- SHA-256 exact duplicate detection for valid images across every report
  format
- Deterministic per-image JSONL manifest export with validation, metadata,
  and duplicate-group membership
- Structured invalid-image diagnostics in every dataset report
- Text, JSON, CSV, and Markdown dataset reports
- UTF-8 report file output
- Formatter registry and legacy JSON shortcut
- Concise dataset-summary filesystem errors with status 1
- Installed `poseidon-dataset-summary` command
- Single-image `poseidon-inspect` command
- Console and rotating-file logging utilities
- Unit and CLI-level test coverage
- GitHub Actions runs the complete test suite on Ubuntu and Windows with
  Python 3.13
- Repository overview, architecture, CLI, and roadmap documentation

### In progress

- No active implementation represented in the current repository

### Planned

- Class-name and dataset-YAML configuration
- Training split validation
- Segmentation and pose annotation support
- Model-training and inference capabilities

### Exploratory

- Video and live-camera processing
- Marine-life detection workflows
- Dataset quality visualization

## BuddySense

### Completed

- No implemented capabilities in the current repository

### In progress

- No active implementation represented in the current repository

### Planned

- Diver and buddy-awareness domain design

### Exploratory

- Diver association and tracking concepts
- Signals that could support buddy-separation awareness

## CurrentAI

### Completed

- No implemented capabilities in the current repository

### In progress

- No active implementation represented in the current repository

### Planned

- Ocean-current intelligence domain design

### Exploratory

- Current estimation and prediction inputs
- Integration of environmental observations

## DecoGuard

### Completed

- No implemented capabilities in the current repository

### In progress

- No active implementation represented in the current repository

### Planned

- Decompression-safety domain design

### Exploratory

- Decompression-risk analysis concepts
- Interfaces for future dive-profile data

## DiveAware

### Completed

- No implemented capabilities in the current repository

### In progress

- No active implementation represented in the current repository

### Planned

- Dive-analytics domain design

### Exploratory

- Dive-event and trend analysis
- Interfaces between dive data and future intelligence components

## SurfaceOps

### Completed

- No implemented capabilities in the current repository

### In progress

- No active implementation represented in the current repository

### Planned

- Surface monitoring and operational-interface design

### Exploratory

- Operator dashboards
- Cross-component status and alert presentation
