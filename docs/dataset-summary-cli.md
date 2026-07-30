# Dataset-summary CLI

The Nautilus Vision dataset-summary CLI analyzes supported images in a
dataset directory and renders one aggregate report. Scanning is top-level
only by default and can optionally include nested directories.

## Syntax

```text
poseidon-dataset-summary DATASET_PATH
    [--recursive]
    [--min-width PIXELS]
    [--min-height PIXELS]
    [--json]
    [--format {text,json,csv,markdown}]
    [--output OUTPUT]
```

`DATASET_PATH` is the required directory to inspect.

The installed command and this equivalent module invocation use the same
`main()` function:

```bash
python -m poseidon_ai.nautilus_vision.dataset_summary DATASET_PATH
```

## Options

| Option | Meaning |
|--------|---------|
| `--recursive` | Search for images in nested directories. |
| `--min-width PIXELS` | Set the minimum valid image width; default: `32`. |
| `--min-height PIXELS` | Set the minimum valid image height; default: `32`. |
| `--format {text,json,csv,markdown}` | Select the report format; default: `text`. |
| `--json` | Backward-compatible shortcut that selects JSON. |
| `--output PATH` | Write UTF-8 output to a file instead of standard output. |
| `-h`, `--help` | Show argparse help and exit. |

If `--json` and `--format` are both supplied, `--json` takes precedence.
Width and height thresholds must be positive integers.

## Examples

### Default text

```bash
poseidon-dataset-summary data/sample_dataset
```

### Recursive text

```bash
poseidon-dataset-summary data/sample_dataset --recursive
```

### JSON

```bash
poseidon-dataset-summary data/sample_dataset --format json
poseidon-dataset-summary data/sample_dataset --json
```

Recursive JSON uses the same report schema:

```bash
poseidon-dataset-summary data/sample_dataset --recursive --format json
```

### CSV

```bash
poseidon-dataset-summary data/sample_dataset --format csv
```

### Markdown

```bash
poseidon-dataset-summary data/sample_dataset --format markdown
```

## Output files

`--output` works with every formatter. The parent directory must already
exist.

Windows PowerShell:

```powershell
poseidon-dataset-summary .\data\sample_dataset --format json --output .\dataset-report.json
```

macOS/Linux:

```bash
poseidon-dataset-summary ./data/sample_dataset --format markdown --output ./dataset-report.md
```

When `--output` is present, the report is written instead of printed.
It can be combined with recursive scanning:

```bash
poseidon-dataset-summary data/sample_dataset --recursive --format json --output dataset-report.json
```

## Recursive scanning

Without `--recursive`, only supported images directly inside `DATASET_PATH`
are analyzed. With `--recursive`, supported images in that directory and all
nested directories contribute to the same report.

Nested valid images contribute to counts, dimensions, format statistics,
decoded channel statistics, pixel-area and megapixel statistics, and dataset
size. Nested corrupt or undersized supported images contribute to invalid
counts, format statistics, and invalid-image diagnostics, including their
portable paths and validation errors. Unsupported nested files remain ignored,
and empty nested directories are not errors.

The equivalent module invocation accepts the same option:

```bash
python -m poseidon_ai.nautilus_vision.dataset_summary data/sample_dataset --recursive
```

## Validation thresholds

The minimum valid width and height both default to 32 pixels. The thresholds
are independent positive integers: either option can be supplied without the
other.

Lower thresholds can allow smaller decodable images:

```bash
poseidon-dataset-summary data/sample_dataset --min-width 10 --min-height 10
```

Higher thresholds can classify an otherwise valid image as invalid:

```bash
poseidon-dataset-summary data/sample_dataset --min-width 100 --min-height 40
```

Thresholds work with recursive scanning and structured output:

```bash
poseidon-dataset-summary data/sample_dataset --recursive --min-width 64 --min-height 64 --format json
```

Images that fail custom thresholds remain successful dataset-analysis
results. Their width and height errors appear in invalid-image diagnostics;
they do not contribute to channel or resolution statistics and are not
operational CLI failures. Zero, negative, or non-integer option values are
argparse usage errors and return status 2.

Threshold values affect validation only. They are not included in the text,
JSON, CSV, or Markdown report schemas.

## Text output

Text output includes:

1. dataset path and image counts;
2. alphabetically ordered, uppercase image-format counts;
3. numerically ordered decoded channel counts for valid images;
4. valid-image minimum, maximum, and average pixel-area and megapixel values;
5. invalid-image paths with every validation error;
6. valid-image width statistics;
7. valid-image height statistics;
8. valid-image dataset size in human-readable units.

An empty candidate set displays `No supported image files found.` and zero
dimension and size values. When there are no diagnostics, the diagnostics
section displays `No invalid images found.` When there are no valid images,
the Image Channels section displays
`No valid image channel data found.` and the Image Resolution section displays
`No valid image resolution data found.` Human-readable pixel values use
thousands separators, average pixels use two decimal places, and megapixels
use two decimal places.

```text
Image Resolution
----------------
Minimum Pixels    : 307,200
Maximum Pixels    : 2,073,600
Average Pixels    : 1,190,400.00
Minimum MP        : 0.31
Maximum MP        : 2.07
Average MP        : 1.19
```

## JSON schema

```json
{
  "dataset_path": "data/sample_dataset",
  "total_images": 3,
  "valid_images": 2,
  "invalid_images": 1,
  "extension_counts": {
    "jpeg": 2,
    "png": 1
  },
  "channel_counts": {
    "3": 2
  },
  "resolution_statistics": {
    "minimum_pixels": 307200,
    "maximum_pixels": 2073600,
    "average_pixels": 1190400.0,
    "minimum_megapixels": 0.3072,
    "maximum_megapixels": 2.0736,
    "average_megapixels": 1.1904
  },
  "invalid_image_diagnostics": [
    {
      "image_path": "data/a-corrupt.jpg",
      "errors": [
        "Image could not be decoded."
      ]
    }
  ],
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

Numbers remain JSON numbers. `extension_counts` is an alphabetically ordered
object with normalized lowercase keys. It counts supported candidates,
including invalid ones. `channel_counts` is numerically ordered before its
integer model keys are explicitly converted to textual JSON object keys; its
values remain JSON numbers. It counts only valid images and is `{}` when no
valid images exist. Width, height, and size fields also describe valid images.
`invalid_image_diagnostics` is a path-sorted array. Each entry has a portable
path string and an `errors` array in original validator order. An analysis
with no invalid images uses an empty array.

Channel values are the numeric decoded channel counts produced by the current
metadata pipeline. They do not infer colour-mode semantics. Recursive valid
images contribute when enabled, while validation thresholds can change
whether an image contributes.

`resolution_statistics` contains `minimum_pixels`, `maximum_pixels`,
`average_pixels`, `minimum_megapixels`, `maximum_megapixels`, and
`average_megapixels`. Pixel counts use each valid image's actual decoded
`width * height` area. Megapixels use decimal conversion
`pixel_count / 1_000_000` and are rounded to six decimal places in structured
reports. All values remain JSON numbers. When no valid images exist, all six
values are zero, with average and megapixel values represented as `0.0`.
Recursive valid images contribute when enabled, and validation thresholds can
change which images contribute.

## CSV schema

The CSV report contains a header and one data row with these fifteen stable
columns:

| Position | Column |
|---------:|--------|
| 1 | `dataset_path` |
| 2 | `total_images` |
| 3 | `valid_images` |
| 4 | `invalid_images` |
| 5 | `extension_counts` |
| 6 | `channel_counts` |
| 7 | `resolution_statistics` |
| 8 | `invalid_image_diagnostics` |
| 9 | `min_width` |
| 10 | `max_width` |
| 11 | `average_width` |
| 12 | `min_height` |
| 13 | `max_height` |
| 14 | `average_height` |
| 15 | `total_size_bytes` |

`extension_counts`, `channel_counts`, and `resolution_statistics` are JSON
objects stored in separate CSV fields:

```csv
data/sample_dataset,3,2,1,"{""jpeg"": 2, ""png"": 1}","{""3"": 2}","{""minimum_pixels"": 307200, ""maximum_pixels"": 2073600, ""average_pixels"": 1190400.0, ""minimum_megapixels"": 0.3072, ""maximum_megapixels"": 2.0736, ""average_megapixels"": 1.1904}","[{""image_path"": ""data/a-corrupt.jpg"", ""errors"": [""Image could not be decoded.""]}]",640,1280,960.00,480,720,600.00,2048
```

CSV quoting is produced by Python's standard-library `csv.writer`. Consumers
should parse the document with a CSV parser, then parse column 5 as a JSON
object, column 6 as the channel-count JSON object, column 7 as the
resolution-statistics JSON object, and column 8 as a JSON array. Empty channel
statistics are `{}`; resolution statistics retain the same zero-valued
six-field object as JSON. The complete diagnostic collection stays in its one
cell; an empty collection is `[]`. Average dimensions are rendered with two
decimal places.

## Markdown output

Markdown output contains eight second-level sections: Overview, Image Formats,
Image Channels, Image Resolution, Invalid Image Diagnostics, Width, Height,
and Dataset Size. Image formats are alphabetically ordered and uppercased.
Image Channels is a numerically ordered two-column table of decoded channel
counts and valid-image totals. Image Resolution is a three-column table with
minimum, maximum, and average rows; raw pixels use thousands separators and
megapixels use two decimal places. When no valid images exist, both sections
use their explicit human-readable empty-state sentences. Diagnostics are
path-sorted, use a third-level code-formatted path heading, and list every
error as a bullet. If no supported images or diagnostics are found, their
sections contain the same empty-state sentences as text output.

```markdown
## Image Resolution

| Metric | Pixels | Megapixels |
|--------|-------:|-----------:|
| Minimum | 307,200 | 0.31 |
| Maximum | 2,073,600 | 2.07 |
| Average | 1,190,400.00 | 1.19 |
```

The Markdown formatter renders the dataset path with `/` separators so saved
reports are portable across operating systems.

## Exit behavior

- Successful terminal output and file writes return status 0.
- `--help` returns status 0.
- Expected dataset and output filesystem failures return status 1. These
  failures are written to standard error as concise `Error:` messages without
  a traceback; standard output remains empty.
- Argparse rejects missing arguments or unsupported `--format` values and
  returns status 2.

Status 1 covers missing dataset paths, files supplied as dataset paths,
dataset directories that cannot be read, missing output directories,
directories supplied as output paths, and output permission or write errors.
The CLI does not create missing output directories.

A valid empty dataset remains successful and retains the existing empty-state
report. Corrupt or undersized supported images also remain successful
analysis results: they contribute to invalid counts and diagnostics rather
than becoming CLI errors.

## Current limitations

- The parent of an `--output` path is not created automatically.

## Troubleshooting

### `poseidon-dataset-summary` command not found

Activate the intended virtual environment and install the repository:

```bash
python -m pip install -e ".[dev]"
```

Confirm the same virtual environment remains active when running the command.

### `No module named poseidon_ai`

Activate the intended virtual environment and install the repository:

```bash
python -m pip install -e ".[dev]"
```

### Dataset path errors

The CLI reports a missing path or a file supplied as the dataset on standard
error and returns status 1. Confirm the argument exists, is a directory, and
can be read by the current process.

### No images are reported

By default, the CLI scans only the top level. If supported images exist only
inside nested directories, add `--recursive`. Confirm files use a supported
suffix: `.bmp`, `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, or `.webp`.

### Images are counted as invalid

A supported file must be decodable by OpenCV and meet the configured minimum
width and height, which both default to 32 pixels. Each report includes the
invalid path and every captured validation error. Adjust `--min-width` and
`--min-height` when smaller images are intentionally acceptable.

### Output-file errors

The CLI reports missing output directories, directory output paths, and
permission or other expected write failures on standard error and returns
status 1. Create the parent directory before using `--output`, supply a file
path, and confirm the process can write to that location.
