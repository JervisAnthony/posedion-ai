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
    [--manifest-output PATH]
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
| `--manifest-output PATH` | Write a per-image JSONL dataset manifest. |
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

## JSONL dataset manifest

`--manifest-output PATH` writes one compact JSON object followed by a newline
for every supported candidate. Valid and invalid supported images are
included; unsupported files remain ignored. Records are sorted by relative
portable path using case-insensitive then case-sensitive ordering, so
absolute machine-specific paths are never exported.

```bash
poseidon-dataset-summary data/sample_dataset \
    --manifest-output dataset-manifest.jsonl
```

Every line uses this stable key order:

```text
path
extension
is_valid
validation_errors
width
height
channels
size_bytes
pixel_count
megapixels
duplicate_group_sha256
```

For example, these are two independent JSONL records:

```jsonl
{"path":"nested/coral.jpg","extension":"jpeg","is_valid":true,"validation_errors":[],"width":640,"height":480,"channels":3,"size_bytes":24576,"pixel_count":307200,"megapixels":0.3072,"duplicate_group_sha256":null}
{"path":"nested/corrupt.png","extension":"png","is_valid":false,"validation_errors":["Image could not be decoded."],"width":null,"height":null,"channels":null,"size_bytes":null,"pixel_count":null,"megapixels":null,"duplicate_group_sha256":null}
```

Paths use forward slashes relative to `DATASET_PATH`. Extensions use the
analyzer's normalized values, including `jpeg` for `.jpg` and `tiff` for
`.tif`. Valid entries contain the width, height, decoded numeric channel
count, file size, pixel count, and megapixels already collected during
analysis. Megapixels are `pixel_count / 1_000_000`, rounded to six decimal
places. Their validation-error array is empty.

Invalid entries preserve every validator message in its original order and
use `null` for all metadata and duplicate fields. A complete lowercase
`duplicate_group_sha256` appears only on valid files belonging to a completed
exact-duplicate group. Unique files and same-size non-duplicates use `null`;
the manifest does not cause unique files to be hashed.

`--recursive` includes supported nested candidates. Validation thresholds
control whether each decodable candidate receives valid metadata or invalid
errors. A dataset with no supported candidates writes a zero-byte manifest.

Manifest export is additional to the aggregate report. With no `--output`,
the selected aggregate report still prints to standard output. With both
options, the two files are written and standard output stays empty:

```bash
poseidon-dataset-summary data/sample_dataset \
    --recursive \
    --min-width 64 \
    --min-height 64 \
    --format json \
    --output dataset-summary.json \
    --manifest-output dataset-manifest.jsonl
```

Manifest export does not change the text, JSON, CSV, or Markdown aggregate
schemas. Its parent directory must already exist. Missing parents, directory
paths, permission failures, and other expected write errors return status 1
with a concise standard-error message and no traceback. The CLI does not
print the aggregate report after a manifest write failure.

Aspect-ratio and orientation statistics are aggregate-report fields only.
The eleven-key JSONL manifest schema remains unchanged because its existing
decoded `width` and `height` values allow consumers to derive per-image
ratios independently.

## Recursive scanning

Without `--recursive`, only supported images directly inside `DATASET_PATH`
are analyzed. With `--recursive`, supported images in that directory and all
nested directories contribute to the same report.

Nested valid images contribute to counts, dimensions, format statistics,
decoded channel statistics, pixel-area and megapixel statistics, aspect-ratio
and orientation statistics, and dataset size. They also participate in exact
duplicate detection. Nested corrupt or undersized supported images contribute
to invalid counts, format statistics, and invalid-image diagnostics,
including their portable paths and validation errors, but never to
valid-image statistics or duplicate groups. Unsupported nested files remain
ignored, and empty nested directories are not errors.

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
they do not contribute to channel, resolution, aspect-ratio, or orientation
statistics and are ineligible for duplicate detection. They are not
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
5. valid-image aspect-ratio statistics and orientation counts;
6. exact duplicate groups and derived counts;
7. invalid-image paths with every validation error;
8. valid-image width statistics;
9. valid-image height statistics;
10. valid-image dataset size in human-readable units.

An empty candidate set displays `No supported image files found.` and zero
dimension and size values. When there are no diagnostics, the diagnostics
section displays `No invalid images found.` When there are no valid images,
the Image Channels section displays
`No valid image channel data found.` and the Image Resolution section displays
`No valid image resolution data found.` The Image Aspect Ratios section
displays `No valid image aspect ratio data found.` Human-readable pixel and
ratio values use two decimal places where applicable.

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

Aspect ratio is calculated from the decoded dimensions as `width / height`
without rounding before aggregation. Orientation compares the same integer
dimensions directly: landscape means `width > height`, portrait means
`width < height`, and square means `width == height`. No EXIF orientation is
read or interpreted.

```text
Image Aspect Ratios
-------------------
Minimum Ratio      : 0.50
Maximum Ratio      : 2.00
Average Ratio      : 1.17
Landscape Images   : 1
Portrait Images    : 1
Square Images      : 1
```

Exact duplicate output follows aspect ratios and uses complete SHA-256
digests and portable, deterministically ordered paths:

```text
Exact Duplicate Images
----------------------
Duplicate Groups   : 1
Files in Groups    : 3
Redundant Copies   : 2

SHA-256            : <complete-digest>
- dataset/copy-a.jpg
- dataset/copy-b.jpg
- dataset/copy-c.jpg
```

When there are no duplicate groups, the section displays
`No exact duplicate images found.`

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
  "aspect_ratio_statistics": {
    "minimum": 0.5,
    "maximum": 2.0,
    "average": 1.25,
    "orientation_counts": {
      "landscape": 1,
      "portrait": 1,
      "square": 0
    }
  },
  "duplicate_images": {
    "group_count": 1,
    "file_count": 2,
    "redundant_copy_count": 1,
    "groups": [
      {
        "sha256": "<complete-digest>",
        "image_paths": [
          "data/copy-a.jpg",
          "data/copy-b.jpg"
        ]
      }
    ]
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

`aspect_ratio_statistics` contains `minimum`, `maximum`, `average`, and
`orientation_counts`. Ratios are calculated for valid images from decoded
`width / height`, aggregated without intermediate rounding, then rounded to
six decimal places for JSON and CSV. The nested orientation object always
uses the order `landscape`, `portrait`, `square`, with numeric counts. Invalid
images do not contribute. Recursive valid images contribute when enabled,
and validation thresholds can change eligibility. No EXIF orientation is
processed.

When there are no valid images, the structured value is:

```json
{
  "minimum": 0.0,
  "maximum": 0.0,
  "average": 0.0,
  "orientation_counts": {
    "landscape": 0,
    "portrait": 0,
    "square": 0
  }
}
```

`duplicate_images` identifies byte-identical valid files by complete SHA-256
content digest. `group_count` is the number of groups, `file_count` is the
number of participating files, and `redundant_copy_count` is the number that
could be removed while retaining one file per group. Every group contains
`sha256` and an `image_paths` string array. Groups and portable paths are
deterministically ordered.

Files are first bucketed by metadata file size. Unique-size valid files are
not hashed because exact duplicates must have the same size. Matching size is
only a candidate filter and never proves duplication; every same-size
candidate is hashed once. Invalid supported images never participate.
Recursive valid files participate when enabled, and validation thresholds can
change eligibility. The empty structure is:

```json
{
  "group_count": 0,
  "file_count": 0,
  "redundant_copy_count": 0,
  "groups": []
}
```

Detection is byte-exact. Visually similar, resized, recompressed, re-encoded,
cropped, or metadata-modified images are not duplicates unless their complete
file bytes match.

## CSV schema

The CSV report contains a header and one data row with these seventeen stable
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
| 8 | `aspect_ratio_statistics` |
| 9 | `duplicate_images` |
| 10 | `invalid_image_diagnostics` |
| 11 | `min_width` |
| 12 | `max_width` |
| 13 | `average_width` |
| 14 | `min_height` |
| 15 | `max_height` |
| 16 | `average_height` |
| 17 | `total_size_bytes` |

`extension_counts`, `channel_counts`, `resolution_statistics`,
`aspect_ratio_statistics`, and `duplicate_images` are JSON objects stored in
separate CSV fields:

```csv
data/sample_dataset,3,2,1,"{""jpeg"": 2, ""png"": 1}","{""3"": 2}","{""minimum_pixels"": 307200, ""maximum_pixels"": 2073600, ""average_pixels"": 1190400.0, ""minimum_megapixels"": 0.3072, ""maximum_megapixels"": 2.0736, ""average_megapixels"": 1.1904}","{""minimum"": 0.5, ""maximum"": 2.0, ""average"": 1.25, ""orientation_counts"": {""landscape"": 1, ""portrait"": 1, ""square"": 0}}","{""group_count"": 0, ""file_count"": 0, ""redundant_copy_count"": 0, ""groups"": []}","[{""image_path"": ""data/a-corrupt.jpg"", ""errors"": [""Image could not be decoded.""]}]",640,1280,960.00,480,720,600.00,2048
```

CSV quoting is produced by Python's standard-library `csv.writer`. Consumers
should parse the document with a CSV parser, then parse column 5 as a JSON
object, column 6 as the channel-count JSON object, column 7 as the
resolution-statistics JSON object, column 8 as the aspect-ratio-statistics
JSON object, column 9 as the duplicate-images JSON object, and column 10 as a
JSON array. Empty channel statistics are `{}`; resolution and aspect-ratio
statistics retain their explicit zero-valued objects, and duplicate images
use the explicit zero-count structure with an empty `groups` array. The
complete diagnostic collection stays in its one cell; an empty collection is
`[]`. Average dimensions are rendered with two decimal places.

## Markdown output

Markdown output contains ten second-level sections: Overview, Image Formats,
Image Channels, Image Resolution, Image Aspect Ratios, Exact Duplicate
Images, Invalid Image Diagnostics, Width, Height, and Dataset Size. Image
formats are alphabetically ordered and uppercased.
Image Channels is a numerically ordered two-column table of decoded channel
counts and valid-image totals. Image Resolution is a three-column table with
minimum, maximum, and average rows; raw pixels use thousands separators and
megapixels use two decimal places. Image Aspect Ratios uses a two-column table
with three two-decimal ratio rows and landscape, portrait, and square counts
in that order. When no valid images exist, these valid-image sections use
their explicit human-readable empty-state sentences. Diagnostics are
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

```markdown
## Image Aspect Ratios

| Metric | Value |
|--------|------:|
| Minimum Ratio | 0.50 |
| Maximum Ratio | 2.00 |
| Average Ratio | 1.17 |
| Landscape Images | 1 |
| Portrait Images | 1 |
| Square Images | 1 |
```

Exact duplicate output includes derived counts and one subsection per
deterministically ordered group. Digests are complete and paths use safe
inline-code delimiters:

```markdown
## Exact Duplicate Images

| Metric | Value |
|--------|------:|
| Duplicate Groups | 1 |
| Files in Groups | 2 |
| Redundant Copies | 1 |

### Duplicate Group 1

**SHA-256:** `<complete-digest>`

- `dataset/copy-a.jpg`
- `dataset/copy-b.jpg`
```

With no groups, Markdown displays `No exact duplicate images found.`

The Markdown formatter renders the dataset path with `/` separators so saved
reports are portable across operating systems.

## Exit behavior

- Successful terminal output and file writes return status 0.
- `--help` returns status 0.
- Expected dataset, report-output, and manifest-output filesystem failures
  return status 1. These
  failures are written to standard error as concise `Error:` messages without
  a traceback; standard output remains empty.
- Argparse rejects missing arguments or unsupported `--format` values and
  returns status 2.

Status 1 covers missing dataset paths, files supplied as dataset paths,
dataset directories that cannot be read, missing output directories,
directories supplied as output paths, and output permission or write errors
for either output. The CLI does not create missing output directories.

A valid empty dataset remains successful and retains the existing empty-state
report. Corrupt or undersized supported images also remain successful
analysis results: they contribute to invalid counts and diagnostics rather
than becoming CLI errors.

## Current limitations

- The parent of an `--output` or `--manifest-output` path is not created
  automatically.

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
status 1. Create the parent directory before using `--output` or
`--manifest-output`, supply a file path, and confirm the process can write to
that location.
