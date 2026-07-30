"""Command-line dataset summary for Nautilus Vision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from poseidon_ai.nautilus_vision.dataset_analyzer import (
    analyze_dataset,
    analyze_dataset_with_manifest,
)
from poseidon_ai.nautilus_vision.dataset_csv import format_dataset_csv
from poseidon_ai.nautilus_vision.dataset_manifest import (
    format_dataset_manifest_jsonl,
)
from poseidon_ai.nautilus_vision.dataset_markdown import (
    format_dataset_markdown,
)
from poseidon_ai.nautilus_vision.dataset_serialization import (
    serialize_aspect_ratio_statistics,
    serialize_channel_counts,
    serialize_duplicate_images,
    serialize_file_size_statistics,
    serialize_format_statistics,
    serialize_invalid_image_diagnostics,
    serialize_resolution_statistics,
)
from poseidon_ai.nautilus_vision.dataset_statistics import DatasetStatistics
from poseidon_ai.nautilus_vision.image_validator import (
    DEFAULT_MIN_HEIGHT,
    DEFAULT_MIN_WIDTH,
)
from poseidon_ai.utils.filesize import format_file_size

DatasetFormatter = Callable[[Path, DatasetStatistics], str]


def format_dataset_summary(
    dataset_path: Path,
    stats: DatasetStatistics,
) -> str:
    """Format dataset statistics for terminal output."""

    if stats.extension_counts:
        image_formats = "\n".join(
            f"{extension.upper():<18}: {count}"
            for extension, count in sorted(
                stats.extension_counts.items()
            )
        )
    else:
        image_formats = "No supported image files found."

    if stats.format_statistics:
        image_format_sections = []
        for extension, statistics in sorted(
            stats.format_statistics.items()
        ):
            image_format_sections.append(
                f"{extension.upper()}\n"
                f"  Total Images        : {statistics.total_images}\n"
                f"  Valid Images        : {statistics.valid_images}\n"
                f"  Invalid Images      : {statistics.invalid_images}\n"
                "  Total Valid Bytes   : "
                f"{statistics.total_valid_size_bytes:,}\n"
                "  Average Valid Bytes : "
                f"{statistics.average_valid_size_bytes:,.2f}"
            )
        image_format_statistics = "\n\n".join(
            image_format_sections
        )
    else:
        image_format_statistics = (
            "No image format statistics found."
        )

    if stats.channel_counts:
        image_channel_lines = []
        for channels, count in sorted(stats.channel_counts.items()):
            unit = "channel" if channels == 1 else "channels"
            label = f"{channels} {unit}"
            image_channel_lines.append(f"{label:<19}: {count}")
        image_channels = "\n".join(image_channel_lines)
    else:
        image_channels = "No valid image channel data found."

    if stats.valid_images:
        image_resolution = (
            f"Minimum Pixels    : {stats.min_pixel_count:,}\n"
            f"Maximum Pixels    : {stats.max_pixel_count:,}\n"
            f"Average Pixels    : {stats.average_pixel_count:,.2f}\n"
            "Minimum MP        : "
            f"{stats.min_pixel_count / 1_000_000:.2f}\n"
            "Maximum MP        : "
            f"{stats.max_pixel_count / 1_000_000:.2f}\n"
            "Average MP        : "
            f"{stats.average_pixel_count / 1_000_000:.2f}"
        )
    else:
        image_resolution = "No valid image resolution data found."

    if stats.valid_images:
        image_aspect_ratios = (
            f"Minimum Ratio      : {stats.min_aspect_ratio:.2f}\n"
            f"Maximum Ratio      : {stats.max_aspect_ratio:.2f}\n"
            f"Average Ratio      : {stats.average_aspect_ratio:.2f}\n"
            "Landscape Images   : "
            f"{stats.orientation_counts.get('landscape', 0)}\n"
            "Portrait Images    : "
            f"{stats.orientation_counts.get('portrait', 0)}\n"
            "Square Images      : "
            f"{stats.orientation_counts.get('square', 0)}"
        )
    else:
        image_aspect_ratios = (
            "No valid image aspect ratio data found."
        )

    if stats.valid_images:
        image_file_sizes = (
            f"Minimum Bytes      : {stats.min_file_size_bytes:,}\n"
            f"Maximum Bytes      : {stats.max_file_size_bytes:,}\n"
            "Average Bytes      : "
            f"{stats.average_file_size_bytes:,.2f}"
        )
    else:
        image_file_sizes = "No valid image file size data found."

    duplicate_data = serialize_duplicate_images(
        stats.duplicate_image_groups
    )
    if stats.duplicate_image_groups:
        duplicate_group_sections = []
        for group in duplicate_data["groups"]:
            duplicate_group_sections.append(
                "\n".join(
                    [
                        f"SHA-256            : {group['sha256']}",
                        *[
                            f"- {image_path}"
                            for image_path in group["image_paths"]
                        ],
                    ]
                )
            )
        duplicate_images = (
            "Duplicate Groups   : "
            f"{duplicate_data['group_count']}\n"
            "Files in Groups    : "
            f"{duplicate_data['file_count']}\n"
            "Redundant Copies   : "
            f"{duplicate_data['redundant_copy_count']}\n\n"
            + "\n\n".join(duplicate_group_sections)
        )
    else:
        duplicate_images = "No exact duplicate images found."

    if stats.invalid_image_diagnostics:
        invalid_image_diagnostics = "\n\n".join(
            "\n".join(
                [
                    diagnostic.image_path.as_posix(),
                    *[
                        f"  - {error}"
                        for error in diagnostic.errors
                    ],
                ]
            )
            for diagnostic in sorted(
                stats.invalid_image_diagnostics,
                key=lambda diagnostic: (
                    diagnostic.image_path.as_posix().casefold(),
                    diagnostic.image_path.as_posix(),
                ),
            )
        )
    else:
        invalid_image_diagnostics = "No invalid images found."

    return (
        "Dataset Summary\n"
        "========================\n"
        f"Dataset Path      : {dataset_path}\n"
        f"Total Images      : {stats.total_images}\n"
        f"Valid Images      : {stats.valid_images}\n"
        f"Invalid Images    : {stats.invalid_images}\n"
        "\n"
        "Image Formats\n"
        "-------------\n"
        f"{image_formats}\n"
        "\n"
        "Image Format Statistics\n"
        "-----------------------\n"
        f"{image_format_statistics}\n"
        "\n"
        "Image Channels\n"
        "--------------\n"
        f"{image_channels}\n"
        "\n"
        "Image Resolution\n"
        "----------------\n"
        f"{image_resolution}\n"
        "\n"
        "Image Aspect Ratios\n"
        "-------------------\n"
        f"{image_aspect_ratios}\n"
        "\n"
        "Image File Sizes\n"
        "----------------\n"
        f"{image_file_sizes}\n"
        "\n"
        "Exact Duplicate Images\n"
        "----------------------\n"
        f"{duplicate_images}\n"
        "\n"
        "Invalid Image Diagnostics\n"
        "-------------------------\n"
        f"{invalid_image_diagnostics}\n"
        "\n"
        "Width\n"
        "-----\n"
        f"Minimum           : {stats.min_width}\n"
        f"Maximum           : {stats.max_width}\n"
        f"Average           : {stats.average_width:.2f}\n"
        "\n"
        "Height\n"
        "------\n"
        f"Minimum           : {stats.min_height}\n"
        f"Maximum           : {stats.max_height}\n"
        f"Average           : {stats.average_height:.2f}\n"
        "\n"
        "Dataset Size\n"
        "------------\n"
        f"Dataset Size      : {format_file_size(stats.total_size_bytes)}\n"
    )


def format_dataset_summary_json(
    dataset_path: Path,
    stats: DatasetStatistics,
) -> str:
    """Format dataset statistics as JSON."""

    payload = {
        "dataset_path": str(dataset_path),
        "total_images": stats.total_images,
        "valid_images": stats.valid_images,
        "invalid_images": stats.invalid_images,
        "extension_counts": dict(
            sorted(stats.extension_counts.items())
        ),
        "format_statistics": serialize_format_statistics(
            stats.format_statistics
        ),
        "channel_counts": serialize_channel_counts(
            stats.channel_counts
        ),
        "resolution_statistics": serialize_resolution_statistics(
            stats.min_pixel_count,
            stats.max_pixel_count,
            stats.average_pixel_count,
        ),
        "aspect_ratio_statistics": serialize_aspect_ratio_statistics(
            stats.min_aspect_ratio,
            stats.max_aspect_ratio,
            stats.average_aspect_ratio,
            stats.orientation_counts,
        ),
        "file_size_statistics": serialize_file_size_statistics(
            stats.min_file_size_bytes,
            stats.max_file_size_bytes,
            stats.average_file_size_bytes,
        ),
        "duplicate_images": serialize_duplicate_images(
            stats.duplicate_image_groups
        ),
        "invalid_image_diagnostics": (
            serialize_invalid_image_diagnostics(
                stats.invalid_image_diagnostics
            )
        ),
        "width": {
            "minimum": stats.min_width,
            "maximum": stats.max_width,
            "average": stats.average_width,
        },
        "height": {
            "minimum": stats.min_height,
            "maximum": stats.max_height,
            "average": stats.average_height,
        },
        "total_size_bytes": stats.total_size_bytes,
        "formatted_size": format_file_size(
            stats.total_size_bytes
        ),
    }

    return json.dumps(
        payload,
        indent=2,
    )


FORMATTER_REGISTRY: dict[str, DatasetFormatter] = {
    "text": format_dataset_summary,
    "json": format_dataset_summary_json,
    "csv": format_dataset_csv,
    "markdown": format_dataset_markdown,
}


def _os_error_message(error: OSError) -> str:
    """Return the useful operating-system portion of an error."""

    return error.strerror or str(error)


def _positive_pixel_count(value: str) -> int:
    """Parse a positive integer pixel count for argparse."""

    try:
        pixels = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        ) from error

    if pixels < 1:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        )

    return pixels


def _write_text_file(
    path: Path,
    content: str,
    *,
    output_name: str,
) -> str | None:
    """Write UTF-8 text or return a concise operational error."""

    output_directory = path.parent

    try:
        if path.is_dir():
            return f"Error: {output_name} path is not a file: {path}"

        if not output_directory.exists():
            return (
                f"Error: {output_name} directory does not exist: "
                f"{output_directory}"
            )

        path.write_text(
            content,
            encoding="utf-8",
        )
    except IsADirectoryError:
        return f"Error: {output_name} path is not a file: {path}"
    except FileNotFoundError:
        return (
            f"Error: {output_name} directory does not exist: "
            f"{output_directory}"
        )
    except OSError as error:
        return (
            f"Error: could not write {output_name} file "
            f"{path}: {_os_error_message(error)}"
        )

    return None


def main() -> int:
    """Run the dataset summary command."""

    parser = argparse.ArgumentParser(
        description="Display statistics for an image dataset."
    )

    parser.add_argument(
        "dataset_path",
        help="Path to the image dataset.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images in nested directories.",
    )

    parser.add_argument(
        "--min-width",
        type=_positive_pixel_count,
        default=DEFAULT_MIN_WIDTH,
        metavar="PIXELS",
        help=(
            "Minimum valid image width in pixels. "
            f"Default: {DEFAULT_MIN_WIDTH}."
        ),
    )

    parser.add_argument(
        "--min-height",
        type=_positive_pixel_count,
        default=DEFAULT_MIN_HEIGHT,
        metavar="PIXELS",
        help=(
            "Minimum valid image height in pixels. "
            f"Default: {DEFAULT_MIN_HEIGHT}."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output dataset statistics as JSON.",
    )

    parser.add_argument(
        "--format",
        choices=tuple(FORMATTER_REGISTRY),
        default="text",
        help="Output format: text, JSON, CSV, or Markdown.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to a file.",
    )

    parser.add_argument(
        "--manifest-output",
        type=Path,
        metavar="PATH",
        help="Write a per-image JSONL dataset manifest to PATH.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    manifest = None

    try:
        if args.manifest_output:
            analysis_result = analyze_dataset_with_manifest(
                dataset_path,
                recursive=args.recursive,
                min_width=args.min_width,
                min_height=args.min_height,
            )
            stats = analysis_result.statistics
            manifest = format_dataset_manifest_jsonl(
                analysis_result.manifest_entries
            )
        else:
            stats = analyze_dataset(
                dataset_path,
                recursive=args.recursive,
                min_width=args.min_width,
                min_height=args.min_height,
            )
    except FileNotFoundError:
        print(
            f"Error: dataset path does not exist: {dataset_path}",
            file=sys.stderr,
        )
        return 1
    except NotADirectoryError:
        print(
            f"Error: dataset path is not a directory: {dataset_path}",
            file=sys.stderr,
        )
        return 1
    except OSError as error:
        print(
            "Error: could not read dataset path "
            f"{dataset_path}: {_os_error_message(error)}",
            file=sys.stderr,
        )
        return 1

    if args.manifest_output:
        manifest_error = _write_text_file(
            args.manifest_output,
            manifest or "",
            output_name="manifest",
        )
        if manifest_error:
            print(manifest_error, file=sys.stderr)
            return 1

    output_format = "json" if args.json else args.format

    formatter = FORMATTER_REGISTRY[output_format]
    summary = formatter(
        dataset_path,
        stats,
    )

    if args.output:
        output_error = _write_text_file(
            args.output,
            summary,
            output_name="output",
        )
        if output_error:
            print(output_error, file=sys.stderr)
            return 1
    else:
        print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
