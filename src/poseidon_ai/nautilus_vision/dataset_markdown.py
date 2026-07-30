from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_serialization import (
    serialize_duplicate_images,
)
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
)
from poseidon_ai.utils.filesize import format_file_size


def _format_inline_code(value: str) -> str:
    """Wrap text in a Markdown code span that tolerates backticks."""

    longest_run = 0
    current_run = 0
    for character in value:
        if character == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    delimiter = "`" * (longest_run + 1)
    padding = " " if value.startswith("`") or value.endswith("`") else ""
    return f"{delimiter}{padding}{value}{padding}{delimiter}"


def _escape_markdown_text(value: str) -> str:
    """Escape Markdown punctuation while preserving rendered error text."""

    escaped = value.replace("\\", "\\\\")
    for character in ("`", "*", "_", "{", "}", "[", "]", "<", ">", "#"):
        escaped = escaped.replace(character, f"\\{character}")

    return " ".join(escaped.splitlines())


def format_dataset_markdown(
    dataset_path: Path,
    stats: DatasetStatistics,
) -> str:
    """Return dataset statistics as Markdown."""

    if stats.extension_counts:
        image_formats = [
            "| Format | Images |",
            "|--------|-------:|",
            *[
                f"| {extension.upper()} | {count} |"
                for extension, count in sorted(
                    stats.extension_counts.items()
                )
            ],
        ]
    else:
        image_formats = ["No supported image files found."]

    if stats.channel_counts:
        image_channels = [
            "| Channels | Images |",
            "|---------:|-------:|",
            *[
                f"| {channels} | {count} |"
                for channels, count in sorted(
                    stats.channel_counts.items()
                )
            ],
        ]
    else:
        image_channels = ["No valid image channel data found."]

    if stats.valid_images:
        image_resolution = [
            "| Metric | Pixels | Megapixels |",
            "|--------|-------:|-----------:|",
            (
                f"| Minimum | {stats.min_pixel_count:,} | "
                f"{stats.min_pixel_count / 1_000_000:.2f} |"
            ),
            (
                f"| Maximum | {stats.max_pixel_count:,} | "
                f"{stats.max_pixel_count / 1_000_000:.2f} |"
            ),
            (
                f"| Average | {stats.average_pixel_count:,.2f} | "
                f"{stats.average_pixel_count / 1_000_000:.2f} |"
            ),
        ]
    else:
        image_resolution = [
            "No valid image resolution data found."
        ]

    if stats.valid_images:
        image_aspect_ratios = [
            "| Metric | Value |",
            "|--------|------:|",
            f"| Minimum Ratio | {stats.min_aspect_ratio:.2f} |",
            f"| Maximum Ratio | {stats.max_aspect_ratio:.2f} |",
            f"| Average Ratio | {stats.average_aspect_ratio:.2f} |",
            (
                "| Landscape Images | "
                f"{stats.orientation_counts.get('landscape', 0)} |"
            ),
            (
                "| Portrait Images | "
                f"{stats.orientation_counts.get('portrait', 0)} |"
            ),
            (
                "| Square Images | "
                f"{stats.orientation_counts.get('square', 0)} |"
            ),
        ]
    else:
        image_aspect_ratios = [
            "No valid image aspect ratio data found."
        ]

    duplicate_data = serialize_duplicate_images(
        stats.duplicate_image_groups
    )
    if stats.duplicate_image_groups:
        duplicate_images = [
            "| Metric | Value |",
            "|--------|------:|",
            (
                "| Duplicate Groups | "
                f"{duplicate_data['group_count']} |"
            ),
            (
                "| Files in Groups | "
                f"{duplicate_data['file_count']} |"
            ),
            (
                "| Redundant Copies | "
                f"{duplicate_data['redundant_copy_count']} |"
            ),
        ]
        for group_number, group in enumerate(
            duplicate_data["groups"],
            start=1,
        ):
            duplicate_images.extend(
                [
                    "",
                    f"### Duplicate Group {group_number}",
                    "",
                    "**SHA-256:** "
                    + _format_inline_code(group["sha256"]),
                    "",
                    *[
                        "- " + _format_inline_code(image_path)
                        for image_path in group["image_paths"]
                    ],
                ]
            )
    else:
        duplicate_images = ["No exact duplicate images found."]

    if stats.invalid_image_diagnostics:
        invalid_image_diagnostics = []
        for diagnostic in sorted(
            stats.invalid_image_diagnostics,
            key=lambda diagnostic: (
                diagnostic.image_path.as_posix().casefold(),
                diagnostic.image_path.as_posix(),
            ),
        ):
            invalid_image_diagnostics.extend(
                [
                    "### "
                    + _format_inline_code(
                        diagnostic.image_path.as_posix()
                    ),
                    "",
                    *[
                        f"- {_escape_markdown_text(error)}"
                        for error in diagnostic.errors
                    ],
                    "",
                ]
            )
        invalid_image_diagnostics.pop()
    else:
        invalid_image_diagnostics = ["No invalid images found."]

    markdown_lines = [
        "# Dataset Summary",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Dataset Path | {dataset_path.as_posix()} |",
        f"| Total Images | {stats.total_images} |",
        f"| Valid Images | {stats.valid_images} |",
        f"| Invalid Images | {stats.invalid_images} |",
        "",
        "## Image Formats",
        "",
        *image_formats,
        "",
        "## Image Channels",
        "",
        *image_channels,
        "",
        "## Image Resolution",
        "",
        *image_resolution,
        "",
        "## Image Aspect Ratios",
        "",
        *image_aspect_ratios,
        "",
        "## Exact Duplicate Images",
        "",
        *duplicate_images,
        "",
        "## Invalid Image Diagnostics",
        "",
        *invalid_image_diagnostics,
        "",
        "## Width",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Minimum | {stats.min_width} |",
        f"| Maximum | {stats.max_width} |",
        f"| Average | {stats.average_width:.2f} |",
        "",
        "## Height",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Minimum | {stats.min_height} |",
        f"| Maximum | {stats.max_height} |",
        f"| Average | {stats.average_height:.2f} |",
        "",
        "## Dataset Size",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Size | {format_file_size(stats.total_size_bytes)} |",
    ]

    return "\n".join(markdown_lines)
