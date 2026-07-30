from pathlib import Path

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
