"""Command-line image inspection for Nautilus Vision."""

from __future__ import annotations

import argparse
from pathlib import Path

from poseidon_ai.nautilus_vision.image_metadata import get_image_metadata


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Inspect image metadata for Nautilus Vision."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the image file.",
    )
    return parser


def main() -> None:
    """Run the image metadata inspection command."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        metadata = get_image_metadata(args.image_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    print(f"Filename: {metadata['filename']}")
    print(f"Width: {metadata['width']}")
    print(f"Height: {metadata['height']}")
    print(f"Channels: {metadata['channels']}")
    print(f"Size: {metadata['size_bytes']} bytes")


if __name__ == "__main__":
    main()