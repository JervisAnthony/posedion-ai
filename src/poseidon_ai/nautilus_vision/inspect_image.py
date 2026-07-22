"""Command-line image inspection for Nautilus Vision."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from poseidon_ai.logging_config import configure_logging
from poseidon_ai.nautilus_vision.image_metadata import get_image_metadata

logger = logging.getLogger(__name__)


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

    configure_logging()

    parser = build_parser()
    args = parser.parse_args()

    logger.info("Inspecting image: %s", args.image_path)

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