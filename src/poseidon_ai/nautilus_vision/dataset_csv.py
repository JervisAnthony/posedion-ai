import csv
import io
import json
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_serialization import (
    serialize_channel_counts,
    serialize_invalid_image_diagnostics,
    serialize_resolution_statistics,
)
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
)


def format_dataset_csv(
    dataset_path: Path,
    stats: DatasetStatistics,
) -> str:
    """Return dataset statistics as CSV."""

    output = io.StringIO(newline="")
    writer = csv.writer(
        output,
        lineterminator="\n",
    )
    writer.writerow(
        [
            "dataset_path",
            "total_images",
            "valid_images",
            "invalid_images",
            "extension_counts",
            "channel_counts",
            "resolution_statistics",
            "invalid_image_diagnostics",
            "min_width",
            "max_width",
            "average_width",
            "min_height",
            "max_height",
            "average_height",
            "total_size_bytes",
        ]
    )
    writer.writerow(
        [
            dataset_path,
            stats.total_images,
            stats.valid_images,
            stats.invalid_images,
            json.dumps(
                dict(sorted(stats.extension_counts.items()))
            ),
            json.dumps(
                serialize_channel_counts(stats.channel_counts)
            ),
            json.dumps(
                serialize_resolution_statistics(
                    stats.min_pixel_count,
                    stats.max_pixel_count,
                    stats.average_pixel_count,
                )
            ),
            json.dumps(
                serialize_invalid_image_diagnostics(
                    stats.invalid_image_diagnostics
                )
            ),
            stats.min_width,
            stats.max_width,
            f"{stats.average_width:.2f}",
            stats.min_height,
            stats.max_height,
            f"{stats.average_height:.2f}",
            stats.total_size_bytes,
        ]
    )

    return output.getvalue().rstrip("\n")
