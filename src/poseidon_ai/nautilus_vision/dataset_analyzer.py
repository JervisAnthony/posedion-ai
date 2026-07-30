"""Dataset analysis utilities for Nautilus Vision."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_loader import load_image_dataset
from poseidon_ai.nautilus_vision.dataset_manifest import (
    DatasetAnalysisResult,
    DatasetManifestEntry,
)
from poseidon_ai.nautilus_vision.dataset_statistics import (
    DatasetStatistics,
    DuplicateImageGroup,
    InvalidImageDiagnostic,
)
from poseidon_ai.nautilus_vision.image_hash import calculate_sha256
from poseidon_ai.nautilus_vision.image_metadata import get_image_metadata
from poseidon_ai.nautilus_vision.image_validator import (
    DEFAULT_MIN_HEIGHT,
    DEFAULT_MIN_WIDTH,
    validate_image,
)


def analyze_dataset(
    dataset_path: str | Path,
    *,
    recursive: bool = False,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> DatasetStatistics:
    """Analyze an image dataset and compute summary statistics.

    Parameters
    ----------
    dataset_path:
        Directory containing the image dataset.
    recursive:
        Search nested directories when True.
    min_width:
        Minimum valid image width in pixels.
    min_height:
        Minimum valid image height in pixels.

    Returns
    -------
    DatasetStatistics
        Summary statistics for valid and invalid images.
    """
    return _analyze_dataset(
        dataset_path,
        recursive=recursive,
        min_width=min_width,
        min_height=min_height,
        collect_manifest=False,
    ).statistics


def analyze_dataset_with_manifest(
    dataset_path: str | Path,
    *,
    recursive: bool = False,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> DatasetAnalysisResult:
    """Analyze a dataset and collect its supported-candidate manifest."""

    return _analyze_dataset(
        dataset_path,
        recursive=recursive,
        min_width=min_width,
        min_height=min_height,
        collect_manifest=True,
    )


def _analyze_dataset(
    dataset_path: str | Path,
    *,
    recursive: bool,
    min_width: int,
    min_height: int,
    collect_manifest: bool,
) -> DatasetAnalysisResult:
    """Run the shared dataset analysis pass."""

    dataset_directory = Path(dataset_path)

    image_paths = load_image_dataset(
        dataset_directory,
        recursive=recursive,
        validate=False,
    )

    stats = DatasetStatistics(dataset_path=dataset_directory)

    widths: list[int] = []
    heights: list[int] = []
    pixel_counts: list[int] = []
    aspect_ratios: list[float] = []
    file_sizes: list[int] = []
    orientation_counts = {
        "landscape": 0,
        "portrait": 0,
        "square": 0,
    }
    size_to_paths: dict[int, list[Path]] = {}
    manifest_entries: list[DatasetManifestEntry] | None = (
        [] if collect_manifest else None
    )

    for image_path in image_paths:
        stats.total_images += 1

        extension = image_path.suffix.lower().removeprefix(".")
        extension = {
            "jpg": "jpeg",
            "tif": "tiff",
        }.get(extension, extension)
        stats.extension_counts[extension] = (
            stats.extension_counts.get(extension, 0) + 1
        )

        validation_result = validate_image(
            image_path,
            min_width=min_width,
            min_height=min_height,
        )

        if not validation_result.is_valid:
            stats.invalid_images += 1

            stats.invalid_image_diagnostics.append(
                InvalidImageDiagnostic(
                    image_path=image_path,
                    errors=validation_result.errors,
                )
            )

            if manifest_entries is not None:
                manifest_entries.append(
                    DatasetManifestEntry(
                        path=image_path.relative_to(dataset_directory),
                        extension=extension,
                        is_valid=False,
                        validation_errors=validation_result.errors,
                    )
                )

            continue

        metadata = get_image_metadata(image_path)

        stats.valid_images += 1
        file_size_bytes = metadata["size_bytes"]
        stats.total_size_bytes += file_size_bytes
        file_sizes.append(file_size_bytes)
        size_to_paths.setdefault(
            file_size_bytes,
            [],
        ).append(image_path)

        channels = metadata["channels"]
        stats.channel_counts[channels] = (
            stats.channel_counts.get(channels, 0) + 1
        )

        width = metadata["width"]
        height = metadata["height"]
        pixel_count = width * height
        widths.append(width)
        heights.append(height)
        pixel_counts.append(pixel_count)
        aspect_ratios.append(width / height)

        if width > height:
            orientation_counts["landscape"] += 1
        elif width < height:
            orientation_counts["portrait"] += 1
        else:
            orientation_counts["square"] += 1

        if manifest_entries is not None:
            manifest_entries.append(
                DatasetManifestEntry(
                    path=image_path.relative_to(dataset_directory),
                    extension=extension,
                    is_valid=True,
                    validation_errors=(),
                    width=width,
                    height=height,
                    channels=channels,
                    size_bytes=file_size_bytes,
                    pixel_count=pixel_count,
                    megapixels=round(pixel_count / 1_000_000, 6),
                )
            )

    if widths:
        stats.min_width = min(widths)
        stats.max_width = max(widths)
        stats.average_width = sum(widths) / len(widths)

    if heights:
        stats.min_height = min(heights)
        stats.max_height = max(heights)
        stats.average_height = sum(heights) / len(heights)

    if pixel_counts:
        stats.min_pixel_count = min(pixel_counts)
        stats.max_pixel_count = max(pixel_counts)
        stats.average_pixel_count = (
            sum(pixel_counts) / len(pixel_counts)
        )

    if aspect_ratios:
        stats.min_aspect_ratio = min(aspect_ratios)
        stats.max_aspect_ratio = max(aspect_ratios)
        stats.average_aspect_ratio = (
            sum(aspect_ratios) / len(aspect_ratios)
        )

    if file_sizes:
        stats.min_file_size_bytes = min(file_sizes)
        stats.max_file_size_bytes = max(file_sizes)
        stats.average_file_size_bytes = (
            sum(file_sizes) / len(file_sizes)
        )

    stats.orientation_counts = orientation_counts

    duplicate_groups: list[DuplicateImageGroup] = []
    for candidate_paths in size_to_paths.values():
        if len(candidate_paths) < 2:
            continue

        digest_to_paths: dict[str, list[Path]] = {}
        for image_path in candidate_paths:
            digest = calculate_sha256(image_path)
            digest_to_paths.setdefault(digest, []).append(image_path)

        for digest, matching_paths in digest_to_paths.items():
            if len(matching_paths) < 2:
                continue

            sorted_paths = tuple(
                sorted(
                    matching_paths,
                    key=lambda path: (
                        path.as_posix().casefold(),
                        path.as_posix(),
                    ),
                )
            )
            duplicate_groups.append(
                DuplicateImageGroup(
                    sha256=digest,
                    image_paths=sorted_paths,
                )
            )

    stats.duplicate_image_groups = sorted(
        duplicate_groups,
        key=lambda group: (
            group.image_paths[0].as_posix().casefold(),
            group.image_paths[0].as_posix(),
            group.sha256,
        ),
    )
    stats.extension_counts = dict(sorted(stats.extension_counts.items()))
    stats.channel_counts = dict(sorted(stats.channel_counts.items()))

    if manifest_entries is None:
        completed_entries: tuple[DatasetManifestEntry, ...] = ()
    else:
        duplicate_digests = {
            image_path.relative_to(dataset_directory): group.sha256
            for group in stats.duplicate_image_groups
            for image_path in group.image_paths
        }
        completed_entries = tuple(
            sorted(
                (
                    replace(
                        entry,
                        duplicate_group_sha256=duplicate_digests.get(
                            entry.path
                        ),
                    )
                    for entry in manifest_entries
                ),
                key=lambda entry: (
                    entry.path.as_posix().casefold(),
                    entry.path.as_posix(),
                ),
            )
        )

    return DatasetAnalysisResult(
        statistics=stats,
        manifest_entries=completed_entries,
    )
