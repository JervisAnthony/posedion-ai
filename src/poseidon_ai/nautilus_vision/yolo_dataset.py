"""Library-level analysis for paired YOLO image and label datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poseidon_ai.nautilus_vision.dataset_loader import load_image_dataset
from poseidon_ai.nautilus_vision.yolo_label import (
    YoloLabelValidationResult,
    validate_yolo_label,
)


@dataclass(frozen=True, slots=True)
class YoloImageLabelPair:
    """One uniquely paired image and validated YOLO label."""

    pairing_key: str
    image_path: Path
    label_path: Path
    label_validation: YoloLabelValidationResult


@dataclass(frozen=True, slots=True)
class YoloPairingConflict:
    """All image and label paths sharing one ambiguous pairing key."""

    pairing_key: str
    image_paths: tuple[Path, ...]
    label_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class YoloClassCount:
    """The annotation count for one numeric YOLO class identifier."""

    class_id: int
    annotation_count: int


@dataclass(frozen=True, slots=True)
class YoloDatasetAnalysisResult:
    """Immutable results from analyzing one image and label directory pair."""

    image_directory: Path
    label_directory: Path
    recursive: bool
    pairs: tuple[YoloImageLabelPair, ...]
    missing_label_images: tuple[Path, ...]
    orphan_label_files: tuple[Path, ...]
    pairing_conflicts: tuple[YoloPairingConflict, ...]
    total_images: int
    total_label_files: int
    valid_label_files: int
    invalid_label_files: int
    empty_label_files: int
    total_annotations: int
    class_counts: tuple[YoloClassCount, ...]


def _relative_posix(path: Path, root: Path) -> str:
    """Return a deterministic POSIX-style path relative to one root."""

    return path.relative_to(root).as_posix()


def _pairing_key(path: Path, root: Path) -> str:
    """Return the case-sensitive relative path without its final suffix."""

    return path.relative_to(root).with_suffix("").as_posix()


def _discover_label_files(
    label_root: Path,
    *,
    recursive: bool,
) -> list[Path]:
    """Discover regular TXT label files without parsing their contents."""

    candidates = (
        label_root.rglob("*")
        if recursive
        else label_root.iterdir()
    )
    label_paths = [
        candidate
        for candidate in candidates
        if candidate.is_file()
        and candidate.suffix.lower() == ".txt"
    ]
    return sorted(
        label_paths,
        key=lambda path: _relative_posix(path, label_root),
    )


def analyze_yolo_dataset(
    image_directory: str | Path,
    label_directory: str | Path,
    *,
    recursive: bool = False,
) -> YoloDatasetAnalysisResult:
    """Pair and analyze supported images and YOLO detection labels."""

    image_root = Path(image_directory)
    label_root = Path(label_directory)

    if not image_root.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_root}"
        )
    if not image_root.is_dir():
        raise NotADirectoryError(
            f"Image path is not a directory: {image_root}"
        )
    if not label_root.exists():
        raise FileNotFoundError(
            f"Label directory does not exist: {label_root}"
        )
    if not label_root.is_dir():
        raise NotADirectoryError(
            f"Label path is not a directory: {label_root}"
        )

    image_paths = sorted(
        load_image_dataset(
            image_root,
            recursive=recursive,
            validate=False,
        ),
        key=lambda path: _relative_posix(path, image_root),
    )
    label_paths = _discover_label_files(
        label_root,
        recursive=recursive,
    )

    images_by_key: dict[str, list[Path]] = {}
    for image_path in image_paths:
        images_by_key.setdefault(
            _pairing_key(image_path, image_root),
            [],
        ).append(image_path)

    labels_by_key: dict[str, list[Path]] = {}
    for label_path in label_paths:
        labels_by_key.setdefault(
            _pairing_key(label_path, label_root),
            [],
        ).append(label_path)

    pairs: list[YoloImageLabelPair] = []
    missing_label_images: list[Path] = []
    orphan_label_files: list[Path] = []
    pairing_conflicts: list[YoloPairingConflict] = []
    valid_label_files = 0
    invalid_label_files = 0
    empty_label_files = 0
    total_annotations = 0
    class_count_values: dict[int, int] = {}

    for pairing_key in sorted(images_by_key.keys() | labels_by_key.keys()):
        matching_images = sorted(
            images_by_key.get(pairing_key, []),
            key=lambda path: _relative_posix(path, image_root),
        )
        matching_labels = sorted(
            labels_by_key.get(pairing_key, []),
            key=lambda path: _relative_posix(path, label_root),
        )

        if len(matching_images) > 1 or len(matching_labels) > 1:
            pairing_conflicts.append(
                YoloPairingConflict(
                    pairing_key=pairing_key,
                    image_paths=tuple(matching_images),
                    label_paths=tuple(matching_labels),
                )
            )
            continue

        if matching_images and matching_labels:
            validation = validate_yolo_label(matching_labels[0])
            pairs.append(
                YoloImageLabelPair(
                    pairing_key=pairing_key,
                    image_path=matching_images[0],
                    label_path=matching_labels[0],
                    label_validation=validation,
                )
            )

            if validation.is_valid:
                valid_label_files += 1
                if not validation.annotations:
                    empty_label_files += 1
                for annotation in validation.annotations:
                    total_annotations += 1
                    class_count_values[annotation.class_id] = (
                        class_count_values.get(annotation.class_id, 0)
                        + 1
                    )
            else:
                invalid_label_files += 1
            continue

        if matching_images:
            missing_label_images.append(matching_images[0])
        else:
            orphan_label_files.append(matching_labels[0])

    class_counts = tuple(
        YoloClassCount(
            class_id=class_id,
            annotation_count=annotation_count,
        )
        for class_id, annotation_count in sorted(
            class_count_values.items()
        )
    )

    return YoloDatasetAnalysisResult(
        image_directory=image_root,
        label_directory=label_root,
        recursive=recursive,
        pairs=tuple(pairs),
        missing_label_images=tuple(
            sorted(
                missing_label_images,
                key=lambda path: _relative_posix(path, image_root),
            )
        ),
        orphan_label_files=tuple(
            sorted(
                orphan_label_files,
                key=lambda path: _relative_posix(path, label_root),
            )
        ),
        pairing_conflicts=tuple(pairing_conflicts),
        total_images=len(image_paths),
        total_label_files=len(label_paths),
        valid_label_files=valid_label_files,
        invalid_label_files=invalid_label_files,
        empty_label_files=empty_label_files,
        total_annotations=total_annotations,
        class_counts=class_counts,
    )
