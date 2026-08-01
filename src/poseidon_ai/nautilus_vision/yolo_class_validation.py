"""In-memory validation of observed YOLO classes against configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poseidon_ai.nautilus_vision.yolo_config import (
    YoloClassDefinition,
    YoloDatasetConfiguration,
)
from poseidon_ai.nautilus_vision.yolo_dataset import (
    YoloDatasetAnalysisResult,
)


@dataclass(frozen=True, slots=True)
class YoloConfiguredClassUsage:
    """The valid-annotation usage for one configured YOLO class."""

    class_id: int
    name: str
    annotation_count: int


@dataclass(frozen=True, slots=True)
class YoloUnknownClassOccurrence:
    """One parsed annotation whose class is not configured."""

    class_id: int
    pairing_key: str
    label_path: Path
    line_number: int
    label_is_valid: bool


@dataclass(frozen=True, slots=True)
class YoloDatasetClassValidationResult:
    """Immutable configured-class usage and validation diagnostics."""

    is_valid: bool
    class_usage: tuple[YoloConfiguredClassUsage, ...]
    unknown_class_occurrences: tuple[
        YoloUnknownClassOccurrence,
        ...,
    ]
    unobserved_classes: tuple[YoloClassDefinition, ...]
    errors: tuple[str, ...]


def validate_yolo_dataset_classes(
    configuration: YoloDatasetConfiguration,
    dataset_analysis: YoloDatasetAnalysisResult,
) -> YoloDatasetClassValidationResult:
    """Compare retained annotations with configured classes in memory."""

    configured_by_id = {
        definition.class_id: definition
        for definition in configuration.classes
    }
    annotation_counts = {
        class_id: 0 for class_id in configured_by_id
    }
    unknown_occurrences: list[YoloUnknownClassOccurrence] = []

    for pair in dataset_analysis.pairs:
        for annotation in pair.label_validation.annotations:
            if annotation.class_id not in configured_by_id:
                unknown_occurrences.append(
                    YoloUnknownClassOccurrence(
                        class_id=annotation.class_id,
                        pairing_key=pair.pairing_key,
                        label_path=pair.label_path,
                        line_number=annotation.line_number,
                        label_is_valid=pair.label_validation.is_valid,
                    )
                )
            elif pair.label_validation.is_valid:
                annotation_counts[annotation.class_id] += 1

    unknown_occurrences.sort(
        key=lambda occurrence: (
            occurrence.pairing_key,
            occurrence.line_number,
            occurrence.class_id,
            occurrence.label_path.as_posix(),
        )
    )

    ordered_definitions = sorted(
        configuration.classes,
        key=lambda definition: definition.class_id,
    )
    class_usage = tuple(
        YoloConfiguredClassUsage(
            class_id=definition.class_id,
            name=definition.name,
            annotation_count=annotation_counts[definition.class_id],
        )
        for definition in ordered_definitions
    )
    unobserved_classes = tuple(
        definition
        for definition in ordered_definitions
        if annotation_counts[definition.class_id] == 0
    )
    occurrences = tuple(unknown_occurrences)
    errors = tuple(
        (
            f"Pair '{occurrence.pairing_key}', line "
            f"{occurrence.line_number}: class_id "
            f"{occurrence.class_id} is not defined in the "
            "dataset configuration."
        )
        for occurrence in occurrences
    )

    return YoloDatasetClassValidationResult(
        is_valid=not occurrences,
        class_usage=class_usage,
        unknown_class_occurrences=occurrences,
        unobserved_classes=unobserved_classes,
        errors=errors,
    )
