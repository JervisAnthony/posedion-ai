"""Pure in-memory summaries for configured YOLO split analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poseidon_ai.nautilus_vision.yolo_class_validation import (
    YoloConfiguredClassUsage,
)
from poseidon_ai.nautilus_vision.yolo_config import (
    YoloClassDefinition,
    YoloDatasetConfiguration,
)
from poseidon_ai.nautilus_vision.yolo_split_analysis import (
    YoloConfiguredSplitAnalysisResult,
    YoloDatasetSplitAnalysis,
    YoloDatasetSplitAnalysisFailure,
)
from poseidon_ai.nautilus_vision.yolo_split_plan import (
    YoloDatasetSplit,
)


@dataclass(frozen=True, slots=True)
class YoloSplitSummary:
    """Flattened counts and class diagnostics for one successful split."""

    split: YoloDatasetSplit
    total_images: int
    total_label_files: int
    paired_images: int
    missing_label_images: int
    orphan_label_files: int
    pairing_conflicts: int
    valid_label_files: int
    invalid_label_files: int
    empty_label_files: int
    total_annotations: int
    configured_annotation_count: int
    unknown_class_occurrences: int
    unknown_class_occurrences_in_valid_labels: int
    unknown_class_occurrences_in_invalid_labels: int
    class_validation_is_valid: bool
    class_usage: tuple[YoloConfiguredClassUsage, ...]
    unobserved_classes: tuple[YoloClassDefinition, ...]


@dataclass(frozen=True, slots=True)
class YoloCrossSplitClassUsage:
    """Configured annotation usage aggregated across successful splits."""

    class_id: int
    name: str
    annotation_count: int
    observed_split_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class YoloCrossSplitSummary:
    """Immutable ordered summary of configured split-analysis outcomes."""

    config_path: Path
    recursive: bool
    outcomes: tuple[
        YoloSplitSummary | YoloDatasetSplitAnalysisFailure,
        ...,
    ]
    total_images: int
    total_label_files: int
    total_paired_images: int
    total_missing_label_images: int
    total_orphan_label_files: int
    total_pairing_conflicts: int
    total_valid_label_files: int
    total_invalid_label_files: int
    total_empty_label_files: int
    total_annotations: int
    configured_annotation_count: int
    unknown_class_occurrences: int
    unknown_class_occurrences_in_valid_labels: int
    unknown_class_occurrences_in_invalid_labels: int
    class_usage: tuple[YoloCrossSplitClassUsage, ...]
    unobserved_classes: tuple[YoloClassDefinition, ...]

    @property
    def is_complete(self) -> bool:
        """Return whether every planned split was analyzed successfully."""

        return not any(
            isinstance(outcome, YoloDatasetSplitAnalysisFailure)
            for outcome in self.outcomes
        )

    @property
    def successful_summaries(self) -> tuple[YoloSplitSummary, ...]:
        """Return successful summaries in original outcome order."""

        return tuple(
            outcome
            for outcome in self.outcomes
            if isinstance(outcome, YoloSplitSummary)
        )

    @property
    def failed_splits(
        self,
    ) -> tuple[YoloDatasetSplitAnalysisFailure, ...]:
        """Return original expected failures in outcome order."""

        return tuple(
            outcome
            for outcome in self.outcomes
            if isinstance(outcome, YoloDatasetSplitAnalysisFailure)
        )

    @property
    def successful_split_count(self) -> int:
        """Return the number of successful split summaries."""

        return len(self.successful_summaries)

    @property
    def failed_split_count(self) -> int:
        """Return the number of preserved expected split failures."""

        return len(self.failed_splits)


def _summarize_success(
    outcome: YoloDatasetSplitAnalysis,
) -> YoloSplitSummary:
    """Flatten one existing successful split-analysis outcome."""

    dataset_analysis = outcome.dataset_analysis
    class_validation = outcome.class_validation
    unknown_occurrences = class_validation.unknown_class_occurrences

    return YoloSplitSummary(
        split=outcome.split,
        total_images=dataset_analysis.total_images,
        total_label_files=dataset_analysis.total_label_files,
        paired_images=len(dataset_analysis.pairs),
        missing_label_images=len(dataset_analysis.missing_label_images),
        orphan_label_files=len(dataset_analysis.orphan_label_files),
        pairing_conflicts=len(dataset_analysis.pairing_conflicts),
        valid_label_files=dataset_analysis.valid_label_files,
        invalid_label_files=dataset_analysis.invalid_label_files,
        empty_label_files=dataset_analysis.empty_label_files,
        total_annotations=dataset_analysis.total_annotations,
        configured_annotation_count=sum(
            usage.annotation_count for usage in class_validation.class_usage
        ),
        unknown_class_occurrences=len(unknown_occurrences),
        unknown_class_occurrences_in_valid_labels=sum(
            occurrence.label_is_valid for occurrence in unknown_occurrences
        ),
        unknown_class_occurrences_in_invalid_labels=sum(
            not occurrence.label_is_valid
            for occurrence in unknown_occurrences
        ),
        class_validation_is_valid=class_validation.is_valid,
        class_usage=class_validation.class_usage,
        unobserved_classes=class_validation.unobserved_classes,
    )


def summarize_yolo_dataset_splits(
    configuration: YoloDatasetConfiguration,
    split_analysis: YoloConfiguredSplitAnalysisResult,
) -> YoloCrossSplitSummary:
    """Aggregate completed configured split outcomes without I/O."""

    if configuration.config_path != split_analysis.config_path:
        raise ValueError(
            "Split analysis config_path does not match the dataset "
            "configuration config_path."
        )

    outcomes: list[
        YoloSplitSummary | YoloDatasetSplitAnalysisFailure
    ] = []
    successful_summaries: list[YoloSplitSummary] = []
    for outcome in split_analysis.outcomes:
        if isinstance(outcome, YoloDatasetSplitAnalysisFailure):
            outcomes.append(outcome)
            continue

        summary = _summarize_success(outcome)
        outcomes.append(summary)
        successful_summaries.append(summary)

    aggregate_counts = {
        definition.class_id: 0 for definition in configuration.classes
    }
    observed_names = {
        definition.class_id: [] for definition in configuration.classes
    }
    for summary in successful_summaries:
        for usage in summary.class_usage:
            aggregate_counts[usage.class_id] += usage.annotation_count
            names = observed_names[usage.class_id]
            if (
                usage.annotation_count > 0
                and summary.split.name not in names
            ):
                names.append(summary.split.name)

    class_usage = tuple(
        YoloCrossSplitClassUsage(
            class_id=definition.class_id,
            name=definition.name,
            annotation_count=aggregate_counts[definition.class_id],
            observed_split_names=tuple(
                observed_names[definition.class_id]
            ),
        )
        for definition in configuration.classes
    )
    unobserved_classes = tuple(
        definition
        for definition in configuration.classes
        if aggregate_counts[definition.class_id] == 0
    )

    return YoloCrossSplitSummary(
        config_path=split_analysis.config_path,
        recursive=split_analysis.recursive,
        outcomes=tuple(outcomes),
        total_images=sum(
            summary.total_images for summary in successful_summaries
        ),
        total_label_files=sum(
            summary.total_label_files for summary in successful_summaries
        ),
        total_paired_images=sum(
            summary.paired_images for summary in successful_summaries
        ),
        total_missing_label_images=sum(
            summary.missing_label_images
            for summary in successful_summaries
        ),
        total_orphan_label_files=sum(
            summary.orphan_label_files
            for summary in successful_summaries
        ),
        total_pairing_conflicts=sum(
            summary.pairing_conflicts
            for summary in successful_summaries
        ),
        total_valid_label_files=sum(
            summary.valid_label_files
            for summary in successful_summaries
        ),
        total_invalid_label_files=sum(
            summary.invalid_label_files
            for summary in successful_summaries
        ),
        total_empty_label_files=sum(
            summary.empty_label_files
            for summary in successful_summaries
        ),
        total_annotations=sum(
            summary.total_annotations for summary in successful_summaries
        ),
        configured_annotation_count=sum(
            summary.configured_annotation_count
            for summary in successful_summaries
        ),
        unknown_class_occurrences=sum(
            summary.unknown_class_occurrences
            for summary in successful_summaries
        ),
        unknown_class_occurrences_in_valid_labels=sum(
            summary.unknown_class_occurrences_in_valid_labels
            for summary in successful_summaries
        ),
        unknown_class_occurrences_in_invalid_labels=sum(
            summary.unknown_class_occurrences_in_invalid_labels
            for summary in successful_summaries
        ),
        class_usage=class_usage,
        unobserved_classes=unobserved_classes,
    )
