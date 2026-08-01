"""Ordered execution of configured YOLO dataset split plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poseidon_ai.nautilus_vision.yolo_class_validation import (
    YoloDatasetClassValidationResult,
    validate_yolo_dataset_classes,
)
from poseidon_ai.nautilus_vision.yolo_config import (
    YoloDatasetConfiguration,
)
from poseidon_ai.nautilus_vision.yolo_dataset import (
    YoloDatasetAnalysisResult,
    analyze_yolo_dataset,
)
from poseidon_ai.nautilus_vision.yolo_split_plan import (
    YoloDatasetSplit,
    YoloDatasetSplitPlan,
)


@dataclass(frozen=True, slots=True)
class YoloDatasetSplitAnalysis:
    """A successfully analyzed and class-validated planned split."""

    split: YoloDatasetSplit
    dataset_analysis: YoloDatasetAnalysisResult
    class_validation: YoloDatasetClassValidationResult


@dataclass(frozen=True, slots=True)
class YoloDatasetSplitAnalysisFailure:
    """An expected root-directory failure for one planned split."""

    split: YoloDatasetSplit
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class YoloConfiguredSplitAnalysisResult:
    """Immutable ordered outcomes from configured split execution."""

    config_path: Path
    recursive: bool
    outcomes: tuple[
        YoloDatasetSplitAnalysis | YoloDatasetSplitAnalysisFailure,
        ...,
    ]

    @property
    def is_complete(self) -> bool:
        """Return whether every planned split was operationally analyzed."""

        return not any(
            isinstance(outcome, YoloDatasetSplitAnalysisFailure)
            for outcome in self.outcomes
        )

    @property
    def successful_splits(
        self,
    ) -> tuple[YoloDatasetSplitAnalysis, ...]:
        """Return successful outcomes in plan-relative order."""

        return tuple(
            outcome
            for outcome in self.outcomes
            if isinstance(outcome, YoloDatasetSplitAnalysis)
        )

    @property
    def failed_splits(
        self,
    ) -> tuple[YoloDatasetSplitAnalysisFailure, ...]:
        """Return expected failure outcomes in plan-relative order."""

        return tuple(
            outcome
            for outcome in self.outcomes
            if isinstance(outcome, YoloDatasetSplitAnalysisFailure)
        )


def analyze_yolo_dataset_splits(
    configuration: YoloDatasetConfiguration,
    split_plan: YoloDatasetSplitPlan,
    *,
    recursive: bool = False,
) -> YoloConfiguredSplitAnalysisResult:
    """Execute every planned split through existing YOLO components."""

    if split_plan.config_path != configuration.config_path:
        raise ValueError(
            "Split plan config_path does not match the dataset "
            "configuration config_path."
        )

    outcomes: list[
        YoloDatasetSplitAnalysis | YoloDatasetSplitAnalysisFailure
    ] = []
    for split in split_plan.splits:
        try:
            dataset_analysis = analyze_yolo_dataset(
                split.image_directory,
                split.label_directory,
                recursive=recursive,
            )
        except (FileNotFoundError, NotADirectoryError) as error:
            outcomes.append(
                YoloDatasetSplitAnalysisFailure(
                    split=split,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            continue

        class_validation = validate_yolo_dataset_classes(
            configuration,
            dataset_analysis,
        )
        outcomes.append(
            YoloDatasetSplitAnalysis(
                split=split,
                dataset_analysis=dataset_analysis,
                class_validation=class_validation,
            )
        )

    return YoloConfiguredSplitAnalysisResult(
        config_path=split_plan.config_path,
        recursive=recursive,
        outcomes=tuple(outcomes),
    )
