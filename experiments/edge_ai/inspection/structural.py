
"""Structural validation for AYNAI Edge AI experiment datasets."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from experiments.edge_ai.config import (
    DEFAULT_CONFIG,
    SCENARIO_NAMES,
    ExperimentConfig,
)
from experiments.edge_ai.inspection.loader import (
    DatasetLoadError,
    DatasetLoader,
)
from experiments.edge_ai.inspection.models import (
    InspectionStatus,
    SeedDataset,
    SeedInspectionStatus,
    StructuralInspectionReport,
)


class StructuralInspector:
    """Inspect structural integrity across Edge AI experiment datasets."""

    def __init__(
        self,
        loader: DatasetLoader,
        experiment_config: ExperimentConfig = DEFAULT_CONFIG,
    ) -> None:
        self.loader = loader
        self.experiment_config = experiment_config

    def _check_interval(
        self,
        data: pd.DataFrame,
        description: str,
        check_id: str,
    ) -> str | None:
        timestamps = pd.to_datetime(data["timestamp"], errors="coerce")

        if len(timestamps) < 2:
            return None

        differences = timestamps.diff().dropna()
        expected = pd.Timedelta(
            seconds=self.experiment_config.sampling_interval_seconds
        )

        if not differences.eq(expected).all():
            return (
                f"{check_id} {description}: expected sampling interval "
                f"of {self.experiment_config.sampling_interval_seconds} seconds"
            )

        return None

    def _check_continuity(
        self,
        previous: pd.DataFrame,
        following: pd.DataFrame,
        previous_name: str,
        following_name: str,
    ) -> str | None:
        if previous.empty or following.empty:
            return None

        previous_end = pd.to_datetime(
            previous["timestamp"].iloc[-1],
            errors="coerce",
        )
        following_start = pd.to_datetime(
            following["timestamp"].iloc[0],
            errors="coerce",
        )

        expected = pd.Timedelta(
            seconds=self.experiment_config.sampling_interval_seconds
        )

        if following_start - previous_end != expected:
            return (
                f"STR-05 {previous_name} -> {following_name}: "
                "timestamps are not contiguous"
            )

        return None

    
    def _run_checks(
        self,
        dataset: SeedDataset,
    ) -> tuple[str, ...]:
        errors: list[str] = []

        # STR-01: Dataset Coverage
        observed_scenarios = set(dataset.evaluation)
        expected_scenarios = set(SCENARIO_NAMES)

        if observed_scenarios != expected_scenarios:
            errors.append(
                "STR-01 Dataset Coverage: expected exactly six "
                "configured evaluation scenarios"
            )

        # STR-02: Training Interval
        error = self._check_interval(
            dataset.training,
            "Training interval",
            "STR-02",
        )
        if error:
            errors.append(error)

        # STR-03: Calibration Interval
        error = self._check_interval(
            dataset.calibration,
            "Calibration interval",
            "STR-03",
        )
        if error:
            errors.append(error)

        # STR-04: Evaluation Interval (all six scenarios = one check)
        invalid_scenarios = []

        for scenario in SCENARIO_NAMES:
            evaluation = dataset.evaluation.get(scenario)

            if evaluation is None:
                continue

            error = self._check_interval(
                evaluation,
                f"Evaluation interval ({scenario})",
                "STR-04",
            )

            if error:
                invalid_scenarios.append(scenario)

        if invalid_scenarios:
            errors.append(
                "STR-04 Evaluation Interval: invalid sampling interval "
                f"in scenarios {', '.join(invalid_scenarios)}"
            )

        # STR-05: Split Continuity (both boundaries = one check)
        continuity_errors = []

        error = self._check_continuity(
            dataset.training,
            dataset.calibration,
            "Training",
            "Calibration",
        )
        if error:
            continuity_errors.append(error)

        available_scenarios = [
            scenario
            for scenario in SCENARIO_NAMES
            if scenario in dataset.evaluation
        ]

        if available_scenarios:
            first_scenario = available_scenarios[0]

            error = self._check_continuity(
                dataset.calibration,
                dataset.evaluation[first_scenario],
                "Calibration",
                "Evaluation",
            )
            if error:
                continuity_errors.append(error)

        if continuity_errors:
            errors.append(
                "STR-05 Split Continuity: "
                + "; ".join(continuity_errors)
            )

        # STR-06: Scenario Alignment
        available_scenarios = [
            scenario
            for scenario in SCENARIO_NAMES
            if scenario in dataset.evaluation
        ]

        if len(available_scenarios) == len(SCENARIO_NAMES):
            reference = pd.to_datetime(
                dataset.evaluation[available_scenarios[0]]["timestamp"]
            ).reset_index(drop=True)

            mismatched_scenarios = []

            for scenario in available_scenarios[1:]:
                observed = pd.to_datetime(
                    dataset.evaluation[scenario]["timestamp"]
                ).reset_index(drop=True)

                if not reference.equals(observed):
                    mismatched_scenarios.append(scenario)

            if mismatched_scenarios:
                errors.append(
                    "STR-06 Scenario Alignment: evaluation timestamps "
                    "differ for scenarios "
                    + ", ".join(mismatched_scenarios)
                )

        return tuple(errors)


    def inspect_seed(self, seed: int) -> SeedInspectionStatus:
        """Inspect one seed independently."""

        try:
            dataset = self.loader.load_seed_dataset(seed)
        except (DatasetLoadError, OSError, ValueError) as exc:
            return SeedInspectionStatus(
                seed=seed,
                status=InspectionStatus.FAIL,
                errors=(f"Seed {seed} loading failed: {exc}",),
            )

        errors = self._run_checks(dataset)

        return SeedInspectionStatus(
            seed=seed,
            status=(
                InspectionStatus.FAIL
                if errors
                else InspectionStatus.PASS
            ),
            errors=errors,
        )

    def inspect(self) -> StructuralInspectionReport:
        """Inspect all configured seeds and aggregate structural results."""

        manifest = self.loader.validate_manifest()

        if not manifest.is_valid:
            return StructuralInspectionReport(
                status=InspectionStatus.FAIL,
                seeds_inspected=0,
                datasets_inspected=0,
                passed_checks=0,
                warning_checks=0,
                failed_checks=0,
                seed_results=(),
                errors=tuple(
                    f"Manifest validation failed: {error}"
                    for error in manifest.errors
                ),
            )

        seed_results: list[SeedInspectionStatus] = []
        datasets_inspected = 0
        errors: list[str] = []

        for seed in self.experiment_config.seeds:
            result = self.inspect_seed(seed)
            seed_results.append(result)

            if result.errors and result.errors[0].startswith(
                f"Seed {seed} loading failed:"
            ):
                errors.extend(result.errors)
                continue

            datasets_inspected += len(SCENARIO_NAMES)
            errors.extend(result.errors)

        structural_check_ids = (
            "STR-01",
            "STR-02",
            "STR-03",
            "STR-04",
            "STR-05",
            "STR-06",
        )

        loaded_seeds = [
            result
            for result in seed_results
            if not (
                result.errors
                and result.errors[0].startswith("Seed ")
            )
        ]

        failed_checks = sum(
            1
            for result in loaded_seeds
            for check_id in structural_check_ids
            if any(
                error.startswith(check_id)
                for error in result.errors
            )
        )

        passed_checks = (len(loaded_seeds) * 6) - failed_checks

        overall_status = (
            InspectionStatus.FAIL
            if errors
            else InspectionStatus.PASS
        )

        return StructuralInspectionReport(
            status=overall_status,
            seeds_inspected=len(seed_results),
            datasets_inspected=datasets_inspected,
            passed_checks=passed_checks,
            warning_checks=0,
            failed_checks=failed_checks,
            seed_results=tuple(seed_results),
            errors=tuple(errors),
        )
