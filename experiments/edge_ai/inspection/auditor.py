"""Read-only auditing of Ground Truth and event metadata."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real

import pandas as pd

from experiments.edge_ai.config import SCENARIO_NAMES
from experiments.edge_ai.inspection.models import (
    AuditCheckResult,
    GroundTruthAuditReport,
    InspectionStatus,
    SeedAuditResult,
    SeedDataset,
)


def _is_valid_seed_value(value: object, expected_seed: int) -> bool:
    """Return whether a value is the expected non-boolean integer seed."""
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and value == expected_seed
    )


def _is_binary_label(value: object) -> bool:
    """Return whether a value is a numeric, non-boolean binary label."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and value in (0, 1)
    )


class GroundTruthAuditor:
    """Audit Ground Truth datasets without modifying source data."""

    CHECKS = (
        ("GT-01", "Seed Consistency"),
        ("GT-02", "Scenario Consistency"),
        ("GT-03", "Phase Consistency"),
        ("GT-04", "Timestamp Alignment"),
        ("GT-05", "Binary Label Domain"),
        ("GT-06", "Label/Metadata Consistency"),
        ("GT-07", "Event ID Integrity"),
    )

    def audit_seed(
        self,
        dataset: SeedDataset,
        manifest: pd.DataFrame,
    ) -> SeedAuditResult:
        """Audit all expected and unexpected scenarios for one seed."""
        if not isinstance(dataset, SeedDataset):
            raise TypeError("dataset must be a SeedDataset.")
        if not isinstance(manifest, pd.DataFrame):
            raise TypeError("manifest must be a pandas DataFrame.")

        callbacks = (
            ("GT-01", "Seed Consistency", self._check_seed),
            ("GT-02", "Scenario Consistency", self._check_scenario),
            ("GT-03", "Phase Consistency", self._check_phase),
            ("GT-04", "Timestamp Alignment", self._check_timestamps),
            ("GT-05", "Binary Label Domain", self._check_labels),
            ("GT-06", "Label/Metadata Consistency", self._check_metadata),
            ("GT-07", "Event ID Integrity", self._check_event_ids),
        )

        expected_scenarios = list(SCENARIO_NAMES)
        unexpected_scenarios = sorted(
            set(dataset.ground_truth) - set(SCENARIO_NAMES),
            key=str,
        )
        scenarios = expected_scenarios + unexpected_scenarios

        checks = []

        for scenario in scenarios:
            if scenario not in dataset.ground_truth:
                for check_id, name, _ in callbacks:
                    if check_id == "GT-02":
                        result = self._result(
                            check_id,
                            name,
                            InspectionStatus.FAIL,
                            "Expected Ground Truth scenario is missing.",
                            scenario,
                            expected="scenario present in Ground Truth",
                            actual="missing scenario",
                        )
                    else:
                        result = self._result(
                            check_id,
                            name,
                            InspectionStatus.NOT_APPLICABLE,
                            "Scenario is missing; this check cannot be evaluated.",
                            scenario,
                        )

                    checks.append(result)

                continue

            ground_truth = dataset.ground_truth[scenario]
            evaluation = dataset.evaluation.get(scenario)

            for check_id, name, callback in callbacks:
                try:
                    result = callback(
                        dataset.seed,
                        scenario,
                        ground_truth,
                        evaluation,
                    )
                except Exception as exc:
                    result = self._result(
                        check_id,
                        name,
                        InspectionStatus.FAIL,
                        f"Check execution failed: {type(exc).__name__}: {exc}",
                        scenario,
                    )

                checks.append(result)

        return SeedAuditResult(
            seed=dataset.seed,
            checks=tuple(checks),
            dataset_count=len(dataset.ground_truth),
        )

    def audit(
        self,
        datasets: Iterable[SeedDataset],
        manifest: pd.DataFrame,
    ) -> GroundTruthAuditReport:
        """Audit seeds independently and preserve seed-level fault isolation."""
        if not isinstance(manifest, pd.DataFrame):
            raise TypeError("manifest must be a pandas DataFrame.")

        try:
            iterator = iter(datasets)
        except TypeError as exc:
            raise TypeError("datasets must be an iterable of SeedDataset objects.") from exc

        results = []
        seen_seeds = set()

        for dataset in iterator:
            if not isinstance(dataset, SeedDataset):
                raise TypeError("datasets must contain only SeedDataset objects.")

            seed = dataset.seed
            if (
                not isinstance(seed, Integral)
                or isinstance(seed, bool)
                or seed < 0
            ):
                raise ValueError(
                    "Each SeedDataset must have a non-negative integer seed."
                )

            if seed in seen_seeds:
                raise ValueError(f"Duplicate seed in audit input: {seed}.")

            seen_seeds.add(seed)

            try:
                result = self.audit_seed(dataset, manifest)
            except Exception as exc:
                result = self._failed_seed_result(dataset, exc)

            results.append(result)

        return GroundTruthAuditReport(
            seed_results=tuple(results),
        )

    def _failed_seed_result(
        self,
        dataset: SeedDataset,
        exc: Exception,
    ) -> SeedAuditResult:
        """Create a complete failure report when a seed cannot be audited."""
        ground_truth = dataset.ground_truth
        if isinstance(ground_truth, dict):
            unexpected = sorted(
                set(ground_truth) - set(SCENARIO_NAMES),
                key=str,
            )
            scenarios = (*SCENARIO_NAMES, *unexpected)
            dataset_count = len(ground_truth)
        else:
            scenarios = SCENARIO_NAMES
            dataset_count = 0

        message = (
            f"Seed audit execution failed: {type(exc).__name__}: {exc}"
        )

        checks = tuple(
            self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                message,
                scenario,
                actual="seed-level audit failure",
            )
            for scenario in scenarios
            for check_id, name in self.CHECKS
        )

        return SeedAuditResult(
            seed=dataset.seed,
            checks=checks,
            dataset_count=dataset_count,
        )

    @staticmethod
    def _result(
        check_id: str,
        name: str,
        status: InspectionStatus,
        message: str,
        scenario: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
        details: tuple[tuple[str, str], ...] = (),
    ) -> AuditCheckResult:
        return AuditCheckResult(
            check_id=check_id,
            name=name,
            status=status,
            message=message,
            expected=expected,
            actual=actual,
            details=(("scenario", str(scenario)), *details),
        )

    def _check_seed(
        self,
        seed: int,
        scenario: str,
        ground_truth: pd.DataFrame,
        evaluation: pd.DataFrame | None,
    ) -> AuditCheckResult:
        check_id, name = "GT-01", "Seed Consistency"

        if "seed" not in ground_truth.columns:
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "Ground Truth is missing the seed column.",
                scenario, expected=str(seed), actual="missing column",
            )

        values = ground_truth["seed"]

        valid = (
            not ground_truth.empty
            and values.notna().all()
            and values.map(lambda value: _is_valid_seed_value(value, seed)).all()
        )

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "All Ground Truth rows belong to the expected seed."
            if valid else "Ground Truth contains missing or inconsistent seed values.",
            scenario,
            expected=str(seed),
            actual=str(values.drop_duplicates().tolist()),
        )

    def _check_scenario(
        self,
        seed: int,
        scenario: str,
        ground_truth: pd.DataFrame,
        evaluation: pd.DataFrame | None,
    ) -> AuditCheckResult:
        check_id, name = "GT-02", "Scenario Consistency"

        if "scenario" not in ground_truth.columns:
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "Ground Truth is missing the scenario column.",
                scenario, expected=scenario, actual="missing column",
            )

        values = ground_truth["scenario"]
        valid = (
            values.notna().all()
            and values.eq(scenario).all()
            and scenario in SCENARIO_NAMES
        )

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "Scenario values match the dataset key."
            if valid else "Scenario values are missing, invalid, or inconsistent.",
            scenario,
            expected=scenario,
            actual=str(values.drop_duplicates().tolist()),
        )

    def _check_phase(
        self,
        seed: int,
        scenario: str,
        ground_truth: pd.DataFrame,
        evaluation: pd.DataFrame | None,
    ) -> AuditCheckResult:
        check_id, name = "GT-03", "Phase Consistency"

        if "phase" not in ground_truth.columns:
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "Ground Truth is missing the phase column.",
                scenario, expected="Evaluation", actual="missing column",
            )

        values = ground_truth["phase"]
        valid = values.notna().all() and values.eq("Evaluation").all()

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "All rows belong to the Evaluation phase."
            if valid else "Ground Truth contains an invalid phase.",
            scenario,
            expected="Evaluation",
            actual=str(values.drop_duplicates().tolist()),
        )

    def _check_timestamps(
        self,
        seed: int,
        scenario: str,
        ground_truth: pd.DataFrame,
        evaluation: pd.DataFrame | None,
    ) -> AuditCheckResult:
        check_id, name = "GT-04", "Timestamp Alignment"

        if evaluation is None:
            return self._result(
                check_id, name, InspectionStatus.NOT_APPLICABLE,
                "The corresponding evaluation dataset is unavailable.",
                scenario,
            )

        if (
            "timestamp" not in ground_truth.columns
            or "timestamp" not in evaluation.columns
        ):
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "A timestamp column is missing.",
                scenario, expected="matching timestamp columns",
                actual="missing column",
            )

        try:
            actual = pd.to_datetime(
                ground_truth["timestamp"], errors="raise"
            )
            expected = pd.to_datetime(
                evaluation["timestamp"], errors="raise"
            )
            valid = (
                len(actual) == len(expected)
                and actual.notna().all()
                and expected.notna().all()
                and actual.reset_index(drop=True).equals(
                    expected.reset_index(drop=True)
                )
            )
        except (TypeError, ValueError, OverflowError):
            valid = False

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "Ground Truth timestamps align with evaluation timestamps."
            if valid else "Ground Truth timestamps do not align with evaluation.",
            scenario,
            expected=f"{len(evaluation)} evaluation timestamps",
            actual=f"{len(ground_truth)} Ground Truth timestamps",
        )

    def _check_labels(
        self,
        seed: int,
        scenario: str,
        ground_truth: pd.DataFrame,
        evaluation: pd.DataFrame | None,
    ) -> AuditCheckResult:
        check_id, name = "GT-05", "Binary Label Domain"

        if "is_anomaly" not in ground_truth.columns:
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "Ground Truth is missing the is_anomaly column.",
                scenario, expected="{0, 1}", actual="missing column",
            )

        values = ground_truth["is_anomaly"]
        valid = (
            not ground_truth.empty
            and values.notna().all()
            and values.map(_is_binary_label).all()
        )

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "All labels belong to the binary domain {0, 1}."
            if valid else "Labels contain missing or out-of-domain values.",
            scenario,
            expected="{0, 1}",
            actual=str(values.drop_duplicates().tolist()),
        )

    def _check_metadata(
        self,
        seed: int,
        scenario: str,
        ground_truth: pd.DataFrame,
        evaluation: pd.DataFrame | None,
    ) -> AuditCheckResult:
        check_id, name = "GT-06", "Label/Metadata Consistency"
        columns = ("event_id", "anomaly_type", "severity")

        missing = [column for column in columns if column not in ground_truth]
        if missing:
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "Ground Truth is missing metadata columns.",
                scenario,
                expected=", ".join(columns),
                actual=f"missing: {', '.join(missing)}",
            )

        if "is_anomaly" not in ground_truth:
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "Ground Truth is missing the anomaly label column.",
                scenario,
                expected="valid binary labels",
                actual="missing is_anomaly column",
            )

        labels = ground_truth["is_anomaly"]
        labels_valid = (
            not ground_truth.empty
            and labels.notna().all()
            and labels.map(_is_binary_label).all()
        )

        if not labels_valid:
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "Metadata consistency requires non-empty valid binary labels.",
                scenario,
                expected="non-empty labels in {0, 1}",
                actual=str(labels.drop_duplicates().tolist()),
            )

        normal = labels.eq(0)
        anomalous = labels.eq(1)

        normal_clean = ground_truth.loc[normal, list(columns)].isna().all().all()

        event_ids = ground_truth.loc[anomalous, "event_id"]
        anomaly_types = ground_truth.loc[anomalous, "anomaly_type"]
        severities = ground_truth.loc[anomalous, "severity"]

        event_ids_valid = event_ids.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()

        anomaly_types_valid = anomaly_types.map(
            lambda value: isinstance(value, str) and value == scenario
        ).all()

        valid_severities = {"weak", "medium", "strong"}
        severities_valid = severities.map(
            lambda value: isinstance(value, str) and value in valid_severities
        ).all()

        anomalous_complete = (
            event_ids_valid and anomaly_types_valid and severities_valid
        )
        valid = normal_clean and anomalous_complete

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "Labels and event metadata are mutually consistent."
            if valid else "Normal/anomalous labels conflict with event metadata.",
            scenario,
            expected=(
                "normal rows: null metadata; anomalous rows: valid event ID, "
                "matching anomaly_type, and supported severity"
            ),
            actual=(
                f"normal_clean={normal_clean}; "
                f"event_ids_valid={event_ids_valid}; "
                f"anomaly_types_valid={anomaly_types_valid}; "
                f"severities_valid={severities_valid}"
            ),
        )

    def _check_event_ids(
        self,
        seed: int,
        scenario: str,
        ground_truth: pd.DataFrame,
        evaluation: pd.DataFrame | None,
    ) -> AuditCheckResult:
        check_id, name = "GT-07", "Event ID Integrity"

        required = ("is_anomaly", "event_id")
        missing = [column for column in required if column not in ground_truth]

        if missing:
            return self._result(
                check_id, name, InspectionStatus.NOT_APPLICABLE,
                "Required columns for Event ID integrity are unavailable.",
                scenario, actual=f"missing: {', '.join(missing)}",
            )

        labels = ground_truth["is_anomaly"]
        labels_valid = (
            not ground_truth.empty
            and labels.notna().all()
            and labels.map(_is_binary_label).all()
        )

        if not labels_valid:
            return self._result(
                check_id, name, InspectionStatus.FAIL,
                "Event ID integrity requires non-empty valid binary labels.",
                scenario,
                expected="non-empty labels in {0, 1}",
                actual=str(labels.drop_duplicates().tolist()),
            )

        ids = ground_truth["event_id"]
        normal_clean = ids.loc[labels.eq(0)].isna().all()

        anomalous_ids = ids.loc[labels.eq(1)]
        anomalous_valid = anomalous_ids.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()

        contiguous = True
        closed_ids = set()
        active_id = None

        if anomalous_valid:
            for label, event_id in zip(labels.tolist(), ids.tolist()):
                if label == 0:
                    if active_id is not None:
                        closed_ids.add(active_id)
                        active_id = None
                    continue

                if event_id != active_id:
                    if event_id in closed_ids:
                        contiguous = False
                        break

                    if active_id is not None:
                        closed_ids.add(active_id)

                    active_id = event_id

        valid = normal_clean and anomalous_valid and contiguous

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "Event IDs are present and internally contiguous."
            if valid else "Event IDs are missing, misplaced, or split across intervals.",
            scenario,
            expected=(
                "normal rows: null IDs; each anomalous Event ID: "
                "one contiguous interval"
            ),
            actual=(
                f"normal_clean={normal_clean}; "
                f"anomalous_valid={anomalous_valid}; "
                f"contiguous={contiguous}"
            ),
        )
