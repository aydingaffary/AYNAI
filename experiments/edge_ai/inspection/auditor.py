"""Read-only auditing of Ground Truth and event metadata."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from experiments.edge_ai.config import SCENARIO_NAMES
from experiments.edge_ai.inspection.models import (
    AuditCheckResult,
    GroundTruthAuditReport,
    InspectionStatus,
    SeedAuditResult,
    SeedDataset,
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
        """Audit the internal Ground Truth consistency of one seed."""
        if not isinstance(dataset, SeedDataset):
            raise TypeError("dataset must be a SeedDataset.")
        if not isinstance(manifest, pd.DataFrame):
            raise TypeError("manifest must be a pandas DataFrame.")

        checks = []

        for scenario, ground_truth in dataset.ground_truth.items():
            evaluation = dataset.evaluation.get(scenario)

            for check_id, name, callback in (
                ("GT-01", "Seed Consistency", self._check_seed),
                ("GT-02", "Scenario Consistency", self._check_scenario),
                ("GT-03", "Phase Consistency", self._check_phase),
                ("GT-04", "Timestamp Alignment", self._check_timestamps),
                ("GT-05", "Binary Label Domain", self._check_labels),
                ("GT-06", "Label/Metadata Consistency", self._check_metadata),
                ("GT-07", "Event ID Integrity", self._check_event_ids),
            ):
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
        """Audit multiple seeds and preserve seed-level fault isolation."""
        if not isinstance(manifest, pd.DataFrame):
            raise TypeError("manifest must be a pandas DataFrame.")

        results = []

        for dataset in datasets:
            results.append(self.audit_seed(dataset, manifest))

        return GroundTruthAuditReport(
            seed_results=tuple(results),
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
            values.notna().all()
            and values.map(
                lambda value: (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and value == seed
                )
            ).all()
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

        def is_binary(value):
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and value in (0, 1)
            )

        valid = values.notna().all() and values.map(is_binary).all()

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
                check_id, name, InspectionStatus.NOT_APPLICABLE,
                "The anomaly label column is unavailable.",
                scenario,
            )

        labels = ground_truth["is_anomaly"]
        if not labels.notna().all() or not labels.isin([0, 1]).all():
            return self._result(
                check_id, name, InspectionStatus.NOT_APPLICABLE,
                "Metadata consistency requires valid binary labels.",
                scenario,
            )

        normal = labels.eq(0)
        anomalous = labels.eq(1)

        normal_clean = ground_truth.loc[normal, list(columns)].isna().all().all()

        anomalous_complete = True
        for column in columns:
            values = ground_truth.loc[anomalous, column]
            anomalous_complete = (
                anomalous_complete
                and values.notna().all()
                and values.map(
                    lambda value: isinstance(value, str) and bool(value.strip())
                ).all()
            )

        valid = normal_clean and anomalous_complete

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "Labels and event metadata are mutually consistent."
            if valid else "Normal/anomalous labels conflict with event metadata.",
            scenario,
            expected="normal rows: null metadata; anomalous rows: complete metadata",
            actual=(
                f"normal_clean={normal_clean}; "
                f"anomalous_complete={anomalous_complete}"
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
        if not labels.notna().all() or not labels.isin([0, 1]).all():
            return self._result(
                check_id, name, InspectionStatus.NOT_APPLICABLE,
                "Event ID integrity requires valid binary labels.",
                scenario,
            )

        ids = ground_truth["event_id"]
        normal = labels.eq(0)
        anomalous = labels.eq(1)

        normal_clean = ids.loc[normal].isna().all()
        anomalous_valid = ids.loc[anomalous].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()

        # A single event ID may span adjacent rows, but it must not
        # disappear and reappear in a later, separate interval.
        contiguous = True
        if anomalous_valid:
            active_ids = ids.loc[anomalous].tolist()
            closed_ids = set()
            previous_id = None

            for event_id in active_ids:
                if event_id != previous_id:
                    if event_id in closed_ids:
                        contiguous = False
                        break
                    if previous_id is not None:
                        closed_ids.add(previous_id)
                    previous_id = event_id

        valid = normal_clean and anomalous_valid and contiguous

        return self._result(
            check_id, name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "Event IDs are present and internally contiguous."
            if valid else "Event IDs are missing, misplaced, or split across intervals.",
            scenario,
            expected="normal rows: null IDs; each anomalous Event ID: one contiguous interval",
            actual=(
                f"normal_clean={normal_clean}; "
                f"anomalous_valid={anomalous_valid}; "
                f"contiguous={contiguous}"
            ),
        )
