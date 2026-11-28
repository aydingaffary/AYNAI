"""Read-only auditing of Ground Truth and event metadata."""

from __future__ import annotations

from collections.abc import Iterable
import json
from numbers import Integral, Real

import pandas as pd

from experiments.edge_ai.config import SCENARIO_NAMES
from experiments.edge_ai.data.telemetry_generator import FEATURE_COLUMNS
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

    GROUND_TRUTH_CHECKS = (
        ("GT-01", "Seed Consistency"),
        ("GT-02", "Scenario Consistency"),
        ("GT-03", "Phase Consistency"),
        ("GT-04", "Timestamp Alignment"),
        ("GT-05", "Binary Label Domain"),
        ("GT-06", "Label/Metadata Consistency"),
        ("GT-07", "Event ID Integrity"),
    )

    EVENT_CHECKS = (
        ("EVT-01", "Event ID Uniqueness"),
        ("EVT-02", "Event Domain"),
        ("EVT-03", "Event Index Validity"),
        ("EVT-04", "Event Interval Non-Overlap"),
        ("EVT-05", "Event Timestamp Alignment"),
        ("EVT-06", "Event JSON Integrity"),
        ("EVT-07", "Event Feature Integrity"),
        ("EVT-08", "Event-to-Ground Truth Alignment"),
    )

    CHECKS = GROUND_TRUTH_CHECKS + EVENT_CHECKS

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

        for check_id, name in self.EVENT_CHECKS:
            try:
                if check_id == "EVT-01":
                    result = self._check_event_id_uniqueness(dataset)
                elif check_id == "EVT-02":
                    result = self._check_event_domain(dataset)
                elif check_id == "EVT-03":
                    result = self._check_event_indices(dataset)
                elif check_id == "EVT-04":
                    result = self._check_event_overlaps(dataset)
                elif check_id == "EVT-05":
                    result = self._check_event_timestamps(dataset)
                elif check_id == "EVT-06":
                    result = self._check_event_json_integrity(dataset)
                elif check_id == "EVT-07":
                    result = self._check_event_feature_integrity(dataset)
                elif check_id == "EVT-08":
                    result = self._check_event_ground_truth_alignment(dataset)
                else:
                    raise NotImplementedError(
                        f"Event check is not implemented: {check_id}"
                    )
            except Exception as exc:
                result = self._result(
                    check_id,
                    name,
                    InspectionStatus.FAIL,
                    f"Check execution failed: {type(exc).__name__}: {exc}",
                    "*",
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
            for check_id, name in self.GROUND_TRUTH_CHECKS
        ) + tuple(
            self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                message,
                "*",
                actual="seed-level audit failure",
            )
            for check_id, name in self.EVENT_CHECKS
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

    def _check_event_id_uniqueness(
        self,
        dataset: SeedDataset,
    ) -> AuditCheckResult:
        """Validate non-empty and seed-wide unique event identifiers."""
        check_id, name = "EVT-01", "Event ID Uniqueness"
        events = dataset.events

        if not isinstance(events, pd.DataFrame):
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata must be a pandas DataFrame.",
                "*",
                expected="events DataFrame with an event_id column",
                actual=type(events).__name__,
            )

        if "event_id" not in events.columns:
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata is missing the event_id column.",
                "*",
                expected="event_id column present",
                actual="missing column",
            )

        identifiers = events["event_id"]
        issues = []
        first_seen = {}

        for row_index, event_id in identifiers.items():
            if not isinstance(event_id, str) or not event_id.strip():
                issues.append(
                    (
                        ("event_row", str(row_index)),
                        ("event_id", repr(event_id)),
                        ("issue", "event_id must be a non-empty string"),
                    )
                )
                continue

            if event_id in first_seen:
                issues.append(
                    (
                        ("event_row", str(row_index)),
                        ("event_id", event_id),
                        ("issue", f"duplicate of row {first_seen[event_id]}"),
                    )
                )
            else:
                first_seen[event_id] = row_index

        details = tuple(
            item
            for issue in issues
            for item in issue
        )

        valid = not issues

        return self._result(
            check_id,
            name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "All event IDs are non-empty and unique within the seed."
            if valid else "Event IDs contain empty or duplicate values.",
            "*",
            expected="non-empty string identifiers unique across the seed",
            actual=(
                f"event_count={len(events)}; "
                f"unique_valid_ids={len(first_seen)}; "
                f"issues={len(issues)}"
            ),
            details=details,
        )

    def _check_event_domain(
        self,
        dataset: SeedDataset,
    ) -> AuditCheckResult:
        """Validate seed ownership, scenario, and severity for all events."""
        check_id, name = "EVT-02", "Event Domain"
        events = dataset.events

        if not isinstance(events, pd.DataFrame):
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata must be a pandas DataFrame.",
                "*",
                expected="events DataFrame with seed, scenario, and severity",
                actual=type(events).__name__,
            )

        required_columns = ("seed", "scenario", "severity")
        missing = [
            column
            for column in required_columns
            if column not in events.columns
        ]

        issues = []

        for column in missing:
            issues.append(
                (
                    ("field", column),
                    ("issue", "missing required column"),
                )
            )

        valid_severities = {"weak", "medium", "strong"}

        for row_index, row in events.iterrows():
            event_id = (
                repr(row["event_id"])
                if "event_id" in events.columns
                else "unavailable"
            )

            values = {
                "seed": row["seed"] if "seed" in events.columns else None,
                "scenario": (
                    row["scenario"] if "scenario" in events.columns else None
                ),
                "severity": (
                    row["severity"] if "severity" in events.columns else None
                ),
            }

            checks = (
                (
                    "seed",
                    lambda value: (
                        isinstance(value, Integral)
                        and not isinstance(value, bool)
                        and value == dataset.seed
                    ),
                    f"must be the integer seed {dataset.seed}",
                ),
                (
                    "scenario",
                    lambda value: (
                        isinstance(value, str)
                        and value in SCENARIO_NAMES
                    ),
                    "must be a supported scenario name",
                ),
                (
                    "severity",
                    lambda value: (
                        isinstance(value, str)
                        and value in valid_severities
                    ),
                    "must be one of weak, medium, strong",
                ),
            )

            for field, validator, reason in checks:
                if field not in events.columns:
                    continue

                value = values[field]

                if not validator(value):
                    issues.append(
                        (
                            ("event_row", str(row_index)),
                            ("event_id", event_id),
                            ("field", field),
                            ("value", repr(value)),
                            ("issue", reason),
                        )
                    )

        details = tuple(
            item
            for issue in issues
            for item in issue
        )

        valid = not issues

        return self._result(
            check_id,
            name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "All event metadata values belong to their supported domains."
            if valid else "Event metadata contains missing or invalid domain values.",
            "*",
            expected=(
                f"seed={dataset.seed}; "
                "scenario in SCENARIO_NAMES; "
                "severity in {weak, medium, strong}"
            ),
            actual=(
                f"event_count={len(events)}; "
                f"issues={len(issues)}"
            ),
            details=details,
        )

    def _check_event_indices(
        self,
        dataset: SeedDataset,
    ) -> AuditCheckResult:
        """Validate event intervals against their scenario evaluation data."""
        check_id, name = "EVT-03", "Event Index Validity"
        events = dataset.events

        if not isinstance(events, pd.DataFrame):
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata must be a pandas DataFrame.",
                "*",
                expected="events DataFrame with scenario, start_index, end_index",
                actual=type(events).__name__,
            )

        required_columns = ("scenario", "start_index", "end_index")
        missing = [
            column for column in required_columns
            if column not in events.columns
        ]

        issues = []

        def add_issue(
            row_index: object | None,
            event_id: str,
            scenario: object,
            field: str,
            value: object,
            reason: str,
        ) -> None:
            details = []
            if row_index is not None:
                details.append(("event_row", str(row_index)))
            details.extend(
                (
                    ("event_id", event_id),
                    ("scenario", repr(scenario)),
                    ("field", field),
                    ("value", repr(value)),
                    ("issue", reason),
                )
            )
            issues.append(tuple(details))

        for column in missing:
            add_issue(
                None,
                "unavailable",
                "unavailable",
                column,
                "unavailable",
                "missing required column",
            )

        for position in range(len(events)):
            row_index = events.index[position]

            event_id = (
                repr(events["event_id"].iloc[position])
                if "event_id" in events.columns
                else "unavailable"
            )

            scenario = (
                events["scenario"].iloc[position]
                if "scenario" in events.columns
                else None
            )

            evaluation = None
            evaluation_available = False

            if "scenario" in events.columns:
                if not isinstance(scenario, str) or scenario not in SCENARIO_NAMES:
                    add_issue(
                        row_index, event_id, scenario,
                        "scenario", scenario,
                        "must be a supported scenario name",
                    )
                elif scenario not in dataset.evaluation:
                    add_issue(
                        row_index, event_id, scenario,
                        "scenario", scenario,
                        "scenario evaluation is missing",
                    )
                elif not isinstance(dataset.evaluation[scenario], pd.DataFrame):
                    add_issue(
                        row_index, event_id, scenario,
                        "evaluation",
                        type(dataset.evaluation[scenario]).__name__,
                        "scenario evaluation must be a pandas DataFrame",
                    )
                else:
                    evaluation = dataset.evaluation[scenario]
                    evaluation_available = True

            start = (
                events["start_index"].iloc[position]
                if "start_index" in events.columns
                else None
            )
            end = (
                events["end_index"].iloc[position]
                if "end_index" in events.columns
                else None
            )

            start_valid = (
                isinstance(start, Integral)
                and not isinstance(start, bool)
            )
            end_valid = (
                isinstance(end, Integral)
                and not isinstance(end, bool)
            )

            if "start_index" in events.columns and not start_valid:
                add_issue(
                    row_index, event_id, scenario,
                    "start_index", start,
                    "must be a non-boolean integer",
                )

            if "end_index" in events.columns and not end_valid:
                add_issue(
                    row_index, event_id, scenario,
                    "end_index", end,
                    "must be a non-boolean integer",
                )

            if start_valid and start < 0:
                add_issue(
                    row_index, event_id, scenario,
                    "start_index", start,
                    "must be greater than or equal to zero",
                )

            if start_valid and end_valid:
                if end <= start:
                    add_issue(
                        row_index, event_id, scenario,
                        "end_index", end,
                        "must be greater than start_index",
                    )

                if evaluation_available and end > len(evaluation):
                    add_issue(
                        row_index, event_id, scenario,
                        "end_index", end,
                        f"must not exceed evaluation length {len(evaluation)}",
                    )

        details = tuple(
            item
            for issue in issues
            for item in issue
        )
        valid = not issues

        return self._result(
            check_id,
            name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "All event intervals are valid for their scenario evaluations."
            if valid else "Event metadata contains invalid index intervals.",
            "*",
            expected="0 <= start_index < end_index <= scenario evaluation length",
            actual=f"event_count={len(events)}; issues={len(issues)}",
            details=details,
        )

    def _check_event_overlaps(
        self,
        dataset: SeedDataset,
    ) -> AuditCheckResult:
        """Ensure event intervals do not overlap within a scenario."""
        check_id, name = "EVT-04", "Event Interval Non-Overlap"
        events = dataset.events

        if not isinstance(events, pd.DataFrame):
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata must be a pandas DataFrame.",
                "*",
                expected="events DataFrame with scenario, start_index, end_index",
                actual=type(events).__name__,
            )

        required_columns = ("scenario", "start_index", "end_index")
        missing = [
            column for column in required_columns
            if column not in events.columns
        ]

        if missing:
            details = tuple(
                item
                for column in missing
                for item in (
                    ("field", column),
                    ("issue", "missing required column"),
                )
            )
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata is missing required columns.",
                "*",
                expected=str(required_columns),
                actual=f"missing={missing}",
                details=details,
            )

        intervals_by_scenario = {}

        for position in range(len(events)):
            scenario = events["scenario"].iloc[position]

            if not isinstance(scenario, str):
                continue

            if scenario not in SCENARIO_NAMES:
                continue

            evaluation = dataset.evaluation.get(scenario)
            if not isinstance(evaluation, pd.DataFrame):
                continue

            start = events["start_index"].iloc[position]
            end = events["end_index"].iloc[position]

            start_valid = (
                isinstance(start, Integral)
                and not isinstance(start, bool)
            )
            end_valid = (
                isinstance(end, Integral)
                and not isinstance(end, bool)
            )

            if not start_valid or not end_valid:
                continue

            if start < 0 or end <= start or end > len(evaluation):
                continue

            event_id = (
                str(events["event_id"].iloc[position])
                if "event_id" in events.columns
                else "unavailable"
            )

            interval = (
                int(start),
                int(end),
                position,
                str(events.index[position]),
                event_id,
            )
            intervals_by_scenario.setdefault(scenario, []).append(interval)

        issues = []

        for scenario, intervals in intervals_by_scenario.items():
            ordered = sorted(
                intervals,
                key=lambda item: (item[0], item[1], item[2]),
            )

            for left_position, left in enumerate(ordered):
                left_start, left_end, _, left_row, left_id = left

                for right_position in range(
                    left_position + 1,
                    len(ordered),
                ):
                    right = ordered[right_position]
                    right_start, right_end, _, right_row, right_id = right

                    # Half-open intervals: adjacency is not overlap.
                    if right_start >= left_end:
                        break

                    issues.append(
                        (
                            ("scenario", scenario),
                            ("event_a_row", left_row),
                            ("event_a_id", left_id),
                            (
                                "event_a_interval",
                                f"[{left_start}, {left_end})",
                            ),
                            ("event_b_row", right_row),
                            ("event_b_id", right_id),
                            (
                                "event_b_interval",
                                f"[{right_start}, {right_end})",
                            ),
                            ("issue", "event intervals overlap"),
                        )
                    )

        details = tuple(
            item
            for issue in issues
            for item in issue
        )
        overlap_pairs = len(issues)
        valid = overlap_pairs == 0

        return self._result(
            check_id,
            name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "No event intervals overlap within any scenario."
            if valid else "Event metadata contains overlapping intervals.",
            "*",
            expected="No overlapping event intervals within each scenario",
            actual=(
                f"event_count={len(events)}; "
                f"overlap_pairs={overlap_pairs}"
            ),
            details=details,
        )

    def _check_event_timestamps(
        self,
        dataset: SeedDataset,
    ) -> AuditCheckResult:
        """Validate event timestamps against indexed evaluation samples."""
        check_id, name = "EVT-05", "Event Timestamp Alignment"
        events = dataset.events

        if not isinstance(events, pd.DataFrame):
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata must be a pandas DataFrame.",
                "*",
                expected=(
                    "events DataFrame with scenario, start_index, "
                    "end_index, start_timestamp, end_timestamp"
                ),
                actual=type(events).__name__,
            )

        required_columns = (
            "scenario",
            "start_index",
            "end_index",
            "start_timestamp",
            "end_timestamp",
        )
        missing = [
            column for column in required_columns
            if column not in events.columns
        ]

        if missing:
            details = tuple(
                item
                for column in missing
                for item in (
                    ("field", column),
                    ("issue", "missing required column"),
                )
            )
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata is missing required timestamp fields.",
                "*",
                expected=str(required_columns),
                actual=f"missing={missing}",
                details=details,
            )

        issues = []
        checked_events = 0

        def add_issue(
            row_index,
            event_id,
            scenario,
            field,
            expected,
            actual,
            reason,
        ):
            issues.append(
                (
                    ("event_row", str(row_index)),
                    ("event_id", event_id),
                    ("scenario", repr(scenario)),
                    ("field", field),
                    ("expected", repr(expected)),
                    ("actual", repr(actual)),
                    ("issue", reason),
                )
            )

        for position in range(len(events)):
            row_index = events.index[position]
            event_id = (
                str(events["event_id"].iloc[position])
                if "event_id" in events.columns
                else "unavailable"
            )
            scenario = events["scenario"].iloc[position]

            # EVT-02 owns scenario-domain validation.
            if (
                not isinstance(scenario, str)
                or scenario not in SCENARIO_NAMES
            ):
                continue

            evaluation = dataset.evaluation.get(scenario)
            if not isinstance(evaluation, pd.DataFrame):
                continue

            # EVT-03 owns interval-index validation.
            start = events["start_index"].iloc[position]
            end = events["end_index"].iloc[position]

            start_valid = (
                isinstance(start, Integral)
                and not isinstance(start, bool)
            )
            end_valid = (
                isinstance(end, Integral)
                and not isinstance(end, bool)
            )

            if not start_valid or not end_valid:
                continue

            if start < 0 or end <= start or end > len(evaluation):
                continue

            if "timestamp" not in evaluation.columns:
                add_issue(
                    row_index,
                    event_id,
                    scenario,
                    "timestamp",
                    "timestamp column in evaluation",
                    "missing column",
                    "scenario evaluation is missing timestamp column",
                )
                continue

            checked_events += 1

            endpoints = (
                (
                    "start_timestamp",
                    evaluation["timestamp"].iloc[int(start)],
                    events["start_timestamp"].iloc[position],
                ),
                (
                    "end_timestamp",
                    evaluation["timestamp"].iloc[int(end) - 1],
                    events["end_timestamp"].iloc[position],
                ),
            )

            for field, expected_raw, actual_raw in endpoints:
                try:
                    expected = pd.to_datetime(
                        expected_raw,
                        errors="coerce",
                    )
                    actual = pd.to_datetime(
                        actual_raw,
                        errors="coerce",
                    )

                    if pd.isna(expected) or pd.isna(actual):
                        reason = "timestamp is invalid or missing"
                        valid = False
                    else:
                        valid = expected == actual
                        reason = (
                            "timestamp matches evaluation sample"
                            if valid
                            else "timestamp does not match evaluation sample"
                        )

                except (TypeError, ValueError, OverflowError):
                    expected = expected_raw
                    actual = actual_raw
                    valid = False
                    reason = "timestamp cannot be parsed or compared"

                if not valid:
                    add_issue(
                        row_index,
                        event_id,
                        scenario,
                        field,
                        expected,
                        actual,
                        reason,
                    )

        details = tuple(
            item
            for issue in issues
            for item in issue
        )
        valid = not issues

        return self._result(
            check_id,
            name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "All event timestamps align with their evaluation samples."
            if valid else "Event metadata contains misaligned timestamps.",
            "*",
            expected=(
                "start_timestamp = evaluation[start_index].timestamp; "
                "end_timestamp = evaluation[end_index - 1].timestamp"
            ),
            actual=(
                f"event_count={len(events)}; "
                f"checked_events={checked_events}; "
                f"issues={len(issues)}"
            ),
            details=details,
        )

    def _check_event_json_integrity(
        self,
        dataset: SeedDataset,
    ) -> AuditCheckResult:
        """Validate serialized JSON fields in event metadata."""
        check_id, name = "EVT-06", "Event JSON Integrity"
        events = dataset.events

        if not isinstance(events, pd.DataFrame):
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata must be a pandas DataFrame.",
                "*",
                expected="events is a pandas DataFrame",
                actual=f"type={type(events).__name__}",
            )

        required = (
            "scenario",
            "affected_features",
            "injection_parameters",
        )
        missing = [
            column for column in required
            if column not in events.columns
        ]

        if missing:
            details = tuple(
                ("missing_column", column)
                for column in missing
            )
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                "Event metadata is missing required JSON columns.",
                "*",
                expected=", ".join(required),
                actual=f"missing_columns={missing}",
                details=details,
            )

        issues = []
        checked_events = 0

        def add_issue(
            row_index: object,
            event_id: object,
            scenario: object,
            field: str,
            expected: str,
            actual: object,
            issue: str,
        ) -> None:
            issues.append((
                ("event_row", str(row_index)),
                ("event_id", str(event_id)),
                ("event_scenario", str(scenario)),
                ("field", field),
                ("expected", expected),
                ("actual", repr(actual)),
                ("issue", issue),
            ))

        for row_index, event in events.iterrows():
            scenario = event["scenario"]

            # EVT-02 owns invalid or unknown scenario values.
            if (
                not isinstance(scenario, str)
                or scenario not in SCENARIO_NAMES
            ):
                continue

            checked_events += 1
            event_id = event.get("event_id", "unavailable")

            for field in ("affected_features", "injection_parameters"):
                raw_value = event[field]

                try:
                    parsed = json.loads(raw_value)
                except (TypeError, ValueError):
                    add_issue(
                        row_index,
                        event_id,
                        scenario,
                        field,
                        "valid JSON string",
                        raw_value,
                        "JSON parsing failed",
                    )
                    continue

                if field == "affected_features":
                    if not isinstance(parsed, list):
                        add_issue(
                            row_index,
                            event_id,
                            scenario,
                            field,
                            "non-empty JSON array of valid feature names",
                            parsed,
                            "decoded JSON value is not an array",
                        )
                    elif not parsed:
                        add_issue(
                            row_index,
                            event_id,
                            scenario,
                            field,
                            "non-empty JSON array of valid feature names",
                            parsed,
                            "feature array is empty",
                        )
                    else:
                        invalid_features = [
                            feature
                            for feature in parsed
                            if (
                                not isinstance(feature, str)
                                or feature not in FEATURE_COLUMNS
                            )
                        ]
                        if invalid_features:
                            add_issue(
                                row_index,
                                event_id,
                                scenario,
                                field,
                                "all entries must belong to FEATURE_COLUMNS",
                                invalid_features,
                                "unknown or invalid feature entries",
                            )

                elif not isinstance(parsed, dict):
                    add_issue(
                        row_index,
                        event_id,
                        scenario,
                        field,
                        "JSON object",
                        parsed,
                        "decoded JSON value is not an object",
                    )

        details = tuple(
            item
            for issue in issues
            for item in issue
        )
        valid = not issues

        return self._result(
            check_id,
            name,
            InspectionStatus.PASS if valid else InspectionStatus.FAIL,
            "All event JSON metadata is valid."
            if valid else "Event metadata contains invalid JSON fields.",
            "*",
            expected=(
                "affected_features: non-empty array of valid feature names; "
                "injection_parameters: JSON object"
            ),
            actual=(
                f"event_count={len(events)}; "
                f"checked_events={checked_events}; "
                f"issues={len(issues)}"
            ),
            details=details,
        )

    def _check_event_feature_integrity(
        self,
        dataset: SeedDataset,
    ) -> AuditCheckResult:
        """Validate affected features against scenario datasets."""
        check_id, name = "EVT-07", "Event Feature Integrity"
        events = dataset.events

        # EVT-06 owns malformed event containers and missing JSON columns.
        required_columns = {
            "scenario",
            "affected_features",
        }
        if (
            not isinstance(events, pd.DataFrame)
            or not required_columns.issubset(events.columns)
        ):
            return self._result(
                check_id,
                name,
                InspectionStatus.PASS,
                "Skipped: event structure is validated by EVT-06.",
                "*",
                actual="No independently auditable event features",
            )

        issues = []
        checked_events = 0
        skipped_events = 0

        for row_index, event in events.iterrows():
            scenario = event["scenario"]

            # EVT-02 owns invalid scenario values.
            if scenario not in SCENARIO_NAMES:
                skipped_events += 1
                continue

            raw_features = event["affected_features"]

            # EVT-06 owns JSON parsing and feature-list validation.
            try:
                features = json.loads(raw_features)
            except (TypeError, ValueError):
                skipped_events += 1
                continue

            if (
                not isinstance(features, list)
                or not features
                or any(
                    not isinstance(feature, str)
                    or feature not in FEATURE_COLUMNS
                    for feature in features
                )
            ):
                skipped_events += 1
                continue

            checked_events += 1
            event_id = event.get("event_id", "")
            seen_features = set()

            evaluation = (
                dataset.evaluation.get(scenario)
                if isinstance(dataset.evaluation, dict)
                else None
            )
            ground_truth = (
                dataset.ground_truth.get(scenario)
                if isinstance(dataset.ground_truth, dict)
                else None
            )

            for feature in features:
                if feature in seen_features:
                    issues.append(
                        (
                            ("event_row", str(row_index)),
                            ("event_id", str(event_id)),
                            ("event_scenario", str(scenario)),
                            ("feature", feature),
                            ("data_source", "event metadata"),
                            ("issue", "Duplicate affected feature"),
                        )
                    )
                else:
                    seen_features.add(feature)

            # Check dataset columns only once per unique feature.
            for feature in dict.fromkeys(features):
                if (
                    not isinstance(evaluation, pd.DataFrame)
                    or feature not in evaluation.columns
                ):
                    issues.append(
                        (
                            ("event_row", str(row_index)),
                            ("event_id", str(event_id)),
                            ("event_scenario", str(scenario)),
                            ("feature", feature),
                            ("data_source", "evaluation"),
                            ("issue", "Affected feature is missing"),
                        )
                    )

                if (
                    not isinstance(ground_truth, pd.DataFrame)
                    or feature not in ground_truth.columns
                ):
                    issues.append(
                        (
                            ("event_row", str(row_index)),
                            ("event_id", str(event_id)),
                            ("event_scenario", str(scenario)),
                            ("feature", feature),
                            ("data_source", "ground_truth"),
                            ("issue", "Affected feature is missing"),
                        )
                    )

        details = tuple(
            detail
            for issue in issues
            for detail in issue
        )

        if issues:
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                (
                    f"Found {len(issues)} event feature integrity issue(s) "
                    f"across {checked_events} valid event(s)."
                ),
                "*",
                expected=(
                    "Unique affected features present in both "
                    "Evaluation and Ground Truth"
                ),
                actual=f"{len(issues)} issue(s)",
                details=details,
            )

        return self._result(
            check_id,
            name,
            InspectionStatus.PASS,
            (
                f"All {checked_events} valid event(s) reference unique "
                "features present in Evaluation and Ground Truth."
            ),
            "*",
            expected="Unique affected features present in both datasets",
            actual=(
                f"{checked_events} valid event(s); "
                f"{skipped_events} delegated event(s)"
            ),
        )

    def _check_event_ground_truth_alignment(
        self,
        dataset: SeedDataset,
    ) -> AuditCheckResult:
        """Compare each valid event interval with its Ground Truth metadata."""
        check_id = "EVT-08"
        name = "Event-to-Ground Truth Alignment"
        events = dataset.events

        required_event_columns = {
            "event_id",
            "scenario",
            "severity",
            "start_index",
            "end_index",
        }
        required_ground_truth_columns = {
            "is_anomaly",
            "event_id",
            "anomaly_type",
            "severity",
        }

        if (
            not isinstance(events, pd.DataFrame)
            or not required_event_columns.issubset(events.columns)
        ):
            return self._result(
                check_id,
                name,
                InspectionStatus.NOT_APPLICABLE,
                "Skipped: event structure or required columns are unavailable; "
                "the relevant event checks own these errors.",
                "*",
                expected=str(sorted(required_event_columns)),
                actual=(
                    type(events).__name__
                    if not isinstance(events, pd.DataFrame)
                    else f"missing={sorted(required_event_columns - set(events.columns))}"
                ),
            )

        if not isinstance(dataset.ground_truth, dict):
            return self._result(
                check_id,
                name,
                InspectionStatus.NOT_APPLICABLE,
                "Skipped: Ground Truth scenario mapping is unavailable.",
                "*",
                expected="Ground Truth mapping by scenario",
                actual=type(dataset.ground_truth).__name__,
            )

        # EVT-01 owns invalid and duplicate identifiers. Do not select an
        # arbitrary row when an identifier is ambiguous.
        id_counts = {}
        for event_id in events["event_id"].tolist():
            if isinstance(event_id, str) and event_id.strip():
                id_counts[event_id] = id_counts.get(event_id, 0) + 1

        issues = []
        checked_events = 0
        skipped_events = 0
        skipped_scenarios = set()

        for position in range(len(events)):
            row = events.iloc[position]
            row_index = str(events.index[position])
            event_id = row["event_id"]
            scenario = row["scenario"]
            severity = row["severity"]
            start = row["start_index"]
            end = row["end_index"]

            if (
                not isinstance(event_id, str)
                or not event_id.strip()
                or id_counts.get(event_id, 0) != 1
            ):
                skipped_events += 1
                continue

            # EVT-02 owns invalid scenario and severity values.
            if (
                not isinstance(scenario, str)
                or scenario not in SCENARIO_NAMES
                or not isinstance(severity, str)
                or severity not in {"weak", "medium", "strong"}
            ):
                skipped_events += 1
                continue

            # EVT-03 owns invalid event intervals and missing evaluations.
            if (
                not isinstance(start, Integral)
                or isinstance(start, bool)
                or not isinstance(end, Integral)
                or isinstance(end, bool)
            ):
                skipped_events += 1
                continue

            evaluation = (
                dataset.evaluation.get(scenario)
                if isinstance(dataset.evaluation, dict)
                else None
            )
            ground_truth = dataset.ground_truth.get(scenario)

            if not isinstance(evaluation, pd.DataFrame):
                skipped_events += 1
                continue

            if not isinstance(ground_truth, pd.DataFrame):
                return self._result(
                    check_id,
                    name,
                    InspectionStatus.NOT_APPLICABLE,
                    "Skipped: Ground Truth for an eligible event scenario "
                    "is unavailable.",
                    scenario,
                    expected="Ground Truth DataFrame for the event scenario",
                    actual=type(ground_truth).__name__,
                )

            # GT-04 owns Evaluation/Ground Truth length alignment.
            if len(ground_truth) != len(evaluation):
                skipped_events += 1
                skipped_scenarios.add(scenario)
                continue

            if not required_ground_truth_columns.issubset(
                ground_truth.columns
            ):
                missing = sorted(
                    required_ground_truth_columns - set(ground_truth.columns)
                )
                return self._result(
                    check_id,
                    name,
                    InspectionStatus.NOT_APPLICABLE,
                    "Skipped: Required Ground Truth columns are unavailable.",
                    scenario,
                    expected=str(sorted(required_ground_truth_columns)),
                    actual=f"missing={missing}",
                )

            if start < 0 or end <= start or end > len(evaluation):
                skipped_events += 1
                continue

            checked_events += 1
            expected_values = (
                ("is_anomaly", 1),
                ("event_id", event_id),
                ("anomaly_type", scenario),
                ("severity", severity),
            )

            for sample_index in range(int(start), int(end)):
                sample = ground_truth.iloc[sample_index]

                for field, expected in expected_values:
                    actual = sample[field]
                    try:
                        matches = bool(actual == expected)
                    except (TypeError, ValueError):
                        matches = False

                    if not matches:
                        issues.append(
                            (
                                ("event_row", row_index),
                                ("event_id", event_id),
                                ("scenario", scenario),
                                ("sample_index", str(sample_index)),
                                ("field", field),
                                ("expected", repr(expected)),
                                ("actual", repr(actual)),
                                ("issue", "Event metadata does not match Ground Truth"),
                            )
                        )

        details = tuple(
            item
            for issue in issues
            for item in issue
        )

        if issues:
            return self._result(
                check_id,
                name,
                InspectionStatus.FAIL,
                (
                    f"Found {len(issues)} event-to-Ground Truth mismatch(es) "
                    f"across {checked_events} checked event(s)."
                ),
                "*",
                expected=(
                    "Each event interval must contain is_anomaly=1 and "
                    "matching event_id, anomaly_type, and severity"
                ),
                actual=(
                    f"checked_events={checked_events}; "
                    f"skipped_events={skipped_events}; issues={len(issues)}"
                ),
                details=details,
            )

        return self._result(
            check_id,
            name,
            InspectionStatus.PASS,
            "All checked event intervals align with their Ground Truth metadata.",
            "*",
            expected=(
                "Each event interval must contain is_anomaly=1 and "
                "matching event_id, anomaly_type, and severity"
            ),
            actual=(
                f"checked_events={checked_events}; "
                f"skipped_events={skipped_events}; "
                f"skipped_scenarios={len(skipped_scenarios)}"
            ),
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
