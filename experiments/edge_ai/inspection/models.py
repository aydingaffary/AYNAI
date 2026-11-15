"""Data models shared by the Edge AI inspection subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class InspectionStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class SeedDataset:
    """All source datasets belonging to one experiment seed."""

    seed: int
    training: pd.DataFrame
    calibration: pd.DataFrame
    evaluation: dict[str, pd.DataFrame]
    ground_truth: dict[str, pd.DataFrame]
    events: pd.DataFrame




@dataclass(frozen=True)
class ManifestValidationResult:
    """Structural validation summary for the dataset manifest."""

    status: InspectionStatus
    expected_seeds: tuple[int, ...]
    observed_seeds: tuple[int, ...]
    dataset_count: int
    missing_seeds: tuple[int, ...] = ()
    unexpected_seeds: tuple[int, ...] = ()
    missing_scenarios: tuple[tuple[int, str], ...] = ()
    duplicate_entries: tuple[tuple[int, str], ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.status != InspectionStatus.FAIL


@dataclass(frozen=True)
class SeedInspectionStatus:
    """Validation status for an individual seed."""

    seed: int
    status: InspectionStatus
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
@dataclass(frozen=True)
class StructuralInspectionReport:
    """Aggregated structural inspection results across experiment seeds."""

    status: InspectionStatus
    seeds_inspected: int
    datasets_inspected: int
    passed_checks: int
    warning_checks: int
    failed_checks: int
    seed_results: tuple[SeedInspectionStatus, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _aggregate_inspection_status(
    statuses: tuple[InspectionStatus, ...],
) -> InspectionStatus:
    """Aggregate inspection statuses using severity precedence."""

    if not statuses:
        return InspectionStatus.NOT_APPLICABLE

    if InspectionStatus.FAIL in statuses:
        return InspectionStatus.FAIL

    if InspectionStatus.WARNING in statuses:
        return InspectionStatus.WARNING

    if InspectionStatus.PASS in statuses:
        return InspectionStatus.PASS

    return InspectionStatus.NOT_APPLICABLE


@dataclass(frozen=True)
class AuditCheckResult:
    """Result of one independent Ground Truth audit check."""

    check_id: str
    name: str
    status: InspectionStatus
    message: str
    expected: str | None = None
    actual: str | None = None
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SeedAuditResult:
    """Aggregated Ground Truth audit results for one experiment seed."""

    seed: int
    checks: tuple[AuditCheckResult, ...]
    dataset_count: int
    started_at: str | None = None
    duration_seconds: float | None = None

    @property
    def status(self) -> InspectionStatus:
        return _aggregate_inspection_status(
            tuple(check.status for check in self.checks)
        )

    @property
    def passed_checks(self) -> int:
        return sum(
            check.status == InspectionStatus.PASS
            for check in self.checks
        )

    @property
    def warning_checks(self) -> int:
        return sum(
            check.status == InspectionStatus.WARNING
            for check in self.checks
        )

    @property
    def failed_checks(self) -> int:
        return sum(
            check.status == InspectionStatus.FAIL
            for check in self.checks
        )

    @property
    def not_applicable_checks(self) -> int:
        return sum(
            check.status == InspectionStatus.NOT_APPLICABLE
            for check in self.checks
        )


@dataclass(frozen=True)
class GroundTruthAuditReport:
    """Aggregated Ground Truth audit report across experiment seeds."""

    seed_results: tuple[SeedAuditResult, ...]
    started_at: str | None = None
    duration_seconds: float | None = None

    @property
    def status(self) -> InspectionStatus:
        return _aggregate_inspection_status(
            tuple(result.status for result in self.seed_results)
        )

    @property
    def total_seeds(self) -> int:
        return len(self.seed_results)

    @property
    def total_datasets(self) -> int:
        return sum(
            result.dataset_count for result in self.seed_results
        )

    @property
    def passed_seeds(self) -> int:
        return sum(
            result.status == InspectionStatus.PASS
            for result in self.seed_results
        )

    @property
    def warning_seeds(self) -> int:
        return sum(
            result.status == InspectionStatus.WARNING
            for result in self.seed_results
        )

    @property
    def failed_seeds(self) -> int:
        return sum(
            result.status == InspectionStatus.FAIL
            for result in self.seed_results
        )

    @property
    def not_applicable_seeds(self) -> int:
        return sum(
            result.status == InspectionStatus.NOT_APPLICABLE
            for result in self.seed_results
        )
