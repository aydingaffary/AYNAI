"""Data models shared by the Edge AI inspection subsystem."""

from __future__ import annotations

import math
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
    evaluation_base: pd.DataFrame
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


def _validate_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_duration(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("duration_seconds must be a number.")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(
                "duration_seconds must be finite and non-negative."
            )
        return
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "duration_seconds must be finite and non-negative."
        )


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

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str):
            raise TypeError("check_id must be a string.")
        if not self.check_id.strip():
            raise ValueError("check_id must not be empty.")

        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")
        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if not isinstance(self.status, InspectionStatus):
            raise TypeError("status must be an InspectionStatus.")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")
        if not self.message.strip():
            raise ValueError("message must not be empty.")


@dataclass(frozen=True)
class SeedAuditResult:
    """Aggregated Ground Truth audit results for one experiment seed."""

    seed: int
    checks: tuple[AuditCheckResult, ...]
    dataset_count: int
    started_at: str | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_integer("seed", self.seed)
        _validate_non_negative_integer("dataset_count", self.dataset_count)
        _validate_duration(self.duration_seconds)

        if not isinstance(self.checks, tuple):
            raise TypeError("checks must be a tuple.")
        if not all(
            isinstance(check, AuditCheckResult) for check in self.checks
        ):
            raise TypeError("checks must contain AuditCheckResult objects.")

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

    def __post_init__(self) -> None:
        _validate_duration(self.duration_seconds)
        if not isinstance(self.seed_results, tuple):
            raise TypeError("seed_results must be a tuple.")
        if not all(
            isinstance(result, SeedAuditResult)
            for result in self.seed_results
        ):
            raise TypeError("seed_results must contain SeedAuditResult objects.")
        seeds = tuple(result.seed for result in self.seed_results)
        if len(seeds) != len(set(seeds)):
            raise ValueError("seed_results must not contain duplicate seeds.")

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
