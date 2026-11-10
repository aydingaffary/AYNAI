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
