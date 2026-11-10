"""Configuration for Edge AI dataset inspection."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InspectionConfig:
    temporal_windows_minutes: tuple[int, ...] = (5, 15, 30)

    distribution_metrics: tuple[str, ...] = (
        "wasserstein",
        "jensen_shannon",
    )

    generate_plots: bool = True

    jsd_bins: int = 50
    jsd_smoothing: float = 1e-10

    rolling_min_periods: int = 1

    def validate(self) -> None:
        if not isinstance(self.temporal_windows_minutes, tuple):
            raise TypeError("temporal_windows_minutes must be a tuple")

        if not self.temporal_windows_minutes:
            raise ValueError("At least one temporal window is required")

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in self.temporal_windows_minutes
        ):
            raise ValueError(
                "Temporal windows must contain positive integers"
            )

        if len(set(self.temporal_windows_minutes)) != len(
            self.temporal_windows_minutes
        ):
            raise ValueError("Temporal windows must be unique")

        allowed_metrics = {"wasserstein", "jensen_shannon"}

        if not isinstance(self.distribution_metrics, tuple):
            raise TypeError("distribution_metrics must be a tuple")

        if not self.distribution_metrics:
            raise ValueError("At least one distribution metric is required")

        if len(set(self.distribution_metrics)) != len(
            self.distribution_metrics
        ):
            raise ValueError("Distribution metrics must be unique")

        if not set(self.distribution_metrics).issubset(allowed_metrics):
            raise ValueError("Unsupported distribution metric")

        if not isinstance(self.generate_plots, bool):
            raise TypeError("generate_plots must be a boolean")

        if (
            isinstance(self.jsd_bins, bool)
            or not isinstance(self.jsd_bins, int)
            or self.jsd_bins < 2
        ):
            raise ValueError("jsd_bins must be an integer greater than 1")

        if (
            isinstance(self.jsd_smoothing, bool)
            or not isinstance(self.jsd_smoothing, (int, float))
            or not math.isfinite(self.jsd_smoothing)
            or self.jsd_smoothing <= 0
        ):
            raise ValueError("jsd_smoothing must be a positive finite number")

        if (
            isinstance(self.rolling_min_periods, bool)
            or not isinstance(self.rolling_min_periods, int)
            or self.rolling_min_periods <= 0
        ):
            raise ValueError(
                "rolling_min_periods must be a positive integer"
            )


DEFAULT_INSPECTION_CONFIG = InspectionConfig()
