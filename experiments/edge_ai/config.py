"""Configuration for reproducible AYNAI Edge AI experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    sampling_interval_seconds: int = 30

    training_samples: int = 720
    calibration_samples: int = 240
    evaluation_samples: int = 2880

    seed_start: int = 1000
    seed_count: int = 30

    @property
    def total_samples(self) -> int:
        return (
            self.training_samples
            + self.calibration_samples
            + self.evaluation_samples
        )

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(
            range(self.seed_start, self.seed_start + self.seed_count)
        )

    def validate(self) -> None:
        values = (
            self.sampling_interval_seconds,
            self.training_samples,
            self.calibration_samples,
            self.evaluation_samples,
            self.seed_count,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in values
        ):
            raise ValueError("Configuration values must be positive integers")

        if (
            isinstance(self.seed_start, bool)
            or not isinstance(self.seed_start, int)
            or self.seed_start < 0
        ):
            raise ValueError("seed_start must be a non-negative integer")


@dataclass(frozen=True)
class TelemetryBaseline:
    latency_ms: float = 50.0
    jitter_ms: float = 5.0
    packet_loss_pct: float = 0.5
    throughput_mbps: float = 100.0
    cpu_usage_pct: float = 35.0
    memory_usage_pct: float = 45.0


DEFAULT_CONFIG = ExperimentConfig()
DEFAULT_BASELINE = TelemetryBaseline()


SCENARIO_NAMES = (
    "normal",
    "sudden_spike",
    "gradual_drift",
    "burst_anomalies",
    "variance_shift",
    "multivariate_anomaly",
)