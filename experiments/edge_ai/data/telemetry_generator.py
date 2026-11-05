"""Generate reproducible, correlated normal network telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.edge_ai.config import (
    DEFAULT_BASELINE,
    DEFAULT_CONFIG,
    ExperimentConfig,
    TelemetryBaseline,
)


FEATURE_COLUMNS = (
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "cpu_usage_pct",
    "memory_usage_pct",
)


def _ar1_process(
    rng: np.random.Generator,
    size: int,
    phi: float = 0.85,
) -> np.ndarray:
    """Generate a standardized first-order autoregressive process."""
    innovations = rng.standard_normal(size)
    values = np.empty(size, dtype=float)

    values[0] = innovations[0]
    innovation_scale = np.sqrt(1.0 - phi**2)

    for index in range(1, size):
        values[index] = (
            phi * values[index - 1]
            + innovation_scale * innovations[index]
        )

    return values


def _validate_baseline(baseline: TelemetryBaseline) -> None:
    """Validate baseline telemetry values."""
    values = {
        "latency_ms": baseline.latency_ms,
        "jitter_ms": baseline.jitter_ms,
        "packet_loss_pct": baseline.packet_loss_pct,
        "throughput_mbps": baseline.throughput_mbps,
        "cpu_usage_pct": baseline.cpu_usage_pct,
        "memory_usage_pct": baseline.memory_usage_pct,
    }

    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    for name in ("latency_ms", "jitter_ms", "throughput_mbps"):
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")

    for name in ("packet_loss_pct", "cpu_usage_pct", "memory_usage_pct"):
        if not 0 <= values[name] <= 100:
            raise ValueError(f"{name} must be between 0 and 100")


def generate_telemetry(
    seed: int = 1000,
    config: ExperimentConfig = DEFAULT_CONFIG,
    baseline: TelemetryBaseline = DEFAULT_BASELINE,
    start_time: str | pd.Timestamp = "2025-01-01T00:00:00",
) -> pd.DataFrame:
    """
    Generate normal network telemetry using correlated latent factors.

    The latent factors are:
    - Network congestion
    - Link instability
    - System load

    Latent factors are intentionally not included in the returned dataset.
    """
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")

    config.validate()
    _validate_baseline(baseline)

    rng = np.random.default_rng(seed)
    size = config.total_samples

    congestion = _ar1_process(rng, size)
    instability = _ar1_process(rng, size)
    system_load = _ar1_process(rng, size)

    def noise(scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, size)

    # Network congestion increases latency and packet loss,
    # while reducing throughput.
    latency = (
        baseline.latency_ms
        + 8.0 * congestion
        + 4.0 * instability
        + 3.0 * system_load
        + noise(2.0)
    )

    jitter = (
        baseline.jitter_ms
        + 0.8 * congestion
        + 2.5 * instability
        + 0.2 * system_load
        + noise(0.5)
    )

    packet_loss = (
        baseline.packet_loss_pct
        + 0.15 * congestion
        + 0.12 * instability
        + noise(0.05)
    )

    throughput = (
        baseline.throughput_mbps
        - 18.0 * congestion
        - 3.0 * instability
        + noise(4.0)
    )

    # System load primarily affects CPU and memory utilization.
    cpu_usage = (
        baseline.cpu_usage_pct
        + 8.0 * system_load
        + noise(2.0)
    )

    memory_usage = (
        baseline.memory_usage_pct
        + 6.0 * system_load
        + noise(1.5)
    )

    timestamps = pd.date_range(
        start=pd.Timestamp(start_time),
        periods=size,
        freq=pd.Timedelta(
            seconds=config.sampling_interval_seconds
        ),
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "latency_ms": np.maximum(latency, 0.0),
            "jitter_ms": np.maximum(jitter, 0.0),
            "packet_loss_pct": np.clip(packet_loss, 0.0, 100.0),
            "throughput_mbps": np.maximum(throughput, 0.0),
            "cpu_usage_pct": np.clip(cpu_usage, 0.0, 100.0),
            "memory_usage_pct": np.clip(memory_usage, 0.0, 100.0),
        }
    )
