"""Deterministic anomaly injection for Edge AI evaluation scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.edge_ai.config import SCENARIO_NAMES
from experiments.edge_ai.data.telemetry_generator import FEATURE_COLUMNS
from experiments.edge_ai.simulation.ground_truth import AnomalyEvent


EVENT_LENGTHS = {
    "sudden_spike": 3,
    "gradual_drift": 60,
    "burst_anomalies": 2,
    "variance_shift": 60,
    "multivariate_anomaly": 10,
}

SCENARIO_CODES = {
    "normal": 0,
    "sudden_spike": 1,
    "gradual_drift": 2,
    "burst_anomalies": 3,
    "variance_shift": 4,
    "multivariate_anomaly": 5,
}

NON_NEGATIVE_FEATURES = (
    "latency_ms",
    "jitter_ms",
    "throughput_mbps",
)

PERCENTAGE_FEATURES = (
    "packet_loss_pct",
    "cpu_usage_pct",
    "memory_usage_pct",
)


def _validate_input(data: pd.DataFrame, scenario: str, seed: int) -> None:
    """Validate the evaluation dataset and injection arguments."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("evaluation_data must be a pandas DataFrame")

    if data.empty:
        raise ValueError("evaluation_data must not be empty")

    if not isinstance(scenario, str) or scenario not in SCENARIO_NAMES:
        raise ValueError(f"scenario must be one of {SCENARIO_NAMES}")

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    required_columns = {"timestamp", *FEATURE_COLUMNS}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data["timestamp"].isna().any():
        raise ValueError("timestamps must not contain missing values")

    timestamps = pd.to_datetime(data["timestamp"], errors="raise")

    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and increasing")

    for column in FEATURE_COLUMNS:
        values = data[column]

        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"{column} must contain numeric values")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")

    for column in NON_NEGATIVE_FEATURES:
        if (data[column] < 0).any():
            raise ValueError(f"{column} must be non-negative")

    for column in PERCENTAGE_FEATURES:
        if not data[column].between(0.0, 100.0).all():
            raise ValueError(f"{column} must be between 0 and 100")


def _make_rng(seed: int, scenario: str) -> np.random.Generator:
    """Create a scenario-specific random stream."""
    sequence = np.random.SeedSequence(
        [seed, SCENARIO_CODES[scenario]]
    )
    return np.random.default_rng(sequence)


def _select_windows(
    rng: np.random.Generator,
    *,
    sample_count: int,
    event_count: int,
    event_length: int,
) -> list[int]:
    """Select non-overlapping event starts."""
    if event_count * event_length > sample_count:
        raise ValueError("Not enough samples for the requested events")

    starts: list[int] = []
    attempts = 0
    maximum_attempts = max(1000, event_count * sample_count * 10)

    while len(starts) < event_count:
        attempts += 1

        if attempts > maximum_attempts:
            raise RuntimeError("Could not place non-overlapping events")

        candidate = int(
            rng.integers(0, sample_count - event_length + 1)
        )

        candidate_end = candidate + event_length

        overlaps = any(
            candidate < start + event_length
            and start < candidate_end
            for start in starts
        )

        if not overlaps:
            starts.append(candidate)

    return sorted(starts)


def _create_event(
    *,
    seed: int,
    scenario: str,
    number: int,
    start: int,
    end: int,
    severity: str,
    affected_features: tuple[str, ...],
    parameters: dict,
) -> AnomalyEvent:
    """Construct a Ground Truth-compatible event."""
    return AnomalyEvent(
        event_id=f"seed-{seed}-{scenario}-{number:02d}",
        start_index=start,
        end_index=end,
        anomaly_type=scenario,
        severity=severity,
        affected_features=affected_features,
        injection_parameters=parameters,
    )


def _inject_spikes(
    result: pd.DataFrame,
    rng: np.random.Generator,
    seed: int,
) -> tuple[AnomalyEvent, ...]:
    """Inject four short, multivariate sudden spikes."""
    multipliers = (2.0, 3.0, 3.0, 5.0)
    severities = ("weak", "medium", "medium", "strong")
    affected = (
        "latency_ms",
        "jitter_ms",
        "packet_loss_pct",
    )

    starts = _select_windows(
        rng,
        sample_count=len(result),
        event_count=4,
        event_length=3,
    )

    events = []

    for number, (start, multiplier, severity) in enumerate(
        zip(starts, multipliers, severities),
        start=1,
    ):
        end = start + 3

        for feature in affected:
            result.loc[result.index[start:end], feature] *= multiplier

        events.append(
            _create_event(
                seed=seed,
                scenario="sudden_spike",
                number=number,
                start=start,
                end=end,
                severity=severity,
                affected_features=affected,
                parameters={
                    "multiplier": multiplier,
                    "latency_multiplier": multiplier,
                    "jitter_multiplier": multiplier,
                    "packet_loss_multiplier": multiplier,
                    "duration_samples": 3,
                },
            )
        )

    return tuple(events)


def _inject_gradual_drift(
    result: pd.DataFrame,
    rng: np.random.Generator,
    seed: int,
) -> tuple[AnomalyEvent, ...]:
    """Inject two linearly increasing latency and jitter events."""
    multipliers = (2.0, 3.0)
    affected = ("latency_ms", "jitter_ms")

    starts = _select_windows(
        rng,
        sample_count=len(result),
        event_count=2,
        event_length=60,
    )

    events = []

    for number, (start, multiplier) in enumerate(
        zip(starts, multipliers),
        start=1,
    ):
        end = start + 60
        ramp = np.linspace(1.0, multiplier, 60)

        for feature in affected:
            values = result[feature].iloc[start:end].to_numpy(dtype=float)
            result.loc[result.index[start:end], feature] = values * ramp

        events.append(
            _create_event(
                seed=seed,
                scenario="gradual_drift",
                number=number,
                start=start,
                end=end,
                severity="medium" if multiplier == 2.0 else "strong",
                affected_features=affected,
                parameters={
                    "start_multiplier": 1.0,
                    "end_multiplier": multiplier,
                    "duration_samples": 60,
                    "ramp": "linear",
                    "jitter_strategy": "proportional",
                },
            )
        )

    return tuple(events)


def _inject_bursts(
    result: pd.DataFrame,
    rng: np.random.Generator,
    seed: int,
) -> tuple[AnomalyEvent, ...]:
    """Inject three clusters containing five short events each."""
    affected = (
        "latency_ms",
        "jitter_ms",
        "packet_loss_pct",
    )
    multipliers = (2.0, 3.0, 5.0, 3.0, 2.0)
    severities = ("weak", "medium", "strong", "medium", "weak")

    clusters = []
    attempts = 0

    while len(clusters) < 3:
        attempts += 1

        if attempts > 10000:
            raise RuntimeError("Could not place non-overlapping clusters")

        start = int(rng.integers(0, len(result) - 14 + 1))
        gaps = [int(value) for value in rng.integers(1, 4, size=4)]

        positions = [start]

        for gap in gaps:
            positions.append(positions[-1] + 2 + gap)

        cluster_end = positions[-1] + 2

        if cluster_end > len(result):
            continue

        overlaps = any(
            start < previous_end and previous_start < cluster_end
            for previous_start, previous_end, _ in clusters
        )

        if not overlaps:
            clusters.append((start, cluster_end, positions))

    clusters.sort(key=lambda item: item[0])

    events = []
    number = 0

    for cluster_number, (_, _, positions) in enumerate(clusters, start=1):
        for event_number, (start, multiplier, severity) in enumerate(
            zip(positions, multipliers, severities),
            start=1,
        ):
            number += 1
            end = start + 2

            for feature in affected:
                result.loc[result.index[start:end], feature] *= multiplier

            gap_before = (
                None
                if event_number == 1
                else start - (positions[event_number - 2] + 2)
            )

            events.append(
                _create_event(
                    seed=seed,
                    scenario="burst_anomalies",
                    number=number,
                    start=start,
                    end=end,
                    severity=severity,
                    affected_features=affected,
                    parameters={
                        "cluster_id": cluster_number,
                        "event_in_cluster": event_number,
                        "multiplier": multiplier,
                        "gap_before_samples": gap_before,
                        "duration_samples": 2,
                    },
                )
            )

    return tuple(events)


def _set_exact_mean_and_std(
    values: np.ndarray,
    multiplier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Set an exact mean and target sample standard deviation."""
    mean = float(np.mean(values))
    original_std = float(np.std(values, ddof=1))

    if not np.isfinite(original_std) or original_std <= 0:
        raise ValueError(
            "Variance shift requires a positive finite baseline standard deviation"
        )

    target_std = original_std * multiplier

    # First attempt: scale deviations around the original mean.
    transformed = mean + multiplier * (values - mean)

    if np.all(transformed >= 0.0):
        if not np.isclose(np.mean(transformed), mean, rtol=1e-10, atol=1e-10):
            raise ArithmeticError("Variance transformation changed the mean")

        if not np.isclose(
            np.std(transformed, ddof=1),
            target_std,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ArithmeticError("Variance transformation missed target std")

        return transformed, "scaled_deviations"

    # Fallback: preserve the exact mean and sample std using
    # n-1 equal low values and one high value.
    sample_count = len(values)
    low_value = mean - target_std / np.sqrt(sample_count)
    high_value = mean + (
        target_std * (sample_count - 1) / np.sqrt(sample_count)
    )

    if low_value < 0.0:
        raise ValueError(
            "Requested variance multiplier is infeasible under non-negative bounds"
        )

    transformed = np.full(sample_count, low_value, dtype=float)
    high_index = int(rng.integers(0, sample_count))
    transformed[high_index] = high_value

    if not np.isclose(np.mean(transformed), mean, rtol=1e-10, atol=1e-10):
        raise ArithmeticError("Fallback transformation changed the mean")

    if not np.isclose(
        np.std(transformed, ddof=1),
        target_std,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ArithmeticError("Fallback transformation missed target std")

    return transformed, "bounded_two_level_fallback"


def _inject_variance_shift(
    result: pd.DataFrame,
    rng: np.random.Generator,
    seed: int,
) -> tuple[AnomalyEvent, ...]:
    """Inject two exact-mean, increased-variance windows."""
    multipliers = (2.0, 3.0)
    affected = ("latency_ms", "jitter_ms")

    starts = _select_windows(
        rng,
        sample_count=len(result),
        event_count=2,
        event_length=60,
    )

    events = []

    for number, (start, multiplier) in enumerate(
        zip(starts, multipliers),
        start=1,
    ):
        end = start + 60
        parameters = {
            "std_multiplier": multiplier,
            "duration_samples": 60,
            "mean_strategy": "exact",
            "variance_strategy": "exact",
            "feature_methods": {},
        }

        for feature in affected:
            values = result[feature].iloc[start:end].to_numpy(dtype=float)

            transformed, method = _set_exact_mean_and_std(
                values,
                multiplier,
                rng,
            )

            result.loc[result.index[start:end], feature] = transformed
            parameters["feature_methods"][feature] = method

        events.append(
            _create_event(
                seed=seed,
                scenario="variance_shift",
                number=number,
                start=start,
                end=end,
                severity="medium" if multiplier == 2.0 else "strong",
                affected_features=affected,
                parameters=parameters,
            )
        )

    return tuple(events)


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Calculate a finite correlation matrix."""
    if np.any(np.std(values, axis=0) == 0):
        raise ValueError(
            "Multivariate injection requires non-constant feature values"
        )

    matrix = np.corrcoef(values, rowvar=False)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Multivariate injection requires non-constant feature values"
        )

    return matrix


def _inject_multivariate(
    result: pd.DataFrame,
    rng: np.random.Generator,
    seed: int,
) -> tuple[AnomalyEvent, ...]:
    """Change feature dependencies while preserving exact marginals."""
    affected = (
        "latency_ms",
        "jitter_ms",
        "packet_loss_pct",
        "throughput_mbps",
    )

    permuted_features = (
        "jitter_ms",
        "packet_loss_pct",
        "throughput_mbps",
    )

    starts = _select_windows(
        rng,
        sample_count=len(result),
        event_count=4,
        event_length=10,
    )

    events = []

    for number, start in enumerate(starts, start=1):
        end = start + 10
        original = result.loc[
            result.index[start:end], affected
        ].to_numpy(dtype=float)

        original_correlation = _correlation_matrix(original)
        best_candidate = None
        best_distance = -1.0

        # Search deterministic candidate permutations and select
        # the one that changes the correlation structure most.
        for _ in range(128):
            candidate = original.copy()

            for feature_index, feature in enumerate(affected):
                if feature in permuted_features:
                    permutation = rng.permutation(len(candidate))
                    candidate[:, feature_index] = original[
                        permutation, feature_index
                    ]

            candidate_correlation = _correlation_matrix(candidate)
            distance = float(
                np.linalg.norm(
                    candidate_correlation - original_correlation,
                    ord="fro",
                )
            )

            if distance > best_distance:
                best_distance = distance
                best_candidate = candidate

        if best_candidate is None or best_distance < 0.15:
            raise ValueError(
                "Could not produce a sufficiently different correlation structure"
            )

        for feature_index, feature in enumerate(affected):
            if feature in permuted_features:
                result.loc[
                    result.index[start:end], feature
                ] = best_candidate[:, feature_index]

                original_values = np.sort(original[:, feature_index])
                injected_values = np.sort(best_candidate[:, feature_index])

                if not np.array_equal(original_values, injected_values):
                    raise ArithmeticError(
                        f"Permutation changed the marginal values of {feature}"
                    )

        events.append(
            _create_event(
                seed=seed,
                scenario="multivariate_anomaly",
                number=number,
                start=start,
                end=end,
                severity="medium",
                affected_features=affected,
                parameters={
                    "duration_samples": 10,
                    "method": "controlled_permutation",
                    "permuted_features": list(permuted_features),
                    "correlation_distance": best_distance,
                    "marginal_values_preserved": True,
                },
            )
        )

    return tuple(events)


def _validate_output(data: pd.DataFrame) -> None:
    """Ensure injected values remain physically valid."""
    for column in FEATURE_COLUMNS:
        values = data[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Injection produced invalid values in {column}")

    for column in NON_NEGATIVE_FEATURES:
        if (data[column] < 0.0).any():
            raise ValueError(f"Injection produced negative values in {column}")

    for column in PERCENTAGE_FEATURES:
        if not data[column].between(0.0, 100.0).all():
            raise ValueError(
                f"Injection produced out-of-range values in {column}"
            )


def inject_anomalies(
    evaluation_data: pd.DataFrame,
    *,
    scenario: str,
    seed: int,
) -> tuple[pd.DataFrame, tuple[AnomalyEvent, ...]]:
    """Inject one reproducible anomaly scenario into evaluation telemetry.

    The input DataFrame is never modified.

    Event intervals use zero-based, end-exclusive indices.
    Training and Calibration data must not be passed to this function.
    """
    _validate_input(evaluation_data, scenario, seed)

    result = evaluation_data.copy(deep=True)
    rng = _make_rng(seed, scenario)

    if scenario == "normal":
        events = ()

    elif scenario == "sudden_spike":
        events = _inject_spikes(result, rng, seed)

    elif scenario == "gradual_drift":
        events = _inject_gradual_drift(result, rng, seed)

    elif scenario == "burst_anomalies":
        events = _inject_bursts(result, rng, seed)

    elif scenario == "variance_shift":
        events = _inject_variance_shift(result, rng, seed)

    elif scenario == "multivariate_anomaly":
        events = _inject_multivariate(result, rng, seed)

    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    _validate_output(result)

    return result, events
