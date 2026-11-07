"""Ground-truth labels and event metadata for Edge AI simulations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd


VALID_PHASES = ("Training", "Calibration", "Evaluation")


@dataclass(frozen=True)
class AnomalyEvent:
    """An anomaly event using a zero-based, end-exclusive index interval."""

    event_id: str
    start_index: int
    end_index: int
    anomaly_type: str
    severity: str
    affected_features: tuple[str, ...]
    injection_parameters: dict[str, Any]


def _validate_event(event: AnomalyEvent, sample_count: int) -> None:
    """Validate one event against the evaluation dataset."""
    if not isinstance(event, AnomalyEvent):
        raise TypeError("Every event must be an AnomalyEvent instance")

    if not isinstance(event.event_id, str) or not event.event_id.strip():
        raise ValueError("event_id must be a non-empty string")

    for name, value in (
        ("start_index", event.start_index),
        ("end_index", event.end_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")

    if event.start_index < 0:
        raise ValueError("start_index must be non-negative")

    if event.end_index <= event.start_index:
        raise ValueError("end_index must be greater than start_index")

    if event.end_index > sample_count:
        raise ValueError("event end_index exceeds the dataset length")

    if not isinstance(event.anomaly_type, str) or not event.anomaly_type.strip():
        raise ValueError("anomaly_type must be a non-empty string")

    if not isinstance(event.severity, str) or not event.severity.strip():
        raise ValueError("severity must be a non-empty string")

    if not isinstance(event.affected_features, tuple) or not event.affected_features:
        raise ValueError("affected_features must be a non-empty tuple")

    if any(
        not isinstance(feature, str) or not feature.strip()
        for feature in event.affected_features
    ):
        raise ValueError("affected_features must contain non-empty strings")

    if len(set(event.affected_features)) != len(event.affected_features):
        raise ValueError("affected_features must not contain duplicates")

    if not isinstance(event.injection_parameters, dict):
        raise ValueError("injection_parameters must be a dictionary")


def build_ground_truth(
    data: pd.DataFrame,
    *,
    seed: int,
    scenario: str,
    phase: str,
    events: tuple[AnomalyEvent, ...] = (),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Attach reproducible sample-level labels and event metadata.

    Event indices are relative to the supplied DataFrame.
    Intervals use [start_index, end_index), so end_index is excluded.

    Returns:
        A copy of the input DataFrame with Ground Truth columns,
        and a list containing one metadata record per event.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if data.empty:
        raise ValueError("data must not be empty")

    if "timestamp" not in data.columns:
        raise ValueError("data must contain a timestamp column")

    if data["timestamp"].isna().any():
        raise ValueError("timestamps must not contain missing values")

    timestamps = pd.to_datetime(data["timestamp"], errors="raise")

    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and increasing")

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario must be a non-empty string")
    if not isinstance(phase, str) or phase not in VALID_PHASES:
        raise ValueError(f"phase must be one of {VALID_PHASES}")

    if not isinstance(events, tuple):
        raise TypeError("events must be provided as a tuple")

    if phase != "Evaluation" and events:
        raise ValueError("Training and Calibration must not contain anomalies")

    # Validate event types and fields before accessing event attributes.
    for event in events:
        _validate_event(event, len(data))

    event_ids = [event.event_id for event in events]

    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event_id values must be unique")

    available_features = set(data.columns) - {"timestamp"}

    for event in events:
        unknown_features = set(event.affected_features) - available_features

        if unknown_features:
            raise ValueError(
                f"Unknown affected features: {sorted(unknown_features)}"
            )

    ordered_events = sorted(events, key=lambda event: event.start_index)

    for previous, current in zip(ordered_events, ordered_events[1:]):
        if current.start_index < previous.end_index:
            raise ValueError("Anomaly events must not overlap")

    result = data.copy()
    result["timestamp"] = timestamps
    result["seed"] = seed
    result["scenario"] = scenario
    result["phase"] = phase
    result["is_anomaly"] = 0
    result["event_id"] = None
    result["anomaly_type"] = None
    result["severity"] = None

    event_metadata: list[dict[str, Any]] = []

    for event in ordered_events:
        start = event.start_index
        end = event.end_index

        result.iloc[
            start:end, result.columns.get_loc("is_anomaly")
        ] = 1

        result.iloc[
            start:end, result.columns.get_loc("event_id")
        ] = event.event_id

        result.iloc[
            start:end, result.columns.get_loc("anomaly_type")
        ] = event.anomaly_type

        result.iloc[
            start:end, result.columns.get_loc("severity")
        ] = event.severity

        event_metadata.append(
            {
                "event_id": event.event_id,
                "scenario": scenario,
                "start_timestamp": timestamps.iloc[start],
                "end_timestamp": timestamps.iloc[end - 1],
                "start_index": start,
                "end_index": end,
                "severity": event.severity,
                "affected_features": list(event.affected_features),
                "injection_parameters": deepcopy(event.injection_parameters),
            }
        )

    return result, event_metadata
