"""Build reproducible Edge AI datasets and save them as CSV files."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.edge_ai.config import (
    DEFAULT_BASELINE,
    DEFAULT_CONFIG,
    SCENARIO_NAMES,
    ExperimentConfig,
    TelemetryBaseline,
)
from experiments.edge_ai.data.telemetry_generator import (
    FEATURE_COLUMNS,
    generate_telemetry,
)
from experiments.edge_ai.simulation.anomaly_injector import (
    inject_anomalies,
)
from experiments.edge_ai.simulation.ground_truth import (
    build_ground_truth,
)


GROUND_TRUTH_COLUMNS = (
    "timestamp",
    "seed",
    "scenario",
    "phase",
    "is_anomaly",
    "event_id",
    "anomaly_type",
    "severity",
)

EVENT_METADATA_COLUMNS = (
    "seed",
    "event_id",
    "scenario",
    "start_timestamp",
    "end_timestamp",
    "start_index",
    "end_index",
    "severity",
    "affected_features",
    "injection_parameters",
)

MANIFEST_COLUMNS = (
    "seed",
    "scenario",
    "training_samples",
    "calibration_samples",
    "evaluation_samples",
    "anomaly_events",
    "training_file",
    "calibration_file",
    "evaluation_file",
    "ground_truth_file",
    "events_file",
)


@dataclass(frozen=True)
class DatasetBuildReport:
    """Summary of a successfully completed dataset build."""

    output_dir: Path
    seeds_processed: int
    evaluation_datasets: int
    total_samples_written: int
    total_events: int


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV without its pandas index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        na_rep="",
    )


def _serialize_event_metadata(
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Convert event metadata into CSV-compatible scalar values."""
    serialized = []

    for record in records:
        item = dict(record)
        item["seed"] = seed

        for column in ("start_timestamp", "end_timestamp"):
            item[column] = pd.Timestamp(item[column]).isoformat()

        item["affected_features"] = json.dumps(
            item["affected_features"],
            separators=(",", ":"),
        )
        item["injection_parameters"] = json.dumps(
            item["injection_parameters"],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

        serialized.append(item)

    return serialized


def _validate_generated_data(
    data: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Validate telemetry generator output before splitting it."""
    expected_columns = {"timestamp", *FEATURE_COLUMNS}

    if len(data) != config.total_samples:
        raise ValueError(
            "Generated telemetry has an unexpected sample count: "
            f"expected {config.total_samples}, got {len(data)}"
        )

    if set(data.columns) != expected_columns:
        raise ValueError(
            "Generated telemetry has an unexpected column schema"
        )

    if data["timestamp"].isna().any():
        raise ValueError("Generated timestamps contain missing values")

    timestamps = pd.to_datetime(data["timestamp"], errors="raise")

    if timestamps.duplicated().any():
        raise ValueError("Generated timestamps are not unique")

    if not timestamps.is_monotonic_increasing:
        raise ValueError("Generated timestamps are not increasing")


def _build_seed(
    staging_dir: Path,
    seed: int,
    config: ExperimentConfig,
    baseline: TelemetryBaseline,
) -> tuple[list[dict[str, Any]], int, int]:
    """Generate and save all datasets belonging to one seed."""
    seed_dir = staging_dir / f"seed_{seed}"
    evaluation_dir = seed_dir / "evaluation"
    ground_truth_dir = seed_dir / "ground_truth"
    metadata_dir = seed_dir / "metadata"

    generated = generate_telemetry(
        seed=seed,
        config=config,
        baseline=baseline,
    )
    _validate_generated_data(generated, config)

    training_end = config.training_samples
    calibration_end = training_end + config.calibration_samples

    training = (
        generated.iloc[:training_end]
        .copy()
        .reset_index(drop=True)
    )
    calibration = (
        generated.iloc[training_end:calibration_end]
        .copy()
        .reset_index(drop=True)
    )
    evaluation_base = (
        generated.iloc[calibration_end:]
        .copy()
        .reset_index(drop=True)
    )

    if len(training) != config.training_samples:
        raise ValueError(f"Invalid Training sample count for seed {seed}")

    if len(calibration) != config.calibration_samples:
        raise ValueError(f"Invalid Calibration sample count for seed {seed}")

    if len(evaluation_base) != config.evaluation_samples:
        raise ValueError(f"Invalid Evaluation sample count for seed {seed}")

    training_path = seed_dir / "training.csv"
    calibration_path = seed_dir / "calibration.csv"

    _write_csv(training, training_path)
    _write_csv(calibration, calibration_path)

    event_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    total_events = 0

    for scenario in SCENARIO_NAMES:
        evaluation, events = inject_anomalies(
            evaluation_base.copy(deep=True),
            scenario=scenario,
            seed=seed,
        )

        labeled, scenario_metadata = build_ground_truth(
            evaluation,
            seed=seed,
            scenario=scenario,
            phase="Evaluation",
            events=events,
        )

        feature_data = evaluation.loc[
            :, ["timestamp", *FEATURE_COLUMNS]
        ].copy()

        ground_truth = labeled.loc[
            :, list(GROUND_TRUTH_COLUMNS)
        ].copy()

        if len(feature_data) != config.evaluation_samples:
            raise ValueError(
                f"Invalid Evaluation output for seed {seed}, "
                f"scenario {scenario}"
            )

        if len(ground_truth) != config.evaluation_samples:
            raise ValueError(
                f"Invalid Ground Truth output for seed {seed}, "
                f"scenario {scenario}"
            )

        expected_anomaly_samples = sum(
            event.end_index - event.start_index
            for event in events
        )

        if int(ground_truth["is_anomaly"].sum()) != expected_anomaly_samples:
            raise ValueError(
                f"Ground Truth labels do not match events for "
                f"seed {seed}, scenario {scenario}"
            )

        evaluation_path = evaluation_dir / f"{scenario}.csv"
        ground_truth_path = ground_truth_dir / f"{scenario}.csv"

        _write_csv(feature_data, evaluation_path)
        _write_csv(ground_truth, ground_truth_path)

        event_records.extend(
            _serialize_event_metadata(scenario_metadata, seed)
        )
        total_events += len(events)

        manifest_records.append(
            {
                "seed": seed,
                "scenario": scenario,
                "training_samples": len(training),
                "calibration_samples": len(calibration),
                "evaluation_samples": len(feature_data),
                "anomaly_events": len(events),
                "training_file": training_path.relative_to(
                    staging_dir
                ).as_posix(),
                "calibration_file": calibration_path.relative_to(
                    staging_dir
                ).as_posix(),
                "evaluation_file": evaluation_path.relative_to(
                    staging_dir
                ).as_posix(),
                "ground_truth_file": ground_truth_path.relative_to(
                    staging_dir
                ).as_posix(),
                "events_file": (
                    seed_dir / "metadata" / "events.csv"
                ).relative_to(staging_dir).as_posix(),
            }
        )

    events_path = metadata_dir / "events.csv"
    events_frame = pd.DataFrame(
        event_records,
        columns=EVENT_METADATA_COLUMNS,
    )
    _write_csv(events_frame, events_path)

    samples_written = (
        len(training)
        + len(calibration)
        + len(SCENARIO_NAMES) * config.evaluation_samples
    )

    return manifest_records, total_events, samples_written


def build_datasets(
    output_dir: str | Path,
    config: ExperimentConfig = DEFAULT_CONFIG,
    baseline: TelemetryBaseline = DEFAULT_BASELINE,
    overwrite: bool = False,
) -> DatasetBuildReport:
    """Build all configured seeds and scenarios as CSV datasets.

    The final output directory is published only after the complete build
    succeeds. Existing output is protected unless overwrite=True.
    """
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be a string or pathlib.Path")

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")

    if not isinstance(baseline, TelemetryBaseline):
        raise TypeError("baseline must be a TelemetryBaseline")

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")

    config.validate()

    target = Path(output_dir).expanduser().resolve()

    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {target}. "
            "Use overwrite=True to replace it."
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    total_events = 0
    total_samples_written = 0

    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory) / target.name
        staging_dir.mkdir()

        for seed in config.seeds:
            records, seed_events, seed_samples = _build_seed(
                staging_dir=staging_dir,
                seed=seed,
                config=config,
                baseline=baseline,
            )
            manifest_records.extend(records)
            total_events += seed_events
            total_samples_written += seed_samples

        manifest = pd.DataFrame(
            manifest_records,
            columns=MANIFEST_COLUMNS,
        )

        expected_datasets = len(config.seeds) * len(SCENARIO_NAMES)

        if len(manifest) != expected_datasets:
            raise ValueError(
                "Unexpected number of Evaluation datasets: "
                f"expected {expected_datasets}, got {len(manifest)}"
            )

        _write_csv(manifest, staging_dir / "manifest.csv")

        backup_dir = Path(temporary_directory) / "previous_output"
        backup_created = False

        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory appeared during build: {target}"
                )

            os.replace(target, backup_dir)
            backup_created = True

        try:
            os.replace(staging_dir, target)
        except Exception:
            if backup_created and backup_dir.exists():
                os.replace(backup_dir, target)
            raise

    return DatasetBuildReport(
        output_dir=target,
        seeds_processed=len(config.seeds),
        evaluation_datasets=len(config.seeds) * len(SCENARIO_NAMES),
        total_samples_written=total_samples_written,
        total_events=total_events,
    )
