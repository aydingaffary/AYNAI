"""Read-only, lazy loader for Edge AI experiment datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from experiments.edge_ai.config import (
    DEFAULT_CONFIG,
    SCENARIO_NAMES,
    ExperimentConfig,
)
from experiments.edge_ai.data.telemetry_generator import FEATURE_COLUMNS
from experiments.edge_ai.inspection.models import (
    InspectionStatus,
    ManifestValidationResult,
    SeedDataset,
)
from experiments.edge_ai.pipeline.dataset_builder import (
    EVENT_METADATA_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    MANIFEST_COLUMNS,
)


class DatasetLoadError(RuntimeError):
    """Raised when a dataset cannot be safely loaded or validated."""


class DatasetLoader:
    """Validate and lazily load read-only experiment datasets."""

    def __init__(
        self,
        dataset_dir: str | Path,
        experiment_config: ExperimentConfig = DEFAULT_CONFIG,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.experiment_config = experiment_config
        self.experiment_config.validate()

    def _safe_path(
        self,
        relative_path: object,
        require_exists: bool = True,
    ) -> Path:
        """Resolve a manifest path without allowing root escape."""
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise DatasetLoadError("Manifest contains an empty file path")

        candidate = Path(relative_path)

        if candidate.is_absolute():
            raise DatasetLoadError(
                f"Absolute manifest paths are not allowed: {relative_path}"
            )

        resolved = (self.dataset_dir / candidate).resolve()

        if not resolved.is_relative_to(self.dataset_dir):
            raise DatasetLoadError(
                f"Manifest path escapes dataset directory: {relative_path}"
            )

        if require_exists and not resolved.is_file():
            raise DatasetLoadError(
                f"Dataset file does not exist: {relative_path}"
            )

        return resolved

    def _manifest_path(self) -> Path:
        path = self.dataset_dir / "manifest.csv"

        if not path.is_file():
            raise DatasetLoadError(
                f"Manifest file not found: {path}"
            )

        return path

    def validate_manifest(
        self,
        check_files: bool = True,
    ) -> ManifestValidationResult:
        """Validate manifest schema, coverage, paths, and sample counts."""
        expected_seeds = self.experiment_config.seeds
        errors: list[str] = []
        warnings: list[str] = []

        try:
            manifest = pd.read_csv(self._manifest_path())
        except Exception as exc:
            return ManifestValidationResult(
                status=InspectionStatus.FAIL,
                expected_seeds=expected_seeds,
                observed_seeds=(),
                dataset_count=0,
                errors=(f"Cannot read manifest: {exc}",),
            )

        if tuple(manifest.columns) != MANIFEST_COLUMNS:
            return ManifestValidationResult(
                status=InspectionStatus.FAIL,
                expected_seeds=expected_seeds,
                observed_seeds=(),
                dataset_count=len(manifest),
                errors=(
                    "Manifest schema mismatch: "
                    f"expected {MANIFEST_COLUMNS}, "
                    f"got {tuple(manifest.columns)}",
                ),
            )

        manifest["seed"] = pd.to_numeric(
            manifest["seed"], errors="coerce"
        )

        if manifest["seed"].isna().any():
            errors.append("Manifest contains invalid seed values")

        observed_seeds = tuple(
            sorted(
                int(value)
                for value in manifest["seed"].dropna().unique()
            )
        )

        expected_set = set(expected_seeds)
        observed_set = set(observed_seeds)

        missing_seeds = tuple(
            sorted(expected_set - observed_set)
        )
        unexpected_seeds = tuple(
            sorted(observed_set - expected_set)
        )

        if missing_seeds:
            errors.append(f"Missing seeds: {missing_seeds}")

        if unexpected_seeds:
            errors.append(f"Unexpected seeds: {unexpected_seeds}")

        valid_seed_rows = manifest["seed"].notna()
        duplicate_mask = manifest.loc[valid_seed_rows].duplicated(
            subset=["seed", "scenario"],
            keep=False,
        )

        duplicate_entries = tuple(
            sorted(
                {
                    (int(row.seed), str(row.scenario))
                    for row in manifest.loc[
                        valid_seed_rows
                    ].loc[duplicate_mask].itertuples()
                }
            )
        )

        if duplicate_entries:
            errors.append(
                f"Duplicate manifest entries: {duplicate_entries}"
            )

        missing_scenarios = tuple(
            (seed, scenario)
            for seed in expected_seeds
            for scenario in SCENARIO_NAMES
            if not (
                (manifest["seed"] == seed)
                & (manifest["scenario"] == scenario)
            ).any()
        )

        if missing_scenarios:
            errors.append(
                f"Missing seed/scenario entries: {missing_scenarios}"
            )

        allowed_scenarios = set(SCENARIO_NAMES)

        for row in manifest.itertuples(index=False):
            seed_value = row.seed

            if pd.isna(seed_value):
                continue

            seed = int(seed_value)
            scenario = row.scenario

            if scenario not in allowed_scenarios:
                errors.append(
                    f"Unsupported scenario for seed {seed}: {scenario}"
                )
                continue

            expected_counts = {
                "training_samples": self.experiment_config.training_samples,
                "calibration_samples": self.experiment_config.calibration_samples,
                "evaluation_samples": self.experiment_config.evaluation_samples,
            }

            for column, expected_count in expected_counts.items():
                value = getattr(row, column)

                try:
                    actual_count = int(value)
                except (TypeError, ValueError, OverflowError):
                    errors.append(
                        f"Invalid {column} for seed {seed}, {scenario}"
                    )
                    continue

                if actual_count != expected_count:
                    errors.append(
                        f"Unexpected {column} for seed {seed}, {scenario}: "
                        f"expected {expected_count}, got {actual_count}"
                    )

            try:
                event_count = int(row.anomaly_events)
                if event_count < 0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                errors.append(
                    f"Invalid anomaly_events for seed {seed}, {scenario}"
                )

            for column in (
                "training_file",
                "calibration_file",
                "evaluation_file",
                "ground_truth_file",
                "events_file",
            ):
                try:
                    self._safe_path(
                        getattr(row, column),
                        require_exists=check_files,
                    )
                except DatasetLoadError as exc:
                    errors.append(
                        f"Seed {seed}, scenario {scenario}, "
                        f"{column}: {exc}"
                    )

        shared_columns = (
            "training_file",
            "calibration_file",
            "events_file",
        )

        for seed in expected_seeds:
            rows = manifest.loc[manifest["seed"] == seed]

            for column in shared_columns:
                if rows[column].nunique(dropna=False) != 1:
                    errors.append(
                        f"Inconsistent {column} paths for seed {seed}"
                    )

        for column in shared_columns:
            paths_by_seed = (
                manifest.groupby("seed", dropna=True)[column]
                .first()
            )

            if paths_by_seed.duplicated().any():
                errors.append(
                    f"{column} must be unique across seeds"
                )

        status = (
            InspectionStatus.FAIL
            if errors
            else InspectionStatus.PASS
        )

        return ManifestValidationResult(
            status=status,
            expected_seeds=expected_seeds,
            observed_seeds=observed_seeds,
            dataset_count=len(manifest),
            missing_seeds=missing_seeds,
            unexpected_seeds=unexpected_seeds,
            missing_scenarios=missing_scenarios,
            duplicate_entries=duplicate_entries,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def _read_csv(self, relative_path: object) -> pd.DataFrame:
        """Read one CSV file after validating its path."""
        path = self._safe_path(relative_path)

        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise DatasetLoadError(
                f"Cannot read dataset file {relative_path}: {exc}"
            ) from exc

    @staticmethod
    def _validate_columns(
        data: pd.DataFrame,
        expected: tuple[str, ...],
        description: str,
    ) -> None:
        if tuple(data.columns) != expected:
            raise DatasetLoadError(
                f"{description} schema mismatch: "
                f"expected {expected}, got {tuple(data.columns)}"
            )

    @staticmethod
    def _parse_timestamps(
        data: pd.DataFrame,
        description: str,
    ) -> pd.Series:
        if data["timestamp"].isna().any():
            raise DatasetLoadError(
                f"{description} contains missing timestamps"
            )

        timestamps = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise DatasetLoadError(
                f"{description} contains invalid timestamps"
            )

        if timestamps.duplicated().any():
            raise DatasetLoadError(
                f"{description} contains duplicate timestamps"
            )

        if not timestamps.is_monotonic_increasing:
            raise DatasetLoadError(
                f"{description} timestamps are not increasing"
            )

        return timestamps

    def _validate_telemetry(
        self,
        data: pd.DataFrame,
        expected_samples: int,
        description: str,
    ) -> pd.DataFrame:
        expected_columns = ("timestamp", *FEATURE_COLUMNS)

        self._validate_columns(
            data,
            expected_columns,
            description,
        )

        if len(data) != expected_samples:
            raise DatasetLoadError(
                f"{description} sample count mismatch: "
                f"expected {expected_samples}, got {len(data)}"
            )

        timestamps = self._parse_timestamps(data, description)

        for feature in FEATURE_COLUMNS:
            values = pd.to_numeric(
                data[feature],
                errors="coerce",
            )

            if values.isna().any():
                raise DatasetLoadError(
                    f"{description} contains invalid values in {feature}"
                )

            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise DatasetLoadError(
                    f"{description} contains non-finite values in {feature}"
                )

            if feature in (
                "latency_ms",
                "jitter_ms",
                "throughput_mbps",
            ):
                if (values < 0).any():
                    raise DatasetLoadError(
                        f"{description}: {feature} must be non-negative"
                    )

            if feature in (
                "packet_loss_pct",
                "cpu_usage_pct",
                "memory_usage_pct",
            ):
                if ((values < 0) | (values > 100)).any():
                    raise DatasetLoadError(
                        f"{description}: {feature} must be between 0 and 100"
                    )

            data[feature] = values

        data["timestamp"] = timestamps
        return data

    def _validate_ground_truth(
        self,
        data: pd.DataFrame,
        evaluation: pd.DataFrame,
        seed: int,
        scenario: str,
    ) -> pd.DataFrame:
        description = f"Ground Truth for seed {seed}, scenario {scenario}"

        self._validate_columns(
            data,
            GROUND_TRUTH_COLUMNS,
            description,
        )

        if len(data) != self.experiment_config.evaluation_samples:
            raise DatasetLoadError(
                f"{description} sample count mismatch"
            )

        timestamps = self._parse_timestamps(data, description)

        if not timestamps.reset_index(drop=True).equals(
            evaluation["timestamp"].reset_index(drop=True)
        ):
            raise DatasetLoadError(
                f"{description} timestamps do not match Evaluation"
            )

        seeds = pd.to_numeric(data["seed"], errors="coerce")

        if seeds.isna().any() or not seeds.eq(seed).all():
            raise DatasetLoadError(
                f"{description} contains incorrect seed values"
            )

        if not data["scenario"].eq(scenario).all():
            raise DatasetLoadError(
                f"{description} contains incorrect scenario values"
            )

        if not data["phase"].eq("Evaluation").all():
            raise DatasetLoadError(
                f"{description} contains an unexpected phase"
            )

        labels = pd.to_numeric(
            data["is_anomaly"],
            errors="coerce",
        )

        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise DatasetLoadError(
                f"{description} contains invalid anomaly labels"
            )

        data["timestamp"] = timestamps
        data["seed"] = seeds.astype(int)
        data["is_anomaly"] = labels.astype(int)

        return data

    def _validate_events(
        self,
        events: pd.DataFrame,
        evaluation: dict[str, pd.DataFrame],
        ground_truth: dict[str, pd.DataFrame],
        seed: int,
    ) -> pd.DataFrame:
        description = f"Events for seed {seed}"

        self._validate_columns(
            events,
            EVENT_METADATA_COLUMNS,
            description,
        )

        if events.empty:
            return events

        seeds = pd.to_numeric(events["seed"], errors="coerce")

        if seeds.isna().any() or not seeds.eq(seed).all():
            raise DatasetLoadError(
                f"{description} contains incorrect seed values"
            )

        if events["event_id"].isna().any():
            raise DatasetLoadError(
                f"{description} contains missing event IDs"
            )

        if events["event_id"].duplicated().any():
            raise DatasetLoadError(
                f"{description} contains duplicate event IDs"
            )

        for event in events.itertuples(index=False):
            scenario = event.scenario

            if scenario not in SCENARIO_NAMES:
                raise DatasetLoadError(
                    f"{description} contains unsupported scenario: {scenario}"
                )

            try:
                affected_features = json.loads(event.affected_features)
                injection_parameters = json.loads(
                    event.injection_parameters
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DatasetLoadError(
                    f"{description} contains invalid event JSON"
                ) from exc

            if (
                not isinstance(affected_features, list)
                or not affected_features
                or not set(affected_features).issubset(FEATURE_COLUMNS)
            ):
                raise DatasetLoadError(
                    f"{description} contains invalid affected_features"
                )

            if not isinstance(injection_parameters, dict):
                raise DatasetLoadError(
                    f"{description} injection_parameters must be a JSON object"
                )

            try:
                start = int(event.start_index)
                end = int(event.end_index)
            except (TypeError, ValueError, OverflowError) as exc:
                raise DatasetLoadError(
                    f"{description} contains invalid event indices"
                ) from exc

            sample_count = self.experiment_config.evaluation_samples

            if start < 0 or end <= start or end > sample_count:
                raise DatasetLoadError(
                    f"{description} contains an invalid event interval"
                )

            feature_data = evaluation[scenario]
            truth_data = ground_truth[scenario]

            expected_start = feature_data["timestamp"].iloc[start]
            expected_end = feature_data["timestamp"].iloc[end - 1]

            actual_start = pd.to_datetime(
                event.start_timestamp,
                errors="coerce",
            )
            actual_end = pd.to_datetime(
                event.end_timestamp,
                errors="coerce",
            )

            if (
                pd.isna(actual_start)
                or pd.isna(actual_end)
                or actual_start != expected_start
                or actual_end != expected_end
            ):
                raise DatasetLoadError(
                    f"{description} event timestamps do not match its interval"
                )

            matching_ids = truth_data["event_id"].eq(event.event_id)
            expected_ids = pd.Series(
                False,
                index=truth_data.index,
            )
            expected_ids.iloc[start:end] = True

            if not matching_ids.equals(expected_ids):
                raise DatasetLoadError(
                    f"{description} event IDs do not match Ground Truth"
                )

            if not truth_data.loc[
                matching_ids, "is_anomaly"
            ].eq(1).all():
                raise DatasetLoadError(
                    f"{description} event contains non-anomalous labels"
                )

        return events

    def _load_seed(
        self,
        manifest: pd.DataFrame,
        seed: int,
    ) -> SeedDataset:
        rows = manifest.loc[manifest["seed"] == seed]

        if rows.empty:
            raise DatasetLoadError(f"Seed {seed} is not in the manifest")

        first = rows.iloc[0]

        training = self._validate_telemetry(
            self._read_csv(first["training_file"]),
            self.experiment_config.training_samples,
            f"Training for seed {seed}",
        )

        calibration = self._validate_telemetry(
            self._read_csv(first["calibration_file"]),
            self.experiment_config.calibration_samples,
            f"Calibration for seed {seed}",
        )

        evaluation: dict[str, pd.DataFrame] = {}
        ground_truth: dict[str, pd.DataFrame] = {}

        for row in rows.itertuples(index=False):
            scenario = row.scenario

            features = self._validate_telemetry(
                self._read_csv(row.evaluation_file),
                self.experiment_config.evaluation_samples,
                f"Evaluation for seed {seed}, scenario {scenario}",
            )

            truth = self._validate_ground_truth(
                self._read_csv(row.ground_truth_file),
                features,
                seed,
                scenario,
            )

            evaluation[scenario] = features
            ground_truth[scenario] = truth

        events = self._read_csv(first["events_file"])

        events = self._validate_events(
            events,
            evaluation,
            ground_truth,
            seed,
        )

        event_counts = (
            events.groupby("scenario").size().to_dict()
            if not events.empty
            else {}
        )

        for row in rows.itertuples(index=False):
            actual_count = int(event_counts.get(row.scenario, 0))

            if actual_count != int(row.anomaly_events):
                raise DatasetLoadError(
                    f"Event count mismatch for seed {seed}, "
                    f"scenario {row.scenario}"
                )

        return SeedDataset(
            seed=seed,
            training=training,
            calibration=calibration,
            evaluation=evaluation,
            ground_truth=ground_truth,
            events=events,
        )

    def load_seed_dataset(
        self,
        seed: int,
        check_manifest_files: bool = True,
    ) -> SeedDataset:
        """Validate the manifest and load one complete seed."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        result = self.validate_manifest(
            check_files=check_manifest_files
        )

        if not result.is_valid:
            raise DatasetLoadError(
                "Manifest validation failed: "
                + "; ".join(result.errors)
            )

        if seed not in result.expected_seeds:
            raise DatasetLoadError(
                f"Seed {seed} is not expected by the experiment configuration"
            )

        manifest = pd.read_csv(self._manifest_path())
        return self._load_seed(manifest, seed)

    def iter_seed_datasets(self) -> Iterator[SeedDataset]:
        """Yield validated seed datasets one at a time."""
        result = self.validate_manifest()

        if not result.is_valid:
            raise DatasetLoadError(
                "Manifest validation failed: "
                + "; ".join(result.errors)
            )

        manifest = pd.read_csv(self._manifest_path())

        for seed in result.expected_seeds:
            yield self._load_seed(manifest, seed)
