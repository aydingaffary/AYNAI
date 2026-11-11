"""Tests for the read-only Edge AI dataset loader."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments.edge_ai.config import (
    ExperimentConfig,
    SCENARIO_NAMES,
)
from experiments.edge_ai.data.telemetry_generator import FEATURE_COLUMNS
from experiments.edge_ai.inspection.loader import (
    DatasetLoadError,
    DatasetLoader,
)
from experiments.edge_ai.inspection.models import InspectionStatus
from experiments.edge_ai.pipeline.dataset_builder import (
    EVENT_METADATA_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    MANIFEST_COLUMNS,
)


class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "datasets"

        self.config = ExperimentConfig(
            training_samples=4,
            calibration_samples=3,
            evaluation_samples=5,
            seed_start=1000,
            seed_count=2,
        )

        self._build_fixture()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _telemetry(self, start, count):
        timestamps = pd.date_range(
            start=start,
            periods=count,
            freq="30s",
        )

        data = {"timestamp": timestamps}

        for index, feature in enumerate(FEATURE_COLUMNS):
            data[feature] = [
                float(index + sample + 1)
                for sample in range(count)
            ]

        data["packet_loss_pct"] = [1.0] * count
        data["cpu_usage_pct"] = [20.0] * count
        data["memory_usage_pct"] = [30.0] * count

        return pd.DataFrame(data)

    def _build_fixture(self):
        self.root.mkdir(parents=True)

        manifest_rows = []

        for seed in self.config.seeds:
            seed_dir = self.root / f"seed_{seed}"

            training = self._telemetry(
                "2025-01-01T00:00:00",
                self.config.training_samples,
            )
            calibration = self._telemetry(
                "2025-01-01T02:00:00",
                self.config.calibration_samples,
            )

            training.to_csv(
                seed_dir / "training.csv",
                index=False,
            ) if seed_dir.exists() else None

            seed_dir.mkdir(parents=True, exist_ok=True)

            training.to_csv(seed_dir / "training.csv", index=False)
            calibration.to_csv(
                seed_dir / "calibration.csv",
                index=False,
            )

            events_dir = seed_dir / "metadata"
            events_dir.mkdir(parents=True)

            pd.DataFrame(
                columns=EVENT_METADATA_COLUMNS
            ).to_csv(
                events_dir / "events.csv",
                index=False,
            )

            for scenario in SCENARIO_NAMES:
                evaluation = self._telemetry(
                    "2025-01-01T04:00:00",
                    self.config.evaluation_samples,
                )

                ground_truth = pd.DataFrame(
                    {
                        "timestamp": evaluation["timestamp"],
                        "seed": seed,
                        "scenario": scenario,
                        "phase": "Evaluation",
                        "is_anomaly": 0,
                        "event_id": None,
                        "anomaly_type": None,
                        "severity": None,
                    }
                )

                evaluation_dir = seed_dir / "evaluation"
                ground_truth_dir = seed_dir / "ground_truth"

                evaluation_dir.mkdir(exist_ok=True)
                ground_truth_dir.mkdir(exist_ok=True)

                evaluation_path = (
                    evaluation_dir / f"{scenario}.csv"
                )
                ground_truth_path = (
                    ground_truth_dir / f"{scenario}.csv"
                )

                evaluation.to_csv(evaluation_path, index=False)
                ground_truth.to_csv(
                    ground_truth_path,
                    index=False,
                )

                manifest_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "training_samples": self.config.training_samples,
                        "calibration_samples": self.config.calibration_samples,
                        "evaluation_samples": self.config.evaluation_samples,
                        "anomaly_events": 0,
                        "training_file": f"seed_{seed}/training.csv",
                        "calibration_file": f"seed_{seed}/calibration.csv",
                        "evaluation_file": (
                            f"seed_{seed}/evaluation/{scenario}.csv"
                        ),
                        "ground_truth_file": (
                            f"seed_{seed}/ground_truth/{scenario}.csv"
                        ),
                        "events_file": (
                            f"seed_{seed}/metadata/events.csv"
                        ),
                    }
                )

        pd.DataFrame(
            manifest_rows,
            columns=MANIFEST_COLUMNS,
        ).to_csv(self.root / "manifest.csv", index=False)

    def _loader(self):
        return DatasetLoader(
            self.root,
            experiment_config=self.config,
        )

    def test_validate_manifest_accepts_complete_dataset(self):
        result = self._loader().validate_manifest()

        self.assertEqual(result.status, InspectionStatus.PASS)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.expected_seeds, self.config.seeds)
        self.assertEqual(result.observed_seeds, self.config.seeds)
        self.assertEqual(result.dataset_count, 12)
        self.assertEqual(result.missing_seeds, ())
        self.assertEqual(result.missing_scenarios, ())
        self.assertEqual(result.duplicate_entries, ())

    def test_load_seed_dataset_returns_all_scenarios(self):
        dataset = self._loader().load_seed_dataset(1000)

        self.assertEqual(dataset.seed, 1000)
        self.assertEqual(len(dataset.training), 4)
        self.assertEqual(len(dataset.calibration), 3)
        self.assertEqual(len(dataset.events), 0)

        self.assertEqual(
            set(dataset.evaluation),
            set(SCENARIO_NAMES),
        )
        self.assertEqual(
            set(dataset.ground_truth),
            set(SCENARIO_NAMES),
        )

        for scenario in SCENARIO_NAMES:
            self.assertEqual(
                len(dataset.evaluation[scenario]),
                self.config.evaluation_samples,
            )
            self.assertEqual(
                len(dataset.ground_truth[scenario]),
                self.config.evaluation_samples,
            )

            self.assertNotIn(
                "is_anomaly",
                dataset.evaluation[scenario].columns,
            )

            self.assertIn(
                "is_anomaly",
                dataset.ground_truth[scenario].columns,
            )

    def test_iterator_loads_seed_datasets_lazily(self):
        loader = self._loader()
        original_read_csv = pd.read_csv
        loaded_paths = []

        def tracked_read_csv(path, *args, **kwargs):
            loaded_paths.append(str(path))
            return original_read_csv(path, *args, **kwargs)

        with patch(
            "experiments.edge_ai.inspection.loader.pd.read_csv",
            side_effect=tracked_read_csv,
        ):
            iterator = loader.iter_seed_datasets()

            first = next(iterator)

            self.assertEqual(first.seed, 1000)
            self.assertTrue(
                all("seed_1001" not in path for path in loaded_paths)
            )

            second = next(iterator)

            self.assertEqual(second.seed, 1001)

            self.assertTrue(
                any("seed_1001" in path for path in loaded_paths)
            )

    def test_missing_scenario_fails_manifest_validation(self):
        manifest_path = self.root / "manifest.csv"
        manifest = pd.read_csv(manifest_path)

        manifest = manifest[
            ~(
                (manifest["seed"] == 1000)
                & (manifest["scenario"] == "normal")
            )
        ]

        manifest.to_csv(manifest_path, index=False)

        result = self._loader().validate_manifest()

        self.assertEqual(result.status, InspectionStatus.FAIL)
        self.assertFalse(result.is_valid)
        self.assertIn(
            (1000, "normal"),
            result.missing_scenarios,
        )

    def test_duplicate_manifest_entry_fails_validation(self):
        manifest_path = self.root / "manifest.csv"
        manifest = pd.read_csv(manifest_path)

        duplicate = manifest.iloc[[0]]
        manifest = pd.concat(
            [manifest, duplicate],
            ignore_index=True,
        )
        manifest.to_csv(manifest_path, index=False)

        result = self._loader().validate_manifest()

        self.assertEqual(result.status, InspectionStatus.FAIL)
        self.assertIn(
            (1000, "normal"),
            result.duplicate_entries,
        )

    def test_path_traversal_fails_manifest_validation(self):
        manifest_path = self.root / "manifest.csv"
        manifest = pd.read_csv(manifest_path)

        manifest.loc[0, "training_file"] = "../outside.csv"
        manifest.to_csv(manifest_path, index=False)

        result = self._loader().validate_manifest()

        self.assertEqual(result.status, InspectionStatus.FAIL)

    def test_missing_dataset_file_fails_manifest_validation(self):
        path = self.root / "seed_1000" / "training.csv"
        path.unlink()

        result = self._loader().validate_manifest()

        self.assertEqual(result.status, InspectionStatus.FAIL)

    def test_invalid_telemetry_values_raise_load_error(self):
        path = self.root / "seed_1000" / "training.csv"
        data = pd.read_csv(path)

        data.loc[0, "latency_ms"] = float("nan")
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_out_of_range_feature_values_raise_load_error(self):
        path = self.root / "seed_1000" / "training.csv"
        data = pd.read_csv(path)

        data.loc[0, "packet_loss_pct"] = 101.0
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_ground_truth_timestamp_mismatch_raises_load_error(self):
        path = (
            self.root
            / "seed_1000"
            / "ground_truth"
            / "normal.csv"
        )

        data = pd.read_csv(path)
        data.loc[0, "timestamp"] = "2025-01-02T00:00:00"
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_invalid_event_json_raises_load_error(self):
        path = (
            self.root
            / "seed_1000"
            / "metadata"
            / "events.csv"
        )

        event = {
            "seed": 1000,
            "event_id": "test-event",
            "scenario": "sudden_spike",
            "start_timestamp": "2025-01-01T04:00:00",
            "end_timestamp": "2025-01-01T04:00:30",
            "start_index": 0,
            "end_index": 2,
            "severity": "weak",
            "affected_features": "[invalid-json",
            "injection_parameters": "{}",
        }

        pd.DataFrame(
            [event],
            columns=EVENT_METADATA_COLUMNS,
        ).to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)


    def test_infinite_telemetry_value_raises_load_error(self):
        path = self.root / "seed_1000" / "training.csv"
        data = pd.read_csv(path)
        data.loc[0, "latency_ms"] = float("inf")
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_duplicate_telemetry_timestamps_raise_load_error(self):
        path = self.root / "seed_1000" / "training.csv"
        data = pd.read_csv(path)
        data.loc[1, "timestamp"] = data.loc[0, "timestamp"]
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_non_monotonic_telemetry_timestamps_raise_load_error(self):
        path = self.root / "seed_1000" / "training.csv"
        data = pd.read_csv(path)
        data.loc[1, "timestamp"], data.loc[2, "timestamp"] = (
            data.loc[2, "timestamp"],
            data.loc[1, "timestamp"],
        )
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_ground_truth_wrong_seed_raises_load_error(self):
        path = (
            self.root / "seed_1000"
            / "ground_truth" / "normal.csv"
        )
        data = pd.read_csv(path)
        data.loc[0, "seed"] = 9999
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_ground_truth_wrong_scenario_raises_load_error(self):
        path = (
            self.root / "seed_1000"
            / "ground_truth" / "normal.csv"
        )
        data = pd.read_csv(path)
        data.loc[0, "scenario"] = "sudden_spike"
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_ground_truth_invalid_label_raises_load_error(self):
        path = (
            self.root / "seed_1000"
            / "ground_truth" / "normal.csv"
        )
        data = pd.read_csv(path)
        data.loc[0, "is_anomaly"] = 2
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_ground_truth_wrong_phase_raises_load_error(self):
        path = (
            self.root / "seed_1000"
            / "ground_truth" / "normal.csv"
        )
        data = pd.read_csv(path)
        data.loc[0, "phase"] = "Training"
        data.to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)

    def test_invalid_event_interval_raises_load_error(self):
        path = (
            self.root / "seed_1000"
            / "metadata" / "events.csv"
        )

        event = {
            "seed": 1000,
            "event_id": "test-event",
            "scenario": "sudden_spike",
            "start_timestamp": "2025-01-01T04:00:00",
            "end_timestamp": "2025-01-01T04:00:30",
            "start_index": -1,
            "end_index": 2,
            "severity": "weak",
            "affected_features": json.dumps(["latency_ms"]),
            "injection_parameters": json.dumps({"multiplier": 2.0}),
        }

        pd.DataFrame(
            [event],
            columns=EVENT_METADATA_COLUMNS,
        ).to_csv(path, index=False)

        with self.assertRaises(DatasetLoadError):
            self._loader().load_seed_dataset(1000)


if __name__ == "__main__":
    unittest.main()
