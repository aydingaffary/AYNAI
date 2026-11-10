"""Tests for the reproducible Edge AI dataset pipeline."""

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
from experiments.edge_ai.pipeline import build_datasets
from experiments.edge_ai.pipeline.dataset_builder import (
    GROUND_TRUTH_COLUMNS,
)
from experiments.edge_ai.simulation.anomaly_injector import inject_anomalies


class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_dir = self.root / "datasets"
        self.config = ExperimentConfig(seed_count=1)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_report_and_manifest(self):
        report = build_datasets(
            self.output_dir,
            config=self.config,
        )

        self.assertEqual(report.seeds_processed, 1)
        self.assertEqual(report.evaluation_datasets, 6)
        self.assertEqual(report.total_samples_written, 18240)
        self.assertGreater(report.total_events, 0)
        self.assertTrue(report.output_dir.exists())

        manifest = pd.read_csv(self.output_dir / "manifest.csv")

        self.assertEqual(len(manifest), len(SCENARIO_NAMES))
        self.assertEqual(
            manifest["scenario"].tolist(),
            list(SCENARIO_NAMES),
        )

        self.assertEqual(
            manifest["training_file"].nunique(),
            1,
        )
        self.assertEqual(
            manifest["calibration_file"].nunique(),
            1,
        )

        self.assertEqual(
            int(manifest["anomaly_events"].sum()),
            report.total_events,
        )

    def test_dataset_structure_and_schema(self):
        build_datasets(
            self.output_dir,
            config=self.config,
        )

        seed_dir = self.output_dir / "seed_1000"

        self.assertTrue((seed_dir / "training.csv").is_file())
        self.assertTrue((seed_dir / "calibration.csv").is_file())
        self.assertTrue((seed_dir / "metadata" / "events.csv").is_file())

        training = pd.read_csv(seed_dir / "training.csv")
        calibration = pd.read_csv(seed_dir / "calibration.csv")

        expected_feature_columns = ["timestamp", *FEATURE_COLUMNS]

        self.assertEqual(
            training.columns.tolist(),
            expected_feature_columns,
        )
        self.assertEqual(
            calibration.columns.tolist(),
            expected_feature_columns,
        )

        self.assertEqual(len(training), self.config.training_samples)
        self.assertEqual(len(calibration), self.config.calibration_samples)

        for scenario in SCENARIO_NAMES:
            feature_path = (
                seed_dir / "evaluation" / f"{scenario}.csv"
            )
            ground_truth_path = (
                seed_dir / "ground_truth" / f"{scenario}.csv"
            )

            self.assertTrue(feature_path.is_file())
            self.assertTrue(ground_truth_path.is_file())

            features = pd.read_csv(feature_path)
            ground_truth = pd.read_csv(ground_truth_path)

            self.assertEqual(
                features.columns.tolist(),
                expected_feature_columns,
            )
            self.assertEqual(
                ground_truth.columns.tolist(),
                list(GROUND_TRUTH_COLUMNS),
            )

            self.assertEqual(
                len(features),
                self.config.evaluation_samples,
            )
            self.assertEqual(
                len(ground_truth),
                self.config.evaluation_samples,
            )

            self.assertTrue(
                features["timestamp"].equals(
                    ground_truth["timestamp"]
                )
            )

            self.assertNotIn("is_anomaly", features.columns)
            self.assertIn("is_anomaly", ground_truth.columns)

    def test_ground_truth_labels_match_scenarios(self):
        build_datasets(
            self.output_dir,
            config=self.config,
        )

        seed_dir = self.output_dir / "seed_1000"

        for scenario in SCENARIO_NAMES:
            ground_truth = pd.read_csv(
                seed_dir / "ground_truth" / f"{scenario}.csv"
            )

            anomaly_count = int(ground_truth["is_anomaly"].sum())

            if scenario == "normal":
                self.assertEqual(anomaly_count, 0)
            else:
                self.assertGreater(anomaly_count, 0)

            self.assertTrue(
                ground_truth["is_anomaly"].isin([0, 1]).all()
            )

    def test_event_metadata_contains_valid_json(self):
        build_datasets(
            self.output_dir,
            config=self.config,
        )

        events = pd.read_csv(
            self.output_dir
            / "seed_1000"
            / "metadata"
            / "events.csv"
        )

        self.assertGreater(len(events), 0)

        for _, event in events.iterrows():
            affected_features = json.loads(
                event["affected_features"]
            )
            injection_parameters = json.loads(
                event["injection_parameters"]
            )

            self.assertIsInstance(affected_features, list)
            self.assertIsInstance(injection_parameters, dict)

    def test_evaluation_datasets_are_deterministic(self):
        first = self.root / "first"
        second = self.root / "second"

        build_datasets(first, config=self.config)
        build_datasets(second, config=self.config)

        for scenario in SCENARIO_NAMES:
            path_a = (
                first / "seed_1000" / "evaluation" / f"{scenario}.csv"
            )
            path_b = (
                second / "seed_1000" / "evaluation" / f"{scenario}.csv"
            )

            self.assertEqual(path_a.read_bytes(), path_b.read_bytes())

    def test_existing_output_is_not_overwritten_by_default(self):
        build_datasets(
            self.output_dir,
            config=self.config,
        )

        manifest_path = self.output_dir / "manifest.csv"
        original_manifest = manifest_path.read_bytes()

        with self.assertRaises(FileExistsError):
            build_datasets(
                self.output_dir,
                config=self.config,
            )

        self.assertEqual(
            manifest_path.read_bytes(),
            original_manifest,
        )

    def test_failed_build_preserves_existing_output(self):
        self.output_dir.mkdir()
        marker = self.output_dir / "important.txt"
        marker.write_text("preserve this output", encoding="utf-8")

        with patch(
            "experiments.edge_ai.pipeline.dataset_builder.inject_anomalies",
            side_effect=RuntimeError("Simulated dataset generation failure"),
        ):
            with self.assertRaises(RuntimeError):
                build_datasets(
                    self.output_dir,
                    config=self.config,
                    overwrite=True,
                )

        self.assertTrue(marker.is_file())
        self.assertEqual(
            marker.read_text(encoding="utf-8"),
            "preserve this output",
        )

        self.assertFalse((self.output_dir / "manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
