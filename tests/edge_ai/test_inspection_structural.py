
"""Tests for structural inspection of Edge AI experiment datasets."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.edge_ai.config import ExperimentConfig, SCENARIO_NAMES
from experiments.edge_ai.data.telemetry_generator import FEATURE_COLUMNS
from experiments.edge_ai.inspection.loader import DatasetLoader
from experiments.edge_ai.inspection.models import (
    InspectionStatus,
    StructuralInspectionReport,
)
from experiments.edge_ai.inspection.structural import StructuralInspector
from experiments.edge_ai.pipeline.dataset_builder import (
    EVENT_METADATA_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    MANIFEST_COLUMNS,
)


class TestStructuralInspector(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "datasets"

        self.config = ExperimentConfig(
            sampling_interval_seconds=30,
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
            freq=f"{self.config.sampling_interval_seconds}s",
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

        interval = self.config.sampling_interval_seconds

        for seed in self.config.seeds:
            seed_dir = self.root / f"seed_{seed}"
            seed_dir.mkdir(parents=True)

            training_start = pd.Timestamp("2025-01-01T00:00:00")
            calibration_start = (
                training_start
                + pd.Timedelta(
                    seconds=self.config.training_samples * interval
                )
            )
            evaluation_start = (
                calibration_start
                + pd.Timedelta(
                    seconds=self.config.calibration_samples * interval
                )
            )

            training = self._telemetry(
                training_start,
                self.config.training_samples,
            )
            calibration = self._telemetry(
                calibration_start,
                self.config.calibration_samples,
            )

            training.to_csv(seed_dir / "training.csv", index=False)
            calibration.to_csv(
                seed_dir / "calibration.csv",
                index=False,
            )

            events_dir = seed_dir / "metadata"
            events_dir.mkdir()

            pd.DataFrame(
                columns=EVENT_METADATA_COLUMNS
            ).to_csv(
                events_dir / "events.csv",
                index=False,
            )

            evaluation_dir = seed_dir / "evaluation"
            ground_truth_dir = seed_dir / "ground_truth"
            evaluation_dir.mkdir()
            ground_truth_dir.mkdir()

            for scenario in SCENARIO_NAMES:
                evaluation = self._telemetry(
                    evaluation_start,
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

                evaluation.to_csv(
                    evaluation_dir / f"{scenario}.csv",
                    index=False,
                )
                ground_truth.to_csv(
                    ground_truth_dir / f"{scenario}.csv",
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

    def _inspector(self):
        return StructuralInspector(
            self._loader(),
            experiment_config=self.config,
        )

    def _rewrite_timestamps(self, relative_path, timestamps):
        path = self.root / relative_path
        frame = pd.read_csv(path)
        frame["timestamp"] = timestamps
        frame.to_csv(path, index=False)

    def test_accepts_structurally_valid_dataset(self):
        report = self._inspector().inspect()

        self.assertIsInstance(report, StructuralInspectionReport)
        self.assertEqual(report.status, InspectionStatus.PASS)
        self.assertEqual(report.seeds_inspected, 2)
        self.assertEqual(report.datasets_inspected, 12)
        self.assertEqual(report.failed_checks, 0)

    def test_verifies_six_scenarios_per_seed(self):
        result = self._inspector().inspect_seed(1000)

        self.assertEqual(result.status, InspectionStatus.PASS)
        self.assertEqual(result.errors, ())

    def test_detects_invalid_training_interval(self):
        timestamps = pd.date_range(
            "2025-01-01T00:00:00",
            periods=4,
            freq="60s",
        )
        self._rewrite_timestamps(
            "seed_1000/training.csv",
            timestamps,
        )

        result = self._inspector().inspect_seed(1000)

        self.assertEqual(result.status, InspectionStatus.FAIL)
        self.assertTrue(any("STR-02" in error for error in result.errors))

    def test_detects_invalid_calibration_interval(self):
        timestamps = pd.date_range(
            "2025-01-01T02:00:00",
            periods=3,
            freq="60s",
        )
        self._rewrite_timestamps(
            "seed_1000/calibration.csv",
            timestamps,
        )

        result = self._inspector().inspect_seed(1000)

        self.assertEqual(result.status, InspectionStatus.FAIL)
        self.assertTrue(any("STR-03" in error for error in result.errors))

    def test_detects_invalid_evaluation_interval(self):
        timestamps = pd.date_range(
            "2025-01-01T04:00:00",
            periods=5,
            freq="60s",
        )
        self._rewrite_timestamps(
            "seed_1000/evaluation/normal.csv",
            timestamps,
        )
        self._rewrite_timestamps(
            "seed_1000/ground_truth/normal.csv",
            timestamps,
        )

        result = self._inspector().inspect_seed(1000)

        self.assertEqual(result.status, InspectionStatus.FAIL)
        self.assertTrue(any("STR-04" in error for error in result.errors))

    def test_detects_training_calibration_discontinuity(self):
        timestamps = pd.date_range(
            "2025-01-01T03:00:00",
            periods=3,
            freq="30s",
        )
        self._rewrite_timestamps(
            "seed_1000/calibration.csv",
            timestamps,
        )

        result = self._inspector().inspect_seed(1000)

        self.assertEqual(result.status, InspectionStatus.FAIL)
        self.assertTrue(any("STR-05" in error for error in result.errors))

    def test_detects_calibration_evaluation_discontinuity(self):
        timestamps = pd.date_range(
            "2025-01-01T05:00:00",
            periods=5,
            freq="30s",
        )
        self._rewrite_timestamps(
            "seed_1000/evaluation/normal.csv",
            timestamps,
        )
        self._rewrite_timestamps(
            "seed_1000/ground_truth/normal.csv",
            timestamps,
        )

        result = self._inspector().inspect_seed(1000)

        self.assertEqual(result.status, InspectionStatus.FAIL)
        self.assertTrue(any("STR-05" in error for error in result.errors))

    def test_detects_scenario_timestamp_mismatch(self):
        timestamps = pd.date_range(
            "2025-01-01T04:00:00",
            periods=5,
            freq="30s",
        )
        timestamps = timestamps + pd.Timedelta(seconds=30)

        self._rewrite_timestamps(
            "seed_1000/evaluation/sudden_spike.csv",
            timestamps,
        )
        self._rewrite_timestamps(
            "seed_1000/ground_truth/sudden_spike.csv",
            timestamps,
        )

        result = self._inspector().inspect_seed(1000)

        self.assertEqual(result.status, InspectionStatus.FAIL)
        self.assertTrue(any("STR-06" in error for error in result.errors))
    def test_reports_six_passed_checks_per_valid_seed(self):
        report = self._inspector().inspect()

        self.assertEqual(report.passed_checks, 12)
        self.assertEqual(report.warning_checks, 0)
        self.assertEqual(report.failed_checks, 0)

    def test_records_failed_seed(self):
        (self.root / "seed_1000/training.csv").unlink()

        report = self._inspector().inspect()

        self.assertEqual(report.status, InspectionStatus.FAIL)
        self.assertEqual(len(report.seed_results), 2)
        self.assertEqual(report.seed_results[0].seed, 1000)
        self.assertEqual(
            report.seed_results[0].status,
            InspectionStatus.FAIL,
        )

    def test_continues_after_one_seed_failure(self):
        (self.root / "seed_1000/training.csv").unlink()

        report = self._inspector().inspect()

        self.assertEqual(report.seeds_inspected, 2)
        self.assertEqual(report.seed_results[1].seed, 1001)
        self.assertEqual(
            report.seed_results[1].status,
            InspectionStatus.PASS,
        )

    def test_invalid_manifest_fails_overall_inspection(self):
        manifest_path = self.root / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        manifest = manifest.iloc[:-1]
        manifest.to_csv(manifest_path, index=False)

        report = self._inspector().inspect()

        self.assertEqual(report.status, InspectionStatus.FAIL)
        self.assertEqual(report.seeds_inspected, 0)
        self.assertTrue(report.errors)
