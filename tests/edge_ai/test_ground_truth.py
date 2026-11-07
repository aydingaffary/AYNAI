"""Tests for sample-level Ground Truth and event metadata."""

import unittest

import pandas as pd

from experiments.edge_ai.simulation.ground_truth import (
    AnomalyEvent,
    build_ground_truth,
)


class TestGroundTruth(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-01-01",
                    periods=20,
                    freq="30s",
                ),
                "latency_ms": [50.0] * 20,
            }
        )

        self.event = AnomalyEvent(
            event_id="seed1000-event001",
            start_index=3,
            end_index=6,
            anomaly_type="sudden_spike",
            severity="medium",
            affected_features=("latency_ms",),
            injection_parameters={"multiplier": 3.0},
        )

    def test_normal_data_has_no_anomaly_labels(self):
        result, metadata = build_ground_truth(
            self.data,
            seed=1000,
            scenario="normal",
            phase="Evaluation",
        )

        self.assertEqual(result["is_anomaly"].sum(), 0)
        self.assertTrue(result["event_id"].isna().all())
        self.assertTrue(result["anomaly_type"].isna().all())
        self.assertTrue(result["severity"].isna().all())
        self.assertEqual(metadata, [])

    def test_event_labels_use_end_exclusive_interval(self):
        result, metadata = build_ground_truth(
            self.data,
            seed=1000,
            scenario="sudden_spike",
            phase="Evaluation",
            events=(self.event,),
        )

        self.assertEqual(
            result["is_anomaly"].tolist(),
            [0, 0, 0, 1, 1, 1] + [0] * 14,
        )

        self.assertEqual(
            result.loc[3:5, "event_id"].tolist(),
            ["seed1000-event001"] * 3,
        )

        self.assertTrue(result.loc[6:, "event_id"].isna().all())
        self.assertEqual(len(metadata), 1)

    def test_event_metadata_matches_sample_interval(self):
        _, metadata = build_ground_truth(
            self.data,
            seed=1000,
            scenario="sudden_spike",
            phase="Evaluation",
            events=(self.event,),
        )

        record = metadata[0]

        self.assertEqual(record["event_id"], self.event.event_id)
        self.assertEqual(record["scenario"], "sudden_spike")
        self.assertEqual(record["start_index"], 3)
        self.assertEqual(record["end_index"], 6)
        self.assertEqual(record["start_timestamp"], self.data["timestamp"].iloc[3])
        self.assertEqual(record["end_timestamp"], self.data["timestamp"].iloc[5])
        self.assertEqual(record["severity"], "medium")
        self.assertEqual(record["affected_features"], ["latency_ms"])
        self.assertEqual(
            record["injection_parameters"],
            {"multiplier": 3.0},
        )

    def test_input_dataframe_is_not_modified(self):
        original = self.data.copy(deep=True)

        build_ground_truth(
            self.data,
            seed=1000,
            scenario="normal",
            phase="Evaluation",
        )

        pd.testing.assert_frame_equal(self.data, original)

    def test_training_cannot_contain_anomalies(self):
        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="sudden_spike",
                phase="Training",
                events=(self.event,),
            )

    def test_calibration_cannot_contain_anomalies(self):
        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="sudden_spike",
                phase="Calibration",
                events=(self.event,),
            )

    def test_overlapping_events_are_rejected(self):
        second_event = AnomalyEvent(
            event_id="event002",
            start_index=5,
            end_index=8,
            anomaly_type="sudden_spike",
            severity="strong",
            affected_features=("latency_ms",),
            injection_parameters={"multiplier": 5.0},
        )

        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="sudden_spike",
                phase="Evaluation",
                events=(self.event, second_event),
            )

    def test_adjacent_events_are_allowed(self):
        second_event = AnomalyEvent(
            event_id="event002",
            start_index=6,
            end_index=8,
            anomaly_type="sudden_spike",
            severity="strong",
            affected_features=("latency_ms",),
            injection_parameters={"multiplier": 5.0},
        )

        result, metadata = build_ground_truth(
            self.data,
            seed=1000,
            scenario="sudden_spike",
            phase="Evaluation",
            events=(self.event, second_event),
        )

        self.assertEqual(result["is_anomaly"].sum(), 5)
        self.assertEqual(len(metadata), 2)

    def test_duplicate_event_ids_are_rejected(self):
        duplicate = AnomalyEvent(
            event_id=self.event.event_id,
            start_index=8,
            end_index=10,
            anomaly_type="sudden_spike",
            severity="weak",
            affected_features=("latency_ms",),
            injection_parameters={"multiplier": 2.0},
        )

        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="sudden_spike",
                phase="Evaluation",
                events=(self.event, duplicate),
            )

    def test_invalid_event_bounds_are_rejected(self):
        invalid_event = AnomalyEvent(
            event_id="invalid",
            start_index=18,
            end_index=21,
            anomaly_type="sudden_spike",
            severity="strong",
            affected_features=("latency_ms",),
            injection_parameters={},
        )

        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="sudden_spike",
                phase="Evaluation",
                events=(invalid_event,),
            )

    def test_invalid_timestamp_order_is_rejected(self):
        invalid_data = self.data.copy()
        invalid_data.loc[5, "timestamp"] = invalid_data.loc[4, "timestamp"]

        with self.assertRaises(ValueError):
            build_ground_truth(
                invalid_data,
                seed=1000,
                scenario="normal",
                phase="Evaluation",
            )

    def test_invalid_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="normal",
                phase="Test",
            )
    def test_invalid_event_type_is_rejected(self):
        with self.assertRaises(TypeError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="sudden_spike",
                phase="Evaluation",
                events=("invalid_event",),
            )

    def test_unknown_affected_feature_is_rejected(self):
        invalid_event = AnomalyEvent(
            event_id="invalid-feature",
            start_index=3,
            end_index=6,
            anomaly_type="sudden_spike",
            severity="medium",
            affected_features=("unknown_feature",),
            injection_parameters={"multiplier": 3.0},
        )

        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="sudden_spike",
                phase="Evaluation",
                events=(invalid_event,),
            )

    def test_timestamp_cannot_be_an_affected_feature(self):
        invalid_event = AnomalyEvent(
            event_id="invalid-timestamp",
            start_index=3,
            end_index=6,
            anomaly_type="sudden_spike",
            severity="medium",
            affected_features=("timestamp",),
            injection_parameters={},
        )

        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="sudden_spike",
                phase="Evaluation",
                events=(invalid_event,),
            )

    def test_non_string_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            build_ground_truth(
                self.data,
                seed=1000,
                scenario="normal",
                phase=None,
            )

if __name__ == "__main__":
    unittest.main()
