import unittest
from unittest.mock import patch

import pandas as pd

from experiments.evaluate_labeled_detectors import evaluate_labeled_detectors


class TestLabeledDetectors(unittest.TestCase):
    def make_data(self):
        latencies = [10, 11, 12, 13, 14, 100, 15]
        labels = [None, None, None, None, None, 1, 0]

        return pd.DataFrame(
            {
                "id": range(1, 8),
                "url": ["https://example.com"] * 7,
                "checked_at": pd.date_range(
                    "2026-01-01",
                    periods=7,
                    freq="h",
                    tz="UTC",
                ),
                "response_time_ms": latencies,
                "is_available": [1] * 7,
                "latency_anomaly_label": labels,
            }
        )

    @patch("experiments.evaluate_labeled_detectors.IsolationForest")
    @patch("experiments.evaluate_labeled_detectors.detect_latency_anomaly")
    def test_unlabeled_records_are_used_as_history(
        self,
        detector,
        isolation_forest,
    ):
        data = self.make_data()
        detector.side_effect = [
            {"is_anomaly": True},
            {"is_anomaly": False},
        ]

        model = isolation_forest.return_value
        model.predict.return_value = [-1]

        result = evaluate_labeled_detectors(data, min_history=5)

        self.assertEqual(detector.call_count, 2)
        first_history = detector.call_args_list[0].args[1]
        self.assertEqual(first_history, [10, 11, 12, 13, 14])
        self.assertEqual(result["evaluated_records"], 2)
        self.assertEqual(result["skipped_insufficient_history"], 0)

        first_training_data = model.fit.call_args_list[0].args[0]
        self.assertEqual(
            first_training_data["response_time_ms"].tolist(),
            [10, 11, 12, 13, 14],
        )

    @patch("experiments.evaluate_labeled_detectors.IsolationForest")
    @patch("experiments.evaluate_labeled_detectors.detect_latency_anomaly")
    def test_both_detectors_use_the_same_labeled_records(
        self,
        detector,
        isolation_forest,
    ):
        data = self.make_data()
        detector.side_effect = [
            {"is_anomaly": True},
            {"is_anomaly": False},
        ]
        isolation_forest.return_value.predict.side_effect = [
            [-1],
            [1],
        ]

        result = evaluate_labeled_detectors(data, min_history=5)

        self.assertEqual(result["evaluated_records"], 2)
        self.assertEqual(result["mad"]["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertEqual(
            result["isolation_forest"]["confusion_matrix"],
            [[1, 0], [0, 1]],
        )

    def test_no_labels_returns_insufficient_history(self):
        data = self.make_data()
        data["latency_anomaly_label"] = None

        result = evaluate_labeled_detectors(data, min_history=5)

        self.assertEqual(result["status"], "insufficient_history")
        self.assertEqual(result["evaluated_records"], 0)

    def test_insufficient_history_is_not_scored(self):
        data = self.make_data().iloc[:5].copy()
        data["latency_anomaly_label"] = [0, 1, 0, 1, 0]

        result = evaluate_labeled_detectors(data, min_history=5)

        self.assertEqual(result["status"], "insufficient_history")
        self.assertEqual(result["evaluated_records"], 0)
        self.assertEqual(result["skipped_insufficient_history"], 5)


if __name__ == "__main__":
    unittest.main()
