
import unittest
from unittest.mock import patch

import pandas as pd

from experiments.evaluate_labeled_anomalies import evaluate_labeled_records


class TestLabeledEvaluation(unittest.TestCase):
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

    def test_unlabeled_records_are_used_as_history(self):
        data = self.make_data()

        with patch(
            "experiments.evaluate_labeled_anomalies.detect_latency_anomaly",
            side_effect=[
                {"is_anomaly": True},
                {"is_anomaly": False},
            ],
        ) as detector:
            result = evaluate_labeled_records(data)

        self.assertEqual(detector.call_count, 2)
        first_history = detector.call_args_list[0].args[1]
        self.assertEqual(first_history, [10, 11, 12, 13, 14])
        self.assertEqual(result["labeled_records"], 2)
        self.assertEqual(result["evaluated_records"], 2)

    def test_metrics_and_confusion_matrix(self):
        data = self.make_data()

        with patch(
            "experiments.evaluate_labeled_anomalies.detect_latency_anomaly",
            side_effect=[
                {"is_anomaly": True},
                {"is_anomaly": False},
            ],
        ):
            result = evaluate_labeled_records(data)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["confusion_matrix"], [[1, 0], [0, 1]])

    def test_no_labels_returns_insufficient_labels(self):
        data = self.make_data()
        data["latency_anomaly_label"] = None

        result = evaluate_labeled_records(data)

        self.assertEqual(result["status"], "insufficient_labels")
        self.assertEqual(result["evaluated_records"], 0)

    def test_insufficient_history_is_not_scored(self):
        data = self.make_data().iloc[:5].copy()
        data["latency_anomaly_label"] = [0, 1, 0, 1, 0]

        result = evaluate_labeled_records(data)

        self.assertEqual(result["status"], "insufficient_history")
        self.assertEqual(result["evaluated_records"], 0)


if __name__ == "__main__":
    unittest.main()
