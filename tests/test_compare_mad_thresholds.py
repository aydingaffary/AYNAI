
import unittest
from unittest.mock import patch

import pandas as pd

from experiments.compare_mad_thresholds import evaluate_configuration


class TestMADConfigurationComparison(unittest.TestCase):
    def make_data(self):
        return pd.DataFrame(
            {
                "id": range(1, 9),
                "url": ["https://example.com"] * 8,
                "checked_at": pd.date_range(
                    "2026-01-01",
                    periods=8,
                    freq="h",
                    tz="UTC",
                ),
                "response_time_ms": [10, 11, 12, 13, 14, 15, 100, 16],
                "is_available": [1] * 8,
                "latency_anomaly_label": [
                    None, None, None, None, None, None, 1, 0
                ],
            }
        )

    @patch("experiments.compare_mad_thresholds.detect_latency_anomaly")
    def test_uses_only_latest_history_and_selected_threshold(
        self,
        detector,
    ):
        data = self.make_data()
        detector.return_value = {"is_anomaly": True}

        result = evaluate_configuration(
            data,
            threshold=2.5,
            history_limit=5,
        )

        self.assertEqual(detector.call_count, 2)

        first_call = detector.call_args_list[0]
        self.assertEqual(first_call.args[0], 100.0)
        self.assertEqual(first_call.args[1], [11, 12, 13, 14, 15])
        self.assertEqual(first_call.kwargs["threshold"], 2.5)

        self.assertEqual(result["history_limit"], 5)
        self.assertEqual(result["threshold"], 2.5)
        self.assertEqual(result["evaluated_records"], 2)

    def test_insufficient_history_is_not_scored(self):
        data = self.make_data().iloc[:5].copy()
        data["latency_anomaly_label"] = [0, 1, 0, 1, 0]

        result = evaluate_configuration(
            data,
            threshold=3.0,
            history_limit=10,
        )

        self.assertEqual(result["status"], "no_evaluable_labels")
        self.assertEqual(result["evaluated_records"], 0)

    def test_invalid_history_limit_is_rejected(self):
        data = self.make_data()

        for value in (4, True, 5.0):
            with self.subTest(history_limit=value):
                with self.assertRaises(ValueError):
                    evaluate_configuration(data, 3.0, value)

    def test_invalid_threshold_is_rejected(self):
        data = self.make_data()

        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(threshold=value):
                with self.assertRaises(ValueError):
                    evaluate_configuration(data, value, 10)


if __name__ == "__main__":
    unittest.main()
