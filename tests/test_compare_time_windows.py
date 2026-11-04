
import unittest
from unittest.mock import patch

import pandas as pd

from experiments.compare_time_windows import evaluate_time_window


class TestTimeWindowComparison(unittest.TestCase):

    def make_data(self):
        return pd.DataFrame(
            {
                "id": range(1, 9),
                "url": ["https://example.com"] * 8,
                "checked_at": pd.date_range(
                    "2026-01-01",
                    periods=8,
                    freq="min",
                    tz="UTC",
                ),
                "response_time_ms": [
                    10, 11, 12, 13, 14, 15, 100, 16
                ],
                "is_available": [1] * 8,
                "latency_anomaly_label": [
                    None, None, None, None, None, None, 1, 0
                ],
            }
        )

    @patch("experiments.compare_time_windows.detect_latency_anomaly")
    def test_uses_only_measurements_inside_time_window(
        self,
        detector,
    ):
        data = self.make_data()
        detector.return_value = {"is_anomaly": True}

        result = evaluate_time_window(
            data,
            window_minutes=5,
            threshold=2.5,
        )

        self.assertEqual(detector.call_count, 2)

        first_call = detector.call_args_list[0]
        self.assertEqual(first_call.args[0], 100.0)
        self.assertEqual(
            first_call.args[1],
            [11, 12, 13, 14, 15],
        )
        self.assertEqual(first_call.kwargs["threshold"], 2.5)

        self.assertEqual(result["evaluated_records"], 2)
        self.assertEqual(result["total_labeled_records"], 2)
        self.assertEqual(result["coverage_percent"], 100.0)

    
    def test_old_measurements_are_excluded(self):
        data = self.make_data()

        data["checked_at"] = pd.to_datetime(
            [
                "2026-01-01 00:00:00",
                "2026-01-01 00:08:00",
                "2026-01-01 00:08:30",
                "2026-01-01 00:09:00",
                "2026-01-01 00:09:30",
                "2026-01-01 00:09:45",
                "2026-01-01 00:10:00",
                "2026-01-01 00:11:00",
            ],
            utc=True,
        )

        with patch(
            "experiments.compare_time_windows.detect_latency_anomaly"
        ) as detector:
            detector.return_value = {"is_anomaly": True}

            evaluate_time_window(
                data,
                window_minutes=2,
                threshold=2.0,
            )

            self.assertEqual(detector.call_count, 1)

            first_call = detector.call_args_list[0]

            self.assertEqual(first_call.args[0], 100.0)
            self.assertEqual(
                first_call.args[1],
                [11, 12, 13, 14, 15],
            )

    def test_insufficient_history_is_not_evaluated(self):
        data = self.make_data()

        result = evaluate_time_window(
            data,
            window_minutes=2,
            min_history=5,
        )

        self.assertEqual(result["evaluated_records"], 0)
        self.assertEqual(result["coverage_percent"], 0.0)
        self.assertEqual(result["status"], "no_evaluable_labels")

    def test_invalid_window_is_rejected(self):
        data = self.make_data()

        for value in (0, -1, 2.5, True):
            with self.subTest(window=value):
                with self.assertRaises(ValueError):
                    evaluate_time_window(data, value)

    def test_invalid_minimum_history_is_rejected(self):
        data = self.make_data()

        for value in (4, 51, True):
            with self.subTest(min_history=value):
                with self.assertRaises(ValueError):
                    evaluate_time_window(
                        data,
                        window_minutes=5,
                        min_history=value,
                    )


if __name__ == "__main__":
    unittest.main()
