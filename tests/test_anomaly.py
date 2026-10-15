import tempfile
import unittest
import math
import math
from pathlib import Path
from unittest.mock import patch

from aynai.anomaly import (
    analyze_service_latency,
    detect_latency_anomaly,
)


class TestLatencyAnomaly(unittest.TestCase):
    def test_detects_high_latency(self):
        history = [100, 102, 98, 101, 99, 103, 100]

        result = detect_latency_anomaly(500, history)

        self.assertTrue(result["is_anomaly"])
        self.assertEqual(result["baseline_ms"], 100)

    def test_accepts_normal_latency(self):
        history = [100, 102, 98, 101, 99, 103, 100]

        result = detect_latency_anomaly(105, history)

        self.assertFalse(result["is_anomaly"])

    def test_requires_enough_history(self):
        result = detect_latency_anomaly(500, [100, 102, 98, 101])

        self.assertFalse(result["is_anomaly"])
        self.assertIsNone(result["baseline_ms"])

    def test_rejects_negative_response_time(self):
        with self.assertRaises(ValueError):
            detect_latency_anomaly(-1, [100, 102, 98, 101, 99])

    def test_rejects_non_finite_threshold(self):
        with self.assertRaises(ValueError):
            detect_latency_anomaly(
                500,
                [100, 102, 98, 101, 99],
                threshold=float("nan"),
            )

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            detect_latency_anomaly(
                500,
                [100, 102, 98, 101, 99],
                threshold=0,
            )

    def test_detects_spike_when_mad_is_zero(self):
        history = [100, 100, 100, 100, 100]

        result = detect_latency_anomaly(200, history)

        self.assertTrue(result["is_anomaly"])
        self.assertEqual(result["baseline_ms"], 100)
        self.assertEqual(result["mad_ms"], 0)

    def test_does_not_detect_small_increase_when_mad_is_zero(self):
        history = [100, 100, 100, 100, 100]

        result = detect_latency_anomaly(140, history)

        self.assertFalse(result["is_anomaly"])
        self.assertEqual(result["baseline_ms"], 100)
        self.assertEqual(result["mad_ms"], 0)

    def test_rejects_nan_response_time(self):
        with self.assertRaises(ValueError):
            detect_latency_anomaly(float("nan"), [100, 102, 98, 101, 99])

    def test_rejects_infinite_response_time(self):
        with self.assertRaises(ValueError):
            detect_latency_anomaly(float("inf"), [100, 102, 98, 101, 99])

    def test_detects_spike_with_variable_historical_latency(self):
        history = [90, 95, 100, 105, 110, 100, 95, 105, 100]

        result = detect_latency_anomaly(300, history)

        self.assertTrue(result["is_anomaly"])
        self.assertEqual(result["baseline_ms"], 100)
        self.assertEqual(result["mad_ms"], 5)

    @patch("aynai.anomaly.get_results")
    def test_uses_most_recent_successful_history(self, mock_get_results):
        mock_get_results.return_value = [
            {"id": 10, "is_available": 1, "response_time_ms": 1000.0},
            {"id": 9, "is_available": 1, "response_time_ms": 101.0},
            {"id": 8, "is_available": 1, "response_time_ms": 99.0},
            {"id": 7, "is_available": 1, "response_time_ms": 100.0},
            {"id": 6, "is_available": 1, "response_time_ms": 102.0},
            {"id": 5, "is_available": 1, "response_time_ms": 98.0},
        ]

        result = analyze_service_latency(
            url="https://example.com",
            current_response_time_ms=500.0,
            history_limit=5,
            exclude_record_id=10,
        )

        self.assertEqual(result["baseline_ms"], 100.0)
        self.assertTrue(result["is_anomaly"])

    @patch("aynai.anomaly.get_results")
    def test_excludes_latest_saved_record(self, mock_get_results):
        mock_get_results.return_value = [
            {"id": 6, "is_available": 1, "response_time_ms": 500.0},
            {"id": 5, "is_available": 1, "response_time_ms": 100.0},
            {"id": 4, "is_available": 1, "response_time_ms": 102.0},
            {"id": 3, "is_available": 1, "response_time_ms": 98.0},
            {"id": 2, "is_available": 1, "response_time_ms": 101.0},
            {"id": 1, "is_available": 1, "response_time_ms": 99.0},
        ]

        result = analyze_service_latency(
            url="https://example.com",
            current_response_time_ms=500.0,
            exclude_record_id=6,
        )

        self.assertEqual(result["baseline_ms"], 100.0)
        self.assertTrue(result["is_anomaly"])

    @patch("aynai.anomaly.get_results")
    def test_ignores_failed_historical_checks(self, mock_get_results):
        mock_get_results.return_value = [
            {"id": 7, "is_available": 1, "response_time_ms": 100.0},
            {"id": 6, "is_available": 0, "response_time_ms": 0.0},
            {"id": 5, "is_available": 1, "response_time_ms": 102.0},
            {"id": 4, "is_available": 1, "response_time_ms": 98.0},
            {"id": 3, "is_available": 1, "response_time_ms": 101.0},
            {"id": 2, "is_available": 1, "response_time_ms": 99.0},
            {"id": 1, "is_available": 1, "response_time_ms": 103.0},
        ]

        result = analyze_service_latency(
            url="https://example.com",
            current_response_time_ms=105.0,
            exclude_record_id=7,
        )

        self.assertEqual(result["baseline_ms"], 101.0)
        self.assertFalse(result["is_anomaly"])

    def test_requires_five_historical_successes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            from aynai.storage import init_db, save_result

            init_db(db_path)

            for value in [100.0, 102.0, 98.0, 101.0]:
                save_result(
                    {
                        "url": "https://example.com",
                        "is_available": True,
                        "status_code": 200,
                        "response_time_ms": value,
                        "error": None,
                    },
                    db_path,
                )

            result = analyze_service_latency(
                url="https://example.com",
                current_response_time_ms=500.0,
                db_path=db_path,
            )

        self.assertFalse(result["is_anomaly"])
        self.assertIsNone(result["baseline_ms"])
        self.assertEqual(result["reason"], "Insufficient historical data")


if __name__ == "__main__":
    unittest.main()
