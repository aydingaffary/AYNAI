import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from aynai.analytics import summarize_results
from aynai.storage import save_result


class TestAnalytics(unittest.TestCase):
    def test_calculates_monitoring_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"

            samples = [
                (True, 100.0),
                (True, 200.0),
                (False, 0.0),
                (True, 300.0),
            ]

            for index, (available, response_time) in enumerate(samples):
                save_result(
                    {
                        "url": "https://example.com",
                        "is_available": available,
                        "status_code": 200 if available else None,
                        "response_time_ms": response_time,
                        "error": None if available else "Connection failed",
                    },
                    db_path,
                )

            summary = summarize_results(
                url="https://example.com",
                db_path=db_path,
            )

            self.assertEqual(summary["total_checks"], 4)
            self.assertEqual(summary["successful_checks"], 3)
            self.assertEqual(summary["failed_checks"], 1)
            self.assertEqual(summary["availability_rate_percent"], 75.0)
            self.assertEqual(summary["average_response_time_ms"], 200.0)
            self.assertEqual(summary["p95_response_time_ms"], 300.0)

    def test_handles_empty_database(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "empty.db"

            summary = summarize_results(db_path=db_path)

            self.assertEqual(summary["total_checks"], 0)
            self.assertEqual(summary["availability_rate_percent"], 0.0)
            self.assertIsNone(summary["average_response_time_ms"])
            self.assertIsNone(summary["p95_response_time_ms"])

    def test_filters_results_by_time_window(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            now = datetime.now(timezone.utc)

            samples = [
                (True, 100.0, now - timedelta(hours=2)),
                (True, 200.0, now - timedelta(minutes=30)),
            ]

            for available, response_time, checked_at in samples:
                with patch("aynai.storage.datetime") as mocked_datetime:
                    mocked_datetime.now.return_value = checked_at
                    mocked_datetime.timezone = timezone

                    save_result(
                        {
                            "url": "https://example.com",
                            "is_available": available,
                            "status_code": 200,
                            "response_time_ms": response_time,
                            "error": None,
                        },
                        db_path,
                    )

            summary = summarize_results(
                url="https://example.com",
                db_path=db_path,
                since_hours=1,
            )

            self.assertEqual(summary["total_checks"], 1)
            self.assertEqual(summary["average_response_time_ms"], 200.0)

    def test_rejects_invalid_time_window(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"

            for invalid_hours in (0, -1, float("nan"), float("inf")):
                with self.subTest(since_hours=invalid_hours):
                    with self.assertRaises(ValueError):
                        summarize_results(
                            db_path=db_path,
                            since_hours=invalid_hours,
                        )

    def test_handles_empty_time_window(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"

            summary = summarize_results(
                db_path=db_path,
                since_hours=1,
            )

            self.assertEqual(summary["total_checks"], 0)
            self.assertEqual(summary["availability_rate_percent"], 0.0)
            self.assertIsNone(summary["average_response_time_ms"])
            self.assertIsNone(summary["p95_response_time_ms"])


if __name__ == "__main__":
    unittest.main()
