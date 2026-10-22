import unittest
import tempfile
from pathlib import Path

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from aynai.analytics import (
    summarize_anomalies,
    summarize_anomaly_trends,
    summarize_results,
)
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


    def test_summarizes_anomalies_and_unevaluated_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            base_result = {
                "url": "https://example.com",
                "is_available": True,
                "status_code": 200,
                "response_time_ms": 150.0,
                "error": None,
            }

            save_result(
                base_result,
                db_path,
                anomaly={
                    "is_anomaly": True,
                    "baseline_ms": 100.0,
                    "mad_ms": 10.0,
                    "reason": "Latency above expected range",
                },
            )
            save_result(
                base_result,
                db_path,
                anomaly={
                    "is_anomaly": False,
                    "baseline_ms": 100.0,
                    "mad_ms": 10.0,
                    "reason": "Latency within expected range",
                },
            )
            save_result(base_result, db_path)

            summary = summarize_anomalies(db_path=db_path)

            self.assertEqual(summary["total_checks"], 3)
            self.assertEqual(summary["evaluated_checks"], 2)
            self.assertEqual(summary["anomaly_count"], 1)
            self.assertEqual(summary["normal_count"], 1)
            self.assertEqual(summary["unevaluated_checks"], 1)
            self.assertEqual(summary["anomaly_rate_percent"], 50.0)

    def test_handles_empty_anomaly_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "empty.db"

            summary = summarize_anomalies(db_path=db_path)

            self.assertEqual(summary["total_checks"], 0)
            self.assertEqual(summary["evaluated_checks"], 0)
            self.assertEqual(summary["anomaly_count"], 0)
            self.assertEqual(summary["normal_count"], 0)
            self.assertEqual(summary["unevaluated_checks"], 0)
            self.assertEqual(summary["anomaly_rate_percent"], 0.0)

    def test_rejects_invalid_anomaly_time_window(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"

            for invalid_hours in (0, -1, float("nan"), float("inf")):
                with self.subTest(since_hours=invalid_hours):
                    with self.assertRaises(ValueError):
                        summarize_anomalies(
                            db_path=db_path,
                            since_hours=invalid_hours,
                        )
    def test_summarizes_anomaly_trends_across_buckets(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "trends.db"
            now = datetime.now(timezone.utc)

            samples = [
                (now - timedelta(hours=2), True),
                (now - timedelta(hours=1), False),
            ]

            for checked_at, is_anomaly in samples:
                with patch("aynai.storage.datetime") as mocked_datetime:
                    mocked_datetime.now.return_value = checked_at
                    mocked_datetime.timezone = timezone

                    save_result(
                        {
                            "url": "https://example.com",
                            "is_available": True,
                            "status_code": 200,
                            "response_time_ms": 150.0,
                            "error": None,
                        },
                        db_path,
                        anomaly={
                            "is_anomaly": is_anomaly,
                            "baseline_ms": 100.0,
                            "mad_ms": 10.0,
                            "reason": "Test result",
                        },
                    )

            summary = summarize_anomaly_trends(
                url="https://example.com",
                db_path=db_path,
                since_hours=3,
                bucket_hours=1,
            )

            self.assertEqual(len(summary["buckets"]), 3)
            self.assertEqual(summary["buckets"][0]["anomaly_count"], 1)
            self.assertEqual(summary["buckets"][0]["anomaly_rate_percent"], 100.0)
            self.assertEqual(summary["buckets"][1]["normal_count"], 1)
            self.assertEqual(summary["buckets"][1]["anomaly_rate_percent"], 0.0)
            self.assertEqual(summary["buckets"][2]["total_checks"], 0)
            self.assertIsNone(summary["buckets"][2]["anomaly_rate_percent"])

    def test_rejects_invalid_trend_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "trends.db"

            invalid_values = (0, -1, float("nan"), float("inf"))

            for value in invalid_values:
                with self.subTest(since_hours=value):
                    with self.assertRaises(ValueError):
                        summarize_anomaly_trends(
                            db_path=db_path,
                            since_hours=value,
                        )

                with self.subTest(bucket_hours=value):
                    with self.assertRaises(ValueError):
                        summarize_anomaly_trends(
                            db_path=db_path,
                            bucket_hours=value,
                        )

    def test_filters_anomaly_trends_by_url(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "trends.db"
            now = datetime.now(timezone.utc)

            for url in ("https://example.com", "https://www.python.org"):
                with patch("aynai.storage.datetime") as mocked_datetime:
                    mocked_datetime.now.return_value = now - timedelta(minutes=10)
                    mocked_datetime.timezone = timezone

                    save_result(
                        {
                            "url": url,
                            "is_available": True,
                            "status_code": 200,
                            "response_time_ms": 100.0,
                            "error": None,
                        },
                        db_path,
                        anomaly={
                            "is_anomaly": True,
                            "baseline_ms": 80.0,
                            "mad_ms": 5.0,
                            "reason": "Test anomaly",
                        },
                    )

            summary = summarize_anomaly_trends(
                url="https://example.com",
                db_path=db_path,
                since_hours=1,
                bucket_hours=1,
            )

            self.assertEqual(
                sum(bucket["total_checks"] for bucket in summary["buckets"]),
                1,
            )
            self.assertEqual(
                sum(bucket["anomaly_count"] for bucket in summary["buckets"]),
                1,
            )



if __name__ == "__main__":
    unittest.main()
