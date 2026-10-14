
import unittest
import tempfile
from pathlib import Path

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
            self.assertEqual(
                summary["availability_rate_percent"], 75.0
            )
            self.assertEqual(
                summary["average_response_time_ms"], 200.0
            )
            self.assertEqual(summary["p95_response_time_ms"], 300.0)

    def test_handles_empty_database(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "empty.db"

            summary = summarize_results(db_path=db_path)

            self.assertEqual(summary["total_checks"], 0)
            self.assertEqual(
                summary["availability_rate_percent"], 0.0
            )
            self.assertIsNone(
                summary["average_response_time_ms"]
            )
            self.assertIsNone(summary["p95_response_time_ms"])


if __name__ == "__main__":
    unittest.main()
