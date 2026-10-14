
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from aynai.storage import get_results
from aynai.monitor import monitor_services


class TestMonitorServices(unittest.TestCase):

    @patch("aynai.monitor.check_http")
    def test_monitor_multiple_services(self, mock_check_http):
        mock_check_http.side_effect = [
            {
                "url": "https://example.com",
                "is_available": True,
                "status_code": 200,
                "response_time_ms": 120.5,
                "error": None,
            },
            {
                "url": "https://www.python.org",
                "is_available": False,
                "status_code": 503,
                "response_time_ms": 250.0,
                "error": "Service Unavailable",
            },
        ]

        urls = [
            "https://example.com",
            "https://www.python.org",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            results = monitor_services(urls, db_path=db_path)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status_code"], 200)
        self.assertEqual(results[1]["status_code"], 503)
        self.assertEqual(results[0]["url"], urls[0])
        self.assertEqual(results[1]["url"], urls[1])
        self.assertEqual(mock_check_http.call_count, 2)

    @patch("aynai.monitor.check_http")
    def test_unexpected_error_does_not_stop_monitoring(
        self, mock_check_http
    ):
        mock_check_http.side_effect = [
            RuntimeError("Unexpected failure"),
            {
                "url": "https://www.python.org",
                "is_available": True,
                "status_code": 200,
                "response_time_ms": 150.0,
                "error": None,
            },
        ]

        urls = [
            "https://example.com",
            "https://www.python.org",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            results = monitor_services(urls, db_path=db_path)

        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0]["error"])
        self.assertEqual(results[1]["status_code"], 200)
        self.assertEqual(mock_check_http.call_count, 2)

    def test_monitor_saves_results_to_database(self):
        result = {
            "url": "https://example.com",
            "is_available": True,
            "status_code": 200,
            "response_time_ms": 120.5,
            "error": None,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            with patch(
                "aynai.monitor.check_http",
                return_value=result,
            ):
                results = monitor_services(
                    ["https://example.com"],
                    db_path=db_path,
                )

            saved_results = get_results(
                url="https://example.com",
                db_path=db_path,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(saved_results), 1)
        self.assertEqual(saved_results[0]["status_code"], 200)
        self.assertEqual(saved_results[0]["is_available"], 1)


if __name__ == "__main__":
    unittest.main()
