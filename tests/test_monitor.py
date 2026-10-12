
import unittest
from unittest.mock import patch

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

        results = monitor_services(urls)

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

        results = monitor_services(urls)

        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0]["error"])
        self.assertEqual(results[1]["status_code"], 200)
        self.assertEqual(mock_check_http.call_count, 2)



if __name__ == "__main__":
    unittest.main()
