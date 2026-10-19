import unittest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError

from aynai.probes.http_probe import check_http


class TestHttpProbe(unittest.TestCase):
    @patch("aynai.probes.http_probe.urllib.request.urlopen")
    def test_successful_http_response(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = check_http("https://example.com")

        self.assertEqual(result["url"], "https://example.com")
        self.assertTrue(result["is_available"])
        self.assertEqual(result["status_code"], 200)
        self.assertIsNone(result["error"])
        self.assertGreaterEqual(result["response_time_ms"], 0)

        mock_urlopen.assert_called_once()

    @patch("aynai.probes.http_probe.urllib.request.urlopen")
    def test_http_error_response(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            url="https://example.com",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=None,
        )

        result = check_http("https://example.com")

        self.assertFalse(result["is_available"])
        self.assertEqual(result["status_code"], 503)
        self.assertIsNotNone(result["error"])
        self.assertGreaterEqual(result["response_time_ms"], 0)

    @patch("aynai.probes.http_probe.urllib.request.urlopen")
    def test_timeout_wrapped_in_url_error(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError(
            TimeoutError("Request timed out")
        )

        result = check_http("https://example.com")

        self.assertFalse(result["is_available"])
        self.assertIsNone(result["status_code"])
        self.assertIsNotNone(result["error"])
        self.assertIn("timed out", result["error"].lower())
    @patch("aynai.probes.http_probe.urllib.request.urlopen")
    def test_timeout_wrapped_in_url_error(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError(
            TimeoutError("Request timed out")
        )

        result = check_http("https://example.com")

        self.assertFalse(result["is_available"])
        self.assertIsNone(result["status_code"])
        self.assertIsNotNone(result["error"])
        self.assertIn("timed out", result["error"].lower())

    @patch("aynai.probes.http_probe.urllib.request.urlopen")
    def test_custom_timeout_is_passed_to_urlopen(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        check_http("https://example.com", timeout=2.5)

        self.assertEqual(
            mock_urlopen.call_args.kwargs["timeout"],
            2.5,
        )

    @patch("aynai.probes.http_probe.urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection failed")

        result = check_http("https://example.com")

        self.assertFalse(result["is_available"])
        self.assertIsNone(result["status_code"])
        self.assertIsNotNone(result["error"])
        self.assertIn("connection failed", result["error"].lower())
        self.assertGreaterEqual(result["response_time_ms"], 0)

if __name__ == "__main__":
    unittest.main()
