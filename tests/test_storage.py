
import tempfile
import unittest
from pathlib import Path

from aynai.storage import get_results, init_db, save_result


class TestStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_db_creates_database(self):
        init_db(self.db_path)

        self.assertTrue(self.db_path.exists())

    def test_save_and_retrieve_result(self):
        result = {
            "url": "https://example.com",
            "is_available": True,
            "status_code": 200,
            "response_time_ms": 120.5,
            "error": None,
        }

        save_result(result, self.db_path)

        results = get_results(
            url="https://example.com",
            db_path=self.db_path,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com")
        self.assertEqual(results[0]["is_available"], 1)
        self.assertEqual(results[0]["status_code"], 200)
        self.assertEqual(results[0]["response_time_ms"], 120.5)
        self.assertIsNone(results[0]["error"])
        self.assertIsNotNone(results[0]["checked_at"])

    def test_results_can_be_filtered_by_url(self):
        first = {
            "url": "https://example.com",
            "is_available": True,
            "status_code": 200,
            "response_time_ms": 100.0,
            "error": None,
        }

        second = {
            "url": "https://python.org",
            "is_available": False,
            "status_code": 503,
            "response_time_ms": 250.0,
            "error": "Service Unavailable",
        }

        save_result(first, self.db_path)
        save_result(second, self.db_path)

        results = get_results(
            url="https://example.com",
            db_path=self.db_path,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com")

        all_results = get_results(db_path=self.db_path)
        self.assertEqual(len(all_results), 2)


if __name__ == "__main__":
    unittest.main()
