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

    def test_init_db_migrates_existing_database_without_losing_data(self):
        import sqlite3

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(self.db_path)

        try:
            connection.execute("""
                CREATE TABLE monitoring_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    is_available INTEGER NOT NULL,
                    status_code INTEGER,
                    response_time_ms REAL NOT NULL,
                    error TEXT,
                    checked_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                INSERT INTO monitoring_results (
                    url, is_available, status_code,
                    response_time_ms, error, checked_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "https://example.com",
                1,
                200,
                120.5,
                None,
                "2026-10-19T10:00:00+00:00",
            ))

            connection.commit()
        finally:
            connection.close()

        init_db(self.db_path)

        connection = sqlite3.connect(self.db_path)

        try:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(monitoring_results)"
                )
            }

            saved_row = connection.execute("""
                SELECT url, response_time_ms
                FROM monitoring_results
                WHERE url = ?
            """, ("https://example.com",)).fetchone()
        finally:
            connection.close()

        self.assertIn("is_anomaly", columns)
        self.assertIn("baseline_ms", columns)
        self.assertIn("mad_ms", columns)
        self.assertIn("anomaly_reason", columns)
        self.assertEqual(
            saved_row,
            ("https://example.com", 120.5),
        )

    def test_new_label_columns_exist(self):
        import sqlite3

        init_db(self.db_path)

        connection = sqlite3.connect(self.db_path)
        try:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(monitoring_results)"
                )
            }
        finally:
            connection.close()

        self.assertIn("latency_anomaly_label", columns)
        self.assertIn("availability_issue_label", columns)
        self.assertIn("label_notes", columns)

    def test_new_label_columns_are_null_for_existing_records(self):
        import sqlite3

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)

        try:
            connection.execute("""
                CREATE TABLE monitoring_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    is_available INTEGER NOT NULL,
                    status_code INTEGER,
                    response_time_ms REAL NOT NULL,
                    error TEXT,
                    checked_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                INSERT INTO monitoring_results (
                    url, is_available, status_code,
                    response_time_ms, error, checked_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "https://example.com",
                1,
                200,
                120.5,
                None,
                "2026-10-19T10:00:00+00:00",
            ))
            connection.commit()
        finally:
            connection.close()

        init_db(self.db_path)

        connection = sqlite3.connect(self.db_path)
        try:
            labels = connection.execute("""
                SELECT
                    latency_anomaly_label,
                    availability_issue_label,
                    label_notes
                FROM monitoring_results
                WHERE url = ?
            """, ("https://example.com",)).fetchone()
        finally:
            connection.close()

        self.assertEqual(labels, (None, None, None))



    def test_save_and_retrieve_independent_labels(self):
        result = {
            "url": "https://example.com",
            "is_available": True,
            "status_code": 200,
            "response_time_ms": 450.0,
            "error": None,
        }

        labels = {
            "latency_anomaly_label": 1,
            "availability_issue_label": 0,
            "label_notes": "Latency anomaly confirmed by review",
        }

        save_result(
            result,
            db_path=self.db_path,
            labels=labels,
        )

        results = get_results(
            url="https://example.com",
            db_path=self.db_path,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["latency_anomaly_label"], 1)
        self.assertEqual(results[0]["availability_issue_label"], 0)
        self.assertEqual(
            results[0]["label_notes"],
            "Latency anomaly confirmed by review",
        )
        self.assertIsNone(results[0]["is_anomaly"])


    def test_invalid_labels_are_rejected(self):
        result = {
            "url": "https://example.com",
            "is_available": True,
            "status_code": 200,
            "response_time_ms": 120.5,
            "error": None,
        }

        invalid_values = (2, -1, "1", True, False)

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    save_result(
                        result,
                        db_path=self.db_path,
                        labels={
                            "latency_anomaly_label": value,
                        },
                    )

        self.assertEqual(get_results(db_path=self.db_path), [])

    def test_invalid_label_notes_are_rejected(self):
        result = {
            "url": "https://example.com",
            "is_available": True,
            "status_code": 200,
            "response_time_ms": 120.5,
            "error": None,
        }

        with self.assertRaises(ValueError):
            save_result(
                result,
                db_path=self.db_path,
                labels={"label_notes": 123},
            )

        self.assertEqual(get_results(db_path=self.db_path), [])



if __name__ == "__main__":
    unittest.main()
