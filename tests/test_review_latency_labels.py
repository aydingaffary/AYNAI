import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import review_latency_labels as reviewer


class TestReviewLatencyLabels(unittest.TestCase):
    def make_records(self):
        records = []

        for i, latency in enumerate([100, 110, 105, 115, 108, 500], start=1):
            records.append(
                {
                    "id": i,
                    "url": "https://example.com",
                    "checked_at": f"2026-01-01T00:0{i}:00+00:00",
                    "response_time_ms": latency,
                    "is_available": 1,
                    "status_code": 200,
                    "latency_anomaly_label": None,
                }
            )

        return records

    def test_help_exits_without_starting_review(self):
        with patch("sys.argv", ["review_latency_labels.py", "--help"]):
            with patch.object(reviewer, "review_batch") as review:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as exc:
                        reviewer.main()

        self.assertEqual(exc.exception.code, 0)
        review.assert_not_called()

    def test_quit_does_not_save_a_label(self):
        records = self.make_records()

        with (
            patch.object(
                reviewer,
                "load_review_candidates",
                return_value=(records, records),
            ),
            patch.object(reviewer, "update_labels") as update,
            patch("builtins.input", return_value="q"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reviewer.review_batch(db_path=Path("unused-test-db.sqlite"))

        update.assert_not_called()

    def test_skip_does_not_save_a_label(self):
        records = self.make_records()

        with (
            patch.object(
                reviewer,
                "load_review_candidates",
                return_value=(records, records),
            ),
            patch.object(reviewer, "update_labels") as update,
            patch("builtins.input", side_effect=["s", "q"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reviewer.review_batch(db_path=Path("unused-test-db.sqlite"))

        update.assert_not_called()

    def test_label_zero_is_saved(self):
        records = self.make_records()

        with (
            patch.object(
                reviewer,
                "load_review_candidates",
                return_value=(records, records),
            ),
            patch.object(reviewer, "update_labels") as update,
            patch("builtins.input", side_effect=["0", ""]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reviewer.review_batch(db_path=Path("unused-test-db.sqlite"))

        update.assert_called_once_with(
            6,
            db_path=Path("unused-test-db.sqlite"),
            latency_anomaly_label=0,
            label_notes=None,
        )

    def test_label_one_and_note_are_saved(self):
        records = self.make_records()

        with (
            patch.object(
                reviewer,
                "load_review_candidates",
                return_value=(records, records),
            ),
            patch.object(reviewer, "update_labels") as update,
            patch("builtins.input", side_effect=["1", "Clearly elevated"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reviewer.review_batch(db_path=Path("unused-test-db.sqlite"))

        update.assert_called_once_with(
            6,
            db_path=Path("unused-test-db.sqlite"),
            latency_anomaly_label=1,
            label_notes="Clearly elevated",
        )


if __name__ == "__main__":
    unittest.main()