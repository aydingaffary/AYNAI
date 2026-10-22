import sys
import unittest
from unittest.mock import patch

import main


class TestCommandLineInterface(unittest.TestCase):
    @patch("main.run_periodic_monitoring")
    @patch("main.update_labels")
    def test_label_record_updates_only_specified_fields(
        self, mock_update_labels, mock_run_periodic_monitoring
    ):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--label-record", "123",
                "--latency-label", "1",
                "--label-notes", "Confirmed latency anomaly",
            ],
        ):
            main.main()

        mock_update_labels.assert_called_once_with(
            123,
            latency_anomaly_label=1,
            label_notes="Confirmed latency anomaly",
        )
        mock_run_periodic_monitoring.assert_not_called()

    @patch("main.run_periodic_monitoring")
    @patch("main.display_analytics")
    def test_analytics_mode_does_not_start_monitoring(
        self, mock_display_analytics, mock_run_periodic_monitoring
    ):
        with patch.object(sys, "argv", ["main.py", "--analytics-hours", "24"]):
            main.main()

        mock_display_analytics.assert_called_once_with(since_hours=24.0)
        mock_run_periodic_monitoring.assert_not_called()

    @patch("main.run_periodic_monitoring")
    @patch("main.display_analytics")
    def test_invalid_analytics_hours_are_rejected(
        self, mock_display_analytics, mock_run_periodic_monitoring
    ):
        invalid_values = ["0", "-1", "nan", "inf", "abc"]

        for value in invalid_values:
            with self.subTest(value=value):
                with patch.object(sys, "argv", ["main.py", "--analytics-hours", value]):
                    with self.assertRaises(SystemExit) as context:
                        main.main()

                self.assertEqual(context.exception.code, 2)

        mock_display_analytics.assert_not_called()
        mock_run_periodic_monitoring.assert_not_called()

    @patch("main.run_periodic_monitoring")
    @patch("main.update_labels")
    def test_label_options_require_record_and_field(
        self, mock_update_labels, mock_run_periodic_monitoring
    ):
        invalid_arguments = [
            ["--latency-label", "1"],
            ["--label-record", "123"],
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with patch.object(sys, "argv", ["main.py", *arguments]):
                    with self.assertRaises(SystemExit) as context:
                        main.main()

                self.assertEqual(context.exception.code, 2)

        mock_update_labels.assert_not_called()
        mock_run_periodic_monitoring.assert_not_called()

    @patch("main.run_periodic_monitoring")
    @patch("main.update_labels")
    @patch("main.display_analytics")
    def test_analytics_and_label_modes_cannot_be_combined(
        self,
        mock_display_analytics,
        mock_update_labels,
        mock_run_periodic_monitoring,
    ):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--analytics-hours", "24",
                "--label-record", "123",
                "--latency-label", "1",
            ],
        ):
            with self.assertRaises(SystemExit) as context:
                main.main()

        self.assertEqual(context.exception.code, 2)
        mock_display_analytics.assert_not_called()
        mock_update_labels.assert_not_called()
        mock_run_periodic_monitoring.assert_not_called()

    @patch("main.run_periodic_monitoring")
    @patch("main.display_analytics")
    def test_default_mode_starts_periodic_monitoring(
        self, mock_display_analytics, mock_run_periodic_monitoring
    ):
        with patch.object(sys, "argv", ["main.py"]):
            main.main()

        mock_run_periodic_monitoring.assert_called_once_with(
            main.URLS,
            interval_seconds=main.INTERVAL_SECONDS,
            monitor_fn=main.monitor_and_display,
        )
        mock_display_analytics.assert_not_called()


if __name__ == "__main__":
    unittest.main()
