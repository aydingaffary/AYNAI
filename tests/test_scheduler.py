import unittest
from unittest.mock import Mock

from aynai.scheduler import run_periodic_monitoring


class TestPeriodicMonitoring(unittest.TestCase):
    def test_runs_expected_number_of_cycles(self):
        monitor = Mock()
        sleep = Mock()

        cycles = run_periodic_monitoring(
            ["https://example.com"],
            interval_seconds=10,
            max_cycles=3,
            monitor_fn=monitor,
            sleep_fn=sleep,
        )

        self.assertEqual(cycles, 3)
        self.assertEqual(monitor.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(10)

    def test_rejects_invalid_interval(self):
        with self.assertRaises(ValueError):
            run_periodic_monitoring(
                ["https://example.com"],
                interval_seconds=0,
                monitor_fn=Mock(),
                sleep_fn=Mock(),
            )

    def test_rejects_invalid_cycle_count(self):
        with self.assertRaises(ValueError):
            run_periodic_monitoring(
                ["https://example.com"],
                max_cycles=0,
                monitor_fn=Mock(),
                sleep_fn=Mock(),
            )


if __name__ == "__main__":
    unittest.main()
