import unittest

from experiments.edge_ai.inspection.config import (
    DEFAULT_INSPECTION_CONFIG,
    InspectionConfig,
)


class TestInspectionConfig(unittest.TestCase):

    def test_default_configuration(self):
        config = DEFAULT_INSPECTION_CONFIG

        self.assertEqual(
            config.temporal_windows_minutes,
            (5, 15, 30),
        )
        self.assertEqual(
            config.distribution_metrics,
            ("wasserstein", "jensen_shannon"),
        )
        self.assertTrue(config.generate_plots)
        config.validate()

    def test_invalid_temporal_window(self):
        with self.assertRaises(ValueError):
            InspectionConfig(
                temporal_windows_minutes=(5, 0, 30)
            ).validate()

    def test_duplicate_temporal_windows(self):
        with self.assertRaises(ValueError):
            InspectionConfig(
                temporal_windows_minutes=(5, 5)
            ).validate()

    def test_unsupported_distribution_metric(self):
        with self.assertRaises(ValueError):
            InspectionConfig(
                distribution_metrics=("unknown",)
            ).validate()

    def test_invalid_jsd_bins(self):
        with self.assertRaises(ValueError):
            InspectionConfig(jsd_bins=1).validate()

    def test_invalid_smoothing(self):
        with self.assertRaises(ValueError):
            InspectionConfig(jsd_smoothing=0).validate()

    def test_invalid_rolling_min_periods(self):
        with self.assertRaises(ValueError):
            InspectionConfig(rolling_min_periods=0).validate()


if __name__ == "__main__":
    unittest.main()
