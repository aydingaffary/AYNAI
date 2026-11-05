import unittest

from experiments.edge_ai.config import (
    DEFAULT_BASELINE,
    DEFAULT_CONFIG,
    ExperimentConfig,
    SCENARIO_NAMES,
)


class TestExperimentConfig(unittest.TestCase):

    def test_default_sample_counts(self):
        self.assertEqual(DEFAULT_CONFIG.training_samples, 720)
        self.assertEqual(DEFAULT_CONFIG.calibration_samples, 240)
        self.assertEqual(DEFAULT_CONFIG.evaluation_samples, 2880)

    def test_total_samples(self):
        self.assertEqual(DEFAULT_CONFIG.total_samples, 3840)

    def test_seed_generation(self):
        self.assertEqual(len(DEFAULT_CONFIG.seeds), 30)
        self.assertEqual(DEFAULT_CONFIG.seeds[0], 1000)
        self.assertEqual(DEFAULT_CONFIG.seeds[-1], 1029)

    def test_sampling_interval(self):
        self.assertEqual(
            DEFAULT_CONFIG.sampling_interval_seconds, 30
        )

    def test_baseline_values(self):
        self.assertEqual(DEFAULT_BASELINE.latency_ms, 50.0)
        self.assertEqual(DEFAULT_BASELINE.jitter_ms, 5.0)
        self.assertEqual(DEFAULT_BASELINE.packet_loss_pct, 0.5)

    def test_scenario_count(self):
        self.assertEqual(len(SCENARIO_NAMES), 6)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(training_samples=-1).validate()

    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(seed_start=-1).validate()


if __name__ == "__main__":
    unittest.main()