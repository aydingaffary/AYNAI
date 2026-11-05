"""Tests for the correlated telemetry generator."""

import unittest

import pandas as pd

from experiments.edge_ai.data.telemetry_generator import (
    FEATURE_COLUMNS,
    generate_telemetry,
)


class TestTelemetryGenerator(unittest.TestCase):

    def test_generated_sample_count(self):
        data = generate_telemetry(seed=1000)

        self.assertEqual(len(data), 3840)

    def test_expected_columns(self):
        data = generate_telemetry(seed=1000)

        expected_columns = ("timestamp",) + FEATURE_COLUMNS

        self.assertEqual(tuple(data.columns), expected_columns)

    def test_reproducibility(self):
        first = generate_telemetry(seed=1000)
        second = generate_telemetry(seed=1000)

        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_produce_different_data(self):
        first = generate_telemetry(seed=1000)
        second = generate_telemetry(seed=1001)

        self.assertFalse(
            first.loc[:, FEATURE_COLUMNS].equals(
                second.loc[:, FEATURE_COLUMNS]
            )
        )

    def test_no_missing_or_infinite_values(self):
        data = generate_telemetry(seed=1000)

        self.assertFalse(data.isna().any().any())

        numeric_data = data.loc[:, FEATURE_COLUMNS]

        self.assertTrue(
            numeric_data.map(
                lambda value: pd.notna(value) and abs(value) != float("inf")
            ).all().all()
        )

    def test_physical_value_ranges(self):
        data = generate_telemetry(seed=1000)

        non_negative_columns = (
            "latency_ms",
            "jitter_ms",
            "throughput_mbps",
        )

        for column in non_negative_columns:
            self.assertTrue((data[column] >= 0).all())

        percentage_columns = (
            "packet_loss_pct",
            "cpu_usage_pct",
            "memory_usage_pct",
        )

        for column in percentage_columns:
            self.assertTrue(data[column].between(0, 100).all())

    def test_sampling_interval(self):
        data = generate_telemetry(seed=1000)

        differences = data["timestamp"].diff().dropna()

        self.assertTrue(
            (differences == pd.Timedelta(seconds=30)).all()
        )

    def test_expected_correlations(self):
        data = generate_telemetry(seed=1000)
        correlation = data.loc[:, FEATURE_COLUMNS].corr()

        self.assertGreater(
            correlation.loc["latency_ms", "jitter_ms"],
            0.50,
        )

        self.assertGreater(
            correlation.loc["latency_ms", "packet_loss_pct"],
            0.60,
        )

        self.assertLess(
            correlation.loc["latency_ms", "throughput_mbps"],
            -0.60,
        )

        self.assertGreater(
            correlation.loc["cpu_usage_pct", "memory_usage_pct"],
            0.80,
        )

    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            generate_telemetry(seed=-1)

    def test_custom_start_time(self):
        data = generate_telemetry(
            seed=1000,
            start_time="2026-01-01T12:00:00",
        )

        self.assertEqual(
            data["timestamp"].iloc[0],
            pd.Timestamp("2026-01-01T12:00:00"),
        )


if __name__ == "__main__":
    unittest.main()
