"""Tests for deterministic Edge AI anomaly injection."""

import unittest

import numpy as np
import pandas as pd

from experiments.edge_ai.config import SCENARIO_NAMES
from experiments.edge_ai.data.telemetry_generator import (
    FEATURE_COLUMNS,
    generate_telemetry,
)
from experiments.edge_ai.simulation.anomaly_injector import (
    inject_anomalies,
)
from experiments.edge_ai.simulation.ground_truth import build_ground_truth


class TestAnomalyInjector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generated = generate_telemetry(seed=1000)

        cls.data = generated.iloc[960:].reset_index(drop=True)

    def test_all_scenarios_produce_expected_event_counts(self):
        expected_counts = {
            "normal": 0,
            "sudden_spike": 4,
            "gradual_drift": 2,
            "burst_anomalies": 15,
            "variance_shift": 2,
            "multivariate_anomaly": 4,
        }

        for scenario in SCENARIO_NAMES:
            with self.subTest(scenario=scenario):
                result, events = inject_anomalies(
                    self.data,
                    scenario=scenario,
                    seed=1000,
                )

                self.assertEqual(
                    len(events),
                    expected_counts[scenario],
                )

                self.assertEqual(len(result), len(self.data))

    def test_all_scenarios_preserve_timestamps(self):
        for scenario in SCENARIO_NAMES:
            with self.subTest(scenario=scenario):
                result, _ = inject_anomalies(
                    self.data,
                    scenario=scenario,
                    seed=1000,
                )

                pd.testing.assert_series_equal(
                    result["timestamp"],
                    self.data["timestamp"],
                )

    def test_input_dataframe_is_not_modified(self):
        original = self.data.copy(deep=True)

        for scenario in SCENARIO_NAMES:
            inject_anomalies(
                self.data,
                scenario=scenario,
                seed=1000,
            )

            pd.testing.assert_frame_equal(self.data, original)

    def test_reproducibility(self):
        for scenario in SCENARIO_NAMES:
            with self.subTest(scenario=scenario):
                first, first_events = inject_anomalies(
                    self.data,
                    scenario=scenario,
                    seed=1000,
                )

                second, second_events = inject_anomalies(
                    self.data,
                    scenario=scenario,
                    seed=1000,
                )

                pd.testing.assert_frame_equal(first, second)
                self.assertEqual(first_events, second_events)

    def test_different_seeds_produce_different_injections(self):
        first, first_events = inject_anomalies(
            self.data,
            scenario="sudden_spike",
            seed=1000,
        )

        second, second_events = inject_anomalies(
            self.data,
            scenario="sudden_spike",
            seed=1001,
        )

        self.assertNotEqual(
            [event.start_index for event in first_events],
            [event.start_index for event in second_events],
        )

        self.assertFalse(first.equals(second))

    def test_event_intervals_are_valid_and_non_overlapping(self):
        for scenario in SCENARIO_NAMES:
            with self.subTest(scenario=scenario):
                _, events = inject_anomalies(
                    self.data,
                    scenario=scenario,
                    seed=1000,
                )

                ordered = sorted(
                    events,
                    key=lambda event: event.start_index,
                )

                for event in ordered:
                    self.assertGreaterEqual(event.start_index, 0)
                    self.assertGreater(event.end_index, event.start_index)
                    self.assertLessEqual(
                        event.end_index,
                        len(self.data),
                    )

                for previous, current in zip(
                    ordered,
                    ordered[1:],
                ):
                    self.assertLessEqual(
                        previous.end_index,
                        current.start_index,
                    )

    def test_normal_scenario_does_not_change_features(self):
        result, events = inject_anomalies(
            self.data,
            scenario="normal",
            seed=1000,
        )

        self.assertEqual(events, ())

        pd.testing.assert_frame_equal(
            result.loc[:, FEATURE_COLUMNS],
            self.data.loc[:, FEATURE_COLUMNS],
        )

    def test_sudden_spikes_use_approved_multipliers(self):
        result, events = inject_anomalies(
            self.data,
            scenario="sudden_spike",
            seed=1000,
        )

        expected_multipliers = (2.0, 3.0, 3.0, 5.0)

        for event, multiplier in zip(events, expected_multipliers):
            self.assertEqual(event.end_index - event.start_index, 3)

            self.assertEqual(
                event.injection_parameters["multiplier"],
                multiplier,
            )

            for feature in (
                "latency_ms",
                "jitter_ms",
                "packet_loss_pct",
            ):
                before = self.data[feature].iloc[
                    event.start_index:event.end_index
                ].to_numpy()

                after = result[feature].iloc[
                    event.start_index:event.end_index
                ].to_numpy()

                np.testing.assert_allclose(
                    after,
                    before * multiplier,
                )

    def test_gradual_drift_has_linear_ramp(self):
        result, events = inject_anomalies(
            self.data,
            scenario="gradual_drift",
            seed=1000,
        )

        for event, multiplier in zip(events, (2.0, 3.0)):
            expected = np.linspace(1.0, multiplier, 60)

            for feature in ("latency_ms", "jitter_ms"):
                before = self.data[feature].iloc[
                    event.start_index:event.end_index
                ].to_numpy(dtype=float)

                after = result[feature].iloc[
                    event.start_index:event.end_index
                ].to_numpy(dtype=float)

                expected_values = before * expected

                np.testing.assert_allclose(
                    after,
                    expected_values,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_burst_events_have_three_clusters(self):
        _, events = inject_anomalies(
            self.data,
            scenario="burst_anomalies",
            seed=1000,
        )

        clusters = {}

        for event in events:
            parameters = event.injection_parameters
            cluster_id = parameters["cluster_id"]

            clusters.setdefault(cluster_id, []).append(event)

            self.assertEqual(event.end_index - event.start_index, 2)

        self.assertEqual(len(clusters), 3)

        for cluster_events in clusters.values():
            self.assertEqual(len(cluster_events), 5)

            cluster_events.sort(
                key=lambda event: event.start_index
            )

            for previous, current in zip(
                cluster_events,
                cluster_events[1:],
            ):
                gap = current.start_index - previous.end_index

                self.assertIn(gap, (1, 2, 3))

    def test_variance_shift_preserves_mean_and_scales_std(self):
        result, events = inject_anomalies(
            self.data,
            scenario="variance_shift",
            seed=1000,
        )

        for event, multiplier in zip(events, (2.0, 3.0)):
            for feature in ("latency_ms", "jitter_ms"):
                before = self.data[feature].iloc[
                    event.start_index:event.end_index
                ].to_numpy()

                after = result[feature].iloc[
                    event.start_index:event.end_index
                ].to_numpy()

                self.assertAlmostEqual(
                    float(np.mean(before)),
                    float(np.mean(after)),
                    places=9,
                )

                self.assertAlmostEqual(
                    float(np.std(after, ddof=1)),
                    float(np.std(before, ddof=1) * multiplier),
                    places=9,
                )

    def test_multivariate_preserves_marginal_values(self):
        result, events = inject_anomalies(
            self.data,
            scenario="multivariate_anomaly",
            seed=1000,
        )

        features = (
            "latency_ms",
            "jitter_ms",
            "packet_loss_pct",
            "throughput_mbps",
        )

        for event in events:
            before = self.data.loc[
                event.start_index:event.end_index - 1,
                features,
            ].to_numpy(dtype=float)

            after = result.loc[
                event.start_index:event.end_index - 1,
                features,
            ].to_numpy(dtype=float)

            before_corr = np.corrcoef(before, rowvar=False)
            after_corr = np.corrcoef(after, rowvar=False)

            distance = np.linalg.norm(
                after_corr - before_corr,
                ord="fro",
            )

            self.assertGreaterEqual(distance, 0.15)

            for feature_index, feature in enumerate(features):
                np.testing.assert_array_equal(
                    np.sort(before[:, feature_index]),
                    np.sort(after[:, feature_index]),
                )

    def test_ground_truth_labels_match_injected_events(self):
        injected, events = inject_anomalies(
            self.data,
            scenario="sudden_spike",
            seed=1000,
        )

        labeled, metadata = build_ground_truth(
            injected,
            seed=1000,
            scenario="sudden_spike",
            phase="Evaluation",
            events=events,
        )

        expected_anomalies = sum(
            event.end_index - event.start_index
            for event in events
        )

        self.assertEqual(
            int(labeled["is_anomaly"].sum()),
            expected_anomalies,
        )

        self.assertEqual(len(metadata), len(events))

        for event, record in zip(events, metadata):
            self.assertEqual(record["event_id"], event.event_id)
            self.assertEqual(record["start_index"], event.start_index)
            self.assertEqual(record["end_index"], event.end_index)

    def test_output_has_no_invalid_physical_values(self):
        for scenario in SCENARIO_NAMES:
            with self.subTest(scenario=scenario):
                result, _ = inject_anomalies(
                    self.data,
                    scenario=scenario,
                    seed=1000,
                )

                for feature in FEATURE_COLUMNS:
                    self.assertTrue(
                        np.isfinite(
                            result[feature].to_numpy(dtype=float)
                        ).all()
                    )

                for feature in (
                    "latency_ms",
                    "jitter_ms",
                    "throughput_mbps",
                ):
                    self.assertTrue((result[feature] >= 0).all())

                for feature in (
                    "packet_loss_pct",
                    "cpu_usage_pct",
                    "memory_usage_pct",
                ):
                    self.assertTrue(
                        result[feature].between(0, 100).all()
                    )

    def test_non_dataframe_input_is_rejected(self):
        with self.assertRaises(TypeError):
            inject_anomalies(
                [],
                scenario="normal",
                seed=1000,
            )

    def test_empty_dataframe_is_rejected(self):
        with self.assertRaises(ValueError):
            inject_anomalies(
                self.data.iloc[0:0],
                scenario="normal",
                seed=1000,
            )

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            inject_anomalies(
                self.data,
                scenario="unknown",
                seed=1000,
            )

    def test_invalid_seed_is_rejected(self):
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    inject_anomalies(
                        self.data,
                        scenario="normal",
                        seed=seed,
                    )

    def test_missing_feature_column_is_rejected(self):
        invalid_data = self.data.drop(columns=["latency_ms"])

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="normal",
                seed=1000,
            )

    def test_non_numeric_feature_is_rejected(self):
        invalid_data = self.data.copy()
        invalid_data["latency_ms"] = "invalid"

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="normal",
                seed=1000,
            )

    def test_non_finite_feature_is_rejected(self):
        invalid_data = self.data.copy()
        invalid_data.loc[0, "latency_ms"] = np.inf

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="normal",
                seed=1000,
            )

    def test_negative_latency_is_rejected(self):
        invalid_data = self.data.copy()
        invalid_data.loc[0, "latency_ms"] = -1.0

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="normal",
                seed=1000,
            )

    def test_percentage_above_100_is_rejected(self):
        invalid_data = self.data.copy()
        invalid_data.loc[0, "packet_loss_pct"] = 101.0

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="normal",
                seed=1000,
            )

    def test_duplicate_timestamps_are_rejected(self):
        invalid_data = self.data.copy()
        invalid_data.loc[1, "timestamp"] = invalid_data.loc[0, "timestamp"]

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="normal",
                seed=1000,
            )

    def test_unordered_timestamps_are_rejected(self):
        invalid_data = self.data.copy()
        invalid_data.loc[1, "timestamp"], invalid_data.loc[2, "timestamp"] = (
            invalid_data.loc[2, "timestamp"],
            invalid_data.loc[1, "timestamp"],
        )

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="normal",
                seed=1000,
            )

    def test_spike_exceeding_percentage_limit_is_rejected(self):
        invalid_data = self.data.copy()
        invalid_data["packet_loss_pct"] = 30.0

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="sudden_spike",
                seed=1000,
            )

    def test_variance_shift_rejects_zero_baseline_variance(self):
        invalid_data = self.data.copy()
        invalid_data["latency_ms"] = 50.0

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="variance_shift",
                seed=1000,
            )

    def test_multivariate_rejects_constant_feature(self):
        invalid_data = self.data.copy()
        invalid_data["jitter_ms"] = 5.0

        with self.assertRaises(ValueError):
            inject_anomalies(
                invalid_data,
                scenario="multivariate_anomaly",
                seed=1000,
            )


if __name__ == "__main__":
    unittest.main()
