"""Tests and fixtures for the Ground Truth auditor."""

import unittest
from unittest.mock import patch

import pandas as pd

from experiments.edge_ai.config import SCENARIO_NAMES
from experiments.edge_ai.inspection.models import (
    InspectionStatus,
    SeedDataset,
)
from experiments.edge_ai.inspection.auditor import GroundTruthAuditor
from experiments.edge_ai.pipeline.dataset_builder import (
    EVENT_METADATA_COLUMNS,
    MANIFEST_COLUMNS,
    _serialize_event_metadata,
)
from experiments.edge_ai.simulation.ground_truth import (
    AnomalyEvent,
    build_ground_truth,
)
from experiments.edge_ai.data.telemetry_generator import FEATURE_COLUMNS


SEED = 1000
EVALUATION_SAMPLES = 8

EVENT_SPECS = {
    "sudden_spike": (1, 4, "strong", ("latency_ms", "jitter_ms")),
    "gradual_drift": (1, 5, "medium", ("latency_ms", "jitter_ms")),
    "burst_anomalies": (
        1, 3, "weak",
        ("latency_ms", "jitter_ms", "packet_loss_pct"),
    ),
    "variance_shift": (1, 5, "medium", ("latency_ms", "jitter_ms")),
    "multivariate_anomaly": (
        1, 4, "strong",
        (
            "latency_ms",
            "jitter_ms",
            "packet_loss_pct",
            "throughput_mbps",
        ),
    ),
}


def _telemetry(start, count):
    """Build deterministic telemetry with the production feature schema."""
    data = {
        "timestamp": pd.date_range(
            start=start,
            periods=count,
            freq="30s",
        )
    }

    for index, feature in enumerate(FEATURE_COLUMNS):
        data[feature] = [
            float(index + sample + 1)
            for sample in range(count)
        ]

    return pd.DataFrame(data)


def _build_audit_fixture(seed=SEED):
    """Build one valid seed with six scenarios and five anomaly events."""
    training = _telemetry("2025-01-01T00:00:00", 4)
    calibration = _telemetry("2025-01-01T01:00:00", 3)
    evaluation_base = _telemetry(
        "2025-01-01T02:00:00",
        EVALUATION_SAMPLES,
    )

    evaluations = {}
    ground_truth = {}
    event_records = []
    manifest_rows = []

    for scenario in SCENARIO_NAMES:
        events = ()

        if scenario in EVENT_SPECS:
            start, end, severity, affected_features = EVENT_SPECS[
                scenario
            ]

            events = (
                AnomalyEvent(
                    event_id=f"seed-{seed}-{scenario}-01",
                    start_index=start,
                    end_index=end,
                    anomaly_type=scenario,
                    severity=severity,
                    affected_features=affected_features,
                    injection_parameters={"test_fixture": True},
                ),
            )

        labeled_data, metadata = build_ground_truth(
            evaluation_base,
            seed=seed,
            scenario=scenario,
            phase="Evaluation",
            events=events,
        )

        evaluations[scenario] = evaluation_base.copy(deep=True)
        ground_truth[scenario] = labeled_data
        event_records.extend(metadata)

        manifest_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "training_samples": len(training),
                "calibration_samples": len(calibration),
                "evaluation_samples": EVALUATION_SAMPLES,
                "anomaly_events": len(events),
                "training_file": f"seed_{seed}/training.csv",
                "calibration_file": f"seed_{seed}/calibration.csv",
                "evaluation_base_file": (
                    f"seed_{seed}/evaluation_base.csv"
                ),
                "evaluation_file": (
                    f"seed_{seed}/evaluation/{scenario}.csv"
                ),
                "ground_truth_file": (
                    f"seed_{seed}/ground_truth/{scenario}.csv"
                ),
                "events_file": f"seed_{seed}/metadata/events.csv",
            }
        )

    serialized_events = _serialize_event_metadata(
        event_records,
        seed,
    )

    events_frame = pd.DataFrame(
        serialized_events,
        columns=EVENT_METADATA_COLUMNS,
    )

    dataset = SeedDataset(
        seed=seed,
        training=training,
        calibration=calibration,
        evaluation_base=evaluation_base,
        evaluation=evaluations,
        ground_truth=ground_truth,
        events=events_frame,
    )

    manifest = pd.DataFrame(
        manifest_rows,
        columns=MANIFEST_COLUMNS,
    )

    return dataset, manifest


class TestGroundTruthAuditorFixture(unittest.TestCase):

    def setUp(self):
        self.dataset, self.manifest = _build_audit_fixture()

    def test_fixture_contains_all_scenarios(self):
        self.assertEqual(
            set(self.dataset.evaluation),
            set(SCENARIO_NAMES),
        )
        self.assertEqual(
            set(self.dataset.ground_truth),
            set(SCENARIO_NAMES),
        )
        self.assertEqual(len(self.manifest), len(SCENARIO_NAMES))

    def test_fixture_contains_five_events(self):
        self.assertEqual(len(self.dataset.events), 5)

    def test_normal_scenario_has_no_anomaly_labels(self):
        ground_truth = self.dataset.ground_truth["normal"]

        self.assertTrue(
            (ground_truth["is_anomaly"] == 0).all()
        )
        self.assertTrue(ground_truth["event_id"].isna().all())

    def test_anomalous_scenarios_have_matching_event_metadata(self):
        for scenario in EVENT_SPECS:
            ground_truth = self.dataset.ground_truth[scenario]
            event_rows = self.dataset.events[
                self.dataset.events["scenario"] == scenario
            ]

            self.assertEqual(len(event_rows), 1)
            self.assertEqual(
                int(ground_truth["is_anomaly"].sum()),
                event_rows.iloc[0]["end_index"]
                - event_rows.iloc[0]["start_index"],
            )

    def test_manifest_event_counts_match_fixture(self):
        counts = self.manifest.set_index("scenario")["anomaly_events"]

        self.assertEqual(counts["normal"], 0)

        for scenario in EVENT_SPECS:
            self.assertEqual(counts[scenario], 1)

    def test_fixture_uses_production_event_serialization(self):
        self.assertEqual(
            tuple(self.dataset.events.columns),
            EVENT_METADATA_COLUMNS,
        )

        for value in self.dataset.events["affected_features"]:
            self.assertIsInstance(value, str)

        for value in self.dataset.events["injection_parameters"]:
            self.assertIsInstance(value, str)


class TestEventMetadataAuditor(unittest.TestCase):

    def setUp(self):
        self.dataset, self.manifest = _build_audit_fixture()
        self.auditor = GroundTruthAuditor()

    def _get_evt01(self):
        result = self.auditor.audit_seed(
            self.dataset,
            self.manifest,
        )
        return next(
            check for check in result.checks
            if check.check_id == "EVT-01"
        )

    def test_evt01_accepts_unique_non_empty_event_ids(self):
        check = self._get_evt01()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt01_rejects_empty_event_id(self):
        self.dataset.events.loc[0, "event_id"] = " "

        check = self._get_evt01()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt01_rejects_duplicate_event_ids_across_scenarios(self):
        self.dataset.events.loc[1, "event_id"] = (
            self.dataset.events.loc[0, "event_id"]
        )

        check = self._get_evt01()

        self.assertEqual(check.status, InspectionStatus.FAIL)


class TestGroundTruthAuditorFaultIsolation(unittest.TestCase):

    def setUp(self):
        self.auditor = GroundTruthAuditor()

    def test_audit_preserves_seed_isolation(self):
        first_dataset, first_manifest = _build_audit_fixture(seed=1000)
        second_dataset, second_manifest = _build_audit_fixture(seed=1001)

        manifest = pd.concat(
            [first_manifest, second_manifest],
            ignore_index=True,
        )

        original_audit_seed = self.auditor.audit_seed

        def audit_seed_with_failure(dataset, current_manifest):
            if dataset.seed == 1000:
                raise RuntimeError("simulated seed failure")
            return original_audit_seed(dataset, current_manifest)

        with patch.object(
            self.auditor,
            "audit_seed",
            side_effect=audit_seed_with_failure,
        ):
            report = self.auditor.audit(
                [first_dataset, second_dataset],
                manifest,
            )

        self.assertEqual(
            [result.seed for result in report.seed_results],
            [1000, 1001],
        )
        self.assertEqual(
            report.seed_results[0].status,
            InspectionStatus.FAIL,
        )
        self.assertEqual(
            report.seed_results[1].status,
            InspectionStatus.PASS,
        )

    def test_failed_seed_generates_complete_failure_report(self):
        dataset, manifest = _build_audit_fixture(seed=1000)

        with patch.object(
            self.auditor,
            "audit_seed",
            side_effect=RuntimeError("simulated seed failure"),
        ):
            report = self.auditor.audit([dataset], manifest)

        result = report.seed_results[0]

        self.assertEqual(result.seed, 1000)
        self.assertEqual(result.dataset_count, len(SCENARIO_NAMES))
        self.assertEqual(len(result.checks), 43)
        self.assertEqual(result.failed_checks, 43)
        self.assertTrue(
            all(
                check.status == InspectionStatus.FAIL
                for check in result.checks
            )
        )
        self.assertTrue(
            all(
                "simulated seed failure" in check.message
                for check in result.checks
            )
        )

    def test_audit_preserves_seed_order(self):
        first_dataset, first_manifest = _build_audit_fixture(seed=1002)
        second_dataset, second_manifest = _build_audit_fixture(seed=1000)

        manifest = pd.concat(
            [first_manifest, second_manifest],
            ignore_index=True,
        )

        report = self.auditor.audit(
            [first_dataset, second_dataset],
            manifest,
        )

        self.assertEqual(
            [result.seed for result in report.seed_results],
            [1002, 1000],
        )

    def test_audit_rejects_duplicate_seeds(self):
        first_dataset, first_manifest = _build_audit_fixture(seed=1000)
        second_dataset, second_manifest = _build_audit_fixture(seed=1000)

        manifest = pd.concat(
            [first_manifest, second_manifest],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            self.auditor.audit(
                [first_dataset, second_dataset],
                manifest,
            )

    def test_audit_rejects_invalid_dataset_type(self):
        _, manifest = _build_audit_fixture()

        with self.assertRaisesRegex(TypeError, "SeedDataset"):
            self.auditor.audit([object()], manifest)


if __name__ == "__main__":
    unittest.main()
