"""Tests and fixtures for the Ground Truth auditor."""

from dataclasses import replace
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


    def _get_evt02(self):
        result = self.auditor.audit_seed(
            self.dataset,
            self.manifest,
        )
        return next(
            check for check in result.checks
            if check.check_id == "EVT-02"
        )

    def test_evt02_accepts_valid_event_domains(self):
        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt02_rejects_seed_mismatch(self):
        self.dataset.events.loc[0, "seed"] = self.dataset.seed + 1

        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt02_rejects_non_integer_seed(self):
        self.dataset.events["seed"] = (
            self.dataset.events["seed"].astype(object)
        )
        self.dataset.events.loc[0, "seed"] = (
            float(self.dataset.seed) + 0.5
        )

        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt02_rejects_unknown_scenario(self):
        self.dataset.events.loc[0, "scenario"] = "unknown_scenario"

        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt02_rejects_unknown_severity(self):
        self.dataset.events.loc[0, "severity"] = "critical"

        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt02_rejects_missing_required_column(self):
        self.dataset.events.drop(
            columns=["scenario"],
            inplace=True,
        )

        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt02_runs_without_event_id_column(self):
        self.dataset.events.drop(
            columns=["event_id"],
            inplace=True,
        )

        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt02_accepts_empty_events_with_valid_schema(self):
        self.dataset.events.drop(
            index=self.dataset.events.index,
            inplace=True,
        )

        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt02_reports_independent_row_errors(self):
        self.dataset.events.loc[0, "scenario"] = "unknown_scenario"
        self.dataset.events.loc[1, "severity"] = "critical"

        check = self._get_evt02()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=2", check.actual)


    def _get_evt03(self, dataset=None):
        result = self.auditor.audit_seed(
            dataset or self.dataset,
            self.manifest,
        )
        return next(
            check for check in result.checks
            if check.check_id == "EVT-03"
        )

    def test_evt03_accepts_valid_events_across_scenarios(self):
        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt03_accepts_start_index_zero(self):
        self.dataset.events.loc[0, "start_index"] = 0

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt03_accepts_end_index_equal_to_evaluation_length(self):
        self.dataset.events.loc[0, "end_index"] = EVALUATION_SAMPLES

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt03_rejects_negative_start_index(self):
        self.dataset.events.loc[0, "start_index"] = -1

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=1", check.actual)

    def test_evt03_rejects_end_index_equal_to_start_index(self):
        start = self.dataset.events.loc[0, "start_index"]
        self.dataset.events.loc[0, "end_index"] = start

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_end_index_before_start_index(self):
        self.dataset.events.loc[0, "start_index"] = 4
        self.dataset.events.loc[0, "end_index"] = 3

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_end_index_beyond_evaluation_length(self):
        self.dataset.events.loc[0, "end_index"] = (
            EVALUATION_SAMPLES + 1
        )

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_float_start_index(self):
        self.dataset.events["start_index"] = (
            self.dataset.events["start_index"].astype(object)
        )
        self.dataset.events.loc[0, "start_index"] = 1.5

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_float_end_index(self):
        self.dataset.events["end_index"] = (
            self.dataset.events["end_index"].astype(object)
        )
        self.dataset.events.loc[0, "end_index"] = 3.5

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_string_index(self):
        self.dataset.events["start_index"] = (
            self.dataset.events["start_index"].astype(object)
        )
        self.dataset.events.loc[0, "start_index"] = "1"

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_boolean_index(self):
        self.dataset.events["start_index"] = (
            self.dataset.events["start_index"].astype(object)
        )
        self.dataset.events.loc[0, "start_index"] = True

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_missing_required_column(self):
        self.dataset.events.drop(
            columns=["start_index"],
            inplace=True,
        )

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=1", check.actual)

    def test_evt03_rejects_non_dataframe_events(self):
        invalid_dataset = replace(
            self.dataset,
            events=[],
        )

        check = self._get_evt03(invalid_dataset)

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_unknown_scenario(self):
        self.dataset.events.loc[0, "scenario"] = "unknown_scenario"

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_missing_scenario_evaluation(self):
        scenario = self.dataset.events.loc[0, "scenario"]
        del self.dataset.evaluation[scenario]

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_rejects_non_dataframe_scenario_evaluation(self):
        scenario = self.dataset.events.loc[0, "scenario"]
        self.dataset.evaluation[scenario] = []

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt03_reports_independent_errors_and_continues(self):
        self.dataset.events.loc[0, "scenario"] = "unknown_scenario"
        self.dataset.events.loc[1, "end_index"] = (
            EVALUATION_SAMPLES + 1
        )

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=2", check.actual)

    def test_evt03_reports_multiple_index_errors_for_one_event(self):
        self.dataset.events.loc[0, "start_index"] = -1
        self.dataset.events.loc[0, "end_index"] = (
            EVALUATION_SAMPLES + 1
        )

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=2", check.actual)

    def test_evt03_runs_without_event_id_column(self):
        self.dataset.events.drop(
            columns=["event_id"],
            inplace=True,
        )

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt03_accepts_empty_events_with_valid_schema(self):
        self.dataset.events.drop(
            index=self.dataset.events.index,
            inplace=True,
        )

        check = self._get_evt03()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt03_does_not_modify_source_dataframes(self):
        events_before = self.dataset.events.copy(deep=True)
        evaluations_before = {
            scenario: frame.copy(deep=True)
            for scenario, frame in self.dataset.evaluation.items()
        }

        self._get_evt03()

        pd.testing.assert_frame_equal(
            self.dataset.events,
            events_before,
        )

        self.assertEqual(
            set(self.dataset.evaluation),
            set(evaluations_before),
        )

        for scenario, expected in evaluations_before.items():
            pd.testing.assert_frame_equal(
                self.dataset.evaluation[scenario],
                expected,
            )



    def _get_evt04(self, dataset=None):
        result = self.auditor.audit_seed(
            dataset or self.dataset,
            self.manifest,
        )
        return next(
            check for check in result.checks
            if check.check_id == "EVT-04"
        )

    def _evt04_dataset(self, intervals):
        template = self.dataset.events.iloc[0].copy()
        rows = []

        for position, (scenario, start, end) in enumerate(intervals):
            row = template.copy()
            row["event_id"] = f"evt04-{position}"
            row["scenario"] = scenario
            row["start_index"] = start
            row["end_index"] = end
            rows.append(row)

        events = pd.DataFrame(
            rows,
            columns=self.dataset.events.columns,
        )
        return replace(self.dataset, events=events)

    def test_evt04_accepts_non_overlapping_intervals(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 3),
            ("sudden_spike", 4, 6),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("overlap_pairs=0", check.actual)

    def test_evt04_accepts_adjacent_intervals(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 3),
            ("sudden_spike", 3, 5),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt04_rejects_partial_overlap(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 4),
            ("sudden_spike", 3, 6),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("overlap_pairs=1", check.actual)

    def test_evt04_rejects_nested_intervals(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 7),
            ("sudden_spike", 3, 5),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("overlap_pairs=1", check.actual)

    def test_evt04_rejects_identical_intervals(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 2, 5),
            ("sudden_spike", 2, 5),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("overlap_pairs=1", check.actual)

    def test_evt04_reports_all_overlapping_pairs(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 6),
            ("sudden_spike", 2, 5),
            ("sudden_spike", 4, 7),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("overlap_pairs=3", check.actual)

    def test_evt04_compares_only_events_in_the_same_scenario(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 5),
            ("gradual_drift", 1, 5),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("overlap_pairs=0", check.actual)

    def test_evt04_is_independent_of_dataframe_row_order(self):
        intervals = [
            ("sudden_spike", 1, 6),
            ("sudden_spike", 2, 5),
            ("sudden_spike", 4, 7),
        ]
        dataset = self._evt04_dataset(intervals)
        reordered = replace(
            dataset,
            events=dataset.events.iloc[::-1].copy(),
        )

        first = self._get_evt04(dataset)
        second = self._get_evt04(reordered)

        self.assertEqual(first.status, second.status)
        self.assertEqual(first.actual, second.actual)

        first_details = dict(first.details)
        second_details = dict(second.details)

        self.assertEqual(
            {
                (key, value)
                for key, value in first_details.items()
                if key.startswith("event_")
            },
            {
                (key, value)
                for key, value in second_details.items()
                if key.startswith("event_")
            },
        )

    def test_evt04_skips_non_integral_start_index(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 4),
            ("sudden_spike", 3, 6),
        ])
        dataset.events["start_index"] = (
            dataset.events["start_index"].astype(object)
        )
        dataset.events.loc[0, "start_index"] = 1.5

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("overlap_pairs=0", check.actual)

    def test_evt04_skips_non_integral_end_index(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 4),
            ("sudden_spike", 3, 6),
        ])
        dataset.events["end_index"] = (
            dataset.events["end_index"].astype(object)
        )
        dataset.events.loc[0, "end_index"] = "4"

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("overlap_pairs=0", check.actual)

    def test_evt04_skips_boolean_indices(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 4),
            ("sudden_spike", 3, 6),
        ])
        dataset.events["start_index"] = (
            dataset.events["start_index"].astype(object)
        )
        dataset.events.loc[0, "start_index"] = True

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("overlap_pairs=0", check.actual)

    def test_evt04_skips_unknown_scenario(self):
        dataset = self._evt04_dataset([
            ("unknown_scenario", 1, 4),
            ("sudden_spike", 3, 6),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt04_skips_out_of_bounds_intervals(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, EVALUATION_SAMPLES + 1),
            ("sudden_spike", 3, 6),
        ])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("overlap_pairs=0", check.actual)

    def test_evt04_runs_without_event_id_column(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 4),
            ("sudden_spike", 3, 6),
        ])
        dataset.events.drop(columns=["event_id"], inplace=True)

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.FAIL)
        details = dict(check.details)
        self.assertIn("unavailable", details.values())

    def test_evt04_accepts_empty_events_with_valid_schema(self):
        dataset = replace(
            self.dataset,
            events=self.dataset.events.iloc[0:0].copy(),
        )

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt04_rejects_missing_required_column(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 4),
        ])
        dataset.events.drop(columns=["start_index"], inplace=True)

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt04_rejects_non_dataframe_events(self):
        dataset = replace(self.dataset, events=[])

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.FAIL)

    def test_evt04_skips_missing_scenario_evaluation(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 4),
            ("sudden_spike", 3, 6),
        ])
        del dataset.evaluation["sudden_spike"]

        check = self._get_evt04(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt04_does_not_modify_source_dataframes(self):
        dataset = self._evt04_dataset([
            ("sudden_spike", 1, 4),
            ("sudden_spike", 3, 6),
        ])
        events_before = dataset.events.copy(deep=True)
        evaluations_before = {
            scenario: frame.copy(deep=True)
            for scenario, frame in dataset.evaluation.items()
        }

        self._get_evt04(dataset)

        pd.testing.assert_frame_equal(
            dataset.events,
            events_before,
        )
        self.assertEqual(
            set(dataset.evaluation),
            set(evaluations_before),
        )

        for scenario, expected in evaluations_before.items():
            pd.testing.assert_frame_equal(
                dataset.evaluation[scenario],
                expected,
            )


    def _get_evt05(self, dataset=None):
        result = self.auditor.audit_seed(
            dataset or self.dataset,
            self.manifest,
        )
        return next(
            check for check in result.checks
            if check.check_id == "EVT-05"
        )

    def test_evt05_accepts_valid_event_timestamps(self):
        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("issues=0", check.actual)

    def test_evt05_rejects_mismatched_start_timestamp(self):
        self.dataset.events.loc[0, "start_timestamp"] = (
            "2025-01-01T00:00:00"
        )

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("start_timestamp", dict(check.details).values())

    def test_evt05_rejects_mismatched_end_timestamp(self):
        self.dataset.events.loc[0, "end_timestamp"] = (
            "2025-01-01T00:00:00"
        )

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("end_timestamp", dict(check.details).values())

    def test_evt05_rejects_invalid_timestamp(self):
        self.dataset.events.loc[0, "start_timestamp"] = "not-a-timestamp"

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=1", check.actual)

    def test_evt05_rejects_missing_event_timestamp_column(self):
        self.dataset.events.drop(
            columns=["start_timestamp"],
            inplace=True,
        )

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("start_timestamp", dict(check.details).values())

    def test_evt05_rejects_missing_evaluation_timestamp_column(self):
        scenario = self.dataset.events.loc[0, "scenario"]
        self.dataset.evaluation[scenario].drop(
            columns=["timestamp"],
            inplace=True,
        )

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=1", check.actual)

    def test_evt05_skips_unknown_scenario(self):
        self.dataset.events.loc[0, "scenario"] = "unknown_scenario"

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("issues=0", check.actual)

    def test_evt05_skips_invalid_indices(self):
        self.dataset.events.loc[0, "start_index"] = -1

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("checked_events=4", check.actual)

    def test_evt05_skips_out_of_bounds_indices(self):
        scenario = self.dataset.events.loc[0, "scenario"]
        evaluation_length = len(self.dataset.evaluation[scenario])
        self.dataset.events.loc[0, "start_index"] = evaluation_length

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("checked_events=4", check.actual)
        self.assertIn("issues=0", check.actual)

    def test_evt05_skips_non_integral_indices(self):
        self.dataset.events["start_index"] = (
            self.dataset.events["start_index"].astype(object)
        )
        self.dataset.events.loc[0, "start_index"] = 1.5

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("checked_events=4", check.actual)
        self.assertIn("issues=0", check.actual)

    def test_evt05_skips_missing_scenario_evaluation(self):
        scenario = self.dataset.events.loc[0, "scenario"]
        del self.dataset.evaluation[scenario]

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("checked_events=4", check.actual)

    def test_evt05_accepts_empty_events_with_valid_schema(self):
        dataset = replace(
            self.dataset,
            events=self.dataset.events.iloc[0:0].copy(),
        )

        check = self._get_evt05(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("checked_events=0", check.actual)

    def test_evt05_runs_without_event_id_column(self):
        self.dataset.events.drop(
            columns=["event_id"],
            inplace=True,
        )

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt05_reports_independent_timestamp_errors(self):
        self.dataset.events.loc[0, "start_timestamp"] = "invalid-start"
        self.dataset.events.loc[1, "end_timestamp"] = "invalid-end"

        check = self._get_evt05()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=2", check.actual)

    def test_evt05_does_not_modify_source_dataframes(self):
        events_before = self.dataset.events.copy(deep=True)
        evaluations_before = {
            scenario: frame.copy(deep=True)
            for scenario, frame in self.dataset.evaluation.items()
        }

        self._get_evt05()

        pd.testing.assert_frame_equal(
            self.dataset.events,
            events_before,
        )
        self.assertEqual(
            set(self.dataset.evaluation),
            set(evaluations_before),
        )

        for scenario, expected in evaluations_before.items():
            pd.testing.assert_frame_equal(
                self.dataset.evaluation[scenario],
                expected,
            )


    def _get_evt06(self, dataset=None):
        result = self.auditor.audit_seed(
            dataset or self.dataset,
            self.manifest,
        )
        return next(
            check for check in result.checks
            if check.check_id == "EVT-06"
        )

    def test_evt06_accepts_valid_event_json(self):
        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("issues=0", check.actual)

    def test_evt06_rejects_invalid_affected_features_json(self):
        self.dataset.events.loc[0, "affected_features"] = "[invalid"

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("affected_features", dict(check.details).values())

    def test_evt06_rejects_invalid_injection_parameters_json(self):
        self.dataset.events.loc[0, "injection_parameters"] = "{invalid"

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("injection_parameters", dict(check.details).values())

    def test_evt06_rejects_empty_affected_features(self):
        self.dataset.events.loc[0, "affected_features"] = "[]"

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=1", check.actual)

    def test_evt06_rejects_non_array_affected_features(self):
        self.dataset.events.loc[0, "affected_features"] = (
            '{"latency_ms": true}'
        )

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=1", check.actual)

    def test_evt06_rejects_unknown_feature_names(self):
        self.dataset.events.loc[0, "affected_features"] = (
            '["latency_ms", "unknown_feature"]'
        )

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=1", check.actual)

    def test_evt06_rejects_non_object_injection_parameters(self):
        self.dataset.events.loc[0, "injection_parameters"] = "[1, 2, 3]"

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=1", check.actual)

    def test_evt06_reports_independent_field_errors(self):
        self.dataset.events.loc[0, "affected_features"] = "invalid"
        self.dataset.events.loc[1, "injection_parameters"] = "invalid"

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=2", check.actual)

        fields = [
            value for key, value in check.details
            if key == "field"
        ]
        self.assertIn("affected_features", fields)
        self.assertIn("injection_parameters", fields)

    def test_evt06_skips_unknown_scenarios(self):
        self.dataset.events.loc[0, "scenario"] = "unknown_scenario"

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("checked_events=4", check.actual)
        self.assertIn("issues=0", check.actual)

    def test_evt06_rejects_missing_required_columns(self):
        self.dataset.events.drop(
            columns=["affected_features"],
            inplace=True,
        )

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn(
            "affected_features",
            dict(check.details).values(),
        )

    def test_evt06_accepts_empty_events_with_valid_schema(self):
        dataset = replace(
            self.dataset,
            events=self.dataset.events.iloc[0:0].copy(),
        )

        check = self._get_evt06(dataset)

        self.assertEqual(check.status, InspectionStatus.PASS)
        self.assertIn("checked_events=0", check.actual)

    def test_evt06_runs_without_event_id_column(self):
        self.dataset.events.drop(
            columns=["event_id"],
            inplace=True,
        )

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.PASS)

    def test_evt06_handles_non_string_json_values(self):
        self.dataset.events["injection_parameters"] = (
            self.dataset.events["injection_parameters"].astype(object)
        )
        self.dataset.events.loc[0, "affected_features"] = None
        self.dataset.events.loc[1, "injection_parameters"] = 123

        check = self._get_evt06()

        self.assertEqual(check.status, InspectionStatus.FAIL)
        self.assertIn("issues=2", check.actual)

    def test_evt06_does_not_modify_source_events(self):
        events_before = self.dataset.events.copy(deep=True)

        self._get_evt06()

        pd.testing.assert_frame_equal(
            self.dataset.events,
            events_before,
        )


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
        self.assertEqual(len(result.checks), 48)
        self.assertEqual(result.failed_checks, 48)
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
