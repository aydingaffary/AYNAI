import unittest
from dataclasses import FrozenInstanceError

import pandas as pd

from experiments.edge_ai.inspection import (
    AuditCheckResult,
    GroundTruthAuditReport,
    InspectionStatus,
    ManifestValidationResult,
    SeedAuditResult,
    SeedDataset,
)


class TestInspectionModels(unittest.TestCase):

    def test_manifest_validation_result(self):
        result = ManifestValidationResult(
            status=InspectionStatus.PASS,
            expected_seeds=(1000,),
            observed_seeds=(1000,),
            dataset_count=6,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.dataset_count, 6)

    def test_failed_manifest_is_invalid(self):
        result = ManifestValidationResult(
            status=InspectionStatus.FAIL,
            expected_seeds=(1000,),
            observed_seeds=(),
            dataset_count=0,
            missing_seeds=(1000,),
        )

        self.assertFalse(result.is_valid)

    def test_seed_dataset_structure(self):
        frame = pd.DataFrame({"timestamp": []})

        dataset = SeedDataset(
            seed=1000,
            training=frame,
            calibration=frame,
            evaluation_base=frame,
            evaluation={"normal": frame},
            ground_truth={"normal": frame},
            events=pd.DataFrame(),
        )

        self.assertEqual(dataset.seed, 1000)
        self.assertIs(dataset.evaluation_base, frame)
        self.assertIn("normal", dataset.evaluation)
        self.assertIn("normal", dataset.ground_truth)

    def make_check(
        self,
        status: InspectionStatus,
        check_id: str = "GT-01",
    ) -> AuditCheckResult:
        return AuditCheckResult(
            check_id=check_id,
            name="Test check",
            status=status,
            message="Test result",
        )

    def test_audit_check_result_is_immutable(self):
        result = self.make_check(InspectionStatus.PASS)

        with self.assertRaises(FrozenInstanceError):
            result.status = InspectionStatus.FAIL

    def test_seed_audit_status_and_check_counts(self):
        result = SeedAuditResult(
            seed=1000,
            checks=(
                self.make_check(InspectionStatus.PASS, "GT-01"),
                self.make_check(InspectionStatus.WARNING, "GT-02"),
                self.make_check(InspectionStatus.NOT_APPLICABLE, "GT-03"),
            ),
            dataset_count=6,
        )

        self.assertEqual(result.status, InspectionStatus.WARNING)
        self.assertEqual(result.passed_checks, 1)
        self.assertEqual(result.warning_checks, 1)
        self.assertEqual(result.failed_checks, 0)
        self.assertEqual(result.not_applicable_checks, 1)

    def test_seed_audit_fail_has_highest_priority(self):
        result = SeedAuditResult(
            seed=1000,
            checks=(
                self.make_check(InspectionStatus.PASS, "GT-01"),
                self.make_check(InspectionStatus.WARNING, "GT-02"),
                self.make_check(InspectionStatus.FAIL, "GT-03"),
            ),
            dataset_count=6,
        )

        self.assertEqual(result.status, InspectionStatus.FAIL)

    def test_seed_audit_all_not_applicable(self):
        result = SeedAuditResult(
            seed=1000,
            checks=(
                self.make_check(
                    InspectionStatus.NOT_APPLICABLE,
                    "GT-01",
                ),
            ),
            dataset_count=0,
        )

        self.assertEqual(
            result.status,
            InspectionStatus.NOT_APPLICABLE,
        )

    def test_seed_audit_empty_checks(self):
        result = SeedAuditResult(
            seed=1000,
            checks=(),
            dataset_count=0,
        )

        self.assertEqual(
            result.status,
            InspectionStatus.NOT_APPLICABLE,
        )
        self.assertEqual(result.passed_checks, 0)
        self.assertEqual(result.failed_checks, 0)

    def test_ground_truth_report_aggregates_seed_results(self):
        passed_seed = SeedAuditResult(
            seed=1000,
            checks=(self.make_check(InspectionStatus.PASS),),
            dataset_count=6,
        )
        warning_seed = SeedAuditResult(
            seed=1001,
            checks=(self.make_check(InspectionStatus.WARNING),),
            dataset_count=6,
        )
        failed_seed = SeedAuditResult(
            seed=1002,
            checks=(self.make_check(InspectionStatus.FAIL),),
            dataset_count=6,
        )

        report = GroundTruthAuditReport(
            seed_results=(
                passed_seed,
                warning_seed,
                failed_seed,
            ),
        )

        self.assertEqual(report.status, InspectionStatus.FAIL)
        self.assertEqual(report.total_seeds, 3)
        self.assertEqual(report.total_datasets, 18)
        self.assertEqual(report.passed_seeds, 1)
        self.assertEqual(report.warning_seeds, 1)
        self.assertEqual(report.failed_seeds, 1)
        self.assertEqual(report.not_applicable_seeds, 0)

    def test_ground_truth_report_without_failures(self):
        report = GroundTruthAuditReport(
            seed_results=(
                SeedAuditResult(
                    seed=1000,
                    checks=(self.make_check(InspectionStatus.PASS),),
                    dataset_count=6,
                ),
                SeedAuditResult(
                    seed=1001,
                    checks=(
                        self.make_check(
                            InspectionStatus.NOT_APPLICABLE
                        ),
                    ),
                    dataset_count=0,
                ),
            ),
        )

        self.assertEqual(report.status, InspectionStatus.PASS)
        self.assertEqual(report.passed_seeds, 1)
        self.assertEqual(report.not_applicable_seeds, 1)

    def test_empty_ground_truth_report(self):
        report = GroundTruthAuditReport(seed_results=())

        self.assertEqual(
            report.status,
            InspectionStatus.NOT_APPLICABLE,
        )
        self.assertEqual(report.total_seeds, 0)
        self.assertEqual(report.total_datasets, 0)

    def test_ground_truth_report_is_immutable(self):
        report = GroundTruthAuditReport(seed_results=())

        with self.assertRaises(FrozenInstanceError):
            report.seed_results = ()



class TestAuditModelValidation(unittest.TestCase):

    def make_check(self, **overrides):
        values = {
            "check_id": "GT-01",
            "name": "Test check",
            "status": InspectionStatus.PASS,
            "message": "Test result",
        }
        values.update(overrides)
        return AuditCheckResult(**values)

    def make_seed_result(self, **overrides):
        values = {
            "seed": 0,
            "checks": (self.make_check(),),
            "dataset_count": 1,
        }
        values.update(overrides)
        return SeedAuditResult(**values)

    def test_zero_seed_is_valid(self):
        result = self.make_seed_result(seed=0)
        self.assertEqual(result.seed, 0)

    def test_negative_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_seed_result(seed=-1)

    def test_boolean_seed_is_rejected(self):
        with self.assertRaises(TypeError):
            self.make_seed_result(seed=True)

    def test_non_integer_seed_is_rejected(self):
        with self.assertRaises(TypeError):
            self.make_seed_result(seed=1.5)

    def test_negative_dataset_count_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_seed_result(dataset_count=-1)

    def test_boolean_dataset_count_is_rejected(self):
        with self.assertRaises(TypeError):
            self.make_seed_result(dataset_count=True)

    def test_checks_must_be_tuple(self):
        with self.assertRaises(TypeError):
            self.make_seed_result(checks=[self.make_check()])

    def test_checks_must_contain_audit_check_results(self):
        with self.assertRaises(TypeError):
            self.make_seed_result(checks=("invalid",))

    def test_negative_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_seed_result(duration_seconds=-0.1)

    def test_non_finite_duration_is_rejected(self):
        for duration in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    self.make_seed_result(duration_seconds=duration)

    def test_invalid_duration_type_is_rejected(self):
        for duration in (True, "1.5"):
            with self.subTest(duration=duration):
                with self.assertRaises(TypeError):
                    self.make_seed_result(duration_seconds=duration)

    def test_empty_check_fields_are_rejected(self):
        for field in ("check_id", "name", "message"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.make_check(**{field: "  "})

    def test_invalid_check_field_types_are_rejected(self):
        for field in ("check_id", "name", "message"):
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    self.make_check(**{field: None})

    def test_invalid_check_status_is_rejected(self):
        with self.assertRaises(TypeError):
            self.make_check(status="PASS")

    def test_report_seed_results_must_be_tuple(self):
        with self.assertRaises(TypeError):
            GroundTruthAuditReport(seed_results=[])

    def test_report_seed_results_must_contain_seed_audit_results(self):
        with self.assertRaises(TypeError):
            GroundTruthAuditReport(seed_results=("invalid",))

    def test_report_rejects_duplicate_seeds(self):
        first = self.make_seed_result(seed=0)
        second = self.make_seed_result(seed=0)

        with self.assertRaises(ValueError):
            GroundTruthAuditReport(seed_results=(first, second))

    def test_report_rejects_invalid_duration(self):
        for duration in (-1, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    GroundTruthAuditReport(
                        seed_results=(),
                        duration_seconds=duration,
                    )

    def test_report_rejects_invalid_duration_type(self):
        with self.assertRaises(TypeError):
            GroundTruthAuditReport(
                seed_results=(),
                duration_seconds=True,
            )


    def test_very_large_integer_duration_is_valid(self):
        result = self.make_seed_result(duration_seconds=10**1000)
        self.assertEqual(result.duration_seconds, 10**1000)


if __name__ == "__main__":
    unittest.main()
