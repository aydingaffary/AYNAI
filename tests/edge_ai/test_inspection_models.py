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
            evaluation={"normal": frame},
            ground_truth={"normal": frame},
            events=pd.DataFrame(),
        )

        self.assertEqual(dataset.seed, 1000)
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


if __name__ == "__main__":
    unittest.main()
