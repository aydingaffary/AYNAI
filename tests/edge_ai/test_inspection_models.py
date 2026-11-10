import unittest

import pandas as pd

from experiments.edge_ai.inspection.models import (
    InspectionStatus,
    ManifestValidationResult,
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


if __name__ == "__main__":
    unittest.main()
