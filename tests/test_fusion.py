import unittest

import numpy as np
import pandas as pd

from original5_strategy import FusionConfig, cross_sectional_rank, fuse_scores


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
                "asset": list("ABCD") * 2,
                "multifactor_score": [4.0, 1.0, 2.0, 3.0, 10.0, 30.0, 20.0, 40.0],
                "model_score": [1.0, 4.0, 2.0, 3.0, 40.0, 10.0, 20.0, 30.0],
            }
        )

    def test_exact_original_weight(self):
        result = fuse_scores(self.frame)
        expected = 0.95 * result["multifactor_rank"] + 0.05 * result["model_rank"]
        np.testing.assert_array_equal(result["fused_score"], expected)

    def test_each_date_is_ranked_independently(self):
        ranks = cross_sectional_rank(self.frame, "multifactor_score")
        np.testing.assert_array_equal(ranks.iloc[:4], [0.5, -0.25, 0.0, 0.25])
        np.testing.assert_array_equal(ranks.iloc[4:], [-0.25, 0.25, 0.0, 0.5])

    def test_positive_affine_transform_does_not_change_fusion(self):
        transformed = self.frame.copy()
        transformed["multifactor_score"] = transformed["multifactor_score"] * 7 + 9
        transformed["model_score"] = transformed["model_score"] * 3 - 4
        np.testing.assert_array_equal(
            fuse_scores(self.frame)["fused_score"],
            fuse_scores(transformed)["fused_score"],
        )

    def test_ties_use_average_percentile_rank(self):
        frame = self.frame.iloc[:4].copy()
        frame["model_score"] = [1.0, 1.0, 3.0, 4.0]
        rank = cross_sectional_rank(frame, "model_score")
        np.testing.assert_array_equal(rank, [-0.125, -0.125, 0.25, 0.5])

    def test_custom_weight(self):
        result = fuse_scores(self.frame, FusionConfig(weight=0.2))
        expected = 0.8 * result["multifactor_rank"] + 0.2 * result["model_rank"]
        np.testing.assert_array_equal(result["fused_score"], expected)

    def test_duplicate_date_asset_is_rejected(self):
        frame = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            fuse_scores(frame)

    def test_missing_and_nonfinite_scores_are_rejected(self):
        for value in (np.nan, np.inf):
            frame = self.frame.copy()
            frame.loc[0, "model_score"] = value
            with self.assertRaises(ValueError):
                fuse_scores(frame)


if __name__ == "__main__":
    unittest.main()
