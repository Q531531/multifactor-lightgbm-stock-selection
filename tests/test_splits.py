import unittest

import numpy as np
import pandas as pd

from original5_strategy.splits import SplitConfig, split_masks, split_receipt


class SplitTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2022-01-03", periods=90)
        self.frame = pd.DataFrame(
            {
                "date": dates,
                "label_end": dates + pd.offsets.BDay(6),
                "forward_5d_return": np.linspace(-0.1, 0.1, len(dates)),
            }
        )
        self.config = SplitConfig(
            train_start=str(dates[0].date()),
            train_end=str(dates[30].date()),
            early_start=str(dates[30].date()),
            early_end=str(dates[50].date()),
            validation_start=str(dates[50].date()),
            validation_end=str(dates[70].date()),
            evaluation_start=str(dates[70].date()),
        )

    def test_labels_mature_before_each_boundary(self):
        masks = split_masks(self.frame, self.config)
        label_end = pd.to_datetime(self.frame.label_end)
        self.assertTrue(
            (label_end[masks["train"]] < pd.Timestamp(self.config.train_end)).all()
        )
        self.assertTrue(
            (label_end[masks["early"]] < pd.Timestamp(self.config.early_end)).all()
        )
        self.assertTrue(
            (
                label_end[masks["validation"]]
                < pd.Timestamp(self.config.validation_end)
            ).all()
        )

    def test_evaluation_is_not_training(self):
        masks = split_masks(self.frame, self.config)
        fitting = masks["train"] | masks["early"] | masks["validation"]
        self.assertFalse((fitting & masks["evaluation"]).any())
        self.assertTrue(split_receipt(self.frame, self.config)["passed"])


if __name__ == "__main__":
    unittest.main()
