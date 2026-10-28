import unittest

import numpy as np
import pandas as pd

from original5_strategy.features import (
    FEATURE_NAMES,
    audit_feature_causality,
    build_feature_frame,
    eligible_mask,
)


def make_panel(sessions=55, assets=4):
    dates = pd.bdate_range("2022-01-03", periods=sessions).to_numpy()
    t = np.arange(sessions)[:, None]
    j = np.arange(assets)[None, :]
    close = 10 + 0.05 * t + 0.2 * j
    return {
        "dates": dates,
        "symbols": np.array([f"S{k}" for k in range(assets)]),
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full((sessions, assets), 1_000_000.0) + t * 100 + j,
        "amount": np.full((sessions, assets), 100_000.0),
        "factor": np.ones((sessions, assets)),
        "member": np.ones((sessions, assets), dtype=bool),
    }


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.panel = make_panel()
        dates = pd.DatetimeIndex(self.panel["dates"])
        self.multifactor = pd.DataFrame(
            {
                "date": np.repeat(dates.to_numpy(), len(self.panel["symbols"])),
                "asset": np.tile(self.panel["symbols"], len(dates)),
                "multifactor_score": np.tile(np.arange(4, dtype=float), len(dates)),
            }
        )

    def test_eligibility_requires_known_32_day_prefix(self):
        mask = eligible_mask(self.panel, 32)
        self.assertFalse(mask[:31].any())
        self.assertTrue(mask[31:].all())

    def test_feature_and_label_definition(self):
        frame = build_feature_frame(self.panel, self.multifactor)
        self.assertTrue(set(FEATURE_NAMES) <= set(frame))
        row = frame[(frame.ti == 31) & (frame.sj == 0)].iloc[0]
        expected = self.panel["open"][37, 0] / self.panel["open"][32, 0] - 1
        self.assertAlmostEqual(row.forward_5d_return, expected, places=7)
        self.assertEqual(
            pd.Timestamp(row.label_end), pd.Timestamp(self.panel["dates"][37])
        )
        self.assertAlmostEqual(
            row.return_5,
            self.panel["close"][31, 0] / self.panel["close"][26, 0] - 1,
            places=7,
        )

    def test_future_mutation_does_not_change_earlier_features(self):
        audit_feature_causality(
            self.panel,
            self.multifactor,
            mutation_start=45,
        )


if __name__ == "__main__":
    unittest.main()
