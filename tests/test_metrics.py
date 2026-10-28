import unittest

import numpy as np
import pandas as pd

from original5_strategy.metrics import account_metrics, daily_cross_sectional_ic


class MetricsTests(unittest.TestCase):
    def test_account_metrics_from_one_consistent_ledger(self):
        returns = np.array([0.01, -0.005, 0.002, -0.003, 0.004])
        nav = 1_000_000 * np.cumprod(1 + returns)
        ledger = pd.DataFrame(
            {
                "return": returns,
                "nav": nav,
                "cost": [10.0] * 5,
                "turnover": [0.1] * 5,
                "max_weight": [0.1] * 5,
                "blocked_orders": [0, 1, 0, 0, 0],
                "gross_exposure": [0.9] * 5,
                "traded_notional": [10_000.0] * 5,
            },
            index=pd.bdate_range("2024-01-02", periods=5),
        )
        metrics = account_metrics(ledger, capital=1_000_000)
        self.assertEqual(metrics["sessions"], 5)
        self.assertEqual(metrics["blocked_orders"], 1)
        self.assertAlmostEqual(metrics["total_return"], np.prod(1 + returns) - 1)

    def test_perfect_daily_rank_ic(self):
        frame = pd.DataFrame(
            {
                "date": np.repeat(pd.to_datetime(["2024-01-02", "2024-01-03"]), 4),
                "fused_score": np.tile([1.0, 2.0, 3.0, 4.0], 2),
                "forward_5d_return": np.tile([10.0, 20.0, 30.0, 40.0], 2),
            }
        )
        daily, summary = daily_cross_sectional_ic(frame)
        np.testing.assert_allclose(daily["RankIC"], 1.0)
        self.assertEqual(summary["RankIC"], 1.0)


if __name__ == "__main__":
    unittest.main()
