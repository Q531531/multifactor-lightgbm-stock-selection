import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from original5_strategy import BacktestConfig, backtest


class BacktestTests(unittest.TestCase):
    def setUp(self):
        sessions = 25
        self.dates = pd.bdate_range("2024-01-02", periods=sessions).to_numpy()
        self.panel = {
            "dates": self.dates,
            "symbols": np.array(["A", "B"]),
            "open": np.full((sessions, 2), 10.0),
            "close": np.full((sessions, 2), 10.0),
            "amount": np.full((sessions, 2), 1_000_000.0),
            "factor": np.ones((sessions, 2)),
            "member": np.ones((sessions, 2), dtype=bool),
        }
        self.config = BacktestConfig(
            capital=100_000,
            topk=1,
            min_hold=1,
            retention=1.0,
            resize_band=0.0,
            side_cap=1.0,
            commission=0.0,
            slippage=0.0,
            stamp_tax=0.0,
            max_participation=1.0,
            lot=100,
        )

    def test_signal_close_executes_next_open(self):
        scores = {19: np.array([1.0, 0.0])}
        ledger, trades, decisions = backtest(
            self.panel,
            scores,
            self.config,
            start=str(pd.Timestamp(self.dates[20]).date()),
            end=str(pd.Timestamp(self.dates[21]).date()),
        )
        self.assertEqual(
            pd.Timestamp(trades.iloc[0]["signal_date"]), pd.Timestamp(self.dates[19])
        )
        self.assertEqual(
            pd.Timestamp(trades.iloc[0]["date"]), pd.Timestamp(self.dates[20])
        )
        self.assertEqual(trades.iloc[0]["symbol"], "A")
        self.assertEqual(len(ledger), 2)
        self.assertEqual(len(decisions), 1)

    def test_repeated_runs_are_exact(self):
        scores = {
            19: np.array([1.0, 0.0]),
            20: np.array([0.0, 1.0]),
            21: np.array([0.0, 1.0]),
        }
        args = dict(
            panel=self.panel,
            scores=scores,
            config=self.config,
            start=str(pd.Timestamp(self.dates[20]).date()),
            end=str(pd.Timestamp(self.dates[22]).date()),
        )
        first = backtest(**args)
        second = backtest(**args)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right, check_exact=True)

    def test_locked_inventory_cannot_be_replaced_early(self):
        scores = {19: np.array([1.0, 0.0])}
        scores.update({i: np.array([0.0, 1.0]) for i in range(20, 24)})
        _, trades, _ = backtest(
            self.panel,
            scores,
            replace(self.config, min_hold=3),
            start=str(pd.Timestamp(self.dates[20]).date()),
            end=str(pd.Timestamp(self.dates[24]).date()),
        )
        sells = trades[trades.side == "sell"]
        self.assertFalse(sells.empty)
        self.assertEqual(pd.Timestamp(sells.date.min()), pd.Timestamp(self.dates[23]))

    def test_open_gap_blocks_entry(self):
        panel = {k: v.copy() for k, v in self.panel.items()}
        panel["open"][20, 0] = 20.0
        ledger, trades, _ = backtest(
            panel,
            {19: np.array([1.0, 0.0])},
            self.config,
            start=str(pd.Timestamp(self.dates[20]).date()),
            end=str(pd.Timestamp(self.dates[21]).date()),
        )
        self.assertTrue(trades.empty)
        self.assertGreater(ledger.blocked_orders.sum(), 0)
        np.testing.assert_array_equal(ledger.cash, self.config.capital)


if __name__ == "__main__":
    unittest.main()
