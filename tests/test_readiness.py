"""输入格式、时间边界、账本审计与配置回归测试。"""

from pathlib import Path
import tempfile
import unittest
import importlib.util

import numpy as np
import pandas as pd

from original5_strategy.audit import audit_account
from original5_strategy.backtest import BacktestConfig, backtest
from original5_strategy.configuration import load_configuration
from original5_strategy.data import panel_from_long, panel_to_long
from original5_strategy.metrics import account_metrics
from original5_strategy.qlib_data import read_binary, read_qlib_panel
from original5_strategy.base_scores import fit_normalizer, normalize_panel
from test_features import make_panel


class ReadinessTests(unittest.TestCase):
    def test_nondeterministic_model_is_detected(self):
        from original5_strategy.model import predict_ensemble

        class Unstable:
            calls = 0

            def predict(self, x, **kwargs):
                self.calls += 1
                return np.full(len(x), self.calls)

        with self.assertRaises(AssertionError):
            predict_ensemble([Unstable()], np.zeros((3, 2)))

    def test_boolean_strings_are_not_all_true(self):
        frame = panel_to_long(make_panel(40, 2))
        frame["member"] = ["False", "True"] * 40
        panel = panel_from_long(frame)
        self.assertFalse(panel["member"][:, 0].any())
        self.assertTrue(panel["member"][:, 1].all())
        frame.loc[0, "asset"] = None
        with self.assertRaises(ValueError):
            panel_from_long(frame)

    def test_panel_roundtrip_preserves_float32_features(self):
        original = make_panel(40, 2)
        for field in ("open", "high", "low", "close", "volume", "amount", "factor"):
            original[field] = original[field].astype(np.float32)
        recovered = panel_from_long(panel_to_long(original))
        for key in original:
            np.testing.assert_array_equal(original[key], recovered[key])
        self.assertEqual(recovered["open"].dtype, np.float32)

    def test_qlib_offsets_and_historical_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calendars").mkdir()
            (root / "instruments").mkdir()
            (root / "features/aaa").mkdir(parents=True)
            dates = pd.bdate_range("2024-01-02", periods=5)
            (root / "calendars/day.txt").write_text(
                "\n".join(d.strftime("%Y-%m-%d") for d in dates)
            )
            (root / "instruments/csi300.txt").write_text(
                f"AAA\t{dates[1].date()}\t{dates[3].date()}\n"
            )
            for name in ("open", "high", "low", "close", "volume", "amount", "factor"):
                np.array([1, 10, 11, 12, 13], dtype="<f4").tofile(
                    root / f"features/aaa/{name}.day.bin"
                )
            panel = read_qlib_panel(
                root, start=str(dates[0].date()), end=str(dates[-1].date())
            )
            np.testing.assert_array_equal(
                panel["member"][:, 0], [False, True, True, True, False]
            )
            self.assertTrue(np.isnan(panel["close"][0, 0]))
            np.testing.assert_array_equal(panel["close"][1:, 0], [10, 11, 12, 13])
            np.array([10, 1], dtype="<f4").tofile(root / "bad.bin")
            with self.assertRaises(ValueError):
                read_binary(root / "bad.bin", 5)

    def test_normalizer_excludes_future(self):
        rng = np.random.default_rng(23)
        panel = dict(
            dates=pd.bdate_range("2020-01-02", periods=30).to_numpy(),
            member=np.ones((30, 5), bool),
            raw_features=rng.normal(size=(30, 5, 158)).astype(np.float32),
            raw_market=rng.normal(size=(30, 63)).astype(np.float32),
        )
        boundary = str(pd.Timestamp(panel["dates"][20]).date())
        first = fit_normalizer(panel, fit_start="2020-01-01", fit_end=boundary)
        panel["raw_features"][20:] = 9999
        panel["raw_market"][20:] = -9999
        second = fit_normalizer(panel, fit_start="2020-01-01", fit_end=boundary)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        features, market = normalize_panel(panel, first)
        self.assertTrue(np.isfinite(features).all() and np.isfinite(market).all())
        self.assertLessEqual(np.max(abs(features)), 3)

    def test_warmup_dates_do_not_expand_study_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calendars").mkdir()
            (root / "instruments").mkdir()
            dates = pd.bdate_range("2017-12-27", periods=9)
            (root / "calendars/day.txt").write_text(
                "\n".join(dates.strftime("%Y-%m-%d"))
            )
            (root / "instruments/csi300.txt").write_text(
                "OLD\t2017-01-01\t2017-12-29\nAAA\t2017-01-01\t2018-12-31\n"
            )
            panel = read_qlib_panel(root, start="2017-12-27", end="2018-01-08")
            np.testing.assert_array_equal(panel["symbols"], ["AAA"])

    def test_toml_applies_model_account_and_split(self):
        settings = load_configuration(
            Path(__file__).parents[1] / "configs/original_protocol.toml"
        )
        self.assertEqual(settings["account"].topk, 30)
        self.assertEqual(settings["model"].num_leaves, 31)
        self.assertEqual(settings["seeds"], (23, 37, 53, 71))
        self.assertEqual(settings["weight"], 0.05)

    def test_independent_ledger_audit_detects_corruption(self):
        panel = make_panel(55, 4)
        config = BacktestConfig(topk=2)
        scores = {i: np.arange(4, dtype=float) for i in range(31, 54)}
        result = backtest(
            panel,
            scores,
            config,
            start=str(pd.Timestamp(panel["dates"][32]).date()),
            end=str(pd.Timestamp(panel["dates"][-1]).date()),
        )
        self.assertTrue(audit_account(result[0], result[1], panel, config)["passed"])
        altered = result[0].copy()
        altered.iloc[-1, altered.columns.get_loc("cash")] += 100
        with self.assertRaises(AssertionError):
            audit_account(altered, result[1], panel, config)

    def test_empty_trade_path_is_auditable(self):
        panel = make_panel(40, 2)
        config = BacktestConfig()
        ledger, trades, _ = backtest(
            panel,
            {},
            config,
            start=str(pd.Timestamp(panel["dates"][32]).date()),
            end=str(pd.Timestamp(panel["dates"][-1]).date()),
        )
        self.assertTrue(trades.empty)
        self.assertTrue(audit_account(ledger, trades, panel, config)["passed"])
        self.assertEqual(account_metrics(ledger)["cost_total"], 0)

    def test_var_uses_all_days_not_only_negative_days(self):
        r = np.r_[np.full(95, 0.01), [-0.10, -0.08, -0.04, -0.02, -0.01]]
        ledger = pd.DataFrame(
            dict(
                return_=r,
                nav=1e7 * np.cumprod(1 + r),
                cost=np.zeros(100),
                turnover=np.zeros(100),
                max_weight=np.zeros(100),
                blocked_orders=np.zeros(100),
                gross_exposure=np.zeros(100),
                traded_notional=np.zeros(100),
            )
        )
        ledger = ledger.rename(columns={"return_": "return"})
        metrics = account_metrics(ledger)
        self.assertAlmostEqual(
            metrics["var95_daily_loss"], max(0.0, -np.quantile(r, 0.05))
        )
        self.assertAlmostEqual(
            metrics["es95_daily_loss"], -r[r <= np.quantile(r, 0.05)].mean()
        )

    @unittest.skipUnless(importlib.util.find_spec("qlib"), "需要 deep 可选依赖")
    def test_optional_qlib_expression_provider(self):
        from original5_strategy.alpha_features import initialize_provider
        from qlib.contrib.data.loader import Alpha158DL
        from qlib.data import D

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calendars").mkdir()
            (root / "features/aaa").mkdir(parents=True)
            dates = pd.bdate_range("2020-01-02", periods=90)
            (root / "calendars/day.txt").write_text(
                "\n".join(dates.strftime("%Y-%m-%d"))
            )
            for field in ("open", "high", "low", "close", "volume", "amount", "factor"):
                values = (
                    np.ones(90, dtype=np.float32)
                    if field == "factor"
                    else np.linspace(10, 20, 90, dtype=np.float32)
                )
                np.r_[np.float32(0), values].astype("<f4").tofile(
                    root / f"features/aaa/{field}.day.bin"
                )
            initialize_provider(root)
            expressions, names = Alpha158DL.get_feature_config()
            frame = D.features(
                ["AAA"],
                expressions,
                start_time=str(dates[-3].date()),
                end_time=str(dates[-1].date()),
            )
            self.assertEqual(frame.shape, (3, 158))
            self.assertEqual(len(names), 158)


if __name__ == "__main__":
    unittest.main()
