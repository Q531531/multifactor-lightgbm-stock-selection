"""市场门控截面网络、权重加载与日截面训练回归测试。"""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class DailyDataset:
    def __init__(self, data, days, stocks):
        self.data = data
        self.index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-02", periods=days), range(stocks)],
            names=["datetime", "instrument"],
        )

    def get_index(self):
        return self.index

    def __getitem__(self, rows):
        return self.data[rows]

    def __len__(self):
        return len(self.index)


@unittest.skipUnless(importlib.util.find_spec("torch"), "需要 deep 可选依赖")
class MarketTransformerTests(unittest.TestCase):
    def test_cpu_inference_is_deterministic(self):
        import torch
        from original5_strategy.market_transformer import MarketGatedTransformer

        torch.set_num_threads(1)
        torch.manual_seed(23)
        network = MarketGatedTransformer().eval()
        x = torch.randn(7, 8, 221)
        with torch.no_grad():
            first, second = network(x), network(x)
        self.assertEqual(tuple(first.shape), (7,))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_market_gate_rescales_factors_with_unit_mean(self):
        import torch
        from original5_strategy.market_transformer import FactorGate

        torch.manual_seed(5)
        weights = FactorGate(63, 158, 5)(torch.randn(4, 63))
        self.assertTrue((weights > 0).all())
        torch.testing.assert_close(weights.mean(dim=1), torch.ones(4))

    def test_stock_attention_mixes_cross_section(self):
        import torch
        from original5_strategy.market_transformer import AttentionBlock

        torch.manual_seed(7)
        x = torch.randn(5, 8, 16)
        changed = x.clone()
        changed[4] += 1
        for across_stocks, mixed in ((False, False), (True, True)):
            block = AttentionBlock(16, 2, 0.0, across_stocks=across_stocks).eval()
            with torch.no_grad():
                delta = (block(changed)[:4] - block(x)[:4]).abs().max().item()
            self.assertEqual(delta > 0, mixed)

    def test_layer_indexed_weights_load_exactly(self):
        import torch
        from original5_strategy.market_transformer import (
            MarketGatedTransformer,
            convert_state_dict,
            load_network,
        )

        renames = {
            "gate.proj.": "feature_gate.trans.",
            "embed.": "layers.0.",
            "position.table": "layers.1.pe",
            "time_attention.": "layers.2.",
            "stock_attention.": "layers.3.",
            "pooling.proj.": "layers.4.trans.",
            "head.": "layers.5.",
            "query.": "qtrans.",
            "key.": "ktrans.",
            "value.": "vtrans.",
            "norm_in.": "norm1.",
            "norm_out.": "norm2.",
            "feed_forward.": "ffn.",
        }
        torch.manual_seed(11)
        source = MarketGatedTransformer().eval()
        state = {}
        for name, tensor in source.state_dict().items():
            for new, old in renames.items():
                name = name.replace(new, old)
            state[name] = tensor
        self.assertIn("layers.2.qtrans.weight", state)
        self.assertEqual(
            convert_state_dict(source.state_dict()).keys(), source.state_dict().keys()
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.pt"
            torch.save(state, path)
            restored = load_network(path)
        x = torch.randn(9, 8, 221)
        with torch.no_grad():
            self.assertTrue(torch.equal(source(x), restored(x)))


@unittest.skipUnless(importlib.util.find_spec("torch"), "需要 deep 可选依赖")
class CrossSectionTrainingTests(unittest.TestCase):
    def test_small_cross_section_keeps_labels_and_trims_larger_extremes(self):
        import torch
        from original5_strategy.cross_section_training import trim_label_tails

        self.assertTrue(trim_label_tails(torch.arange(12, dtype=torch.float32)).all())
        keep = trim_label_tails(torch.arange(80, dtype=torch.float32))
        self.assertEqual(int(keep.sum()), 76)
        self.assertFalse(keep[:2].any())
        self.assertFalse(keep[-2:].any())

    def test_sampler_length_counts_daily_batches(self):
        from original5_strategy.cross_section_training import DailySampler

        sampler = DailySampler(DailyDataset(None, 3, 4))
        batches = list(sampler)
        self.assertEqual(len(sampler), 3)
        self.assertEqual([len(batch) for batch in batches], [4, 4, 4])
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(12))
        shuffled = list(DailySampler(DailyDataset(None, 3, 4), shuffle=True, seed=1))
        np.testing.assert_array_equal(np.sort(np.concatenate(shuffled)), np.arange(12))

    def test_training_without_threshold_saves_reloadable_model(self):
        import torch
        from original5_strategy.cross_section_training import CrossSectionTrainer

        generator = torch.Generator().manual_seed(23)
        features = torch.randn(12, 2, 3, generator=generator)
        labels = features[:, -1, :].sum(dim=1)
        data = torch.cat([features, labels[:, None, None].expand(-1, 2, 1)], dim=2)
        dataset = DailyDataset(data, 2, 6)

        def build_network():
            return torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(6, 1), torch.nn.Flatten(0)
            )

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "nested/models"
            options = dict(
                epochs=1, lr=0.01, seed=23, save_dir=directory, save_prefix="regression"
            )
            with patch("torch.cuda.is_available", return_value=True):
                trained = CrossSectionTrainer(build_network, **options)
            self.assertEqual(trained.device.type, "cpu")
            weights = trained.fit(dataset)
            self.assertEqual(weights, directory / "regression_23.pt")
            self.assertTrue(weights.is_file())
            first, metrics = trained.predict(dataset)
            self.assertTrue(np.isfinite(first).all())
            self.assertEqual(set(metrics), {"IC", "ICIR", "RankIC", "RankICIR"})
            restored = CrossSectionTrainer(build_network, **options)
            with self.assertRaises(ValueError):
                restored.predict(dataset)
            restored.load(weights)
            second, _ = restored.predict(dataset)
            pd.testing.assert_series_equal(first, second, check_exact=True)

    def test_stop_loss_ends_training_early(self):
        import torch
        from original5_strategy.cross_section_training import CrossSectionTrainer

        data = torch.randn(12, 8, 222, generator=torch.Generator().manual_seed(3))
        with tempfile.TemporaryDirectory() as temp:
            trainer = CrossSectionTrainer(
                epochs=5, lr=1e-4, seed=3, stop_loss=1e9, save_dir=temp
            )
            trainer.fit(DailyDataset(data, 2, 6))
            self.assertEqual(trainer.trained_epochs, 0)
            self.assertTrue(trainer.weights_path.is_file())


if __name__ == "__main__":
    unittest.main()
