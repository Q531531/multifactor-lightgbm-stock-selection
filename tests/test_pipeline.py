import importlib.util
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import json

import pandas as pd

from original5_strategy.model import RankerConfig
from original5_strategy.pipeline import predict_and_fuse, prepare_dataset, train_models
from original5_strategy.splits import SplitConfig


HAS_OPTIONAL = (
    importlib.util.find_spec("lightgbm") is not None
    and importlib.util.find_spec("pyarrow") is not None
)


@unittest.skipUnless(HAS_OPTIONAL, "需要 train 和 parquet 可选依赖")
class PipelineTests(unittest.TestCase):
    def test_installed_cli_runs_registered_protocol(self):
        from examples.run_synthetic_pipeline import synthetic_market

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            market = synthetic_market(sessions=180, assets=12)
            market.to_parquet(root / "market.parquet", index=False)
            days = pd.DatetimeIndex(sorted(market.date.unique()))
            boundaries = dict(
                train_start=days[31],
                train_end=days[76],
                early_start=days[76],
                early_end=days[106],
                validation_start=days[106],
                validation_end=days[136],
                evaluation_start=days[136],
            )
            text = "[split]\n" + "".join(
                f'{k} = "{v.date()}"\n' for k, v in boundaries.items()
            )
            text += "\n[model]\nseeds=[23]\nn_estimators=8\nnum_leaves=7\nmin_child_samples=5\nearly_stopping_rounds=2\nn_jobs=1\n"
            text += (
                "\n[account]\ncapital=1000000\ntopk=3\nside_cap=0.5\nformal_repeats=4\n"
            )
            (root / "config.toml").write_text(text)
            base = [sys.executable, "-m", "original5_strategy.pipeline"]
            config = ["--config", str(root / "config.toml")]
            commands = [
                [
                    "prepare",
                    "--market",
                    str(root / "market.parquet"),
                    "--output",
                    str(root / "prepared.parquet"),
                    "--allow-demo-multifactor",
                ],
                [
                    "audit",
                    "--prepared",
                    str(root / "prepared.parquet"),
                    "--output",
                    str(root / "audit.json"),
                ],
                [
                    "train",
                    "--prepared",
                    str(root / "prepared.parquet"),
                    "--models-dir",
                    str(root / "models"),
                    "--metadata",
                    str(root / "deep/training.json"),
                ],
                [
                    "predict",
                    "--prepared",
                    str(root / "prepared.parquet"),
                    "--models-dir",
                    str(root / "models"),
                    "--output",
                    str(root / "scores.parquet"),
                ],
                [
                    "backtest",
                    "--market",
                    str(root / "market.parquet"),
                    "--scores",
                    str(root / "scores.parquet"),
                    "--output-dir",
                    str(root / "account"),
                    "--start",
                    str(days[137].date()),
                    "--end",
                    str(days[-1].date()),
                ],
            ]
            for cmd in commands:
                subprocess.run(
                    base + cmd + config, check=True, capture_output=True, text=True
                )
            receipt = json.loads((root / "account/account_receipt.json").read_text())
            self.assertTrue(receipt["independent_account_audit"]["passed"])
            self.assertEqual(receipt["config"]["topk"], 3)
            self.assertEqual(receipt["repeats"], 4)
            self.assertTrue(receipt["all_repeats_exact"])

    def test_small_end_to_end_training_and_prediction(self):
        from examples.run_synthetic_pipeline import synthetic_market

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            market_path = root / "market.parquet"
            prepared_path = root / "prepared.parquet"
            models_dir = root / "models"
            metadata_path = root / "training.json"
            scores_path = root / "scores.parquet"
            synthetic_market(sessions=180, assets=12).to_parquet(
                market_path, index=False
            )
            frame = prepare_dataset(
                market_path,
                prepared_path,
                allow_demo_multifactor=True,
            )
            days = pd.DatetimeIndex(sorted(frame.date.unique()))
            config = SplitConfig(
                train_start=str(days[0].date()),
                train_end=str(days[45].date()),
                early_start=str(days[45].date()),
                early_end=str(days[75].date()),
                validation_start=str(days[75].date()),
                validation_end=str(days[105].date()),
                evaluation_start=str(days[105].date()),
            )
            metadata = train_models(
                prepared_path,
                models_dir,
                metadata_path,
                split_config=config,
                ranker_config=RankerConfig(
                    n_estimators=8,
                    max_depth=3,
                    num_leaves=7,
                    min_child_samples=5,
                    early_stopping_rounds=2,
                    n_jobs=1,
                ),
                seeds=(23,),
            )
            scores = predict_and_fuse(
                prepared_path,
                models_dir,
                scores_path,
                evaluation_start=config.evaluation_start,
            )
            self.assertTrue(metadata["passed"])
            self.assertEqual(len(metadata["model_files"]), 1)
            self.assertFalse(scores.empty)
            self.assertTrue(scores["fused_score"].notna().all())
            with self.assertRaises(ValueError):
                predict_and_fuse(
                    prepared_path,
                    models_dir,
                    scores_path,
                    evaluation_start=config.train_start,
                )
            corrupted = models_dir / "ranker_0.txt"
            original = corrupted.read_bytes()
            corrupted.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                predict_and_fuse(
                    prepared_path,
                    models_dir,
                    scores_path,
                    evaluation_start=config.evaluation_start,
                )
            corrupted.write_bytes(original)
            with self.assertRaises(FileExistsError):
                train_models(
                    prepared_path,
                    models_dir,
                    metadata_path,
                    split_config=config,
                    seeds=(23,),
                )


if __name__ == "__main__":
    unittest.main()
