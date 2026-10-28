"""使用合成行情执行数据准备、模型训练、融合预测和账户回测。"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from original5_strategy.backtest import BacktestConfig
from original5_strategy.model import RankerConfig
from original5_strategy.pipeline import (
    predict_and_fuse,
    prepare_dataset,
    run_account,
    train_models,
)
from original5_strategy.splits import SplitConfig


def synthetic_market(
    seed: int = 20261028, sessions: int = 240, assets: int = 24
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=sessions)
    symbols = np.array([f"S{i:03d}" for i in range(assets)])
    common = rng.normal(0.0003, 0.008, (sessions, 1))
    idiosyncratic = rng.normal(0.0, 0.012, (sessions, assets))
    close = 20 * np.exp(np.cumsum(common + idiosyncratic, axis=0))
    overnight = rng.normal(0.0, 0.003, (sessions, assets))
    open_price = close * np.exp(overnight)
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.02, close.shape))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.02, close.shape))
    volume = rng.lognormal(13.0, 0.25, close.shape)
    amount = volume * close / 1000
    return pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), assets),
            "asset": np.tile(symbols, sessions),
            "open": open_price.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": close.ravel(),
            "volume": volume.ravel(),
            "amount": amount.ravel(),
            "factor": np.ones(close.size),
            "member": np.ones(close.size, dtype=bool),
        }
    )


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    market = synthetic_market()
    market_path = output / "market.parquet"
    prepared_path = output / "prepared.parquet"
    models_dir = output / "models"
    metadata_path = output / "training.json"
    scores_path = output / "scores.parquet"
    market.to_parquet(market_path, index=False)
    prepared = prepare_dataset(
        market_path,
        prepared_path,
        allow_demo_multifactor=True,
        start="2020-02-17",
    )
    days = pd.DatetimeIndex(sorted(prepared["date"].unique()))
    split = SplitConfig(
        train_start=str(days[0].date()),
        train_end=str(days[75].date()),
        early_start=str(days[75].date()),
        early_end=str(days[115].date()),
        validation_start=str(days[115].date()),
        validation_end=str(days[155].date()),
        evaluation_start=str(days[155].date()),
    )
    ranker = RankerConfig(
        n_estimators=20,
        max_depth=3,
        num_leaves=7,
        min_child_samples=10,
        early_stopping_rounds=4,
        n_jobs=1,
    )
    train_models(
        prepared_path,
        models_dir,
        metadata_path,
        split_config=split,
        ranker_config=ranker,
        seeds=(23, 37),
    )
    scores = predict_and_fuse(
        prepared_path,
        models_dir,
        scores_path,
        evaluation_start=split.evaluation_start,
    )
    panel_dates = pd.DatetimeIndex(market["date"].unique()).sort_values()
    first_signal = pd.Timestamp(scores["date"].min())
    start = panel_dates[panel_dates.get_loc(first_signal) + 1]
    run_account(
        market_path,
        scores_path,
        output / "account",
        start=str(start.date()),
        end=str(panel_dates[-1].date()),
        repeats=4,
        config=BacktestConfig(
            capital=1_000_000,
            topk=10,
            min_hold=5,
            retention=2.0,
            resize_band=0.2,
            side_cap=0.2,
            commission=0.0003,
            slippage=0.001,
            stamp_tax=0.0005,
            max_participation=0.01,
            lot=100,
        ),
    )
    print(f"合成端到端流水线完成: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("synthetic_output"))
    run(parser.parse_args().output)
