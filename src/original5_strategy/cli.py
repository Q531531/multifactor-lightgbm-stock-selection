"""Command-line interface for score fusion."""

import argparse
from pathlib import Path

import pandas as pd

from .fusion import FusionConfig, fuse_scores


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError("input must be CSV or Parquet")


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in (".parquet", ".pq"):
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("output must be CSV or Parquet")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多因子与 LightGBM 的 95%/5% 横截面排名融合"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weight", type=float, default=0.05)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--asset-col", default="asset")
    parser.add_argument("--multifactor-col", default="multifactor_score")
    parser.add_argument("--model-col", default="model_score")
    parser.add_argument("--output-col", default="fused_score")
    args = parser.parse_args()
    config = FusionConfig(
        weight=args.weight,
        date_col=args.date_col,
        asset_col=args.asset_col,
        multifactor_col=args.multifactor_col,
        model_col=args.model_col,
        output_col=args.output_col,
    )
    _write(fuse_scores(_read(args.input), config), args.output)


if __name__ == "__main__":
    main()
