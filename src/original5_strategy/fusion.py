"""多因子评分与模型评分的 95%/5% 横截面排名融合。"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FusionConfig:
    """Column names and the model-head weight."""

    weight: float = 0.05
    date_col: str = "date"
    asset_col: str = "asset"
    multifactor_col: str = "multifactor_score"
    model_col: str = "model_score"
    output_col: str = "fused_score"

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")


def _validate_scores(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    asset_col: str | None = None,
) -> None:
    required = list(
        dict.fromkeys([*group_cols, *value_cols] + ([asset_col] if asset_col else []))
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    if frame.empty:
        raise ValueError("score frame is empty")
    if frame[required].isna().any().any():
        raise ValueError("score frame contains missing values")
    values = frame[list(value_cols)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("score columns must contain only finite values")
    if asset_col and frame.duplicated([*group_cols, asset_col]).any():
        raise ValueError("duplicate date/asset rows are not allowed")


def cross_sectional_rank(
    frame: pd.DataFrame,
    value_col: str,
    group_cols: Sequence[str] = ("date",),
) -> pd.Series:
    """Return same-date percentile ranks centered on zero.

    Ranking definition:
    ``groupby(date).rank(method="average", pct=True) - 0.5``.
    """

    _validate_scores(frame, group_cols, [value_col])
    return (
        frame.groupby(list(group_cols), sort=False)[value_col]
        .rank(method="average", pct=True)
        .sub(0.5)
    )


def fuse_scores(
    frame: pd.DataFrame, config: FusionConfig | None = None
) -> pd.DataFrame:
    """Fuse multifactor and model scores after same-date ranking.

    ``fused = (1 - weight) * multifactor_rank + weight * model_rank``
    """

    config = config or FusionConfig()
    _validate_scores(
        frame,
        [config.date_col],
        [config.multifactor_col, config.model_col],
        config.asset_col,
    )
    result = frame.copy()
    result["multifactor_rank"] = cross_sectional_rank(
        result, config.multifactor_col, (config.date_col,)
    )
    result["model_rank"] = cross_sectional_rank(
        result, config.model_col, (config.date_col,)
    )
    result[config.output_col] = (1.0 - config.weight) * result[
        "multifactor_rank"
    ] + config.weight * result["model_rank"]
    return result
