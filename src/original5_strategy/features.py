"""可交易性过滤、17 个日频技术特征与五日收益标签。"""

from typing import Mapping

import numpy as np
import pandas as pd

from .data import validate_multifactor_scores


FEATURE_NAMES = (
    "return_1",
    "return_2",
    "return_5",
    "return_10",
    "return_20",
    "return_31",
    "volatility_5",
    "volume_ratio_5",
    "ma_distance_5",
    "volatility_10",
    "volume_ratio_10",
    "ma_distance_10",
    "volatility_20",
    "volume_ratio_20",
    "ma_distance_20",
    "intraday_return",
    "daily_range",
)
MODEL_FEATURES = (*FEATURE_NAMES, "multifactor_rank")


def eligible_mask(panel: Mapping[str, np.ndarray], lookback: int = 32) -> np.ndarray:
    """仅使用截至信号日可知的连续 OHLCVA 与股票池状态。"""

    raw = np.stack(
        [
            np.asarray(panel[name], dtype=float)
            for name in ("open", "high", "low", "close", "volume", "amount")
        ],
        axis=-1,
    )
    valid = (
        np.isfinite(raw).all(axis=2)
        & (raw[:, :, :4] > 0).all(axis=2)
        & (raw[:, :, 4:] > 0).all(axis=2)
        & (np.asarray(panel["factor"], dtype=float) > 0)
    )
    windows = (
        pd.DataFrame(valid.astype(np.int16))
        .rolling(lookback, min_periods=lookback)
        .sum()
        .to_numpy()
        == lookback
    )
    result = windows & np.asarray(panel["member"], dtype=bool)
    result[: lookback - 1] = False
    return result


def technical_feature_arrays(panel: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """计算 17 个日频收益、波动率、量价和区间特征。"""

    # Preserve input precision to avoid rounding changes before feature arithmetic.
    close = np.asarray(panel["close"])
    open_price = np.asarray(panel["open"])
    high = np.asarray(panel["high"])
    low = np.asarray(panel["low"])
    volume = np.asarray(panel["volume"])
    close_frame = pd.DataFrame(close)
    volume_frame = pd.DataFrame(volume)
    daily_return = close_frame.pct_change(fill_method=None)
    result: dict[str, np.ndarray] = {}
    for lag in (1, 2, 5, 10, 20, 31):
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"return_{lag}"] = close / np.roll(close, lag, axis=0) - 1
        result[f"return_{lag}"][:lag] = np.nan
    for lag in (5, 10, 20):
        result[f"volatility_{lag}"] = (
            daily_return.rolling(lag, min_periods=lag).std(ddof=1).to_numpy()
        )
        result[f"volume_ratio_{lag}"] = (
            volume / volume_frame.rolling(lag, min_periods=lag).mean().to_numpy() - 1
        )
        result[f"ma_distance_{lag}"] = (
            close / close_frame.rolling(lag, min_periods=lag).mean().to_numpy() - 1
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        result["intraday_return"] = close / open_price - 1
        result["daily_range"] = (high - low) / close
    return result


def _same_date_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return (
        frame.groupby("date", sort=False)[column]
        .rank(method="average", pct=True)
        .sub(0.5)
    )


def build_demo_multifactor(frame: pd.DataFrame) -> pd.Series:
    """等权技术因子演示基线，用于合成流程验证。"""

    positive = (
        "return_5",
        "return_10",
        "return_20",
        "ma_distance_10",
        "ma_distance_20",
        "intraday_return",
    )
    negative = ("volatility_10", "volatility_20", "daily_range")
    components = [_same_date_rank(frame, name) for name in positive]
    components += [-_same_date_rank(frame, name) for name in negative]
    return pd.concat(components, axis=1).mean(axis=1)


def build_feature_frame(
    panel: Mapping[str, np.ndarray],
    multifactor_scores: pd.DataFrame | None = None,
    *,
    start: str | None = None,
    lookback: int = 32,
    allow_demo_multifactor: bool = False,
) -> pd.DataFrame:
    """从行情面板生成可训练长表。

    五日标签定义为 ``open[t+6] / open[t+1] - 1``，因此信号日为 ``t``，
    标签成熟日为 ``t+6``。
    """

    dates = pd.DatetimeIndex(panel["dates"])
    symbols = np.asarray(panel["symbols"]).astype(str)
    eligible = eligible_mask(panel, lookback)
    if start is not None:
        eligible &= dates.to_numpy()[:, None] >= np.datetime64(start)
    ti, sj = np.where(eligible)
    arrays = technical_feature_arrays(panel)
    frame = pd.DataFrame(
        {
            "date": dates.to_numpy()[ti],
            "asset": symbols[sj],
            "ti": ti,
            "sj": sj,
        }
    )
    for name in FEATURE_NAMES:
        values = arrays[name][ti, sj]
        frame[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
    if multifactor_scores is not None:
        scores = validate_multifactor_scores(multifactor_scores)
        frame = frame.merge(
            scores, on=["date", "asset"], how="left", validate="one_to_one"
        )
        if frame["multifactor_score"].isna().any():
            missing = int(frame["multifactor_score"].isna().sum())
            raise ValueError(f"可交易样本缺少 {missing} 行多因子分数")
        frame["multifactor_source"] = "provided"
    elif allow_demo_multifactor:
        frame["multifactor_score"] = build_demo_multifactor(frame)
        frame["multifactor_source"] = "equal_weight_demo"
    else:
        raise ValueError("必须提供多因子分数；仅演示时可启用 allow_demo_multifactor")
    frame["multifactor_rank"] = _same_date_rank(frame, "multifactor_score")
    end = ti + 6
    known = end < len(dates)
    target = np.full(len(frame), np.nan, dtype=np.float32)
    open_price = np.asarray(panel["open"])
    target[known] = (
        open_price[end[known], sj[known]] / open_price[ti[known] + 1, sj[known]] - 1
    )
    label_end = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
    label_end[known] = dates.to_numpy()[end[known]]
    frame["forward_5d_return"] = target
    frame["label_end"] = label_end
    return frame.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def audit_feature_causality(
    panel: Mapping[str, np.ndarray],
    multifactor_scores: pd.DataFrame | None = None,
    *,
    mutation_start: int,
    lookback: int = 32,
    allow_demo_multifactor: bool = False,
) -> None:
    """改变未来行情，要求更早信号日的特征逐值不变。"""

    original = build_feature_frame(
        panel,
        multifactor_scores,
        lookback=lookback,
        allow_demo_multifactor=allow_demo_multifactor,
    )
    changed = {name: np.asarray(value).copy() for name, value in panel.items()}
    for name in ("open", "high", "low", "close", "volume", "amount", "factor"):
        changed[name][mutation_start:] = np.where(
            np.isfinite(changed[name][mutation_start:]),
            changed[name][mutation_start:] * 7.0 + 11.0,
            changed[name][mutation_start:],
        )
    rebuilt = build_feature_frame(
        changed,
        multifactor_scores,
        lookback=lookback,
        allow_demo_multifactor=allow_demo_multifactor,
    )
    cutoff = pd.Timestamp(panel["dates"][mutation_start - 1])
    columns = ["date", "asset", *MODEL_FEATURES]
    left = original.loc[original.date <= cutoff, columns].reset_index(drop=True)
    right = rebuilt.loc[rebuilt.date <= cutoff, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
