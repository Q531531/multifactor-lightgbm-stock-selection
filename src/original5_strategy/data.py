"""行情长表、股票池和 NumPy 面板之间的转换。"""

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


MARKET_FIELDS = ("open", "high", "low", "close", "volume", "amount", "factor")
REQUIRED_LONG_COLUMNS = ("date", "asset", *MARKET_FIELDS, "member")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"不支持的表格格式: {suffix}")
    return frame


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in (".parquet", ".pq"):
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"不支持的表格格式: {suffix}")


def validate_long_market(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_LONG_COLUMNS if column not in frame]
    if missing:
        raise KeyError(f"行情缺少字段: {missing}")
    if frame.empty:
        raise ValueError("行情表为空")
    result = frame.copy()
    if result[["date", "asset", "member"]].isna().any().any():
        raise ValueError("date/asset/member 不允许为空")
    result["date"] = pd.to_datetime(result["date"])
    result["asset"] = result["asset"].astype(str)
    if result.duplicated(["date", "asset"]).any():
        raise ValueError("行情存在重复的 date/asset")
    if result[["date", "asset"]].isna().any().any():
        raise ValueError("date/asset 不允许为空")

    def membership(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in (
            "true",
            "false",
            "0",
            "1",
        ):
            return value.strip().lower() in ("true", "1")
        raise ValueError(f"member 必须为布尔值或 0/1，收到: {value!r}")

    result["member"] = result["member"].map(membership).astype(bool)
    return result.sort_values(["date", "asset"], kind="mergesort").reset_index(
        drop=True
    )


def panel_from_long(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """把行情长表转换为完整的 ``[date, asset]`` 面板。"""

    frame = validate_long_market(frame)
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    symbols = np.array(sorted(frame["asset"].unique()), dtype=str)
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "asset"])
    aligned = frame.set_index(["date", "asset"]).reindex(index)
    panel = {
        "dates": dates.to_numpy(),
        "symbols": symbols,
        "member": aligned["member"]
        .fillna(False)
        .to_numpy(bool)
        .reshape(len(dates), len(symbols)),
    }
    for field in MARKET_FIELDS:
        values = pd.to_numeric(aligned[field], errors="raise")
        dtype = np.float32 if frame[field].dtype == np.float32 else np.float64
        panel[field] = values.to_numpy(dtype=dtype).reshape(len(dates), len(symbols))
    return panel


def panel_to_long(panel: Mapping[str, np.ndarray]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = np.asarray(panel["symbols"]).astype(str)
    rows = {
        "date": np.repeat(dates.to_numpy(), len(symbols)),
        "asset": np.tile(symbols, len(dates)),
    }
    for field in MARKET_FIELDS + ("member",):
        rows[field] = np.asarray(panel[field]).reshape(-1)
    return pd.DataFrame(rows)


def save_panel(panel: Mapping[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **panel)


def load_panel(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def validate_multifactor_scores(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("date", "asset", "multifactor_score")
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"多因子分数缺少字段: {missing}")
    result = frame[list(required)].copy()
    result["date"] = pd.to_datetime(result["date"])
    result["asset"] = result["asset"].astype(str)
    if result.duplicated(["date", "asset"]).any():
        raise ValueError("多因子分数存在重复的 date/asset")
    values = result["multifactor_score"].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("多因子分数必须是有限数值")
    return result.sort_values(["date", "asset"], kind="mergesort").reset_index(
        drop=True
    )
