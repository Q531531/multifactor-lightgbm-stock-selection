"""时间边界、独立账本核对与运行收据。"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .features import MODEL_FEATURES
from .splits import SplitConfig, split_masks, split_receipt


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(current) for key, current in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(current) for current in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, (pd.Timestamp, datetime, Path)):
            return str(item)
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            clean(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def audit_prepared_frame(
    frame: pd.DataFrame, config: SplitConfig | None = None
) -> dict:
    config = config or SplitConfig()
    required = {
        "date",
        "asset",
        "label_end",
        "forward_5d_return",
        "multifactor_score",
        *MODEL_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"训练表缺少字段: {missing}")
    if frame.duplicated(["date", "asset"]).any():
        raise AssertionError("训练表存在重复 date/asset")
    feature_values = frame[list(MODEL_FEATURES)].to_numpy(float)
    if not np.isfinite(feature_values).all():
        raise AssertionError("模型特征含非有限值")
    masks = split_masks(frame, config)
    train_end = pd.Timestamp(config.train_end)
    early_end = pd.Timestamp(config.early_end)
    validation_end = pd.Timestamp(config.validation_end)
    label_end = pd.to_datetime(frame["label_end"])
    assert (label_end[masks["train"]] < train_end).all()
    assert (label_end[masks["early"]] < early_end).all()
    assert (label_end[masks["validation"]] < validation_end).all()
    assert not (
        masks["evaluation"] & (masks["train"] | masks["early"] | masks["validation"])
    ).any()
    receipt = split_receipt(frame, config)
    receipt.update(
        passed=True,
        rows=len(frame),
        unique_rows=True,
        all_model_features_finite=True,
        label_maturity_boundaries_passed=True,
        model_features=list(MODEL_FEATURES),
        multifactor_sources=frame.get("multifactor_source", pd.Series(["unknown"]))
        .value_counts()
        .to_dict(),
    )
    return receipt


def artifact_manifest(paths: Iterable[str | Path], root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    result = {}
    for raw in paths:
        path = Path(raw).resolve()
        result[str(path.relative_to(root))] = sha256(path)
    return dict(sorted(result.items()))


def run_receipt(
    *,
    command: str,
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    root: str | Path,
    configuration,
) -> dict:
    return {
        "passed": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "configuration": asdict(configuration)
        if hasattr(configuration, "__dataclass_fields__")
        else configuration,
        "inputs": artifact_manifest(inputs, root),
        "outputs": artifact_manifest(outputs, root),
    }


def audit_account(ledger, trades, panel, config):
    """从成交事件独立重建现金、库存、费用、仓龄和每日净值。"""
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = pd.Index(panel["symbols"])
    mark = pd.DataFrame(panel["close"].astype(float)).ffill().fillna(0).to_numpy()
    h = np.zeros(len(symbols))
    age = np.full(len(symbols), -1000)
    cash = nav = config.capital
    grouped = (
        {} if trades.empty else {pd.Timestamp(d): g for d, g in trades.groupby("date")}
    )
    minimum = cash
    for date, row in ledger.iterrows():
        t = dates.get_loc(date)
        fees = amount_sum = 0.0
        for fill in grouped.get(date, pd.DataFrame()).itertuples(index=False):
            j = symbols.get_loc(fill.symbol)
            if pd.Timestamp(fill.signal_date) != dates[t - 1]:
                raise AssertionError("成交信号不是前一交易日")
            np.testing.assert_allclose(
                h[j], fill.position_before, atol=1e-8, rtol=1e-12
            )
            amount = fill.units * float(panel["open"][t, j])
            commission = max(5, amount * config.commission)
            slip = amount * config.slippage
            tax = amount * config.stamp_tax if fill.side == "sell" else 0.0
            np.testing.assert_allclose(
                [amount, commission, slip, tax],
                [fill.notional, fill.commission, fill.slippage, fill.stamp_tax],
                atol=1e-7,
                rtol=1e-12,
            )
            cost = commission + slip + tax
            if fill.side == "buy":
                cash -= amount + cost
                h[j] += fill.units
                age[j] = t
                np.testing.assert_allclose(
                    fill.raw_shares / config.lot,
                    round(fill.raw_shares / config.lot),
                    atol=1e-8,
                )
            elif fill.side == "sell":
                if t - age[j] < config.min_hold:
                    raise AssertionError("卖出违反最短持仓期")
                cash += amount - cost
                h[j] -= fill.units
            else:
                raise AssertionError("未知成交方向")
            if (
                cash < -1e-5
                or h[j] < -1e-8
                or fill.participation > config.max_participation + 1e-12
            ):
                raise AssertionError("现金、库存或参与率不满足约束")
            minimum = min(minimum, cash)
            np.testing.assert_allclose(
                [cash, h[j]],
                [fill.cash_after, fill.position_after],
                atol=1e-6,
                rtol=1e-12,
            )
            fees += cost
            amount_sum += amount
        daily_fills = grouped.get(date)
        if daily_fills is not None:
            for side, subset in daily_fills.groupby("side"):
                if subset.notional.sum() > nav * config.side_cap + 1e-5:
                    raise AssertionError("成交超出单边预算")
        newnav = cash + h @ mark[t]
        np.testing.assert_allclose(
            [newnav, cash, newnav / nav - 1, fees, amount_sum],
            [row.nav, row.cash, row["return"], row.cost, row.traded_notional],
            atol=1e-6,
            rtol=1e-12,
        )
        if hashlib.sha256(h.tobytes()).hexdigest() != row.position_hash:
            raise AssertionError("独立重建的持仓与账本记录不一致")
        nav = newnav
    return dict(
        passed=True,
        sessions=len(ledger),
        fills=len(trades),
        minimum_cash=minimum,
        cash_inventory_fees_reconstructed=True,
        holding_period_and_budget_checked=True,
    )
