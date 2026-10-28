"""Next-open, cash-funded execution for fused daily scores."""

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    capital: float = 10_000_000
    topk: int = 30
    min_hold: int = 5
    retention: float = 2.0
    resize_band: float = 0.2
    side_cap: float = 0.10
    commission: float = 0.0003
    slippage: float = 0.001
    stamp_tax: float = 0.0005
    max_participation: float = 0.01
    lot: int = 100

    def __post_init__(self):
        if self.capital <= 0 or self.topk < 1 or self.min_hold < 1 or self.lot < 1:
            raise ValueError("资金、持仓数、持仓期和整手数量必须为正")
        if not 0 < self.side_cap <= 1 or not 0 < self.max_participation <= 1:
            raise ValueError("单边预算与参与率必须在 (0, 1] 内")
        if (
            self.retention < 1
            or self.resize_band < 0
            or min(self.commission, self.slippage, self.stamp_tax) < 0
        ):
            raise ValueError("留存和费用参数无效")


def scores_from_frame(
    frame: pd.DataFrame,
    dates: np.ndarray,
    symbols: np.ndarray,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    score_col: str = "fused_score",
) -> dict[int, np.ndarray]:
    """Convert a long score table to the backtester's daily arrays."""

    required = [date_col, asset_col, score_col]
    if any(column not in frame for column in required):
        raise KeyError(f"required columns: {required}")
    if frame.duplicated([date_col, asset_col]).any():
        raise ValueError("duplicate date/asset rows are not allowed")
    date_index = pd.Index(pd.to_datetime(dates))
    symbol_index = pd.Index(symbols)
    if not np.isfinite(frame[score_col].to_numpy(float)).all():
        raise ValueError("交易分数必须为有限值")
    result: dict[int, np.ndarray] = {}
    for date, group in frame.groupby(date_col, sort=False):
        ti = int(date_index.get_loc(pd.Timestamp(date)))
        sj = symbol_index.get_indexer(group[asset_col])
        if (sj < 0).any():
            raise KeyError("score frame contains an unknown asset")
        values = np.full(len(symbols), np.nan)
        values[sj] = group[score_col].to_numpy(dtype=float)
        result[ti] = values
    return result


def _validate_panel(panel: Mapping[str, np.ndarray]) -> None:
    required = ("dates", "symbols", "open", "close", "amount", "factor", "member")
    missing = [name for name in required if name not in panel]
    if missing:
        raise KeyError(f"panel is missing: {missing}")
    shape = np.asarray(panel["open"]).shape
    if len(shape) != 2:
        raise ValueError("market arrays must have shape [date, asset]")
    for name in ("close", "amount", "factor", "member"):
        if np.asarray(panel[name]).shape != shape:
            raise ValueError(f"{name} shape does not match open")
    if len(panel["dates"]) != shape[0] or len(panel["symbols"]) != shape[1]:
        raise ValueError("date/symbol dimensions do not match market arrays")
    dates = pd.DatetimeIndex(panel["dates"])
    if (
        not len(dates)
        or not shape[1]
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("面板日期必须非空、递增且唯一")
    if pd.Index(panel["symbols"]).has_duplicates:
        raise ValueError("证券代码必须唯一")


def backtest(
    panel: Mapping[str, np.ndarray],
    scores: Mapping[int, np.ndarray],
    config: BacktestConfig | None = None,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate next-open execution with a cash-funded account.

    Scores from signal close ``t`` are used at open ``t+1``. Trading is
    cash-funded and enforces lot size, T+1, participation, side budget,
    commission, slippage and stamp tax.
    """

    _validate_panel(panel)
    config = config or BacktestConfig()
    dates = pd.DatetimeIndex(panel["dates"])
    if pd.Timestamp(start) > pd.Timestamp(end) or dates.searchsorted(
        start
    ) >= dates.searchsorted(end, side="right"):
        raise ValueError("回测区间为空或顺序错误")
    if dates.searchsorted(start) == 0:
        raise ValueError("回测开始前至少需要一个交易日作为信号和估值前缀")
    symbols = np.asarray(panel["symbols"])
    n = len(symbols)
    close = np.asarray(panel["close"], dtype=float)
    open_price = np.asarray(panel["open"], dtype=float)
    factor = np.asarray(panel["factor"], dtype=float)
    member = np.asarray(panel["member"], dtype=bool)
    mark = pd.DataFrame(close).ffill().fillna(0).to_numpy()
    liquidity = (
        pd.DataFrame(panel["amount"]).rolling(20, min_periods=15).median().to_numpy()
        * 1000
    )
    raw_open = np.divide(
        open_price, factor, out=np.full_like(open_price, np.nan), where=factor > 0
    )
    raw_close = np.divide(
        close, factor, out=np.full_like(close, np.nan), where=factor > 0
    )
    holdings = np.zeros(n)
    age = np.full(n, -1000)
    cash = nav = config.capital
    ledger_rows: list[dict] = []
    trades: list[dict] = []
    decisions: list[dict] = []

    for t in range(dates.searchsorted(start), dates.searchsorted(end, side="right")):
        signal_index = t - 1
        cost = traded_notional = 0.0
        blocked = 0
        values = scores.get(signal_index)
        opening_mark = np.where(
            np.isfinite(open_price[t]) & (open_price[t] > 0),
            open_price[t],
            mark[signal_index],
        )
        if values is not None:
            values = np.asarray(values, dtype=float)
            if values.shape != (n,):
                raise ValueError(
                    "each daily score vector must have one value per asset"
                )
            valid = member[signal_index] & np.isfinite(values)
            ranked = np.flatnonzero(valid)
            ranked = ranked[np.lexsort((ranked, -values[ranked]))]
            rank = {int(asset): position for position, asset in enumerate(ranked)}
            locked = list(
                np.flatnonzero((holdings > 1e-10) & (t - age < config.min_hold))
            )
            retained = sorted(
                [
                    asset
                    for asset in ranked
                    if holdings[asset] > 0
                    and rank[asset] < config.topk * config.retention
                    and asset not in locked
                ],
                key=lambda asset: rank[asset],
            )
            selected = locked + retained[: max(0, config.topk - len(locked))]
            selected += [asset for asset in ranked if asset not in selected][
                : max(0, config.topk - len(selected))
            ]
            desired = np.zeros(n)
            target_value = nav * 0.99 / config.topk
            for asset in selected:
                if np.isfinite(raw_open[t, asset]) and raw_open[t, asset] > 0:
                    desired[asset] = (
                        np.floor(target_value / raw_open[t, asset] / config.lot)
                        * config.lot
                        / factor[t, asset]
                    )
                else:
                    desired[asset] = holdings[asset]
                if (
                    holdings[asset] > 0
                    and abs(desired[asset] - holdings[asset])
                    * mark[signal_index, asset]
                    < config.resize_band * target_value
                ):
                    desired[asset] = holdings[asset]
                if t - age[asset] < config.min_hold:
                    desired[asset] = max(desired[asset], holdings[asset])
            with np.errstate(divide="ignore", invalid="ignore"):
                gap = raw_open[t] / raw_close[signal_index] - 1
                adjustment = factor[t] / factor[signal_index] - 1
            executable = (
                np.isfinite(raw_open[t])
                & (raw_open[t] > 0)
                & np.isfinite(gap)
                & (abs(gap) < 0.045)
                & np.isfinite(adjustment)
                & (abs(adjustment) < 0.002)
                & (liquidity[signal_index] > 0)
            )
            difference = desired - holdings
            sell_order = sorted(
                np.flatnonzero(difference < -1e-10),
                key=lambda asset: (
                    values[asset] if np.isfinite(values[asset]) else -np.inf,
                    asset,
                ),
            )
            buy_order = sorted(
                np.flatnonzero(difference > 1e-10),
                key=lambda asset: (-values[asset], asset),
            )
            for side, order in (("sell", sell_order), ("buy", buy_order)):
                budget = nav * config.side_cap
                for asset in order:
                    if not executable[asset] or (side == "sell" and t - age[asset] < 1):
                        blocked += 1
                        continue
                    quantity = min(
                        abs(difference[asset]),
                        budget / open_price[t, asset],
                        liquidity[signal_index, asset]
                        * config.max_participation
                        / open_price[t, asset],
                    )
                    if side == "sell":
                        quantity = min(quantity, holdings[asset])
                    else:
                        quantity = min(
                            quantity,
                            max(0, cash - 5)
                            / (
                                open_price[t, asset]
                                * (1 + config.commission + config.slippage)
                            ),
                        )
                    if side == "buy" or quantity < holdings[asset] - 1e-8:
                        quantity = (
                            np.floor(quantity * factor[t, asset] / config.lot)
                            * config.lot
                            / factor[t, asset]
                        )
                    if quantity < 1e-10:
                        continue
                    amount = quantity * open_price[t, asset]
                    commission = max(5, amount * config.commission)
                    slippage = amount * config.slippage
                    tax = amount * config.stamp_tax if side == "sell" else 0.0
                    fees = commission + slippage + tax
                    if side == "sell" and cash + amount < fees:
                        blocked += 1
                        continue
                    before = holdings[asset]
                    if side == "sell":
                        holdings[asset] -= quantity
                        cash += amount - fees
                    else:
                        holdings[asset] += quantity
                        cash -= amount + fees
                        age[asset] = t
                    if cash < -1e-5 or holdings[asset] < -1e-8:
                        raise AssertionError("cash or holdings became negative")
                    budget -= amount
                    cost += fees
                    traded_notional += amount
                    trades.append(
                        dict(
                            date=dates[t],
                            signal_date=dates[signal_index],
                            symbol=symbols[asset],
                            side=side,
                            units=quantity,
                            raw_shares=quantity * factor[t, asset],
                            open_price=open_price[t, asset],
                            raw_price=raw_open[t, asset],
                            notional=amount,
                            commission=commission,
                            slippage=slippage,
                            stamp_tax=tax,
                            cash_after=cash,
                            position_before=before,
                            position_after=holdings[asset],
                            opening_equity_after=cash + holdings @ opening_mark,
                            prior_nav=nav,
                            participation=amount / liquidity[signal_index, asset],
                        )
                    )
            decisions.append(
                dict(
                    date=dates[t],
                    signal_date=dates[signal_index],
                    selected=len(selected),
                    target_value=target_value,
                    side_budget=nav * config.side_cap,
                )
            )
        new_nav = cash + holdings @ mark[t]
        if not np.isfinite(new_nav) or new_nav <= 0:
            raise AssertionError("account NAV is invalid")
        ledger_rows.append(
            dict(
                date=dates[t],
                nav=new_nav,
                return_=new_nav / nav - 1,
                cash=cash,
                position_value=new_nav - cash,
                holdings_count=int((holdings > 1e-10).sum()),
                cost=cost,
                traded_notional=traded_notional,
                turnover=traded_notional / nav,
                blocked_orders=blocked,
                gross_exposure=(new_nav - cash) / new_nav,
                max_weight=float(np.max(holdings * mark[t]) / new_nav),
                position_hash=hashlib.sha256(holdings.tobytes()).hexdigest(),
            )
        )
        nav = new_nav
    ledger = (
        pd.DataFrame(ledger_rows)
        .set_index("date")
        .rename(columns={"return_": "return"})
    )
    return ledger, pd.DataFrame(trades), pd.DataFrame(decisions)


def backtest_metrics(ledger: pd.DataFrame) -> dict[str, float]:
    returns = ledger["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("at least two sessions are required")
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    risk_free = np.expm1(np.log1p(0.02) / 252)
    return {
        "sessions": int(len(returns)),
        "net_cagr": float(wealth[-1] ** (252 / len(returns)) - 1),
        "total_return": float(wealth[-1] - 1),
        "sharpe": float(
            (returns.mean() - risk_free) / returns.std(ddof=1) * np.sqrt(252)
        ),
        "max_drawdown": float(np.max(1 - wealth / peak)),
        "cost_total": float(ledger["cost"].sum()),
        "annual_turnover_buys_plus_sells": float(ledger["turnover"].mean() * 252),
        "mean_exposure": float(ledger["gross_exposure"].mean()),
        "max_weight": float(ledger["max_weight"].max()),
        "blocked_orders": int(ledger["blocked_orders"].sum()),
    }
