"""账户收益、风险、费用与横截面 IC 指标。"""

import numpy as np
import pandas as pd


def _drawdown(wealth: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return 1.0 - wealth / peak


def _rolling_loss(returns: np.ndarray, window: int) -> float:
    if len(returns) < window:
        return float("nan")
    compounded = (
        pd.Series(returns)
        .add(1.0)
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True)
        .sub(1.0)
    )
    return float(max(0.0, -compounded.min()))


def account_metrics(
    ledger: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    benchmark_returns: np.ndarray | None = None,
    *,
    capital: float = 10_000_000,
) -> dict[str, float | int]:
    returns = ledger["return"].to_numpy(float)
    if len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError("账户收益至少需要两个有限观测")
    wealth = np.cumprod(1.0 + returns)
    drawdown = _drawdown(wealth)
    risk_free = np.expm1(np.log1p(0.02) / 252)
    excess_cash = returns - risk_free
    downside = np.minimum(excess_cash, 0.0)
    downside_deviation = np.sqrt(np.mean(downside**2)) * np.sqrt(252)
    cagr = wealth[-1] ** (252 / len(returns)) - 1
    # VaR/ES use the complete daily return distribution.
    q05 = float(np.quantile(returns, 0.05))
    var95 = max(0.0, -q05)
    es95 = max(0.0, -float(returns[returns <= q05].mean()))
    underwater = drawdown > 1e-12
    episodes = int(np.sum(underwater & ~np.r_[False, underwater[:-1]]))
    longest = 0
    current = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    result: dict[str, float | int] = {
        "sessions": len(returns),
        "total_return": float(wealth[-1] - 1),
        "net_cagr": float(cagr),
        "sharpe": float(excess_cash.mean() / returns.std(ddof=1) * np.sqrt(252))
        if returns.std(ddof=1) > 0
        else float("nan"),
        "sortino": float(excess_cash.mean() * 252 / downside_deviation)
        if downside_deviation > 0
        else float("nan"),
        "calmar": float(cagr / drawdown.max()) if drawdown.max() > 0 else float("nan"),
        "omega0": float(
            returns[returns > 0].sum() / max(-returns[returns < 0].sum(), 1e-15)
        ),
        "positive_day_fraction": float(np.mean(returns > 0)),
        "negative_day_fraction": float(np.mean(returns < 0)),
        "annual_volatility": float(returns.std(ddof=1) * np.sqrt(252)),
        "downside_deviation": float(downside_deviation),
        "max_drawdown": float(drawdown.max()),
        "ulcer_index": float(np.sqrt(np.mean(drawdown**2))),
        "mean_drawdown": float(drawdown.mean()),
        "underwater_fraction": float(underwater.mean()),
        "worst_day_loss": float(max(0.0, -returns.min())),
        "var95_daily_loss": var95,
        "es95_daily_loss": es95,
        "loss_days_below_2pct": int(np.sum(returns < -0.02)),
        "loss_days_below_3pct": int(np.sum(returns < -0.03)),
        "worst_5day_loss": _rolling_loss(returns, 5),
        "worst_20day_loss": _rolling_loss(returns, 20),
        "longest_underwater_sessions": longest,
        "drawdown_episodes": episodes,
        "cost_total": float(ledger["cost"].sum()),
        "annual_turnover_buys_plus_sells": float(ledger["turnover"].mean() * 252),
        "max_single_weight": float(ledger["max_weight"].max()),
        "blocked_orders": int(ledger["blocked_orders"].sum()),
        "mean_exposure": float(ledger["gross_exposure"].mean()),
    }
    previous_nav = np.r_[capital, ledger["nav"].to_numpy(float)[:-1]]
    result["annual_normalized_fee"] = float(
        np.mean(ledger["cost"] / previous_nav) * 252
    )
    traded = float(ledger["traded_notional"].sum())
    result["fee_rate_per_traded_notional"] = (
        float(ledger["cost"].sum() / traded) if traded > 0 else 0.0
    )
    if trades is not None and not trades.empty:
        for column in ("commission", "slippage", "stamp_tax"):
            if column in trades:
                result[column + "_total"] = float(trades[column].sum())
        result["fills"] = len(trades)
    if benchmark_returns is not None:
        benchmark_returns = np.asarray(benchmark_returns, dtype=float)
        if benchmark_returns.shape != returns.shape:
            raise ValueError("基准收益长度与账户收益不一致")
        excess = returns - benchmark_returns
        result["AER252"] = float(excess.mean() * 252)
        result["IR252"] = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252))
    return result


def daily_cross_sectional_ic(
    frame: pd.DataFrame,
    *,
    score_col: str = "fused_score",
    target_col: str = "forward_5d_return",
) -> tuple[pd.DataFrame, dict[str, float]]:
    required = ("date", score_col, target_col)
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"IC 输入缺少字段: {missing}")
    rows = []
    for date, group in frame.groupby("date", sort=True):
        group = group[np.isfinite(group[score_col]) & np.isfinite(group[target_col])]
        if len(group) < 3:
            continue
        rows.append(
            {
                "date": pd.Timestamp(date),
                "count": len(group),
                "IC": float(group[score_col].corr(group[target_col])),
                "RankIC": float(
                    group[score_col].corr(group[target_col], method="spearman")
                ),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("没有足够样本计算 IC")
    rank_ic_std = float(daily["RankIC"].std(ddof=1))
    summary = {
        "days": len(daily),
        "IC": float(daily["IC"].mean()),
        "RankIC": float(daily["RankIC"].mean()),
        "RankICIR252": (
            float(daily["RankIC"].mean() / rank_ic_std * np.sqrt(252))
            if rank_ic_std > 0
            else float("nan")
        ),
    }
    return daily, summary
