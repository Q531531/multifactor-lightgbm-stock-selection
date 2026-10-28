"""市场门控截面网络的固定权重推理与训练期 median/MAD 标准化。"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .audit import sha256, write_json
from .data import load_panel, write_table


def fit_normalizer(panel, *, fit_start, fit_end):
    dates = pd.DatetimeIndex(panel["dates"])
    mask = (dates >= fit_start) & (dates < fit_end)
    if not mask.any():
        raise ValueError("标准化训练区间为空")
    raw = panel["raw_features"]
    market = panel["raw_market"]
    fit = raw[mask][panel["member"][mask]].astype(float)
    if not len(fit):
        raise ValueError("标准化训练期没有成员样本")
    median = np.nanmedian(fit, axis=0)
    mad = np.nanmedian(abs(fit - median), axis=0) * 1.4826
    median_m = np.nanmedian(market[mask].astype(float), axis=0)
    mad_m = np.nanmedian(abs(market[mask].astype(float) - median_m), axis=0) * 1.4826
    if not np.isfinite(np.r_[median, median_m, mad, mad_m]).all():
        raise ValueError("训练期存在完全不可观测的特征")
    return dict(
        median=median,
        mad=np.where(mad < 1e-12, 1, mad),
        median_m=median_m,
        mad_m=np.where(mad_m < 1e-12, 1, mad_m),
        fit_start=np.array(str(dates[mask][0].date())),
        fit_end=np.array(str(dates[mask][-1].date())),
        boundary=np.array(fit_end),
    )


def normalize_panel(panel, scaler):
    features = np.nan_to_num(
        np.clip((panel["raw_features"] - scaler["median"]) / scaler["mad"], -3, 3),
        nan=0,
    ).astype(np.float32)
    market = np.nan_to_num(
        np.clip((panel["raw_market"] - scaler["median_m"]) / scaler["mad_m"], -3, 3),
        nan=0,
    ).astype(np.float32)
    return features, market


def predict_base_scores(
    panel_path,
    weights,
    output,
    *,
    start="2019-01-01",
    end="2026-10-20",
    fit_start="2018-01-01",
    fit_end="2023-01-01",
    repeats=4,
    threads=2,
):
    """2019—2022 使用信号年前统计，2023 以后冻结 fit_end 前统计。"""
    import torch
    from .market_transformer import load_network

    if repeats < 1:
        raise ValueError("重复次数应为正")
    panel = load_panel(panel_path)
    if panel["raw_features"].shape[2] != 158 or panel["raw_market"].shape[1] != 63:
        raise ValueError("网络输入必须为 158 股票特征和 63 市场特征")
    dates = pd.DatetimeIndex(panel["dates"])
    chosen = np.flatnonzero((dates >= start) & (dates <= end))
    if not len(chosen) or chosen.min() < 7:
        raise ValueError("推理区间为空或缺少 8 日历史")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    model = load_network(weights)
    cached = {}
    rows = []
    checks = []
    for i in chosen:
        boundary = min(pd.Timestamp(fit_end), pd.Timestamp(f"{dates[i].year}-01-01"))
        if boundary <= pd.Timestamp(fit_start) or boundary > dates[i]:
            raise ValueError("推理开始前必须存在独立标准化训练区间")
        key = str(boundary.date())
        if key not in cached:
            scaler = fit_normalizer(panel, fit_start=fit_start, fit_end=key)
            cached[key] = normalize_panel(panel, scaler)
        features, market = cached[key]
        members = np.flatnonzero(
            panel["member"][i]
            & np.isfinite(panel["close"][i])
            & (panel["close"][i] > 0)
        )
        if not len(members):
            continue
        stock = features[i - 7 : i + 1, members, :].transpose(1, 0, 2)
        state = np.repeat(market[i - 7 : i + 1][None], len(members), axis=0)
        x = torch.from_numpy(np.concatenate([stock, state], axis=2).astype(np.float32))
        hashes = []
        with torch.no_grad():
            for _ in range(repeats):
                prediction = model(x).numpy()
                hashes.append(hashlib.sha256(prediction.tobytes()).hexdigest())
        if len(set(hashes)) != 1:
            raise AssertionError("网络重复推理不一致")
        rows.extend(
            dict(
                date=dates[i],
                asset=panel["symbols"][j],
                multifactor_score=float(prediction[k]),
            )
            for k, j in enumerate(members)
        )
        checks.append(
            dict(
                date=str(dates[i].date()),
                members=len(members),
                normalization_boundary=key,
                prediction_sha256=hashes[0],
            )
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("网络没有生成有效分数")
    write_table(frame, output)
    write_json(
        Path(output).with_suffix(".receipt.json"),
        dict(
            passed=True,
            rows=len(frame),
            repeats=repeats,
            all_repeats_exact=True,
            input_sha256=sha256(panel_path),
            weights_sha256=sha256(weights),
            output_sha256=sha256(output),
            checks=checks,
            upstream_pretraining_cutoff_verified=False,
        ),
    )
    return frame
