"""基于冻结实验资产验证特征、模型、融合分数与账户保真。"""

from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd

from original5_strategy.audit import audit_account, sha256, write_json
from original5_strategy.backtest import BacktestConfig, backtest
from original5_strategy.data import load_panel
from original5_strategy.features import (
    FEATURE_NAMES,
    MODEL_FEATURES,
    technical_feature_arrays,
)
from original5_strategy.fusion import fuse_scores
from original5_strategy.model import DEFAULT_SEEDS, fit_ranker


def verify(root, output):
    panel = load_panel(root / "inputs/panel.npz")
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = panel["symbols"]
    with np.load(root / "inputs/fit_meta.npz", allow_pickle=False) as z:
        meta = {k: z[k] for k in z.files}
    arrays = technical_feature_arrays(panel)
    actual = np.column_stack(
        [np.nan_to_num(arrays[name][meta["ti"], meta["sj"]]) for name in FEATURE_NAMES]
    ).astype(np.float32)
    np.testing.assert_array_equal(actual, meta["technical"])
    labels = np.full(len(meta["ti"]), np.nan, np.float32)
    labels[:] = (
        panel["open"][meta["ti"] + 6, meta["sj"]]
        / panel["open"][meta["ti"] + 1, meta["sj"]]
        - 1
    )
    np.testing.assert_array_equal(labels, meta["y5"])
    base = pd.read_parquet(root / "inputs/base_train_validation.parquet")
    ids = pd.DataFrame(dict(ti=meta["ti"], sj=meta["sj"]))
    ids = ids.merge(base[["ti", "sj", "score"]], on=["ti", "sj"], validate="one_to_one")
    ranks = ids.groupby("ti").score.rank(pct=True).to_numpy() - 0.5
    np.testing.assert_array_equal(ranks, meta["base_rank"])
    x = np.column_stack([actual, ranks]).astype(np.float32)
    np.testing.assert_array_equal(x, np.load(root / "inputs/fit_technical.npy"))
    frame = pd.DataFrame(x, columns=MODEL_FEATURES)
    frame["date"] = dates.to_numpy()[meta["ti"]]
    frame["asset"] = symbols[meta["sj"]]
    frame["forward_5d_return"] = labels
    ends = dates.to_numpy()[meta["ti"] + 6]
    tr = (
        (meta["split"] == 0)
        & np.isfinite(labels)
        & (ends < np.datetime64("2023-01-01"))
    )
    early = (
        (meta["split"] == 1)
        & np.isfinite(labels)
        & (frame.date.to_numpy() < np.datetime64("2023-07-01"))
        & (ends < np.datetime64("2023-07-01"))
    )
    models = []
    model_checks = []
    prefix = "technical_5d_lambdarank_medium"
    for seed in DEFAULT_SEEDS:
        model = fit_ranker(frame.loc[tr], frame.loc[early], MODEL_FEATURES, seed=seed)
        text = model.booster_.model_to_string()
        expected_path = root / f"models/{prefix}_s{seed}.txt"
        assert text == expected_path.read_text(), (
            f"model text differs for seed {seed}; check dependency versions"
        )
        expected = pd.read_parquet(
            root / f"results/{prefix}_s{seed}_validation.parquet"
        )
        np.testing.assert_array_equal(
            model.predict(x[meta["split"] == 1], num_threads=1), expected.score
        )
        models.append(model)
        model_checks.append(
            dict(
                seed=seed,
                model_sha256=hashlib.sha256(text.encode()).hexdigest(),
                validation_predictions=len(expected),
                exact=True,
            )
        )
    with np.load(root / "inputs/evaluation_meta.npz", allow_pickle=False) as z:
        em = {k: z[k] for k in z.files}
    ex = np.load(root / "inputs/evaluation_technical.npy")
    predicted = np.mean([m.predict(ex, num_threads=1) for m in models], axis=0)
    eb = pd.read_parquet(root / "inputs/base_evaluation.parquet")
    joined = pd.DataFrame(dict(ti=em["ti"], sj=em["sj"])).merge(
        eb[["ti", "sj", "score"]], on=["ti", "sj"], validate="one_to_one"
    )
    scores = pd.DataFrame(
        dict(
            date=dates.to_numpy()[joined.ti.to_numpy()],
            asset=symbols[joined.sj.to_numpy()],
            multifactor_score=joined.score,
            model_score=predicted,
        )
    )
    fused = fuse_scores(scores)
    old_head = pd.read_parquet(root / f"results/{prefix}_evaluation.parquet")
    np.testing.assert_array_equal(predicted, old_head.score)
    daily = {}
    for i, g in joined.groupby("ti"):
        v = np.full(len(symbols), np.nan)
        v[g.sj.to_numpy()] = fused.loc[g.index, "fused_score"]
        daily[int(i)] = v
    accounts = []
    for bps in (10, 30):
        config = BacktestConfig(slippage=bps / 10000)
        for repeat in range(4):
            result = backtest(
                panel, daily, config, start="2024-07-01", end="2026-10-20"
            )
            for kind, current in zip(("ledger", "trades", "decisions"), result):
                expected = pd.read_parquet(
                    root
                    / f"results/evaluation_accounts/{prefix}_w5_{bps}bp_{kind}.parquet"
                )
                pd.testing.assert_frame_equal(current, expected, check_exact=True)
        audit = audit_account(result[0], result[1], panel, config)
        accounts.append(
            dict(bps=bps, repeats=4, all_original_fields_exact=True, **audit)
        )
    receipt = dict(
        passed=True,
        scope="engineering fidelity on supplied frozen assets; not unseen performance",
        feature_rows=len(actual),
        features=17,
        labels_exact=True,
        base_ranks_exact=True,
        models=model_checks,
        fused_rows=len(fused),
        accounts=accounts,
        panel_sha256=sha256(root / "inputs/panel.npz"),
    )
    write_json(output, receipt)
    print(
        "HISTORICAL_ENGINEERING_FIDELITY_PASSED",
        len(actual),
        len(fused),
        len(models),
        flush=True,
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.reference_dir, args.output)
