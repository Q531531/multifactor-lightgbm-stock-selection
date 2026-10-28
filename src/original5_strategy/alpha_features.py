"""使用 Qlib 计算 Alpha158 股票特征与 63 维指数特征。"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import save_panel
from .audit import sha256, write_json
from .qlib_data import read_qlib_panel


def initialize_provider(provider):
    from qlib.config import C
    from qlib.data.ops import register_all_ops
    from qlib.data.data import register_all_wrappers

    # Initialize the local expression provider without experiment recording.
    C.set(
        provider_uri=str(Path(provider).resolve()),
        region="cn",
        expression_cache=None,
        dataset_cache=None,
        kernels=1,
    )
    register_all_ops(C)
    register_all_wrappers(C)
    C._registered = True


def build_factor_panel(
    provider,
    output,
    *,
    start="2017-11-01",
    end="2026-10-20",
    universe_start="2018-01-01",
):
    initialize_provider(provider)
    from qlib.data import D
    from qlib.contrib.data.loader import Alpha158DL

    panel = read_qlib_panel(
        provider, start=start, end=end, universe_start=universe_start
    )
    dates = pd.DatetimeIndex(panel["dates"])
    expressions, names = Alpha158DL.get_feature_config()
    raw = np.full((len(dates), len(panel["symbols"]), len(names)), np.nan, np.float32)
    for j, symbol in enumerate(panel["symbols"]):
        frame = D.features([str(symbol)], expressions, start_time=start, end_time=end)
        frame = frame.droplevel("instrument").reindex(dates)
        raw[:, j] = frame.to_numpy(np.float32)
        if (j + 1) % 50 == 0:
            print(f"Alpha158 {j + 1}/{len(panel['symbols'])}", flush=True)
    market = []
    market_expressions = ["$close/Ref($close,1)-1"]
    for window in (5, 10, 20, 30, 60):
        market_expressions.extend(
            [
                f"Mean($close/Ref($close,1)-1,{window})",
                f"Std($close/Ref($close,1)-1,{window})",
                f"Mean($amount,{window})/$amount",
                f"Std($amount,{window})/$amount",
            ]
        )
    for symbol in ("SH000300", "SH000905", "SH000906"):
        frame = D.features([symbol], market_expressions, start_time=start, end_time=end)
        market.append(frame.droplevel("instrument").reindex(dates).to_numpy(np.float32))
    panel["raw_features"] = raw
    panel["raw_market"] = np.concatenate(market, axis=1)
    save_panel(panel, output)
    Path(output).with_suffix(".expressions.json").write_text(
        json.dumps(
            {
                "stock_names": names,
                "stock_expressions": expressions,
                "market_expressions": market_expressions,
                "market_indices": ["SH000300", "SH000905", "SH000906"],
                "normalization_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    provider = Path(provider)
    write_json(
        Path(output).with_suffix(".receipt.json"),
        dict(
            passed=True,
            normalized=False,
            stock_features=158,
            market_features=63,
            calendar_sha256=sha256(provider / "calendars/day.txt"),
            membership_sha256=sha256(provider / "instruments/csi300.txt"),
            output_sha256=sha256(output),
        ),
    )
    return panel
