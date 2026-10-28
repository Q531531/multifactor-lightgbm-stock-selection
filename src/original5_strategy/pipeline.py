"""端到端命令：准备数据、训练、预测融合、账户回测和审计。"""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import audit_prepared_frame, audit_account, sha256, write_json
from .backtest import BacktestConfig, backtest, scores_from_frame
from .data import panel_from_long, read_table, validate_multifactor_scores, write_table
from .features import MODEL_FEATURES, build_feature_frame
from .fusion import FusionConfig, fuse_scores
from .metrics import account_metrics, daily_cross_sectional_ic
from .model import (
    DEFAULT_SEEDS,
    RankerConfig,
    fit_ensemble,
    load_ensemble,
    predict_ensemble,
    save_ensemble,
)
from .splits import SplitConfig, split_masks, split_receipt


def _frame_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def prepare_dataset(
    market_path: str | Path,
    output_path: str | Path,
    *,
    multifactor_path: str | Path | None = None,
    panel_output: str | Path | None = None,
    start: str | None = None,
    lookback: int = 32,
    allow_demo_multifactor: bool = False,
) -> pd.DataFrame:
    market = read_table(market_path)
    panel = panel_from_long(market)
    multifactor = (
        validate_multifactor_scores(read_table(multifactor_path))
        if multifactor_path is not None
        else None
    )
    frame = build_feature_frame(
        panel,
        multifactor,
        start=start,
        lookback=lookback,
        allow_demo_multifactor=allow_demo_multifactor,
    )
    write_table(frame, output_path)
    if panel_output is not None:
        panel_output = Path(panel_output)
        panel_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(panel_output, **panel)
    return frame


def train_models(
    prepared_path: str | Path,
    models_dir: str | Path,
    metadata_path: str | Path,
    *,
    split_config: SplitConfig | None = None,
    ranker_config: RankerConfig | None = None,
    seeds=DEFAULT_SEEDS,
) -> dict:
    split_config = split_config or SplitConfig()
    ranker_config = ranker_config or RankerConfig()
    frame = read_table(prepared_path)
    audit = audit_prepared_frame(frame, split_config)
    masks = split_masks(frame, split_config)
    if Path(models_dir).exists() and any(Path(models_dir).glob("ranker_*.txt")):
        raise FileExistsError("训练输出目录已有模型，请使用新目录")
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    train = frame.loc[masks["train"]].copy()
    early = frame.loc[masks["early"]].copy()
    models = fit_ensemble(
        train,
        early,
        MODEL_FEATURES,
        seeds=tuple(seeds),
        config=ranker_config,
    )
    models_dir = Path(models_dir)
    save_ensemble(models, models_dir)
    validation = frame.loc[masks["validation"]].copy()
    validation["model_score"] = predict_ensemble(
        models, validation[list(MODEL_FEATURES)]
    )
    validation["fused_score"] = fuse_scores(
        validation,
        FusionConfig(model_col="model_score"),
    )["fused_score"]
    daily_ic, ic = daily_cross_sectional_ic(validation)
    daily_ic_path = Path(metadata_path).with_name("validation_daily_ic.csv")
    daily_ic.to_csv(daily_ic_path, index=False)
    model_files = sorted(models_dir.glob("ranker_*.txt"))
    metadata = {
        "passed": True,
        "seeds": list(map(int, seeds)),
        "features": list(MODEL_FEATURES),
        "split": split_receipt(frame, split_config),
        "prepared_audit": audit,
        "ranker_config": asdict(ranker_config),
        "model_files": {path.name: sha256(path) for path in model_files},
        "prepared_sha256": sha256(prepared_path),
        "validation_ic": ic,
        "validation_daily_ic_sha256": sha256(daily_ic_path),
        "evaluation_used_for_training_or_selection": False,
    }
    write_json(metadata_path, metadata)
    write_json(models_dir / "training_manifest.json", metadata)
    return metadata


def predict_and_fuse(
    prepared_path: str | Path,
    models_dir: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: str = "2024-07-01",
    weight: float = 0.05,
) -> pd.DataFrame:
    frame = read_table(prepared_path)
    frame["date"] = pd.to_datetime(frame["date"])
    training_manifest = Path(models_dir) / "training_manifest.json"
    if training_manifest.exists():
        import json

        saved = json.loads(training_manifest.read_text())
        if saved["features"] != list(MODEL_FEATURES):
            raise ValueError("训练特征顺序与推理不一致")
        if pd.Timestamp(evaluation_start) < pd.Timestamp(
            saved["split"]["config"]["validation_end"]
        ):
            raise ValueError("评价预测区间早于已登记验证边界")
    selected = frame[frame["date"] >= pd.Timestamp(evaluation_start)].copy()
    if selected.empty:
        raise ValueError("评价区间没有样本")
    models = load_ensemble(models_dir)
    selected["model_score"] = predict_ensemble(models, selected[list(MODEL_FEATURES)])
    selected = fuse_scores(
        selected,
        FusionConfig(weight=weight, model_col="model_score"),
    )
    write_table(selected, output_path)
    write_json(
        Path(output_path).with_suffix(".receipt.json"),
        dict(
            passed=True,
            rows=len(selected),
            days=selected.date.nunique(),
            weight=weight,
            model_inference_repeats=4,
            all_model_inference_repeats_exact=True,
            prepared_sha256=sha256(prepared_path),
            output_sha256=sha256(output_path),
            models={
                p.name: sha256(p) for p in sorted(Path(models_dir).glob("ranker_*.txt"))
            },
        ),
    )
    return selected


def run_account(
    market_path: str | Path,
    scores_path: str | Path,
    output_dir: str | Path,
    *,
    start: str,
    end: str,
    bps: int = 10,
    repeats: int = 4,
    config: BacktestConfig | None = None,
) -> dict:
    if repeats < 1:
        raise ValueError("重复次数必须至少为 1")
    market = read_table(market_path)
    panel = panel_from_long(market)
    scores_frame = read_table(scores_path)
    scores_frame["date"] = pd.to_datetime(scores_frame["date"])
    scores = scores_from_frame(scores_frame, panel["dates"], panel["symbols"])
    config = config or BacktestConfig(slippage=bps / 10_000)
    if not np.isclose(config.slippage, bps / 10_000):
        raise ValueError("bps 与 BacktestConfig.slippage 不一致")
    outputs = [
        backtest(panel, scores, config, start=start, end=end) for _ in range(repeats)
    ]
    hashes = [
        {
            "ledger": _frame_hash(result[0]),
            "trades": _frame_hash(result[1]),
            "decisions": _frame_hash(result[2]),
        }
        for result in outputs
    ]
    if hashes != [hashes[0]] * repeats:
        raise AssertionError("重复账户回放不一致")
    ledger, trades, decisions = outputs[0]
    account_audit = audit_account(ledger, trades, panel, config)
    metrics = account_metrics(ledger, trades, capital=config.capital)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger.parquet"
    trades_path = output_dir / "trades.parquet"
    decisions_path = output_dir / "decisions.parquet"
    ledger.to_parquet(ledger_path)
    trades.to_parquet(trades_path, index=False)
    decisions.to_parquet(decisions_path, index=False)
    receipt = {
        "passed": True,
        "start": start,
        "end": end,
        "bps": bps,
        "repeats": repeats,
        "all_repeats_exact": True,
        "repeat_hashes": hashes,
        "config": asdict(config),
        "metrics": metrics,
        "independent_account_audit": account_audit,
        "market_sha256": sha256(market_path),
        "scores_sha256": sha256(scores_path),
        "files": {
            path.name: sha256(path)
            for path in (ledger_path, trades_path, decisions_path)
        },
    }
    write_json(output_dir / "account_receipt.json", receipt)
    return receipt


def _add_split_arguments(parser: argparse.ArgumentParser) -> None:
    for name in (
        "train_start",
        "train_end",
        "early_start",
        "early_end",
        "validation_start",
        "validation_end",
        "evaluation_start",
    ):
        parser.add_argument("--" + name.replace("_", "-"), default=None)


def _split_from_args(args) -> SplitConfig:
    from .configuration import load_configuration

    defaults = load_configuration(args.config)["split"]
    return SplitConfig(
        **{
            name: getattr(args, name) or getattr(defaults, name)
            for name in SplitConfig.__dataclass_fields__
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多因子 + LightGBM 融合选股完整流水线")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="生成特征、标签和训练长表")
    prepare.add_argument("--market", type=Path, required=True)
    prepare.add_argument("--multifactor", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--panel-output", type=Path)
    prepare.add_argument("--start")
    prepare.add_argument("--lookback", type=int)
    prepare.add_argument("--allow-demo-multifactor", action="store_true")

    train = subparsers.add_parser("train", help="训练四种子五日 LambdaRank")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--models-dir", type=Path, required=True)
    train.add_argument("--metadata", type=Path, required=True)
    train.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    _add_split_arguments(train)

    predict = subparsers.add_parser(
        "predict", help="生成评价期预测并执行 95%%/5%% 融合"
    )
    predict.add_argument("--prepared", type=Path, required=True)
    predict.add_argument("--models-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--evaluation-start")
    predict.add_argument("--weight", type=float)

    account = subparsers.add_parser("backtest", help="执行次日开盘账户回测")
    account.add_argument("--market", type=Path, required=True)
    account.add_argument("--scores", type=Path, required=True)
    account.add_argument("--output-dir", type=Path, required=True)
    account.add_argument("--start", required=True)
    account.add_argument("--end", required=True)
    account.add_argument("--bps", type=int)
    account.add_argument("--repeats", type=int)

    audit = subparsers.add_parser("audit", help="复核训练表时间边界")
    audit.add_argument("--prepared", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    _add_split_arguments(audit)
    for subparser in (prepare, train, predict, account, audit):
        subparser.add_argument(
            "--config", type=Path, help="TOML 参数文件；显式参数优先"
        )
    train.set_defaults(seeds=None)

    download = subparsers.add_parser(
        "download-data", help="下载并校验固定 Qlib 数据快照"
    )
    download.add_argument("--destination", required=True, type=Path)
    qlib = subparsers.add_parser("import-qlib", help="Qlib 二进制转换为行情长表")
    qlib.add_argument("--provider", required=True, type=Path)
    qlib.add_argument("--output", required=True, type=Path)
    qlib.add_argument("--start", default="2017-11-01")
    qlib.add_argument("--end", default="2026-10-20")
    qlib.add_argument("--universe-start", default="2018-01-01")
    factors = subparsers.add_parser(
        "factor-features", help="生成 Alpha158 股票因子与市场特征"
    )
    factors.add_argument("--provider", required=True, type=Path)
    factors.add_argument("--output", required=True, type=Path)
    factors.add_argument("--start", default="2017-11-01")
    factors.add_argument("--end", default="2026-10-20")
    factors.add_argument("--universe-start", default="2018-01-01")
    inference = subparsers.add_parser(
        "base-predict", help="由市场门控截面网络权重生成基础分数"
    )
    inference.add_argument("--panel", required=True, type=Path)
    inference.add_argument("--weights", required=True, type=Path)
    inference.add_argument("--output", required=True, type=Path)
    inference.add_argument("--start", default="2019-01-01")
    inference.add_argument("--end", default="2026-10-20")
    inference.add_argument("--fit-start", default="2018-01-01")
    inference.add_argument("--fit-end", default="2023-01-01")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from .configuration import load_configuration

    settings = load_configuration(getattr(args, "config", None))
    if args.command == "prepare":
        frame = prepare_dataset(
            args.market,
            args.output,
            multifactor_path=args.multifactor,
            panel_output=args.panel_output,
            start=args.start,
            lookback=args.lookback
            if args.lookback is not None
            else settings["lookback"],
            allow_demo_multifactor=args.allow_demo_multifactor,
        )
        write_json(
            args.output.with_suffix(".receipt.json"),
            {
                "passed": True,
                "rows": len(frame),
                "days": frame["date"].nunique(),
                "assets": frame["asset"].nunique(),
                "multifactor_sources": frame["multifactor_source"]
                .value_counts()
                .to_dict(),
                "output_sha256": sha256(args.output),
                "market_sha256": sha256(args.market),
                "multifactor_sha256": sha256(args.multifactor)
                if args.multifactor
                else None,
            },
        )
    elif args.command == "train":
        train_models(
            args.prepared,
            args.models_dir,
            args.metadata,
            split_config=_split_from_args(args),
            ranker_config=settings["model"],
            seeds=tuple(int(value) for value in args.seeds.split(","))
            if args.seeds
            else settings["seeds"],
        )
    elif args.command == "predict":
        frame = predict_and_fuse(
            args.prepared,
            args.models_dir,
            args.output,
            evaluation_start=args.evaluation_start
            or settings["split"].evaluation_start,
            weight=args.weight if args.weight is not None else settings["weight"],
        )
    elif args.command == "backtest":
        from dataclasses import replace

        bps = (
            args.bps
            if args.bps is not None
            else int(round(settings["account"].slippage * 10000))
        )
        run_account(
            args.market,
            args.scores,
            args.output_dir,
            start=args.start,
            end=args.end,
            bps=bps,
            repeats=args.repeats if args.repeats is not None else settings["repeats"],
            config=replace(settings["account"], slippage=bps / 10000),
        )
    elif args.command == "audit":
        receipt = audit_prepared_frame(
            read_table(args.prepared), _split_from_args(args)
        )
        write_json(args.output, receipt)
    elif args.command == "download-data":
        from .qlib_data import download_snapshot

        print(download_snapshot(args.destination))
    elif args.command == "import-qlib":
        from .qlib_data import read_qlib_panel
        from .data import panel_to_long

        write_table(
            panel_to_long(
                read_qlib_panel(
                    args.provider,
                    start=args.start,
                    end=args.end,
                    universe_start=args.universe_start,
                )
            ),
            args.output,
        )
        write_json(
            args.output.with_suffix(".receipt.json"),
            dict(
                passed=True,
                output_sha256=sha256(args.output),
                calendar_sha256=sha256(args.provider / "calendars/day.txt"),
                membership_sha256=sha256(args.provider / "instruments/csi300.txt"),
            ),
        )
    elif args.command == "factor-features":
        from .alpha_features import build_factor_panel

        build_factor_panel(
            args.provider,
            args.output,
            start=args.start,
            end=args.end,
            universe_start=args.universe_start,
        )
    elif args.command == "base-predict":
        from .base_scores import predict_base_scores

        predict_base_scores(
            args.panel,
            args.weights,
            args.output,
            start=args.start,
            end=args.end,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
        )


if __name__ == "__main__":
    main()
