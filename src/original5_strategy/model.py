"""LightGBM five-session ranking head used before 5% score fusion.

Model weights and datasets are separate inputs. Callers must provide causally
split training and validation frames.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_SEEDS = (23, 37, 53, 71)


@dataclass(frozen=True)
class RankerConfig:
    n_estimators: int = 600
    learning_rate: float = 0.035
    num_leaves: int = 31
    max_depth: int = 7
    min_child_samples: int = 200
    reg_alpha: float = 0.05
    reg_lambda: float = 2.0
    colsample_bytree: float = 0.85
    subsample: float = 0.8
    subsample_freq: int = 1
    lambdarank_truncation_level: int = 40
    early_stopping_rounds: int = 60
    n_jobs: int = 2


def _prepare(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    date_col: str,
    asset_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = [date_col, asset_col, target_col, *feature_cols]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    if frame.empty:
        raise ValueError("training frame is empty")
    if frame.duplicated([date_col, asset_col]).any():
        raise ValueError("duplicate date/asset rows are not allowed")
    ordered = frame.sort_values([date_col, asset_col], kind="mergesort").reset_index(
        drop=True
    )
    numeric = ordered[[target_col, *feature_cols]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("target and features must be finite")
    rank_target = (
        ordered.groupby(date_col, sort=False)[target_col]
        .rank(method="average", pct=True)
        .sub(0.5)
        .to_numpy()
    )
    relevance = np.minimum(np.floor((rank_target + 0.5) * 10), 9).astype(int)
    groups = ordered.groupby(date_col, sort=False).size().to_numpy(dtype=int)
    return ordered, relevance, groups


def fit_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    seed: int,
    config: RankerConfig | None = None,
    date_col: str = "date",
    asset_col: str = "asset",
    target_col: str = "forward_5d_return",
):
    """Fit one deterministic LightGBM LambdaRank model.

    The caller is responsible for ensuring every training label matures before
    the validation period. This function never constructs a temporal split.
    """

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "install the 'train' extra to use LightGBM training"
        ) from exc

    config = config or RankerConfig()
    train_ordered, train_target, train_groups = _prepare(
        train, feature_cols, date_col, asset_col, target_col
    )
    valid_ordered, valid_target, valid_groups = _prepare(
        validation, feature_cols, date_col, asset_col, target_col
    )
    train_x = train_ordered[list(feature_cols)].to_numpy(dtype=np.float32)
    valid_x = valid_ordered[list(feature_cols)].to_numpy(dtype=np.float32)
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(range(10)),
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        colsample_bytree=config.colsample_bytree,
        subsample=config.subsample,
        subsample_freq=config.subsample_freq,
        lambdarank_truncation_level=config.lambdarank_truncation_level,
        random_state=seed,
        n_jobs=config.n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model = lgb.LGBMRanker(**params)
    model.fit(
        train_x,
        train_target,
        group=train_groups,
        eval_set=[(valid_x, valid_target)],
        eval_group=[valid_groups],
        eval_at=[30],
        callbacks=[
            lgb.early_stopping(
                config.early_stopping_rounds, verbose=False, first_metric_only=True
            )
        ],
    )
    return model


def fit_ensemble(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    config: RankerConfig | None = None,
    **columns,
) -> list:
    """Fit a LambdaRank ensemble with the configured random seeds."""

    return [
        fit_ranker(
            train,
            validation,
            feature_cols,
            seed=int(seed),
            config=config,
            **columns,
        )
        for seed in seeds
    ]


def predict_ensemble(
    models: Sequence, features: pd.DataFrame | np.ndarray, *, repeats=4
) -> np.ndarray:
    if not models:
        raise ValueError("at least one model is required")
    if repeats < 1:
        raise ValueError("重复推理次数应为正")
    x = np.asarray(features, dtype=np.float32)
    predictions = []
    for model in models:
        previous = None
        for _ in range(repeats):
            values = np.asarray(model.predict(x, num_threads=1), dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("模型生成非有限分数")
            digest = hashlib.sha256(values.tobytes()).hexdigest()
            if previous is not None and digest != previous:
                raise AssertionError("同一模型重复推理不一致")
            previous = digest
        predictions.append(values)
    if any(values.shape != predictions[0].shape for values in predictions[1:]):
        raise ValueError("model prediction shapes differ")
    return np.mean(predictions, axis=0)


def save_ensemble(models: Sequence, directory: str | Path) -> None:
    """Save trained models and their loading metadata to a new output directory."""

    directory = Path(directory)
    if not models:
        raise ValueError("至少需要一个模型")
    if directory.exists() and any(directory.glob("ranker_*.txt")):
        raise FileExistsError("模型目录已有权重，请使用新的输出目录")
    directory.mkdir(parents=True, exist_ok=True)
    for index, model in enumerate(models):
        model.booster_.save_model(str(directory / f"ranker_{index}.txt"))
    (directory / "ensemble.json").write_text(
        json.dumps(
            {
                "model_files": {
                    f"ranker_{i}.txt": hashlib.sha256(
                        (directory / f"ranker_{i}.txt").read_bytes()
                    ).hexdigest()
                    for i in range(len(models))
                }
            },
            indent=2,
        )
        + "\n"
    )


def load_ensemble(directory: str | Path) -> list:
    """加载模型，并核对 ensemble.json 中登记的文件集合与保存记录。"""

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("install the 'train' extra to load LightGBM models") from exc
    paths = sorted(Path(directory).glob("ranker_*.txt"))
    if not paths:
        raise FileNotFoundError(f"没有找到模型: {directory}/ranker_*.txt")
    manifest = Path(directory) / "ensemble.json"
    if manifest.exists():
        expected = json.loads(manifest.read_text())["model_files"]
        if set(expected) != {p.name for p in paths}:
            raise ValueError("模型目录与 ensemble.json 成员集合不一致")
        for p in paths:
            if hashlib.sha256(p.read_bytes()).hexdigest() != expected[p.name]:
                raise ValueError(f"模型文件与保存记录不一致: {p.name}")
    return [lgb.Booster(model_file=str(path)) for path in paths]
