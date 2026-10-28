"""严格按标签成熟时间切分训练、早停、验证和评价样本。"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2019-01-01"
    train_end: str = "2023-01-01"
    early_start: str = "2023-01-01"
    early_end: str = "2023-07-01"
    validation_start: str = "2023-07-01"
    validation_end: str = "2024-01-01"
    evaluation_start: str = "2024-07-01"

    def __post_init__(self) -> None:
        values = [
            pd.Timestamp(self.train_start),
            pd.Timestamp(self.train_end),
            pd.Timestamp(self.early_start),
            pd.Timestamp(self.early_end),
            pd.Timestamp(self.validation_start),
            pd.Timestamp(self.validation_end),
            pd.Timestamp(self.evaluation_start),
        ]
        if values != sorted(values):
            raise ValueError("时间边界必须按训练、早停、验证、评价顺序排列")
        if values[1] > values[2] or values[3] > values[4] or values[5] > values[6]:
            raise ValueError("相邻区间不允许重叠")


def split_masks(
    frame: pd.DataFrame, config: SplitConfig | None = None
) -> dict[str, np.ndarray]:
    config = config or SplitConfig()
    required = ("date", "label_end", "forward_5d_return")
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"切分输入缺少字段: {missing}")
    signal = pd.to_datetime(frame["date"])
    label_end = pd.to_datetime(frame["label_end"])
    known = (
        np.isfinite(frame["forward_5d_return"].to_numpy(float))
        & label_end.notna().to_numpy()
    )
    masks = {
        "train": (
            (signal >= pd.Timestamp(config.train_start))
            & (signal < pd.Timestamp(config.train_end))
            & (label_end < pd.Timestamp(config.train_end))
            & known
        ).to_numpy(),
        "early": (
            (signal >= pd.Timestamp(config.early_start))
            & (signal < pd.Timestamp(config.early_end))
            & (label_end < pd.Timestamp(config.early_end))
            & known
        ).to_numpy(),
        "validation": (
            (signal >= pd.Timestamp(config.validation_start))
            & (signal < pd.Timestamp(config.validation_end))
            & (label_end < pd.Timestamp(config.validation_end))
            & known
        ).to_numpy(),
        "evaluation": (signal >= pd.Timestamp(config.evaluation_start)).to_numpy(),
    }
    for name in ("train", "early", "validation"):
        if not masks[name].any():
            raise ValueError(f"{name} 切分没有样本")
    if (masks["train"] & masks["early"]).any() or (
        masks["early"] & masks["validation"]
    ).any():
        raise AssertionError("训练、早停和验证样本发生重叠")
    return masks


def split_receipt(frame: pd.DataFrame, config: SplitConfig | None = None) -> dict:
    config = config or SplitConfig()
    masks = split_masks(frame, config)
    receipt = {"config": asdict(config), "parts": {}}
    for name, mask in masks.items():
        part = frame.loc[mask]
        receipt["parts"][name] = {
            "rows": int(mask.sum()),
            "days": int(part["date"].nunique()),
            "first_signal": str(pd.Timestamp(part["date"].min()).date())
            if len(part)
            else None,
            "last_signal": str(pd.Timestamp(part["date"].max()).date())
            if len(part)
            else None,
            "maximum_label_end": (
                str(pd.Timestamp(part["label_end"].max()).date())
                if len(part) and part["label_end"].notna().any()
                else None
            ),
        }
    receipt["passed"] = True
    receipt["evaluation_used_for_training"] = False
    return receipt
