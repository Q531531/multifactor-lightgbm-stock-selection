"""多因子评分与 LightGBM 收益头的 95%/5% 横截面融合策略。"""

from .backtest import BacktestConfig, backtest, backtest_metrics, scores_from_frame
from .features import FEATURE_NAMES, MODEL_FEATURES, build_feature_frame, eligible_mask
from .fusion import FusionConfig, cross_sectional_rank, fuse_scores
from .splits import SplitConfig, split_masks

__all__ = [
    "BacktestConfig",
    "FEATURE_NAMES",
    "FusionConfig",
    "MODEL_FEATURES",
    "SplitConfig",
    "backtest",
    "backtest_metrics",
    "build_feature_frame",
    "cross_sectional_rank",
    "eligible_mask",
    "fuse_scores",
    "scores_from_frame",
    "split_masks",
]

__version__ = "1.1.0"
