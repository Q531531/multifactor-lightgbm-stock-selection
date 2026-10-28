"""加载 TOML 策略、模型和账户参数。"""

from pathlib import Path
import tomllib

from .backtest import BacktestConfig
from .model import RankerConfig, DEFAULT_SEEDS
from .splits import SplitConfig


def load_configuration(path=None):
    data = {}
    if path is not None:
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
    split = SplitConfig(**data.get("split", {}))
    params = dict(data.get("model", {}))
    seeds = tuple(params.pop("seeds", DEFAULT_SEEDS))
    objective = params.pop("objective", "lambdarank")
    if objective != "lambdarank" or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("模型应为 lambdarank 且种子非空、不重复")
    model = RankerConfig(**params)
    account = dict(data.get("account", {}))
    repeats = int(account.pop("formal_repeats", 4))
    account = BacktestConfig(**account)
    strategy = data.get("strategy", {})
    weight = float(strategy.get("lightgbm_weight", 0.05))
    base_weight = float(strategy.get("multifactor_weight", 1 - weight))
    if not 0 <= weight <= 1 or abs(weight + base_weight - 1) > 1e-12 or repeats < 1:
        raise ValueError("融合权重应合计 1，重复数必须为正")
    lookback = int(data.get("features", {}).get("eligibility_lookback", 32))
    if lookback < 32:
        raise ValueError("17 项技术特征至少需要 32 日前缀")
    return dict(
        split=split,
        model=model,
        account=account,
        seeds=seeds,
        repeats=repeats,
        weight=weight,
        lookback=lookback,
    )
