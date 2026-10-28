"""按交易日组批的截面回归训练：标签去尾、截面标准化、忽略缺失标签的均方误差。"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .market_transformer import MarketGatedTransformer


class DailySampler:
    """每个交易日的全部样本组成一批；数据集需按日期连续排列。"""

    def __init__(self, dataset, *, shuffle=False, seed=None):
        index = dataset.get_index()
        counts = pd.Series(np.ones(len(index)), index=index).groupby(
            level="datetime", sort=False
        )
        self.sizes = counts.size().to_numpy()
        self.starts = np.r_[0, np.cumsum(self.sizes)[:-1]]
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = np.arange(len(self.sizes))
        if self.shuffle:
            self.rng.shuffle(order)
        for day in order:
            yield np.arange(self.starts[day], self.starts[day] + self.sizes[day])

    def __len__(self):
        return len(self.sizes)


def trim_label_tails(label, fraction=0.025):
    """按排序位置去掉两端各 ``int(fraction * N)`` 个标签；小截面全部保留。"""
    cut = int(fraction * len(label))
    keep = torch.zeros(len(label), dtype=torch.bool, device=label.device)
    keep[torch.argsort(label)[cut : len(label) - cut]] = True
    return keep


def standardize(values):
    return (values - values.mean()) / values.std()


def masked_mse(prediction, label):
    valid = ~torch.isnan(label)
    return torch.mean((prediction[valid] - label[valid]) ** 2)


def daily_ic(prediction, label):
    frame = pd.DataFrame({"prediction": prediction, "label": label})
    return (
        frame["prediction"].corr(frame["label"]),
        frame["prediction"].corr(frame["label"], method="spearman"),
    )


class CrossSectionTrainer:
    """训练截面打分网络并保存、重载权重。

    数据集 ``dataset[indices]`` 返回 ``[N, T, F + 1]``，最后一列在最后一个回看日
    存放标签；``dataset.get_index()`` 返回含 ``datetime`` 层级的行索引。
    """

    def __init__(
        self,
        build_network=MarketGatedTransformer,
        *,
        epochs,
        lr,
        gpu=None,
        seed=None,
        stop_loss=None,
        save_dir="models",
        save_prefix="network",
        clip_value=3.0,
    ):
        self.epochs = epochs
        self.seed = seed
        self.stop_loss = stop_loss
        self.save_dir = Path(save_dir)
        self.save_prefix = save_prefix
        self.clip_value = clip_value
        use_gpu = gpu is not None and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{gpu}" if use_gpu else "cpu")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        self.network = build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)
        self.trained_epochs = -1

    @property
    def weights_path(self):
        return self.save_dir / f"{self.save_prefix}_{self.seed}.pt"

    def _split(self, batch):
        batch = torch.as_tensor(batch)
        return batch[:, :, :-1].to(self.device), batch[:, -1, -1].to(self.device)

    def train_epoch(self, dataset, sampler):
        self.network.train()
        losses = []
        for rows in sampler:
            features, label = self._split(dataset[rows])
            keep = trim_label_tails(label)
            prediction = self.network(features[keep].float())
            loss = masked_mse(prediction, standardize(label[keep]))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), self.clip_value)
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def fit(self, train, valid=None):
        """训练到最大轮数或训练损失不高于 ``stop_loss``，保存最后一轮权重。"""
        sampler = DailySampler(train, shuffle=True, seed=self.seed)
        for epoch in range(self.epochs):
            loss = self.train_epoch(train, sampler)
            self.trained_epochs = epoch
            message = f"epoch {epoch}, train_loss {loss:.6f}"
            if valid is not None:
                _, metrics = self.predict(valid)
                message += ", " + ", ".join(
                    f"{name} {value:.4f}" for name, value in metrics.items()
                )
            print(message, flush=True)
            if self.stop_loss is not None and loss <= self.stop_loss:
                break
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(copy.deepcopy(self.network.state_dict()), self.weights_path)
        return self.weights_path

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state)
        self.trained_epochs = 0

    def predict(self, dataset):
        """逐日推理，返回预测序列及日度 IC/RankIC 均值和信息比率。"""
        if self.trained_epochs < 0:
            raise ValueError("网络尚未训练或加载权重")
        self.network.eval()
        predictions, ic, rank_ic = [], [], []
        for rows in DailySampler(dataset):
            features, label = self._split(dataset[rows])
            with torch.no_grad():
                prediction = self.network(features.float()).cpu().numpy().ravel()
            predictions.append(prediction)
            pearson, spearman = daily_ic(prediction, label.cpu().numpy())
            ic.append(pearson)
            rank_ic.append(spearman)
        series = pd.Series(np.concatenate(predictions), index=dataset.get_index())
        metrics = {
            "IC": np.mean(ic),
            "ICIR": np.mean(ic) / np.std(ic),
            "RankIC": np.mean(rank_ic),
            "RankICIR": np.mean(rank_ic) / np.std(rank_ic),
        }
        return series, metrics
