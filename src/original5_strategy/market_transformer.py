"""市场状态门控的双轴注意力网络：单股时间注意力 + 同日跨股票截面注意力。

输入形状为 ``[股票数 N, 回看日 T, 股票因子 + 市场特征]``，输出每只股票一个分数。

1. 门控：最后一日市场特征经线性层和带温度 softmax 生成股票因子权重，
   乘以因子数使权重均值为 1，再缩放全部回看日的股票因子。
2. 嵌入：线性投影到隐藏维度并叠加正弦位置编码。
3. 时间注意力：每只股票在自身回看窗口内做多头自注意力，前置 LayerNorm，
   残差后接 LayerNorm 与前馈网络。
4. 截面注意力：同一回看日内跨全部股票做多头自注意力，结构同上，
   注意力分数除以 sqrt(隐藏维度 / 头数)。
5. 时间聚合：以最后一日表示为查询，对回看日加权求和。
6. 输出：线性层得到股票分数。
"""

import math

import torch
from torch import nn


class FactorGate(nn.Module):
    """由市场特征生成均值为 1 的股票因子权重。"""

    def __init__(self, market_dim, stock_dim, temperature):
        super().__init__()
        self.proj = nn.Linear(market_dim, stock_dim)
        self.stock_dim = stock_dim
        self.temperature = temperature

    def forward(self, market):
        weights = torch.softmax(self.proj(market) / self.temperature, dim=-1)
        return self.stock_dim * weights


class SinusoidalPosition(nn.Module):
    """偶数维正弦、奇数维余弦的位置编码。"""

    def __init__(self, width, max_len=100):
        super().__init__()
        frequency = torch.exp(
            torch.arange(0, width, 2).float() * (-math.log(10000.0) / width)
        )
        angle = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) * frequency
        table = torch.zeros(max_len, width)
        table[:, 0::2] = torch.sin(angle)
        table[:, 1::2] = torch.cos(angle)
        self.register_buffer("table", table)

    def forward(self, x):
        return x + self.table[: x.shape[1], :]


class AttentionBlock(nn.Module):
    """前置归一化多头自注意力与前馈残差块。

    ``across_stocks=False`` 时沿回看日做注意力且不缩放分数；
    ``across_stocks=True`` 时交换前两维，在同一日的股票之间做注意力。
    各头逐一计算，最后一个头承接隐藏维度无法整除的剩余部分。
    """

    def __init__(self, width, heads, dropout, *, across_stocks):
        super().__init__()
        self.width = width
        self.heads = heads
        self.across_stocks = across_stocks
        self.scale = math.sqrt(width / heads) if across_stocks else None
        self.norm_in = nn.LayerNorm(width, eps=1e-5)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.attention_dropout = nn.Dropout(p=dropout)
        self.norm_out = nn.LayerNorm(width, eps=1e-5)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(width, width),
            nn.Dropout(p=dropout),
        )

    def head_slices(self):
        size = int(self.width / self.heads)
        return [
            slice(i * size, None if i == self.heads - 1 else (i + 1) * size)
            for i in range(self.heads)
        ]

    def forward(self, x):
        x = self.norm_in(x)
        q, k, v = self.query(x), self.key(x), self.value(x)
        if self.across_stocks:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        outputs = []
        for part in self.head_slices():
            scores = torch.matmul(q[:, :, part], k[:, :, part].transpose(1, 2))
            if self.scale is not None:
                scores = scores / self.scale
            weights = self.attention_dropout(torch.softmax(scores, dim=-1))
            context = torch.matmul(weights, v[:, :, part])
            outputs.append(context.transpose(0, 1) if self.across_stocks else context)
        x = self.norm_out(x + torch.concat(outputs, dim=-1))
        return x + self.feed_forward(x)


class LastStepPooling(nn.Module):
    """以最后一个回看日为查询，对全部回看日加权聚合。"""

    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, z):
        h = self.proj(z)
        query = h[:, -1, :].unsqueeze(-1)
        weights = torch.softmax(torch.matmul(h, query).squeeze(-1), dim=1)
        return torch.matmul(weights.unsqueeze(1), z).squeeze(1)


class MarketGatedTransformer(nn.Module):
    """默认配置：158 股票因子、63 市场特征、256 隐藏维度、4/2 头。"""

    def __init__(
        self,
        stock_dim=158,
        market_dim=63,
        width=256,
        time_heads=4,
        stock_heads=2,
        time_dropout=0.5,
        stock_dropout=0.5,
        gate_temperature=5,
    ):
        super().__init__()
        self.stock_dim = stock_dim
        self.market_dim = market_dim
        self.gate = FactorGate(market_dim, stock_dim, gate_temperature)
        self.embed = nn.Linear(stock_dim, width)
        self.position = SinusoidalPosition(width)
        self.time_attention = AttentionBlock(
            width, time_heads, time_dropout, across_stocks=False
        )
        self.stock_attention = AttentionBlock(
            width, stock_heads, stock_dropout, across_stocks=True
        )
        self.pooling = LastStepPooling(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        stock = x[:, :, : self.stock_dim]
        market = x[:, -1, self.stock_dim : self.stock_dim + self.market_dim]
        stock = stock * torch.unsqueeze(self.gate(market), dim=1)
        h = self.position(self.embed(stock))
        h = self.stock_attention(self.time_attention(h))
        return self.head(self.pooling(h)).squeeze(-1)


# 顺序层编号格式的权重键到本网络参数名的对应关系。
_LAYER_PREFIXES = {
    "feature_gate.trans.": "gate.proj.",
    "layers.0.": "embed.",
    "layers.1.pe": "position.table",
    "layers.2.": "time_attention.",
    "layers.3.": "stock_attention.",
    "layers.4.trans.": "pooling.proj.",
    "layers.5.": "head.",
}
_ATTENTION_NAMES = {
    "qtrans.": "query.",
    "ktrans.": "key.",
    "vtrans.": "value.",
    "norm1.": "norm_in.",
    "norm2.": "norm_out.",
    "ffn.": "feed_forward.",
}


def convert_state_dict(state):
    """将顺序层编号格式的权重键转换为本网络参数名；本网络格式原样返回。"""
    converted = {}
    for name, tensor in state.items():
        for old, new in _LAYER_PREFIXES.items():
            if name.startswith(old):
                name = new + name[len(old) :]
                break
        if name.startswith(("time_attention.", "stock_attention.")):
            block, rest = name.split(".", 1)
            for old, new in _ATTENTION_NAMES.items():
                if rest.startswith(old):
                    rest = new + rest[len(old) :]
                    break
            name = f"{block}.{rest}"
        converted[name] = tensor
    return converted


def load_network(weights, **kwargs):
    """在 CPU 上构建网络并严格加载权重，返回推理模式的网络。"""
    network = MarketGatedTransformer(**kwargs)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    network.load_state_dict(convert_state_dict(state))
    return network.eval()
