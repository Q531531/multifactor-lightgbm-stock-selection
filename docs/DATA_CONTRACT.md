# 数据接口约定

历史数据下载和版本说明见 [数据来源与获取方式](DATA_SOURCES.md)。
以下约定适用于完成格式转换后的输入，可用 `import-qlib` 命令直接转换 Qlib 数据。

## 分数融合输入

命令行工具接受 CSV 或 Parquet 文件，每个信号日、每只股票对应一行：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `date` | date/datetime | 信号日期 |
| `asset` | string | 唯一证券标识 |
| `multifactor_score` | float | 原始多因子分数 |
| `model_score` | float | 原始五日 LightGBM 分数 |

`(date, asset)` 必须唯一，分数必须是有限数值。两个输入会先在各自交易日内独立排名，
再进行 95%/5% 融合。

## 原始行情输入

`strategy-pipeline prepare` 接受 CSV 或 Parquet 长表：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `date` | date/datetime | 交易日期 |
| `asset` | string | 证券标识 |
| `open/high/low/close` | float | 复权 OHLC |
| `volume` | float | 成交量 |
| `amount` | float | 成交额，单位为千元 |
| `factor` | float | 复权因子 |
| `member` | bool | 当日可知的股票池成员状态 |

行情必须在 `(date, asset)` 上唯一。缺失的日期与股票组合会在面板中保留为空，
不会向前填充成交价格或股票池状态。

## 模型训练输入

`original5_strategy.model.fit_ranker` 接受两个已完成时间切分的数据表：

- `train`：标签在验证期开始前已经成熟的训练样本。
- `validation`：独立的早停区间。
- `feature_cols`：调用方指定的数值特征字段。
- `forward_5d_return`：从下一交易日开盘到第五个持有期的收益，或有明确说明的等价五日目标。

代码不会自动推断时间切分。调用训练函数前，使用者必须自行保证不存在未来标签泄漏。
完整流水线默认使用 `configs/original_protocol.toml` 中记录的时间边界。

## 回测行情面板

`backtest` 接受由 NumPy 数组组成的映射：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `dates` | `[T]` | 按顺序排列的交易日期 |
| `symbols` | `[N]` | 证券标识 |
| `open` | `[T, N]` | 用于成交的复权开盘价 |
| `close` | `[T, N]` | 用于每日估值的复权收盘价 |
| `amount` | `[T, N]` | 日成交额，单位为千元 |
| `factor` | `[T, N]` | 复权因子 |
| `member` | `[T, N]` | 当日可知的股票池成员状态 |

分数映射以整数信号日索引为键，每日对应一个 `[N]` 向量。收盘时点 `t` 生成的分数，
最早只能在 `t+1` 的开盘成交。
