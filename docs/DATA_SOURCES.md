# 数据来源与获取方式

本项目需要两类真实输入：**日线行情及历史股票池**、**多因子基础分数**。
两者来源不同；下载行情文件不会自动得到基础模型预测。

## 历史实验使用的行情

| 项目 | 历史来源 |
| --- | --- |
| 发布者 | [chenditc/investment_data](https://github.com/chenditc/investment_data) |
| 固定版本 | [2026-10-20 release](https://github.com/chenditc/investment_data/releases/tag/2026-10-20) |
| 行情文件 | [qlib_bin.tar.gz](https://github.com/chenditc/investment_data/releases/download/2026-10-20/qlib_bin.tar.gz) |
| 股票池 | 快照中的 `instruments/csi300.txt` 历史沪深300成员区间 |
| 日历 | 快照中的 `calendars/day.txt` |
| 日线字段 | `features/<证券代码>/<字段>.day.bin` 中的 open、high、low、close、volume、amount、factor |
| 历史研究证券范围 | 564 只曾进入所用区间股票池的证券，每个信号日按当日成员状态筛选 |

行情采用 **Qlib 格式**。[Microsoft Qlib](https://github.com/microsoft/qlib)
提供数据格式和读取工具，快照由 `chenditc/investment_data` 整理发布。
完整上游行情供应链及成分公告生效时点未独立核实。
重新获取 latest 或其他日期的快照可能得到不同数据与结果。

## 获取行情

以下命令仅在使用者执行时下载数据，仓库不携带行情文件：

```bash
mkdir -p data/downloads data/cn_data
curl --fail --location \
  https://github.com/chenditc/investment_data/releases/download/2026-10-20/qlib_bin.tar.gz \
  --output data/downloads/qlib_bin.tar.gz
tar -xzf data/downloads/qlib_bin.tar.gz --strip-components=1 -C data/cn_data
```

命令行支持固定快照下载与 Qlib 二进制转换：

```bash
strategy-pipeline download-data --destination data/snapshot
strategy-pipeline import-qlib \
  --provider data/snapshot/cn_data --output data/market.parquet
```

`strategy-pipeline prepare` 接受转换后的 CSV/Parquet 行情长表：

```text
date, asset, open, high, low, close, volume, amount, factor, member
```

`member` 应由当日是否落在该证券的历史成员区间内生成，不能用最新成分列表回填历史。
默认从 2017-11-01 读取预热行情，但用 `--universe-start 2018-01-01` 确定研究成员区间；
避免把预热期结束前已退池的证券误计入历史 564 只研究证券。
Qlib 数据读取方法见
[Qlib 数据文档](https://qlib.readthedocs.io/en/latest/component/data.html)。
导出的字段含义见 [数据接口约定](DATA_CONTRACT.md)。

回测将 `amount × 1000` 转换为人民币成交额。
输入的 `amount` 应以千元计量；转换其他数据源时需核对单位与复权定义。

## 多因子基础分数从哪里来

历史 95% 基础分数由市场门控截面注意力网络推理生成。历史流程使用沪深300训练的固定权重、
158 个股票因子和 63 个市场特征，
先按训练期统计量归一化，再对全部当日有效成分推理，最后与策略的 32 日可用样本对齐。
根据项目作者确认，基础 Transformer 权重训练截至 **2022-12-31（含）**，
固定历史测试区间为 **2024-07-01 至 2026-10-20（两端含）**；测试期使用固定权重。
用于本地验证和评价的归一化统计固定在 2018—2022 年，统计拟合边界与权重训练截止分别记录。

网络先用最后一日市场特征生成因子权重并重标定股票因子，再依次做单股 8 日时间注意力、
同日跨股票截面注意力和以最后一日为查询的时间聚合，最后线性输出分数；
结构说明见 [README](../README.md#基础分数网络)。

本仓库提供该网络、日截面训练器、Qlib Alpha158/市场变量构造及训练期归一化调度。
可提供历史同源分数，或准备与网络匹配的权重后运行以下流程。

Alpha158 构造还需要 Qlib 证券目录中的 `vwap.day.bin`；
市场特征需要 `SH000300`、`SH000905`、`SH000906` 三个指数的日线数据。
每个指数取日收益率，以及 5/10/20/30/60 日收益均值、收益标准差、
成交额均值与当日成交额之比、成交额标准差与当日成交额之比，共 3 × 21 = 63 维。

模型权重需单独准备，并通过 `base-predict --weights` 指定路径；
核对历史结果时需使用固定实验的同一权重文件。

```bash
strategy-pipeline factor-features \
  --provider data/snapshot/cn_data --output outputs/factor_panel.npz
strategy-pipeline base-predict \
  --panel outputs/factor_panel.npz --weights /path/to/base_weights.pt \
  --output data/multifactor_scores.parquet
```

也可由其他已定义的基础模型生成以下格式的分数：

```text
date, asset, multifactor_score
```

更换基础模型会改变输入分数和回测结果，属于新的实验配置。
基础分数不能使用当时未知的行情、未来标签或评价期拟合统计量。

## 仓库中的示例数据

`examples/run_synthetic_pipeline.py` 使用固定种子的 NumPy 随机数生成合成行情。
`--allow-demo-multifactor` 使用代码中的等权技术因子示例分数，不是历史网络输出。
这些输入用于测试准备、训练、融合和账户回放是否能执行，不能作为真实历史表现证据。

固定输入与策略代码的复现范围见 [复现说明](REPRODUCIBILITY.md)。
