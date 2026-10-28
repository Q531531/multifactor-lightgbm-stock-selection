"""无需 Qlib 依赖的日线二进制读取、历史股票池装配和固定快照下载。"""

from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import urllib.request

import numpy as np
import pandas as pd

from .data import MARKET_FIELDS

SNAPSHOT_URL = "https://github.com/chenditc/investment_data/releases/download/2026-10-20/qlib_bin.tar.gz"
SNAPSHOT_SHA256 = "ff60683d0459a12e9432b2b0ce2aca1a6fe24a3114aed3247c87d157c709f1e4"
SNAPSHOT_BYTES = 568654372


def read_binary(path: Path, calendar_size: int) -> np.ndarray:
    result = np.full(calendar_size, np.nan, dtype=np.float32)
    if not path.exists():
        return result
    values = np.fromfile(path, dtype="<f4")
    if len(values) < 2:
        raise ValueError(f"无效 Qlib 日线文件: {path.name}")
    if not np.isfinite(values[0]):
        raise ValueError(f"无效 Qlib 日历偏移: {path.name}")
    offset = int(values[0])
    if (
        not np.isfinite(values[0])
        or values[0] != offset
        or offset < 0
        or offset + len(values) - 1 > calendar_size
    ):
        raise ValueError(f"Qlib 文件日历偏移越界: {path.name}")
    result[offset : offset + len(values) - 1] = values[1:]
    return result


def read_qlib_panel(
    provider: str | Path,
    *,
    market="csi300",
    start="2017-11-01",
    end="2026-10-20",
    universe_start="2018-01-01",
) -> dict:
    root = Path(provider)
    calendar = pd.DatetimeIndex(
        pd.to_datetime((root / "calendars/day.txt").read_text().splitlines())
    )
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Qlib 日历必须严格递增且唯一")
    if "/" in market or "\\" in market:
        raise ValueError("market 应为股票池名称")
    intervals = pd.read_csv(
        root / "instruments" / f"{market}.txt",
        sep="\t",
        header=None,
        names=["asset", "start", "end"],
    )
    intervals["start"] = pd.to_datetime(intervals.start)
    intervals["end"] = pd.to_datetime(intervals.end)
    intervals["asset"] = intervals.asset.str.upper()
    if (
        intervals[["asset", "start", "end"]].isna().any().any()
        or (intervals.end < intervals.start).any()
    ):
        raise ValueError("股票池成员区间无效")
    study_start = max(pd.Timestamp(start), pd.Timestamp(universe_start))
    intervals = intervals[(intervals.end >= study_start) & (intervals.start <= end)]
    symbols = np.array(sorted(intervals.asset.unique()), dtype=str)
    keep = (calendar >= start) & (calendar <= end)
    dates = calendar[keep]
    if not len(dates) or not len(symbols):
        raise ValueError("数据区间或股票池为空")
    member = np.zeros((len(dates), len(symbols)), dtype=bool)
    for j, symbol in enumerate(symbols):
        if not symbol.isalnum():
            raise ValueError("证券代码必须为字母数字")
        for row in intervals[intervals.asset == symbol].itertuples():
            member[:, j] |= (dates >= row.start) & (dates <= row.end)
    panel = dict(dates=dates.to_numpy(), symbols=symbols, member=member)
    for field in MARKET_FIELDS:
        panel[field] = np.column_stack(
            [
                read_binary(
                    root / "features" / symbol.lower() / f"{field}.day.bin",
                    len(calendar),
                )[keep]
                for symbol in symbols
            ]
        )
    return panel


def download_snapshot(destination: str | Path) -> Path:
    """只有显式调用时联网；流式校验后提取，无数据随代码分发。"""
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("下载目的目录必须为空，以避免混合数据版本")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "qlib_bin.tar.gz"
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(SNAPSHOT_URL, timeout=45) as source,
            archive.open("wb") as out,
        ):
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                out.write(chunk)
        if (
            archive.stat().st_size != SNAPSHOT_BYTES
            or digest.hexdigest() != SNAPSHOT_SHA256
        ):
            raise ValueError("下载快照与登记版本不一致")
        root = destination / "cn_data"
        root.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            for item in tar:
                parts = Path(item.name).parts
                if (
                    not parts
                    or parts[0] != "qlib_bin"
                    or ".." in parts
                    or not (item.isdir() or item.isfile())
                ):
                    raise ValueError("快照包含不符合约定的成员")
                target = root.joinpath(*parts[1:])
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(item) as stream, target.open("wb") as out:
                        shutil.copyfileobj(stream, out)
        (destination / "snapshot_receipt.json").write_text(
            json.dumps(
                {
                    "url": SNAPSHOT_URL,
                    "sha256": digest.hexdigest(),
                    "bytes": SNAPSHOT_BYTES,
                    "release": "2026-10-20",
                    "passed": True,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        archive.unlink(missing_ok=True)
    return root
