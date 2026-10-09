"""为四折扫描建折目录别名（硬链接，零磁盘占用）。

背景：``rerank_eval.py`` 按 ``outputs/long_horizon/<study>/<fold>/raw_predictions.parquet``
取预测。但四折里——

* ``wf2023`` / ``confirm2024`` 的窗口与 validation 的 ``fold1`` / ``fold2`` 逐位相同，
  已在 LH003（长周期族）、LH004（窗口目标族）训过；
* ``wf2021`` / ``wf2022`` 由 LH006 新训。

所以这里把源文件硬链接成统一折名，让一次扫描能跨四个折跑完。列多的那边不做裁剪
（rerank 只取 directions 里用到的列）。

用法（仓库根目录）::

    python src/long_horizon/fold_aliases.py
"""

import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
LH = ROOT / "outputs" / "long_horizon"

PAIRS = [
    ("LH003", "wf2021", "LH006", "wf2021"),
    ("LH003", "wf2022", "LH006", "wf2022"),
    ("LH003", "wf2023", "LH003", "fold1"),
    ("LH003", "confirm2024", "LH003", "fold2"),
    ("LH004", "wf2021", "LH006", "wf2021"),
    ("LH004", "wf2022", "LH006", "wf2022"),
    ("LH004", "wf2023", "LH004", "fold1"),
    ("LH004", "confirm2024", "LH004", "fold2"),
]

for study, fold, src_study, src_fold in PAIRS:
    dest_dir = LH / study / fold
    dest_dir.mkdir(parents=True, exist_ok=True)
    link = dest_dir / "raw_predictions.parquet"
    target = LH / src_study / src_fold / "raw_predictions.parquet"
    if not target.exists():
        sys.exit(f"[FAIL] 缺少源文件 {target}")
    if link.exists():
        link.unlink()
    os.link(target, link)
    print(f"link {study}/{fold}  <-  {src_study}/{src_fold}")

print()
for study in ("LH003", "LH004"):
    for fold in ("wf2021", "wf2022", "wf2023", "confirm2024"):
        p = LH / study / fold / "raw_predictions.parquet"
        names = pq.read_schema(p).names
        models = [c for c in names if c not in ("ts_code", "trade_date")]
        rows = pq.ParquetFile(p).metadata.num_rows
        print(f"{study}/{fold}: {rows:,} 行 | {len(models)} 列 | {models}")
print("OK")
