"""wf2023 评估窗的弱，是 H01 独有还是整个长周期模型族共有？"""

# --- LH024 archival shim: 按脚本位置反查仓库根与数据目录，与当前工作目录无关 ---
import os as _os
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = (
    _P(_os.environ["FINTECH_ROOT"])
    if _os.environ.get("FINTECH_ROOT")
    else next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
)
_LH024_DIR = _P(_os.environ.get("LH024_DIR", _REPO / "experiments" / "LH024"))
if _LH024_DIR.is_dir():
    _os.chdir(_LH024_DIR)  # 脚本内部使用 cv_design_audit/ 相对路径
# ---------------------------------------------------------------------------

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = _REPO
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "research" / "LH023"))
from _probe_root_cause import Panel, score, to_wide  # noqa: E402
from _probe_dgtw import pct_rank_2d  # noqa: E402
M = ["H01", "H05_F2T2", "H10_F2T2", "H20_F2T2", "H30_F2T2"]
res = {}
for fold in ["wf2021", "wf2022", "wf2023", "confirm2024"]:
    c = pd.read_parquet(REPO / "outputs/long_horizon" / f"_probe_rc_{fold}.parquet")
    c["trade_date"] = c["trade_date"].astype("int64")
    c = c.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    p = Panel(c)
    dates = sorted(c.trade_date.unique())
    n = int(len(dates) * 0.6)
    ev = np.arange(n, len(dates))
    for m in M:
        if m not in c.columns:
            continue
        P0 = pct_rank_2d(to_wide(p, c[m].to_numpy(dtype="float64")))
        ic = score(P0, p)["_ic"]
        res.setdefault(m, {})[fold] = float(np.nanmean(ic[ev]))
df = pd.DataFrame(res).T
df["其余三折均值"] = df[[c for c in df.columns if c != "wf2023"]].mean(axis=1)
df["wf2023 差"] = df["wf2023"] - df["其余三折均值"]
df["相对降幅"] = df["wf2023 差"] / df["其余三折均值"]
pd.set_option("display.width", 300)
print("=== 各模型在四折评估窗的 pooled IC（20230809~20231229 = wf2023）===")
print(df.round(5).to_string())
print("\n=== 同一口径下 wf2023 的 IC 是否对全部模型都偏低 ===")
print(f"  全部模型 wf2023 都低于其余三折：{bool((df['wf2023 差'] < 0).all())}")
