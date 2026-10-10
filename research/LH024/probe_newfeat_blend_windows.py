"""blend 扫描的稳健性判定：内部极值是真信号还是噪声。

只用已落盘的 newfeat_blend_windows.csv（8 窗 × 11 w），不重训。
判据（脚本预注册）：
  ① w* → 1（无极值） ⇒ 关闭
  ② 存在 w 使 confirm2024 Δα ≥ −0.005 且 wf2023 Δα ≥ +0.02 ⇒ 保险成立
"""

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

import numpy as np, pandas as pd

W = pd.read_csv("cv_design_audit/newfeat_blend_windows.csv")
REF = pd.read_csv("cv_design_audit/newfeat_halves.csv")
pd.set_option("display.width", 340)

# 用 w=1（纯 H01）作为参考；逐窗数值取 windows 文件里 w=1 行（应等于 halves 的 H01 行）
base = W[W.w == 1.0].set_index("win")
print("=== 自检：blend w=1 是否逐窗复现 newfeat_halves 的 H01 ===")
h = REF[REF.tag == "H01"].set_index("win")
chk = pd.DataFrame({
    "alpha_blend": base["alpha"], "alpha_halves": h["alpha"],
    "ic_blend": base["ic"], "ic_halves": h["ic"],
})
chk["d_alpha"] = (chk.alpha_blend - chk.alpha_halves).abs()
chk["d_ic"] = (chk.ic_blend - chk.ic_halves).abs()
print(chk[["d_alpha", "d_ic"]].max().to_string())
print()

# 逐窗 Δ（相对 H01）
d = W.copy()
for m in ("alpha", "ic", "turnover", "profile", "final"):
    b = base[m]
    d[f"d{m}"] = d[m] - d.win.map(b)
d["strict"] = 0.4 * d.dic - 0.3 * d.dturnover
d["withalpha"] = 0.4 * d.dic + 0.3 * d.dalpha - 0.3 * d.dturnover

print("=== 逐窗 Δα（相对纯 H01），行=窗，列=w ===")
pv = d.pivot_table(index="win", columns="w", values="dalpha")
pv = pv.sort_index()
print(pv.round(5).to_string())
print()
print("=== 逐窗 strict Δ（0.4ΔIC − 0.3ΔT），行=窗 ===")
pv2 = d.pivot_table(index="win", columns="w", values="strict").sort_index()
print(pv2.round(5).to_string())
print()
print("=== 逐窗 withalpha Δ，行=窗 ===")
pv3 = d.pivot_table(index="win", columns="w", values="withalpha").sort_index()
print(pv3.round(5).to_string())
