"""补充诊断：层内化到底改了什么 + 层内 IC 谱。

1) 名单层分布：beta=0 与 beta=1 的 band 名单在 vol60 十层上的占比（对比全池占比）。
2) 名单重叠：两个 beta 的逐日名单 Jaccard。
3) 层内 IC vs 全池 IC：H01 / mix 的逐日全池 Spearman 与「逐层 Spearman 按层规模加权」。
全程只用 2021-2024 四折。
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

import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1"); os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve()
REPO = _REPO
CACHE = REPO/"outputs/long_horizon"
OUT = _LH024_DIR / "cv_design_audit"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"research/LH023"))
from _probe_dgtw import pct_rank_2d, stratify
from _probe_e_attr import load_factors, wide_of
from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide

FOLDS=["wf2021","wf2022","wf2023","confirm2024"]; KQ=0.0022778298112255263; NB=10
DEPLOY={"H01":0.5,"H05_F1T2":0.1204,"H05_F2T2":0.3796}

def pr_row(x):
    out=np.full(x.shape,np.nan); m=np.isfinite(x); n=int(m.sum())
    if n<2: return out
    v=x[m]; o=np.argsort(v,kind="mergesort"); rk=np.empty(n); rk[o]=np.arange(n,dtype=float)
    s=v[o]; i=0
    while i<n:
        j=i+1
        while j<n and s[j]==s[i]: j+=1
        rk[o[i:j]]=(i+j-1)/2.0; i=j
    out[m]=rk/(n-1); return out

def within_pct(P,L,nbin=NB):
    out=np.full(P.shape,np.nan)
    for t in range(P.shape[0]):
        b=stratify(L[t],nbin)
        for k in range(nbin):
            m=(b==k)&np.isfinite(P[t])
            if m.sum()<20: continue
            out[t,m]=pr_row(P[t][m])
    return out

def within_ic(P,Y,L,nbin=NB):
    """逐日逐层 Spearman(P,Y)，按层内有效样本数加权平均。"""
    val=np.full(P.shape[0],np.nan)
    for t in range(P.shape[0]):
        b=stratify(L[t],nbin); num=0.0; den=0.0
        for k in range(nbin):
            m=(b==k)&np.isfinite(P[t])&np.isfinite(Y[t])
            if m.sum()<20: continue
            rp=rankdata(P[t][m]); ry=rankdata(Y[t][m])
            if rp.std()==0 or ry.std()==0: continue
            num+=float(np.corrcoef(rp,ry)[0,1])*int(m.sum()); den+=int(m.sum())
        if den>0: val[t]=num/den
    return val

fdf=load_factors(20210101,20241231)
rows=[]; comp=[]
for fold in FOLDS:
    df=load_fold(fold)
    pred=pd.read_parquet(CACHE/"LH003"/fold/"raw_predictions.parquet")
    pred["ts_code"]=pred["ts_code"].astype(str); pred["trade_date"]=pred["trade_date"].astype("int64")
    ex=[m for m in DEPLOY if m in pred.columns and m not in df.columns]
    if ex:
        df=df.merge(pred[["ts_code","trade_date"]+ex],on=["ts_code","trade_date"],how="left")
        df=df.sort_values(["trade_date","ts_code"],kind="stable").reset_index(drop=True)
    panel=Panel(df); W=wide_of(fdf,panel); vol=W["vol60"]
    raw={m:to_wide(panel,df[m].to_numpy(dtype="float64")) for m in DEPLOY}
    pct={m:pct_rank_2d(P) for m,P in raw.items()}
    mix=np.nansum(np.stack([w*pct[m] for m,w in DEPLOY.items()]),axis=0)
    base={"H01":pct["H01"],"mix":pct_rank_2d(mix)}
    for name,G in base.items():
        r_all=[]
        for tag,Pv in [("pool",raw["H01"] if name=="H01" else mix)]:
            pass
        # 层内 IC / 全池 IC 用原始分数（尺度无关）
        Pv = raw["H01"] if name=="H01" else mix
        pool=score(Pv,panel)["ic_mean"]
        wic=np.nanmean(within_ic(Pv,panel.Y,vol))
        rows.append(dict(fold=fold,name=name,pool_ic=pool,within_ic=float(wic)))
        # 名单层分布
        Lw=within_pct(G,vol); okb=np.isfinite(Lw)&np.isfinite(G)
        tops={}
        for beta in (0.0,1.0):
            P0=G.copy(); P0[okb]=(1-beta)*G[okb]+beta*Lw[okb]
            tops[beta]=fast_band(P0,panel.LU,KQ)[1]
        B=stratify(vol,NB)
        for beta in (0.0,1.0):
            sh=np.zeros(NB); tot=0
            for t in range(panel.T):
                sel=tops[beta][t]
                if not sel.any(): continue
                for k in range(NB):
                    sh[k]+=int((sel&(B[t]==k)).sum())
                tot+=int(sel.sum())
            sh=sh/tot
            comp.append(dict(fold=fold,name=name,beta=beta,**{f"L{k}":sh[k] for k in range(NB)}))
        jac=[]
        for t in range(panel.T):
            a,b=tops[0.0][t],tops[1.0][t]
            u=int((a|b).sum())
            if u: jac.append(int((a&b).sum())/u)
        comp.append(dict(fold=fold,name=name,beta="jaccard01",**{f"L{k}":np.nan for k in range(NB)}))
        rows[-1]["jaccard_b0_b1"]=float(np.mean(jac))
    print(f"[{fold}] done", flush=True)

R=pd.DataFrame(rows); R.to_csv(OUT/"within_ic_spectrum.csv",index=False)
C=pd.DataFrame(comp); C.to_csv(OUT/"within_layer_composition.csv",index=False)
pd.set_option("display.width",300)
print("\n=== 全池 IC vs 层内 IC ===")
print(R.round(4).to_string(index=False))
print("\n=== 四折均值 ===")
print(R.groupby("name")[["pool_ic","within_ic","jaccard_b0_b1"]].mean().round(4).to_string())
print("\n=== 名单层分布（四折均值，L0=最低波 L9=最高波）===")
cc=C[C.beta!="jaccard01"].copy()
cc["beta"]=cc["beta"].astype(float)
g=cc.groupby(["name","beta"])[[f"L{k}" for k in range(NB)]].mean()
print(g.round(4).to_string())
