"""Summary of the label-permutation null (results/labelnull/*.json).
Per dataset: B, folds, share of features with p<0.05, share with BH q<0.05 (within fold), and the same shares
among the features CASPER-t retained and among those it removed; median p of retained vs removed."""
import os, glob, json
import numpy as np, pandas as pd
from exp_common import RESULTS


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


rows = []
for f in sorted(glob.glob(os.path.join(RESULTS, "labelnull", "*.json"))):
    d = json.load(open(f)); p = np.array(d["p"]); q = bh(p); s = np.zeros(len(p), bool); s[d["selected"]] = True
    rows.append(dict(Dataset=d["Dataset"], Fold=d["Fold"], B=d["B"], n_feat=len(p),
                     sig05=(p < 0.05).mean(), sigBH=(q < 0.05).mean(),
                     sel_sigBH=(q[s] < 0.05).mean() if s.any() else np.nan,
                     rem_sigBH=(q[~s] < 0.05).mean() if (~s).any() else np.nan,
                     sel_p_med=np.median(p[s]), rem_p_med=np.median(p[~s]) if (~s).any() else np.nan,
                     null_K=np.mean(d["null_K"]), K=int(s.sum())))
R = pd.DataFrame(rows)
A = R.groupby("Dataset").agg(B=("B", "first"), folds=("Fold", "count"), p=("n_feat", "first"), K=("K", "mean"),
                             null_K=("null_K", "mean"), sig05=("sig05", "mean"), sigBH=("sigBH", "mean"),
                             sel_sigBH=("sel_sigBH", "mean"), rem_sigBH=("rem_sigBH", "mean"),
                             sel_p_med=("sel_p_med", "median"), rem_p_med=("rem_p_med", "median"))
print(A.round(3).to_string())
A.to_markdown(os.path.join(RESULTS, "tables", "T_labelnull.md"), floatfmt=".3f")
