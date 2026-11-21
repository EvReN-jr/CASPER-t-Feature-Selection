"""Experiment: is the value of CASPER-t in the gamma score or only in the cut?
Random-forest permutation importance (same settings as the StdPFI baseline, but n_repeats=30) is computed on the
training fold, its Monte Carlo SE is std/sqrt(30), and the same CASPER-t cut (Q1 + kappa*SE, kappa in {0, 1.5})
is applied. Each subset is evaluated at its own K on the same outer folds with the same classifiers.
Output: results/pfirule/<ds>.csv"""
import os, sys, json, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, evaluate, eval_rows, set_threads, MAIN9, RESULTS
from casper_core import stratified_subsample
from run_kappa_eps import select

OUT = os.path.join(RESULTS, "pfirule"); os.makedirs(OUT, exist_ok=True)
N_REP = 30


def run_fold(ds, fi, tr, te, X, y):
    set_threads(2)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    ytr, yte = y[tr], y[te]; cols = list(Xtr.columns)
    A = Xtr.values.astype(float); yy = np.asarray(ytr).astype(int)
    sub = stratified_subsample(yy.astype(float), 3000, 42)
    t = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(A, yy)
    imp = permutation_importance(rf, A[sub], yy[sub], n_repeats=N_REP, random_state=42, scoring="accuracy").importances
    sec = time.time() - t
    mu = imp.mean(1); se = imp.std(1) / np.sqrt(N_REP)
    rows = []
    for k in (0.0, 1.5):
        idx = select(mu, se, k)
        res, kk = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in idx])
        for r in eval_rows(dict(Dataset=ds, Fold=fi, Method=f"RF-PFI+cut(kappa={k})", K=len(idx), p=len(cols),
                                Seconds=sec, zero_se=int((se <= 1e-9).sum()), neg_mu=int((mu < 0).sum())), res, kk):
            rows.append(r)
    return rows


if __name__ == "__main__":
    jobs_n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for ds in MAIN9:
        f = os.path.join(OUT, f"{ds}.csv")
        if os.path.exists(f):
            continue
        X, y = load(ds)
        res = Parallel(n_jobs=jobs_n)(delayed(run_fold)(ds, fi, tr, te, X, y) for fi, (tr, te) in enumerate(folds(X, y), 1))
        pd.DataFrame([x for r in res for x in r]).to_csv(f, index=False)
        print("ok", ds, flush=True)
