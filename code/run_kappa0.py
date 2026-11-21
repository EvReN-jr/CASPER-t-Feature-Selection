"""kappa=0 (pure quartile cut, no margin) rebuilt from the stored mu/se of the main run, evaluated on the same
outer folds with the same classifiers as run_kappa_eps.py. kappa=1.5 is recomputed as a consistency check
against the stored selection. Output: results/kappa0/<ds>.csv"""
import os, sys, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, evaluate, set_threads, MAIN9, RESULTS
from run_kappa_eps import select

OUT = os.path.join(RESULTS, "kappa0"); os.makedirs(OUT, exist_ok=True)


def run_fold(ds, fi, tr, te, X, y, row, kappas):
    set_threads(2)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    cols = list(Xtr.columns)
    mu, se = json.loads(row.mu), json.loads(row.se)
    stored = set(json.loads(row.Features))
    out, cache = [], {}
    for k in kappas:
        idx = select(mu, se, k)
        names = [cols[i] for i in idx]
        if k == 1.5:
            assert set(names) == stored, (ds, fi, set(names) ^ stored)
        key = tuple(sorted(idx))
        if key not in cache:
            res, _ = evaluate(Xtr, Xte, y[tr], y[te], names)
            cache[key] = np.mean([m["ACCU"] for c, m in res.items() if c != "HYBRID"])
        out.append(dict(Dataset=ds, Fold=fi, kappa=k, K=len(idx), p=len(cols), ACCU=cache[key],
                        Features=json.dumps(names)))
    return out


if __name__ == "__main__":
    jobs_n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for ds in MAIN9:
        f = os.path.join(OUT, f"{ds}.csv")
        if os.path.exists(f):
            continue
        S = pd.read_csv(os.path.join(RESULTS, "main", f"{ds}_selection.csv"))
        c = S[S.Method == "CASPER-t"].sort_values("Fold")
        X, y = load(ds)
        res = Parallel(n_jobs=jobs_n)(delayed(run_fold)(ds, fi, tr, te, X, y, c[c.Fold == fi].iloc[0], [0.0, 1.5])
                                      for fi, (tr, te) in enumerate(folds(X, y), 1))
        pd.DataFrame([x for r in res for x in r]).to_csv(f, index=False)
        print("ok", ds, flush=True)
