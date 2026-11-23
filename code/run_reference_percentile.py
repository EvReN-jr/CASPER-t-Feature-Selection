"""Reference point of the elimination rule: the first quartile against the 10th and the 50th percentile.
The selection of every fold is rebuilt from the stored permutation means and standard errors (results/main) with
kappa=1.5, and each subset is evaluated on the same outer test fold with the classifiers of the article
(mean over the seven base classifiers, as in the kappa table).
Usage: python run_reference_percentile.py [jobs]   Output: results/reference_percentile/folds.csv, summary.csv"""
import os, sys, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, evaluate, set_threads, MAIN9, RESULTS

QS = [10, 25, 50]
OUT = os.path.join(RESULTS, "reference_percentile"); os.makedirs(OUT, exist_ok=True)


def select(mu, se, q, kappa=1.5):
    mu, se = np.asarray(mu, float), np.asarray(se, float)
    ref = np.percentile(mu, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, (mu - ref) / se, np.where(mu > ref, np.inf, -np.inf))
    idx = [i for i in range(len(mu)) if t[i] > kappa]
    return idx if idx else [int(np.argmax(mu))]


def run_fold(ds, fi, tr, te, X, y, row):
    set_threads(2)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    cols = list(Xtr.columns); mu, se = json.loads(row.mu), json.loads(row.se)
    base = set(select(mu, se, 25)); assert {cols[i] for i in base} == set(json.loads(row.Features))
    out, cache = [], {}
    for q in QS:
        idx = select(mu, se, q); key = tuple(sorted(idx))
        if key not in cache:
            res, _ = evaluate(Xtr, Xte, y[tr], y[te], [cols[i] for i in idx])
            cache[key] = np.mean([m["ACCU"] for c, m in res.items() if c != "HYBRID"])
        s = set(idx)
        out.append(dict(Dataset=ds, Fold=fi, q=q, K=len(idx), p=len(cols), ACCU=cache[key],
                        Jaccard=len(s & base) / len(s | base)))
    return out


def main(jobs):
    rows = []
    for ds in MAIN9:
        X, y = load(ds)
        S = pd.read_csv(os.path.join(RESULTS, "main", f"{ds}_selection.csv")); c = S[S.Method == "CASPER-t"].set_index("Fold")
        out = Parallel(n_jobs=jobs)(delayed(run_fold)(ds, fi, tr, te, X, y, c.loc[fi])
                                    for fi, (tr, te) in enumerate(folds(X, y), 1))
        rows += [r for o in out for r in o]; print(ds, "ok", flush=True)
    R = pd.DataFrame(rows); R["ret"] = R.K / R.p; R.to_csv(os.path.join(OUT, "folds.csv"), index=False)
    A = R.groupby("q").agg(K=("K", "mean"), retained=("ret", "mean"), ACCU=("ACCU", "mean"), Jaccard=("Jaccard", "mean"))
    A["retained"] *= 100; A["ACCU"] *= 100; A.to_csv(os.path.join(OUT, "summary.csv")); print(A.round(2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
