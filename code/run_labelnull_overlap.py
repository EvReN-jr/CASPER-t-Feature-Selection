"""Does the selected SET depend on the labels, and does the score depend on the number of Jenks levels k_f?
(1) For every outer training fold, L permuted labelings are refitted (same seeds 10000+b as run_label_null.py)
    and the Jaccard index between each null subset and the subset on the real labels is recorded, together with
    the Jaccard index expected for two random subsets of the same sizes.
(2) Spearman correlation, within each fold, between k_f and the observed mean drop, the null mean drop
    (results/labelnull) and retention.
Usage: python run_labelnull_overlap.py [L] [jobs]     Output: results/labelnull_overlap/summary.csv, folds.csv"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, Discretizer, set_threads, RESULTS, MAIN9
from casper_core import CasperSelector

OUT = os.path.join(RESULTS, "labelnull_overlap"); os.makedirs(OUT, exist_ok=True)


def jac(a, b):
    a, b = set(a), set(b); return len(a & b) / len(a | b) if a | b else 1.0


def exp_jac_random(p, k1, k2, rng, R=2000):
    return float(np.mean([jac(rng.choice(p, k1, replace=False), rng.choice(p, k2, replace=False)) for _ in range(R)]))


def null_sel(Xd, ytr, b):
    set_threads(1)
    yp = np.random.RandomState(10000 + b).permutation(ytr)
    return list(CasperSelector(max_n=3000).fit(Xd, yp).support_idx_)


def main(L, jobs):
    rows = []
    for ds in MAIN9:
        X, y = load(ds)
        S = pd.read_csv(os.path.join(RESULTS, "main", f"{ds}_selection.csv")); c = S[S.Method == "CASPER-t"].set_index("Fold")
        for fi, (tr, _) in enumerate(folds(X, y), 1):
            Xtr = FoldPreprocessor().fit(X.iloc[tr]).transform(X.iloc[tr]).reset_index(drop=True)
            cols = list(Xtr.columns); p = len(cols)
            disc = Discretizer().fit(Xtr); Xd = disc.transform(Xtr).values.astype(float)
            kf = np.array([disc.base_.column_k_.get(cname, len(np.unique(Xd[:, j]))) for j, cname in enumerate(cols)], float)
            real = [cols.index(f) for f in json.loads(c.loc[fi, "Features"])]
            nulls = Parallel(n_jobs=jobs)(delayed(null_sel)(Xd, y[tr], b) for b in range(L))
            rng = np.random.RandomState(fi)
            J = [jac(real, s) for s in nulls]
            Jr = [exp_jac_random(p, len(real), len(s), rng, 300) for s in nulls]
            mu_obs = np.array(json.loads(c.loc[fi, "mu"]))
            ln = os.path.join(RESULTS, "labelnull", f"{ds}_f{fi}.json")
            mu_null = np.array(json.load(open(ln))["null_mu_mean"]) if os.path.exists(ln) else np.full(p, np.nan)
            ret = np.isin(np.arange(p), real).astype(float)
            sp = lambda a, b: spearmanr(a, b)[0] if np.ptp(kf) > 0 and np.ptp(a) > 0 else np.nan
            rows.append(dict(Dataset=ds, Fold=fi, p=p, K=len(real), J_null=np.mean(J), J_random=np.mean(Jr),
                             kf_min=kf.min(), kf_max=kf.max(), rho_kf_mu_obs=sp(mu_obs, kf), rho_kf_mu_null=sp(mu_null, kf),
                             rho_kf_retained=sp(ret, kf)))
        d = pd.DataFrame([r for r in rows if r["Dataset"] == ds])
        print(f"{ds}: J_null {d.J_null.mean():.2f} vs random {d.J_random.mean():.2f} | rho(kf,mu_obs) {d.rho_kf_mu_obs.mean():.2f} "
              f"rho(kf,mu_null) {d.rho_kf_mu_null.mean():.2f}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "folds.csv"), index=False)
    A = R.groupby("Dataset", sort=False).mean(numeric_only=True).drop(columns="Fold")
    A.to_csv(os.path.join(OUT, "summary.csv")); print(A.round(2).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
