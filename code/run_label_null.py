"""Label-permutation null for the Gamma-PFI drops (PIMP-style, Altmann et al. 2010).
For each outer training fold the labels are permuted B times and the whole CASPER-t fit is repeated
(Jenks codes, epsilon* rule, N=30 shuffles). For each feature the observed mean drop of the main run is compared
with its null distribution: p_f = (1 + #{b: mu_null_bf >= mu_obs_f}) / (B + 1).
Usage: run_label_null.py <dataset> <B> <folds e.g. 1-10> <jobs> <threads>
Output: results/labelnull/<ds>_f<fold>.json (one file per fold, so the run can resume)."""
import os, sys, json, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, Discretizer, set_threads, RESULTS
from casper_core import CasperSelector

OUT = os.path.join(RESULTS, "labelnull"); os.makedirs(OUT, exist_ok=True)


def null_fit(Xd, ytr, b, max_n, threads):
    set_threads(threads)
    yp = np.random.RandomState(10000 + b).permutation(ytr)
    cs = CasperSelector(max_n=max_n).fit(Xd, yp)
    return cs.mu_.tolist(), len(cs.support_idx_), float(cs.eps_star_), float(cs.gamma_base_)


def run(ds, B, fold_ids, jobs, threads, max_n=3000):
    X, y = load(ds)
    S = pd.read_csv(os.path.join(RESULTS, "main", f"{ds}_selection.csv"))
    c = S[S.Method == "CASPER-t"].set_index("Fold")
    for fi, (tr, te) in enumerate(folds(X, y), 1):
        if fi not in fold_ids:
            continue
        f = os.path.join(OUT, f"{ds}_f{fi}.json")
        if os.path.exists(f):
            continue
        pp = FoldPreprocessor().fit(X.iloc[tr])
        Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True)
        cols = list(Xtr.columns)
        Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
        ytr = y[tr]
        t = time.time()
        res = Parallel(n_jobs=jobs)(delayed(null_fit)(Xd, ytr, b, max_n, threads) for b in range(B))
        mu_obs = np.array(json.loads(c.loc[fi, "mu"]))
        sel = json.loads(c.loc[fi, "Features"])
        assert len(mu_obs) == len(cols)
        M = np.array([r[0] for r in res])                      # B x p
        pv = (1 + (M >= mu_obs[None, :]).sum(0)) / (B + 1)
        json.dump(dict(Dataset=ds, Fold=fi, B=B, cols=cols, mu_obs=mu_obs.tolist(), p=pv.tolist(),
                       selected=[cols.index(s) for s in sel], null_K=[r[1] for r in res],
                       null_eps=[r[2] for r in res], null_gbase=[r[3] for r in res],
                       null_mu_mean=M.mean(0).tolist(), null_mu_q95=np.quantile(M, 0.95, axis=0).tolist(),
                       seconds=time.time() - t), open(f, "w"))
        print(f"ok {ds} fold {fi} B={B} {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    ds, B, fr, jobs, threads = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    a, b = map(int, fr.split("-"))
    run(ds, B, set(range(a, b + 1)), jobs, threads)
