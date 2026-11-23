"""Synthetic data of run_synthetic.py: ranking AUC of the Gamma-PFI means across the whole threshold grid.
For every candidate epsilon of the training fold, gamma(eps) and the N=30 permutation means are computed with the
exact engine, and the ranking AUC against the five informative features is recorded. The row with is_star=True
is the threshold chosen by the CASPER-t rule.
Usage: python run_synthetic_eps.py      Output: results/synthetic/eps_sweep.csv"""
import os
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from exp_common import Discretizer, set_threads, RESULTS
from casper_core import CasperSelector, stratified_subsample, _fast_gamma_op, pairwise
import casper_fast

OUT = os.path.join(RESULTS, "synthetic"); os.makedirs(OUT, exist_ok=True)


def one(p, seed):
    set_threads(1)
    X, y = make_classification(n_samples=600, n_features=p, n_informative=5, n_redundant=0, n_repeated=0,
                               n_clusters_per_class=2, class_sep=1.0, flip_y=0.0, shuffle=False, random_state=seed)
    truth = np.r_[np.ones(5), np.zeros(p - 5)]
    idx = np.random.RandomState(seed).permutation(600); tr = idx[:420]
    Xtr = pd.DataFrame(X[tr], columns=[f"f{i}" for i in range(p)]); ytr = y[tr]
    Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
    cs = CasperSelector().fit(Xd, ytr)                      # rule threshold, same as run_synthetic.py
    sub = stratified_subsample(ytr.astype(float), 3000, 42); Xs, ys = Xd[sub], ytr[sub]
    gfn = _fast_gamma_op(Xs, ys); D = pairwise(Xs, "l1", None)
    rows = []
    for eps in cs.eps_grid_:
        gb = gfn(D, ys, eps)
        _, _, deltas = casper_fast.gamma_pfi(Xs, ys, eps + 1e-12, gb, n_rep=30, rng=np.random.RandomState(42),
                                             return_deltas=True)
        mu = deltas.mean(1)
        auc = roc_auc_score(truth, mu) if np.ptp(mu) > 0 else 0.5
        rows.append(dict(p=p, seed=seed, eps=float(eps), gamma=float(gb), AUC=float(auc),
                         mu_inf=float(mu[:5].mean()), mu_noise=float(mu[5:].mean()), is_star=bool(eps == cs.eps_star_)))
    return rows


if __name__ == "__main__":
    jobs = [(p, s) for p in (10, 20, 50) for s in range(5)]
    out = Parallel(n_jobs=8)(delayed(one)(p, s) for p, s in jobs)
    R = pd.DataFrame([r for o in out for r in o]); R.to_csv(os.path.join(OUT, "eps_sweep.csv"), index=False)
    star = R[R.is_star].groupby("p")[["AUC", "gamma"]].mean()
    best = R.loc[R.groupby(["p", "seed"]).AUC.idxmax()].groupby("p")[["AUC", "gamma", "eps"]].mean()
    print("rule threshold:\n", star.round(3)); print("best threshold on the grid:\n", best.round(3))
