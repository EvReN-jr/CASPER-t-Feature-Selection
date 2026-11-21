# -*- coding: utf-8 -*-
"""
Permütasyon sayısı N ve bootstrap B duyarlılığı.
Her katta öznitelik başına N_max=200 permütasyon ve B_max=100 bootstrap replikasyonu bir kez üretilir;
N∈{10,30,50,100,200} için ilk N replikasyonla μ, MC-SEM ve t_f; B∈{10,30,100} için bootstrap SE.
Kayıt: K, seçilen küme, ortalama |t_f|, ve seçilen kümenin doğruluğu (8 sınıflandırıcı).
κ taraması: MC için {1.5}, bootstrap için {0.25,0.5,1.0,1.5}.
"""
import sys, json, time, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, Discretizer, evaluate, eval_rows, set_threads, save_rows, MAIN9
from casper_core import (stratified_subsample, pairwise, select_eps, gamma_op, gamma_pfi_mc, gamma_pfi_boot,
                         tstat_rule)

NS = [10, 30, 50, 100, 200]
BS = [10, 30, 100]
KAPPA_BOOT = [0.25, 0.5, 1.0, 1.5]


def run_fold(ds, fi, tr, te, X, y, threads):
    set_threads(threads)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    ytr, yte = y[tr], y[te]; cols = list(Xtr.columns)
    Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
    sub = stratified_subsample(ytr.astype(float), 3000, 42); Xs, ys = Xd[sub], ytr[sub]
    eps, gb, _, _ = select_eps(pairwise(Xs), ys, gamma_op)
    t = time.time()
    mu_all, _, _, deltas, _ = gamma_pfi_mc(Xs, ys, eps, "op", max(NS), np.random.RandomState(42), gamma_base=gb)
    t_mc = time.time() - t
    t = time.time()
    _, bd = gamma_pfi_boot(Xs, ys, eps, "op", max(BS), np.random.RandomState(43))
    t_bt = time.time() - t
    sel, rows, cache = [], [], {}

    def ev(tag, idx):
        key = tuple(sorted(idx))
        if key not in cache:
            res, k = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in key]); cache[key] = (res, k)
        res, k = cache[key]
        rows.extend(eval_rows(dict(Dataset=ds, Fold=fi, Setting=tag), res, k))

    for N in NS:
        d = deltas[:, :N]; mu = d.mean(1); sem = d.std(1) / np.sqrt(N)
        idx, tf, q1 = tstat_rule(mu, sem, 1.5)
        tag = f"MC_N{N}_k1.5"
        sel.append(dict(Dataset=ds, Fold=fi, Setting=tag, SE="mc", N=N, kappa=1.5, K=len(idx),
                        mean_abs_t=float(np.mean(np.abs(tf))), median_sem=float(np.median(sem)),
                        Features=json.dumps(sorted(int(i) for i in idx)), Seconds=t_mc * N / max(NS)))
        ev(tag, idx)
    mu30 = deltas[:, :30].mean(1)
    for B in BS:
        se = bd[:, :B].std(1, ddof=1)
        for k in KAPPA_BOOT:
            idx, tf, q1 = tstat_rule(mu30, se, k)
            tag = f"BOOT_B{B}_k{k}"
            sel.append(dict(Dataset=ds, Fold=fi, Setting=tag, SE="boot", N=30, B=B, kappa=k, K=len(idx),
                            mean_abs_t=float(np.mean(np.abs(tf))), median_se=float(np.median(se)),
                            Features=json.dumps(sorted(int(i) for i in idx)), Seconds=t_bt * B / max(BS)))
            ev(tag, idx)
    print(f"[{ds}] fold {fi}", flush=True)
    return rows, sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="*", default=MAIN9)
    ap.add_argument("--jobs", type=int, default=5); a = ap.parse_args()
    for ds in a.datasets:
        X, y = load(ds); F = folds(X, y)
        out = Parallel(n_jobs=a.jobs)(delayed(run_fold)(ds, i + 1, tr, te, X, y, max(1, 16 // a.jobs))
                                      for i, (tr, te) in enumerate(F))
        save_rows([r for o in out for r in o[0]], "nperm", f"{ds}_metrics")
        save_rows([s for o in out for s in o[1]], "nperm", f"{ds}_selection")
        print("===", ds, flush=True)


if __name__ == "__main__":
    main()
