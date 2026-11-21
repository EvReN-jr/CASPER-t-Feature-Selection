# -*- coding: utf-8 -*-
"""
Alt örneklemenin seçime etkisi + ölçeklenme.
(a) MAGIC ve HTRU2, ilk 5 dış kat: O(n^3) topolojik aşamanın alt örneklem boyutu
    m∈{500,1000,2000,3000,5000,8000}; her m için 3 farklı alt örnek tohumu.
    Ölçüm: K, seçilen küme, tohumlar arası Jaccard, m=8000 kümesine Jaccard, süre, doğruluk.
(b) Ölçeklenme eğrisi: MAGIC eğitim katından n∈{250,...,8000} için tek γ değerlendirmesi ve tam
    CASPER-t süresi (log-log eğim ≈ 3 beklenir: boolean matris çarpımı).
"""
import json, time, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, Discretizer, evaluate, eval_rows, set_threads, save_rows, jaccard
from casper_core import CasperSelector, pairwise, gamma_op, gamma_near

MS = [500, 1000, 2000, 3000, 5000]


def run_fold(ds, fi, tr, te, X, y, threads):
    set_threads(threads)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    ytr, yte = y[tr], y[te]; cols = list(Xtr.columns)
    Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
    rows, sel, cache = [], [], {}
    for m in MS:
        for s in (42, 7):
            t = time.time(); cs = CasperSelector(max_n=m, random_state=s).fit(Xd, ytr); sec = time.time() - t
            idx = sorted(map(int, cs.support_idx_))
            sel.append(dict(Dataset=ds, Fold=fi, m=m, seed=s, K=len(idx), eps=cs.eps_star_, Seconds=sec,
                            Features=json.dumps(idx)))
            key = tuple(idx)
            if key not in cache:
                cache[key] = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in idx])
            res, k = cache[key]
            rows.extend(eval_rows(dict(Dataset=ds, Fold=fi, m=m, seed=s), res, k))
    print(f"[{ds}] fold {fi}", flush=True)
    return rows, sel


def scaling():
    set_threads(16)
    X, y = load("magic")
    Xd = Discretizer().fit(X).transform(X).values.astype(float)
    rng = np.random.RandomState(0); out = []
    for n in (250, 500, 1000, 2000, 3000, 4000, 6000):
        idx = rng.choice(len(y), n, replace=False); D = pairwise(Xd[idx]); yy = y[idx]
        for fn in (gamma_op, gamma_near):
            ts = []
            for _ in range(3):
                t = time.time(); fn(D, yy, 6); ts.append(time.time() - t)
            out.append(dict(n=n, what=fn.__name__, seconds=float(np.median(ts))))
        t = time.time(); CasperSelector(max_n=n).fit(Xd[idx], yy); out.append(dict(n=n, what="CASPER-t full fit", seconds=time.time() - t))
        print("scaling", n, flush=True)
    save_rows(out, "subsample", "scaling_magic")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=3); a = ap.parse_args()
    scaling()
    for ds in ("magic",):
        X, y = load(ds); F = folds(X, y)[:3]
        out = Parallel(n_jobs=a.jobs)(delayed(run_fold)(ds, i + 1, tr, te, X, y, max(1, 16 // a.jobs))
                                      for i, (tr, te) in enumerate(F))
        save_rows([r for o in out for r in o[0]], "subsample", f"{ds}_metrics")
        save_rows([s for o in out for s in o[1]], "subsample", f"{ds}_selection")
        print("===", ds, flush=True)


if __name__ == "__main__":
    main()
