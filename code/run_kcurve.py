# -*- coding: utf-8 -*-
"""
Doğruluk–K eğrileri.
Her K∈{1..p} (p>30 ise ~12 noktalı ızgara) için CASPER-t sıralaması (μ_Δγ), rastgele (10 çekim ort.),
ANOVA, RF, mRMR ve NRS(Hu) sıralamaları aynı K'da değerlendirilir. Eğri altı alan (normalize) ve
rastgeleye göre fark raporlanır. Büyük veri setlerinde yalnız ilk 5 kat (maliyet; belgelenir).
Sınıflandırıcı: 8 sınıflandırıcı ortalaması.
"""
import json, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from exp_common import load, folds, FoldPreprocessor, Discretizer, evaluate, set_threads, save_rows, MAIN9
from casper_core import CasperSelector
import baselines as B


def kgrid(p):
    return list(range(1, p + 1)) if p <= 30 else sorted(set(np.unique(np.round(np.linspace(1, p, 12)).astype(int))))


def run_fold(ds, fi, tr, te, X, y, threads):
    set_threads(threads)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    ytr, yte = y[tr], y[te]; cols = list(Xtr.columns); p = len(cols)
    Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
    cs = CasperSelector().fit(Xd, ytr)
    ranks = {"CASPER-t": list(cs.ranking_),
             "ANOVA": list(np.argsort(-np.nan_to_num(f_classif(Xtr.values, ytr)[0]), kind="stable")),
             "RF": list(np.argsort(-RandomForestClassifier(100, random_state=42, n_jobs=1).fit(Xtr.values, ytr).feature_importances_, kind="stable")),
             "mRMR": B.mrmr(Xtr, ytr)[0], "NRS(Hu)": B.nrs_hu(Xtr, ytr)[0]}
    rows = []
    rng = np.random.RandomState(500 + fi)
    for K in kgrid(p):
        for m, r in ranks.items():
            res, _ = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in r[:K]])
            rows.append(dict(Dataset=ds, Fold=fi, Method=m, K=K, p=p, Kcasper=len(cs.support_idx_),
                             ACCU=np.mean([v["ACCU"] for v in res.values()]), HYB=res["HYBRID"]["ACCU"]))
        acc = []
        for _ in range(10 if p > K else 1):
            idx = rng.choice(p, K, replace=False)
            res, _ = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in idx])
            acc.append((np.mean([v["ACCU"] for v in res.values()]), res["HYBRID"]["ACCU"]))
        rows.append(dict(Dataset=ds, Fold=fi, Method="Random", K=K, p=p, Kcasper=len(cs.support_idx_),
                         ACCU=np.mean([a[0] for a in acc]), HYB=np.mean([a[1] for a in acc])))
    print(f"[{ds}] fold {fi}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="*", default=MAIN9)
    ap.add_argument("--jobs", type=int, default=5); a = ap.parse_args()
    for ds in a.datasets:
        X, y = load(ds); F = folds(X, y)
        if len(y) > 1000:
            F = F[:5]
        out = Parallel(n_jobs=a.jobs)(delayed(run_fold)(ds, i + 1, tr, te, X, y, max(1, 16 // a.jobs))
                                      for i, (tr, te) in enumerate(F))
        save_rows([r for o in out for r in o], "kcurve", ds)
        print("===", ds, flush=True)


if __name__ == "__main__":
    main()
