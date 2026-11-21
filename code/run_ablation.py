# -*- coding: utf-8 -*-
"""
Sistematik ablasyon. Referans = CASPER-t
(jenks_km, L1, minimal komşuluk γ_op, permütasyon, t-kuralı κ=1.5, ε=argmax γ<0.999).
Her varyant TEK bileşeni değiştirir. İki ölçüm:
  * own   : varyantın kendi durma kuralının seçtiği küme (stopping + ranking birlikte)
  * matched: varyantın sıralamasının referansın K'sında kesilmesi (yalnız sıralama kalitesi)
"""
import json, time, argparse
import numpy as np
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, Discretizer, evaluate, eval_rows, set_threads, save_rows, MAIN9
from casper_core import CasperSelector

VARIANTS = {
    "REF": ({}, {}),
    "disc=jenks_exact": ({"disc": "jenks_exact"}, {}),
    "disc=equal_width": ({"disc": "ew"}, {}),
    "disc=equal_freq": ({"disc": "ef"}, {}),
    "disc=none(raw)": ({"disc": "none"}, {}),
    "metric=L2": ({}, {"metric": "l2"}),
    "metric=Linf": ({}, {"metric": "linf"}),
    "relation=equivalence(eps=0)": ({}, {"eps_rule": "zero"}),
    "eps_rule=coarsest": ({}, {"eps_rule": "coarsest"}),
    "nbhd=ball T(x) (Hu)": ({}, {"gamma": "tol"}),
    "nbhd=near-gamma": ({}, {"gamma": "near"}),
    "importance=deletion": ({}, {"importance": "delete", "rule": "q1"}),
    "rule=Q1 point estimate": ({}, {"rule": "q1"}),
    "rule=positive mean": ({}, {"rule": "positive"}),
}


def run_fold(ds, fi, tr, te, X, y, threads):
    set_threads(threads)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    ytr, yte = y[tr], y[te]; cols = list(Xtr.columns)
    discs, rows, sel, Kref = {}, [], [], None
    for name, (dkw, ckw) in VARIANTS.items():
        dm = dkw.get("disc", "jenks_km")
        if dm not in discs:
            discs[dm] = Discretizer(dm).fit(Xtr).transform(Xtr).values.astype(float)
        t = time.time()
        try:
            cs = CasperSelector(**ckw).fit(discs[dm], ytr)
        except Exception as e:
            print(f"[{ds} f{fi}] {name} hata: {e}", flush=True); continue
        sec = time.time() - t
        own = list(cs.support_idx_)
        if name == "REF":
            Kref = len(own)
        matched = list(cs.ranking_[:Kref])
        sel.append(dict(Dataset=ds, Fold=fi, Variant=name, K=len(own), Kref=Kref, eps=cs.eps_star_,
                        gamma_base=cs.gamma_base_, Seconds=sec, Own=json.dumps(sorted(map(int, own))),
                        Matched=json.dumps(sorted(map(int, matched)))))
        for kind, idx in (("own", own), ("matched", matched)):
            res, k = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in idx])
            rows.extend(eval_rows(dict(Dataset=ds, Fold=fi, Variant=name, Kind=kind), res, k))
    print(f"[{ds}] fold {fi}", flush=True)
    return rows, sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="*", default=MAIN9)
    ap.add_argument("--jobs", type=int, default=5); a = ap.parse_args()
    for ds in a.datasets:
        X, y = load(ds); F = folds(X, y)
        out = Parallel(n_jobs=a.jobs)(delayed(run_fold)(ds, i + 1, tr, te, X, y, max(1, 16 // a.jobs))
                                      for i, (tr, te) in enumerate(F))
        save_rows([r for o in out for r in o[0]], "ablation", f"{ds}_metrics")
        save_rows([s for o in out for s in o[1]], "ablation", f"{ds}_selection")
        print("===", ds, flush=True)


if __name__ == "__main__":
    main()
