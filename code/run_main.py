# -*- coding: utf-8 -*-
"""
Ana karşılaştırma koşusu.
Her dış katta: kat-içi winsor+korelasyon → Jenks(eğitim) → 4 CASPER adayı {op,near}×{mc,boot}
+ klasik 9 + mRMR, Boruta, HSIC-Lasso, FRFS, QuickReduct, NRS(Hu) + rastgele-K.
Değerlendirme: ALL; adayların kendi K'sı; CASPER-t (op,mc) K'sına eşlenmiş tüm sıralamalar;
kendi durma kuralı olanların kendi K'sı. Tüm seçicilerin duvar-saati süresi kaydedilir.
Kullanım: python run_main.py [dataset ...] [--jobs 5]
"""
import sys, json, time, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import (load, folds, FoldPreprocessor, Discretizer, evaluate, eval_rows, set_threads,
                        save_rows, MAIN9)
import baselines as B
from casper_core import CasperSelector

CANDS = {"CASPER-t": ("op", "mc"), "CASPER-t[boot]": ("op", "boot"),
         "CASPER-t[near]": ("near", "mc"), "CASPER-t[near,boot]": ("near", "boot")}


def run_fold(ds, fi, tr, te, X, y, R, threads, lite=False):
    set_threads(threads)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True)
    Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    ytr, yte = y[tr], y[te]
    cols = list(Xtr.columns); p = len(cols)
    t = time.time(); disc = Discretizer().fit(Xtr); Xd = disc.transform(Xtr).values.astype(float)
    t_disc = time.time() - t
    sel, rows = [], []

    def add_sel(method, idx, sec, **kw):
        sel.append(dict(Dataset=ds, Fold=fi, Method=method, K=len(idx), p=p, Seconds=sec,
                        Features=json.dumps([cols[i] for i in idx]), **kw))

    cache = {}

    def cached_eval(idx):  # aynı alt küme tekrar değerlendirilmez (küçük p'de çok sık)
        key = tuple(sorted(int(i) for i in idx))
        if key not in cache:
            cache[key] = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in key])
        return cache[key]

    def ev(method, idx, kind):
        res, k = cached_eval(idx)
        rows.extend(eval_rows(dict(Dataset=ds, Fold=fi, Method=method, Kind=kind), res, k))

    ev("ALL", list(range(p)), "all")
    K = None
    cands = {"CASPER-t": CANDS["CASPER-t"]} if lite else CANDS   # yüksek boyut: elenen boot/near adayları atlanır
    for name, (g, se) in cands.items():
        t = time.time()
        cs = CasperSelector(gamma=g, se=se, max_n=1500 if lite else 3000).fit(Xd, ytr)
        sec = time.time() - t + t_disc
        idx = list(cs.support_idx_)
        add_sel(name, idx, sec, eps=cs.eps_star_, gamma_base=cs.gamma_base_, n_top=cs.n_topological_,
                mu=json.dumps(np.round(cs.mu_, 6).tolist()), se=json.dumps(np.round(cs.se_, 6).tolist()))
        ev(name, idx, "own")
        if name == "CASPER-t":
            K, rank_c = len(idx), list(cs.ranking_)
    rankers = {}
    if lite:  # yüksek boyut: RFE/SHAP/StdPFI/Boruta/FRFS/HSIC p=500'de çok pahalı → hızlı alt küme
        rankers.update(B.fast_classical(Xtr, ytr))
        extra = (B.mrmr, B.nrs_hu) if p <= 100 else (B.mrmr,)  # NRS ileri arama p>100'de K·p adım: atlanır (belgelendi)
    else:
        for m, (r, o, s) in B.classical9(Xtr, ytr).items():
            rankers[m] = (r, o, s)
        extra = (B.mrmr, B.boruta, B.hsic_lasso, B.frfs, B.nrs_hu)
    for f in extra:
        try:
            rankers[f.__name__] = f(Xtr, ytr, k_cap=K) if (lite and f is B.nrs_hu) else f(Xtr, ytr)
        except Exception as e:
            print(f"[{ds} f{fi}] {f.__name__} hata: {e}", flush=True)
    try:
        rankers["quickreduct"] = B.quickreduct(Xd, ytr)
    except Exception as e:
        print(f"[{ds} f{fi}] quickreduct hata: {e}", flush=True)
    for m, (r, o, s) in rankers.items():
        add_sel(m, r[:K], s, kind="matched")
        ev(m, r[:K], "matched")
        if o is not None:
            add_sel(m + "[own]", o, s, kind="own")
            ev(m, o, "own")
    accs = []
    for j, idx in enumerate(B.random_k(p, K, R, seed=1000 + fi)):
        res, k = cached_eval(idx)
        for r_ in eval_rows(dict(Dataset=ds, Fold=fi, Method="Random-K", Kind="matched", Draw=j), res, k):
            accs.append(r_)
    if accs:
        a = pd.DataFrame(accs).groupby(["Dataset", "Fold", "Method", "Kind", "Classifier", "K"], as_index=False)[["ACCU", "F1_S", "AUC"]].mean()
        rows.extend(a.to_dict("records"))
    print(f"[{ds}] fold {fi} bitti (p={p}, K={K})", flush=True)
    return rows, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=MAIN9)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--exp", default="main")
    ap.add_argument("--lite", action="store_true")
    a = ap.parse_args()
    for ds in a.datasets:
        X, y = load(ds)
        R = 30 if len(y) <= 1000 else 10
        F = folds(X, y)
        t0 = time.time()
        out = Parallel(n_jobs=a.jobs, backend="loky")(
            delayed(run_fold)(ds, i + 1, tr, te, X, y, R, max(1, 16 // a.jobs), a.lite) for i, (tr, te) in enumerate(F))
        rows = [r for o in out for r in o[0]]; sel = [s for o in out for s in o[1]]
        save_rows(rows, a.exp, f"{ds}_metrics"); save_rows(sel, a.exp, f"{ds}_selection")
        print(f"=== {ds} tamam: {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
