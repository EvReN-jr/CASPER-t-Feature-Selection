# -*- coding: utf-8 -*-
"""
(a) Karışık tipli, eksik değerli, gürültülü gerçek veriler:
OpenML credit-g, colic, hepatitis, sick. Kat içinde: eksik değer doldurma (sayısal medyan, nominal mod),
sayısal sütunlara winsor+Jenks, nominal sütunlara eğitimden öğrenilen tamsayı kodu; CASPER-t
HEOM-tipi tolerans (sayısal L1 + nominal Hamming) ile. Karşılaştırma: MI, RF, mRMR, Rastgele-K (eşlenik K).
Sınıflandırıcılar seçilen özniteliklerin one-hot kodlanmış hâlinde eğitilir.
(b) Etiket gürültüsü: 5 küçük veri + Banknote, eğitim etiketlerinin %10/%20'si çevrilir (test temiz).
CASPER-t ve ANOVA/RF/mRMR için: doğruluk ve gürültüsüz seçime Jaccard benzerliği.
"""
import os, json, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from exp_common import (RESULTS, load, folds, FoldPreprocessor, Discretizer, evaluate, eval_rows, set_threads, save_rows,
                        jaccard, REV, SMALL)
from casper_core import CasperSelector
import baselines as B

MIXED = {"credit-g": "good", "colic": "1", "hepatitis": "LIVE", "sick": "sick"}


def load_mixed(name):
    b = fetch_openml(name=name, version=1, as_frame=True, parser="liac-arff",
                     data_home=os.path.join(REV, "data_raw", "openml"))
    X = b.data.copy()
    X = X.loc[:, X.notna().any()]          # tamamen eksik sütun (ör. sick: TBG) atılır
    X = X.loc[:, X.nunique(dropna=True) > 1]  # sabit sütunlar atılır
    nom = [str(X[c].dtype) in ("category", "object") for c in X.columns]
    y = (b.target.astype(str) == MIXED[name]).astype(int).values
    return X, np.array(nom), y


def prep_mixed(Xtr, Xte, nom):
    cols = list(Xtr.columns)
    num = [c for c, n in zip(cols, nom) if not n]; cat = [c for c, n in zip(cols, nom) if n]
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in num:
        med = Xtr[c].astype(float).median()
        Xtr[c] = Xtr[c].astype(float).fillna(med); Xte[c] = Xte[c].astype(float).fillna(med)
    codes = {}
    for c in cat:
        s = Xtr[c].map(lambda v: "nan" if pd.isna(v) else str(v)).astype(object); mode = s[s != "nan"].mode().iloc[0] if (s != "nan").any() else "nan"
        s = s.replace("nan", mode); levels = sorted(s.unique()); codes[c] = {v: i + 1 for i, v in enumerate(levels)}
        Xtr[c] = s.map(codes[c]).astype(float)
        Xte[c] = Xte[c].map(lambda v: "nan" if pd.isna(v) else str(v)).astype(object).replace("nan", mode).map(codes[c]).fillna(0).astype(float)
    pp = FoldPreprocessor(corr=None).fit(Xtr[num]) if num else None
    if num:
        Xtr[num] = pp.transform(Xtr[num]); Xte[num] = pp.transform(Xte[num])
    disc = Discretizer().fit(Xtr[num]) if num else None
    Xd = Xtr.copy()
    if num:
        Xd[num] = disc.transform(Xtr[num])
    oh_tr = pd.get_dummies(Xtr, columns=cat, dtype=float); oh_te = pd.get_dummies(Xte, columns=cat, dtype=float)
    oh_te = oh_te.reindex(columns=oh_tr.columns, fill_value=0.0)
    groups = {c: [k for k in oh_tr.columns if k == c or k.startswith(c + "_")] for c in cols}
    return Xtr, Xd, oh_tr, oh_te, groups


def run_mixed_fold(name, fi, tr, te, X, nom, y, threads):
    set_threads(threads)
    Xtr, Xd, oh_tr, oh_te, groups = prep_mixed(X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), nom)
    ytr, yte = y[tr], y[te]; cols = list(X.columns); p = len(cols)
    cs = CasperSelector(nominal=nom).fit(Xd.values.astype(float), ytr)
    K = len(cs.support_idx_)
    ranks = {"CASPER-t[HEOM]": list(cs.support_idx_),
             "MI": list(np.argsort(-mutual_info_classif(Xtr.values, ytr, discrete_features=nom, random_state=42), kind="stable"))[:K],
             "RF": list(np.argsort(-RandomForestClassifier(200, random_state=42, n_jobs=1).fit(Xtr.values, ytr).feature_importances_, kind="stable"))[:K],
             "mRMR": B.mrmr(Xtr, ytr)[0][:K]}
    rng = np.random.RandomState(fi)
    rows = []

    def ev(m, idx, extra=None):
        cc = [k for i in idx for k in groups[cols[i]]]
        res, _ = evaluate(oh_tr, oh_te, ytr, yte, cc)
        rows.extend(eval_rows(dict(Dataset=name, Fold=fi, Method=m, **(extra or {})), res, len(idx)))
    ev("ALL", list(range(p)))
    for m, idx in ranks.items():
        ev(m, idx)
    for j in range(10):
        ev("Random-K", list(rng.choice(p, K, replace=False)), {"Draw": j})
    print(f"[{name}] fold {fi} K={K}/{p}", flush=True)
    return rows


def run_noise_fold(ds, fi, tr, te, X, y, threads):
    set_threads(threads)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    yte = y[te]; cols = list(Xtr.columns)
    Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
    rows, ref = [], {}
    for flip in (0.0, 0.1, 0.2):
        ytr = y[tr].copy()
        if flip:
            m = np.random.RandomState(100 * fi + int(flip * 100)).rand(len(ytr)) < flip; ytr[m] = 1 - ytr[m]
        cs = CasperSelector().fit(Xd, ytr); K = len(cs.support_idx_)
        sets = {"CASPER-t": sorted(map(int, cs.support_idx_))}
        if flip == 0.0:
            ref["K"] = K
        Kc = ref["K"]
        sets["ANOVA"] = sorted(np.argsort(-np.nan_to_num(f_classif(Xtr.values, ytr)[0]), kind="stable")[:Kc].tolist())
        sets["RF"] = sorted(np.argsort(-RandomForestClassifier(100, random_state=42, n_jobs=1).fit(Xtr.values, ytr).feature_importances_, kind="stable")[:Kc].tolist())
        sets["mRMR"] = sorted(B.mrmr(Xtr, ytr)[0][:Kc])
        for m, idx in sets.items():
            if flip == 0.0:
                ref[m] = idx
            res, k = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in idx])
            rows.extend(eval_rows(dict(Dataset=ds, Fold=fi, Method=m, flip=flip,
                                       Jaccard_vs_clean=jaccard(idx, ref[m])), res, k))
    print(f"[noise {ds}] fold {fi}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); a = ap.parse_args()
    th = max(1, 16 // a.jobs)
    for name in MIXED:
        if os.path.exists(os.path.join(RESULTS, "mixed", f"{name}.csv")):
            continue   # önceki koşuda kaydedildi
        X, nom, y = load_mixed(name); F = folds(X, y)
        out = Parallel(n_jobs=a.jobs)(delayed(run_mixed_fold)(name, i + 1, tr, te, X, nom, y, th) for i, (tr, te) in enumerate(F))
        save_rows([r for o in out for r in o], "mixed", name)
    for ds in SMALL + ["banknote"]:
        X, y = load(ds); F = folds(X, y)
        out = Parallel(n_jobs=a.jobs)(delayed(run_noise_fold)(ds, i + 1, tr, te, X, y, th) for i, (tr, te) in enumerate(F))
        save_rows([r for o in out for r in o], "labelnoise", ds)


if __name__ == "__main__":
    main()
