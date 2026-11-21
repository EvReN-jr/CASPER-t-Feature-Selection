# -*- coding: utf-8 -*-
"""
Bilinen-gerçek sentetik veri.
make_classification: 5 bilgilendirici öznitelik + (p−5) saf gürültü, n=600, p∈{10,20,50,100,200,500}.
Ölçümler (her p için 5 tohum, eğitim %70):
  * ε*/p, gamma_base
  * bilgilendirici vs gürültü özniteliklerinin ortalama |μ_Δγ| değerleri (Önerme prop:highdim: p büyüdükçe küçülür)
  * sıralama AUC'si (bilgilendirici=1, gürültü=0) — CASPER-t μ, ANOVA, RF, mRMR, NRS(Hu)
  * CASPER-t seçilen kümenin gerçek-öznitelik duyarlılık/kesinliği ve test doğruluğu (RF)
Ek: etiket gürültüsü (%0/10/20, yalnız eğitim etiketleri) p=20'de.
"""
import json, time, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from exp_common import Discretizer, set_threads, save_rows
from casper_core import CasperSelector
import baselines as B

PS = [10, 20, 50, 100, 200, 500]


def one(p, seed, flip=0.0, threads=3):
    set_threads(threads)
    X, y = make_classification(n_samples=600, n_features=p, n_informative=5, n_redundant=0, n_repeated=0,
                               n_clusters_per_class=2, class_sep=1.0, flip_y=0.0, shuffle=False, random_state=seed)
    truth = np.r_[np.ones(5), np.zeros(p - 5)]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(600); tr, te = idx[:420], idx[420:]
    Xtr = pd.DataFrame(X[tr], columns=[f"f{i}" for i in range(p)]); Xte = pd.DataFrame(X[te], columns=Xtr.columns)
    ytr, yte = y[tr].copy(), y[te]
    if flip > 0:
        m = rng.rand(len(ytr)) < flip; ytr[m] = 1 - ytr[m]
    Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
    t = time.time(); cs = CasperSelector().fit(Xd, ytr); sec = time.time() - t
    scores = {"CASPER-t": cs.mu_, "ANOVA": f_classif(Xtr.values, ytr)[0],
              "RF": RandomForestClassifier(200, random_state=0, n_jobs=1).fit(Xtr.values, ytr).feature_importances_}
    for nm, f in (("mRMR", B.mrmr), ("NRS(Hu)", B.nrs_hu)):
        r, o, s = f(Xtr, ytr); sc = np.zeros(p); sc[r] = np.arange(p, 0, -1); scores[nm] = sc
    out = dict(p=p, seed=seed, flip=flip, eps=cs.eps_star_, eps_over_p=cs.eps_star_ / p, gamma_base=cs.gamma_base_,
               mu_inf=float(np.mean(np.abs(cs.mu_[:5]))), mu_noise=float(np.mean(np.abs(cs.mu_[5:]))),
               sem_inf=float(np.mean(cs.sem_[:5])), K=len(cs.support_idx_), seconds=sec)
    for nm, sc in scores.items():
        out[f"AUC_{nm}"] = roc_auc_score(truth, sc)
    s = set(cs.support_idx_)
    out["recall_true"] = len(s & set(range(5))) / 5
    out["precision_true"] = len(s & set(range(5))) / max(1, len(s))
    rf = RandomForestClassifier(200, random_state=0, n_jobs=1)
    out["acc_casper"] = rf.fit(Xtr.values[:, sorted(s)], ytr).score(Xte.values[:, sorted(s)], yte)
    out["acc_true5"] = rf.fit(Xtr.values[:, :5], ytr).score(Xte.values[:, :5], yte)
    out["acc_all"] = rf.fit(Xtr.values, ytr).score(Xte.values, yte)
    print(p, seed, flip, "AUC", round(out["AUC_CASPER-t"], 3), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); a = ap.parse_args()
    tasks = [(p, s, 0.0) for p in PS for s in range(5)] + [(20, s, f) for s in range(5) for f in (0.1, 0.2)]
    out = Parallel(n_jobs=a.jobs)(delayed(one)(p, s, f, max(1, 16 // a.jobs)) for p, s, f in tasks)
    save_rows(out, "synthetic", "synthetic")


if __name__ == "__main__":
    main()
