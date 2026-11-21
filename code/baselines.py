# -*- coding: utf-8 -*-
"""
Karşılaştırma yöntemleri. Hepsi YALNIZCA eğitim katında çalışır. Her fonksiyon
  (ranking: list[int] — en önemliden başlayarak tüm öznitelik indeksleri,
   own: list[int] | None — yöntemin kendi durma kuralının seçtiği küme,
   seconds: float) döndürür.

Kaynaklar:
  * Klasik 9: orijinal ClassicalMethods (ANOVA, MI, LASSO, RF, GB, RFE, SHAP, ReliefF, StdPFI)
  * mRMR (Peng 2005; mrmr_selection), Boruta (Kursa 2010; boruta_py), HSIC-Lasso (Yamada 2014; pyHSICLasso)
  * GBRS — granular-ball rough set (Xia et al., IEEE TNNLS 2023/2025, DOI 10.1109/TNNLS.2023.3325199),
    yazarların kodu third_party/GBRS (değiştirilmeden import edilir)
  * FRFS (Jensen & Shen; frlearn), QuickReduct (Jensen & Shen 2004), NRS (Hu et al. 2008, δ-top, monoton)
"""
from __future__ import annotations

import os, sys, time, contextlib, io
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
# authors' code of GBRS, GBFRS and GBNRS (not redistributed here, see code/README.md)
THIRD_PARTY = os.environ.get("CASPER_THIRD_PARTY", os.path.join(HERE, "..", "third_party"))
sys.path.insert(0, HERE)
from orig import P  # noqa: E402
from casper_core import pairwise, gamma_tol, stratified_subsample  # noqa: E402


def _timed(fn):
    def w(*a, **k):
        t = time.time()
        r, o = fn(*a, **k)
        return list(map(int, r)), (None if o is None else list(map(int, o))), time.time() - t
    w.__name__ = fn.__name__
    return w


def _pad(order, p):
    order = list(dict.fromkeys(int(i) for i in order))
    return order + [i for i in range(p) if i not in order]


# ---------------------------------------------------------------- klasik 9 (yöntem başına süre)
def classical9(Xtr, ytr, cap=3000):
    """Orijinal ClassicalMethods ile aynı ayarlar; fark: (i) her yöntemin süresi ayrı ölçülür,
    (ii) SHAP ve StdPFI n>cap iken tabakalı cap-satırlık alt örneklemde hesaplanır (ReliefF orijinalde de böyleydi).
    n<=cap olan tüm verilerde sonuç orijinalle aynıdır."""
    from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from skrebate import ReliefF
    import shap
    X = Xtr.values.astype(float); y = np.asarray(ytr).astype(int); p = X.shape[1]
    sub = stratified_subsample(y.astype(float), cap, 42)
    out = {}
    def put(name, fn):
        t = time.time(); s = np.nan_to_num(np.asarray(fn(), float))
        out[name] = (list(np.argsort(-s, kind="stable")), None, time.time() - t)
    put("ANOVA", lambda: f_classif(X, y)[0])
    put("MI", lambda: mutual_info_classif(X, y, random_state=42))
    def lasso():
        c = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=42).fit(X, y).coef_
        return np.abs(c).max(0)
    put("LASSO", lasso)
    put("RF", lambda: RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y).feature_importances_)
    put("GB", lambda: GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X, y).feature_importances_)
    def rfe():
        r = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=1).fit(X, y)
        return (p + 1 - r.ranking_).astype(float)
    put("RFE", rfe)
    def shp():
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
        sv = shap.TreeExplainer(rf).shap_values(X[sub])
        if isinstance(sv, list):
            return np.mean([np.abs(v).mean(0) for v in sv], axis=0)
        return np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)
    put("SHAP", shp)
    put("ReliefF", lambda: ReliefF(n_neighbors=max(3, min(50, len(sub) - 1))).fit(X[sub], y[sub]).feature_importances_)
    def stdpfi():
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
        return permutation_importance(rf, X[sub], y[sub], n_repeats=20, random_state=42, scoring="accuracy").importances_mean
    put("StdPFI", stdpfi)
    return out


def fast_classical(Xtr, ytr):
    """Yüksek boyut için: ANOVA, MI, LASSO, RF, ReliefF (orijinal ayarlarla)."""
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from skrebate import ReliefF
    X = Xtr.values.astype(float); out = {}
    def put(name, fn):
        t = time.time(); s = np.nan_to_num(np.asarray(fn(), float)); out[name] = (list(np.argsort(-s, kind="stable")), None, time.time() - t)
    put("ANOVA", lambda: f_classif(X, ytr)[0])
    put("MI", lambda: mutual_info_classif(X, ytr, random_state=42))
    put("LASSO", lambda: np.abs(LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=42).fit(X, ytr).coef_).max(0))
    put("RF", lambda: RandomForestClassifier(100, random_state=42, n_jobs=1).fit(X, ytr).feature_importances_)
    idx = stratified_subsample(ytr.astype(float), 3000, 42)
    put("ReliefF", lambda: ReliefF(n_neighbors=max(3, min(50, len(idx) - 1))).fit(X[idx], ytr[idx]).feature_importances_)
    return out


@_timed
def mrmr(Xtr, ytr):
    from mrmr import mrmr_classif
    sel = mrmr_classif(X=Xtr.reset_index(drop=True), y=pd.Series(ytr), K=Xtr.shape[1], show_progress=False)
    cols = list(Xtr.columns)
    return _pad([cols.index(c) for c in sel], len(cols)), None


@_timed
def boruta(Xtr, ytr):
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_jobs=1, class_weight="balanced", max_depth=5, random_state=42)
    b = BorutaPy(rf, n_estimators="auto", random_state=42, max_iter=50, verbose=0)
    X = Xtr.values.astype(float)
    b.fit(X, ytr)
    imp = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1).fit(X, ytr).feature_importances_
    rank = np.lexsort((-imp, b.ranking_))  # önce Boruta sırası, eşitlikte RF önemi
    own = np.where(b.support_)[0]
    if own.size == 0:
        own = rank[:1]
    return rank, own


@_timed
def hsic_lasso(Xtr, ytr, max_n=1500):
    from pyHSICLasso import HSICLasso
    idx = stratified_subsample(ytr.astype(float), max_n, 42)
    X = Xtr.values.astype(float)[idx]
    h = HSICLasso()
    h.input(X, ytr[idx].astype(float), featname=[str(i) for i in range(X.shape[1])])
    h.classification(num_feat=min(X.shape[1], 50), B=20, M=3, n_jobs=1)
    sel = [int(i) for i in h.get_index()]
    return _pad(sel, X.shape[1]), None


@_timed
def gbrs(Xtr, ytr, max_n=1000, d2_grid=None, n_d2=3):
    """Yazarların protokolü: MinMax ölçek, attribute_reduce(pur=1, d2) ve d2'yi iç 5-katlı KNN(5) CV ile seç.
    Maliyet nedeniyle eğitim katı max_n=1000 nesneye tabakalı alt örneklenir (belgelenir)."""
    sys.path.insert(0, os.path.join(THIRD_PARTY, "GBRS", "GBRS", "Code"))
    import GBRS as G
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    idx = stratified_subsample(ytr.astype(float), max_n, 42)
    X = MinMaxScaler().fit_transform(Xtr.values.astype(float)[idx])
    y = ytr[idx].astype(float)
    n, p = X.shape
    data = np.hstack([X, y[:, None], np.arange(n)[:, None]])
    grid = d2_grid or sorted(set(int(v) for v in np.linspace(0, p, min(p + 1, n_d2))))
    best, best_row = -1, None
    for d2 in grid:
        np.random.seed(42)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                row = G.attribute_reduce(data, pur=1, d2=d2)
        except Exception:
            continue
        if not row:
            continue
        sc = cross_val_score(KNeighborsClassifier(5), X[:, row], y, cv=5).mean()
        if sc > best:
            best, best_row = sc, row
    if best_row is None:
        best_row = [0]
    return _pad(best_row, p), best_row


@_timed
def frfs(Xtr, ytr, max_n=1500):
    from frlearn.feature_preprocessors import FRFS
    idx = stratified_subsample(ytr.astype(float), max_n, 42)
    X = MinMaxScaler().fit_transform(Xtr.values.astype(float)[idx])
    sel = FRFS()(X, ytr[idx])
    probe = np.arange(X.shape[1], dtype=float)[None, :]
    own = [int(round(v)) for v in sel(probe)[0]]
    return _pad(own, X.shape[1]), own


def _pawlak_gamma(D, y, attrs):
    if not attrs:
        return 0.0
    keys = D[:, attrs]
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    pos = 0
    for g in np.unique(inv):
        m = inv == g
        if len(np.unique(y[m])) == 1:
            pos += m.sum()
    return pos / len(y)


@_timed
def quickreduct(Xd, ytr):
    """Jensen & Shen (2004) — ayrıklaştırılmış eğitim katında Pawlak γ ile ileri açgözlü."""
    D = np.asarray(Xd, int)
    p = D.shape[1]
    gfull = _pawlak_gamma(D, ytr, list(range(p)))
    sel = []
    rem = list(range(p))
    while rem:
        g, a = max((_pawlak_gamma(D, ytr, sel + [a]), -a) for a in rem)
        sel.append(-a); rem.remove(-a)
        if g >= gfull - 1e-12:
            break
    return _pad(sel, p), sel


@_timed
def nrs_hu(Xtr, ytr, delta=0.15, max_n=3000, k_cap=None):
    """Hu et al. (2008) komşuluk kaba küme reduct'ı: MinMax ölçekli ham veride δ-top (L∞, Hu'nun
    önerdiği 0.1–0.2 aralığında δ=0.15), monoton γ_tol ile ileri açgözlü; kazanç ≤0 olunca durur."""
    idx = stratified_subsample(ytr.astype(float), max_n, 42)
    X = MinMaxScaler().fit_transform(Xtr.values.astype(float)[idx])
    y = ytr[idx]
    n, p = X.shape
    Dsel = np.zeros((n, n))
    sel, rem, gprev, own = [], list(range(p)), 0.0, None
    while rem and (k_cap is None or len(sel) < k_cap):
        best = None
        for f in rem:
            Df = np.maximum(Dsel, np.abs(X[:, f][:, None] - X[:, f][None, :]))
            g = gamma_tol(Df, y, delta)
            if best is None or g > best[0]:
                best = (g, f, Df)
        if own is None and best[0] <= gprev + 1e-9:
            own = list(sel)
        sel.append(best[1]); rem.remove(best[1]); Dsel = best[2]; gprev = best[0]
    own = own if own else (sel[:1] if k_cap is None else None)
    return _pad(sel, p), own


def random_k(p, k, R, seed):
    rng = np.random.RandomState(seed)
    return [sorted(rng.choice(p, size=k, replace=False)) for _ in range(R)]


# ---------------------------------------------------------------- granular-ball ailesi (yazar kodları, değiştirilmeden)
import contextlib, io, importlib.util


def _load_tp(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(THIRD_PARTY, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _gb_prepare(Xtr, ytr, max_n, with_index=True):
    idx = stratified_subsample(ytr.astype(float), max_n, 42)
    X = MinMaxScaler().fit_transform(Xtr.values.astype(float)[idx]); y = ytr[idx].astype(float)
    data = np.hstack([X, y[:, None]] + ([np.arange(len(y))[:, None]] if with_index else []))
    return X, y, data


def _knn_cv(X, y, row):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    return cross_val_score(KNeighborsClassifier(5), X[:, row], y, cv=5).mean()


@_timed
def gbfrs(Xtr, ytr, max_n=1000, purities=(0.95, 0.85, 0.75)):
    """GBFRS (Lian, Xia vd., IEEE TNNLS 2026). Yazar protokolü: saflık ∈ {1−0.05i}, iç 5-katlı KNN(5) ile seçim.
    Maliyet nedeniyle 3 saflık değeri ve ≤1000 nesne (belgelendi)."""
    G = _load_tp("GBFRS_mod", "GBFRS_x/GBFRS_github/GBFRS.py")
    X, y, data = _gb_prepare(Xtr, ytr, max_n)
    best, best_row = -1, None
    for pur in purities:
        np.random.seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                row = G.attribute_reduce(data.copy(), pur=pur)
            except Exception:
                continue
        if row:
            sc = _knn_cv(X, y, row)
            if sc > best:
                best, best_row = sc, list(row)
    best_row = best_row or [0]
    return _pad(best_row, X.shape[1]), best_row


@_timed
def gbnrs(Xtr, ytr, max_n=1000, n_d2=3):
    """GBNRS-new (Xia vd.; GBNRS IEEE TKDE 2022 yazar kodu, geriye eleme). d2 ızgarası yazar kuralından 3 değer;
    seçim iç 5-katlı KNN(5) ile (yazar kodu eğitim doğruluğu kullanıyordu; sızıntıyı önlemek için CV). ≤1000 nesne."""
    G = _load_tp("GBNRS_mod", "GBNRS_x/GBNRS_new/xin_de.py")
    X, y, data = _gb_prepare(Xtr, ytr, max_n, with_index=False)
    p = X.shape[1]
    grid = sorted(set(int(v) for v in np.linspace(2, max(2, 2 * p - 1), n_d2)))
    best, best_row = -1, None
    for d2 in grid:
        np.random.seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                row = G.attribute_reduce(data.copy(), pur=1, d2=d2)
            except Exception:
                continue
        if row:
            sc = _knn_cv(X, y, row)
            if sc > best:
                best, best_row = sc, list(row)
    best_row = best_row or [0]
    # geriye eleme: sıralama = kalan küme + kalanların dışındakiler (ANOVA sırasıyla)
    from sklearn.feature_selection import f_classif
    rest = [i for i in np.argsort(-np.nan_to_num(f_classif(X, y)[0])) if i not in best_row]
    return list(best_row) + rest, best_row
