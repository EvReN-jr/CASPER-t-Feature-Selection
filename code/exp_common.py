# -*- coding: utf-8 -*-
"""
Deneylerin ortak altyapısı.

* Veri: 5 küçük veri HAM hâlinden (data/raw/), 4 büyük + 7 yüksek boyutlu veri data/<ad>_READY_FOR_RS.xlsx
  (OpenML ham, winsorize edilmemiş; pipeline/gamma_pfi_uhem.py OpenML'den hazırlar). Konum CASPER_DATA ile değiştirilebilir.
* FoldPreprocessor: winsorization (1.5·IQR) + korelasyon filtresi (|ρ|>0.98) YALNIZCA eğitim katında
  fit edilir, test katına aynı sınırlarla uygulanır.
* Ayrıklaştırıcılar: jenks_km (orijinal), jenks_exact (jenkspy, Fisher-Jenks DP), ew, ef, none.
* Değerlendirme: orijinal 8 sınıflandırıcı (yz_algorithms.train_models_given_split, SVC için 4000 üst sınır).
* Dış döngü: 5×2 RepeatedStratifiedKFold(random_state=42) — makaleyle aynı katlar.
"""
from __future__ import annotations

import os, sys, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(REV, ".."))
sys.path.insert(0, HERE)
from orig import P  # noqa: E402  (orijinal pipeline; değiştirilmez)

from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

DATA = os.environ.get("CASPER_DATA", os.path.join(HERE, "data"))
UHEM_DATA = DATA
RAW = os.path.join(DATA, "raw")
RESULTS = os.environ.get("CASPER_RESULTS", os.path.join(REV, "results"))

SMALL = ["cyron", "diabet", "kanser", "parkinson", "sonar"]
LARGE = ["banknote", "phoneme", "htru2", "magic"]
MAIN9 = SMALL + LARGE
HIGHDIM = ["ozone", "hillvalley", "mfeat", "usps", "madeline", "philippine", "madelon"]
PRETTY = {"cyron": "Cryotherapy", "diabet": "Diabetes", "kanser": "Breast Cancer",
          "parkinson": "Parkinson's", "sonar": "Sonar", "banknote": "Banknote", "phoneme": "Phoneme",
          "htru2": "HTRU2", "magic": "MAGIC", "ozone": "Ozone", "hillvalley": "Hill-Valley",
          "mfeat": "MFeat", "usps": "USPS", "madeline": "Madeline", "philippine": "Philippine",
          "madelon": "Madelon"}
LARGE_FILES = {"banknote": "banknote", "phoneme": "phoneme", "htru2": "htru2", "magic": "magicdata3"}


FAST_MAXN = int(os.environ.get("CASPER_FAST_MAXN", "0"))  # >0 ise HIZLI MOD: n>FAST_MAXN veriler tabakalı alt örneğe iner


def load(name):
    """(X_df, y). HIZLI MOD (CASPER_FAST_MAXN>0): n>FAST_MAXN ise sabit tohumlu tabakalı alt örnek (yön bulma amaçlı;
    kesin sonuçlar tam veriyle üretilir)."""
    X, y = _load_full(name)
    if FAST_MAXN and len(y) > FAST_MAXN:
        from casper_core import stratified_subsample
        idx = stratified_subsample(y.astype(float), FAST_MAXN, 2026)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X, y


def _load_full(name):
    """(X_df, y) döndürür. Hedef 0/1 tamsayı."""
    if name in SMALL:
        df = pd.read_csv(os.path.join(RAW, f"{name}_raw.csv"))
        y = df.pop("target").values.astype(int)
        return df.astype(float), y
    fn = LARGE_FILES.get(name, name)
    cache = os.path.join(RESULTS, "_cache", f"{fn}.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        df = pd.read_excel(os.path.join(UHEM_DATA, f"{fn}_READY_FOR_RS.xlsx"))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache)
    df = df.drop(columns=[c for c in P.EXCLUDE_COLS if c in df.columns])
    y = df.pop("target").values.astype(int)
    return df.astype(float), y


def folds(X, y, n_splits=5, n_repeats=2, seed=42):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(X, y))


# ---------------------------------------------------------------- kat-içi ön işleme
class FoldPreprocessor:
    def __init__(self, winsor=True, corr="default", iqr=1.5, min_unique=11):
        if corr == "default":   # CASPER_NOCORR=1 → korelasyon filtresi kapalı (Hill-Valley istisnası)
            corr = None if os.environ.get("CASPER_NOCORR") == "1" else 0.98
        self.winsor, self.corr, self.iqr, self.min_unique = winsor, corr, iqr, min_unique

    def fit(self, X):
        self.cont_ = [c for c in X.columns if X[c].nunique() >= self.min_unique]
        self.bounds_ = {}
        if self.winsor:
            for c in self.cont_:
                q1, q3 = X[c].quantile(0.25), X[c].quantile(0.75)
                self.bounds_[c] = (q1 - self.iqr * (q3 - q1), q3 + self.iqr * (q3 - q1))
        Xw = self._clip(X)
        self.drop_ = []
        if self.corr is not None and X.shape[1] > 1:
            cm = Xw.corr().abs().values
            cols = list(X.columns)
            var = Xw.var().values
            drop = set()
            for j in range(len(cols)):
                for i in range(j):
                    if cm[i, j] > self.corr and cols[i] not in drop and cols[j] not in drop:
                        drop.add(cols[j] if var[j] <= var[i] else cols[i])
            self.drop_ = sorted(drop)
        return self

    def _clip(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X

    def transform(self, X):
        return self._clip(X).drop(columns=self.drop_)


# ---------------------------------------------------------------- ayrıklaştırıcılar
class Discretizer:
    """method ∈ {'jenks_km','jenks_exact','ew','ef','none'}. Hepsi aynı bin sayısını (jenks_km'nin GVF≥0.8
    kuralıyla seçtiği k_f) kullanır; böylece ablasyonda yalnızca KESİM NOKTALARI değişir."""

    def __init__(self, method="jenks_km"):
        self.method = method

    def fit(self, X):
        self.base_ = P.AutoJenksDiscretizer(random_state=42).fit(X)
        self.edges_ = {}
        for c in X.columns:
            if self.base_.column_mode_.get(c) != "continuous":
                continue
            k = self.base_.column_k_[c]
            v = X[c].values.astype(float)
            if self.method == "jenks_exact":
                import jenkspy
                u = np.unique(v)
                kk = min(k, len(u) - 1)
                br = jenkspy.jenks_breaks(v if len(v) <= 3000 else np.random.RandomState(0).choice(v, 3000, replace=False), n_classes=kk) if kk >= 2 else [v.min(), v.max()]
                self.edges_[c] = np.unique(br[1:-1])
            elif self.method == "ew":
                self.edges_[c] = np.linspace(v.min(), v.max(), k + 1)[1:-1]
            elif self.method == "ef":
                self.edges_[c] = np.unique(np.quantile(v, np.linspace(0, 1, k + 1)[1:-1]))
            elif self.method == "none":
                self.edges_[c] = (v.min(), v.max(), k)
        return self

    def transform(self, X):
        if self.method == "jenks_km":
            return self.base_.transform(X)
        out = X.copy()
        for c, e in self.edges_.items():
            v = X[c].values.astype(float)
            if self.method == "none":
                lo, hi, k = e
                out[c] = 1 + (k - 1) * np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)  # sürekli, aynı ölçek
            else:
                out[c] = 1 + np.searchsorted(e, v, side="right")
        return out


# ---------------------------------------------------------------- değerlendirme
CLASSIFIERS = ["GBC", "GNB", "LDA", "LOGREG", "RFC", "SVC", "XGBC", "HYBRID"]


def evaluate(Xtr, Xte, ytr, yte, cols):
    """{classifier: {ACCU,PREC,RECA,F1_S,AUC}} — orijinal _evaluate_feature_set."""
    res, k = P._evaluate_feature_set(Xtr, Xte, ytr, yte, list(cols))
    return res, k


def eval_rows(base, res, k):
    rows = []
    for clf, m in res.items():
        r = dict(base); r.update(Classifier=clf, K=k)
        r.update({kk: m.get(kk) for kk in ("ACCU", "F1_S", "AUC")})
        rows.append(r)
    return rows


def set_threads(n):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except Exception:
        pass


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def nogueira(sets, p):
    """Nogueira, Sechidis & Brown (2018) kararlılık indeksi."""
    M = len(sets)
    Z = np.zeros((M, p))
    for i, s in enumerate(sets):
        Z[i, list(s)] = 1
    kbar = Z.sum(1).mean()
    if kbar in (0, p):
        return np.nan
    s2 = Z.var(0, ddof=1)
    return 1 - s2.mean() / ((kbar / p) * (1 - kbar / p))


def save_rows(rows, exp, name):
    d = os.path.join(RESULTS, exp)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, f"{name}.csv"), index=False)
