# -*- coding: utf-8 -*-
"""Uygulama testleri: sızıntı yokluğu, orijinal pipeline ile birebir eşitlik, ayrıklaştırıcılar, HEOM."""
import os, sys, warnings
import numpy as np, pandas as pd
import pytest
warnings.filterwarnings("ignore")
for _m in ("xgboost", "shap", "skrebate", "scikit_posthocs", "openpyxl", "jenkspy"):
    pytest.importorskip(_m)   # the reference pipeline needs them (it would otherwise pip-install them)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp_common import load, folds, FoldPreprocessor, Discretizer
from casper_core import CasperSelector, pairwise
from orig import P


@pytest.fixture(scope="module")
def cryo():
    X, y = load("cyron")
    return X, y, folds(X, y)


def _fit_fold(X, y, tr):
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True)
    disc = Discretizer().fit(Xtr)
    cs = CasperSelector().fit(disc.transform(Xtr).values.astype(float), y[tr])
    return pp, cs


def test_no_leakage_test_rows_do_not_affect_fit(cryo):
    """Test katındaki satırlar keyfi değiştirildiğinde ön işleme sınırları ve seçilen küme değişmemeli."""
    X, y, F = cryo
    tr, te = F[0]
    pp1, cs1 = _fit_fold(X, y, tr)
    X2 = X.copy(); X2.iloc[te] = X2.iloc[te] * 100 + 7   # test satırlarını boz
    y2 = y.copy(); y2[te] = 1 - y2[te]
    pp2, cs2 = _fit_fold(X2, y2, tr)
    assert pp1.bounds_ == pp2.bounds_ and pp1.drop_ == pp2.drop_
    assert np.array_equal(cs1.support_idx_, cs2.support_idx_) and cs1.eps_star_ == cs2.eps_star_


def test_reproduces_original_selector_exactly(cryo):
    """Yeni çekirdek = makale sayılarını üreten orijinal V1Selector + _tstat_select (READY verisinde)."""
    df, t = P._load_dataset("cyrondata3_READY_FOR_RS.xlsx")
    feats = [c for c in df.columns if c != t]; y = df[t].values.astype(int)
    tr, _ = folds(df[feats], y)[0]
    Xtr = df[feats].iloc[tr].reset_index(drop=True)
    Xd = P.AutoJenksDiscretizer(random_state=42).fit(Xtr).transform(Xtr)
    v1 = P.V1Selector(t, random_state=42).fit(pd.concat([Xd, pd.Series(y[tr], name=t)], axis=1))
    sel, _, _ = P._tstat_select(v1.ranking_, "Delta_Gamma", "SEM_Gamma", 1.5)
    cs = CasperSelector().fit(Xd.values.astype(float), y[tr])
    r = v1.ranking_.set_index("Feature").loc[feats]
    assert v1.eps_star == cs.eps_star_
    assert np.allclose(r.Delta_Gamma.values, cs.mu_, atol=0) and np.allclose(r.SEM_Gamma.values, cs.sem_, atol=0)
    assert set(sel) == set(np.array(feats)[cs.support_idx_])


@pytest.mark.parametrize("m", ["jenks_km", "jenks_exact", "ew", "ef"])
def test_discretizers_produce_ordinal_codes(cryo, m):
    X, y, F = cryo
    tr, te = F[0]
    d = Discretizer(m).fit(X.iloc[tr])
    Z = d.transform(X.iloc[te])
    for c in d.edges_ if m != "jenks_km" else d.base_.column_centers_:
        k = d.base_.column_k_[c]
        assert Z[c].min() >= 1 and Z[c].max() <= k


def test_heom_distance():
    X = np.array([[1, 3], [2, 3], [1, 5]], float)
    nom = np.array([False, True])
    D = pairwise(X, "l1", nom)
    assert D[0, 1] == 1 and D[0, 2] == 1 and D[1, 2] == 2


def test_fold_preprocessor_matches_global_recipe_on_kanser():
    """Kat-içi filtre, tüm veride uygulandığında READY dosyasındaki ile aynı 4 sütunu atmalı."""
    X, y = load("kanser")
    pp = FoldPreprocessor().fit(X)
    assert sorted(pp.drop_) == sorted(["mean perimeter", "mean radius", "worst perimeter", "worst radius"])
