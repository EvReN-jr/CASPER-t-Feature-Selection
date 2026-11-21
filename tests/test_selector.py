"""Tests for the CasperT selector: API contract, leakage-free Pipeline use,
and a small reproducibility check."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from casper import CasperT


@pytest.fixture
def data():
    X, y = make_classification(n_samples=200, n_features=12, n_informative=5,
                               n_redundant=3, random_state=0)
    return X, y


def test_fit_transform_shapes(data):
    X, y = data
    sel = CasperT(random_state=0).fit(X, y)
    mask = sel.get_support()
    assert mask.dtype == bool and mask.shape == (X.shape[1],)
    assert mask.any()                                   # at least one feature
    Xr = sel.transform(X)
    assert Xr.shape == (X.shape[0], int(mask.sum()))


def test_diagnostics_present(data):
    X, y = data
    sel = CasperT(random_state=0).fit(X, y)
    for attr in ("delta_gamma_", "sem_", "t_stats_", "eps_star_", "gamma_base_", "q1_"):
        assert hasattr(sel, attr)
    assert sel.t_stats_.shape == (X.shape[1],)


def test_kappa_monotonicity(data):
    X, y = data
    n_low = CasperT(kappa=0.5, random_state=0).fit(X, y).get_support().sum()
    n_high = CasperT(kappa=3.0, random_state=0).fit(X, y).get_support().sum()
    assert n_high <= n_low                              # stricter margin -> fewer


def test_pipeline_cross_val_no_error(data):
    X, y = data
    pipe = Pipeline([("casper", CasperT(random_state=0)), ("clf", SVC())])
    scores = cross_val_score(pipe, X, y, cv=3)
    assert scores.shape == (3,) and np.all(scores > 0)


def test_deterministic(data):
    X, y = data
    a = CasperT(random_state=0).fit(X, y).get_support()
    b = CasperT(random_state=0).fit(X, y).get_support()
    assert np.array_equal(a, b)


def test_constant_column_not_selected():
    """A zero-variance column carries no information and is never selected,
    whatever kappa.  The fixture has mostly noise features, so Q1 < 0 and a
    division by a near-zero SEM would otherwise give the constant column a
    huge margin."""
    X, y = make_classification(n_samples=300, n_features=40, n_informative=3,
                               n_redundant=0, n_repeated=0, random_state=0)
    X_aug = np.hstack([X, np.zeros((X.shape[0], 1))])
    const_idx = X_aug.shape[1] - 1
    for kappa in (0.5, 1.5, 5.0):
        sel = CasperT(kappa=kappa, random_state=0).fit(X_aug, y)
        assert sel.q1_ < 0, "fixture no longer exercises the q1<0 condition this test needs"
        assert not sel.get_support()[const_idx], f"constant column selected at kappa={kappa}"
