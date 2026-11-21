"""The exact engine on distinct code vectors must reproduce the dense reference for
gamma_op, gamma_tol and gamma_near, every metric, and every selector configuration
of the ablation (bootstrap SE included), bit for bit."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import casper_core as C


def test_pattern_gamma_matches_definitions():
    rng = np.random.default_rng(5)
    for t in range(60):
        n = int(rng.integers(6, 60)); p = int(rng.integers(1, 5))
        X = rng.integers(1, int(rng.integers(2, 6)) + 1, size=(n, p)).astype(float)
        if t % 3 == 0:
            X = X[rng.integers(0, max(2, n // 4), n)]          # many duplicate rows
        y = rng.integers(0, int(rng.integers(2, 4)), n)
        for g in ("op", "tol", "near"):
            for m in ("l1", "l2", "linf"):
                D = C.pairwise(X, m)
                for e in (0.0, 1.0, 1.5, 2.0, 3.0):
                    assert C.GAMMAS[g](D, y, e) == C.pattern_gamma(X, y, e, g, m)


CFGS = [dict(), dict(se="boot"), dict(gamma="near"), dict(gamma="near", se="boot"), dict(metric="l2"),
        dict(metric="linf"), dict(eps_rule="zero"), dict(eps_rule="coarsest"), dict(gamma="tol"),
        dict(importance="delete"), dict(rule="q1"), dict(rule="positive")]


@pytest.mark.parametrize("cfg", CFGS, ids=[str(c) for c in CFGS])
def test_selector_fast_equals_reference(cfg):
    rng = np.random.default_rng(11)
    for _ in range(6):
        n = int(rng.integers(30, 100)); p = int(rng.integers(2, 6))
        X = rng.integers(1, 5, size=(n, p)).astype(float); y = rng.integers(0, 2, n)
        a = C.CasperSelector(engine="reference", n_rep=5, n_boot=4, **cfg).fit(X, y)
        b = C.CasperSelector(engine="fast", n_rep=5, n_boot=4, **cfg).fit(X, y)
        assert a.eps_star_ == b.eps_star_ and a.gamma_base_ == b.gamma_base_
        assert np.array_equal(a.mu_, b.mu_) and np.array_equal(a.se_, b.se_)
        assert np.array_equal(a.support_idx_, b.support_idx_)
